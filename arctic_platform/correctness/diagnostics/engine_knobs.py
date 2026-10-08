# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Which engine setting produces the residual gradient-norm disagreement at sp 1?

At sequence-parallel degree 1 there is no sequence split and no cross-rank reduction, and 27 of 374 tensors
still disagree with the single-GPU reference by more than 1e-3 absolute -- slightly more than at sp 8. So
the disagreement comes from what a single rank does, and the settings below are what that rank does
differently from the reference.

Each arm changes one entry of the job config and leaves the rest at the config's values. The reference is
run once: it does not depend on any of them.
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arctic_platform.correctness.harness.batches import build_batch  # noqa: E402
from arctic_platform.correctness.harness.batches import save  # noqa: E402
from arctic_platform.correctness.harness.config import load_config  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import build_payload  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import fwd_bwd_step  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import gateway  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import pack  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import running_job  # noqa: E402
from arctic_platform.correctness.harness.names import align  # noqa: E402
from arctic_platform.correctness.harness.runner import run_reference  # noqa: E402
from arctic_platform.correctness.harness.seeds import SEED  # noqa: E402
from arctic_platform.correctness.harness.spec import STATED_CRITERION_ABS  # noqa: E402
from arctic_platform.correctness.harness.spec import TestSpec  # noqa: E402
from arctic_platform.correctness.onboarding.synth_model import materialize_pretrained  # noqa: E402

ROW_TOKENS = int(os.environ.get("PROBE_ROW_TOKENS", 2048))
LAYERS = int(os.environ.get("PROBE_LAYERS", 0))
SOURCE = "/data-fast/base-models/Qwen/Qwen3.8-27B"
CACHE_ROOT = "/data-fast/base-models/synthetic"
SPEC = "arctic_platform/correctness/specs/qwen3.8-27b-h200-train-sft-8gpus-2k.json"
# The two tensors the backward pass reaches first, in order. A disagreement that starts at norm.weight
# while lm_head.weight agrees puts the origin in the backward rather than the forward.
WATCH = ("lm_head.weight", "norm.weight")


def with_ac_disabled(training):
    training["activation_checkpointing"] = False
    training["gradient_checkpointing"] = False


def with_ac_offload_disabled(training):
    training.setdefault("ac_config", {}).setdefault("offload_config", {})["enabled"] = False


def with_zero_stage_0(training):
    training.setdefault("ds_config", {}).setdefault("zero_optimization", {})["stage"] = 0


def with_optimizer_on_gpu(training):
    zero = training.setdefault("ds_config", {}).setdefault("zero_optimization", {})
    zero.pop("offload_optimizer", None)
    training.setdefault("optimizer", {})["name"] = "adamw"


def with_bf16_reduce(training):
    training["reduce_dtype"] = "bfloat16"
    training.setdefault("ds_config", {})["communication_data_type"] = "bfloat16"


SELECTED = {a for a in os.environ.get("PROBE_ARMS", "").split(",") if a}
REPEAT = int(os.environ.get("PROBE_REPEAT", 1))

ARMS = (
    ("config as written", None),
    ("activation checkpointing off", with_ac_disabled),
    ("ac offload off", with_ac_offload_disabled),
    ("zero stage 0", with_zero_stage_0),
    ("optimizer on gpu", with_optimizer_on_gpu),
    ("bfloat16 reduce", with_bf16_reduce),
)


def main(config_path: str, spec_path: str) -> int:
    from transformers import AutoConfig

    cfg = load_config(Path(config_path))
    spec = TestSpec.read(Path(spec_path))
    layers = LAYERS or spec.model.num_hidden_layers
    model_path = (
        spec.model.cache_path
        if layers == spec.model.num_hidden_layers
        else materialize_pretrained(SOURCE, f"{CACHE_ROOT}/Qwen3.8-27B-{layers}L", layers).cache_path
    )
    fp32_lm_head = bool(cfg.training.get("fp32_lm_head", False))
    # Both engines run whatever this resolves to. Flash attention's backward accumulates the query gradient
    # with atomics and ignores torch.use_deterministic_algorithms, so two identical runs of one engine
    # already disagree by several times the criterion; sdpa repeats exactly and leaves only differences
    # that belong to the engines.
    attn = os.environ.get("PROBE_ATTN") or cfg.training.get("attn_implementation", "flash_attention_3")

    model_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vocab = getattr(model_cfg, "text_config", model_cfg).vocab_size

    work = Path(tempfile.mkdtemp(prefix="engine-knobs-"))
    batch = build_batch("gas1", 1, ROW_TOKENS, vocab, seed=SEED)
    if os.environ.get("PROBE_FULL_ROWS") == "1":
        # The row is random token ids at every position; only the labels past the row length are masked.
        # The engine packs the labelled span and forwards that, while the reference forwards the whole
        # row, so the two see different token counts through the recurrent mixer. Labelling every position
        # makes the spans identical, which separates that asymmetry from a difference in the engines.
        batch.labels = batch.input_ids.clone()
    batch_path = work / "batch.pt"
    save(batch, batch_path)
    body = pack(batch)
    print(
        f"model {model_path} ({layers} layers), {ROW_TOKENS:,}-token sequence, "
        f"{batch.active_tokens:,} real tokens, attn {attn}, fp32_lm_head {fp32_lm_head}, sp 1\n",
        flush=True,
    )

    print("[ref] single GPU ...", flush=True)
    reference = run_reference(
        model_path,
        batch_path,
        work / "ref.json",
        token_budget=65536,
        ce_chunk=2048,
        attn=attn,
        fp32_lm_head=fp32_lm_head,
    )
    print(f"[ref] loss {reference['loss']:.6f}\n", flush=True)

    rows = []
    with gateway(work, 1) as url:
        selected = [(label, mutate) for label, mutate in ARMS if not SELECTED or label in SELECTED]
        if REPEAT > 1:
            selected = [(f"{label} #{n + 1}", mutate) for label, mutate in selected for n in range(REPEAT)]
        repeats = {}
        for label, mutate in selected:
            training = copy.deepcopy(cfg.training)
            training["sp_size"] = 1
            training["n_gpus"] = 1
            if mutate is not None:
                mutate(training)
            payload = build_payload(training, model_path, SEED, attn_implementation=attn)
            print(f"[dss] {label} ...", flush=True)
            try:
                with running_job(url, payload) as job_id:
                    dss = fwd_bwd_step(url, job_id, body)
            except Exception as exc:  # a setting the engine rejects is a result, not a crash
                print(f"[dss] {label}: unavailable -- {type(exc).__name__}: {exc}\n", flush=True)
                rows.append((label, None))
                continue

            pairs, only_dss, only_ref = align(dss.grad_norms, reference["grad_norms"])
            ratios = sorted(a / b for _, a, b in pairs if b)
            diffs = [(abs(a - b), name) for name, a, b in pairs]
            worst, worst_name = max(diffs)
            repeats[label] = {name: a for name, a, _b in pairs}
            watched = {name: (a, b) for name, a, b in pairs if name in WATCH}
            failing = sorted(((d, n) for d, n in diffs if d > STATED_CRITERION_ABS), reverse=True)
            for d, name in failing:
                ref = next(b for n, _a, b in pairs if n == name)
                print(f"    {name:44} {d:>11.3e}  (reference norm {ref:.6f})", flush=True)
            rows.append(
                (
                    label,
                    (
                        ratios[len(ratios) // 2],
                        sum(1 for r in ratios if r > 1.0),
                        len(ratios),
                        sum(1 for d, _ in diffs if d > STATED_CRITERION_ABS),
                        worst,
                        worst_name,
                        abs(dss.avg_loss - reference["loss"]),
                        watched,
                    ),
                )
            )
            print(f"[dss] {label}: over 1e-3 {rows[-1][1][3]}  worst {worst:.3e} {worst_name}\n", flush=True)

    print(
        f"{'arm':32} {'median ratio':>13} {'above 1.0':>11} {'over 1e-3':>10} {'worst abs':>11} "
        f"{'loss delta':>11}  worst tensor"
    )
    for label, stats in rows:
        if stats is None:
            print(f"{label:32} {'unavailable':>13}")
            continue
        median_ratio, above, total, over, worst, worst_name, loss_delta, _ = stats
        print(
            f"{label:32} {median_ratio:>13.6f} {above:>7}/{total:<3} {over:>10} {worst:>11.3e} "
            f"{loss_delta:>11.3e}  {worst_name}"
        )
    if REPEAT > 1 and len(repeats) > 1:
        names = sorted(set.intersection(*(set(v) for v in repeats.values())))
        runs = list(repeats.values())
        ranges = sorted(((max(r[n] for r in runs) - min(r[n] for r in runs), n) for n in names), reverse=True)
        print("")
        print(
            f"Arctic Platform against itself over {len(runs)} runs of the same config, "
            f"{sum(1 for r, _ in ranges if r > STATED_CRITERION_ABS)} tensors past "
            f"{STATED_CRITERION_ABS:.0e}"
        )
        print(f"{'tensor':44} {'range':>11}")
        for rng, name in ranges[:10]:
            print(f"{name:44} {rng:>11.3e}")

    for tensor in WATCH:
        print("")
        print(f"{'arm':32} {tensor + ' Arctic Platform':>24} {'reference':>14} {'abs diff':>11}")
        for label, stats in rows:
            if stats and tensor in stats[7]:
                a, b = stats[7][tensor]
                print(f"{label:32} {a:>24.6f} {b:>14.6f} {abs(a - b):>11.3e}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(
        main(
            args[0] if args else "arctic_platform/correctness/configs/qwen3.8-27b/h200/train-sft-8gpus-2k.config",
            args[1] if len(args) > 1 else SPEC,
        )
    )
