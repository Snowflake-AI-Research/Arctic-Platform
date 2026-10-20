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

"""Does handing the reference the engine's sequence boundaries close the remaining disagreement?

The gated delta net reads its boundaries from two keyword arguments and builds neither. The engine passes
both, so its short convolution and delta rule run the varlen kernels; the reference passes neither, so they
run the batched ones. On a single sequence the two are the same mathematical function, and at module level
the stock forward given both arguments reproduces the engine's packed forward bit for bit.

This measures what that choice is worth end to end: one Arctic Platform run against two reference runs on the same
batch, differing only in whether the boundaries are passed.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from attention_ablation import build_ablated  # noqa: E402
from attention_ablation import layer_type_field  # noqa: E402

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
from arctic_platform.correctness.harness.workdir import correctness_workdir  # noqa: E402
from arctic_platform.correctness.onboarding.synth_model import materialize_pretrained  # noqa: E402

ROW_TOKENS = int(os.environ.get("PROBE_ROW_TOKENS", 2048))
LAYERS = int(os.environ.get("PROBE_LAYERS", 4))
SOURCE = "/data-fast/base-models/Qwen/Qwen3.8-27B"
CACHE_ROOT = "/data-fast/base-models/synthetic"
SPEC = "arctic_platform/correctness/specs/qwen3.8-27b-h200-train-sft-8gpus-2k.json"


def deep_merge(base: dict, over: dict) -> dict:
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def main(config_path: str) -> int:
    from transformers import AutoConfig

    cfg = load_config(Path(config_path))
    fp32_lm_head = bool(cfg.training.get("fp32_lm_head", False))
    attn = os.environ.get("PROBE_ATTN") or cfg.training.get("attn_implementation", "flash_attention_3")
    reference_attn = os.environ.get("PROBE_REF_ATTN", "sdpa")
    matmul_precision = os.environ.get("PROBE_MATMUL_PRECISION", "highest")
    sp = int(os.environ.get("PROBE_SP", 0) or cfg.training.get("sp_size", 1))
    # One sequence cannot fill eight data-parallel shards, so a single-sequence arm runs on one GPU.
    n_gpus = int(os.environ.get("PROBE_GPUS", 0) or cfg.training.get("n_gpus", 8))

    path = materialize_pretrained(SOURCE, f"{CACHE_ROOT}/Qwen3.8-27B-{LAYERS}L", LAYERS).cache_path
    # ``full`` removes the full-attention layers, which is the only way to take the engine's flash
    # attention and the reference's sdpa out of the comparison: whatever the last attention layer's
    # backward produces flows into every layer below it.
    if os.environ.get("PROBE_DROP") == "full":
        source_cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        types = layer_type_field(getattr(source_cfg, "text_config", source_cfg))
        full_types = {t for t in types if "full" in t or "self" in t or t == "attention"}
        path, _, _ = build_ablated(path, f"{CACHE_ROOT}/Qwen3.8-27B-{LAYERS}L-no-full-attn", full_types)
    model_cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    vocab = getattr(model_cfg, "text_config", model_cfg).vocab_size

    work = correctness_workdir("mixer-packing-")
    batch = build_batch("gas1", 1, ROW_TOKENS, vocab, seed=SEED)
    batch_path = work / "batch.pt"
    save(batch, batch_path)

    training = copy.deepcopy(cfg.training)
    override = os.environ.get("PROBE_ENGINE_OVERRIDE")
    if override:
        deep_merge(training, json.loads(override))
        print(f"engine override {override}", flush=True)
    training["sp_size"] = sp
    training["n_gpus"] = n_gpus
    # One or more engine float32 matmul precisions, measured against the same reference and the same
    # weights. Each host builds its own copy of an ablated model, so two precisions compared across two
    # jobs are compared across two sets of weights.
    precisions = [
        p for p in os.environ.get("PROBE_ENGINE_MATMUL", training.get("matmul_precision", "high")).split(",") if p
    ]
    print(
        f"{LAYERS} layers, {ROW_TOKENS:,} tokens in one sequence, {batch.active_tokens:,} scored, "
        f"engine on {attn} at sp {sp} across {n_gpus} GPU(s), reference on {reference_attn} with "
        f"float32 matmul precision {matmul_precision}",
        flush=True,
    )

    engine_runs = {}
    for precision in precisions:
        training["matmul_precision"] = precision
        payload = build_payload(training, path, SEED, attn_implementation=attn)
        with gateway(work, n_gpus) as url:
            with running_job(url, payload) as job_id:
                engine_runs[precision] = fwd_bwd_step(url, job_id, pack(batch))
        print(f"[dss] float32 matmul precision {precision}: loss {engine_runs[precision].avg_loss:.6f}", flush=True)

    # The engine against itself. A disagreement with the reference that is no larger than this is the
    # engine's own reproducibility and not a difference between the two engines.
    if len(precisions) == 2:
        left, right = (engine_runs[p] for p in precisions)
        pairs, _, _ = align(left.grad_norms, right.grad_norms)
        spread = sorted(((abs(a - b), name) for name, a, b in pairs), reverse=True)
        print("")
        print(
            f"--- the engine against itself, {precisions[0]} against {precisions[1]}: "
            f"{sum(1 for d, _ in spread if d > STATED_CRITERION_ABS)} of {len(pairs)} tensors over "
            f"{STATED_CRITERION_ABS:.0e}, worst {spread[0][0]:.3e} {spread[0][1]}",
            flush=True,
        )
        for diff, name in spread[:6]:
            print(f"{name:52} {diff:>11.3e}")

    mixer_arms = ((f"engine at {p}, reference as it runs today", p, False) for p in precisions)
    if os.environ.get("PROBE_MIXER_ARM") == "1":
        mixer_arms = [
            (f"engine at {p}, reference {r}", p, packing)
            for p in precisions
            for r, packing in (("as it runs today", False), ("given the engine's boundaries", True))
        ]
    references = {}
    for label, precision, packing in mixer_arms:
        dss = engine_runs[precision]
        if packing not in references:
            references[packing] = run_reference(
                path,
                batch_path,
                work / f"ref-{int(packing)}.json",
                token_budget=65536,
                ce_chunk=2048,
                attn=reference_attn,
                fp32_lm_head=fp32_lm_head,
                mixer_packing=packing,
                matmul_precision=matmul_precision,
            )
        reference = references[packing]
        pairs, only_dss, only_ref = align(dss.grad_norms, reference["grad_norms"])
        failures = sorted(
            ((abs(a - b), name, a, b) for name, a, b in pairs if abs(a - b) > STATED_CRITERION_ABS), reverse=True
        )
        print("")
        print(
            f"--- {label}: loss {reference['loss']:.6f}, {len(failures)} of {len(pairs)} tensors "
            f"over {STATED_CRITERION_ABS:.0e}"
        )
        print(f"{'tensor':52} {'dss':>13} {'reference':>13} {'abs diff':>11}")
        for diff, name, a, b in failures:
            print(f"{name:52} {a:>13.6f} {b:>13.6f} {diff:>11.3e}")
        if only_dss or only_ref:
            print(f"unmatched: {len(only_dss)} on the engine side, {len(only_ref)} on the reference side")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(
        main(args[0] if args else "arctic_platform/correctness/configs/qwen3.8-27b/h200/train-sft-8gpus-2k.config")
    )
