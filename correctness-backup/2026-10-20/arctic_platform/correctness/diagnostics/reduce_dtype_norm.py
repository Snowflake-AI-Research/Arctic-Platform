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

"""Leave one precision knob out of a passing config and see which tensors stop matching.

The final RMSNorm's gradient is a sum over every token on every sequence-parallel rank, so it is the
tensor most exposed to the precision of whatever accumulates it. A config carrying fp32_lm_head, fp32
gradient communication and an fp32 reduce dtype matches the single-GPU reference on that tensor; a config
carrying none of them is off by about 19%. Removing one knob at a time from the passing config says which
one is responsible, which adding knobs to the failing config cannot, because the two configs differ in
more than these three settings.

Each knob is named on the command line; the baseline always runs first so the comparison is against this
node and this model rather than against a number from an earlier run.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arctic_platform.correctness.harness.arms import arms_for  # noqa: E402
from arctic_platform.correctness.harness.batches import build_batch  # noqa: E402
from arctic_platform.correctness.harness.batches import save  # noqa: E402
from arctic_platform.correctness.harness.config import load_config  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import build_payload  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import fwd_bwd_step  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import gateway  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import pack  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import running_job  # noqa: E402
from arctic_platform.correctness.harness.names import align  # noqa: E402
from arctic_platform.correctness.harness.seeds import SEED  # noqa: E402
from arctic_platform.correctness.harness.spec import STATED_CRITERION_ABS  # noqa: E402
from arctic_platform.correctness.harness.spec import TestSpec  # noqa: E402
from arctic_platform.correctness.harness.workdir import correctness_workdir  # noqa: E402

# label -> (where it lives, key). "ds" is the DeepSpeed config block, "tc" the training config itself.
KNOBS = {
    "fp32_lm_head": ("tc", "fp32_lm_head"),
    "comm_fp32": ("ds", "communication_data_type"),
    "reduce_fp32": ("tc", "reduce_dtype"),
}

WATCH = "norm.weight"


def without(training: dict, label: str) -> dict:
    where, key = KNOBS[label]
    out = copy.deepcopy(training)
    target = out.get("ds_config", {}) if where == "ds" else out
    target.pop(key, None)
    if label == "fp32_lm_head":
        # Removing the key would fall back to whatever the engine defaults to; state the off value.
        out["fp32_lm_head"] = False
    return out


def run(training: dict, model_path: str, attn: str, n_gpus: int, body: bytes):
    payload = build_payload(training, model_path, SEED, attn_implementation=attn)
    work = correctness_workdir("leaveout-")
    with gateway(work, n_gpus) as url:
        with running_job(url, payload) as job_id:
            return fwd_bwd_step(url, job_id, body)


def main(config_path: str, drop: list) -> int:
    cfg = load_config(Path(config_path))
    spec = TestSpec.read(Path("arctic_platform/correctness/specs") / f"{cfg.config_id}.json")
    probe = arms_for(cfg.max_seq_len, cfg.n_gpus)[0]

    from transformers import AutoConfig

    model_cfg = AutoConfig.from_pretrained(spec.model.cache_path, trust_remote_code=True)
    vocab = getattr(model_cfg, "text_config", model_cfg).vocab_size
    batch = build_batch(probe.name, probe.global_batch_size, probe.max_seq_len, vocab, seed=SEED)
    body = pack(batch)
    attn = next(iter(spec.tolerance)) if spec.tolerance else "flash_attention_3"

    print(
        f"config {cfg.config_id}, case {probe.name}: gbs {probe.global_batch_size}, "
        f"seq len {probe.max_seq_len:,}, {batch.pad_fraction:.1%} padding"
    )
    print(f"criterion {STATED_CRITERION_ABS:.0e} absolute on per-parameter gradient L2 norms")
    knobs = {
        key: (cfg.training.get("ds_config", {}) if where == "ds" else cfg.training).get(config_key)
        for key, (where, config_key) in KNOBS.items()
    }
    print(f"knobs present in this config: {knobs}\n", flush=True)

    tmp = correctness_workdir("leaveout-ref-")
    batch_path = tmp / "batch.pt"
    save(batch, batch_path)
    print("[probe] reference (single GPU, no parallelism) ...", flush=True)
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "arctic_platform.correctness.reference",
            "--model",
            spec.model.cache_path,
            "--batch",
            str(batch_path),
            "--out",
            str(tmp / "ref.json"),
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        print(out.stdout[-3000:])
        print(out.stderr[-3000:])
        return 1
    reference = json.loads((tmp / "ref.json").read_text())
    ref_norms = reference["grad_norms"]
    print(f"[probe] reference loss {reference['loss']:.6f}\n", flush=True)

    arms = [("baseline (all knobs as written)", cfg.training)]
    arms += [(f"without {label}", without(cfg.training, label)) for label in drop]

    rows = []
    for label, training in arms:
        step = run(training, spec.model.cache_path, attn, cfg.n_gpus, body)
        pairs, _, _ = align(step.grad_norms, ref_norms)
        failing = sorted(((abs(a - b), k) for k, a, b in pairs if abs(a - b) > STATED_CRITERION_ABS), reverse=True)
        watch = next(((a, b) for k, a, b in pairs if k == WATCH), None)
        rows.append((label, step, len(pairs), failing, watch))
        ratio = watch[0] / watch[1] if watch and watch[1] else float("nan")
        print(
            f"[probe] {label:<34} loss {step.avg_loss:.6f}  {len(failing):>3}/{len(pairs)} failing  "
            f"{WATCH} ratio {ratio:.6f}",
            flush=True,
        )

    print(f"\n{'arm':<34} {'failing':>9} {WATCH + ' Arctic Platform':>18} {'reference':>13} {'ratio':>10}")
    for label, _step, ncmp, failing, watch in rows:
        d, r = watch if watch else (float("nan"), float("nan"))
        print(f"{label:<34} {f'{len(failing)}/{ncmp}':>9} {d:>18.8f} {r:>13.8f} {d / r if r else float('nan'):>10.6f}")

    for label, _step, _ncmp, failing, _watch in rows:
        if failing:
            print(f"\nworst tensors, {label}:")
            for diff, name in failing[:8]:
                print(f"  {name[:56]:<56} abs {diff:.3e}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    cfgp = args[0] if args else "qwen3.8-27b-8gpu-offload.config"
    raise SystemExit(main(cfgp, args[1:] or ["fp32_lm_head"]))
