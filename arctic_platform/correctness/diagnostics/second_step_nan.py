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

"""Two steps at lr=0 on one job: does the first step's gradient poison the second step's weights?

The run-to-run floor is measured by stepping the same job twice on identical data at ``lr=0``, which should
leave the weights untouched. On this model the second step raises ``FloatingPointError`` while the first
returns a finite loss.

An infinite gradient explains that without any weight update: AdamW at ``lr=0`` forms ``lr * m / (sqrt(v) +
eps)``, and where a gradient overflowed, ``m`` and ``v`` are both infinite, so the quotient is NaN and the
product is ``0 * NaN`` rather than zero. The weight becomes NaN and every later forward inherits it.
So the first step's per-parameter gradient norms are the evidence, not the second step's traceback: if any
of them is non-finite the mechanism above is available, and if all of them are finite it is not. Both depths
run, since the deeper model is the one the single-GPU budget now selects.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arctic_platform.correctness.harness.batches import build_batch  # noqa: E402
from arctic_platform.correctness.harness.config import load_config  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import build_payload  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import fwd_bwd_step  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import gateway  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import pack  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import running_job  # noqa: E402
from arctic_platform.correctness.harness.seeds import SEED  # noqa: E402
from arctic_platform.correctness.harness.workdir import correctness_workdir  # noqa: E402
from arctic_platform.correctness.onboarding.synth_model import materialize  # noqa: E402

SOURCE = "/data-fast/base-models/Qwen/Qwen3.8-27B"
CACHE_ROOT = "/data-fast/base-models/synthetic"
ROW_TOKENS = 2048


def model_for(layers: int):
    from transformers import AutoConfig

    spec = materialize(SOURCE, f"{CACHE_ROOT}/synthetic-Qwen3.8-27B-{layers}L-seed{SEED}", layers)
    cfg = AutoConfig.from_pretrained(spec.cache_path, trust_remote_code=True)
    return spec.cache_path, getattr(cfg, "text_config", cfg).vocab_size


def describe(norms: dict) -> str:
    bad = {k: v for k, v in norms.items() if not math.isfinite(v)}
    finite = [v for v in norms.values() if math.isfinite(v)]
    largest = max(((v, k) for k, v in zip(norms.keys(), norms.values()) if math.isfinite(v)), default=(0, "-"))
    text = f"{len(norms)} norms, {len(bad)} non-finite"
    if bad:
        text += " (" + ", ".join(list(bad)[:3]) + ")"
    text += f", largest finite {largest[0]:.4g} on {largest[1]}"
    if finite:
        text += f", median {sorted(finite)[len(finite) // 2]:.4g}"
    return text


def main(config_path: str) -> int:
    cfg = load_config(Path(config_path))
    attn = cfg.training.get("attn_implementation", "flash_attention_3")
    print(f"config {cfg.config_id}, attn {attn}, n_gpus {cfg.n_gpus}, row {ROW_TOKENS:,} tokens\n", flush=True)
    for layers in (28, 16):
        model_path, vocab = model_for(layers)
        body_bytes = pack(build_batch("probe", 1, ROW_TOKENS, vocab, seed=SEED))
        payload = build_payload(cfg.training, model_path, SEED, attn_implementation=attn)
        work = correctness_workdir("twostep-")
        print(f"[probe] {layers}L ...", flush=True)
        try:
            with gateway(work, cfg.n_gpus) as url:
                with running_job(url, payload) as job_id:
                    first = fwd_bwd_step(url, job_id, body_bytes)
                    print(f"[probe] {layers}L step 1 loss {first.avg_loss:.6f}", flush=True)
                    print(f"[probe] {layers}L step 1 grads: {describe(first.grad_norms)}", flush=True)
                    second = fwd_bwd_step(url, job_id, body_bytes)
                    print(f"[probe] {layers}L step 2 loss {second.avg_loss:.6f}", flush=True)
                    print(f"[probe] {layers}L step 2 grads: {describe(second.grad_norms)}\n", flush=True)
        except Exception as exc:
            text = str(exc)
            marker = "FloatingPointError:"
            tail = text.split(marker)[-1].strip()[:120] if marker in text else text[:160]
            print(f"[probe] {layers}L raised: {tail}\n", flush=True)
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(main(args[0] if args else "qwen3.8-27b-8gpu-plain.config"))
