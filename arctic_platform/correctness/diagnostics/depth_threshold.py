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

"""Where between 16 and 28 layers does an lr=0 step start breaking the next forward, and does memory explain it?

Established: the forward-backward is repeatable, every per-parameter gradient norm is finite, the global
gradient norm is finite at 44.5, clipping off changes nothing, the sampled weight delta is exactly zero, and
the step reports success. Sixteen layers survives two steps; twenty-eight does not. Both a bf16-master GPU
AdamW and an fp32-master CPU Adam fail at the larger size, so the optimizer implementation is not the
variable either.

Depth is, and depth moves several things at once: parameter count, the flattened gradient buffer, and the
memory the step needs. This walks the depths and reports peak memory beside the outcome, so a threshold that
coincides with a memory ceiling is visible as one.
"""

from __future__ import annotations

import copy
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sp"))

from arctic_platform.correctness.harness.batches import build_batch  # noqa: E402
from arctic_platform.correctness.harness.config import load_config  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import build_payload  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import gateway  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import pack  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import running_job  # noqa: E402
from arctic_platform.correctness.harness.seeds import SEED  # noqa: E402
from arctic_platform.correctness.onboarding.synth_model import materialize  # noqa: E402

SOURCE = "/data-fast/base-models/Qwen/Qwen3.8-27B"
CACHE_ROOT = "/data-fast/base-models/synthetic"
ROW_TOKENS = 2048
DEPTHS = (16, 20, 24, 28)


def peak_of(payload_metrics: dict) -> str:
    for key in ("step_peak_memory_gib", "peak_allocated_gib", "cuda_peak_allocated_gib"):
        if key in payload_metrics:
            return f"{float(payload_metrics[key]):.1f} GiB"
    keys = [k for k in payload_metrics if "peak" in k.lower()]
    return ", ".join(f"{k}={payload_metrics[k]}" for k in keys[:3]) or "-"


def main(config_path: str) -> int:
    from sp_gateway_harness import fwd_bwd_response
    from sp_gateway_harness import step
    from transformers import AutoConfig

    cfg = load_config(Path(config_path))
    attn = cfg.training.get("attn_implementation", "flash_attention_3")
    training = copy.deepcopy(cfg.training)
    training["step_peak_memory_log"] = True
    training["training_memory_telemetry"] = True

    print(f"config {cfg.config_id}, row {ROW_TOKENS:,} tokens, attn {attn}, n_gpus {cfg.n_gpus}\n", flush=True)

    rows = []
    for layers in DEPTHS:
        spec = materialize(SOURCE, f"{CACHE_ROOT}/synthetic-Qwen3.8-27B-{layers}L-seed{SEED}", layers)
        model_cfg = AutoConfig.from_pretrained(spec.cache_path, trust_remote_code=True)
        vocab = getattr(model_cfg, "text_config", model_cfg).vocab_size
        body_bytes = pack(build_batch("probe", 1, ROW_TOKENS, vocab, seed=SEED))
        payload = build_payload(training, spec.cache_path, SEED, attn_implementation=attn)
        work = Path(tempfile.mkdtemp(prefix=f"depth{layers}-"))

        print(f"[probe] {layers}L ({spec.param_count/1e9:.2f}B params) ...", flush=True)
        outcome, gnorm, peak = "?", "?", "-"
        try:
            with gateway(work, cfg.n_gpus) as url:
                with running_job(url, payload) as job_id:
                    first = fwd_bwd_response(url, job_id, body_bytes)
                    peak = peak_of(first.get("metrics") or {})
                    stepped = step(url, job_id, 0.0)
                    gnorm = stepped.get("grad_norm")
                    after = float(fwd_bwd_response(url, job_id, body_bytes)["avg_loss"])
                    outcome = f"{after:.6f}" if math.isfinite(after) else "non-finite"
        except Exception as exc:
            text = str(exc)
            outcome = "non-finite" if "non-finite" in text else text[-70:]
        rows.append((layers, spec.param_count, gnorm, peak, outcome))
        print(
            f"[probe] {layers}L -> loss after lr=0 step: {outcome} (grad_norm {gnorm}, fwd-bwd peak {peak})\n",
            flush=True,
        )

    print(f"{'layers':>7} {'params':>14} {'grad_norm':>12} {'fwd-bwd peak':>16}  loss after lr=0 step")
    for layers, params, gnorm, peak, outcome in rows:
        gn = f"{gnorm:.4f}" if isinstance(gnorm, (int, float)) else str(gnorm)
        print(f"{layers:>7} {params:>14,} {gn:>12} {peak:>16}  {outcome}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(main(args[0] if args else "qwen3.8-27b-8gpu-plain.config"))
