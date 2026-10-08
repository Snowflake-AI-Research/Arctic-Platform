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

"""Which call corrupts the weights: the backward, or the optimizer step at lr=0?

A job that returns a finite loss on its first forward-backward-step raises FloatingPointError on its second,
at 28 layers but not at 16, with every gradient norm from the first step finite. Two calls are candidates
for introducing the non-finite value, and they can be separated by running them apart.

Phases on one job, deepest model only:

1. forward-backward, no step. Twice. Identical finite losses mean the backward is repeatable on its own.
2. one optimizer step at lr=0, which is supposed to leave every weight where it is.
3. forward-backward again. A non-finite loss here places the corruption in the step.

Gradients accumulate across phase 1 because nothing zeroes them, which does not affect the loss.
"""

from __future__ import annotations

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
LAYERS = 28


def main(config_path: str) -> int:
    from sp_gateway_harness import fwd_bwd_response
    from sp_gateway_harness import step

    cfg = load_config(Path(config_path))
    attn = cfg.training.get("attn_implementation", "flash_attention_3")

    from transformers import AutoConfig

    spec = materialize(SOURCE, f"{CACHE_ROOT}/synthetic-Qwen3.8-27B-{LAYERS}L-seed{SEED}", LAYERS)
    model_cfg = AutoConfig.from_pretrained(spec.cache_path, trust_remote_code=True)
    vocab = getattr(model_cfg, "text_config", model_cfg).vocab_size
    body_bytes = pack(build_batch("probe", 1, ROW_TOKENS, vocab, seed=SEED))
    payload = build_payload(cfg.training, spec.cache_path, SEED, attn_implementation=attn)

    print(f"config {cfg.config_id}, {LAYERS}L, row {ROW_TOKENS:,} tokens, attn {attn}\n", flush=True)

    work = Path(tempfile.mkdtemp(prefix="phase-"))
    with gateway(work, cfg.n_gpus) as url:
        with running_job(url, payload) as job_id:
            for label in ("fwd-bwd #1 (no step)", "fwd-bwd #2 (no step)"):
                try:
                    loss = float(fwd_bwd_response(url, job_id, body_bytes)["avg_loss"])
                    print(f"[probe] {label:<28} loss {loss:.6f} finite={math.isfinite(loss)}", flush=True)
                except Exception as exc:
                    print(f"[probe] {label:<28} raised: {str(exc)[-150:]}", flush=True)
                    return 1

            print("[probe] optimizer step at lr=0 ...", flush=True)
            stepped = step(url, job_id, 0.0)
            norms = (stepped.get("gradient_norms_per_param") or {}).values()
            bad = sum(1 for v in norms if not math.isfinite(float(v)))
            print(f"[probe] step returned {len(list(norms))} grad norms, {bad} non-finite", flush=True)

            try:
                loss = float(fwd_bwd_response(url, job_id, body_bytes)["avg_loss"])
                print(
                    f"[probe] {'fwd-bwd #3 (after step)':<28} loss {loss:.6f} finite={math.isfinite(loss)}", flush=True
                )
            except Exception as exc:
                text = str(exc)
                marker = "FloatingPointError:"
                tail = text.split(marker)[-1].strip()[:130] if marker in text else text[-150:]
                print(f"[probe] {'fwd-bwd #3 (after step)':<28} raised: {tail}", flush=True)
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(main(args[0] if args else "qwen3.8-27b-8gpu-plain.config"))
