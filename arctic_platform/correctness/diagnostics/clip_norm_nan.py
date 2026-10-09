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

"""Is the global gradient norm the non-finite value that an lr=0 step writes into the weights?

At 28 layers a step at lr=0 leaves the model producing a non-finite loss, while every per-parameter gradient
norm the same step reports is finite, and two forward-backward passes without a step are bit-identical. A
quantity computed over the flattened gradient buffer rather than per parameter would behave that way, and
the clipping norm is the one the step consumes.

The engine reports it: ``/step`` returns ``grad_norm``, nulled when non-finite, and ``update_successful``.
Reading those distinguishes a non-finite clipping norm from a corruption elsewhere in the step, and running
once with clipping disabled says whether removing the consumer removes the failure.
"""

from __future__ import annotations

import copy
import math
import sys
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
from arctic_platform.correctness.harness.workdir import correctness_workdir  # noqa: E402
from arctic_platform.correctness.onboarding.synth_model import materialize  # noqa: E402

SOURCE = "/data-fast/base-models/Qwen/Qwen3.8-27B"
CACHE_ROOT = "/data-fast/base-models/synthetic"
ROW_TOKENS = 2048
LAYERS = 28
INTERESTING = ("grad_norm", "update_successful", "last_lr", "global_steps")


def arm(label, training, model_path, vocab, attn, n_gpus):
    body_bytes = pack(build_batch("probe", 1, ROW_TOKENS, vocab, seed=SEED))
    payload = build_payload(training, model_path, SEED, attn_implementation=attn)
    work = correctness_workdir("clipnorm-")
    from sp_gateway_harness import fwd_bwd_response
    from sp_gateway_harness import step

    print(f"[probe] {label} ...", flush=True)
    with gateway(work, n_gpus) as url:
        with running_job(url, payload) as job_id:
            before = float(fwd_bwd_response(url, job_id, body_bytes)["avg_loss"])
            print(f"[probe] {label}: loss before step {before:.6f}", flush=True)

            stepped = step(url, job_id, 0.0)
            reported = {k: stepped.get(k) for k in INTERESTING}
            print(f"[probe] {label}: step reported {reported}", flush=True)
            deltas = {k: v for k, v in stepped.items() if "weight_delta" in k}
            if deltas:
                print(f"[probe] {label}: weight deltas {deltas}", flush=True)

            try:
                after = float(fwd_bwd_response(url, job_id, body_bytes)["avg_loss"])
                print(f"[probe] {label}: loss after step {after:.6f} finite={math.isfinite(after)}\n", flush=True)
                return reported, f"{after:.6f}"
            except Exception as exc:
                text = str(exc)
                tail = text.split("FloatingPointError:")[-1].strip()[:90]
                print(f"[probe] {label}: loss after step raised: {tail}\n", flush=True)
                return reported, "non-finite"


def main(config_path: str) -> int:
    cfg = load_config(Path(config_path))
    attn = cfg.training.get("attn_implementation", "flash_attention_3")

    from transformers import AutoConfig

    spec = materialize(SOURCE, f"{CACHE_ROOT}/synthetic-Qwen3.8-27B-{LAYERS}L-seed{SEED}", LAYERS)
    model_cfg = AutoConfig.from_pretrained(spec.cache_path, trust_remote_code=True)
    vocab = getattr(model_cfg, "text_config", model_cfg).vocab_size

    unclipped = copy.deepcopy(cfg.training)
    unclipped["gradient_clipping"] = 0.0

    print(
        f"config {cfg.config_id}, {LAYERS}L, row {ROW_TOKENS:,} tokens, attn {attn}, "
        f"clipping as written {cfg.training.get('gradient_clipping')}\n",
        flush=True,
    )

    rows = []
    for label, training in (("clipping as written", cfg.training), ("clipping disabled", unclipped)):
        rows.append((label, *arm(label, training, spec.cache_path, vocab, attn, cfg.n_gpus)))

    print(f"{'arm':<22} {'grad_norm':>12} {'update_ok':>10}  loss after lr=0 step")
    for label, reported, after in rows:
        gn = reported.get("grad_norm")
        print(f"{label:<22} {str(gn):>12} {str(reported.get('update_successful')):>10}  {after}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(main(args[0] if args else "qwen3.8-27b-8gpu-plain.config"))
