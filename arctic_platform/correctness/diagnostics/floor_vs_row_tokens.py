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

"""How much of the run-to-run spread comes from row length rather than depth?

Depth compounds: the spread is zero at the last layer, which the backward reaches first, and grows
monotonically toward layer 0. But depth is not the large term. Sixteen layers at 2048-token rows spreads
3.011e-03, against 0.073e-03 for the same 16 layers at 65536-token rows -- a factor of forty from row
length alone, where 16 to 28 layers moves it by a quarter.

Shorter rows mean fewer tokens per sequence-parallel rank: a 2048-token row over eight ranks leaves 256
slots and about 192 real tokens each. Every reduction in the backward then sums fewer, larger terms, and
the kernels that pick their split by token count pick differently.

This walks the row length at one depth and reports the spread against the tokens each rank actually holds,
so the relationship can be read off rather than inferred from two points.
"""

from __future__ import annotations

import statistics
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
LAYERS = 16
ROW_LENGTHS = (2048, 8192, 32768, 65536)


def main(config_path: str) -> int:
    from transformers import AutoConfig

    cfg = load_config(Path(config_path))
    attn = cfg.training.get("attn_implementation", "flash_attention_3")
    spec = materialize(SOURCE, f"{CACHE_ROOT}/synthetic-Qwen3.8-27B-{LAYERS}L-seed{SEED}", LAYERS)
    model_cfg = AutoConfig.from_pretrained(spec.cache_path, trust_remote_code=True)
    vocab = getattr(model_cfg, "text_config", model_cfg).vocab_size

    print(f"config {cfg.config_id}, {LAYERS}L, sp_size {cfg.sp_size}, two separate jobs per measurement\n", flush=True)

    table = []
    payload = build_payload(cfg.training, spec.cache_path, SEED, attn_implementation=attn)
    work = correctness_workdir("rowfloor-")
    with gateway(work, cfg.n_gpus) as url:
        for row_tokens in ROW_LENGTHS:
            batch = build_batch("floor", 1, row_tokens, vocab, seed=SEED)
            body_bytes = pack(batch)
            per_rank = batch.active_tokens / cfg.sp_size
            print(
                f"[rows] {row_tokens:,} tokens ({batch.active_tokens:,} real, {per_rank:.0f} per rank) ...", flush=True
            )
            with running_job(url, payload) as job_id:
                first = fwd_bwd_step(url, job_id, body_bytes)
            with running_job(url, payload) as job_id:
                second = fwd_bwd_step(url, job_id, body_bytes)

            shared = sorted(set(first.grad_norms) & set(second.grad_norms))
            diffs = [
                (
                    abs(first.grad_norms[k] - second.grad_norms[k]),
                    abs(first.grad_norms[k] - second.grad_norms[k]) / max(abs(first.grad_norms[k]), 1e-12),
                    k,
                )
                for k in shared
            ]
            max_abs = max(d[0] for d in diffs)
            max_rel = max(d[1] for d in diffs)
            median_rel = statistics.median(d[1] for d in diffs)
            worst = max(diffs)[2]
            table.append((row_tokens, batch.active_tokens, per_rank, max_abs, max_rel, median_rel, worst))
            print(
                f"[rows] {row_tokens:,}: max_abs {max_abs:.3e}  max_rel {max_rel:.3e}  "
                f"median_rel {median_rel:.3e}  worst {worst}\n",
                flush=True,
            )

    print(
        f"{'row tokens':>11} {'real':>9} {'per rank':>9} {'max_abs':>11} {'max_rel':>11} "
        f"{'median_rel':>11}  largest contributor"
    )
    for row_tokens, real, per_rank, max_abs, max_rel, median_rel, worst in table:
        print(
            f"{row_tokens:>11,} {real:>9,} {per_rank:>9.0f} {max_abs:>11.3e} {max_rel:>11.3e} "
            f"{median_rel:>11.3e}  {worst}"
        )
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(main(args[0] if args else "qwen3.8-27b-8gpu-plain.config"))
