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

"""Does sequence length alone move Arctic Platform away from the single-GPU reference?

Holds the model, the config, the engine code, and the comparison rule fixed, and varies only the length of
the single sequence in the batch. Both arms carry one sequence, so gradient accumulation depth, padding
fraction, and parallel topology are identical; the per-rank token count is what changes.

Reports the median Arctic Platform/reference ratio and the count over the 1e-3 criterion for each length, so a length
effect appears as a trend in the median rather than as a verdict that could also move with the gate.
"""

from __future__ import annotations

import statistics
import sys
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
from arctic_platform.correctness.harness.workdir import correctness_workdir  # noqa: E402

LENGTHS = (2048, 8192, 32768, 65536)
SPEC = "arctic_platform/correctness/specs/qwen3.8-27b-h200-train-sft-8gpus-2k.json"


def main(config_path: str, spec_path: str) -> int:
    from transformers import AutoConfig

    cfg = load_config(Path(config_path))
    spec = TestSpec.read(Path(spec_path))
    model_path = spec.model.cache_path
    fp32_lm_head = bool(cfg.training.get("fp32_lm_head", False))
    attn = cfg.training.get("attn_implementation", "flash_attention_3")

    model_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vocab = getattr(model_cfg, "text_config", model_cfg).vocab_size

    print(
        f"model {model_path} ({spec.model.num_hidden_layers} layers), config {cfg.config_id}, "
        f"sp_size {cfg.sp_size}, fp32_lm_head {fp32_lm_head}\n",
        flush=True,
    )

    work = correctness_workdir("seqlen-ab-")
    rows = []

    # Every reference runs before the gateway starts, so the single-GPU pass never shares the node with it.
    references = {}
    for length in LENGTHS:
        batch = build_batch("gas1", 1, length, vocab, seed=SEED)
        path = work / f"batch-{length}.pt"
        save(batch, path)
        print(f"[ref] {length:,} tokens ({batch.active_tokens:,} real) ...", flush=True)
        references[length] = (
            batch,
            path,
            run_reference(
                model_path,
                path,
                work / f"ref-{length}.json",
                token_budget=65536,
                ce_chunk=2048,
                attn="sdpa",
                fp32_lm_head=fp32_lm_head,
            ),
        )

    payload = build_payload(cfg.training, model_path, SEED, attn_implementation=attn)
    with gateway(work, cfg.n_gpus) as url:
        for length in LENGTHS:
            batch, path, reference = references[length]
            with running_job(url, payload) as job_id:
                dss = fwd_bwd_step(url, job_id, pack(batch))

            pairs, only_dss, only_ref = align(dss.grad_norms, reference["grad_norms"])
            ratios = sorted(a / b for _, a, b in pairs if b)
            diffs = [(abs(a - b), name) for name, a, b in pairs]
            over = [d for d, _ in diffs if d > STATED_CRITERION_ABS]
            worst, worst_name = max(diffs)
            median_norm = statistics.median(b for _, _, b in pairs)
            rows.append(
                (
                    length,
                    batch.active_tokens,
                    batch.active_tokens / cfg.sp_size,
                    ratios[len(ratios) // 2],
                    len(over),
                    len(pairs),
                    worst,
                    worst_name,
                    median_norm,
                )
            )
            print(
                f"[dss] {length:,}: median ratio {ratios[len(ratios) // 2]:.6f}  "
                f"over 1e-3 {len(over)}/{len(pairs)}  worst {worst:.3e} {worst_name}  "
                f"median ref norm {median_norm:.4f}"
                f"{f'  UNMATCHED {len(only_dss)}/{len(only_ref)}' if only_dss or only_ref else ''}\n",
                flush=True,
            )

    print(
        f"{'seq len':>9} {'real':>9} {'per rank':>9} {'median ratio':>13} {'over 1e-3':>10} "
        f"{'worst abs':>11} {'median norm':>12}  worst tensor"
    )
    for length, real, per_rank, ratio, over, total, worst, worst_name, median_norm in rows:
        print(
            f"{length:>9,} {real:>9,} {per_rank:>9.0f} {ratio:>13.6f} {over:>6}/{total:<3} "
            f"{worst:>11.3e} {median_norm:>12.4f}  {worst_name}"
        )
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(
        main(
            args[0] if args else "arctic_platform/correctness/configs/qwen3.8-27b/h200/train-sft-8gpus-2k.config",
            args[1] if len(args) > 1 else SPEC,
        )
    )
