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

"""How reproducible is the run-to-run gradient-norm floor, and does tensor size explain it?

The floor a spec records is a maximum over every parameter of the difference between two identical steps.
A maximum over hundreds of samples is a high-variance statistic, so a single draw of it cannot distinguish
"the engine got quieter" from "this draw missed the tail". This probe takes several consecutive draws
inside one job, so weights, data and gateway are held fixed and only reduction order varies, and reports
each draw with the parameter that dominated it and that parameter's element count.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arctic_platform.correctness.harness.arms import arms_for  # noqa: E402
from arctic_platform.correctness.harness.batches import build_batch  # noqa: E402
from arctic_platform.correctness.harness.batches import pack  # noqa: E402
from arctic_platform.correctness.harness.config import load_config  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import build_payload  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import fwd_bwd_step  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import gateway  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import running_job  # noqa: E402
from arctic_platform.correctness.harness.seeds import SEED  # noqa: E402
from arctic_platform.correctness.harness.spec import TestSpec  # noqa: E402
from arctic_platform.correctness.harness.workdir import correctness_workdir  # noqa: E402

DRAWS = 8


def numel_by_name(model_path: str) -> dict:
    import torch
    from transformers import AutoConfig
    from transformers import AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    return {n: p.numel() for n, p in model.named_parameters()}


def main(config_path: str) -> int:
    cfg = load_config(Path(config_path))
    spec = TestSpec.read(Path("arctic_platform/correctness/specs") / f"{cfg.config_id}.json")
    probe = arms_for(cfg.max_seq_len, cfg.n_gpus)[0]

    from transformers import AutoConfig

    model_cfg = AutoConfig.from_pretrained(spec.model.cache_path, trust_remote_code=True)
    vocab = getattr(model_cfg, "text_config", model_cfg).vocab_size
    sizes = numel_by_name(spec.model.cache_path)

    batch = build_batch(probe.name, probe.global_batch_size, probe.max_seq_len, vocab, seed=SEED)
    body = pack(batch)
    attn = next(iter(spec.tolerance)) if spec.tolerance else "flash_attention_3"
    payload = build_payload(cfg.training, spec.model.cache_path, SEED, attn_implementation=attn)

    tmp = correctness_workdir("floor-variance-")
    steps = []
    with gateway(tmp, cfg.n_gpus) as url:
        with running_job(url, payload) as job_id:
            for i in range(DRAWS):
                steps.append(fwd_bwd_step(url, job_id, body).grad_norms)
                print(f"[probe] step {i + 1}/{DRAWS} done", flush=True)

    print(
        f"\nconfig {cfg.config_id}, case {probe.name}, {probe.total_tokens:,} tokens, "
        f"{batch.pad_fraction:.1%} padding, lr=0, one gateway, one job"
    )
    print(
        f"{spec.model.param_count:,} parameters, {spec.model.num_hidden_layers} layers, "
        f"hash {spec.model.content_hash}\n"
    )

    print(f"{'draw':>4} {'max_abs':>11} {'max_rel':>11}  {'dominated by':<44} {'elements':>13}")
    abs_draws = []
    for i in range(len(steps) - 1):
        a, b = steps[i], steps[i + 1]
        shared = sorted(set(a) & set(b))
        worst = max(shared, key=lambda k: abs(a[k] - b[k]))
        max_abs = abs(a[worst] - b[worst])
        max_rel = max(abs(a[k] - b[k]) / max(abs(a[k]), 1e-12) for k in shared)
        abs_draws.append(max_abs)
        short = worst.split("module.")[-1]
        n = next((v for k, v in sizes.items() if k.endswith(short) or short.endswith(k)), 0)
        print(f"{i + 1:>4} {max_abs:>11.3e} {max_rel:>11.3e}  {short[:44]:<44} {n:>13,}")

    lo, hi = min(abs_draws), max(abs_draws)
    print(
        f"\nmax_abs over {len(abs_draws)} draws: min {lo:.3e}  median "
        f"{statistics.median(abs_draws):.3e}  max {hi:.3e}  spread {hi / max(lo, 1e-30):.1f}x"
    )

    # Relative spread against element count, to test whether wider tensors are quieter.
    last_a, last_b = steps[0], steps[1]
    rows = []
    for k in sorted(set(last_a) & set(last_b)):
        short = k.split("module.")[-1]
        n = next((v for kk, v in sizes.items() if kk.endswith(short) or short.endswith(kk)), 0)
        if n:
            rows.append((n, abs(last_a[k] - last_b[k]) / max(abs(last_a[k]), 1e-12)))
    rows.sort()
    print(f"\n{'element count bucket':>22} {'tensors':>8} {'median relative spread':>24}")
    buckets = [(0, 10**6), (10**6, 10**7), (10**7, 10**8), (10**8, 10**12)]
    for lo_n, hi_n in buckets:
        vals = [r for n, r in rows if lo_n <= n < hi_n]
        if vals:
            print(f"{f'{lo_n:,}-{hi_n:,}':>22} {len(vals):>8} {statistics.median(vals):>24.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "qwen3.8-27b-8gpu-offload.config"))
