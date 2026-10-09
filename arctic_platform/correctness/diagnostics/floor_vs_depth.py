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

"""Why does adding layers widen the run-to-run spread?

Two runs of one configuration on identical data differ by some amount, and that floor bounds what a
cross-engine comparison can resolve. At 16 layers it is 0.073e-03 absolute; at 28 it is 3.811e-03. The
arithmetic per layer did not change, so the question is what depth does to it.

Two mechanisms both widen the floor and they are distinguishable by what they do to the *relative* spread
and by where it sits:

- Magnitudes grow. A randomly initialized residual stack grows its activations with depth, so gradients are
  larger at 28 layers and an unchanged relative error reads as a larger absolute one. Relative spread would
  be flat across depths while the gradient norms climb.
- Perturbations compound. A difference introduced in the backward at one layer feeds every layer below it,
  so deeper stacks amplify what shallower ones absorb. Relative spread would climb with depth, and within
  one model it would be worst at layer 0, which the backward reaches last.

Each depth reports both, plus the spread profiled by layer index, so the answer comes from the shape rather
than from a single maximum.
"""

from __future__ import annotations

import re
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
ROW_TOKENS = 2048
DEPTHS = (16, 20, 24, 28)
LAYER_RE = re.compile(r"layers\.(\d+)\.")


def measure(training, model_path, vocab, attn, n_gpus):
    """Two runs, separate jobs, identical data."""
    body_bytes = pack(build_batch("floor", 1, ROW_TOKENS, vocab, seed=SEED))
    payload = build_payload(training, model_path, SEED, attn_implementation=attn)
    work = correctness_workdir("depthfloor-")
    with gateway(work, n_gpus) as url:
        with running_job(url, payload) as job_id:
            first = fwd_bwd_step(url, job_id, body_bytes)
        with running_job(url, payload) as job_id:
            second = fwd_bwd_step(url, job_id, body_bytes)
    return first.grad_norms, second.grad_norms


def report(layers, first, second):
    shared = sorted(set(first) & set(second))
    rows = [
        (abs(first[k] - second[k]), abs(first[k] - second[k]) / max(abs(first[k]), 1e-12), k, first[k]) for k in shared
    ]

    max_abs = max(r[0] for r in rows)
    max_rel = max(r[1] for r in rows)
    median_rel = statistics.median(r[1] for r in rows)
    median_norm = statistics.median(r[3] for r in rows)
    print(
        f"[depth] {layers}L: max_abs {max_abs:.3e}  max_rel {max_rel:.3e}  "
        f"median_rel {median_rel:.3e}  median grad norm {median_norm:.4g}  ({len(rows)} tensors)",
        flush=True,
    )

    for d, r, name, norm in sorted(rows, reverse=True)[:3]:
        print(f"        {name[:56]:<56} abs {d:.3e}  rel {r:.3e}  norm {norm:.4g}", flush=True)

    by_layer = {}
    for _d, r, name, _norm in rows:
        m = LAYER_RE.search(name)
        if m:
            by_layer.setdefault(int(m.group(1)), []).append(r)
    if by_layer:
        indices = sorted(by_layer)
        sample = [indices[0], indices[len(indices) // 2], indices[-1]]
        profile = "  ".join(f"layer {i}: {statistics.median(by_layer[i]):.3e}" for i in sample)
        print(f"        median relative spread by layer -- {profile}", flush=True)
    print(flush=True)
    return max_abs, max_rel, median_rel, median_norm


def main(config_path: str) -> int:
    from transformers import AutoConfig

    cfg = load_config(Path(config_path))
    attn = cfg.training.get("attn_implementation", "flash_attention_3")
    print(
        f"config {cfg.config_id}, row {ROW_TOKENS:,} tokens, attn {attn}, two separate jobs per measurement\n",
        flush=True,
    )

    table = []
    for layers in DEPTHS:
        spec = materialize(SOURCE, f"{CACHE_ROOT}/synthetic-Qwen3.8-27B-{layers}L-seed{SEED}", layers)
        model_cfg = AutoConfig.from_pretrained(spec.cache_path, trust_remote_code=True)
        text_cfg = getattr(model_cfg, "text_config", model_cfg)
        print(
            f"[depth] {layers}L ({spec.param_count/1e9:.2f}B params, "
            f"head_dim {getattr(text_cfg, 'head_dim', None)}) ...",
            flush=True,
        )
        try:
            first, second = measure(cfg.training, spec.cache_path, text_cfg.vocab_size, attn, cfg.n_gpus)
        except Exception as exc:
            print(f"[depth] {layers}L raised: {str(exc)[-150:]}\n", flush=True)
            continue
        table.append((layers, *report(layers, first, second)))

    print(f"{'layers':>7} {'max_abs':>11} {'max_rel':>11} {'median_rel':>11} {'median norm':>12}")
    for layers, max_abs, max_rel, median_rel, median_norm in table:
        print(f"{layers:>7} {max_abs:>11.3e} {max_rel:>11.3e} {median_rel:>11.3e} {median_norm:>12.4g}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(main(args[0] if args else "qwen3.8-27b-8gpu-plain.config"))
