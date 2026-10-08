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

"""Why is the run-to-run floor 3.8e-03 here when it was 0.073e-03 before?

The floor is the spread between two runs of one configuration on identical data, and it sets what any
cross-engine comparison can resolve. It moved by a factor of fifty while three things changed together: the
model went from 16 layers to 28, rows went from 65536 tokens to 2048, and the two runs stopped sharing a
job. Each arm below moves one of them.

The fourth arm asks whether the spread is kernel nondeterminism at all: under ``debug.full_determinism`` a
bf16 run is documented to repeat bit-exactly, except in the GatedDeltaNet layers and the expert combine,
which that flag does not pin. A floor that survives the flag lives in those.

Every arm reports the tensors carrying the largest spread, since a floor concentrated in the head or the
final norm is a different problem from one spread across the model.
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arctic_platform.correctness.harness.batches import build_batch  # noqa: E402
from arctic_platform.correctness.harness.config import load_config  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import build_payload  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import fwd_bwd_step  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import gateway  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import pack  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import pin_suite_determinism  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import running_job  # noqa: E402
from arctic_platform.correctness.harness.seeds import SEED  # noqa: E402
from arctic_platform.correctness.onboarding.synth_model import materialize  # noqa: E402

SOURCE = "/data-fast/base-models/Qwen/Qwen3.8-27B"
CACHE_ROOT = "/data-fast/base-models/synthetic"

# label, layers, row tokens, share one job between the two runs, full determinism
ARMS = [
    ("28L 2048 two jobs", 28, 2048, False, False),
    ("28L 2048 two jobs, determinism", 28, 2048, False, True),
    ("28L 65536 two jobs", 28, 65536, False, False),
    ("16L 65536 two jobs", 16, 65536, False, False),
    ("16L 65536 one job", 16, 65536, True, False),
]


def spread(first: dict, second: dict):
    shared = sorted(set(first) & set(second))
    diffs = [(abs(first[k] - second[k]), abs(first[k] - second[k]) / max(abs(first[k]), 1e-12), k) for k in shared]
    diffs.sort(reverse=True)
    max_abs = diffs[0][0] if diffs else float("nan")
    max_rel = max(d[1] for d in diffs) if diffs else float("nan")
    return max_abs, max_rel, diffs[:3], len(shared)


def measure(training, model_path, vocab, attn, n_gpus, row_tokens, share_job):
    body_bytes = pack(build_batch("floor", 1, row_tokens, vocab, seed=SEED))
    payload = build_payload(training, model_path, SEED, attn_implementation=attn)
    work = Path(tempfile.mkdtemp(prefix="floor-"))
    with gateway(work, n_gpus) as url:
        if share_job:
            with running_job(url, payload) as job_id:
                first = fwd_bwd_step(url, job_id, body_bytes)
                second = fwd_bwd_step(url, job_id, body_bytes)
        else:
            with running_job(url, payload) as job_id:
                first = fwd_bwd_step(url, job_id, body_bytes)
            with running_job(url, payload) as job_id:
                second = fwd_bwd_step(url, job_id, body_bytes)
    return spread(first.grad_norms, second.grad_norms)


def main(config_path: str) -> int:
    from transformers import AutoConfig

    cfg = load_config(Path(config_path))
    attn = cfg.training.get("attn_implementation", "flash_attention_3")

    rows = []
    for label, layers, row_tokens, share_job, determinism in ARMS:
        spec = materialize(SOURCE, f"{CACHE_ROOT}/synthetic-Qwen3.8-27B-{layers}L-seed{SEED}", layers)
        model_cfg = AutoConfig.from_pretrained(spec.cache_path, trust_remote_code=True)
        text_cfg = getattr(model_cfg, "text_config", model_cfg)
        vocab = text_cfg.vocab_size
        training = copy.deepcopy(cfg.training)
        if determinism:
            training["debug"] = pin_suite_determinism(dict(training.get("debug") or {}))

        head_dim = getattr(text_cfg, "head_dim", None)
        print(f"[floor] {label} (head_dim {head_dim}) ...", flush=True)
        try:
            max_abs, max_rel, worst, ncmp = measure(
                training, spec.cache_path, vocab, attn, cfg.n_gpus, row_tokens, share_job
            )
        except Exception as exc:
            print(f"[floor] {label}: raised {str(exc)[-160:]}\n", flush=True)
            rows.append((label, None, None, [], 0))
            continue
        rows.append((label, max_abs, max_rel, worst, ncmp))
        print(f"[floor] {label}: max_abs {max_abs:.3e}  max_rel {max_rel:.3e}  over {ncmp} tensors", flush=True)
        for d, r, name in worst:
            print(f"           {name[:58]:<58} abs {d:.3e}  rel {r:.3e}", flush=True)
        print(flush=True)

    print(f"{'arm':<34} {'max_abs':>11} {'max_rel':>11}  largest contributor")
    for label, max_abs, max_rel, worst, _ in rows:
        if max_abs is None:
            print(f"{label:<34} {'-':>11} {'-':>11}  (raised)")
            continue
        print(f"{label:<34} {max_abs:>11.3e} {max_rel:>11.3e}  {worst[0][2] if worst else '-'}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(main(args[0] if args else "qwen3.8-27b-8gpu-plain.config"))
