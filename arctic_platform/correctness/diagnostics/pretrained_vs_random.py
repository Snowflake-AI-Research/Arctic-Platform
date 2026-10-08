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

"""Does the gradient-norm bias survive real weights?

Arctic Platform reports per-parameter gradient L2 norms about 4.7e-04 relative above the single-GPU reference, and the
figure holds across a 32x span of sequence length, across batch sizes, and across sequence-parallel degrees
8, 4, and 2. Inverting ||g + e|| ~ ||g||(1 + rho^2/2) on that median puts the per-element disagreement at
about 3.1e-02, four bf16 rounding units.

A freshly initialized model on uniform random tokens sits at maximum entropy, so each parameter's gradient
is a small residual of many opposing per-token contributions. Relative rounding error is largest exactly
there. Filling the same reduced config with the checkpoint's own weights removes that condition while
changing nothing else: identical architecture, identical batch, identical kernels on both sides.

A bias that collapses with real weights is the fixture cancelling its own signal. A bias that holds is in
the engine.
"""

from __future__ import annotations

import statistics
import sys
import tempfile
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
from arctic_platform.correctness.onboarding.synth_model import materialize_pretrained  # noqa: E402

ROW_TOKENS = 2048
SOURCE = "/data-fast/base-models/Qwen/Qwen3.8-27B"
CACHE_ROOT = "/data-fast/base-models/synthetic"
SPEC = "arctic_platform/correctness/specs/qwen3.8-27b-h200-train-sft-8gpus-2k.json"


def main(config_path: str, spec_path: str) -> int:
    from transformers import AutoConfig

    cfg = load_config(Path(config_path))
    spec = TestSpec.read(Path(spec_path))
    layers = spec.model.num_hidden_layers
    fp32_lm_head = bool(cfg.training.get("fp32_lm_head", False))
    attn = cfg.training.get("attn_implementation", "flash_attention_3")
    n_gpus = int(cfg.training.get("n_gpus", 8))

    source_cfg = AutoConfig.from_pretrained(SOURCE, trust_remote_code=True)
    source_text = getattr(source_cfg, "text_config", source_cfg)
    print(f"source {SOURCE}: {source_text.num_hidden_layers} layers, keeping {layers}", flush=True)

    pretrained = materialize_pretrained(SOURCE, f"{CACHE_ROOT}/pretrained-Qwen3.8-27B-{layers}L", layers)
    arms = (("random init (seed 1234)", spec.model.cache_path), ("checkpoint weights", pretrained.cache_path))

    model_cfg = AutoConfig.from_pretrained(spec.model.cache_path, trust_remote_code=True)
    vocab = getattr(model_cfg, "text_config", model_cfg).vocab_size

    work = Path(tempfile.mkdtemp(prefix="pretrained-vs-random-"))
    batch = build_batch("gas1", 1, ROW_TOKENS, vocab, seed=SEED)
    batch_path = work / "batch.pt"
    save(batch, batch_path)
    body = pack(batch)
    print(
        f"{ROW_TOKENS:,}-token sequence, {batch.active_tokens:,} real tokens, attn {attn}, "
        f"fp32_lm_head {fp32_lm_head}, sp {cfg.training.get('sp_size')}\n",
        flush=True,
    )

    rows = []
    for label, model_path in arms:
        print(f"[ref] {label} ...", flush=True)
        reference = run_reference(
            model_path,
            batch_path,
            work / f"ref-{label[:6]}.json",
            token_budget=65536,
            ce_chunk=2048,
            attn=attn,
            fp32_lm_head=fp32_lm_head,
        )
        print(f"[ref] {label}: loss {reference['loss']:.6f}", flush=True)

        payload = build_payload(cfg.training, model_path, SEED, attn_implementation=attn)
        print(f"[dss] {label} ...", flush=True)
        with gateway(work, n_gpus) as url:
            with running_job(url, payload) as job_id:
                dss = fwd_bwd_step(url, job_id, body)

        pairs, only_dss, only_ref = align(dss.grad_norms, reference["grad_norms"])
        ratios = sorted(a / b for _, a, b in pairs if b)
        median_ratio = ratios[len(ratios) // 2]
        above = sum(1 for r in ratios if r > 1.0)
        diffs = [(abs(a - b), name) for name, a, b in pairs]
        over = sum(1 for d, _ in diffs if d > STATED_CRITERION_ABS)
        worst, worst_name = max(diffs)
        median_norm = statistics.median(b for _, _, b in pairs)
        rows.append(
            (
                label,
                reference["loss"],
                median_norm,
                median_ratio,
                above,
                len(ratios),
                over,
                worst,
                worst_name,
                abs(dss.avg_loss - reference["loss"]),
            )
        )
        print(
            f"[dss] {label}: median ratio {median_ratio:.6f}  above 1.0 {above}/{len(ratios)}  "
            f"over 1e-3 {over}  worst {worst:.3e} {worst_name}"
            f"{f'  UNMATCHED {len(only_dss)}/{len(only_ref)}' if only_dss or only_ref else ''}\n",
            flush=True,
        )

    print(
        f"{'arm':26} {'ref loss':>10} {'median norm':>12} {'median ratio':>13} {'above 1.0':>11} "
        f"{'over 1e-3':>10} {'worst abs':>11} {'loss delta':>11}"
    )
    for label, loss, norm, ratio, above, total, over, worst, _, loss_delta in rows:
        print(
            f"{label:26} {loss:>10.4f} {norm:>12.4f} {ratio:>13.6f} {above:>7}/{total:<3} "
            f"{over:>10} {worst:>11.3e} {loss_delta:>11.3e}"
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
