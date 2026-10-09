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

"""Which difference produces the 0.047% gradient-norm bias?

Arctic Platform reports per-parameter gradient L2 norms about 4.7e-04 relative above the single-GPU reference. The
figure holds across a 32x span of sequence length, across batch sizes, and across sequence-parallel degrees
8, 4, and 2, and roughly 310 of 374 tensors sit above the reference rather than all of them. That is how an
independent perturbation behaves and not how a scale factor behaves: for a gradient g and a perturbation e
uncorrelated with it, ||g + e|| is about ||g||(1 + rho^2/2), where rho is the per-element relative size
of e.

Two candidate sources are testable without the gateway, by running the reference against itself with one
thing changed:

- the attention kernel, since the reference runs sdpa while the engine runs flash_attention_3
- the gradient accumulator's dtype

Each arm reports the statistics used against Arctic Platform, so the magnitudes are directly comparable.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from arctic_platform.correctness.harness.batches import build_batch  # noqa: E402
from arctic_platform.correctness.harness.batches import load  # noqa: E402
from arctic_platform.correctness.harness.batches import save  # noqa: E402
from arctic_platform.correctness.harness.config import load_config  # noqa: E402
from arctic_platform.correctness.harness.seeds import SEED  # noqa: E402
from arctic_platform.correctness.harness.spec import STATED_CRITERION_ABS  # noqa: E402
from arctic_platform.correctness.harness.spec import TestSpec  # noqa: E402
from arctic_platform.correctness.harness.workdir import correctness_workdir  # noqa: E402
from arctic_platform.correctness.reference.hf_single_gpu import run as reference_run  # noqa: E402

ROW_TOKENS = 2048
SPEC = "arctic_platform/correctness/specs/qwen3.8-27b-h200-train-sft-8gpus-2k.json"

ARMS = (
    ("baseline: sdpa, fp32 accumulator", "sdpa", None),
    ("flash_attention_3 instead of sdpa", "flash_attention_3", None),
    ("bfloat16 accumulator instead of fp32", "sdpa", torch.bfloat16),
)


def compare(base, other):
    shared = sorted(set(base) & set(other))
    ratios = sorted(other[k] / base[k] for k in shared if base[k])
    above = sum(1 for r in ratios if r > 1.0)
    worst, worst_name = max((abs(other[k] - base[k]), k) for k in shared)
    over = sum(1 for k in shared if abs(other[k] - base[k]) > STATED_CRITERION_ABS)
    return ratios[len(ratios) // 2], above, len(ratios), over, worst, worst_name


def main(config_path: str, spec_path: str) -> int:
    from transformers import AutoConfig

    cfg = load_config(Path(config_path))
    spec = TestSpec.read(Path(spec_path))
    model_path = spec.model.cache_path
    fp32_lm_head = bool(cfg.training.get("fp32_lm_head", False))

    model_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vocab = getattr(model_cfg, "text_config", model_cfg).vocab_size

    work = correctness_workdir("bias-source-")
    batch = build_batch("gas1", 1, ROW_TOKENS, vocab, seed=SEED)
    path = work / "batch.pt"
    save(batch, path)

    print(
        f"model {model_path} ({spec.model.num_hidden_layers} layers), {ROW_TOKENS:,}-token sequence, "
        f"{batch.active_tokens:,} real tokens, fp32_lm_head {fp32_lm_head}",
        flush=True,
    )
    print("", flush=True)

    results = []
    for label, attn, accum in ARMS:
        print(f"[ref] {label} ...", flush=True)
        kwargs = {"accumulate_dtype": accum} if accum is not None else {}
        try:
            out = reference_run(
                model_path,
                load(path),
                token_budget=65536,
                ce_chunk=2048,
                attn_implementation=attn,
                fp32_lm_head=fp32_lm_head,
                **kwargs,
            )
        except Exception as exc:  # an unavailable kernel is a result, not a crash
            print(f"[ref] {label}: unavailable -- {type(exc).__name__}: {exc}", flush=True)
            results.append((label, None, None))
            continue
        print(f"[ref] {label}: loss {out.loss:.6f}", flush=True)
        results.append((label, out.loss, out.grad_norms))

    base_label, base_loss, base_norms = results[0]
    print("", flush=True)
    print(
        f"{'arm':38} {'median ratio':>13} {'above 1.0':>11} {'over 1e-3':>10} {'worst abs':>11} "
        f"{'loss delta':>11}  worst tensor"
    )
    for label, loss, norms in results[1:]:
        if norms is None:
            print(f"{label:38} {'unavailable':>13}")
            continue
        median_ratio, above, total, over, worst, worst_name = compare(base_norms, norms)
        print(
            f"{label:38} {median_ratio:>13.6f} {above:>7}/{total:<3} {over:>10} {worst:>11.3e} "
            f"{abs(loss - base_loss):>11.3e}  {worst_name}"
        )
    print("", flush=True)
    print(f"median reference norm {statistics.median(base_norms.values()):.4f}")
    print(
        "Arctic Platform against this baseline measures median ratio 1.000491, 314/374 above 1.0, "
        "worst 2.332e-02, loss delta 1.602e-03."
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
