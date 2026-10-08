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

"""Does the residual gradient-norm disagreement track sequence parallelism?

With the reduced model carrying the checkpoint's own weights, the systematic offset between Arctic Platform and the
single-GPU reference is gone -- the median ratio sits at 0.9997 with the tensors split evenly either side
of 1.0 -- but individual tensors still disagree by up to 1%, enough to cross a 1e-3 absolute gate on the
larger norms. Self-attention layers are a quarter of the stack and supply about half of those tensors.

Varying the sequence-parallel degree with data parallelism held at one changes only how many ranks a single
sequence is split across, and therefore how attention and the per-token reductions are grouped. At sp 1 the
split is gone altogether. A disagreement that falls with sp points at the sequence-parallel path. A
disagreement that holds at sp 1 is in numerics a single rank produces on its own.
"""

from __future__ import annotations

import copy
import os
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

SP_DEGREES = (8, 4, 2, 1)
ROW_TOKENS = int(os.environ.get("PROBE_ROW_TOKENS", 2048))
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
        f"model {model_path} ({spec.model.num_hidden_layers} layers), {ROW_TOKENS:,}-token sequence, "
        f"fp32_lm_head {fp32_lm_head}\n",
        flush=True,
    )

    work = Path(tempfile.mkdtemp(prefix="bias-sp-"))
    batch = build_batch("gas1", 1, ROW_TOKENS, vocab, seed=SEED)
    batch_path = work / "batch.pt"
    save(batch, batch_path)
    body = pack(batch)

    print(f"[ref] single GPU, {batch.active_tokens:,} real tokens ...", flush=True)
    reference = run_reference(
        model_path,
        batch_path,
        work / "ref.json",
        token_budget=65536,
        ce_chunk=2048,
        attn=attn,
        fp32_lm_head=fp32_lm_head,
    )

    rows = []
    for sp in SP_DEGREES:
        training = copy.deepcopy(cfg.training)
        training["sp_size"] = sp
        # Data parallelism stays at one, so the gradient is reduced across the sequence split and nothing
        # else. The GPU count moves with it: the actor validates the config against the slots the gateway
        # actually has, and a config asking for more than the gateway owns fails initialization with a 500.
        training["n_gpus"] = sp
        payload = build_payload(training, model_path, SEED, attn_implementation=attn)
        print(f"[dss] sp_size {sp} ({batch.active_tokens // sp:,} real tokens per rank) ...", flush=True)
        with gateway(work, sp) as url:
            with running_job(url, payload) as job_id:
                dss = fwd_bwd_step(url, job_id, body)

        pairs, only_dss, only_ref = align(dss.grad_norms, reference["grad_norms"])
        ratios = sorted(a / b for _, a, b in pairs if b)
        median_ratio = ratios[len(ratios) // 2]
        above = sum(1 for r in ratios if r > 1.0)
        diffs = [(abs(a - b), name) for name, a, b in pairs]
        over = [d for d, _ in diffs if d > STATED_CRITERION_ABS]
        worst, worst_name = max(diffs)
        loss_delta = abs(dss.avg_loss - reference["loss"])
        final_norm = next(((a, b) for name, a, b in pairs if name == "norm.weight"), None)
        rows.append((sp, median_ratio, above, len(ratios), len(over), worst, worst_name, loss_delta, final_norm))
        print(
            f"[dss] sp {sp}: median ratio {median_ratio:.6f}  ratios above 1.0 {above}/{len(ratios)}  "
            f"over 1e-3 {len(over)}/{len(pairs)}  worst {worst:.3e} {worst_name}  "
            f"loss delta {loss_delta:.3e}"
            f"{f'  UNMATCHED {len(only_dss)}/{len(only_ref)}' if only_dss or only_ref else ''}\n",
            flush=True,
        )

    print(
        f"{'sp':>4} {'median ratio':>13} {'above 1.0':>11} {'over 1e-3':>10} {'worst abs':>11} "
        f"{'loss delta':>11}  worst tensor"
    )
    for sp, ratio, above, total, over, worst, worst_name, loss_delta, _ in rows:
        print(
            f"{sp:>4} {ratio:>13.6f} {above:>7}/{total:<3} {over:>10} {worst:>11.3e} {loss_delta:>11.3e}  {worst_name}"
        )
    print("")
    print(f"{'sp':>4} {'norm.weight Arctic Platform':>17} {'reference':>12} {'abs diff':>11}")
    for sp, *_rest, final_norm in rows:
        if final_norm:
            a, b = final_norm
            print(f"{sp:>4} {a:>17.6f} {b:>12.6f} {abs(a - b):>11.3e}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(
        main(
            args[0] if args else "arctic_platform/correctness/configs/qwen3.8-27b/h200/train-sft-8gpus-2k.config",
            args[1] if len(args) > 1 else SPEC,
        )
    )
