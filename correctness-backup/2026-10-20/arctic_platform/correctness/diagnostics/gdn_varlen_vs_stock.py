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

"""Does the engine's packed gated-delta-net forward compute what the stock HuggingFace one computes?

The engine replaces ``Qwen3_5GatedDeltaNet.forward`` so a packed request cannot leak convolution or
recurrent state across sequence boundaries. That replacement also changes kernels: the short convolution
becomes ``causal_conv1d_fn`` with a segment index, and the delta rule is given ``cu_seqlens``. The
reference runs the stock forward on a padded batch.

With a single sequence filling the window the two are the same mathematical function, so whatever they
disagree on is kernel choice. Both are given the same hidden states and the same upstream gradient, and
every parameter of the module is compared.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from arctic_platform.correctness.harness.seeds import SEED  # noqa: E402
from arctic_platform.correctness.onboarding.synth_model import materialize_pretrained  # noqa: E402

TOKENS = int(os.environ.get("PROBE_ROW_TOKENS", 2048))
LAYERS = int(os.environ.get("PROBE_LAYERS", 4))
SOURCE = "/data-fast/base-models/Qwen/Qwen3.8-27B"
CACHE_ROOT = "/data-fast/base-models/synthetic"


def run_once(module, hidden, upstream, **call_kwargs):
    module.zero_grad(set_to_none=True)
    x = hidden.detach().clone().requires_grad_(True)
    out = module(x, **call_kwargs)
    out.backward(upstream)
    grads = {name: p.grad.detach().float().clone() for name, p in module.named_parameters() if p.grad is not None}
    return out.detach().float(), grads, x.grad.detach().float()


def compare(name, a, b):
    diff = (a - b).abs()
    na, nb = float(a.norm()), float(b.norm())
    print(
        f"{name:34} {na:>13.6f} {nb:>13.6f} {abs(na - nb):>11.3e} {float(diff.max()):>11.3e} "
        f"{int((diff == 0).sum()):>9}/{diff.numel()}"
    )


def main() -> int:
    from transformers import AutoModelForCausalLM

    from arctic_platform.model.implementations.qwen35.model_builder import _patch_qwen3_5_linear_attn_varlen

    path = materialize_pretrained(SOURCE, f"{CACHE_ROOT}/Qwen3.8-27B-{LAYERS}L", LAYERS).cache_path
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True)
    inner = getattr(model, "model", model)
    gdn = next(layer.linear_attn for layer in inner.layers if hasattr(layer, "linear_attn")).cuda()
    hidden_size = gdn.in_proj_qkv.weight.shape[1]
    print(f"gated delta net with hidden size {hidden_size}, {TOKENS:,} tokens, one sequence")

    torch.manual_seed(SEED)
    hidden = torch.randn(1, TOKENS, hidden_size, dtype=torch.bfloat16, device="cuda") * 0.05
    upstream = torch.randn(1, TOKENS, hidden_size, dtype=torch.bfloat16, device="cuda")

    # The stock forward takes the same convolution kernel and forwards whatever ``seq_idx`` it is given,
    # so running it twice -- once without and once with the segment index the packed path builds --
    # separates the segment index from everything else the packed forward does.
    seq_idx = torch.zeros(1, TOKENS, dtype=torch.int32, device="cuda")
    cu_boundaries = torch.tensor([0, TOKENS], dtype=torch.int32, device="cuda")
    stock = run_once(gdn, hidden, upstream)
    stock_seq_idx = run_once(gdn, hidden, upstream, seq_idx=seq_idx)
    # The stock forward also forwards sequence boundaries to the delta rule, under its own keyword. Given
    # both arguments it is doing everything the packed forward does, so a remaining difference is not an
    # argument the engine passes.
    stock_both = run_once(gdn, hidden, upstream, seq_idx=seq_idx, cu_seq_lens_q=cu_boundaries)

    _patch_qwen3_5_linear_attn_varlen()
    cu_seqlens = torch.tensor([0, TOKENS], dtype=torch.int32, device="cuda")
    packed = run_once(gdn, hidden, upstream, cu_seqlens=cu_seqlens)
    # The same call twice. Any difference here belongs to the kernels rather than to the two forwards,
    # and it is the scale every other row in this probe has to be read against.
    packed_again = run_once(gdn, hidden, upstream, cu_seqlens=cu_seqlens)

    for title, left, right in (
        ("packed against packed, the same call twice", packed, packed_again),
        ("stock against packed", stock, packed),
        ("stock with the same segment index against packed", stock_seq_idx, packed),
        ("stock given both arguments against packed", stock_both, packed),
        ("stock against stock with a segment index", stock, stock_seq_idx),
    ):
        left_out, left_grads, left_input = left
        right_out, right_grads, right_input = right
        print("")
        print(f"--- {title}")
        print(f"{'quantity':34} {'left':>13} {'right':>13} {'norm diff':>11} {'max elem':>11}  equal")
        compare("forward output", left_out, right_out)
        compare("input gradient", left_input, right_input)
        for name in sorted(set(left_grads) & set(right_grads)):
            compare(name, left_grads[name], right_grads[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
