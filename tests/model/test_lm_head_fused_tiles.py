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
"""The chunked LM head's Triton tile kernels against its PyTorch tile path: log-probs, entropy and gradients."""

import torch

from arctic_platform.model.implementations.moe.layers import lm_head
from arctic_platform.model.implementations.moe.layers.lm_head import FusedOutputLinear
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close


def _run(fused, monkeypatch, hidden, weight, labels, temperature, grad):
    monkeypatch.setattr(lm_head, "_fused_tiles", lambda *args: fused)
    head = FusedOutputLinear(weight.shape[1], weight.shape[0], chunk_size=3000).cuda().bfloat16()
    head.weight.data.copy_(weight)
    x = hidden.clone().requires_grad_(True)
    out = head(x, labels, temperature=temperature)
    (out["logprobs"] * grad).sum().backward()
    return out["logprobs"], out["entropy"], x.grad, head.weight.grad


@require_torch_gpu
def test_fused_tiles_match_the_pytorch_tiles(monkeypatch):
    """A vocab that is not a multiple of the 8192 tile, a token count that is not a multiple of the chunk, ignored
    positions (labels -100) and a temperature."""
    set_seed(0)
    tokens, hidden_size, vocab = 7000, 256, 20000
    hidden = (torch.randn(1, tokens, hidden_size, device="cuda") * 2).bfloat16()
    weight = (torch.randn(vocab, hidden_size, device="cuda") * 0.1).bfloat16()
    labels = torch.randint(0, vocab, (1, tokens), device="cuda")
    labels[0, ::97] = -100
    labels[0, 5] = vocab - 1  # the last column of the short tile
    labels[0, 6] = 8192  # the first column of a tile
    grad = torch.randn(1, tokens, device="cuda")
    temp = torch.full((1, tokens), 0.7, device="cuda")

    want = _run(False, monkeypatch, hidden, weight, labels, temp, grad)
    got = _run(True, monkeypatch, hidden, weight, labels, temp, grad)
    torch_assert_close(got[0], want[0], rtol=1e-5, atol=1e-5)  # log-probs, FP32
    torch_assert_close(got[1], want[1], rtol=1e-5, atol=1e-5)  # entropy, FP32
    for g, w in zip(got[2:], want[2:]):  # BF16 gradients of the hidden states and the weight
        # The BF16 tiles round a few entries one ulp apart (Triton's exp against PyTorch's), so a summed weight
        # gradient can differ where its terms cancel: compare against the tensor's scale, not entrywise.
        diff = (g.float() - w.float()).abs()
        assert (diff.norm() / w.float().norm()).item() < 2e-3
        assert (diff.max() / w.float().abs().max()).item() < 2e-3
