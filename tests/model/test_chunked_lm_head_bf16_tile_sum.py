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

"""A bf16 chunked LM head returns the same gradient however many tiles it cuts the call into.

Each fixture makes every nonzero tile contribution exactly +1, so the exact sum counts those contributions.
bf16 holds every integer up to 256 but not 257, so a bf16 running sum stops growing at 256 while an fp32 sum
reaches the count; the result is then rounded to bf16 once, as a single-tile call is.
"""

from __future__ import annotations

import pytest
import torch

from arctic_platform.model.implementations.gpu.lm_head import chunked_lm_head_logprobs
from arctic_platform.model.implementations.moe.layers.lm_head import FusedOutputLinear

TILES = 512


def _dense_weight_grad(hidden, weight, labels):
    weight = weight.clone().requires_grad_()
    chunked_lm_head_logprobs(
        hidden, weight, labels, token_chunk_size=1, vocab_chunk_size=8192, fp32_lm_head=False
    ).sum().backward()
    return weight.grad


def _moe_weight_grad(hidden, weight, labels):
    head = FusedOutputLinear(weight.shape[1], weight.shape[0], chunk_size=1, fp32_lm_head=False).to(torch.bfloat16)
    with torch.no_grad():
        head.weight.copy_(weight)
    head(hidden, labels=labels)["logprobs"].sum().backward()
    return head.weight.grad


@pytest.mark.parametrize("weight_grad", [_dense_weight_grad, _moe_weight_grad], ids=["dense", "moe"])
def test_bf16_head_sums_token_tiles_in_fp32(weight_grad):
    # Two-token vocabulary with zero weights: p = 1/2 everywhere, so the label-0 logit gradient is 1/2 per token,
    # and a hidden value of 2 makes each one-token tile add exactly 1 to grad_weight[0, 0].
    hidden = torch.full((1, TILES, 1), 2.0, dtype=torch.bfloat16)
    weight = torch.zeros(2, 1, dtype=torch.bfloat16)
    labels = torch.zeros(1, TILES, dtype=torch.long)

    grad = weight_grad(hidden, weight, labels)

    assert grad.dtype == torch.bfloat16
    assert grad[0, 0].item() == TILES  # a bf16 running sum stops at 256
    assert grad[1, 0].item() == -TILES


def test_bf16_dense_head_sums_vocab_tiles_in_fp32():
    # Zero hidden states make every logit 0, so p = 1/1024 for each of 1024 tokens and each non-target logit
    # gradient is -2**-10. A weight of -1024 then makes each one-token vocabulary tile add exactly 1 to the
    # hidden gradient: 1023 nonzero contributions across 1024 tiles, the target's own weight being 0.
    vocab = 1024
    hidden = torch.zeros(1, 1, 1, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.full((vocab, 1), -float(vocab), dtype=torch.bfloat16)
    weight[0] = 0.0
    labels = torch.zeros(1, 1, dtype=torch.long)

    chunked_lm_head_logprobs(
        hidden, weight, labels, token_chunk_size=1, vocab_chunk_size=1, fp32_lm_head=False
    ).sum().backward()

    exact = torch.tensor(float(vocab - 1)).to(torch.bfloat16)
    assert hidden.grad.dtype == torch.bfloat16
    assert hidden.grad.item() == exact.item()  # 1024; a bf16 running sum stops at 256
