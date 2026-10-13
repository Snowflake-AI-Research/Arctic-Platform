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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from arctic_platform.model.implementations.moe.layers.lm_head import FusedCrossEntropyOutputLinear
from arctic_platform.model.implementations.moe.layers.lm_head import FusedOutputLinear
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import torch_assert_close


def _check_native_tile_gradient_accumulation(device, accumulation_axis, dtype):
    # Uniform logits expose both token-tile sum loss and vocabulary-tile product rounding.
    rows, vocab = (1, 16384) if accumulation_axis == "hidden" else (4096, 8)
    hidden_value, weight_value = (0.0, 1.0) if accumulation_axis == "hidden" else (1.0, 0.0)
    hidden = torch.full((1, rows, 1), hidden_value, device=device, dtype=dtype, requires_grad=True)
    head = FusedOutputLinear(1, vocab, chunk_size=8).to(device=device, dtype=dtype)
    nn.init.constant_(head.weight, weight_value)
    labels = torch.full((1, rows), vocab - 1, device=device, dtype=torch.long)
    actual = head(hidden, labels=labels)["logprobs"]
    expected = (
        torch.log_softmax((hidden @ head.weight.t()).float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    )
    torch_assert_close(actual, expected, rtol=0, atol=1e-6)
    (-actual.mean()).backward(retain_graph=True)
    actual_hidden_grad, actual_weight_grad = hidden.grad.clone(), head.weight.grad.clone()
    hidden.grad = head.weight.grad = None
    (-expected.mean()).backward()
    torch_assert_close(actual_hidden_grad, hidden.grad, rtol=0, atol=1e-6)
    torch_assert_close(actual_weight_grad, head.weight.grad, rtol=0, atol=1e-6)


@pytest.mark.parametrize("accumulation_axis", ["hidden", "weight"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_chunked_lm_head_preserves_small_tile_gradients(accumulation_axis, dtype):
    _check_native_tile_gradient_accumulation("cpu", accumulation_axis, dtype)


@require_torch_gpu
@pytest.mark.parametrize("accumulation_axis", ["hidden", "weight"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_chunked_lm_head_preserves_small_tile_gradients_cuda(accumulation_axis, dtype):
    _check_native_tile_gradient_accumulation("cuda", accumulation_axis, dtype)


class _PerTokenLoss(nn.Module):
    def forward(self, weight, hidden_states, labels):
        assert weight.shape == (5, 3)
        assert hidden_states.shape == (6, 3)
        return labels.float() + 0.5


def test_fused_cross_entropy_can_return_target_logprobs():
    head = FusedCrossEntropyOutputLinear(3, 5)
    head.fused_logprobs = _PerTokenLoss()
    hidden_states = torch.randn(2, 3, 3)
    labels = torch.tensor([[0, 1, 2], [3, 4, -100]])

    output = head(hidden_states, labels=labels, dss_compute_logprobs=True)

    torch.testing.assert_close(output["logprobs"], -(labels.float() + 0.5))
    assert "loss" not in output
