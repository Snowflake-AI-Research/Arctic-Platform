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

import pytest
import torch

from arctic_platform.model.implementations.debug.token_combine import fixed_order_sum_rows_by_token
from arctic_platform.model.implementations.moe.token_combine import sum_rows_by_token


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_tokens,padding_rows", [(0, 0), (3, 0), (3, 2)])
def test_fixed_order_empty_routing_keeps_collective_backward(dtype, num_tokens, padding_rows):
    backward_calls = []

    class Receive(torch.autograd.Function):
        @staticmethod
        def forward(ctx, source):
            ctx.source_shape = source.shape
            return source.new_full((padding_rows, source.shape[1]), float("nan"))

        @staticmethod
        def backward(ctx, gradient):
            backward_calls.append(gradient.clone())
            return gradient.new_zeros(ctx.source_shape)

    source = torch.ones((2, 4), dtype=dtype, requires_grad=True)
    rows = Receive.apply(source)
    indices = torch.empty(0, dtype=torch.long)
    combined = fixed_order_sum_rows_by_token(rows, indices, num_tokens, top_k=4)
    expected = sum_rows_by_token(rows[:0], indices, num_tokens)

    torch.testing.assert_close(combined, expected, rtol=0, atol=0)
    assert combined.requires_grad, "an empty rank must still join the collective backward"
    combined.sum().backward()

    assert len(backward_calls) == 1
    torch.testing.assert_close(backward_calls[0], torch.zeros_like(rows), rtol=0, atol=0)
    torch.testing.assert_close(source.grad, torch.zeros_like(source), rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens,padding_rows", [(0, 0), (3, 0), (3, 2)])
def test_fixed_order_empty_routing_without_grad(num_tokens, padding_rows):
    rows = torch.full((padding_rows, 4), float("nan"))
    combined = fixed_order_sum_rows_by_token(rows, torch.empty(0, dtype=torch.long), num_tokens, top_k=4)
    assert not combined.requires_grad
    torch.testing.assert_close(combined, torch.zeros((num_tokens, 4)), rtol=0, atol=0)


def test_fixed_order_nonempty_routing_preserves_values_and_gradients():
    rows = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    indices = torch.tensor([1, 0, 1])
    combined = fixed_order_sum_rows_by_token(rows, indices, 3, top_k=4)
    torch.testing.assert_close(combined, torch.tensor([[3.0, 4.0], [6.0, 8.0], [0.0, 0.0]]))
    combined.square().sum().backward()
    torch.testing.assert_close(rows.grad, 2 * combined.detach().index_select(0, indices))
