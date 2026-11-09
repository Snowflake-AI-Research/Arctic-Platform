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
"""The expert LoRA epilogue adds into the GEMM outputs in place without autograd's CopySlices."""

from __future__ import annotations

import pytest
import torch

from arctic_platform.model.implementations.moe.layers.moe import _add_into_rows


def _graph_names(tensor: torch.Tensor) -> set[str]:
    names, stack, seen = set(), [tensor.grad_fn], set()
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        names.add(type(fn).__name__)
        stack.extend(next_fn for next_fn, _ in fn.next_functions)
    return names


@pytest.mark.parametrize("cols", [None, slice(3, 7)])
def test_add_into_rows_matches_an_out_of_place_add_without_copy_slices(cols):
    torch.manual_seed(0)
    source = torch.randn(9, 5, requires_grad=True)
    weight = torch.randn(5, 10)
    lora_input = torch.randn(7, 4, requires_grad=True)
    lora_weight = torch.randn(4, 10 if cols is None else 4)
    grad = torch.randn(9, 10)
    rows, span = 7, cols or slice(0, 10)

    out = _add_into_rows(source @ weight, lora_input @ lora_weight, rows, cols)
    out.backward(grad)
    expected_source, expected_lora = source.grad.clone(), lora_input.grad.clone()
    source.grad = lora_input.grad = None
    base = source @ weight
    delta = torch.zeros_like(base)
    delta[:rows, span] = lora_input @ lora_weight
    reference = base + delta
    reference.backward(grad)

    torch.testing.assert_close(out, reference)
    torch.testing.assert_close(expected_source, source.grad)
    torch.testing.assert_close(expected_lora, lora_input.grad)
    assert "CopySlices" not in _graph_names(out)
