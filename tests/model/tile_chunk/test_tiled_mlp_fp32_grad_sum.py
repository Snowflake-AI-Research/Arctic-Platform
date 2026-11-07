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
"""A BF16 weight's gradient over the tiles of a tiled MLP is summed in FP32 and rounded once."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from arctic_platform.model.implementations.gpu.tiled_mlp import apply_tiled_mlp

TILES = 8


class _Scale(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states @ self.weight.t()


def _tile(module: _Scale, compute_params=lambda module: [module.weight]) -> _Scale:
    apply_tiled_mlp(
        module,
        is_target=lambda candidate: candidate is module,
        mlp_forward=lambda candidate, hidden_states: hidden_states @ candidate.weight.t(),
        compute_params=compute_params,
        token_chunk_size=1,
    )
    return module


def _backward(module: _Scale, values: list[float], requires_grad: bool = True) -> torch.Tensor:
    x = torch.tensor(values, dtype=torch.bfloat16).view(-1, 1).requires_grad_(requires_grad)
    module(x).sum().backward()
    return x


def test_bf16_weight_gradient_over_eight_tiles_is_rounded_once():
    # One tile contributes 256, seven contribute 1: the sum is 263, which BF16 rounds to 264. A BF16 running sum
    # adds each 1 to 256 and rounds it away every time.
    module = _tile(_Scale())
    _backward(module, [256.0] + [1.0] * (TILES - 1))

    assert module.weight.grad.dtype == torch.bfloat16
    assert module.weight.grad.item() == 264.0


def test_weight_post_accumulate_hook_fires_once():
    module = _tile(_Scale())
    fired = []
    module.weight.register_post_accumulate_grad_hook(lambda parameter: fired.append(parameter.grad.clone()))

    _backward(module, [1.0] * TILES)

    assert len(fired) == 1
    assert fired[0].item() == float(TILES)


def test_input_gradient_is_unchanged():
    module = _tile(_Scale())
    module.weight.data.fill_(3.0)

    x = _backward(module, [1.0] * TILES)

    assert x.grad.view(-1).tolist() == [3.0] * TILES


def test_frozen_weight_gets_no_gradient():
    module = _tile(_Scale())
    module.weight.requires_grad_(False)

    x = _backward(module, [1.0] * TILES)

    assert module.weight.grad is None
    assert x.grad is not None


def test_input_without_grad_still_trains_the_weight():
    module = _tile(_Scale())

    x = _backward(module, [1.0] * TILES, requires_grad=False)

    assert module.weight.grad.item() == float(TILES)
    assert x.grad is None


def test_gradients_accumulate_across_backward_calls():
    module = _tile(_Scale())

    _backward(module, [1.0] * TILES)
    _backward(module, [1.0] * TILES)

    assert module.weight.grad.item() == 2.0 * TILES


def test_float32_weight_gradient_stays_float32():
    module = _tile(_Scale(torch.float32))
    module(torch.ones(TILES, 1)).sum().backward()

    assert module.weight.grad.dtype == torch.float32
    assert module.weight.grad.item() == float(TILES)


def test_trainable_weight_missing_from_compute_params_is_rejected():
    module = _tile(_Scale(), compute_params=lambda module: [])
    x = torch.ones(TILES, 1, dtype=torch.bfloat16, requires_grad=True)

    with pytest.raises(RuntimeError, match="compute_params does not list"):
        module(x).sum().backward()
