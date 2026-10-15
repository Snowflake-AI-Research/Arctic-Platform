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

"""Unit test: ``apply_tiled_mlp`` token-sharding is mathematically identical to the un-tiled dense FFN.

Tiling is a pure memory transform, so for the same weights and input the forward output and every gradient
must match the un-tiled FFN to fp tolerance. Runs on CPU in float64 with tiny tensors (no GPU/gateway/model),
mirroring the wiring in ``qwen35/deepspeed_integration.py``.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from arctic_platform.model.implementations.gpu.tiled_mlp import apply_dense_tiled_mlp
from arctic_platform.model.implementations.gpu.tiled_mlp import apply_tiled_mlp
from arctic_platform.model.implementations.gpu.tiled_mlp import enable_tiled_mlp


class SwiGLUFeedForward(nn.Module):
    """Same shape as the Qwen3.5 dense shared-expert FFN: ``w2(silu(w1 x) * w3 x)``."""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.w1 = nn.Linear(hidden, intermediate, bias=False)
        self.w3 = nn.Linear(hidden, intermediate, bias=False)
        self.w2 = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))


def _mlp_forward(module: SwiGLUFeedForward, hidden_states: torch.Tensor) -> torch.Tensor:
    return module.w2(F.silu(module.w1(hidden_states)) * module.w3(hidden_states))


def _compute_params(module: SwiGLUFeedForward) -> list[torch.Tensor]:
    return [module.w1.weight, module.w2.weight, module.w3.weight]


def _forward_backward(model: SwiGLUFeedForward, x: torch.Tensor):
    """Run forward + a scalar backward; return (output, input_grad, {param_name: grad})."""
    output = model(x)
    # Weight the reduction so the backward exercises a non-trivial (non-constant) incoming grad.
    (output * torch.arange(1, output.numel() + 1, dtype=output.dtype).reshape(output.shape)).sum().backward()
    param_grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    return output.detach().clone(), x.grad.detach().clone(), param_grads


@pytest.mark.parametrize(
    "n_tokens, token_chunk_size, expect_shards",
    [
        (32, 8, True),  # 32 tokens / 8 -> 4 shards: the real tiled path
        (30, 8, True),  # not divisible: 4 shards (8,8,8,6) -> exercises the ragged last shard
        (5, 8, False),  # tokens <= chunk -> wrapper takes the identity (no-tiling) path
    ],
)
def test_tiled_matches_untiled(n_tokens: int, token_chunk_size: int, expect_shards: bool):
    torch.manual_seed(0)
    hidden, intermediate = 16, 40

    reference = SwiGLUFeedForward(hidden, intermediate).double()
    tiled = copy.deepcopy(reference)  # identical weights

    patched = apply_tiled_mlp(
        tiled,
        is_target=lambda module: isinstance(module, SwiGLUFeedForward),
        mlp_forward=_mlp_forward,
        compute_params=_compute_params,
        token_chunk_size=token_chunk_size,
    )
    assert patched == 1, "apply_tiled_mlp should patch exactly the one target FFN"

    base = torch.randn(n_tokens, hidden, dtype=torch.float64)
    x_ref = base.clone().requires_grad_(True)
    x_tiled = base.clone().requires_grad_(True)

    out_ref, gin_ref, gp_ref = _forward_backward(reference, x_ref)
    out_tiled, gin_tiled, gp_tiled = _forward_backward(tiled, x_tiled)

    # forward output identical
    torch.testing.assert_close(out_tiled, out_ref, rtol=1e-10, atol=1e-10)
    # input gradient identical
    torch.testing.assert_close(gin_tiled, gin_ref, rtol=1e-10, atol=1e-10)
    # every weight gradient identical (tiling only defers/re-splits the reduction, never changes its value)
    assert gp_tiled.keys() == gp_ref.keys()
    for name in gp_ref:
        torch.testing.assert_close(gp_tiled[name], gp_ref[name], rtol=1e-10, atol=1e-10, msg=f"grad {name}")


def test_tiled_matches_untiled_for_more_than_one_batch_row():
    """Forward and backward must shard the same flattened token stream for a documented ``[B,S,H]`` input."""
    torch.manual_seed(2)
    hidden, intermediate = 16, 40
    reference = SwiGLUFeedForward(hidden, intermediate).double()
    tiled = copy.deepcopy(reference)
    apply_tiled_mlp(
        tiled,
        is_target=lambda module: isinstance(module, SwiGLUFeedForward),
        mlp_forward=_mlp_forward,
        compute_params=_compute_params,
        token_chunk_size=4,
    )
    base = torch.randn(2, 16, hidden, dtype=torch.float64)

    out_ref, gin_ref, gp_ref = _forward_backward(reference, base.clone().requires_grad_(True))
    out_tiled, gin_tiled, gp_tiled = _forward_backward(tiled, base.clone().requires_grad_(True))

    torch.testing.assert_close(out_tiled, out_ref, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(gin_tiled, gin_ref, rtol=1e-10, atol=1e-10)
    for name in gp_ref:
        torch.testing.assert_close(gp_tiled[name], gp_ref[name], rtol=1e-10, atol=1e-10, msg=f"grad {name}")


def _apply(model, token_chunk_size):
    return apply_tiled_mlp(
        model,
        is_target=lambda module: isinstance(module, SwiGLUFeedForward),
        mlp_forward=_mlp_forward,
        compute_params=_compute_params,
        token_chunk_size=token_chunk_size,
    )


@pytest.mark.parametrize("invalid_value", [None, 0, -1, True, 1.5, "8"])
def test_apply_tiled_mlp_rejects_non_positive_chunk_sizes(invalid_value):
    model = SwiGLUFeedForward(16, 40).double()
    with pytest.raises(ValueError, match="positive integer"):
        _apply(model, invalid_value)


def test_apply_tiled_mlp_rejects_a_model_with_no_matching_modules():
    model = nn.Linear(16, 16)
    with pytest.raises(ValueError, match="matched zero modules"):
        apply_tiled_mlp(
            model,
            is_target=lambda module: isinstance(module, SwiGLUFeedForward),
            mlp_forward=_mlp_forward,
            compute_params=_compute_params,
            token_chunk_size=8,
        )


def test_enable_tiled_mlp_leaves_the_model_unchanged_when_unset():
    model = SwiGLUFeedForward(16, 40).double()
    original_forward = model.forward

    assert (
        enable_tiled_mlp(
            model,
            is_target=lambda module: isinstance(module, SwiGLUFeedForward),
            mlp_forward=_mlp_forward,
            token_chunk_size=None,
        )
        == 0
    )
    assert model.forward == original_forward


def test_apply_dense_tiled_mlp_leaves_the_model_unchanged_when_unset():
    model = SwiGLUFeedForward(16, 40).double()
    original_forward = model.forward

    assert apply_dense_tiled_mlp(model, token_chunk_size=None) == 0
    assert model.forward == original_forward
