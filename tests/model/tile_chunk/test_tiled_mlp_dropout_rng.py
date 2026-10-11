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

"""Unit test: a tiled FFN that draws from the RNG must be recomputed under the state its forward used.

``TiledMLP`` runs the FFN once per token shard in the forward and runs it again, shard by shard, inside the
backward to rebuild the intermediates it did not keep. It restores no RNG state between the two, so a shard
whose forward draws -- PEFT installs ``lora_dropout`` on the projections, and dropout draws under
``no_grad`` like anywhere else -- would be differentiated against a mask that never produced the output. The
gradient that comes back is then the gradient of a different network.

CPU, float64, tiny tensors: the reference is the same FFN run shard by shard with grad enabled, which
consumes the RNG in exactly the order the tiled forward does, so the two must agree exactly.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from arctic_platform.model.implementations.gpu.tiled_mlp import apply_tiled_mlp

TOKENS, HIDDEN, INTERMEDIATE, SHARDS = 32, 16, 40, 4
TOKEN_CHUNK_SIZE = TOKENS // SHARDS
DROPOUT_P = 0.5
SEED = 1234


class DropoutSwiGLUFeedForward(nn.Module):
    """The Qwen3.5 dense FFN with dropout in front of the projections, as a LoRA adapter leaves it."""

    def __init__(self, hidden: int, intermediate: int, p: float):
        super().__init__()
        self.w1 = nn.Linear(hidden, intermediate, bias=False)
        self.w3 = nn.Linear(hidden, intermediate, bias=False)
        self.w2 = nn.Linear(intermediate, hidden, bias=False)
        self.dropout = nn.Dropout(p)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _mlp_forward(self, hidden_states)


def _mlp_forward(module: DropoutSwiGLUFeedForward, hidden_states: torch.Tensor) -> torch.Tensor:
    gated = F.silu(module.w1(module.dropout(hidden_states)))
    return module.w2(gated * module.w3(module.dropout(hidden_states)))


def _compute_params(module: DropoutSwiGLUFeedForward) -> list[torch.Tensor]:
    return [module.w1.weight, module.w2.weight, module.w3.weight]


def _tiled(model: DropoutSwiGLUFeedForward) -> DropoutSwiGLUFeedForward:
    tiled = copy.deepcopy(model)
    patched = apply_tiled_mlp(
        tiled,
        is_target=lambda module: isinstance(module, DropoutSwiGLUFeedForward),
        mlp_forward=_mlp_forward,
        compute_params=_compute_params,
        token_chunk_size=TOKEN_CHUNK_SIZE,
    )
    assert patched == 1
    return tiled


def _backward(output: torch.Tensor) -> None:
    # A non-constant incoming gradient, so a shard differentiated against the wrong mask cannot cancel out.
    weights = torch.arange(1, output.numel() + 1, dtype=output.dtype).reshape(output.shape)
    (output * weights).sum().backward()


def _run(model: DropoutSwiGLUFeedForward, x: torch.Tensor):
    output = model(x)
    _backward(output)
    grads = {name: param.grad.detach().clone() for name, param in model.named_parameters()}
    return output.detach().clone(), x.grad.detach().clone(), grads


def _run_shard_by_shard(model: DropoutSwiGLUFeedForward, x: torch.Tensor):
    """The gradient owed when both passes shard the same flattened token stream."""
    original_shape = x.shape
    flat = x.reshape(-1, x.shape[-1])
    num_shards = (flat.shape[0] + TOKEN_CHUNK_SIZE - 1) // TOKEN_CHUNK_SIZE
    shards = [_mlp_forward(model, shard) for shard in torch.chunk(flat, chunks=num_shards, dim=0)]
    output = torch.cat(shards, dim=0).view(*original_shape[:-1], -1)
    _backward(output)
    grads = {name: param.grad.detach().clone() for name, param in model.named_parameters()}
    return output.detach().clone(), x.grad.detach().clone(), grads


def test_the_tiled_ffn_differentiates_the_dropout_masks_its_forward_drew():
    torch.manual_seed(0)
    reference = DropoutSwiGLUFeedForward(HIDDEN, INTERMEDIATE, DROPOUT_P).double()
    tiled = _tiled(reference)

    base = torch.randn(TOKENS, HIDDEN, dtype=torch.float64)
    x_reference = base.clone().requires_grad_(True)
    x_tiled = base.clone().requires_grad_(True)

    torch.manual_seed(SEED)
    drawn_once = _mlp_forward(reference, base)
    torch.manual_seed(SEED + 1)
    drawn_again = _mlp_forward(reference, base)
    assert not torch.allclose(drawn_once, drawn_again), "dropout must fire, or this test proves nothing"

    torch.manual_seed(SEED)
    out_reference, gin_reference, grads_reference = _run_shard_by_shard(reference, x_reference)
    torch.manual_seed(SEED)
    out_tiled, gin_tiled, grads_tiled = _run(tiled, x_tiled)
    torch.testing.assert_close(out_tiled, out_reference, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(gin_tiled, gin_reference, rtol=1e-12, atol=1e-12)
    for name, expected in grads_reference.items():
        torch.testing.assert_close(grads_tiled[name], expected, rtol=1e-12, atol=1e-12, msg=f"grad {name}")


def test_the_backward_recompute_leaves_the_rng_where_the_forward_left_it():
    """Every later consumer in the step shares this generator, so the recompute must not advance it."""
    torch.manual_seed(0)
    tiled = _tiled(DropoutSwiGLUFeedForward(HIDDEN, INTERMEDIATE, DROPOUT_P).double())
    x = torch.randn(TOKENS, HIDDEN, dtype=torch.float64).requires_grad_(True)

    torch.manual_seed(SEED)
    output = tiled(x)
    after_forward = torch.rand(4, dtype=torch.float64)

    torch.manual_seed(SEED)
    output = tiled(x)
    _backward(output)
    after_backward = torch.rand(4, dtype=torch.float64)

    torch.testing.assert_close(after_backward, after_forward, rtol=0, atol=0)


def test_rng_replay_matches_forward_shards_for_more_than_one_batch_row():
    """DeepSpeed flattens ``[B,S,H]`` in backward, so forward must record RNG for those same token groups."""
    torch.manual_seed(0)
    reference = DropoutSwiGLUFeedForward(HIDDEN, INTERMEDIATE, DROPOUT_P).double()
    tiled = _tiled(reference)
    base = torch.randn(2, TOKENS, HIDDEN, dtype=torch.float64)

    torch.manual_seed(SEED)
    out_reference, gin_reference, grads_reference = _run_shard_by_shard(reference, base.clone().requires_grad_(True))
    torch.manual_seed(SEED)
    out_tiled, gin_tiled, grads_tiled = _run(tiled, base.clone().requires_grad_(True))

    torch.testing.assert_close(out_tiled, out_reference, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(gin_tiled, gin_reference, rtol=1e-12, atol=1e-12)
    for name, expected in grads_reference.items():
        torch.testing.assert_close(grads_tiled[name], expected, rtol=1e-12, atol=1e-12, msg=f"grad {name}")


def test_rng_replay_covers_dropout_added_after_tiling_is_installed():
    """Production applies PEFT after dense tiling, so preserving RNG cannot be decided during installation."""
    torch.manual_seed(0)
    reference = DropoutSwiGLUFeedForward(HIDDEN, INTERMEDIATE, DROPOUT_P).double()
    tiled = copy.deepcopy(reference)
    tiled.dropout = nn.Identity()
    tiled = _tiled(tiled)
    tiled.dropout = nn.Dropout(DROPOUT_P)
    base = torch.randn(TOKENS, HIDDEN, dtype=torch.float64)

    torch.manual_seed(SEED)
    out_reference, gin_reference, grads_reference = _run_shard_by_shard(reference, base.clone().requires_grad_(True))
    torch.manual_seed(SEED)
    out_tiled, gin_tiled, grads_tiled = _run(tiled, base.clone().requires_grad_(True))

    torch.testing.assert_close(out_tiled, out_reference, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(gin_tiled, gin_reference, rtol=1e-12, atol=1e-12)
    for name, expected in grads_reference.items():
        torch.testing.assert_close(grads_tiled[name], expected, rtol=1e-12, atol=1e-12, msg=f"grad {name}")
