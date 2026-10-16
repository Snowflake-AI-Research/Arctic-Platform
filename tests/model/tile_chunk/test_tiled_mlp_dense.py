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

"""Dense SwiGLU MLPs on the Hugging Face / Liger load path are token-tiled by ``tiled_mlp_token_chunk_size``.

The MoE load path tiles its shared-expert FFN through the Prime-RL wrapper, which a dense Hugging Face or
Liger model never enters, so the dense path needs its own hook. These tests assert the tiling happens (one
projection call per token shard) and that it is numerically transparent. CPU-only, tiny tensors.
"""

from __future__ import annotations

import copy
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from arctic_platform.model.implementations.gpu.tiled_mlp import apply_dense_tiled_mlp

HIDDEN, INTERMEDIATE, TOKENS = 16, 40, 32


class Qwen3MLP(nn.Module):
    """The projection names and forward of a Hugging Face dense SwiGLU MLP."""

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class VendorFeedForwardMLP(Qwen3MLP):
    """A class name the hook does not know, carrying the projections that identify a dense SwiGLU."""


class DropoutQwen3MLP(Qwen3MLP):
    """A dense MLP with the positive dropout PEFT adds to projection paths."""

    def __init__(self):
        super().__init__()
        self.lora_dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        hidden_states = self.lora_dropout(hidden_states)
        return super().forward(hidden_states)


class Phi3MLP(nn.Module):
    """Phi-3 uses one fused gate/up projection rather than separate projections."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(HIDDEN, 2 * INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)

    def forward(self, hidden_states):
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Zamba2MLP(Qwen3MLP):
    """Zamba2 passes a static layer index alongside the hidden states."""

    def forward(self, hidden_states, layer_idx):
        return super().forward(hidden_states) + float(layer_idx)


class Qwen3MoeMLP(Qwen3MLP):
    """A routed expert whose invocation set may differ between data-parallel ranks."""


class DeepseekV3MLP(Qwen3MLP):
    """DeepSeek uses the same class for dense MLPs and routed experts."""


class BiasedQwen3MLP(nn.Module):
    """A matched dense MLP whose projection biases participate in every tile."""

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=True)
        self.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=True)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=True)

    def forward(self, hidden_states):
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Decoder(nn.Module):
    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp

    def forward(self, hidden_states):
        return self.mlp(hidden_states)


def _count_shards(model: nn.Module, x: torch.Tensor) -> tuple[int, torch.Tensor]:
    """Run a forward and return (calls to gate_proj, output). One call per token shard."""
    calls = 0

    def hook(*_args):
        nonlocal calls
        calls += 1

    handle = model.mlp.gate_proj.register_forward_hook(hook)
    try:
        output = model(x)
    finally:
        handle.remove()
    return calls, output


@pytest.mark.parametrize("mlp_cls", [Qwen3MLP, VendorFeedForwardMLP])
@pytest.mark.parametrize(
    "token_chunk_size, expected_shards",
    [
        (8, 4),
        (12, 3),  # 32 tokens / 12 -> 3 shards (12, 12, 8): the ragged last shard
        (64, 1),  # chunk >= tokens: the wrapper takes the identity path
    ],
)
def test_dense_mlp_is_tiled_over_the_token_dim(mlp_cls, token_chunk_size, expected_shards):
    torch.manual_seed(0)
    reference = Decoder(mlp_cls()).double()
    tiled = copy.deepcopy(reference)

    apply_dense_tiled_mlp(tiled, token_chunk_size=token_chunk_size)

    x = torch.randn(1, TOKENS, HIDDEN, dtype=torch.float64)
    shards, output = _count_shards(tiled, x)
    baseline_shards, expected_output = _count_shards(reference, x)

    assert baseline_shards == 1
    assert shards == expected_shards
    # Tiling is a memory transform only: same weights and input must give the same output.
    assert torch.allclose(output, expected_output, atol=1e-12, rtol=1e-12)


def test_dense_tiling_excludes_qwen_routed_experts_by_module_role():
    model = nn.Module()
    model.dense = Qwen3MLP().double()
    model.experts = nn.ModuleList([Qwen3MoeMLP().double()])
    dense_calls = []
    routed_calls = []
    model.dense.gate_proj.register_forward_hook(lambda *_args: dense_calls.append(None))
    model.experts[0].gate_proj.register_forward_hook(lambda *_args: routed_calls.append(None))

    apply_dense_tiled_mlp(model, token_chunk_size=8)
    hidden_states = torch.randn(1, TOKENS, HIDDEN, dtype=torch.float64)
    model.dense(hidden_states)
    model.experts[0](hidden_states)

    assert len(dense_calls) == 4
    assert len(routed_calls) == 1


def test_dense_tiling_excludes_deepseek_routed_experts_by_module_role():
    model = nn.Module()
    model.dense = DeepseekV3MLP().double()
    model.shared_expert = DeepseekV3MLP().double()
    model.experts = nn.ModuleList([DeepseekV3MLP().double()])
    calls = {"dense": [], "shared": [], "routed": []}
    model.dense.gate_proj.register_forward_hook(lambda *_args: calls["dense"].append(None))
    model.shared_expert.gate_proj.register_forward_hook(lambda *_args: calls["shared"].append(None))
    model.experts[0].gate_proj.register_forward_hook(lambda *_args: calls["routed"].append(None))

    apply_dense_tiled_mlp(model, token_chunk_size=8)
    hidden_states = torch.randn(1, TOKENS, HIDDEN, dtype=torch.float64)
    model.dense(hidden_states)
    model.shared_expert(hidden_states)
    model.experts[0](hidden_states)

    assert len(calls["dense"]) == 4
    assert len(calls["shared"]) == 4
    assert len(calls["routed"]) == 1


def test_dense_tiling_forwards_zamba2_layer_index_to_every_tile():
    model = Decoder(Zamba2MLP()).double()
    apply_dense_tiled_mlp(model, token_chunk_size=8)
    hidden_states = torch.randn(1, TOKENS, HIDDEN, dtype=torch.float64)

    actual = model.mlp(hidden_states, 3)

    reference = Zamba2MLP().double()
    reference.load_state_dict(model.mlp.state_dict())
    torch.testing.assert_close(actual, reference(hidden_states, 3), rtol=1e-12, atol=1e-12)


def test_dense_tiling_invokes_the_bound_forward_that_liger_installed():
    """An instance-level Liger forward must remain the computation run by every tile."""
    model = Decoder(Qwen3MLP()).double()
    original_forward = model.mlp.forward
    calls = []

    def liger_forward(self, hidden_states):
        calls.append(tuple(hidden_states.shape))
        return original_forward(hidden_states) + 1.0

    model.mlp.forward = types.MethodType(liger_forward, model.mlp)
    apply_dense_tiled_mlp(model, token_chunk_size=8)

    output = model(torch.randn(1, TOKENS, HIDDEN, dtype=torch.float64))

    assert calls == [(8, HIDDEN)] * 4
    assert torch.all(output > -1000)  # force evaluation before inspecting the recorded calls


def test_dense_tiling_supports_phi3_fused_gate_up_projection():
    """A configured Phi-3 MLP must tile rather than silently report zero matched modules."""
    torch.manual_seed(0)
    reference = Decoder(Phi3MLP()).double()
    tiled = copy.deepcopy(reference)
    calls = 0

    def hook(*_args):
        nonlocal calls
        calls += 1

    handle = tiled.mlp.gate_up_proj.register_forward_hook(hook)
    try:
        apply_dense_tiled_mlp(tiled, token_chunk_size=8)
        x = torch.randn(1, TOKENS, HIDDEN, dtype=torch.float64)
        output = tiled(x)
    finally:
        handle.remove()

    assert calls == 4
    torch.testing.assert_close(output, reference(x), rtol=1e-12, atol=1e-12)


def test_configured_dense_tiling_rejects_a_model_with_no_matching_mlp():
    """A memory setting that changed nothing must fail during initialization, not log modules=0 and continue."""
    model = Decoder(nn.Linear(HIDDEN, HIDDEN)).double()

    with pytest.raises(ValueError, match="tiled_mlp_token_chunk_size.*zero modules"):
        apply_dense_tiled_mlp(model, token_chunk_size=8)


def test_tiled_dense_mlp_defers_zero_reduction_for_projection_biases():
    """Every trainable parameter must stay unready until the last tile reaches ZeRO.

    This hook models ZeRO's per-parameter reduction guard. A parameter omitted from ``compute_params`` remains
    ready on the first tile, gets reduced there, and raises when the second tile presents it again.
    """
    torch.manual_seed(0)
    model = Decoder(BiasedQwen3MLP()).double()
    apply_dense_tiled_mlp(model, token_chunk_size=8)

    reduced: set[str] = set()
    handles = []
    for name, param in model.mlp.named_parameters():
        param.ds_grad_is_ready = True

        def reduce_once(p, *, param_name=name):
            if getattr(p, "ds_grad_is_ready", True):
                if param_name in reduced:
                    raise RuntimeError(f"parameter {param_name} has already been reduced")
                reduced.add(param_name)

        handles.append(param.register_post_accumulate_grad_hook(reduce_once))

    try:
        x = torch.randn(1, TOKENS, HIDDEN, dtype=torch.float64, requires_grad=True)
        model(x).sum().backward()
    finally:
        for handle in handles:
            handle.remove()

    assert reduced == {name for name, _ in model.mlp.named_parameters()}


def test_tiled_dense_mlp_matches_untiled_gradients():
    torch.manual_seed(0)
    reference = Decoder(Qwen3MLP()).double()
    tiled = copy.deepcopy(reference)
    apply_dense_tiled_mlp(tiled, token_chunk_size=8)

    grads = []
    for model in (reference, tiled):
        x = torch.randn(1, TOKENS, HIDDEN, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
        x.requires_grad_(True)
        output = model(x)
        # A non-constant incoming gradient, so a shard-boundary error cannot cancel out.
        (output * torch.arange(1, output.numel() + 1, dtype=output.dtype).reshape(output.shape)).sum().backward()
        grads.append((x.grad.clone(), {name: p.grad.clone() for name, p in model.named_parameters()}))

    (ref_input_grad, ref_param_grads), (tiled_input_grad, tiled_param_grads) = grads
    assert torch.allclose(tiled_input_grad, ref_input_grad, atol=1e-12, rtol=1e-12)
    for name, ref_grad in ref_param_grads.items():
        assert torch.allclose(tiled_param_grads[name], ref_grad, atol=1e-12, rtol=1e-12), name


def test_modules_without_swiglu_projections_are_rejected_when_tiling_is_configured():
    model = Decoder(nn.Linear(HIDDEN, HIDDEN)).double()

    with pytest.raises(ValueError, match="tiled_mlp_token_chunk_size.*zero modules"):
        apply_dense_tiled_mlp(model, token_chunk_size=8)
