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

from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn as nn

from arctic_platform.model import ActivationOffloadConfig
from arctic_platform.model.implementations.gpu.lm_head import chunked_lm_head_logprobs
from arctic_platform.model.implementations.gpu.lm_head import enable_fp32_lm_head
from arctic_platform.model.implementations.gpu.sp import gated_delta_net
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import torch_assert_close


def _check_tile_gradient_accumulation(device, accumulation_axis, dtype):
    # Uniform logits give exact gradients; many tiles expose contributions
    # lost when a running BF16 sum grows larger than each new contribution.
    rows, vocab = (1, 4096) if accumulation_axis == "hidden" else (4096, 8)
    hidden_value, weight_value = (0.0, 1.0) if accumulation_axis == "hidden" else (1.0, 0.0)
    hidden = torch.full((1, rows, 1), hidden_value, device=device, dtype=dtype, requires_grad=True)
    weight = torch.full((vocab, 1), weight_value, device=device, dtype=dtype, requires_grad=True)
    bias = torch.zeros(vocab, device=device, dtype=dtype, requires_grad=True)
    labels = torch.full((1, rows), vocab - 1, device=device, dtype=torch.long)
    actual = chunked_lm_head_logprobs(
        hidden,
        weight,
        labels,
        bias=bias,
        token_chunk_size=8,
        vocab_chunk_size=8,
        fp32_lm_head=False,
    )
    expected = (
        torch.log_softmax((hidden @ weight.t() + bias).float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    )
    torch_assert_close(actual, expected, rtol=0, atol=1e-6)
    (-actual.mean()).backward(retain_graph=True)
    actual_hidden_grad, actual_weight_grad, actual_bias_grad = (
        hidden.grad.clone(),
        weight.grad.clone(),
        bias.grad.clone(),
    )
    hidden.grad = weight.grad = bias.grad = None
    (-expected.mean()).backward()
    torch_assert_close(actual_hidden_grad, hidden.grad, rtol=0, atol=1e-6)
    torch_assert_close(actual_weight_grad, weight.grad, rtol=0, atol=1e-6)
    torch_assert_close(actual_bias_grad, bias.grad, rtol=0, atol=1e-6)


@pytest.mark.parametrize("accumulation_axis", ["hidden", "weight"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_chunked_lm_head_preserves_small_tile_gradients(accumulation_axis, dtype):
    _check_tile_gradient_accumulation("cpu", accumulation_axis, dtype)


@require_torch_gpu
@pytest.mark.parametrize("accumulation_axis", ["hidden", "weight"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_chunked_lm_head_preserves_small_tile_gradients_cuda(accumulation_axis, dtype):
    _check_tile_gradient_accumulation("cuda", accumulation_axis, dtype)


def test_chunked_lm_head_matches_full_projection_and_gradients():
    torch.manual_seed(7)
    hidden = torch.randn(2, 5, 6, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(11, 6, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 11, (2, 5))
    temperature = torch.rand(2, 5, dtype=torch.float64) + 0.5

    actual = chunked_lm_head_logprobs(
        hidden,
        weight,
        labels,
        temperature=temperature,
        token_chunk_size=3,
        vocab_chunk_size=4,
        fp32_lm_head=False,
    )
    expected = (
        torch.log_softmax(
            (hidden @ weight.t()).float() / temperature.float().unsqueeze(-1),
            dim=-1,
        )
        .gather(-1, labels.unsqueeze(-1))
        .squeeze(-1)
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    actual.sum().backward(retain_graph=True)
    actual_hidden_grad = hidden.grad.detach().clone()
    actual_weight_grad = weight.grad.detach().clone()
    hidden.grad = None
    weight.grad = None
    expected.sum().backward()
    torch.testing.assert_close(hidden.grad, actual_hidden_grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(weight.grad, actual_weight_grad, rtol=1e-6, atol=1e-6)


def test_fp32_lm_head_projection():
    model = nn.Module()
    model.lm_head = nn.Linear(4, 7, bias=False, dtype=torch.bfloat16)
    enable_fp32_lm_head(model)

    output = model.lm_head(torch.randn(3, 4, dtype=torch.bfloat16))

    assert output.dtype == torch.float32
    assert model.lm_head._dss_fp32_lm_head is True


def test_activation_offload_config_rejects_negative_pin_memory_limit():
    with pytest.raises(ValueError, match="non-negative"):
        ActivationOffloadConfig(pin_memory_max_size_gib=-1)


def test_gated_delta_net_resolves_transformers_5_17_module_level_kernels(monkeypatch):
    implementation = types.ModuleType("fake_transformers_gated_delta_net")

    def causal_conv1d_fn(*args, **kwargs):
        return args, kwargs

    def torch_chunk_gated_delta_rule(*args, **kwargs):
        return args, kwargs

    implementation.causal_conv1d_fn = causal_conv1d_fn
    implementation.torch_chunk_gated_delta_rule = torch_chunk_gated_delta_rule
    monkeypatch.setitem(sys.modules, implementation.__name__, implementation)

    class TransformersGatedDeltaNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_kernel_size = 4

    TransformersGatedDeltaNet.__module__ = implementation.__name__
    module = TransformersGatedDeltaNet()

    assert gated_delta_net._is_gated_delta_net_module(module)
    assert gated_delta_net._resolve_gated_delta_net_kernel(module, "causal_conv1d_fn") is causal_conv1d_fn
    assert (
        gated_delta_net._resolve_gated_delta_net_kernel(module, "chunk_gated_delta_rule")
        is torch_chunk_gated_delta_rule
    )
