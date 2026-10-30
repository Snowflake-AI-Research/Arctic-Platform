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
"""The Triton hyper-connection mix and combine of Qwen3.8-Flash-Next round once from fp32, as vLLM's kernels do."""

import pytest
import torch

from arctic_platform.model.implementations.qwen38 import hyper_connections as hc
from arctic_platform.testing_utils import require_torch_gpu

HC, HIDDEN, TOKENS = 4, 640, 300


def _assert_rounded_once(actual: torch.Tensor, reference: torch.Tensor) -> None:
    """``actual`` is the fp64 ``reference`` rounded once to its dtype, but for at most 1 value in 10,000: Triton's
    sigmoid, which vLLM's kernels use as well, is not correctly rounded."""
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual.double(), reference, rtol=1e-5, atol=1e-6)
        return
    differ = int((actual != reference.to(actual.dtype)).sum())
    assert differ <= actual.numel() // 10_000, f"{differ} of {actual.numel()} values were rounded more than once"


def _inputs(dtype=torch.bfloat16, hc_count=HC):
    torch.manual_seed(0)
    x = (2 * torch.randn(TOKENS, hc_count * HIDDEN, device="cuda")).to(dtype).requires_grad_()
    gate = (2 * torch.randn(TOKENS, hc_count * HIDDEN, device="cuda")).to(dtype).requires_grad_()
    block = (2 * torch.randn(TOKENS, HIDDEN, device="cuda")).to(dtype).requires_grad_()
    logits = (4 * torch.randn(TOKENS, hc_count, device="cuda")).to(dtype).requires_grad_()
    return x, gate, block, logits


def _mix_fp64(x, gate, hc_count=HC):
    streams = (hc_count, -1)
    return (torch.sigmoid(gate.double()).unflatten(-1, streams) * x.double().unflatten(-1, streams)).mean(-2)


def _combine_fp64(residual, block, logits, hc_count=HC):
    weight = 2 * torch.sigmoid(logits.double() / hc_count)
    streams = residual.double().unflatten(-1, (hc_count, -1))
    return (streams + block.double().unsqueeze(-2) * weight.unsqueeze(-1)).flatten(-2)


@require_torch_gpu
@pytest.mark.parametrize("hc_count", [3, 4])
def test_forward_rounds_once(hc_count):
    x, gate, block, logits = _inputs(hc_count=hc_count)
    _assert_rounded_once(hc.gate_mix(x, gate, hc_count), _mix_fp64(x, gate, hc_count))
    _assert_rounded_once(hc.combine(x, block, logits, hc_count), _combine_fp64(x, block, logits, hc_count))


@require_torch_gpu
def test_forward_matches_vllm_kernels_bit_for_bit():
    vllm_hc = pytest.importorskip("vllm.models.qwen4_exp.nvidia.ops.hc")
    x, gate, block, logits = (tensor.detach() for tensor in _inputs())
    torch.testing.assert_close(hc.gate_mix(x, gate, HC), vllm_hc._hc_gate_mix(x, gate, HC), rtol=0, atol=0)
    torch.testing.assert_close(
        hc.combine(x, block, logits, HC), vllm_hc._hc_combine(x, block, logits, HC), rtol=0, atol=0
    )


@require_torch_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_backward_matches_fp64(dtype):
    x, gate, block, logits = _inputs(dtype)
    x64, gate64, block64, logits64 = (tensor.detach().double().requires_grad_() for tensor in (x, gate, block, logits))
    grad_mix = torch.randn(TOKENS, HIDDEN, device="cuda").to(dtype)
    grad_combine = torch.randn(TOKENS, HC * HIDDEN, device="cuda").to(dtype)
    cases = (
        ((x, gate), hc.gate_mix(x, gate, HC), (x64, gate64), _mix_fp64(x64, gate64), grad_mix),
        (
            (x, block, logits),
            hc.combine(x, block, logits, HC),
            (x64, block64, logits64),
            _combine_fp64(x64, block64, logits64),
            grad_combine,
        ),
    )
    for leaves, output, leaves64, reference, grad in cases:
        actual = torch.autograd.grad(output, leaves, grad)
        expected = torch.autograd.grad(reference, leaves64, grad.double())
        for got, want, leaf in zip(actual, expected, leaves, strict=True):
            assert got.dtype == leaf.dtype
            _assert_rounded_once(got, want)


@require_torch_gpu
def test_double_backward_uses_differentiable_ops():
    x, gate, block, logits = _inputs(torch.float32)
    for output in (hc.gate_mix(x, gate, HC), hc.combine(x, block, logits, HC)):
        inputs = (x, gate) if output.shape[-1] == HIDDEN else (block, logits)
        grads = torch.autograd.grad(output.square().sum(), inputs, create_graph=True)
        second = torch.autograd.grad(sum(grad.square().sum() for grad in grads), inputs, allow_unused=True)
        assert all(grad is not None and bool(torch.isfinite(grad).all()) for grad in second)


@require_torch_gpu
def test_saves_only_its_inputs():
    x, gate, block, logits = _inputs()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        hc.gate_mix(x, gate, HC)
        hc.combine(x, block, logits, HC)
    assert [tensor.data_ptr() for tensor in saved] == [t.data_ptr() for t in (x, gate, block, logits)]
