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
"""Routing weights folded into the frozen routed-expert down projection: same outputs and gradients as scaling the
expert outputs afterwards, without keeping those outputs for backward."""

from __future__ import annotations

import pytest
import torch

from arctic_platform.model.implementations import fp8
from arctic_platform.model.implementations.moe.layers import moe as moe_module
from arctic_platform.model.implementations.moe.routing_weights import apply_routing_weights

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the scored down projection runs on CUDA")

_EXPERTS, _DIM, _HIDDEN = 4, 768, 512
_COUNTS = [100, 0, 37, 255]


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _quantized(*shape: int) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [
        fp8.quantize_to_fp8_blockwise(torch.randn(*shape[1:], device="cuda") * 0.02, 128) for _ in range(shape[0])
    ]
    return torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])


def _experts(fp8_block_size: int | None) -> moe_module.GroupedExperts:
    torch.manual_seed(0)
    experts = moe_module.GroupedExperts(_DIM, _HIDDEN, _EXPERTS, use_grouped_mm=True, fp8_block_size=fp8_block_size)
    experts = experts.cuda()
    if fp8_block_size is None:
        for name in ("w1", "w2", "w3"):
            param = getattr(experts, name)
            param.data = (torch.randn(param.shape, device="cuda") * 0.02).bfloat16()
            param.requires_grad_(False)
    else:
        for name, shape in (
            ("w1", (_EXPERTS, _HIDDEN, _DIM)),
            ("w2", (_EXPERTS, _DIM, _HIDDEN)),
            ("w3", (_EXPERTS, _HIDDEN, _DIM)),
        ):
            weight, scale = _quantized(*shape)
            getattr(experts, name).data = weight
            getattr(experts, f"{name}_scale_inv").data = scale
    experts.set_ep_comm_backend("deepep")
    return experts


def _inputs(pad: int):
    torch.manual_seed(1)
    counts = torch.tensor(_COUNTS, device="cuda")
    rows = int(counts.sum()) + pad
    x = torch.randn(rows, _DIM, device="cuda").bfloat16().requires_grad_(True)
    scores = torch.rand(rows, device="cuda").requires_grad_(True)
    grad = torch.randn(rows, _DIM, device="cuda").bfloat16()
    return counts, x, scores, grad


def _saved_activation_bytes(fn, rows: int):
    """Bytes of the per-row tensors autograd keeps for backward (weights excluded)."""
    saved = []

    def pack(tensor):
        if tensor.ndim and tensor.shape[0] == rows:
            saved.append(tensor.numel() * tensor.element_size())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = fn()
    return out, sum(saved)


@pytest.mark.parametrize("pad", [0, 9])
def test_scored_experts_match_scaling_the_outputs_and_keep_no_output_copy(pad):
    experts = _experts(128)
    counts, x, scores, grad = _inputs(pad)
    routed = int(counts.sum())

    expected, expected_saved = _saved_activation_bytes(
        lambda: apply_routing_weights(experts(x, counts), scores), x.shape[0]
    )
    expected.backward(grad)
    expected_dx, expected_ds = x.grad.clone(), scores.grad.clone()
    x.grad = scores.grad = None

    out, saved = _saved_activation_bytes(lambda: experts(x, counts, scores), x.shape[0])
    out.backward(grad)

    assert type(out.grad_fn).__name__ == "_ScoredSwiGLUDownFnBackward"
    # The same math as scaling afterwards; gate and up come from one stacked GEMM instead of two.
    assert _rel(out[:routed], expected[:routed]) < 1e-3
    assert _rel(x.grad[:routed], expected_dx[:routed]) < 1e-2
    assert _rel(scores.grad[:routed], expected_ds[:routed]) < 1e-2
    assert (out[routed:] == 0).all() and (scores.grad[routed:] == 0).all()
    assert (x.grad[routed:] == 0).all()
    # Scaling afterwards keeps the [rows, dim] expert output for the score gradient; the folded path keeps nothing else.
    assert expected_saved - saved == x.shape[0] * _DIM * 2


@pytest.mark.parametrize("limit", [None, 1.0])
def test_scored_swiglu_down_gradients_match_autograd_reference(limit):
    from arctic_platform.model.implementations.kernels.scored_swiglu_down import scored_swiglu_down

    torch.manual_seed(0)
    counts = torch.tensor(_COUNTS, device="cuda")
    offsets = torch.cumsum(counts, 0, dtype=torch.int32)
    rows = int(offsets[-1])
    weight = (torch.randn(_EXPERTS, _DIM, _HIDDEN, device="cuda") * 0.05).bfloat16()
    gate_up = (torch.randn(rows, 2 * _HIDDEN, device="cuda") * 2).bfloat16().requires_grad_(True)
    scores = torch.rand(rows, device="cuda").requires_grad_(True)
    grad = torch.randn(rows, _DIM, device="cuda").bfloat16()

    out = scored_swiglu_down(gate_up, scores, weight, None, offsets, rows, limit=limit)
    out.backward(grad)
    gate_grad, up_grad = gate_up.grad[:, :_HIDDEN], gate_up.grad[:, _HIDDEN:]

    gate, up = gate_up.detach()[:, :_HIDDEN], gate_up.detach()[:, _HIDDEN:]
    g64, u64, s64 = (t.detach().double().requires_grad_(True) for t in (gate, up, scores))
    g_c, u_c = (g64, u64) if limit is None else (g64.clamp(max=limit), u64.clamp(min=-limit, max=limit))
    h = torch.nn.functional.silu(g_c) * u_c
    segments = torch.split(h, counts.tolist())
    y = torch.cat([seg @ weight[i].double().T for i, seg in enumerate(segments)])
    (s64[:, None] * y).backward(grad.double())

    assert _rel(out, s64[:, None].detach() * y.detach()) < 1e-2
    assert _rel(gate_grad, g64.grad) < 2e-2
    assert _rel(up_grad, u64.grad) < 2e-2
    assert _rel(scores.grad, s64.grad) < 1e-2


def test_bf16_experts_keep_scaling_the_outputs():
    # BF16 routed experts run through SonicMoE on the EP receive buffer; the grouped path scales afterwards.
    experts = _experts(None)
    counts, x, scores, _ = _inputs(9)

    out = experts(x, counts, scores)
    expected = apply_routing_weights(experts(x, counts), scores)

    assert type(out.grad_fn).__name__ == "_ApplyRoutingWeightsBackward"
    # Padding rows past the expert counts are never written by the grouped GEMM.
    rows = int(counts.sum())
    torch.testing.assert_close(out[:rows], expected[:rows], rtol=0, atol=0)


def test_down_projection_lora_keeps_scaling_the_outputs():
    experts = _experts(128)
    rank = 8
    weight_a = torch.randn(_EXPERTS * rank, _HIDDEN, device="cuda").bfloat16() * 0.1
    weight_b = torch.randn(_DIM, _EXPERTS * rank, device="cuda").bfloat16() * 0.1
    experts._ap_lora_ab = {"w2": (weight_a, weight_b, rank, 0.5)}
    counts, x, scores, grad = _inputs(0)

    out = experts(x, counts, scores)
    expected = apply_routing_weights(experts(x, counts), scores)

    assert type(out.grad_fn).__name__ == "_ApplyRoutingWeightsBackward"
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_gate_up_lora_without_down_lora_keeps_the_scored_path():
    experts = _experts(128)
    rank = 8
    weight_a = torch.randn(_EXPERTS * rank, _DIM, device="cuda").bfloat16() * 0.1
    weight_b = torch.randn(_HIDDEN, _EXPERTS * rank, device="cuda").bfloat16() * 0.1
    experts._ap_lora_ab = {"w1": (weight_a, weight_b, rank, 0.5)}
    counts, x, scores, grad = _inputs(0)

    expected = apply_routing_weights(experts(x, counts), scores)
    expected.backward(grad)
    expected_dx = x.grad.clone()
    x.grad = None
    out = experts(x, counts, scores)
    out.backward(grad)

    assert type(out.grad_fn).__name__ == "_ScoredSwiGLUDownFnBackward"
    assert _rel(out, expected) < 1e-3
    assert _rel(x.grad, expected_dx) < 1e-2
