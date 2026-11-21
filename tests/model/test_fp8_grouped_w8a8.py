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
"""Block-FP8 routed experts with W8A8 forward GEMMs (moe.w8a8_experts, kernels/fp8_grouped_w8a8.py) against the FP32
emulation of W8A8 and the BF16-decode path, on one Hopper GPU."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from arctic_platform.model.implementations import fp8
from arctic_platform.model.implementations.kernels import fp8_grouped_w8a8 as w8
from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts

hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9, reason="DeepGEMM's SM90 FP8 kernels"
)
# Empty experts first, inside and last; rows on, just past and well past 128-row boundaries.
COUNTS = [0, 5, 128, 129, 0, 300, 1, 0]


@pytest.fixture
def w8a8_on():
    w8.configure_from_options(True)
    yield
    w8.configure_from_options(None)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _quantized(*shape: int) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [
        fp8.quantize_to_fp8_blockwise(torch.randn(*shape[1:], device="cuda") * 0.05, 128) for _ in range(shape[0])
    ]
    return torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])


def _quant_ste(t: torch.Tensor) -> torch.Tensor:
    """FP32 emulation of the 1 x 128 E4M3 activation quantization, straight through in backward."""
    groups = t.reshape(t.shape[0], t.shape[1] // 128, 128)
    scale = groups.detach().abs().amax(-1, keepdim=True).clamp(min=1e-10) / 448.0
    q = ((groups.detach() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).reshape(t.shape)
    return t + (q - t).detach()


def _bf16_ste(t: torch.Tensor) -> torch.Tensor:
    """The BF16 rounding of a kernel's output, straight through in backward."""
    return t + (t.bfloat16().float() - t).detach()


def _experts():
    """GLM-5.3's LoRA layout: LoRA on w1, w2 and w3, no SwiGLU limit."""
    torch.manual_seed(0)
    experts, dim, hidden, rank = len(COUNTS), 768, 512, 8
    module = GroupedExperts(dim, hidden, experts, use_grouped_mm=True, fp8_block_size=128).cuda()
    for name, shape in (("w1", (hidden, dim)), ("w2", (dim, hidden)), ("w3", (hidden, dim))):
        w, s = _quantized(experts, *shape)
        getattr(module, name).data.copy_(w)
        getattr(module, f"{name}_scale_inv").data.copy_(s)
    module._ap_lora_ab = {}
    for name, (out_f, in_f) in (("w1", (hidden, dim)), ("w2", (dim, hidden)), ("w3", (hidden, dim))):
        a = (torch.randn(experts * rank, in_f, device="cuda") * 0.05).bfloat16().requires_grad_(True)
        b = (torch.randn(out_f, rank * experts, device="cuda") * 0.05).bfloat16().requires_grad_(True)
        module._ap_lora_ab[name] = (a, b, rank, 0.5)
    module.set_ep_comm_backend("deepep")
    return module


def _reference(module, x, scores, quantize: bool):
    """FP32 experts with LoRA on the decoded weights; ``quantize`` emulates the W8A8 forward, straight through, with
    every tensor rounded to BF16 where the kernels round it (the quantization of SwiGLU makes those whole FP8 steps).
    """
    w = {
        n: fp8.dequantize_from_fp8_blockwise(
            getattr(module, n), getattr(module, f"{n}_scale_inv"), dtype=torch.float32
        )
        for n in ("w1", "w2", "w3")
    }
    ab = {n: (a.float(), b.float(), r, sc) for n, (a, b, r, sc) in module._ap_lora_ab.items()}
    q = _quant_ste if quantize else (lambda t: t)
    r = _bf16_ste if quantize else (lambda t: t)
    pieces, start = [], 0
    for e, count in enumerate(COUNTS):
        xs, s = x[start : start + count], scores[start : start + count, None]
        start += count

        def lora(name, inputs):
            a, b, rank, sc = ab[name]
            down = r(inputs @ a.view(len(COUNTS), rank, -1)[e].T) * sc
            return r(down @ b.view(b.shape[0], rank, -1)[:, :, e].T)

        gate = r(r(q(xs) @ w["w1"][e].T) + lora("w1", xs))
        up = r(r(q(xs) @ w["w3"][e].T) + lora("w3", xs))
        h = r(F.silu(gate) * up)
        pieces.append((r(q(h) @ w["w2"][e].T) + lora("w2", h)) * s)
    return torch.cat(pieces)


@hopper
def test_w8a8_experts_match_the_emulated_forward_and_its_straight_through_gradients(w8a8_on):
    module = _experts()
    assert module.w8a8
    rows = sum(COUNTS)
    x = torch.randn(rows + 5, 768, device="cuda").bfloat16().requires_grad_(True)
    scores = torch.rand(rows + 5, device="cuda", requires_grad=True)
    out = module(x, torch.tensor(COUNTS, device="cuda"), scores)
    grad = torch.randn_like(out)
    out.backward(grad)

    x_ref = x.detach()[:rows].float().requires_grad_(True)
    scores_ref = scores.detach()[:rows].clone().requires_grad_(True)
    factors = [t for name in sorted(module._ap_lora_ab) for t in module._ap_lora_ab[name][:2]]
    grads = [t.grad.clone() for t in factors]
    for t in factors:
        t.grad = None
    want = _reference(module, x_ref, scores_ref, quantize=True)
    want.backward(grad[:rows].float())
    assert (out[rows:] == 0).all() and (x.grad[rows:] == 0).all() and (scores.grad[rows:] == 0).all()
    assert _rel(out[:rows], want) < 1e-2
    assert _rel(x.grad[:rows], x_ref.grad) < 1e-2
    assert _rel(scores.grad[:rows], scores_ref.grad) < 1e-2
    for got, t in zip(grads, factors):
        assert _rel(got, t.grad) < 1e-2
    # Against the unquantized reference (what the BF16-decode path computes) only the activation quantization differs.
    with torch.no_grad():
        exact = _reference(module, x_ref.detach(), scores_ref.detach(), quantize=False)
    assert _rel(out[:rows], exact) < 6e-2
