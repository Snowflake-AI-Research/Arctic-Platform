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
"""Block-FP8 routed experts with W8A8 forward GEMMs (moe.w8a8_experts, kernels/fp8_grouped_w8a8.py): the padded
layout, the quantization into it and the gather back, the grouped GEMM against the FP32 emulation of W8A8 and the
BF16-decode path, and the experts' outputs and straight-through gradients. One Hopper GPU; the refusals run on CPU."""

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


def _layout(counts):
    offsets = torch.cumsum(torch.tensor(counts, device="cuda"), 0, dtype=torch.int32)
    return offsets, w8.padded_layout(offsets, sum(counts))


@hopper
def test_padded_layout_starts_every_expert_on_a_128_row_block():
    offsets, layout = _layout(COUNTS)
    dst = layout.dst.long().tolist()
    starts = []
    for e, (count, end) in enumerate(zip(COUNTS, offsets.tolist())):
        if count:
            rows = dst[end - count : end]
            assert rows[0] % 128 == 0 and rows == list(range(rows[0], rows[0] + count)), (e, rows[0])
            starts.append(rows[0])
    assert starts == sorted(starts) and len(set(starts)) == len(starts)
    expected = torch.full((layout.padded_rows,), -1, dtype=torch.int32)
    for e, (count, end) in enumerate(zip(COUNTS, offsets.tolist())):
        expected[dst[end - count : end]] = e
    assert torch.equal(layout.m_indices.cpu(), expected)
    assert layout.padded_rows % 128 == 0 and layout.padded_rows >= dst[-1] + 1
    assert layout.dst.dtype == layout.m_indices.dtype == torch.int32

    _, empty = _layout([0, 0, 0])
    assert empty.rows == 0 and empty.dst.numel() == 0 and (empty.m_indices == -1).all()


@hopper
def test_quantize_into_padded_writes_the_per_group_quantization_to_the_padded_rows():
    from arctic_platform.model.implementations.kernels.fp8_quant import per_token_group_quant_fp8

    torch.manual_seed(0)
    _, layout = _layout(COUNTS)
    x = (torch.randn(layout.rows, 384, device="cuda") * 3).bfloat16()
    x[7] = 0  # all-zero groups take the eps floor
    q, s = w8.quantize_into_padded(x, layout)
    q_ref, s_ref = per_token_group_quant_fp8(x, 128, use_ue8m0=False)
    dst = layout.dst.long()
    assert torch.equal(q[dst].view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(s[dst], s_ref)
    # DeepGEMM's SM90 activation scales: MN-major over the padded rows.
    assert s.shape == (layout.padded_rows, 3) and s.stride() == (1, layout.padded_rows)


@hopper
@pytest.mark.parametrize("limit", [None, 0.75])
def test_swiglu_quantize_quantizes_the_bf16_swiglu(limit):
    from arctic_platform.model.implementations.kernels.fused_swiglu import packed_swiglu

    torch.manual_seed(0)
    _, layout = _layout(COUNTS)
    gate_up = torch.randn(layout.rows, 512, device="cuda").bfloat16()
    q, s = w8.swiglu_quantize_into_padded(gate_up, layout, limit)
    q_ref, s_ref = w8.quantize_into_padded(packed_swiglu(gate_up, limit), layout)
    dst = layout.dst.long()
    assert torch.equal(q[dst].view(torch.uint8), q_ref[dst].view(torch.uint8))
    assert torch.equal(s[dst], s_ref[dst])


@hopper
def test_unpad_rows_gathers_adds_the_delta_and_applies_the_scores():
    torch.manual_seed(0)
    _, layout = _layout(COUNTS)
    y = torch.randn(layout.padded_rows, 640, device="cuda").bfloat16()
    delta = torch.randn(layout.rows, 256, device="cuda").bfloat16()
    scores = torch.rand(layout.rows, device="cuda")
    out = torch.full((layout.rows + 3, 640), 7.0, device="cuda", dtype=torch.bfloat16)
    w8.unpad_rows_into(out[:, 384:], y[:, 384:], layout, delta, scores)
    w8.unpad_rows_into(out[:, :384], y[:, :384], layout)
    rows = y[layout.dst.long()].float()
    rows[:, 384:] = (rows[:, 384:] + delta.float()) * scores[:, None]
    assert torch.equal(out[: layout.rows], rows.bfloat16())
    assert (out[layout.rows :] == 7).all(), "rows past the routed rows are left as they are"


@hopper
@pytest.mark.parametrize("stacked", [False, True], ids=["one", "side-by-side"])
def test_grouped_w8a8_mm_forward_and_its_straight_through_input_gradient(stacked):
    torch.manual_seed(0)
    offsets, layout = _layout(COUNTS)
    rows, experts = layout.rows, len(COUNTS)
    w1, s1 = _quantized(experts, 256, 768)
    w3, s3 = _quantized(experts, 384, 768)
    weight, scale = ((w1, w3), (s1, s3)) if stacked else (w1, s1)
    width = 640 if stacked else 256
    x = torch.randn(rows + 9, 768, device="cuda").bfloat16().requires_grad_(True)
    deltas = (torch.randn(rows, 256, device="cuda").bfloat16().requires_grad_(True),) + ((None,) if stacked else ())
    out = fp8.grouped_fp8_w8a8_mm(x, weight, scale, offsets, rows, 128, layout, deltas=deltas)
    grad = torch.randn_like(out)
    out.backward(grad)

    decoded = torch.cat(
        [fp8.dequantize_from_fp8_blockwise(w, s, dtype=torch.float32) for w, s in zip((w1, w3), (s1, s3))], 1
    )[:, :width]
    xq = _quant_ste(x[:rows].detach().float())
    tokens, exact = torch.split(xq, COUNTS), torch.split(x[:rows].detach().float(), COUNTS)
    emulated = torch.cat([t @ decoded[e].T for e, t in enumerate(tokens)])
    emulated[:, :256] += deltas[0].detach().float()
    unquantized = torch.cat([t @ decoded[e].T for e, t in enumerate(exact)])
    unquantized[:, :256] += deltas[0].detach().float()
    assert out.shape == (rows + 9, width) and (out[rows:] == 0).all()
    assert _rel(out[:rows], emulated) < 5e-3
    assert _rel(out[:rows], unquantized) < 5e-2  # the activation quantization: ~3% per GEMM

    # The backward is the decoded path's, bit for bit; the delta gets its columns of the output gradient.
    x_ref = x.detach().clone().requires_grad_(True)
    fp8.grouped_fp8_decoded_mm(x_ref, weight, scale, offsets, rows, 128).backward(grad)
    assert torch.equal(x.grad, x_ref.grad) and (x.grad[rows:] == 0).all()
    assert torch.equal(deltas[0].grad, grad[:rows, :256])


def _experts(lora: tuple, limit: float | None):
    torch.manual_seed(0)
    experts, dim, hidden, rank = len(COUNTS), 768, 512, 8
    module = GroupedExperts(dim, hidden, experts, use_grouped_mm=True, fp8_block_size=128, swiglu_limit=limit).cuda()
    for name, shape in (("w1", (hidden, dim)), ("w2", (dim, hidden)), ("w3", (hidden, dim))):
        w, s = _quantized(experts, *shape)
        getattr(module, name).data.copy_(w)
        getattr(module, f"{name}_scale_inv").data.copy_(s)
    module._ap_lora_ab = {}
    for name, (out_f, in_f) in (("w1", (hidden, dim)), ("w2", (dim, hidden)), ("w3", (hidden, dim))):
        if name in lora:
            a = (torch.randn(experts * rank, in_f, device="cuda") * 0.05).bfloat16().requires_grad_(True)
            b = (torch.randn(out_f, rank * experts, device="cuda") * 0.05).bfloat16().requires_grad_(True)
            module._ap_lora_ab[name] = (a, b, rank, 0.5)
    module.set_ep_comm_backend("deepep")
    return module


def _reference(module, x, scores, quantize: bool):
    """FP32 experts with LoRA on the decoded weights; ``quantize`` emulates the W8A8 forward, straight through, with
    every tensor rounded to BF16 where the kernels round it (GEMM outputs, the LoRA intermediate and delta, their sum
    and SwiGLU): the quantization of the SwiGLU output turns those roundings into whole FP8 steps. The scored down
    projection (no down-projection LoRA) takes its score gradient from the unquantized SwiGLU, as
    ``scored_swiglu_down`` does."""
    w = {
        n: fp8.dequantize_from_fp8_blockwise(
            getattr(module, n), getattr(module, f"{n}_scale_inv"), dtype=torch.float32
        )
        for n in ("w1", "w2", "w3")
    }
    ab = {n: (a.float(), b.float(), r, sc) for n, (a, b, r, sc) in module._ap_lora_ab.items()}
    q = _quant_ste if quantize else (lambda t: t)
    r = _bf16_ste if quantize else (lambda t: t)
    limit = module.swiglu_limit
    pieces, start = [], 0
    for e, count in enumerate(COUNTS):
        xs, s = x[start : start + count], scores[start : start + count, None]
        start += count

        def lora(name, inputs):
            if name not in ab:
                return 0
            a, b, rank, sc = ab[name]
            down = r(inputs @ a.view(len(COUNTS), rank, -1)[e].T) * sc
            return r(down @ b.view(b.shape[0], rank, -1)[:, :, e].T)

        gate = r(r(q(xs) @ w["w1"][e].T) + lora("w1", xs))
        up = r(r(q(xs) @ w["w3"][e].T) + lora("w3", xs))
        if limit is not None:
            gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        h = r(F.silu(gate) * up)
        y = r(q(h) @ w["w2"][e].T) + lora("w2", h)
        if "w2" in ab:
            pieces.append(y * s)
        else:
            plain = h @ w["w2"][e].T
            pieces.append(y * s.detach() + plain * (s - s.detach()))
    return torch.cat(pieces)


@hopper
@pytest.mark.parametrize(
    "lora, limit",
    [(("w1", "w2", "w3"), None), (("w1", "w3"), None), (("w1", "w2", "w3"), 3.0), ((), None)],
    ids=["lora-all", "lora-gate-up-scored", "lora-all-limit", "no-lora-scored"],
)
def test_w8a8_experts_match_the_emulated_forward_and_its_straight_through_gradients(w8a8_on, lora, limit):
    module = _experts(lora, limit)
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


@hopper
def test_w8a8_and_decoded_experts_share_the_backward(w8a8_on):
    """Fed the same output gradient and the same saved activations, the W8A8 down projection's input gradient is the
    decoded path's bit for bit: the forward GEMMs are the only difference."""
    module = _experts(("w1", "w2", "w3"), None)
    rows = sum(COUNTS)
    offsets, layout = _layout(COUNTS)
    h = torch.randn(rows, 512, device="cuda").bfloat16()
    grad = torch.randn(rows, 768, device="cuda").bfloat16()
    dx = []
    for w8a8 in (True, False):
        hi = h.clone().requires_grad_(True)
        if w8a8:
            out = fp8.grouped_fp8_w8a8_mm(hi, module.w2, module.w2_scale_inv, offsets, rows, 128, layout)
        else:
            out = fp8.grouped_fp8_decoded_mm(hi, module.w2, module.w2_scale_inv, offsets, rows, 128)
        out.backward(grad)
        dx.append(hi.grad)
    assert torch.equal(dx[0], dx[1])


@hopper
def test_w8a8_experts_take_all_empty_calls(w8a8_on):
    module = _experts(("w1", "w2", "w3"), None)
    x = torch.randn(4, 768, device="cuda").bfloat16().requires_grad_(True)
    out = module(x, torch.zeros(len(COUNTS), dtype=torch.int64, device="cuda"), torch.rand(4, device="cuda"))
    out.float().sum().backward()
    assert out.shape == (4, 768) and not out.any() and not x.grad.any()


def test_w8a8_is_refused_for_experts_that_are_not_block_fp8(w8a8_on):
    with pytest.raises(ValueError, match="w8a8_experts runs block-FP8 routed experts"):
        GroupedExperts(256, 128, 4, use_grouped_mm=True)
    with pytest.raises(ValueError, match="moe.grouped_mm"):
        GroupedExperts(256, 128, 4, use_grouped_mm=False, fp8_block_size=128)
    assert GroupedExperts(256, 128, 4, use_grouped_mm=True, fp8_block_size=128).w8a8


def test_w8a8_is_off_by_default():
    assert not GroupedExperts(256, 128, 4, use_grouped_mm=True, fp8_block_size=128).w8a8
    assert not GroupedExperts(256, 128, 4, use_grouped_mm=True).w8a8


def test_w8a8_experts_refuse_paths_without_the_grouped_gemm(w8a8_on):
    """CPU activations (and fused PEFT deltas) take the per-expert FP8 path, which has no W8A8 forward."""
    module = GroupedExperts(256, 128, 4, use_grouped_mm=True, fp8_block_size=128)
    module.set_ep_comm_backend("deepep")
    with pytest.raises(NotImplementedError, match="w8a8_experts"):
        module(torch.randn(6, 256).bfloat16(), torch.tensor([1, 2, 3, 0]))
