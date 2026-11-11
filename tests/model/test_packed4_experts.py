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
"""Packed 4-bit routed experts (W4AFP8 INT4, NVFP4): decode kernels, the grouped forward and backward, PEFT shapes."""

import pytest
import torch
from torch import nn

from arctic_platform.model.implementations.kernels.packed4_dequant import dequantize_int4_groups_reference
from arctic_platform.model.implementations.kernels.packed4_dequant import dequantize_mxfp4_reference
from arctic_platform.model.implementations.kernels.packed4_dequant import dequantize_nvfp4_reference

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the decode kernels run on CUDA")


def _random_int4(experts, rows, cols, group, device="cpu"):
    codes = torch.randint(-7, 8, (experts, rows, cols), dtype=torch.int16, device=device)
    packed = ((codes[..., 0::2] & 0x0F) | ((codes[..., 1::2] & 0x0F) << 4)).to(torch.uint8).view(torch.int8)
    scale = (torch.rand(experts, rows, cols // group, device=device) * 0.02 + 1e-3).bfloat16()
    return packed, scale, codes


def _random_nvfp4(experts, rows, cols, device="cpu"):
    packed = torch.randint(0, 256, (experts, rows, cols // 2), dtype=torch.uint8, device=device)
    scale = (torch.rand(experts, rows, cols // 16, device=device) * 400 + 1).to(torch.float8_e4m3fn)
    global_scale = torch.rand(experts, device=device) * 1e-4 + 1e-5
    return packed, scale, global_scale


def test_int4_reference_unpacks_low_nibble_first_and_signed():
    packed, scale, codes = _random_int4(2, 4, 256, 128)
    restored = dequantize_int4_groups_reference(packed, torch.ones_like(scale))
    assert torch.equal(restored, codes.float())
    scaled = dequantize_int4_groups_reference(packed, scale)
    torch.testing.assert_close(scaled, codes.float() * scale.float().repeat_interleave(128, dim=-1))


def test_nvfp4_reference_matches_the_e2m1_table():
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
    packed = torch.arange(16, dtype=torch.uint8)[0::2] | (torch.arange(16, dtype=torch.uint8)[1::2] << 4)
    restored = dequantize_nvfp4_reference(packed.view(1, 8), torch.ones(1, 1).to(torch.float8_e4m3fn), torch.ones(1))
    assert torch.equal(restored.view(-1), values)


@cuda
@pytest.mark.parametrize("cols", [256, 2048, 6144])
def test_int4_decode_matches_reference(cols):
    from arctic_platform.model.implementations.kernels.packed4_dequant import dequantize_int4_groups

    packed, scale, _ = _random_int4(3, 96, cols, 128, device="cuda")
    out = dequantize_int4_groups(packed, scale)
    assert torch.equal(out, dequantize_int4_groups_reference(packed, scale).bfloat16())


@cuda
def test_int4_decode_side_by_side_rows():
    from arctic_platform.model.implementations.kernels.packed4_dequant import dequantize_int4_groups

    gate, gate_scale, _ = _random_int4(2, 64, 512, 128, device="cuda")
    up, up_scale, _ = _random_int4(2, 64, 512, 128, device="cuda")
    out = torch.full((2, 128, 512), float("nan"), dtype=torch.bfloat16, device="cuda")
    dequantize_int4_groups(gate, gate_scale, out=out, out_row=0)
    dequantize_int4_groups(up, up_scale, out=out, out_row=64)
    want = torch.cat(
        [dequantize_int4_groups_reference(gate, gate_scale), dequantize_int4_groups_reference(up, up_scale)], dim=1
    )
    assert torch.equal(out, want.bfloat16())


@cuda
def test_nvfp4_decode_matches_reference():
    from arctic_platform.model.implementations.kernels.packed4_dequant import dequantize_nvfp4

    packed, scale, global_scale = _random_nvfp4(3, 80, 512, device="cuda")
    out = dequantize_nvfp4(packed, scale, global_scale)
    assert torch.equal(out, dequantize_nvfp4_reference(packed, scale, global_scale).bfloat16())


def _int4_experts(experts, dim, hidden, device):
    from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts

    module = GroupedExperts(dim, hidden, experts, use_grouped_mm=True, fp8_block_size=128, packed4="int4").to(device)
    reference = {}
    for name, (rows, cols) in (("w1", (hidden, dim)), ("w2", (dim, hidden)), ("w3", (hidden, dim))):
        packed, scale, _ = _random_int4(experts, rows, cols, 128, device=device)
        getattr(module, name).data.copy_(packed)
        getattr(module, f"{name}_scale_inv").data.copy_(scale)
        reference[name] = dequantize_int4_groups_reference(packed, scale).bfloat16()
    return module, reference


def _dense_experts(reference, x, counts, scores):
    outs = []
    for e, rows in enumerate(torch.split(x, counts)):
        gate = rows @ reference["w1"][e].float().T
        up = rows @ reference["w3"][e].float().T
        outs.append((torch.nn.functional.silu(gate) * up) @ reference["w2"][e].float().T)
    return torch.cat(outs) * scores[:, None]


@cuda
def test_int4_grouped_experts_match_the_decoded_bf16_experts():
    torch.manual_seed(0)
    experts, dim, hidden = 4, 512, 256
    module, reference = _int4_experts(experts, dim, hidden, "cuda")
    counts = [5, 0, 17, 9]
    x = torch.randn(sum(counts), dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scores = torch.rand(sum(counts), device="cuda", requires_grad=True)
    tokens = torch.tensor(counts, device="cuda")
    out = module(x, tokens, scores)

    x_ref = x.detach().clone().requires_grad_(True)
    scores_ref = scores.detach().clone().requires_grad_(True)
    want = _dense_experts(reference, x_ref.float(), counts, scores_ref)
    torch.testing.assert_close(out.float(), want, rtol=2e-2, atol=2e-2)
    grad = torch.randn_like(want)
    out.float().backward(grad)
    want.backward(grad)
    torch.testing.assert_close(x.grad.float(), x_ref.grad.float(), rtol=3e-2, atol=3e-2)
    # Each score gradient is a dot product over dim of the output gradient and the expert output; the kernel reads
    # the SwiGLU output rounded to BF16, so compare the whole vector rather than elements that cancel.
    error = (scores.grad.float() - scores_ref.grad.float()).norm() / scores_ref.grad.float().norm()
    assert error < 2e-2, error


def test_peft_sizes_int4_expert_lora_from_the_logical_shape():
    import peft

    from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts
    from arctic_platform.model.patches.peft import apply_peft

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = GroupedExperts(512, 256, 4, use_grouped_mm=True, fp8_block_size=128, packed4="int4")

    model = Block()
    wrapped = apply_peft(
        model,
        {
            "peft_type": "LORA",
            "r": 8,
            "lora_alpha": 8,
            "target_parameters": ["experts.w1", "experts.w2", "experts.w3"],
        },
    )
    from peft.tuners.lora.layer import ParamWrapper

    shapes = {
        module.parameter_name: (
            tuple(module.lora_A["default"].weight.shape),
            tuple(module.lora_B["default"].weight.shape),
        )
        for module in wrapped.modules()
        if isinstance(module, ParamWrapper)
    }
    # PEFT's ParamWrapper layout: A is [E * r, in], B is [out, E * r], with in and out unpacked.
    assert shapes == {"w1": ((32, 512), (256, 32)), "w2": ((32, 256), (512, 32)), "w3": ((32, 512), (256, 32))}
    assert not any(hasattr(p, "ds_shape") for p in model.parameters())
    assert isinstance(wrapped, peft.PeftModel)


def _random_mxfp4(experts, rows, cols, device="cpu"):
    packed = torch.randint(-128, 128, (experts, rows, cols // 2), dtype=torch.int8, device=device)
    # E8M0 exponents around 2^-7 .. 2^-3, the official checkpoint's range for 2304 x 5120 experts.
    exponent = torch.randint(120, 125, (experts, rows, cols // 32), dtype=torch.uint8, device=device)
    return packed, exponent.view(torch.float8_e8m0fnu).float(), exponent


def _lora(experts, rank, in_f, out_f, device):
    a = torch.nn.Parameter((torch.randn(experts * rank, in_f, device=device) * 0.05).bfloat16())
    b = torch.nn.Parameter((torch.randn(out_f, rank * experts, device=device) * 0.05).bfloat16())
    return a, b


def _lora_delta(a, b, e, experts, rank, scaling):
    # PEFT's ParamWrapper layout: A [E * r, in] expert-major, B [out, r * E] with the expert index fastest.
    a_e = a.float().view(experts, rank, -1)[e]
    b_e = b.float().view(b.shape[0], rank, experts)[:, :, e]
    return (b_e @ a_e) * scaling


def test_mxfp4_reference_matches_deepseek_v4_dequantize():
    from arctic_platform.model.implementations.deepseek_v4.dequantize import dequantize_mxfp4

    packed, scale, exponent = _random_mxfp4(3, 16, 256)
    want = dequantize_mxfp4(packed, scale)
    assert torch.equal(dequantize_mxfp4_reference(packed, scale).bfloat16(), want)
    assert torch.equal(
        dequantize_mxfp4_reference(packed, exponent).bfloat16(), want
    ), "E8M0 bytes decode to the same scales"
    assert torch.equal(
        dequantize_mxfp4_reference(packed, torch.zeros_like(exponent)),
        dequantize_mxfp4_reference(packed, torch.full_like(scale, 2.0**-127)),
    )


@cuda
@pytest.mark.parametrize("e8m0", [False, True])
@pytest.mark.parametrize("cols", [256, 2304, 5120])
def test_mxfp4_decode_matches_reference(cols, e8m0):
    from arctic_platform.model.implementations.kernels.packed4_dequant import dequantize_mxfp4_packed

    packed, scale, exponent = _random_mxfp4(3, 96, cols, device="cuda")
    stored = exponent if e8m0 else scale
    assert torch.equal(dequantize_mxfp4_packed(packed, stored), dequantize_mxfp4_reference(packed, scale).bfloat16())
    out = torch.full((3, 200, cols), float("nan"), dtype=torch.bfloat16, device="cuda")
    dequantize_mxfp4_packed(packed, stored, out=out, out_row=104)
    assert torch.equal(out[:, 104:], dequantize_mxfp4_reference(packed, scale).bfloat16())


@cuda
def test_mxfp4_grouped_experts_with_lora_match_the_decoded_bf16_experts():
    from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts

    torch.manual_seed(0)
    experts, dim, hidden, rank, scaling = 4, 512, 256, 8, 0.5
    module = GroupedExperts(dim, hidden, experts, use_grouped_mm=True, packed4="mxfp4", swiglu_limit=10.0).cuda()
    reference, ab = {}, {}
    for name, (rows, cols) in (("w1", (hidden, dim)), ("w2", (dim, hidden)), ("w3", (hidden, dim))):
        packed, scale, _ = _random_mxfp4(experts, rows, cols, device="cuda")
        getattr(module, name).data.copy_(packed)
        getattr(module, f"{name}_scale_inv").data.copy_(scale)
        reference[name] = dequantize_mxfp4_reference(packed, scale).bfloat16().float()
        ab[name] = (*_lora(experts, rank, cols, rows, "cuda"), rank, scaling)
    module._ap_lora_ab = ab
    counts = [5, 0, 17, 9]
    x = torch.randn(sum(counts), dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scores = torch.rand(sum(counts), device="cuda", requires_grad=True)
    out = module(x, torch.tensor(counts, device="cuda"), scores)

    from arctic_platform.model.implementations.moe.layers.moe import _apply_swiglu

    x_ref = x.detach().float().requires_grad_(True)
    ab_ref = {
        n: (a.detach().clone().requires_grad_(True), b.detach().clone().requires_grad_(True))
        for n, (a, b, _, _) in ab.items()
    }
    pieces = []
    for e, rows in enumerate(torch.split(x_ref, counts)):
        w = {n: reference[n][e] + _lora_delta(*ab_ref[n], e, experts, rank, scaling) for n in reference}
        pieces.append(_apply_swiglu(rows @ w["w1"].T, rows @ w["w3"].T, 10.0) @ w["w2"].T)
    want = torch.cat(pieces) * scores[:, None].detach()
    error = ((out.float() - want).norm() / want.norm()).item()
    assert error < 2e-2, error
    grad = torch.randn_like(want)
    out.float().backward(grad)
    want.backward(grad)
    for got, ref, label in [(x.grad, x_ref.grad, "x")] + [
        (ab[n][k].grad, ab_ref[n][k].grad, f"{n}.{'AB'[k]}") for n in ab for k in (0, 1)
    ]:
        err = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        assert err < 3e-2, (label, err)


def test_peft_sizes_mxfp4_expert_lora_from_the_logical_shape():
    from peft.tuners.lora.layer import ParamWrapper

    from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts
    from arctic_platform.model.patches.peft import apply_peft

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = GroupedExperts(512, 256, 4, use_grouped_mm=True, packed4="mxfp4")

    wrapped = apply_peft(
        Block(),
        {
            "peft_type": "LORA",
            "r": 8,
            "lora_alpha": 8,
            "target_parameters": ["experts.w1", "experts.w2", "experts.w3"],
        },
    )
    shapes = {
        module.parameter_name: (
            tuple(module.lora_A["default"].weight.shape),
            tuple(module.lora_B["default"].weight.shape),
        )
        for module in wrapped.modules()
        if isinstance(module, ParamWrapper)
    }
    assert shapes == {"w1": ((32, 512), (256, 32)), "w2": ((32, 256), (512, 32)), "w3": ((32, 512), (256, 32))}


def test_mxfp4_scales_compact_to_e8m0_bytes_exactly():
    from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts
    from arctic_platform.model.implementations.moe.layers.moe import compact_mxfp4_expert_scales

    module = GroupedExperts(64, 32, 2, use_grouped_mm=True, packed4="mxfp4")
    decoded = {}
    for name in ("w1", "w2", "w3"):
        packed = getattr(module, name)
        packed.data.copy_(torch.randint(-128, 128, packed.shape, dtype=torch.int8))
        exponent = torch.randint(0, 255, getattr(module, f"{name}_scale_inv").shape, dtype=torch.uint8)
        exponent.view(-1)[0] = 0  # the subnormal 2^-127
        getattr(module, f"{name}_scale_inv").data.copy_(exponent.view(torch.float8_e8m0fnu).float())
        decoded[name] = dequantize_mxfp4_reference(packed.data, getattr(module, f"{name}_scale_inv").data)
    before = sum(getattr(module, f"{n}_scale_inv").numel() * 4 for n in ("w1", "w2", "w3"))
    assert compact_mxfp4_expert_scales(nn.ModuleList([module])) == before * 3 // 4
    for name in ("w1", "w2", "w3"):
        scale = getattr(module, f"{name}_scale_inv")
        assert scale.dtype == torch.uint8 and not scale.requires_grad
        assert torch.equal(dequantize_mxfp4_reference(getattr(module, name).data, scale.data), decoded[name])
    assert compact_mxfp4_expert_scales(nn.ModuleList([module])) == 0, "idempotent"
    module.w1_scale_inv.data = torch.full(module.w1_scale_inv.shape, 0.3)
    with pytest.raises(ValueError, match="powers of two"):
        module.compact_mxfp4_scales()
