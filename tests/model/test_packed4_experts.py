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


def test_per_tensor_fp8_dense_mlp_becomes_exact_block_fp8():
    from arctic_platform.model.implementations.fp8 import dequantize_from_fp8_blockwise
    from arctic_platform.model.implementations.glm52.models.glm_moe_dsa.converting_glm_moe_dsa import (
        convert_hf_layer_to_tt,
    )

    weight = (torch.randn(300, 256) * 50).to(torch.float8_e4m3fn)
    scale = torch.tensor(0.0123)
    state = {
        "model.layers.0.mlp.gate_proj.weight": weight,
        "model.layers.0.mlp.gate_proj.weight_scale": scale,
        "model.layers.0.mlp.gate_proj.input_scale": torch.tensor(0.5),
    }
    convert_hf_layer_to_tt(state, 0)
    assert set(state) == {"model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.gate_proj.weight_scale_inv"}
    blocks = state["model.layers.0.mlp.gate_proj.weight_scale_inv"]
    assert blocks.shape == (3, 2)
    restored = dequantize_from_fp8_blockwise(weight, blocks, dtype=torch.float32)
    assert torch.equal(restored, weight.float() * scale)


def test_nvfp4_expert_scales_stack_per_projection():
    from arctic_platform.model.implementations.glm52.models.glm_moe_dsa.converting_glm_moe_dsa import (
        convert_hf_layer_to_tt,
    )

    experts, dim, hidden = 2, 64, 32
    state = {
        "model.layers.3.mlp.gate.weight": torch.randn(experts, dim),
        "model.layers.3.mlp.gate.e_score_correction_bias": torch.zeros(experts),
    }
    for j in range(experts):
        for proj, (rows, cols) in (
            ("gate_proj", (hidden, dim)),
            ("up_proj", (hidden, dim)),
            ("down_proj", (dim, hidden)),
        ):
            stem = f"model.layers.3.mlp.experts.{j}.{proj}"
            state[f"{stem}.weight"] = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8)
            state[f"{stem}.weight_scale"] = torch.rand(rows, cols // 16).to(torch.float8_e4m3fn)
            state[f"{stem}.weight_scale_2"] = torch.tensor(1e-3 * (j + 1))
            state[f"{stem}.input_scale"] = torch.tensor(1.0)
    for proj, (rows, cols) in (("gate_proj", (hidden, dim)), ("up_proj", (hidden, dim)), ("down_proj", (dim, hidden))):
        state[f"model.layers.3.mlp.shared_experts.{proj}.weight"] = torch.randn(rows, cols).bfloat16()
    w1_first = state["model.layers.3.mlp.experts.1.gate_proj.weight"].clone()
    convert_hf_layer_to_tt(state, 3)
    prefix = "model.layers.3.mlp.experts"
    assert state[f"{prefix}.w1"].shape == (experts, hidden, dim // 2) and torch.equal(
        state[f"{prefix}.w1"][1], w1_first
    )
    assert state[f"{prefix}.w2"].shape == (experts, dim, hidden // 2)
    assert state[f"{prefix}.w1_scale_inv"].shape == (experts, hidden, dim // 16)
    assert state[f"{prefix}.w1_scale_inv"].dtype == torch.float8_e4m3fn
    torch.testing.assert_close(state[f"{prefix}.w2_scale_2"], torch.tensor([1e-3, 2e-3]))
    assert not [k for k in state if "input_scale" in k or ".experts.0." in k]
