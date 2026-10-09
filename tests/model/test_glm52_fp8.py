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

"""Native GLM-5.2 finegrained-FP8 weights stay quantized until the GEMM."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from arctic_platform.model.implementations import fp8 as fp8_mod
from arctic_platform.model.implementations.fp8 import BlockFp8Linear
from arctic_platform.model.implementations.fp8 import _require_deep_gemm
from arctic_platform.model.implementations.fp8 import _require_fp8_state
from arctic_platform.model.implementations.fp8 import dequantize_from_fp8_blockwise
from arctic_platform.model.implementations.fp8 import fp8_weight_block_size
from arctic_platform.model.implementations.fp8 import make_linear
from arctic_platform.model.implementations.fp8 import quantize_to_fp8_blockwise
from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts
from arctic_platform.model.implementations.moe.layers.moe import _packed_expert_linear


def test_fp8_roundtrip_matches_blockwise_scale():
    torch.manual_seed(0)
    weight = torch.randn(256, 128, dtype=torch.bfloat16)
    q, scale = quantize_to_fp8_blockwise(weight, block_size=128)
    restored = dequantize_from_fp8_blockwise(q, scale, block_size=128, dtype=torch.bfloat16)
    assert q.dtype is torch.float8_e4m3fn
    assert restored.dtype is torch.bfloat16
    torch.testing.assert_close(restored.float(), weight.float(), rtol=0.15, atol=0.15)


def test_block_fp8_linear_matches_dequant_gemm():
    torch.manual_seed(2)
    linear = BlockFp8Linear(128, 256, block_size=128)
    bf16 = torch.randn(256, 128, dtype=torch.bfloat16)
    q, scale = quantize_to_fp8_blockwise(bf16, block_size=128)
    with torch.no_grad():
        linear.weight.copy_(q)
        linear.weight_scale_inv.copy_(scale)
    x = torch.randn(3, 128, dtype=torch.bfloat16)
    out = linear(x)
    ref = torch.nn.functional.linear(x, dequantize_from_fp8_blockwise(q, scale, dtype=x.dtype))
    # CPU falls back to dequant+BF16 GEMM. CUDA uses DeepGEMM W8A8 (1x128 act quant).
    if out.device.type == "cpu":
        torch.testing.assert_close(out.float(), ref.float())
    else:
        torch.testing.assert_close(out.float(), ref.float(), rtol=0.05, atol=0.5)
    assert isinstance(make_linear(128, 256, fp8_block_size=128), BlockFp8Linear)
    assert isinstance(make_linear(128, 256, fp8_block_size=None), nn.Linear)


def test_block_fp8_linear_backward_matches_dequant_dx():
    torch.manual_seed(4)
    linear = BlockFp8Linear(128, 256, block_size=128)
    q, scale = quantize_to_fp8_blockwise(torch.randn(256, 128, dtype=torch.bfloat16), block_size=128)
    with torch.no_grad():
        linear.weight.copy_(q)
        linear.weight_scale_inv.copy_(scale)
    x = torch.randn(5, 128, dtype=torch.bfloat16, requires_grad=True)
    linear(x).sum().backward()
    w = dequantize_from_fp8_blockwise(q, scale, dtype=torch.float32)
    x_ref = x.detach().clone().float().requires_grad_(True)
    torch.nn.functional.linear(x_ref, w).sum().backward()
    torch.testing.assert_close(x.grad.float(), x_ref.grad.float(), rtol=0.05, atol=0.05)


@pytest.mark.parametrize("default_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_expert_scales_are_fp32_regardless_of_default_dtype(default_dtype):
    """The model is built under a bf16 default dtype, which the scales must not inherit."""
    from arctic_platform.model.implementations.moe.layers.moe import BCFeedForward

    prev = torch.get_default_dtype()
    torch.set_default_dtype(default_dtype)
    try:
        grouped = GroupedExperts(dim=128, hidden_dim=256, num_experts=2, use_grouped_mm=False, fp8_block_size=128)
        shared = BCFeedForward(dim=128, hidden_dim=256, fp8_block_size=128)
    finally:
        torch.set_default_dtype(prev)

    for module in (grouped, shared):
        for name in ("w1_scale_inv", "w2_scale_inv", "w3_scale_inv"):
            assert getattr(module, name).dtype == torch.float32, f"{type(module).__name__}.{name}"


def test_fp8_weight_block_size_from_hf_config():
    class Cfg:
        quantization_config = {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        }

    assert fp8_weight_block_size(Cfg()) == 128
    assert fp8_weight_block_size(type("Empty", (), {})()) is None


def test_bf16_expert_lora_still_applies_pefts_fused_parametrization():
    """Only the FP8 forward reads the unfused A/B store.

    A BF16 base has to keep reading the delta through ``self.w1``, so anything
    that skips PEFT's ``_activate_lora`` for these wrappers has to be gated on
    FP8 or the expert adapter silently trains nothing.
    """

    class _FusedLoraProxy(nn.Module):
        """PEFT's parametrization folds the delta into the weight it returns."""

        def __init__(self, delta: torch.Tensor):
            super().__init__()
            self.delta_weight = delta

        def forward(self, weight: torch.Tensor) -> torch.Tensor:
            return weight + self.delta_weight

    torch.manual_seed(70)
    dim, hidden, n = 128, 256, 2
    experts = GroupedExperts(dim=dim, hidden_dim=hidden, num_experts=n, use_grouped_mm=False)
    with torch.no_grad():
        for w in (experts.w1, experts.w2, experts.w3):
            w.copy_(torch.randn_like(w))
    x = torch.randn(4, dim)
    counts = torch.tensor([2, 2], dtype=torch.int32)
    out_base = experts._forward_deepep(x, counts)

    delta = torch.randn_like(experts.w1) * 0.05
    nn.utils.parametrize.register_parametrization(experts, "w1", _FusedLoraProxy(delta))
    out_lora = experts._forward_deepep(x, counts)

    assert torch.isfinite(out_lora).all()
    assert not torch.allclose(out_lora, out_base, atol=1e-3)


def test_deepgemm_token_pad_unpad_roundtrip_and_tail():
    """Pad/unpad is the grouped CUDA path; CPU grouped_mm skips it."""
    torch.manual_seed(3)
    x = torch.arange(4 * 8, dtype=torch.float32).view(4, 8)
    counts = torch.tensor([1, 3], dtype=torch.int32)
    xp, m_indices, got_counts, n_tail, padded_counts = fp8_mod._pad_tokens_for_deepgemm(x, counts, align=2)
    assert got_counts == [1, 3]
    assert n_tail == 0
    assert padded_counts == [2, 4]
    assert xp.shape == (6, 8)
    assert m_indices.tolist() == [0, 0, 1, 1, 1, 1]
    torch.testing.assert_close(xp[0], x[0])
    torch.testing.assert_close(xp[1], torch.zeros_like(x[0]))
    torch.testing.assert_close(xp[2:5], x[1:4])
    torch.testing.assert_close(xp[5], torch.zeros_like(x[0]))

    y = torch.arange(6 * 5, dtype=torch.float32).view(6, 5)
    out = fp8_mod._unpad_tokens_from_deepgemm(y, got_counts, padded_counts, n_tail=0)
    assert out.shape == (4, 5)
    torch.testing.assert_close(out[:1], y[:1])
    torch.testing.assert_close(out[1:], y[2:5])

    x_tail = torch.cat([x, torch.ones(2, 8)])
    xp_t, _, _, n_tail, padded_t = fp8_mod._pad_tokens_for_deepgemm(x_tail, counts, align=2)
    assert n_tail == 2
    assert xp_t.shape[0] == 6
    y_t = fp8_mod._unpad_tokens_from_deepgemm(y, got_counts, padded_t, n_tail=2)
    assert y_t.shape == (6, 5)
    torch.testing.assert_close(y_t[-2:], torch.zeros(2, 5))


def test_packed_expert_linear_uses_b_out_dim_and_pads_tail():
    torch.manual_seed(5)
    dim, hidden, n, r, n_pad = 8, 16, 2, 2, 3
    counts = torch.tensor([1, 3], dtype=torch.int32)
    x = torch.randn(sum(counts.tolist()) + n_pad, dim)
    ab = (
        torch.randn(r * n, dim),
        torch.randn(hidden, r * n),
        r,
        1.0,
    )
    out = _packed_expert_linear(x, None, counts, ab)
    assert out.shape == (x.shape[0], hidden)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out[:4], torch.zeros(4, hidden), atol=1e-6)
    torch.testing.assert_close(out[4:], torch.zeros(n_pad, hidden))


def test_fp8_expert_lora_unfused_ab_is_applied():
    torch.manual_seed(11)
    dim, hidden, n, r = 128, 256, 2, 4
    experts = GroupedExperts(dim=dim, hidden_dim=hidden, num_experts=n, use_grouped_mm=False, fp8_block_size=128)
    w1 = torch.randn(n, hidden, dim, dtype=torch.bfloat16)
    w2 = torch.randn(n, dim, hidden, dtype=torch.bfloat16)
    w3 = torch.randn(n, hidden, dim, dtype=torch.bfloat16)
    with torch.no_grad():
        for src, dst, scale_dst in (
            (w1, experts.w1, experts.w1_scale_inv),
            (w2, experts.w2, experts.w2_scale_inv),
            (w3, experts.w3, experts.w3_scale_inv),
        ):
            qs, ss = zip(*(quantize_to_fp8_blockwise(src[i]) for i in range(n)))
            dst.copy_(torch.stack(qs))
            scale_dst.copy_(torch.stack(ss))
    x = torch.randn(4, dim, dtype=torch.bfloat16)
    counts = torch.tensor([2, 2], dtype=torch.int32)
    out_base = experts._forward_deepep(x, counts)
    experts._ap_lora_ab = {
        "w1": (
            torch.randn(r * n, dim, dtype=torch.bfloat16),
            torch.randn(hidden, r * n, dtype=torch.bfloat16),
            r,
            2.0,
        )
    }
    out_ab = experts._forward_deepep(x, counts)
    assert torch.isfinite(out_ab.float()).all()
    assert not torch.allclose(out_ab.float(), out_base.float(), atol=1e-3)


def test_fp8_expert_lora_unfused_ab_grouped_mm_is_applied():
    """Production GLM path is use_grouped_mm=True plus unfused A/B, not the for-loop."""
    torch.manual_seed(13)
    dim, hidden, n, r = 128, 256, 2, 4
    experts = GroupedExperts(dim=dim, hidden_dim=hidden, num_experts=n, use_grouped_mm=True, fp8_block_size=128)
    w1 = torch.randn(n, hidden, dim, dtype=torch.bfloat16)
    w2 = torch.randn(n, dim, hidden, dtype=torch.bfloat16)
    w3 = torch.randn(n, hidden, dim, dtype=torch.bfloat16)
    with torch.no_grad():
        for src, dst, scale_dst in (
            (w1, experts.w1, experts.w1_scale_inv),
            (w2, experts.w2, experts.w2_scale_inv),
            (w3, experts.w3, experts.w3_scale_inv),
        ):
            qs, ss = zip(*(quantize_to_fp8_blockwise(src[i]) for i in range(n)))
            dst.copy_(torch.stack(qs))
            scale_dst.copy_(torch.stack(ss))
    x = torch.randn(4, dim, dtype=torch.bfloat16)
    counts = torch.tensor([1, 3], dtype=torch.int32)
    out_base = experts._forward_deepep(x, counts)
    experts._ap_lora_ab = {
        "w1": (
            torch.randn(r * n, dim, dtype=torch.bfloat16),
            torch.randn(hidden, r * n, dtype=torch.bfloat16),
            r,
            2.0,
        )
    }
    out_ab = experts._forward_deepep(x, counts)
    assert out_ab.shape == (4, dim)
    assert torch.isfinite(out_ab.float()).all()
    assert not torch.allclose(out_ab.float(), out_base.float(), atol=1e-3)


def test_missing_deep_gemm_raises_instead_of_dequantizing(monkeypatch):
    """Dequantizing would restore the BF16 weight this path exists to avoid."""
    monkeypatch.setattr(fp8_mod, "_deep_gemm", lambda: None)

    with pytest.raises(RuntimeError, match="requires DeepGEMM"):
        _require_deep_gemm()


def test_recast_fp8_state_raises_instead_of_being_repaired():
    """A bf16 weight or scale here means the DeepSpeed cast guard failed."""
    codes = torch.empty(4, 4, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, 1, dtype=torch.float32)
    _require_fp8_state(codes, scale, "expert")

    with pytest.raises(TypeError, match="float8_e4m3fn"):
        _require_fp8_state(torch.empty(4, 4, dtype=torch.bfloat16), scale, "expert")
    with pytest.raises(TypeError, match="float32"):
        _require_fp8_state(codes, scale.to(torch.bfloat16), "expert")
