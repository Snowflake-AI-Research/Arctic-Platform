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
"""Packed MXFP4 routed experts (DeepSeek-V4) with LoRA against the decoded BF16 experts."""

import pytest
import torch

from arctic_platform.model.implementations.kernels.packed4_dequant import dequantize_mxfp4_reference

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the decode kernels run on CUDA")


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
