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

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused SwiGLU kernel runs on CUDA")


def _reference(gate, up, limit):
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


@pytest.mark.parametrize("limit", [None, 10.0, 1.5])
def test_fused_swiglu_matches_reference_forward_and_backward(limit):
    from arctic_platform.model.implementations.kernels.fused_swiglu import fused_swiglu

    torch.manual_seed(0)
    gate = (torch.randn(3001, 512, device="cuda") * 4).double()
    up = (torch.randn(3001, 512, device="cuda") * 4).double()
    if limit is not None:
        # Values exactly at the bounds take the gradient, as torch.clamp's do.
        gate[0, :4] = limit
        up[0, :4] = torch.tensor([limit, -limit, limit, -limit], device="cuda")
    grad = torch.randn(3001, 512, device="cuda").double()
    # The reference sees the kernel's BF16 inputs, so values rounded onto a clamp bound agree on its mask.
    gate, up = gate.bfloat16().double(), up.bfloat16().double()
    g64, u64 = gate.clone().requires_grad_(), up.clone().requires_grad_()
    _reference(g64, u64, limit).backward(grad)

    g, u = gate.bfloat16().requires_grad_(), up.bfloat16().requires_grad_()
    out = fused_swiglu(g, u, limit)
    out.backward(grad.bfloat16())
    expected = _reference(gate.bfloat16().double(), up.bfloat16().double(), limit)
    torch.testing.assert_close(out.double(), expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(g.grad.double(), g64.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(u.grad.double(), u64.grad, rtol=2e-2, atol=2e-2)
    if limit is not None:
        assert (g.grad[0, :4] != 0).all() and (u.grad[0, :4] != 0).all()


def test_grouped_expert_lora_rows_matches_padded_lora():
    from arctic_platform.model.implementations.fp8 import grouped_expert_lora
    from arctic_platform.model.implementations.fp8 import grouped_expert_lora_rows

    torch.manual_seed(0)
    # Rank 8: CUDA grouped GEMMs need 16-byte strides on the rank-wide intermediate.
    experts, rank, din, dout = 6, 8, 64, 48
    weight_a = torch.randn(experts * rank, din, device="cuda").bfloat16()
    weight_b = torch.randn(dout, experts * rank, device="cuda").bfloat16()
    ab = (weight_a, weight_b, rank, 0.5)
    counts = torch.tensor([5, 0, 7, 3, 0, 9], device="cuda")
    offsets = torch.cumsum(counts, 0, dtype=torch.int32)
    rows = int(offsets[-1])
    x = torch.randn(rows + 4, din, device="cuda").bfloat16()
    padded = grouped_expert_lora(x, ab, offsets, rows)
    torch.testing.assert_close(grouped_expert_lora_rows(x, ab, offsets, rows), padded[:rows], rtol=2e-2, atol=2e-2)
    assert (padded[rows:] == 0).all()
