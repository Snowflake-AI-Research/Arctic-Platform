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
"""The Triton kernels of ``FusedAddRMSNorm`` / ``FusedAddRMSNormFp32Residual`` against their PyTorch formulas (GPU)."""

from __future__ import annotations

import pytest
import torch

from arctic_platform.model.implementations.gpu import vllm_rounding
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close


@require_torch_gpu
@pytest.mark.parametrize("fp32_residual", [False, True])
@pytest.mark.parametrize("hidden", [2048, 2560, 5120])
def test_add_norm_kernels_match_formula(monkeypatch, fp32_residual, hidden):
    set_seed(0)
    rows = 4099
    x = torch.randn(1, rows, hidden, device="cuda").bfloat16().requires_grad_(True)
    residual_dtype = torch.float32 if fp32_residual else torch.bfloat16
    residual = (torch.randn(1, rows, hidden, device="cuda") * 3).to(residual_dtype).requires_grad_(True)
    weight = (torch.randn(hidden, device="cuda") * 0.2).bfloat16().requires_grad_(True)
    function = vllm_rounding.FusedAddRMSNormFp32Residual if fp32_residual else vllm_rounding.FusedAddRMSNorm
    grad_out = torch.randn(1, rows, hidden, device="cuda").bfloat16()

    def run(kernels: bool):
        monkeypatch.setattr(vllm_rounding, "_uses_add_norm_kernels", lambda *args: kernels)
        for t in (x, residual, weight):
            t.grad = None
        out, summed = function.apply(x, residual, weight, 1e-6)
        grad_sum = torch.full_like(summed, 0.25)
        torch.autograd.backward([out, summed], [grad_out, grad_sum])
        return out.detach(), summed.detach(), x.grad.clone(), residual.grad.clone(), weight.grad.clone()

    expected = run(False)
    got = run(True)
    assert got[0].dtype == expected[0].dtype and got[1].dtype == expected[1].dtype
    # Same fp32 math, row sums reduced in another order: at most one bf16 ulp apart in the outputs.
    torch_assert_close(got[0].float(), expected[0].float(), rtol=2**-7, atol=1e-5)
    torch_assert_close(got[1].float(), expected[1].float(), rtol=0, atol=0)
    torch_assert_close(got[2].float(), expected[2].float(), rtol=2**-7, atol=1e-4)
    torch_assert_close(got[3].float(), expected[3].float(), rtol=2**-7, atol=1e-4)
    # The weight gradient sums 4099 rows in fp32 in another order before its bf16 rounding.
    torch_assert_close(got[4].float(), expected[4].float(), rtol=2**-6, atol=1e-2)
