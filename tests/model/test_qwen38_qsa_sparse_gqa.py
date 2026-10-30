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
"""The fused QSA sparse GQA kernels against a float64 reference and the chunked PyTorch path."""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused QSA kernels run on CUDA")


def _inputs(S, S_KV, R, HQ=24, HKV=2, D=256):
    torch.manual_seed(0)
    q = (torch.randn(1, S, HQ, D, device="cuda") * 0.5).bfloat16()
    k = (torch.randn(1, S_KV, HKV, D, device="cuda") * 0.5).bfloat16()
    v = torch.randn(1, S_KV, HKV, D, device="cuda").bfloat16()
    positions = S_KV - S + torch.arange(S, device="cuda")
    idx = (torch.rand(1, S, R, device="cuda") * (positions[None, :, None] + 1)).long()
    idx = torch.where(torch.rand(1, S, R, device="cuda") < 0.05, -1, idx).to(torch.int32)
    idx[:, 0] = -1  # a query with no selected key gets a zero output
    idx[:, 1, 3:6] = idx[:, 1, 2:3]  # a repeated slot counts once per slot, as in the PyTorch path
    return q, k, v, idx


def _reference64(q, k, v, idx, scale):
    B, S, HQ, D = q.shape
    HKV = k.shape[2]
    valid = (idx >= 0) & (idx < k.shape[1])
    safe = idx.long().clamp(0, k.shape[1] - 1)
    b = torch.arange(B, device=q.device)[:, None, None]
    scores = torch.einsum("bshgd,bsrhd->bshgr", q.view(B, S, HKV, HQ // HKV, D), k[b, safe]) * scale
    probabilities = torch.softmax(scores.masked_fill(~valid[:, :, None, None], -torch.inf), -1).nan_to_num(0.0)
    return torch.einsum("bshgr,bsrhd->bshgd", probabilities, v[b, safe]).reshape(B, S, HQ, D)


def _run(fn, q, k, v, idx, scale, dtype=None):
    q, k, v = (t.detach().clone().to(dtype or t.dtype).requires_grad_(True) for t in (q, k, v))
    out = fn(q, k, v, idx, scale)
    torch.manual_seed(1)
    out.backward(torch.randn(out.shape, device=out.device).to(out.dtype))
    return out.detach(), q.grad, k.grad, v.grad


@pytest.mark.parametrize("S,S_KV,R", [(64, 64, 32), (300, 1200, 131), (256, 4096, 2051)])
def test_fused_sparse_gqa_matches_float64_at_least_as_well_as_the_pytorch_path(S, S_KV, R):
    from arctic_platform.model.implementations.qwen38.qsa_flex import _SparseGQAAttention
    from arctic_platform.model.implementations.qwen38.qsa_sparse_gqa import fused_sparse_gqa_attention

    q, k, v, idx = _inputs(S, S_KV, R)
    scale = 1 / math.sqrt(q.shape[-1])
    exact = _run(_reference64, q, k, v, idx, scale, torch.float64)
    fused = _run(fused_sparse_gqa_attention, q, k, v, idx, scale)
    # The fused kernel rounds the unnormalized probabilities to BF16 before P·V, as vLLM's QSA kernel does; the
    # fp32_parity reference keeps them in FP32, so it is the BF16-probability reference this is measured against.
    chunked = _run(lambda *a: _SparseGQAAttention.apply(*a, 128, False), q, k, v, idx, scale)

    assert (fused[0][0, 0] == 0).all()
    for name, f, c, e in zip(("out", "dq", "dk", "dv"), fused, chunked, exact):
        fused_rel = ((f.double() - e).norm() / e.norm()).item()
        chunked_rel = ((c.double() - e).norm() / e.norm()).item()
        assert fused_rel < 1e-2, (name, fused_rel)  # BF16 outputs
        assert fused_rel <= 1.25 * chunked_rel, (name, fused_rel, chunked_rel)
