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
"""DeepSeek-V4.1's indexer scores ``sum_h w[q, h] * relu(q[q, h] . k[e])`` in one pass over the keys.

The reference contracts in FP32 through a ``[queries, heads, entries]`` intermediate (heads times the scores'
size, written and re-read by the ReLU, the weighting and the head sum). Here each tile of keys is loaded once and
the heads are reduced in registers. The queries and weights are BF16, so their products are exact on BF16 tensor
cores; the keys come out of an FP32-weighted norm, so they are split into a BF16 high part and a BF16 residual and
contracted twice, which keeps them to 2^-17 of their value instead of rounding them to BF16.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=w, num_stages=s)
        for m, n, w, s in ((64, 128, 4, 2), (128, 128, 8, 2), (64, 256, 8, 2), (128, 64, 4, 3))
    ],
    key=["H", "D", "SPLIT"],
)
@triton.jit
def _index_scores_kernel(
    q_ptr,
    k_hi_ptr,
    k_lo_ptr,
    w_ptr,
    out_ptr,
    Q,
    E,
    stride_qm,
    stride_qh,
    stride_wm,
    H: tl.constexpr,
    D: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < Q
    mask_n = offs_n < E
    k_offsets = offs_n[:, None].to(tl.int64) * D + offs_d[None, :]
    k_hi = tl.load(k_hi_ptr + k_offsets, mask=mask_n[:, None], other=0.0)
    if SPLIT:
        k_lo = tl.load(k_lo_ptr + k_offsets, mask=mask_n[:, None], other=0.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for h in range(H):
        q = tl.load(
            q_ptr + offs_m[:, None].to(tl.int64) * stride_qm + h * stride_qh + offs_d[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        w = tl.load(w_ptr + offs_m.to(tl.int64) * stride_wm + h, mask=mask_m, other=0.0).to(tl.float32)
        s = tl.dot(q, tl.trans(k_hi))
        if SPLIT:
            s = tl.dot(q, tl.trans(k_lo), s)
        acc += tl.maximum(s, 0.0) * w[:, None]
    out = out_ptr + offs_m[:, None].to(tl.int64) * E + offs_n[None, :]
    tl.store(out, acc, mask=mask_m[:, None] & mask_n[None, :])


def split_keys(keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``keys [entries, dim]`` as a BF16 high part and, unless they are BF16 already, the BF16 residual."""
    if keys.dtype == torch.bfloat16:
        return keys.contiguous(), None
    high = keys.bfloat16()
    return high.contiguous(), (keys.float() - high.float()).bfloat16().contiguous()


def index_scores_applies(query: torch.Tensor, weights: torch.Tensor) -> bool:
    return query.is_cuda and query.dtype == torch.bfloat16 and weights.dtype == torch.bfloat16


def index_scores(
    query: torch.Tensor, keys: tuple[torch.Tensor, torch.Tensor | None], weights: torch.Tensor
) -> torch.Tensor:
    """``query [queries, heads, dim]`` BF16, ``keys`` from ``split_keys``, ``weights [queries, heads]`` BF16 ->
    FP32 ``[queries, entries]``."""
    k_hi, k_lo = keys
    Q, H, D = query.shape
    E = k_hi.shape[0]
    if query.stride(-1) != 1:
        query = query.contiguous()
    if weights.stride(-1) != 1:
        weights = weights.contiguous()
    out = torch.empty(Q, E, dtype=torch.float32, device=query.device)
    if Q == 0 or E == 0:
        return out
    grid = lambda meta: (triton.cdiv(Q, meta["BLOCK_M"]), triton.cdiv(E, meta["BLOCK_N"]))
    _index_scores_kernel[grid](
        query,
        k_hi,
        k_hi if k_lo is None else k_lo,
        weights,
        out,
        Q,
        E,
        query.stride(0),
        query.stride(1),
        weights.stride(0),
        H=H,
        D=D,
        SPLIT=k_lo is not None,
    )
    return out
