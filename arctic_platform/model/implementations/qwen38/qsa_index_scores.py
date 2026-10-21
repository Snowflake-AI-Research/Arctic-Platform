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
"""Qwen3.8 QSA index scores in one Triton pass, and the top-k over them.

``select_qsa_token_ids`` scores every visible compressed block for a query as ``sum_h relu(q_h . k) / sqrt(d)``. The
PyTorch path built the ``[rows, heads, blocks]`` FP32 product with an FP32 (SIMT) GEMM, then ReLU, a head sum and a
visibility mask as separate passes. Here one kernel computes each ``[64 rows, 128 blocks]`` tile from BF16 operands
with FP32 accumulation (the products of BF16 values are exact in FP32 either way), sums the heads in registers and
writes the masked FP32 scores. FlashInfer's deterministic radix top-k then selects the blocks, ties going to the
smaller index.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

try:
    from flashinfer import top_k as _flashinfer_top_k
    from flashinfer.topk import TopKTieBreak
except ImportError:  # pragma: no cover - FlashInfer is optional
    _flashinfer_top_k = None


@triton.jit
def _qsa_index_scores_kernel(q_ptr, k_ptr, visible_ptr, out_ptr, R, P, sqrt_d, stride_qr, stride_qh, stride_kp,
                             stride_or, H: tl.constexpr, D: tl.constexpr, BLOCK_R: tl.constexpr,
                             BLOCK_P: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
    dims = tl.arange(0, D)
    row_mask = rows < R
    col_mask = cols < P
    k = tl.load(k_ptr + cols[:, None].to(tl.int64) * stride_kp + dims[None, :], mask=col_mask[:, None], other=0.0)
    acc = tl.zeros([BLOCK_R, BLOCK_P], tl.float32)
    for h in tl.static_range(H):
        q = tl.load(q_ptr + rows[:, None].to(tl.int64) * stride_qr + h * stride_qh + dims[None, :],
                    mask=row_mask[:, None], other=0.0)
        acc += tl.maximum(tl.dot(q, tl.trans(k)), 0.0)
    acc = acc / sqrt_d
    visible = tl.load(visible_ptr + rows, mask=row_mask, other=0)
    acc = tl.where(cols[None, :] < visible[:, None], acc, -float("inf"))
    tl.store(out_ptr + rows[:, None].to(tl.int64) * stride_or + cols[None, :], acc,
             mask=row_mask[:, None] & col_mask[None, :])


def fused_index_scores_supported(queries: torch.Tensor) -> bool:
    head_dim = queries.shape[-1]
    return queries.is_cuda and head_dim >= 16 and head_dim & (head_dim - 1) == 0


def qsa_index_scores(queries: torch.Tensor, keys: torch.Tensor, visible_blocks: torch.Tensor) -> torch.Tensor:
    """``[rows, blocks]`` FP32 scores ``sum_h relu(queries[:, h] . keys) / sqrt(d)``, ``-inf`` at blocks a row does not
    see (``block >= visible_blocks[row]``). ``queries`` ``[rows, heads, d]``, ``keys`` ``[blocks, d]``."""
    rows, heads, head_dim = queries.shape
    blocks = keys.shape[0]
    queries = queries.to(torch.bfloat16)
    keys = keys.to(torch.bfloat16).contiguous()
    if queries.stride(-1) != 1:
        queries = queries.contiguous()
    out = torch.empty(rows, blocks, dtype=torch.float32, device=queries.device)
    block_r, block_p = 64, 128
    grid = (triton.cdiv(rows, block_r), triton.cdiv(blocks, block_p))
    _qsa_index_scores_kernel[grid](
        queries, keys, visible_blocks.to(torch.int32).contiguous(), out, rows, blocks, math.sqrt(head_dim),
        queries.stride(0), queries.stride(1), keys.stride(0), out.stride(0),
        H=heads, D=head_dim, BLOCK_R=block_r, BLOCK_P=block_p, num_warps=4, num_stages=2,
    )
    return out


def top_block_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the ``k`` largest scores per row (any order), ties to the smaller index."""
    if _flashinfer_top_k is None or not scores.is_cuda:
        return torch.topk(scores, k=k, dim=-1).indices
    return _flashinfer_top_k(scores, k, sorted=False, deterministic=True, tie_break=TopKTieBreak.SMALL)[1].long()
