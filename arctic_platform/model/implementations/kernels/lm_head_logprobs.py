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
"""The per-tile FP32 work of the chunked LM head (``moe/layers/lm_head.py``) in one Triton pass per tile.

The chunked LM head computes per-token log-probabilities and entropies tile by tile: one ``[tokens, 8192]`` logits
GEMM, then, in PyTorch, an FP32 copy of the tile, the temperature, an online log-sum-exp update (max, ``isfinite``
masks, ``exp``, two ``where``s, two row sums) and a host-synchronizing ``nonzero`` for the target column; and in the
backward the same GEMM again, an FP32 copy, ``exp``, gathers, scatters and two BF16 casts. At 8192 x 8192 each of
those passes moves 256 MiB, several times the GEMM's own traffic. These kernels read the GEMM's BF16 tile once (the
forward reads it twice from L2) and write the BF16 gradient tile once, with the same FP32 arithmetic per element:

* ``lse_tile_update_``: ``m, s, t`` (running max, sum of exp, exp-weighted logit sum, all over the non-target
  columns) and the target logit of each row whose label falls in the tile.
* ``logprob_tile_grad``: the gradient of the per-token log-probability w.r.t. the tile's logits, times the
  temperature: ``-g * softmax`` off the target, ``g * sigmoid(others_minus_target)`` at it.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 2048


@triton.jit
def _lse_tile_update_kernel(
    logits_ptr,
    stride_row,
    labels_ptr,
    inv_t_ptr,
    m_ptr,
    s_ptr,
    t_ptr,
    target_ptr,
    n_cols,
    vocab_start,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride_row
    label = tl.load(labels_ptr + row).to(tl.int64) - vocab_start
    inv_t = tl.load(inv_t_ptr + row).to(tl.float32)
    # Pass 1: the tile's max over every column, as torch.amax.
    tile_max = tl.full([BLOCK], float("-inf"), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(base + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32) * inv_t
        tile_max = tl.maximum(tile_max, x)
    m_old = tl.load(m_ptr + row)
    m_new = tl.maximum(m_old, tl.max(tile_max, axis=0))
    exp_old = tl.where(tl.abs(m_old) < float("inf"), tl.exp(m_old - m_new), 0.0)
    # Pass 2: finite non-target columns into s and t; the row's target logit.
    s_acc = tl.zeros([BLOCK], dtype=tl.float32)
    t_acc = tl.zeros([BLOCK], dtype=tl.float32)
    target = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        in_tile = cols < n_cols
        x = tl.load(base + cols, mask=in_tile, other=0.0).to(tl.float32) * inv_t
        finite = in_tile & (tl.abs(x) < float("inf"))
        is_target = cols == label
        e = tl.where(finite & (~is_target), tl.exp(x - m_new), 0.0)
        s_acc += e
        t_acc += e * tl.where(finite, x, 0.0)
        target += tl.where(is_target & in_tile, x, 0.0)
    s_new = tl.load(s_ptr + row) * exp_old + tl.sum(s_acc, axis=0)
    t_new = tl.load(t_ptr + row) * exp_old + tl.sum(t_acc, axis=0)
    tl.store(m_ptr + row, m_new)
    tl.store(s_ptr + row, s_new)
    tl.store(t_ptr + row, t_new)
    if (label >= 0) & (label < n_cols):
        tl.store(target_ptr + row, tl.sum(target, axis=0))


@triton.jit
def _logprob_tile_grad_kernel(
    logits_ptr,
    stride_row,
    grad_ptr,
    stride_grad,
    labels_ptr,
    inv_t_ptr,
    logz_ptr,
    omt_ptr,
    g_ptr,
    n_cols,
    vocab_start,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    label = tl.load(labels_ptr + row).to(tl.int64) - vocab_start
    inv_t = tl.load(inv_t_ptr + row).to(tl.float32)
    logz = tl.load(logz_ptr + row)
    g = tl.load(g_ptr + row).to(tl.float32)
    target_grad = g * tl.sigmoid(tl.load(omt_ptr + row))
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        in_tile = cols < n_cols
        x = tl.load(logits_ptr + row * stride_row + cols, mask=in_tile, other=0.0).to(tl.float32) * inv_t
        grad = tl.where(cols == label, target_grad, -g * tl.exp(x - logz)) * inv_t
        tl.store(grad_ptr + row * stride_grad + cols, grad.to(grad_ptr.dtype.element_ty), mask=in_tile)


def lse_tile_update_(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_start: int,
    inv_temperature: torch.Tensor,
    m: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """Fold one ``[rows, cols]`` logits tile (vocab ids ``vocab_start + col``) into the rows' running ``m, s, t``
    (FP32, in place) and write ``target`` for the rows whose label is in the tile."""
    rows, cols = logits.shape
    assert logits.stride(1) == 1
    assert all(x.is_contiguous() for x in (m, s, t, target)), "the running statistics are written in place"
    if rows == 0:
        return
    _lse_tile_update_kernel[(rows,)](
        logits,
        logits.stride(0),
        labels.contiguous(),
        inv_temperature.contiguous(),
        m,
        s,
        t,
        target,
        cols,
        vocab_start,
        BLOCK=_BLOCK,
        num_warps=8,
    )


def logprob_tile_grad(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_start: int,
    inv_temperature: torch.Tensor,
    logz: torch.Tensor,
    others_minus_target: torch.Tensor,
    grad_logprobs: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """d(sum_r g_r * logprob_r) / d(logits tile), times the temperature, as ``out_dtype``."""
    rows, cols = logits.shape
    assert logits.stride(1) == 1
    grad = torch.empty((rows, cols), dtype=out_dtype, device=logits.device)
    if rows == 0:
        return grad
    _logprob_tile_grad_kernel[(rows,)](
        logits,
        logits.stride(0),
        grad,
        grad.stride(0),
        labels.contiguous(),
        inv_temperature.contiguous(),
        logz.contiguous(),
        others_minus_target.contiguous(),
        grad_logprobs.contiguous(),
        cols,
        vocab_start,
        BLOCK=_BLOCK,
        num_warps=8,
    )
    return grad
