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
"""Fused sparse GQA attention for Qwen3.8-Flash-Next QSA (Triton).

Every query token attends to its own selected key positions (``selected_token_ids``, -1 = unused slot), shared by
all heads. One program per (token, KV head) runs that KV head's query heads (padded to 16) over the selected rows in
blocks of 64: the K and V rows are gathered into shared memory once for all of them, softmax runs online in FP32,
and the log-sum-exp is kept. Backward recomputes the probabilities from it, writes dQ and scatters dK/dV with FP32
atomics. Against an FP64 reference the outputs and dQ match the chunked PyTorch path, and dK/dV are 5x closer
(it accumulates them in BF16).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    idx_ptr,
    o_ptr,
    lse_ptr,
    S,
    S_KV,
    R,
    scale,
    sq_b,
    sq_s,
    sq_h,
    sk_b,
    sk_s,
    sk_h,
    si_b,
    si_s,
    so_b,
    so_s,
    so_h,
    sl_b,
    sl_s,
    G: tl.constexpr,
    GP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tok = tl.program_id(0)
    kvh = tl.program_id(1)
    b = tl.program_id(2)
    heads = tl.arange(0, GP)
    dims = tl.arange(0, D)
    hmask = heads < G
    q_base = q_ptr + b.to(tl.int64) * sq_b + tok.to(tl.int64) * sq_s + (kvh * G) * sq_h
    q = tl.load(q_base + heads[:, None] * sq_h + dims[None, :], mask=hmask[:, None], other=0.0)
    m = tl.full([GP], -float("inf"), tl.float32)
    denom = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, D], tl.float32)
    kv_base = b.to(tl.int64) * sk_b + kvh * sk_h
    idx_base = idx_ptr + b.to(tl.int64) * si_b + tok.to(tl.int64) * si_s
    for start in range(0, R, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        ids = tl.load(idx_base + offs, mask=offs < R, other=-1)
        valid = (ids >= 0) & (ids < S_KV)
        rows = tl.where(valid, ids, 0).to(tl.int64)
        k = tl.load(k_ptr + kv_base + rows[:, None] * sk_s + dims[None, :], mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(valid[None, :], s, -float("inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m - m_safe)
        p = tl.exp(s - m_safe[:, None])
        denom = denom * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + kv_base + rows[:, None] * sk_s + dims[None, :], mask=valid[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new
    out = tl.where(denom[:, None] > 0, acc / tl.where(denom > 0, denom, 1.0)[:, None], 0.0)
    o_base = o_ptr + b.to(tl.int64) * so_b + tok.to(tl.int64) * so_s + (kvh * G) * so_h
    tl.store(o_base + heads[:, None] * so_h + dims[None, :], out.to(o_ptr.dtype.element_ty), mask=hmask[:, None])
    lse = tl.where(denom > 0, m + tl.log(tl.where(denom > 0, denom, 1.0)), float("inf"))
    tl.store(lse_ptr + b.to(tl.int64) * sl_b + tok.to(tl.int64) * sl_s + kvh * G + heads, lse, mask=hmask)


@triton.jit
def _qsa_bwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    idx_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    S,
    S_KV,
    R,
    scale,
    sq_b,
    sq_s,
    sq_h,
    sk_b,
    sk_s,
    sk_h,
    si_b,
    si_s,
    sl_b,
    sl_s,
    sdk_b,
    sdk_s,
    sdk_h,
    G: tl.constexpr,
    GP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tok = tl.program_id(0)
    kvh = tl.program_id(1)
    b = tl.program_id(2)
    heads = tl.arange(0, GP)
    dims = tl.arange(0, D)
    hmask = heads < G
    # q, do and dq share one layout ([B, S, Hq, D] contiguous).
    q_off = b.to(tl.int64) * sq_b + tok.to(tl.int64) * sq_s + (kvh * G) * sq_h + heads[:, None] * sq_h + dims[None, :]
    q = tl.load(q_ptr + q_off, mask=hmask[:, None], other=0.0)
    do = tl.load(do_ptr + q_off, mask=hmask[:, None], other=0.0)
    l_off = b.to(tl.int64) * sl_b + tok.to(tl.int64) * sl_s + kvh * G + heads
    lse = tl.load(lse_ptr + l_off, mask=hmask, other=float("inf"))
    delta = tl.load(delta_ptr + l_off, mask=hmask, other=0.0)
    dq = tl.zeros([GP, D], tl.float32)
    kv_base = b.to(tl.int64) * sk_b + kvh * sk_h
    dkv_base = b.to(tl.int64) * sdk_b + kvh * sdk_h
    idx_base = idx_ptr + b.to(tl.int64) * si_b + tok.to(tl.int64) * si_s
    for start in range(0, R, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        ids = tl.load(idx_base + offs, mask=offs < R, other=-1)
        valid = (ids >= 0) & (ids < S_KV)
        rows = tl.where(valid, ids, 0).to(tl.int64)
        k = tl.load(k_ptr + kv_base + rows[:, None] * sk_s + dims[None, :], mask=valid[:, None], other=0.0)
        v = tl.load(v_ptr + kv_base + rows[:, None] * sk_s + dims[None, :], mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        p = tl.exp(s - lse[:, None])
        p = tl.where(valid[None, :] & hmask[:, None], p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        # dQ sums ~2K slots per row: split ds into BF16 high and low parts so it keeps ~16 mantissa bits.
        ds_hi = ds.to(k.dtype)
        dq += tl.dot(ds_hi, k) + tl.dot((ds - ds_hi.to(tl.float32)).to(k.dtype), k)
        dv_tile = tl.dot(tl.trans(p.to(do.dtype)), do)
        dk_tile = tl.dot(tl.trans(ds.to(q.dtype)), q) * scale
        dst = dkv_base + rows[:, None] * sdk_s + dims[None, :]
        tl.atomic_add(dv_ptr + dst, dv_tile, mask=valid[:, None], sem="relaxed")
        tl.atomic_add(dk_ptr + dst, dk_tile, mask=valid[:, None], sem="relaxed")
    tl.store(dq_ptr + q_off, (dq * scale).to(dq_ptr.dtype.element_ty), mask=hmask[:, None])


_BLOCK_N = 64


def _qsa_fwd(q, k, v, idx, scale):
    B, S, HQ, D = q.shape
    HKV = k.shape[2]
    G = HQ // HKV
    o = torch.empty_like(q)
    lse = torch.empty(B, S, HQ, device=q.device, dtype=torch.float32)
    grid = (S, HKV, B)
    _qsa_fwd_kernel[grid](
        q,
        k,
        v,
        idx,
        o,
        lse,
        S,
        k.shape[1],
        idx.shape[2],
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        idx.stride(0),
        idx.stride(1),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        lse.stride(0),
        lse.stride(1),
        G=G,
        GP=max(16, triton.next_power_of_2(G)),
        D=D,
        BLOCK_N=_BLOCK_N,
        num_warps=8,
        num_stages=2,
    )
    return o, lse


def _qsa_bwd(q, k, v, idx, o, lse, do, scale):
    B, S, HQ, D = q.shape
    HKV = k.shape[2]
    G = HQ // HKV
    delta = (o.float() * do.float()).sum(-1)
    dq = torch.empty_like(q)
    dk = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
    dv = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
    grid = (S, HKV, B)
    _qsa_bwd_kernel[grid](
        q,
        k,
        v,
        idx,
        do.contiguous(),
        lse,
        delta,
        dq,
        dk,
        dv,
        S,
        k.shape[1],
        idx.shape[2],
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        idx.stride(0),
        idx.stride(1),
        lse.stride(0),
        lse.stride(1),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        G=G,
        GP=max(16, triton.next_power_of_2(G)),
        D=D,
        BLOCK_N=_BLOCK_N,
        num_warps=8,
        num_stages=1,
    )
    return dq, dk.to(k.dtype), dv.to(v.dtype)


class _QSASparseGQA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, idx, scale):
        q, k, v, idx = q.contiguous(), k.contiguous(), v.contiguous(), idx.contiguous()
        o, lse = _qsa_fwd(q, k, v, idx, scale)
        ctx.save_for_backward(q, k, v, idx, o, lse)
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, idx, o, lse = ctx.saved_tensors
        dq, dk, dv = _qsa_bwd(q, k, v, idx, o, lse, do, ctx.scale)
        return dq, dk, dv, None, None


def fused_sparse_gqa_supported(query: torch.Tensor, key: torch.Tensor) -> bool:
    """Shapes the kernels tile: power-of-two head dim of at least 16 and up to 64 query heads per KV head."""
    head_dim = query.shape[-1]
    groups = query.shape[2] // key.shape[2]
    return query.is_cuda and head_dim >= 16 and head_dim & (head_dim - 1) == 0 and groups <= 64


def fused_sparse_gqa_attention(query, key, value, selected_token_ids, softmax_scale: float) -> torch.Tensor:
    """``[B, S, Hq, D]`` queries over ``[B, S_kv, Hkv, D]`` keys/values at ``selected_token_ids`` ``[B, S, R]``."""
    return _QSASparseGQA.apply(query, key, value, selected_token_ids.to(torch.int32), float(softmax_scale))
