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
"""Causal grouped-query attention for head size 512 (Gemma 4 full-attention layers), forward and backward, in TileLang.

FlashAttention 3 and the fused torch SDPA backends stop at head size 256, and cuDNN runs 512 only forward. Compiled flex
attention covers it at 37-150 TF/s forward and 44-59 TF/s backward on an H200. These kernels follow FlashMLA's layout for
a 512-wide head: two warpgroups split the 64 x N score tile and the 64 x 512 output tile along N, so the fp32 output
accumulator is 128 registers per thread.

* forward: one CTA per (64-query block, query head); the score tile goes through shared memory into the P @ V GEMM.
* backward: Delta = rowsum(O * dO); then dK / dV per 32-row KV block, accumulated transposed ([512, 32], so the head
  axis is the GEMM's M) over every later query block of every query head in the group, so no atomics; and dQ per
  64-query block. Seven score-sized GEMMs instead of five; the 512 x 32 / 64 x 512 accumulators fit in registers.
* packed rows: per-token segment starts (from the packer's ``cu_seqlens``) mask each score, so a token sees only its
  own row; the forward's KV loop also starts at its block's first segment start. The backward loops keep their
  causal bounds (bounds read from memory made TileLang predicate every load), so packing saves no backward work.

Layouts: ``query`` [B, H, S, D], ``key`` / ``value`` [B, HK, S, D] (what Ulysses hands its core attention), output
[B, S, H, D] (what HF's attention interface returns). The log-sum-exp is kept in log2 units.
"""

# No ``from __future__ import annotations``: TileLang reads the prim_funcs' tensor annotations, which name closure
# variables (heads, kv_heads, ...); as strings they would not be captured.

# TileLang's libcudart stub needs the real runtime loaded globally first (see kernels/deepseek_v4/dsv4_sparse_attn.py).
import ctypes as _ctypes

try:
    _ctypes.CDLL("libcudart.so", mode=_ctypes.RTLD_GLOBAL)
except OSError:  # CPU-only machines
    pass

import functools

import torch

HEAD_DIM = 512
SEQ_TILE = 64
_LOG2E = 1.44269504


def unsupported_reason(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> str | None:
    """Why these inputs cannot take the kernels, or ``None`` when they can."""
    if not query.is_cuda:
        return "the kernels need CUDA tensors"
    if any(tensor.dtype != torch.bfloat16 for tensor in (query, key, value)):
        return "the kernels take bfloat16 only"
    if query.ndim != 4 or key.shape != value.shape or key.ndim != 4:
        return f"expected [B, H, S, D] query and matching key / value, got {tuple(query.shape)} {tuple(key.shape)}"
    batch, heads, seq_len, dim = query.shape
    if dim != HEAD_DIM or key.shape[-1] != HEAD_DIM:
        return f"the kernels are built for head size {HEAD_DIM}, got {dim}"
    if key.shape[0] != batch or key.shape[2] != seq_len:
        return f"query {tuple(query.shape)} and key {tuple(key.shape)} differ in batch or sequence"
    if heads % key.shape[1]:
        return f"{heads} query heads do not group over {key.shape[1]} KV heads"
    if seq_len % SEQ_TILE:
        return f"sequence length {seq_len} is not a multiple of the {SEQ_TILE}-token tile"
    return None


@functools.cache
def _kernels(batch: int, heads: int, kv_heads: int, seq_len: int, softmax_scale: float):
    """The four kernels for one shape, compiled on first use (about a minute; cached on disk by TileLang after that).

    The sequence length is static: with ``T.dynamic`` the backward ran 2.3x slower at 64K (2,932 ms against 1,250 ms).
    """
    import tilelang
    import tilelang.language as T

    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    dim = HEAD_DIM
    groups = heads // kv_heads
    scale = float(softmax_scale * _LOG2E)
    sm_scale = float(softmax_scale)
    dtype = T.bfloat16
    accum = T.float32
    threads = 256

    @tilelang.jit(out_idx=[4, 5], pass_configs=pass_configs)
    def forward(block_M=64, block_N=64):
        @T.prim_func
        def main(
            Q: T.Tensor([batch, heads, seq_len, dim], dtype),
            K: T.Tensor([batch, kv_heads, seq_len, dim], dtype),
            V: T.Tensor([batch, kv_heads, seq_len, dim], dtype),
            SegStart: T.Tensor([batch, seq_len], T.int32),
            Output: T.Tensor([batch, seq_len, heads, dim], dtype),
            Lse: T.Tensor([batch, heads, seq_len], accum),
        ):
            with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                S_shared = T.alloc_shared([block_M, block_N], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum)
                acc_o = T.alloc_fragment([block_M, dim], accum)
                scores_max = T.alloc_fragment([block_M], accum)
                scores_max_prev = T.alloc_fragment([block_M], accum)
                scores_scale = T.alloc_fragment([block_M], accum)
                scores_sum = T.alloc_fragment([block_M], accum)
                logsum = T.alloc_fragment([block_M], accum)
                max_safe = T.alloc_fragment([block_M], accum)

                qb = T.ceildiv(seq_len, block_M) - 1 - bx  # the longest (latest) query blocks first
                kvh = by // groups
                T.copy(Q[bz, by, qb * block_M : (qb + 1) * block_M, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum))
                # The block's first row starts the earliest segment any of its rows attends to. The segment starts
                # are read from global memory in the mask: staging them through shared memory deadlocked the
                # warp-specialized pipeline in TileLang 0.1.12.
                k_first = SegStart[bz, qb * block_M] // block_N
                n_k = T.ceildiv(T.min((qb + 1) * block_M, seq_len), block_N) - k_first
                for t in T.Pipelined(n_k, num_stages=1):
                    k = k_first + t
                    T.copy(K[bz, kvh, k * block_N : (k + 1) * block_N, :], K_shared)
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            (qb * block_M + i >= k * block_N + j)
                            & (k * block_N + j >= SegStart[bz, qb * block_M + i]),
                            0,
                            -T.infinity(accum),
                        )
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                    T.copy(V[bz, kvh, k * block_N : (k + 1) * block_N, :], V_shared)
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    # A row of a later packed segment can have no visible key in this block yet: keep its
                    # exponents at exp2(-inf) = 0 instead of -inf - -inf.
                    for i in T.Parallel(block_M):
                        max_safe[i] = T.if_then_else(scores_max[i] == -T.infinity(accum), 0, scores_max[i])
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - max_safe[i] * scale)
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - max_safe[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, S_shared)
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scores_scale[i]
                    T.gemm(S_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, Output[bz, qb * block_M : (qb + 1) * block_M, by, :])
                for i in T.Parallel(block_M):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, Lse[bz, by, qb * block_M : (qb + 1) * block_M])

        return main

    @tilelang.jit(out_idx=[2], pass_configs=pass_configs)
    def delta(blk=32):
        @T.prim_func
        def main(
            O: T.Tensor([batch, seq_len, heads, dim], dtype),
            dO: T.Tensor([batch, seq_len, heads, dim], dtype),
            Delta: T.Tensor([batch, heads, seq_len], accum),
        ):
            with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
                o = T.alloc_fragment([blk, blk], dtype)
                do = T.alloc_fragment([blk, blk], dtype)
                acc = T.alloc_fragment([blk, blk], accum)
                row = T.alloc_fragment([blk], accum)
                T.clear(acc)
                for k in range(T.ceildiv(dim, blk)):
                    T.copy(O[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], o)
                    T.copy(dO[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], do)
                    for i, j in T.Parallel(blk, blk):
                        # Multiply in fp32: a bf16 product rounds every term of the row sum, and Delta - dP
                        # cancels where one key takes the whole softmax (an attention sink).
                        acc[i, j] += T.cast(o[i, j], accum) * T.cast(do[i, j], accum)
                T.reduce_sum(acc, row, 1)
                T.copy(row, Delta[bz, bx, by * blk : (by + 1) * blk])

        return main

    @tilelang.jit(out_idx=[7, 8], pass_configs=pass_configs)
    def dkdv(block_M=64, block_N=32):
        @T.prim_func
        def main(
            Q: T.Tensor([batch, heads, seq_len, dim], dtype),
            K: T.Tensor([batch, kv_heads, seq_len, dim], dtype),
            V: T.Tensor([batch, kv_heads, seq_len, dim], dtype),
            dO: T.Tensor([batch, seq_len, heads, dim], dtype),
            Lse: T.Tensor([batch, heads, seq_len], accum),
            Delta: T.Tensor([batch, heads, seq_len], accum),
            SegStart: T.Tensor([batch, seq_len], T.int32),
            dK: T.Tensor([batch, kv_heads, seq_len, dim], dtype),
            dV: T.Tensor([batch, kv_heads, seq_len, dim], dtype),
        ):
            with T.Kernel(T.ceildiv(seq_len, block_N), kv_heads, batch, threads=threads) as (bx, by, bz):
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                q_shared = T.alloc_shared([block_M, dim], dtype)
                do_shared = T.alloc_shared([block_M, dim], dtype)
                p_shared = T.alloc_shared([block_M, block_N], dtype)
                ds_shared = T.alloc_shared([block_M, block_N], dtype)
                lse_shared = T.alloc_shared([block_M], accum)
                delta_shared = T.alloc_shared([block_M], accum)
                seg_shared = T.alloc_shared([block_M], T.int32)
                s = T.alloc_fragment([block_M, block_N], accum)
                dp = T.alloc_fragment([block_M, block_N], accum)
                dvT = T.alloc_fragment([dim, block_N], accum)
                dkT = T.alloc_fragment([dim, block_N], accum)
                dvT_shared = T.alloc_shared([dim, block_N], dtype)
                dkT_shared = T.alloc_shared([dim, block_N], dtype)

                kb = bx  # early KV blocks meet the most query blocks: launch them first
                T.copy(K[bz, by, kb * block_N : (kb + 1) * block_N, :], K_shared)
                T.copy(V[bz, by, kb * block_N : (kb + 1) * block_N, :], V_shared)
                T.clear(dvT)
                T.clear(dkT)
                loop_st = T.floordiv(kb * block_N, block_M)
                # The loop bounds stay affine in the block index: bounds read from the segment ends made TileLang
                # predicate every load and the backward 2.3x slower. Packed rows are kept apart by the mask alone.
                n_q = T.ceildiv(seq_len, block_M) - loop_st
                for it in T.Pipelined(groups * n_q, num_stages=0):
                    h = by * groups + it // n_q
                    qb = loop_st + it % n_q
                    T.copy(Q[bz, h, qb * block_M : (qb + 1) * block_M, :], q_shared)
                    T.clear(s)
                    T.gemm(q_shared, K_shared, s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                    T.copy(Lse[bz, h, qb * block_M : (qb + 1) * block_M], lse_shared)
                    T.copy(SegStart[bz, qb * block_M : (qb + 1) * block_M], seg_shared)
                    for i, j in T.Parallel(block_M, block_N):
                        s[i, j] = T.if_then_else(
                            (qb * block_M + i >= kb * block_N + j) & (kb * block_N + j >= seg_shared[i]),
                            T.exp2(s[i, j] * scale - lse_shared[i]),
                            0,
                        )
                    T.copy(dO[bz, qb * block_M : (qb + 1) * block_M, h, :], do_shared)
                    T.clear(dp)
                    T.gemm(do_shared, V_shared, dp, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                    T.copy(s, p_shared)
                    # TileLang 0.1.12 computes a transpose_A GEMM wrong under FullRow with two warpgroups; Square is right.
                    T.gemm(do_shared, p_shared, dvT, transpose_A=True, policy=T.GemmWarpPolicy.Square)
                    T.copy(Delta[bz, h, qb * block_M : (qb + 1) * block_M], delta_shared)
                    for i, j in T.Parallel(block_M, block_N):
                        dp[i, j] = s[i, j] * (dp[i, j] - delta_shared[i]) * sm_scale
                    T.copy(dp, ds_shared)
                    T.gemm(q_shared, ds_shared, dkT, transpose_A=True, policy=T.GemmWarpPolicy.Square)
                T.copy(dvT, dvT_shared)
                T.copy(dkT, dkT_shared)
                for i, j in T.Parallel(block_N, dim):
                    dV[bz, by, kb * block_N + i, j] = dvT_shared[j, i]
                for i, j in T.Parallel(block_N, dim):
                    dK[bz, by, kb * block_N + i, j] = dkT_shared[j, i]

        return main

    @tilelang.jit(out_idx=[7], pass_configs=pass_configs)
    def dq(block_M=64, block_N=32):
        @T.prim_func
        def main(
            Q: T.Tensor([batch, heads, seq_len, dim], dtype),
            K: T.Tensor([batch, kv_heads, seq_len, dim], dtype),
            V: T.Tensor([batch, kv_heads, seq_len, dim], dtype),
            dO: T.Tensor([batch, seq_len, heads, dim], dtype),
            Lse: T.Tensor([batch, heads, seq_len], accum),
            Delta: T.Tensor([batch, heads, seq_len], accum),
            SegStart: T.Tensor([batch, seq_len], T.int32),
            dQ: T.Tensor([batch, heads, seq_len, dim], dtype),
        ):
            with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
                q_shared = T.alloc_shared([block_M, dim], dtype)
                do_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                ds_shared = T.alloc_shared([block_M, block_N], dtype)
                lse_shared = T.alloc_shared([block_M], accum)
                delta_shared = T.alloc_shared([block_M], accum)
                s = T.alloc_fragment([block_M, block_N], accum)
                dp = T.alloc_fragment([block_M, block_N], accum)
                acc_dq = T.alloc_fragment([block_M, dim], accum)
                seg_shared = T.alloc_shared([block_M], T.int32)

                qb = T.ceildiv(seq_len, block_M) - 1 - bx
                kvh = by // groups
                T.copy(Q[bz, by, qb * block_M : (qb + 1) * block_M, :], q_shared)
                T.copy(dO[bz, qb * block_M : (qb + 1) * block_M, by, :], do_shared)
                T.copy(Lse[bz, by, qb * block_M : (qb + 1) * block_M], lse_shared)
                T.copy(Delta[bz, by, qb * block_M : (qb + 1) * block_M], delta_shared)
                T.copy(SegStart[bz, qb * block_M : (qb + 1) * block_M], seg_shared)
                T.clear(acc_dq)
                for k in T.Pipelined(T.ceildiv((qb + 1) * block_M, block_N), num_stages=0):
                    T.copy(K[bz, kvh, k * block_N : (k + 1) * block_N, :], K_shared)
                    T.clear(s)
                    T.gemm(q_shared, K_shared, s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                    T.copy(V[bz, kvh, k * block_N : (k + 1) * block_N, :], V_shared)
                    T.clear(dp)
                    T.gemm(do_shared, V_shared, dp, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                    for i, j in T.Parallel(block_M, block_N):
                        s[i, j] = T.if_then_else(
                            (qb * block_M + i >= k * block_N + j) & (k * block_N + j >= seg_shared[i]),
                            T.exp2(s[i, j] * scale - lse_shared[i]),
                            0,
                        )
                    for i, j in T.Parallel(block_M, block_N):
                        dp[i, j] = s[i, j] * (dp[i, j] - delta_shared[i]) * sm_scale
                    T.copy(dp, ds_shared)
                    T.gemm(ds_shared, K_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)
                T.copy(acc_dq, dQ[bz, by, qb * block_M : (qb + 1) * block_M, :])

        return main

    return forward(), delta(), dkdv(), dq()


def _shape(query: torch.Tensor, key: torch.Tensor) -> tuple[int, int, int, int]:
    return int(query.shape[0]), int(query.shape[1]), int(key.shape[1]), int(query.shape[2])


def segment_starts(cu_seqlens: torch.Tensor, seq_len: int, batch: int = 1) -> torch.Tensor:
    """Per-token start of the packed row each token belongs to, int32 [batch, seq_len].

    ``cu_seqlens`` is the packer's ``[0, len_0, len_0 + len_1, ..., seq_len]`` over the flattened tokens; the same
    segments repeat on every batch row.
    """
    cu = cu_seqlens.to(torch.int64).flatten()
    if cu.numel() < 2 or int(cu[0]) != 0 or int(cu[-1]) != seq_len:
        raise ValueError(f"cu_seqlens must run from 0 to the sequence length {seq_len}, got {cu.tolist()[:8]}...")
    lengths = cu[1:] - cu[:-1]
    if bool((lengths < 0).any()):
        raise ValueError("cu_seqlens must be non-decreasing")
    starts = torch.repeat_interleave(cu[:-1], lengths).to(torch.int32)
    return starts.view(1, -1).expand(batch, -1).contiguous()


@torch.library.custom_op("arctic_hd512::forward", mutates_args=())
def _forward_op(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, seg_start: torch.Tensor, softmax_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """(output [B, S, H, D], log2-sum-exp [B, H, S]) for contiguous inputs. A torch op, so a selective-checkpoint
    policy can keep its outputs (``FORWARD_OP``)."""
    forward, _, _, _ = _kernels(*_shape(query, key), softmax_scale)
    output, lse = forward(query, key, value, seg_start)
    return output, lse


@_forward_op.register_fake
def _forward_fake(query, key, value, seg_start, softmax_scale):
    batch, heads, seq_len, dim = query.shape
    return query.new_empty(batch, seq_len, heads, dim), query.new_empty(batch, heads, seq_len, dtype=torch.float32)


@torch.library.custom_op("arctic_hd512::backward", mutates_args=())
def _backward_op(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    seg_start: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, delta, dkdv, dq = _kernels(*_shape(query, key), softmax_scale)
    row_delta = delta(output, grad_output)
    grad_key, grad_value = dkdv(query, key, value, grad_output, lse, row_delta, seg_start)
    grad_query = dq(query, key, value, grad_output, lse, row_delta, seg_start)
    return grad_query, grad_key, grad_value


@_backward_op.register_fake
def _backward_fake(grad_output, query, key, value, output, lse, seg_start, softmax_scale):
    return torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)


def _setup_context(ctx, inputs, output):
    query, key, value, seg_start, softmax_scale = inputs
    ctx.save_for_backward(query, key, value, output[0], output[1], seg_start)
    ctx.softmax_scale = softmax_scale


def _backward(ctx, grad_output, grad_lse):
    query, key, value, output, lse, seg_start = ctx.saved_tensors
    grad_query, grad_key, grad_value = _backward_op(
        grad_output.contiguous(), query, key, value, output, lse, seg_start, ctx.softmax_scale
    )
    return grad_query, grad_key, grad_value, None, None


_forward_op.register_autograd(_backward, setup_context=_setup_context)

#: The forward op, for selective activation checkpointing policies that keep attention outputs.
FORWARD_OP = torch.ops.arctic_hd512.forward.default


def causal_gqa_attention_hd512(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """``softmax(scale * Q K^T + causal) V`` for [B, H, S, 512] ``query`` over [B, HK, S, 512] ``key`` / ``value``.

    ``cu_seqlens`` (the packer's row boundaries over S, shared by every batch row) keeps packed rows apart: a token
    attends only to earlier tokens of its own row. Returns [B, S, H, 512]. Raises ``ValueError`` for inputs
    ``unsupported_reason`` rejects.
    """
    reason = unsupported_reason(query, key, value)
    if reason is not None:
        raise ValueError(reason)
    batch, _, seq_len, _ = query.shape
    if cu_seqlens is None:  # one row: no host round trip for the bounds
        seg_start = torch.zeros(batch, seq_len, device=query.device, dtype=torch.int32)
    else:
        seg_start = segment_starts(cu_seqlens.to(query.device), int(seq_len), int(batch))
    query, key, value = (tensor.contiguous() for tensor in (query, key, value))
    output, _ = _forward_op(query, key, value, seg_start, float(softmax_scale))
    return output
