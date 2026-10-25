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
"""Fused GLM-5.3-Flash manifold hyper-connections (mHC) for frozen hyper-connection weights.

Each decoder layer carries ``hc_mult`` residual streams ``[tokens, hc, hidden]``. At its
attention and MLP sites, ``Glm5NextTextHyperConnection`` RMS-normalizes the flattened streams,
projects them to ``(2 + hc) * hc`` mixing logits, collapses the streams into the sublayer input,
and the layer then remixes ``post * sublayer_out + comb^T @ streams``. The reference runs these as
FP32 elementwise passes over copies of the streams; here three Triton kernels each read the
streams once per direction:

- ``hc_logits``: RMS and the logit dot products (3xTF32) in one pass. Its backward is closed-form,
  ``dX = r * (dL @ fn) - X * (r^3 / K) * sum(dL * dots)``, with no FP32 copy of ``X``.
- ``hc_collapse``: ``sum_h pre[h] * X[h]``.
- ``hc_mix``: ``post[j] * y + sum_i comb[i, j] * X[i]``.

A fourth kernel, ``hc_gates``, turns each token's ``(2 + hc) * hc`` logits into ``pre``, ``post`` and
the Sinkhorn-normalized ``comb``; the reference spends ~80 small kernels on that per site. Its
backward recomputes the Sinkhorn iterates in registers instead of saving them. When ``fn``/``base``/``scale``
train (full fine-tuning), their gradients come from the same backward: ``fn``'s as ``(dL * r)^T X`` in FP32
row chunks, ``base``'s and ``scale``'s from the gates' pre-scale logit gradient.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch import nn

# Measured at [32768, 16384] x [24, 16384] on B300: 0.8 ms at 64 x 128, 6.4 ms at 32 x 256 (spills).
_LOGIT_ROWS = 64
_LOGIT_COLS = 128
_LOGIT_BWD_ROWS = 32
_LOGIT_BWD_COLS = 256
_STREAM_ROWS = 4
_STREAM_COLS = 256


@triton.jit
def _hc_logits_fwd_kernel(
    x_ptr,
    fn_ptr,
    dots_ptr,
    sumsq_ptr,
    T,
    K,
    M: tl.constexpr,
    M_PAD: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    m = tl.arange(0, M_PAD)
    acc = tl.zeros((ROWS, M_PAD), dtype=tl.float32)
    sq = tl.zeros((ROWS,), dtype=tl.float32)
    for k0 in range(0, K, COLS):
        col = k0 + tl.arange(0, COLS)
        x = tl.load(
            x_ptr + row[:, None].to(tl.int64) * K + col[None, :],
            mask=(row[:, None] < T) & (col[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        w = tl.load(fn_ptr + m[:, None] * K + col[None, :], mask=(m[:, None] < M) & (col[None, :] < K), other=0.0)
        acc += tl.dot(x, tl.trans(w), input_precision="tf32x3")
        sq += tl.sum(x * x, axis=1)
    tl.store(dots_ptr + row[:, None] * M + m[None, :], acc, mask=(row[:, None] < T) & (m[None, :] < M))
    tl.store(sumsq_ptr + row, sq, mask=row < T)


@triton.jit
def _hc_logits_bwd_kernel(
    x_ptr,
    fn_ptr,
    dl_ptr,
    a_ptr,
    b_ptr,
    dx_ptr,
    T,
    K,
    M: tl.constexpr,
    M_PAD: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.program_id(1) * COLS + tl.arange(0, COLS)
    m = tl.arange(0, M_PAD)
    dl = tl.load(dl_ptr + row[:, None] * M + m[None, :], mask=(row[:, None] < T) & (m[None, :] < M), other=0.0)
    w = tl.load(fn_ptr + m[:, None] * K + col[None, :], mask=(m[:, None] < M) & (col[None, :] < K), other=0.0)
    g = tl.dot(dl, w, input_precision="tf32x3")
    mask = (row[:, None] < T) & (col[None, :] < K)
    offset = row[:, None].to(tl.int64) * K + col[None, :]
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + row, mask=row < T, other=0.0)
    b = tl.load(b_ptr + row, mask=row < T, other=0.0)
    tl.store(dx_ptr + offset, (a[:, None] * g - b[:, None] * x).to(dx_ptr.dtype.element_ty), mask=mask)


_FN_GRAD_ROWS = 2048  # rows of x upcast to FP32 at a time for fn's gradient


class _HcLogitsFn(torch.autograd.Function):
    """``F.linear(x * rsqrt(mean(x^2) + eps), fn)`` over flattened streams ``[T, K]``; gradients reach ``x`` and,
    when it trains, ``fn``."""

    @staticmethod
    def forward(ctx, x: Tensor, fn: Tensor, eps: float) -> Tensor:
        T, K = x.shape
        M = fn.shape[0]
        dots = torch.empty(T, M, dtype=torch.float32, device=x.device)
        sumsq = torch.empty(T, dtype=torch.float32, device=x.device)
        _hc_logits_fwd_kernel[(triton.cdiv(T, _LOGIT_ROWS),)](
            x, fn, dots, sumsq, T, K, M, triton.next_power_of_2(max(M, 16)), _LOGIT_ROWS, _LOGIT_COLS
        )
        r = torch.rsqrt(sumsq / K + eps)
        ctx.save_for_backward(x, fn, dots, r)
        return dots * r[:, None]

    @staticmethod
    def backward(ctx, grad: Tensor):
        x, fn, dots, r = ctx.saved_tensors
        T, K = x.shape
        M = fn.shape[0]
        grad = grad.float().contiguous()
        b = (r**3 / K) * (grad * dots).sum(-1)
        dx = torch.empty_like(x)
        grid = (triton.cdiv(T, _LOGIT_BWD_ROWS), triton.cdiv(K, _LOGIT_BWD_COLS))
        _hc_logits_bwd_kernel[grid](
            x,
            fn,
            grad,
            r.contiguous(),
            b.contiguous(),
            dx,
            T,
            K,
            M,
            triton.next_power_of_2(max(M, 16)),
            _LOGIT_BWD_ROWS,
            _LOGIT_BWD_COLS,
        )
        dfn = None
        if ctx.needs_input_grad[1]:
            scaled = grad * r[:, None]
            dfn = torch.zeros(M, K, dtype=torch.float32, device=x.device)
            for start in range(0, T, _FN_GRAD_ROWS):
                end = min(start + _FN_GRAD_ROWS, T)
                dfn.addmm_(scaled[start:end].t(), x[start:end].float())
            dfn = dfn.to(fn.dtype)
        return dx, dfn, None


@triton.jit
def _hc_collapse_fwd_kernel(x_ptr, pre_ptr, out_ptr, T, D, H: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.program_id(1) * COLS + tl.arange(0, COLS)
    mask = (row[:, None] < T) & (col[None, :] < D)
    stream = row[:, None].to(tl.int64) * (H * D) + col[None, :]
    acc = tl.zeros((ROWS, COLS), dtype=tl.float32)
    for h in tl.static_range(H):
        pre = tl.load(pre_ptr + row * H + h, mask=row < T, other=0.0)
        acc += pre[:, None] * tl.load(x_ptr + stream + h * D, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row[:, None].to(tl.int64) * D + col[None, :], acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _hc_collapse_bwd_kernel(
    x_ptr, pre_ptr, dout_ptr, dx_ptr, dpre_ptr, T, D, H: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    for h in tl.static_range(H):
        pre = tl.load(pre_ptr + row * H + h, mask=row < T, other=0.0)
        dpre = tl.zeros((ROWS,), dtype=tl.float32)
        for c0 in range(0, D, COLS):
            col = c0 + tl.arange(0, COLS)
            mask = (row[:, None] < T) & (col[None, :] < D)
            dout = tl.load(dout_ptr + row[:, None].to(tl.int64) * D + col[None, :], mask=mask, other=0.0)
            dout = dout.to(tl.float32)
            stream = row[:, None].to(tl.int64) * (H * D) + h * D + col[None, :]
            x = tl.load(x_ptr + stream, mask=mask, other=0.0).to(tl.float32)
            tl.store(dx_ptr + stream, (pre[:, None] * dout).to(dx_ptr.dtype.element_ty), mask=mask)
            dpre += tl.sum(x * dout, axis=1)
        tl.store(dpre_ptr + row * H + h, dpre, mask=row < T)


class _HcCollapseFn(torch.autograd.Function):
    """``sum_h pre[:, h] * x[:, h]`` for streams ``x [T, H, D]`` and FP32 weights ``pre [T, H]``."""

    @staticmethod
    def forward(ctx, x: Tensor, pre: Tensor) -> Tensor:
        T, H, D = x.shape
        out = torch.empty(T, D, dtype=x.dtype, device=x.device)
        grid = (triton.cdiv(T, _STREAM_ROWS), triton.cdiv(D, _STREAM_COLS))
        _hc_collapse_fwd_kernel[grid](x, pre, out, T, D, H, _STREAM_ROWS, _STREAM_COLS)
        ctx.save_for_backward(x, pre)
        return out

    @staticmethod
    def backward(ctx, grad: Tensor):
        x, pre = ctx.saved_tensors
        T, H, D = x.shape
        dx = torch.empty_like(x)
        dpre = torch.empty(T, H, dtype=torch.float32, device=x.device)
        _hc_collapse_bwd_kernel[(triton.cdiv(T, _STREAM_ROWS),)](
            x, pre, grad.contiguous(), dx, dpre, T, D, H, _STREAM_ROWS, _STREAM_COLS
        )
        return dx, dpre


@triton.jit
def _hc_mix_fwd_kernel(
    y_ptr, x_ptr, post_ptr, comb_ptr, out_ptr, T, D, H: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.program_id(1) * COLS + tl.arange(0, COLS)
    h = tl.arange(0, H)
    rows = row[:, None, None].to(tl.int64)
    mask3 = (row[:, None, None] < T) & (col[None, None, :] < D)
    x = tl.load(x_ptr + (rows * H + h[None, :, None]) * D + col[None, None, :], mask=mask3, other=0.0).to(tl.float32)
    y = tl.load(
        y_ptr + row[:, None].to(tl.int64) * D + col[None, :], mask=(row[:, None] < T) & (col[None, :] < D), other=0.0
    ).to(tl.float32)
    post = tl.load(post_ptr + row[:, None] * H + h[None, :], mask=row[:, None] < T, other=0.0)
    comb = tl.load(
        comb_ptr + (row[:, None, None] * H + h[None, :, None]) * H + h[None, None, :],
        mask=row[:, None, None] < T,
        other=0.0,
    )
    # out[:, j] = post[:, j] * y + sum_i comb[:, i, j] * x[:, i]
    mixed = tl.sum(comb[:, :, :, None] * x[:, :, None, :], axis=1)
    out = post[:, :, None] * y[:, None, :] + mixed
    tl.store(
        out_ptr + (rows * H + h[None, :, None]) * D + col[None, None, :], out.to(out_ptr.dtype.element_ty), mask=mask3
    )


@triton.jit
def _hc_mix_bwd_kernel(
    y_ptr,
    x_ptr,
    post_ptr,
    comb_ptr,
    dout_ptr,
    dy_ptr,
    dx_ptr,
    dpost_ptr,
    dcomb_ptr,
    T,
    D,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    # Four streams, unrolled: 2-D tiles keep every load and reduction along the contiguous axis.
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    live = row < T
    p0 = tl.load(post_ptr + row * 4 + 0, mask=live, other=0.0)
    p1 = tl.load(post_ptr + row * 4 + 1, mask=live, other=0.0)
    p2 = tl.load(post_ptr + row * 4 + 2, mask=live, other=0.0)
    p3 = tl.load(post_ptr + row * 4 + 3, mask=live, other=0.0)
    c00 = tl.load(comb_ptr + row * 16 + 0, mask=live, other=0.0)
    c01 = tl.load(comb_ptr + row * 16 + 1, mask=live, other=0.0)
    c02 = tl.load(comb_ptr + row * 16 + 2, mask=live, other=0.0)
    c03 = tl.load(comb_ptr + row * 16 + 3, mask=live, other=0.0)
    c10 = tl.load(comb_ptr + row * 16 + 4, mask=live, other=0.0)
    c11 = tl.load(comb_ptr + row * 16 + 5, mask=live, other=0.0)
    c12 = tl.load(comb_ptr + row * 16 + 6, mask=live, other=0.0)
    c13 = tl.load(comb_ptr + row * 16 + 7, mask=live, other=0.0)
    c20 = tl.load(comb_ptr + row * 16 + 8, mask=live, other=0.0)
    c21 = tl.load(comb_ptr + row * 16 + 9, mask=live, other=0.0)
    c22 = tl.load(comb_ptr + row * 16 + 10, mask=live, other=0.0)
    c23 = tl.load(comb_ptr + row * 16 + 11, mask=live, other=0.0)
    c30 = tl.load(comb_ptr + row * 16 + 12, mask=live, other=0.0)
    c31 = tl.load(comb_ptr + row * 16 + 13, mask=live, other=0.0)
    c32 = tl.load(comb_ptr + row * 16 + 14, mask=live, other=0.0)
    c33 = tl.load(comb_ptr + row * 16 + 15, mask=live, other=0.0)
    zero = tl.zeros((ROWS,), dtype=tl.float32)
    dp0, dp1, dp2, dp3 = zero, zero, zero, zero
    dc00, dc01, dc02, dc03 = zero, zero, zero, zero
    dc10, dc11, dc12, dc13 = zero, zero, zero, zero
    dc20, dc21, dc22, dc23 = zero, zero, zero, zero
    dc30, dc31, dc32, dc33 = zero, zero, zero, zero
    for c0 in range(0, D, COLS):
        col = c0 + tl.arange(0, COLS)
        mask = live[:, None] & (col[None, :] < D)
        stream = row[:, None].to(tl.int64) * (4 * D) + col[None, :]
        single = row[:, None].to(tl.int64) * D + col[None, :]
        y = tl.load(y_ptr + single, mask=mask, other=0.0).to(tl.float32)
        o0 = tl.load(dout_ptr + stream + 0 * D, mask=mask, other=0.0).to(tl.float32)
        o1 = tl.load(dout_ptr + stream + 1 * D, mask=mask, other=0.0).to(tl.float32)
        o2 = tl.load(dout_ptr + stream + 2 * D, mask=mask, other=0.0).to(tl.float32)
        o3 = tl.load(dout_ptr + stream + 3 * D, mask=mask, other=0.0).to(tl.float32)
        x0 = tl.load(x_ptr + stream + 0 * D, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + stream + 1 * D, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + stream + 2 * D, mask=mask, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + stream + 3 * D, mask=mask, other=0.0).to(tl.float32)
        dy = p0[:, None] * o0 + p1[:, None] * o1 + p2[:, None] * o2 + p3[:, None] * o3
        tl.store(dy_ptr + single, dy.to(dy_ptr.dtype.element_ty), mask=mask)
        dx0 = c00[:, None] * o0 + c01[:, None] * o1 + c02[:, None] * o2 + c03[:, None] * o3
        tl.store(dx_ptr + stream + 0 * D, dx0.to(dx_ptr.dtype.element_ty), mask=mask)
        dx1 = c10[:, None] * o0 + c11[:, None] * o1 + c12[:, None] * o2 + c13[:, None] * o3
        tl.store(dx_ptr + stream + 1 * D, dx1.to(dx_ptr.dtype.element_ty), mask=mask)
        dx2 = c20[:, None] * o0 + c21[:, None] * o1 + c22[:, None] * o2 + c23[:, None] * o3
        tl.store(dx_ptr + stream + 2 * D, dx2.to(dx_ptr.dtype.element_ty), mask=mask)
        dx3 = c30[:, None] * o0 + c31[:, None] * o1 + c32[:, None] * o2 + c33[:, None] * o3
        tl.store(dx_ptr + stream + 3 * D, dx3.to(dx_ptr.dtype.element_ty), mask=mask)
        dp0 += tl.sum(y * o0, axis=1)
        dp1 += tl.sum(y * o1, axis=1)
        dp2 += tl.sum(y * o2, axis=1)
        dp3 += tl.sum(y * o3, axis=1)
        dc00 += tl.sum(x0 * o0, axis=1)
        dc01 += tl.sum(x0 * o1, axis=1)
        dc02 += tl.sum(x0 * o2, axis=1)
        dc03 += tl.sum(x0 * o3, axis=1)
        dc10 += tl.sum(x1 * o0, axis=1)
        dc11 += tl.sum(x1 * o1, axis=1)
        dc12 += tl.sum(x1 * o2, axis=1)
        dc13 += tl.sum(x1 * o3, axis=1)
        dc20 += tl.sum(x2 * o0, axis=1)
        dc21 += tl.sum(x2 * o1, axis=1)
        dc22 += tl.sum(x2 * o2, axis=1)
        dc23 += tl.sum(x2 * o3, axis=1)
        dc30 += tl.sum(x3 * o0, axis=1)
        dc31 += tl.sum(x3 * o1, axis=1)
        dc32 += tl.sum(x3 * o2, axis=1)
        dc33 += tl.sum(x3 * o3, axis=1)
    tl.store(dpost_ptr + row * 4 + 0, dp0, mask=live)
    tl.store(dpost_ptr + row * 4 + 1, dp1, mask=live)
    tl.store(dpost_ptr + row * 4 + 2, dp2, mask=live)
    tl.store(dpost_ptr + row * 4 + 3, dp3, mask=live)
    tl.store(dcomb_ptr + row * 16 + 0, dc00, mask=live)
    tl.store(dcomb_ptr + row * 16 + 1, dc01, mask=live)
    tl.store(dcomb_ptr + row * 16 + 2, dc02, mask=live)
    tl.store(dcomb_ptr + row * 16 + 3, dc03, mask=live)
    tl.store(dcomb_ptr + row * 16 + 4, dc10, mask=live)
    tl.store(dcomb_ptr + row * 16 + 5, dc11, mask=live)
    tl.store(dcomb_ptr + row * 16 + 6, dc12, mask=live)
    tl.store(dcomb_ptr + row * 16 + 7, dc13, mask=live)
    tl.store(dcomb_ptr + row * 16 + 8, dc20, mask=live)
    tl.store(dcomb_ptr + row * 16 + 9, dc21, mask=live)
    tl.store(dcomb_ptr + row * 16 + 10, dc22, mask=live)
    tl.store(dcomb_ptr + row * 16 + 11, dc23, mask=live)
    tl.store(dcomb_ptr + row * 16 + 12, dc30, mask=live)
    tl.store(dcomb_ptr + row * 16 + 13, dc31, mask=live)
    tl.store(dcomb_ptr + row * 16 + 14, dc32, mask=live)
    tl.store(dcomb_ptr + row * 16 + 15, dc33, mask=live)


_GATE_ROWS = 64


@triton.jit
def _hc_gates_load(logits_ptr, base_ptr, scale_ptr, rows, mask):
    """The four-stream site's pre/post logits and its comb logits ``[rows, 4, 4]``, scaled and biased."""
    h = tl.arange(0, 4)
    pre = tl.load(logits_ptr + rows[:, None] * 24 + h[None, :], mask=mask[:, None], other=0.0)
    post = tl.load(logits_ptr + rows[:, None] * 24 + 4 + h[None, :], mask=mask[:, None], other=0.0)
    cell = h[:, None] * 4 + h[None, :]
    comb = tl.load(logits_ptr + rows[:, None, None] * 24 + 8 + cell[None], mask=mask[:, None, None], other=0.0)
    pre = pre * tl.load(scale_ptr) + tl.load(base_ptr + h)[None, :]
    post = post * tl.load(scale_ptr + 1) + tl.load(base_ptr + 4 + h)[None, :]
    comb = comb * tl.load(scale_ptr + 2) + tl.load(base_ptr + 8 + cell)[None]
    return pre, post, comb, cell


@triton.jit
def _hc_softmax(z):
    e = tl.exp(z - tl.max(z, axis=2)[:, :, None])
    return e / tl.sum(e, axis=2)[:, :, None]


@triton.jit
def _hc_sinkhorn_step(x, is_row, eps):
    """Normalize rows (``is_row``) or columns of each token's matrix."""
    rows = x / (tl.sum(x, axis=2)[:, :, None] + eps)
    cols = x / (tl.sum(x, axis=1)[:, None, :] + eps)
    return tl.where(is_row, rows, cols)


@triton.jit
def _hc_gates_fwd_kernel(
    logits_ptr, base_ptr, scale_ptr, pre_ptr, post_ptr, comb_ptr, T, eps, ITERS: tl.constexpr, ROWS: tl.constexpr
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    mask = rows < T
    h = tl.arange(0, 4)
    pre, post, comb, cell = _hc_gates_load(logits_ptr, base_ptr, scale_ptr, rows, mask)
    tl.store(pre_ptr + rows[:, None] * 4 + h[None, :], tl.sigmoid(pre) + eps, mask=mask[:, None])
    tl.store(post_ptr + rows[:, None] * 4 + h[None, :], 2 * tl.sigmoid(post), mask=mask[:, None])
    # Sinkhorn step k normalizes columns for even k and rows for odd k; there are 2 * ITERS - 1 steps.
    x = _hc_softmax(comb) + eps
    for k in range(2 * ITERS - 1):
        x = _hc_sinkhorn_step(x, (k % 2) == 1, eps)
    tl.store(comb_ptr + rows[:, None, None] * 16 + cell[None], x, mask=mask[:, None, None])


@triton.jit
def _hc_gates_bwd_kernel(
    logits_ptr,
    base_ptr,
    scale_ptr,
    dpre_ptr,
    dpost_ptr,
    dcomb_ptr,
    dlogits_ptr,
    T,
    eps,
    ITERS: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    mask = rows < T
    h = tl.arange(0, 4)
    pre, post, comb, cell = _hc_gates_load(logits_ptr, base_ptr, scale_ptr, rows, mask)
    sig_pre = tl.sigmoid(pre)
    sig_post = tl.sigmoid(post)
    dpre = tl.load(dpre_ptr + rows[:, None] * 4 + h[None, :], mask=mask[:, None], other=0.0)
    dpost = tl.load(dpost_ptr + rows[:, None] * 4 + h[None, :], mask=mask[:, None], other=0.0)
    # The gradients of the scaled, biased logits; the caller applies ``scale`` (and reduces ``base`` / ``scale``).
    out = dlogits_ptr + rows[:, None] * 24 + h[None, :]
    tl.store(out, dpre * sig_pre * (1 - sig_pre), mask=mask[:, None])
    tl.store(out + 4, dpost * 2 * sig_post * (1 - sig_post), mask=mask[:, None])

    p = _hc_softmax(comb)
    x0 = p + eps
    g = tl.load(dcomb_ptr + rows[:, None, None] * 16 + cell[None], mask=mask[:, None, None], other=0.0)
    # Back through each step ``y = x / s`` (``dx = (dy - sum(dy * y)) / s``), recomputing its input x;
    # 39 steps make the recomputation cheap next to the memory a saved trajectory would take.
    steps = 2 * ITERS - 1
    for back in range(steps):
        k = steps - 1 - back
        x = x0
        for j in range(k):
            x = _hc_sinkhorn_step(x, (j % 2) == 1, eps)
        is_row = (k % 2) == 1
        s = tl.where(is_row, tl.sum(x, axis=2)[:, :, None] + eps, tl.sum(x, axis=1)[:, None, :] + eps)
        gy = g * (x / s)
        g = (g - tl.where(is_row, tl.sum(gy, axis=2)[:, :, None], tl.sum(gy, axis=1)[:, None, :])) / s
    dz = p * (g - tl.sum(g * p, axis=2)[:, :, None])
    tl.store(dlogits_ptr + rows[:, None, None] * 24 + 8 + cell[None], dz, mask=mask[:, None, None])


class _HcGatesFn(torch.autograd.Function):
    """``(pre, post, comb)`` of a four-stream site from its ``[T, 24]`` FP32 logits; gradients reach the logits and,
    when they train, ``base`` and ``scale``."""

    @staticmethod
    def forward(ctx, logits: Tensor, base: Tensor, scale: Tensor, eps: float, iters: int):
        T = logits.shape[0]
        pre = torch.empty(T, 4, dtype=torch.float32, device=logits.device)
        post = torch.empty_like(pre)
        comb = torch.empty(T, 4, 4, dtype=torch.float32, device=logits.device)
        _hc_gates_fwd_kernel[(triton.cdiv(T, _GATE_ROWS),)](
            logits, base, scale, pre, post, comb, T, eps, iters, _GATE_ROWS
        )
        ctx.save_for_backward(logits, base, scale)
        ctx.eps, ctx.iters = eps, iters
        return pre, post, comb

    @staticmethod
    def backward(ctx, dpre: Tensor | None, dpost: Tensor | None, dcomb: Tensor | None):
        logits, base, scale = ctx.saved_tensors
        T = logits.shape[0]
        dpre = torch.zeros(T, 4, device=logits.device) if dpre is None else dpre.float().contiguous()
        dpost = torch.zeros(T, 4, device=logits.device) if dpost is None else dpost.float().contiguous()
        dcomb = torch.zeros(T, 4, 4, device=logits.device) if dcomb is None else dcomb.float().contiguous()
        dz = torch.empty_like(logits)
        _hc_gates_bwd_kernel[(triton.cdiv(T, _GATE_ROWS),)](
            logits, base, scale, dpre, dpost, dcomb, dz, T, ctx.eps, ctx.iters, _GATE_ROWS
        )
        flat_scale = scale.reshape(-1)
        groups = (slice(0, 4), slice(4, 8), slice(8, 24))
        per_logit = torch.cat([flat_scale[i].expand(g.stop - g.start) for i, g in enumerate(groups)])
        dlogits = dz * per_logit
        dbase = dz.sum(0).reshape(base.shape).to(base.dtype) if ctx.needs_input_grad[1] else None
        dscale = None
        if ctx.needs_input_grad[2]:
            dscale = torch.stack([(dz[:, g] * logits[:, g]).sum() for g in groups]).reshape(scale.shape).to(scale.dtype)
        return dlogits, dbase, dscale, None, None


class _HcMixFn(torch.autograd.Function):
    """``post[:, j] * y + sum_i comb[:, i, j] * x[:, i]`` for ``y [T, D]``, streams ``x [T, H, D]``."""

    @staticmethod
    def forward(ctx, y: Tensor, x: Tensor, post: Tensor, comb: Tensor) -> Tensor:
        T, H, D = x.shape
        out = torch.empty_like(x)
        grid = (triton.cdiv(T, _STREAM_ROWS), triton.cdiv(D, _STREAM_COLS))
        _hc_mix_fwd_kernel[grid](y, x, post, comb, out, T, D, H, _STREAM_ROWS, _STREAM_COLS)
        ctx.save_for_backward(y, x, post, comb)
        return out

    @staticmethod
    def backward(ctx, grad: Tensor):
        y, x, post, comb = ctx.saved_tensors
        T, H, D = x.shape
        dy, dx = torch.empty_like(y), torch.empty_like(x)
        dpost = torch.empty(T, H, dtype=torch.float32, device=x.device)
        dcomb = torch.empty(T, H, H, dtype=torch.float32, device=x.device)
        _hc_mix_bwd_kernel[(triton.cdiv(T, _STREAM_ROWS),)](
            y, x, post, comb, grad.contiguous(), dy, dx, dpost, dcomb, T, D, _STREAM_ROWS, _STREAM_COLS
        )
        return dy, dx, dpost, dcomb


def fused_hyper_connection_applies(hc: nn.Module, streams: Tensor) -> bool:
    """CUDA streams and GLM-5.3-Flash's four streams; ``fn``/``base``/``scale`` may be frozen or train."""
    return streams.is_cuda and hc.hc_mult == 4


def hyper_connection_gates(hc: nn.Module, streams: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """A site's ``(pre, post, comb)`` from its streams ``[batch, length, hc, hidden]``, without the collapse; for
    DeepSeek-V4.1, whose shifted mHC collapses each site's streams with the previous site's ``pre``."""
    batch, length, hc_mult, hidden = streams.shape
    x = streams.reshape(batch * length, hc_mult * hidden).contiguous()
    logits = _HcLogitsFn.apply(x, hc.fn.float().contiguous(), hc.input_norm.eps)
    pre, post, comb = _HcGatesFn.apply(
        logits.contiguous(),
        hc.base.float().contiguous(),
        hc.scale.float().contiguous(),
        hc.hc_eps,
        hc.hc_sinkhorn_iters,
    )
    return (
        pre.view(batch, length, hc_mult),
        post.view(batch, length, hc_mult),
        comb.view(batch, length, hc_mult, hc_mult),
    )


def collapse_streams(streams: Tensor, pre: Tensor) -> Tensor:
    """``sum_h pre[..., h] * streams[..., h, :]`` in FP32, cast back to the streams' dtype."""
    batch, length, hc_mult, hidden = streams.shape
    collapsed = _HcCollapseFn.apply(
        streams.reshape(batch * length, hc_mult, hidden).contiguous(),
        pre.reshape(batch * length, hc_mult).float().contiguous(),
    )
    return collapsed.view(batch, length, hidden)


def hyper_connection(hc: nn.Module, streams: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``Glm5NextTextHyperConnection.forward`` with fused stream passes: ``(post, comb, collapsed)``."""
    pre, post, comb = hyper_connection_gates(hc, streams)
    return post, comb, collapse_streams(streams, pre)


def mix_streams(post: Tensor, comb: Tensor, sublayer_out: Tensor, streams: Tensor) -> Tensor:
    """``post * sublayer_out + comb^T @ streams`` per token, as the decoder layer remixes after a site."""
    batch, length, hc_mult, hidden = streams.shape
    out = _HcMixFn.apply(
        sublayer_out.reshape(batch * length, hidden).contiguous(),
        streams.reshape(batch * length, hc_mult, hidden).contiguous(),
        post.reshape(batch * length, hc_mult).float().contiguous(),
        comb.reshape(batch * length, hc_mult, hc_mult).float().contiguous(),
    )
    return out.view(batch, length, hc_mult, hidden)
