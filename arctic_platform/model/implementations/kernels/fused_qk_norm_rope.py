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
"""Per-head RMSNorm followed by the rotary embedding (Qwen3's ``q_norm`` / ``k_norm`` + ``apply_rotary_pos_emb``),
in one pass each way.

``[tokens, heads, head_dim]`` in, the normalized and rotated tensor out. The composition it replaces upcasts the
projection output to FP32, squares, averages, scales and rounds it, multiplies the weight, and then rotates it
(each a pass over the tensor, with an FP32 copy saved for backward); here each token's heads are read once and
written once, and backward saves only the input and one FP32 reciprocal RMS per head.

The norm rounds as transformers' module does: ``n = bf16(x * rsqrt(mean(x^2) + eps))`` in FP32, ``y = bf16(w * n)``.
The rotation ``y * cos + rotate_half(y) * sin`` is rounded once, where transformers' BF16 ops round each product and
the sum (and Liger's Triton rope, as compiled, rounds one product and the sum): at most one BF16 rounding apart. The
backward runs in FP32 and rounds the input gradient once; the weight gradient is summed over tokens and heads in
FP32.
"""

import torch
import triton
import triton.language as tl

_TOKENS_PER_BWD_PROGRAM = 32


@triton.jit
def _norm_rope_fwd_kernel(
    x_ptr,
    x_tok_stride,
    w_ptr,
    cos_ptr,
    sin_ptr,
    cos_batch_stride,
    out_ptr,
    rstd_ptr,
    seq_len,
    n_heads,
    eps,
    HALF: tl.constexpr,
    PAD_H: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    b = tok // seq_len
    s = tok % seq_len
    heads = tl.arange(0, PAD_H)[:, None]
    cols = tl.arange(0, HALF)[None, :]
    mask = heads < n_heads
    x_row = x_ptr + tok * x_tok_stride + heads * (2 * HALF)
    x1 = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_row + HALF + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt((tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)) / (2 * HALF) + eps)
    tl.store(rstd_ptr + tok * n_heads + tl.arange(0, PAD_H), rstd, mask=tl.arange(0, PAD_H) < n_heads)
    r = rstd[:, None]
    dtype = out_ptr.dtype.element_ty
    w1 = tl.load(w_ptr + cols).to(tl.float32)
    w2 = tl.load(w_ptr + HALF + cols).to(tl.float32)
    y1 = (w1 * (x1 * r).to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
    y2 = (w2 * (x2 * r).to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
    trig = b * cos_batch_stride + s * (2 * HALF)
    c1 = tl.load(cos_ptr + trig + cols).to(tl.float32)
    c2 = tl.load(cos_ptr + trig + HALF + cols).to(tl.float32)
    s1 = tl.load(sin_ptr + trig + cols).to(tl.float32)
    s2 = tl.load(sin_ptr + trig + HALF + cols).to(tl.float32)
    # rotate_half(y) = [-y2, y1]. Each product of two BF16 values is exact in FP32, so the rotation rounds once (to
    # FP32, then BF16) whether or not the compiler contracts it into an FMA.
    o1 = y1 * c1 - y2 * s1
    o2 = y2 * c2 + y1 * s2
    out_row = out_ptr + tok * (n_heads * 2 * HALF) + heads * (2 * HALF)
    tl.store(out_row + cols, o1.to(dtype), mask=mask)
    tl.store(out_row + HALF + cols, o2.to(dtype), mask=mask)


@triton.jit
def _norm_rope_bwd_kernel(
    grad_ptr,
    x_ptr,
    x_tok_stride,
    w_ptr,
    cos_ptr,
    sin_ptr,
    cos_batch_stride,
    rstd_ptr,
    dx_ptr,
    dw_partial_ptr,
    n_tokens,
    seq_len,
    n_heads,
    HALF: tl.constexpr,
    PAD_H: tl.constexpr,
    TOKENS: tl.constexpr,
    NEED_DW: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    heads = tl.arange(0, PAD_H)[:, None]
    cols = tl.arange(0, HALF)[None, :]
    head_mask = heads < n_heads
    w1 = tl.load(w_ptr + cols).to(tl.float32)
    w2 = tl.load(w_ptr + HALF + cols).to(tl.float32)
    dw1 = tl.zeros((HALF,), tl.float32)
    dw2 = tl.zeros((HALF,), tl.float32)
    dtype = dx_ptr.dtype.element_ty
    for i in range(TOKENS):
        tok = pid * TOKENS + i
        live = tok < n_tokens
        mask = head_mask & live
        b = tok // seq_len
        s = tok % seq_len
        trig = b * cos_batch_stride + s * (2 * HALF)
        c1 = tl.load(cos_ptr + trig + cols, mask=live, other=0.0).to(tl.float32)
        c2 = tl.load(cos_ptr + trig + HALF + cols, mask=live, other=0.0).to(tl.float32)
        s1 = tl.load(sin_ptr + trig + cols, mask=live, other=0.0).to(tl.float32)
        s2 = tl.load(sin_ptr + trig + HALF + cols, mask=live, other=0.0).to(tl.float32)
        g_row = grad_ptr + tok * (n_heads * 2 * HALF) + heads * (2 * HALF)
        g1 = tl.load(g_row + cols, mask=mask, other=0.0).to(tl.float32)
        g2 = tl.load(g_row + HALF + cols, mask=mask, other=0.0).to(tl.float32)
        # out1 = y1 c1 - y2 s1, out2 = y2 c2 + y1 s2
        dy1 = g1 * c1 + g2 * s2
        dy2 = g2 * c2 - g1 * s1
        x_row = x_ptr + tok * x_tok_stride + heads * (2 * HALF)
        x1 = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_row + HALF + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(
            rstd_ptr + tok * n_heads + tl.arange(0, PAD_H), mask=(tl.arange(0, PAD_H) < n_heads) & live, other=0.0
        )[:, None]
        if NEED_DW:
            dw1 += tl.sum(dy1 * (x1 * r).to(dtype).to(tl.float32), axis=0)
            dw2 += tl.sum(dy2 * (x2 * r).to(dtype).to(tl.float32), axis=0)
        dn1 = dy1 * w1
        dn2 = dy2 * w2
        # n = x r, r = (mean(x^2) + eps)^-1/2:  dx = r dn - x r^3 mean(dn x)
        dot = (tl.sum(dn1 * x1, axis=1) + tl.sum(dn2 * x2, axis=1))[:, None] / (2 * HALF)
        dx1 = r * dn1 - x1 * (r * r * r) * dot
        dx2 = r * dn2 - x2 * (r * r * r) * dot
        dx_row = dx_ptr + tok * (n_heads * 2 * HALF) + heads * (2 * HALF)
        tl.store(dx_row + cols, dx1.to(dtype), mask=mask)
        tl.store(dx_row + HALF + cols, dx2.to(dtype), mask=mask)
    if NEED_DW:
        tl.store(dw_partial_ptr + pid * (2 * HALF) + tl.arange(0, HALF), dw1)
        tl.store(dw_partial_ptr + pid * (2 * HALF) + HALF + tl.arange(0, HALF), dw2)


def _num_warps(n_heads: int, head_dim: int) -> int:
    # One program holds a token's [heads, head_dim] tile in several FP32 copies.
    return 4 if triton.next_power_of_2(n_heads) * head_dim <= 1024 else 8


def _trig_rows(trig: torch.Tensor, batch: int, seq_len: int, head_dim: int) -> tuple[torch.Tensor, int]:
    """``cos`` / ``sin`` as contiguous ``[B or 1, S, D]`` and the stride between batch rows (0 when shared)."""
    trig = trig.reshape(-1, seq_len, head_dim).contiguous()
    if trig.shape[0] not in (1, batch):
        raise ValueError(f"rotary table batch {trig.shape[0]} matches neither 1 nor the input batch {batch}")
    return trig, (seq_len * head_dim if trig.shape[0] == batch and batch > 1 else 0)


class FusedNormRope(torch.autograd.Function):
    """``apply_rotary(weight * rmsnorm(x))`` per head; ``x`` is ``[B, S, H, D]`` with unit stride along ``D``."""

    @staticmethod
    def forward(ctx, x, weight, cos, sin, eps: float):
        batch, seq_len, n_heads, head_dim = x.shape
        if head_dim % 2 or head_dim & (head_dim - 1):
            raise ValueError(f"head_dim must be a power of two, got {head_dim}")
        if x.stride(3) != 1 or x.stride(2) != head_dim or x.stride(0) != seq_len * x.stride(1):
            x = x.contiguous()
        cos, cos_batch_stride = _trig_rows(cos, batch, seq_len, head_dim)
        sin, _ = _trig_rows(sin, batch, seq_len, head_dim)
        out = torch.empty(batch, seq_len, n_heads, head_dim, device=x.device, dtype=x.dtype)
        rstd = torch.empty(batch * seq_len, n_heads, device=x.device, dtype=torch.float32)
        n_tokens = batch * seq_len
        if n_tokens:
            _norm_rope_fwd_kernel[(n_tokens,)](
                x,
                x.stride(1),
                weight,
                cos,
                sin,
                cos_batch_stride,
                out,
                rstd,
                seq_len,
                n_heads,
                eps,
                HALF=head_dim // 2,
                PAD_H=triton.next_power_of_2(n_heads),
                num_warps=_num_warps(n_heads, head_dim),
            )
        ctx.save_for_backward(x, weight, cos, sin, rstd)
        ctx.cos_batch_stride = cos_batch_stride
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, cos, sin, rstd = ctx.saved_tensors
        batch, seq_len, n_heads, head_dim = x.shape
        grad_out = grad_out.contiguous()
        n_tokens = batch * seq_len
        need_dw = ctx.needs_input_grad[1]
        dx = torch.empty(batch, seq_len, n_heads, head_dim, device=x.device, dtype=x.dtype)
        programs = triton.cdiv(n_tokens, _TOKENS_PER_BWD_PROGRAM)
        dw_partial = torch.empty(programs if need_dw else 1, head_dim, device=x.device, dtype=torch.float32)
        if n_tokens:
            _norm_rope_bwd_kernel[(programs,)](
                grad_out,
                x,
                x.stride(1),
                weight,
                cos,
                sin,
                ctx.cos_batch_stride,
                rstd,
                dx,
                dw_partial,
                n_tokens,
                seq_len,
                n_heads,
                HALF=head_dim // 2,
                PAD_H=triton.next_power_of_2(n_heads),
                TOKENS=_TOKENS_PER_BWD_PROGRAM,
                NEED_DW=need_dw,
                num_warps=_num_warps(n_heads, head_dim),
            )
        dw = None
        if need_dw:
            dw = (dw_partial.sum(0) if n_tokens else torch.zeros_like(dw_partial[0])).to(weight.dtype)
        return dx, dw, None, None, None


def fused_norm_rope(x, weight, cos, sin, eps: float):
    """Per-head ``weight * rmsnorm(x)`` rotated by ``cos`` / ``sin`` (transformers' ``rotate_half`` convention).

    ``x``: ``[B, S, H, D]``; ``cos`` / ``sin``: ``[B or 1, S, D]``. Returns a contiguous ``[B, S, H, D]`` tensor.
    """
    return FusedNormRope.apply(x, weight, cos, sin, float(eps))
