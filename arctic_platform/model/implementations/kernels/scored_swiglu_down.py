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
"""Routed-expert SwiGLU and frozen down projection with the routing weights folded in: ``s * (swiglu(g, u) @ W2.T)``.

Scaling the expert outputs afterwards (``apply_routing_weights``) keeps each ``[rows, dim]`` output alive for the
score gradient ``<grad, y>``: at 256K tokens over 8 GPUs and top-8 routing, 1 GiB per MoE layer. With ``W2`` frozen
the same gradient is ``<grad @ W2, h>``, and backward computes ``grad @ W2`` for the input gradient anyway; ``h`` is
rebuilt from the gate and up projections that SwiGLU keeps regardless. So only the gate and up projections and the
scores are saved, and one Triton pass per row turns ``grad @ W2`` into the gate, up and score gradients. The forward
output is the same as the unfused path's: the same GEMM on the same BF16 ``h``, then the same per-row scaling.

Gate and up arrive packed in one ``[rows, 2 * hidden]`` tensor, the output of one grouped GEMM against the stacked
gate|up weights, and their gradient leaves in the same layout, so the input gradient is one GEMM too.
"""

import torch
import triton
import triton.language as tl

from arctic_platform.model.implementations.kernels.fused_swiglu import _packed_swiglu_fwd_kernel
from arctic_platform.model.implementations.moe.routing_weights import scale_rows

_BLOCK_D = 1024


@triton.jit
def _scored_swiglu_bwd_kernel(
    gate_ptr,
    up_ptr,
    gh_ptr,
    s_ptr,
    dgate_ptr,
    dup_ptr,
    ds_ptr,
    D,
    stride_gu,
    stride_gh,
    limit,
    HAS_LIMIT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # gh = grad @ W2, the gradient of the unscored h: dh = s * gh, ds = <gh, h>.
    row = tl.program_id(0).to(tl.int64)
    scale = tl.load(s_ptr + row).to(tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)
        mask = cols < D
        g = tl.load(gate_ptr + row * stride_gu + cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + row * stride_gu + cols, mask=mask, other=0.0).to(tl.float32)
        gh = tl.load(gh_ptr + row * stride_gh + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_LIMIT:
            pass_g = g <= limit
            pass_u = (u >= -limit) & (u <= limit)
            g = tl.minimum(g, limit)
            u = tl.minimum(tl.maximum(u, -limit), limit)
        sig = tl.sigmoid(g)
        # The forward GEMM read h rounded to the activation dtype; so does its score gradient.
        h = (g * sig * u).to(gate_ptr.dtype.element_ty).to(tl.float32)
        acc += gh * h
        dh = gh * scale
        dg = dh * u * sig * (1.0 + g * (1.0 - sig))
        du = dh * g * sig
        if HAS_LIMIT:
            dg = tl.where(pass_g, dg, 0.0)
            du = tl.where(pass_u, du, 0.0)
        tl.store(dgate_ptr + row * stride_gu + cols, dg.to(dgate_ptr.dtype.element_ty), mask=mask)
        tl.store(dup_ptr + row * stride_gu + cols, du.to(dup_ptr.dtype.element_ty), mask=mask)
    tl.store(ds_ptr + row, tl.sum(acc, axis=0))


def _down_weight(weight: torch.Tensor, scale: torch.Tensor | None, block_size: int | None) -> torch.Tensor:
    if scale is None:
        return weight.bfloat16()
    if block_size is None:
        raise ValueError("block FP8 down-projection decode requires block_size")
    from arctic_platform.model.implementations.fp8 import _decode_fp8_to_bf16

    return _decode_fp8_to_bf16(weight, scale, block_size)


def _w8a8_forward(gate_up, scores, weight, scale, rows, limit, layout):
    """SwiGLU quantized straight into the padded layout, DeepGEMM, then one gather back that applies the scores."""
    from arctic_platform.model.implementations.kernels.fp8_grouped_w8a8 import grouped_gemm_padded
    from arctic_platform.model.implementations.kernels.fp8_grouped_w8a8 import swiglu_quantize_into_padded
    from arctic_platform.model.implementations.kernels.fp8_grouped_w8a8 import unpad_rows_into

    q, q_scale = swiglu_quantize_into_padded(gate_up[:rows], layout, limit)
    y = grouped_gemm_padded(q, q_scale, weight, scale, layout)
    del q, q_scale
    out = torch.empty(gate_up.shape[0], weight.shape[-2], dtype=gate_up.dtype, device=gate_up.device)
    unpad_rows_into(out, y, layout, scores=scores.reshape(-1)[:rows])
    if rows < out.shape[0]:
        out[rows:].zero_()
    return out


class _ScoredSwiGLUDownFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_up, scores, weight, scale, offsets, rows, block_size, limit, layout):
        gate_up = gate_up.contiguous()
        hidden = gate_up.shape[1] // 2
        from arctic_platform.model.implementations.fp8 import stash_frozen

        ctx.rows, ctx.block_size, ctx.limit = rows, block_size, limit
        ctx.save_for_backward(gate_up, scores, offsets, *stash_frozen(ctx, weight, scale))
        if layout is not None:
            return _w8a8_forward(gate_up, scores, weight, scale, rows, limit, layout)
        h = torch.empty(gate_up.shape[0], hidden, dtype=gate_up.dtype, device=gate_up.device)
        if h.numel():
            _packed_swiglu_fwd_kernel[(h.shape[0],)](
                gate_up,
                h,
                hidden,
                gate_up.stride(0),
                0.0 if limit is None else float(limit),
                HAS_LIMIT=limit is not None,
                BLOCK_D=_BLOCK_D,
            )
        y = torch._grouped_mm(h, _down_weight(weight, scale, block_size).transpose(-2, -1), offs=offsets)
        del h
        if rows < y.shape[0]:
            y[rows:].zero_()
        return scale_rows(y, scores, y.dtype)

    @staticmethod
    def backward(ctx, grad):
        from arctic_platform.model.implementations.fp8 import unstash_frozen

        (weight, scale), (gate_up, scores, offsets) = unstash_frozen(ctx, ctx.saved_tensors)
        rows = ctx.rows
        hidden = gate_up.shape[1] // 2
        grad_h = torch._grouped_mm(
            grad.contiguous().to(gate_up.dtype), _down_weight(weight, scale, ctx.block_size), offs=offsets
        )
        d_gate_up = torch.empty_like(gate_up)
        dscores = torch.empty(scores.shape[0], dtype=torch.float32, device=gate_up.device)
        if rows:
            limit = ctx.limit
            _scored_swiglu_bwd_kernel[(rows,)](
                gate_up,
                gate_up[:, hidden:],
                grad_h,
                scores.reshape(-1).contiguous(),
                d_gate_up,
                d_gate_up[:, hidden:],
                dscores,
                hidden,
                gate_up.stride(0),
                grad_h.stride(0),
                0.0 if limit is None else float(limit),
                HAS_LIMIT=limit is not None,
                BLOCK_D=_BLOCK_D,
            )
        if rows < gate_up.shape[0]:
            d_gate_up[rows:].zero_()
            dscores[rows:].zero_()
        return d_gate_up, dscores.reshape(scores.shape).to(scores.dtype), None, None, None, None, None, None, None


def scored_swiglu_down(
    gate_up: torch.Tensor,
    scores: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    offsets: torch.Tensor,
    rows: int,
    block_size: int | None = None,
    limit: float | None = None,
    layout=None,
) -> torch.Tensor:
    """``scores[:, None] * (swiglu(gate, up) @ W2_e.T)`` per expert for rows packed by expert, ``W2`` frozen.

    ``gate_up`` is ``[rows, 2 * hidden]``: gate in the first half of each row, up in the second. ``weight`` is the
    ``[experts, dim, hidden]`` down-projection stack, BF16 or block FP8 with ``scale`` (decoded per GEMM).
    ``offsets`` is the cumulative row count per expert and ``rows`` its last value; rows past it are padding and
    come out zero. Gradients flow to ``gate_up`` (same layout) and ``scores``. With ``layout``
    (``kernels.fp8_grouped_w8a8.padded_layout``) the forward GEMM runs block-FP8 W8A8 on SwiGLU quantized per token
    per 128 inputs; the backward is unchanged (straight through the quantization).
    """
    if gate_up.ndim != 2 or gate_up.shape[1] % 2 or scores.numel() != gate_up.shape[0]:
        raise ValueError(
            f"gate_up {tuple(gate_up.shape)} must be [rows, 2 * hidden] with one score per row, got {scores.numel()}"
        )
    if layout is not None and scale is None:
        raise ValueError("a W8A8 down projection needs block-FP8 weights")
    return _ScoredSwiGLUDownFn.apply(gate_up, scores, weight, scale, offsets, rows, block_size, limit, layout)
