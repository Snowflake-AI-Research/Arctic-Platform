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
"""Forward GEMMs of frozen block-FP8 routed experts as W8A8 on Hopper, through DeepGEMM (``moe.w8a8_experts``).

The FP8 checkpoints these experts load (GLM-5.3's release: ``activation_scheme: dynamic``) are published for this
arithmetic: E4M3 weights with FP32 128 x 128 block scales, activations quantized to E4M3 per token per 128 inputs
on the fly. DeepGEMM's m-grouped contiguous GEMM reads each expert's rows from a 128-row boundary. ``padded_layout``
places the routed rows (packed by expert) on those boundaries without a host sync, with the expert of each padded row
(-1 for padding; DeepGEMM skips 128-row blocks that start with one). Its per-expert "psum" layout gives the same
outputs but runs the gate projection ~9% slower on H200 at GLM-5.3's shapes. The activations are quantized straight
into the padded layout (scales in DeepGEMM's MN-major TMA-aligned layout) and one gather per GEMM brings the output
back to the routed rows, adding the expert LoRA delta or applying the routing weights on the way.
The GEMMs read the experts' own codes and scales in place: a frozen expert offload's per-call copy or UltraEP's
native + spare stacks.

Padding rows of the quantized input are left unwritten and their GEMM outputs are never read: a GEMM output row
depends only on its own input row.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

W8A8_EXPERTS_ENV = "ARL_W8A8_EXPERTS"
_FP8_MAX = 448.0
_EPS = 1e-10  # the 1 x 128 activation quantization's amax floor (kernels/fp8_quant.py)
_GROUP = 128
_QUANT_BLOCK_M = 32
_UNPAD_BLOCK_M = 8
_UNPAD_BLOCK_N = 512


_configured: bool | None = None  # loader option moe.w8a8_experts (moe/runtime_options.py); None: the env


def configure_from_options(value: bool | None) -> None:
    global _configured
    _configured = value


def w8a8_experts_requested() -> bool:
    if _configured is not None:
        return _configured
    return os.environ.get(W8A8_EXPERTS_ENV, "0") not in ("", "0", "false", "False")


def check_w8a8_device(device: torch.device) -> None:
    """DeepGEMM reads FP32 activation and weight scales only on SM90; elsewhere it would round them to UE8M0."""
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] != 9:
        raise NotImplementedError(f"moe.w8a8_experts runs DeepGEMM's SM90 (Hopper) FP8 kernels, not on {device}")


@dataclass(frozen=True)
class PaddedLayout:
    """Where the routed rows sit in DeepGEMM's padded grouped layout."""

    dst: torch.Tensor  # [rows] int32: the padded row of each routed row
    m_indices: torch.Tensor  # [padded_rows] int32: the expert of each padded row, -1 for padding
    rows: int
    padded_rows: int


def mk_alignment() -> int:
    from arctic_platform.model.implementations.fp8 import _require_deep_gemm

    return int(_require_deep_gemm().get_mk_alignment_for_contiguous_layout())


def padded_layout(offsets: torch.Tensor, rows: int) -> PaddedLayout:
    """Layout of ``rows`` routed rows packed by expert, ``offsets`` their cumulative count per expert.

    Sized for the worst case (``align - 1`` padding rows per expert), so it needs no host sync.
    """
    align = mk_alignment()
    experts = offsets.numel()
    ends = offsets.to(torch.int64)
    counts = torch.diff(ends, prepend=ends.new_zeros(1))
    padded = (counts + align - 1) // align * align
    shift = torch.cumsum(padded, 0) - padded - (ends - counts)
    padded_rows = triton.cdiv(rows + experts * (align - 1), align) * align
    m_indices = torch.full((padded_rows,), -1, dtype=torch.int32, device=offsets.device)
    dst = torch.arange(rows, device=offsets.device, dtype=torch.int64)
    if rows:
        expert = torch.repeat_interleave(
            torch.arange(experts, device=offsets.device, dtype=torch.int64), counts, output_size=rows
        )
        dst += shift[expert]
        m_indices.scatter_(0, dst, expert.to(torch.int32))
    return PaddedLayout(dst=dst.to(torch.int32), m_indices=m_indices, rows=rows, padded_rows=padded_rows)


@triton.jit
def _quant_into_padded_kernel(
    x_ptr,
    dst_ptr,
    q_ptr,
    s_ptr,
    rows,
    stride_x,
    hidden,
    cols,
    padded_rows,
    limit,
    eps,
    fp8_max,
    SWIGLU: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP: tl.constexpr,
):
    # One [BLOCK_M, GROUP] tile of routed rows per program, written to their padded rows; the BLOCK_M scales of one
    # group sit together in the MN-major layout.
    row = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    group = tl.program_id(1)
    col = group * GROUP + tl.arange(0, GROUP)
    mask = row < rows
    ptrs = x_ptr + row[:, None] * stride_x + col[None, :]
    if SWIGLU:
        g = tl.load(ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
        u = tl.load(ptrs + hidden, mask=mask[:, None], other=0.0).to(tl.float32)
        if HAS_LIMIT:
            g = tl.minimum(g, limit)
            u = tl.minimum(tl.maximum(u, -limit), limit)
        # The BF16 SwiGLU output the decoded path's GEMM reads, then its quantization.
        x = (g * tl.sigmoid(g) * u).to(tl.bfloat16).to(tl.float32)
    else:
        x = tl.load(ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), axis=1), eps) / fp8_max
    q = tl.clamp(x / scale[:, None], -fp8_max, fp8_max).to(q_ptr.dtype.element_ty)
    dst = tl.load(dst_ptr + row, mask=mask, other=0).to(tl.int64)
    tl.store(q_ptr + dst[:, None] * cols + col[None, :], q, mask=mask[:, None])
    tl.store(s_ptr + group * padded_rows + dst, scale, mask=mask)


def _quantize(x: torch.Tensor, layout: PaddedLayout, cols: int, swiglu: bool, limit: float | None):
    q = torch.empty(layout.padded_rows, cols, dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(cols // _GROUP, layout.padded_rows, dtype=torch.float32, device=x.device)
    if layout.rows:
        if x.stride(-1) != 1:
            x = x.contiguous()
        _quant_into_padded_kernel[(triton.cdiv(layout.rows, _QUANT_BLOCK_M), cols // _GROUP)](
            x,
            layout.dst,
            q,
            s,
            layout.rows,
            x.stride(0),
            cols,
            cols,
            layout.padded_rows,
            0.0 if limit is None else float(limit),
            _EPS,
            _FP8_MAX,
            SWIGLU=swiglu,
            HAS_LIMIT=limit is not None,
            BLOCK_M=_QUANT_BLOCK_M,
            GROUP=_GROUP,
            num_warps=4,
        )
    # DeepGEMM's SM90 activation scales: [rows, groups] FP32, MN-major (padded_rows is a multiple of 128, so aligned).
    return q, s.t()


def quantize_into_padded(x: torch.Tensor, layout: PaddedLayout) -> tuple[torch.Tensor, torch.Tensor]:
    """The routed rows of ``x`` quantized to E4M3 per 128 inputs into ``layout``: ``(codes, scales)`` for DeepGEMM."""
    if x.shape[-1] % _GROUP:
        raise ValueError(f"W8A8 activations need a multiple of {_GROUP} columns, got {tuple(x.shape)}")
    return _quantize(x, layout, x.shape[-1], swiglu=False, limit=None)


def swiglu_quantize_into_padded(
    gate_up: torch.Tensor, layout: PaddedLayout, limit: float | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``quantize_into_padded`` of ``swiglu(gate, up)`` for packed ``[rows, 2 * hidden]`` gate|up rows."""
    hidden = gate_up.shape[-1] // 2
    if hidden % _GROUP:
        raise ValueError(f"W8A8 SwiGLU needs a multiple of {_GROUP} hidden columns, got {tuple(gate_up.shape)}")
    return _quantize(gate_up, layout, hidden, swiglu=True, limit=limit)


def grouped_gemm_padded(
    q: torch.Tensor, q_scale: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, layout: PaddedLayout
) -> torch.Tensor:
    """DeepGEMM ``m_grouped_fp8_gemm_nt_contiguous`` against ``[experts, out, in]`` block-FP8 ``weight`` with its
    ``[experts, out / 128, in / 128]`` FP32 scales: ``[padded rows, out]`` BF16, padding rows unspecified."""
    from arctic_platform.model.implementations.fp8 import _disable_ue8m0_cast
    from arctic_platform.model.implementations.fp8 import _require_deep_gemm
    from arctic_platform.model.implementations.fp8 import _require_fp8_state

    _require_fp8_state(weight, scale, "W8A8 expert")
    out = torch.empty(layout.padded_rows, weight.shape[-2], dtype=torch.bfloat16, device=q.device)
    if layout.rows:
        _require_deep_gemm().m_grouped_fp8_gemm_nt_contiguous(
            (q, q_scale), (weight, scale), out, layout.m_indices, disable_ue8m0_cast=_disable_ue8m0_cast()
        )
    return out


@triton.jit
def _unpad_rows_kernel(
    y_ptr,
    dst_ptr,
    delta_ptr,
    s_ptr,
    out_ptr,
    rows,
    cols,
    stride_y,
    stride_delta,
    stride_out,
    HAS_DELTA: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (row < rows)[:, None] & (col < cols)[None, :]
    src = tl.load(dst_ptr + row, mask=row < rows, other=0).to(tl.int64)
    y = tl.load(y_ptr + src[:, None] * stride_y + col[None, :], mask=mask, other=0.0).to(tl.float32)
    if HAS_DELTA:
        y += tl.load(delta_ptr + row[:, None] * stride_delta + col[None, :], mask=mask, other=0.0).to(tl.float32)
    if HAS_SCALE:
        y *= tl.load(s_ptr + row, mask=row < rows, other=0.0).to(tl.float32)[:, None]
    tl.store(out_ptr + row[:, None] * stride_out + col[None, :], y.to(out_ptr.dtype.element_ty), mask=mask)


def unpad_rows_into(
    out: torch.Tensor,
    y: torch.Tensor,
    layout: PaddedLayout,
    delta: torch.Tensor | None = None,
    scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """``out[:rows] = (y[dst] + delta) * scores[:, None]`` (FP32 math, one rounding) for padded ``y``; ``out`` may be a
    column slice. Rows of ``out`` past the routed rows are left as they are."""
    rows, cols = layout.rows, y.shape[1]
    if out.shape[1] != cols or out.shape[0] < rows or out.stride(-1) != 1:
        raise ValueError(f"out {tuple(out.shape)} {out.stride()} for {rows} rows of {cols} columns")
    if delta is not None and (delta.shape != (rows, cols) or delta.stride(-1) != 1):
        raise ValueError(f"LoRA delta {tuple(delta.shape)} {delta.stride()} for {rows} rows of {cols} columns")
    if rows:
        _unpad_rows_kernel[(triton.cdiv(rows, _UNPAD_BLOCK_M), triton.cdiv(cols, _UNPAD_BLOCK_N))](
            y,
            layout.dst,
            y if delta is None else delta,
            layout.dst if scores is None else scores.reshape(-1),
            out,
            rows,
            cols,
            y.stride(0),
            0 if delta is None else delta.stride(0),
            out.stride(0),
            HAS_DELTA=delta is not None,
            HAS_SCALE=scores is not None,
            BLOCK_M=_UNPAD_BLOCK_M,
            BLOCK_N=_UNPAD_BLOCK_N,
            num_warps=4,
        )
    return out
