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
"""vLLM's ``fused_add_rms_norm`` with a (1 + weight) scale, in fp32, as one Triton pass each way.

``vllm_rounding.FusedAddRMSNorm`` writes it as a dozen PyTorch operations over ``[tokens, hidden]``: two fp32 upcasts,
the add, the square, the mean, the scale, and two casts back, and its backward about as many again. At 64K tokens per
GPU each is a full pass over the activations, twice per decoder layer. These kernels read the inputs once and write
the outputs once in each direction, with the same math: the sum of ``x`` and ``residual`` is formed in fp32, the norm
of that fp32 sum is computed in fp32 and the weight is applied as ``1 + weight`` in fp32. The backward recomputes the
sum from the saved inputs and the per-row ``rstd`` and reduces the weight gradient in fp32, per program and then over
programs. Only the reduction order of the row sums differs from PyTorch's, an fp32 rounding difference.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    y_ptr,
    s_ptr,
    rstd_ptr,
    x_stride,
    r_stride,
    y_stride,
    s_stride,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * r_stride + cols, mask=mask, other=0.0).to(tl.float32)
    s = x + r
    rstd = 1.0 / tl.sqrt(tl.sum(s * s, axis=0) / n_cols + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = s * rstd * (1.0 + w)
    tl.store(y_ptr + row * y_stride + cols, y.to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(s_ptr + row * s_stride + cols, s.to(s_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _bwd_kernel(
    dy_ptr,
    ds_ptr,
    x_ptr,
    r_ptr,
    w_ptr,
    rstd_ptr,
    dx_ptr,
    dr_ptr,
    dw_ptr,
    dy_stride,
    ds_stride,
    x_stride,
    r_stride,
    dx_stride,
    dr_stride,
    n_rows,
    n_cols,
    rows_per_program,
    HAS_DS: tl.constexpr,
    SEPARATE_DR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    scale = 1.0 + tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros((BLOCK,), dtype=tl.float32)
    row_start = program * rows_per_program
    row_end = tl.minimum(row_start + rows_per_program, n_rows)
    for row in range(row_start, row_end):
        x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + row * r_stride + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row)
        normed = (x + r) * rstd
        dy = tl.load(dy_ptr + row * dy_stride + cols, mask=mask, other=0.0).to(tl.float32)
        dw += dy * normed
        dn = dy * scale
        ds = rstd * (dn - normed * (tl.sum(dn * normed, axis=0) / n_cols))
        if HAS_DS:
            ds += tl.load(ds_ptr + row * ds_stride + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(dx_ptr + row * dx_stride + cols, ds.to(dx_ptr.dtype.element_ty), mask=mask)
        if SEPARATE_DR:
            tl.store(dr_ptr + row * dr_stride + cols, ds.to(dr_ptr.dtype.element_ty), mask=mask)
    tl.store(dw_ptr + program * n_cols + cols, dw, mask=mask)


def _block(n_cols: int) -> tuple[int, int]:
    block = triton.next_power_of_2(n_cols)
    return block, min(max(block // 256, 4), 16)


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _rows(t: torch.Tensor, n_cols: int) -> torch.Tensor:
    t = t.reshape(-1, n_cols)
    return t if t.stride(-1) == 1 else t.contiguous()


def supported(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether these kernels take the call: CUDA tensors of the 16- or 32-bit float types, one weight per channel."""
    floats = (torch.bfloat16, torch.float16, torch.float32)
    return (
        x.is_cuda
        and residual.is_cuda
        and weight.is_cuda
        and x.dtype in floats
        and residual.dtype in floats
        and weight.dtype in floats
        and weight.ndim == 1
        and x.shape == residual.shape
        and x.shape[-1] == weight.shape[0]
        and x.shape[-1] <= 65536
    )


def fused_add_rms_norm_fwd(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float, sum_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(normalized in x.dtype, fp32 sum rounded to sum_dtype, per-row rstd in fp32)``."""
    n_cols = x.shape[-1]
    x2 = _rows(x, n_cols)
    r2 = _rows(residual, n_cols)
    rows = x2.shape[0]
    y = torch.empty((rows, n_cols), dtype=x.dtype, device=x.device)
    s = torch.empty((rows, n_cols), dtype=sum_dtype, device=x.device)
    rstd = torch.empty((rows,), dtype=torch.float32, device=x.device)
    if rows > 0:
        block, warps = _block(n_cols)
        _fwd_kernel[(rows,)](
            x2,
            r2,
            weight,
            y,
            s,
            rstd,
            x2.stride(0),
            r2.stride(0),
            y.stride(0),
            s.stride(0),
            n_cols,
            eps,
            BLOCK=block,
            num_warps=warps,
        )
    return y.view(x.shape), s.view(x.shape), rstd


def fused_add_rms_norm_bwd(
    grad_out: torch.Tensor | None,
    grad_sum: torch.Tensor | None,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(grad x in x.dtype, grad residual in residual.dtype, grad weight in weight.dtype)``.

    The two input gradients are the same values; when the dtypes agree the same tensor is returned for both.
    """
    n_cols = x.shape[-1]
    x2 = _rows(x, n_cols)
    r2 = _rows(residual, n_cols)
    rows = x2.shape[0]
    dy = torch.zeros_like(x2) if grad_out is None else _rows(grad_out, n_cols)
    ds = dy if grad_sum is None else _rows(grad_sum, n_cols)
    dx = torch.empty((rows, n_cols), dtype=x.dtype, device=x.device)
    separate = residual.dtype != x.dtype
    dr = torch.empty((rows, n_cols), dtype=residual.dtype, device=x.device) if separate else dx
    programs = max(1, min(rows, _sm_count(x.device.index if x.device.index is not None else 0) * 4))
    rows_per_program = triton.cdiv(rows, programs) if rows > 0 else 1
    programs = triton.cdiv(rows, rows_per_program) if rows > 0 else 1
    dw_partial = torch.zeros((programs, n_cols), dtype=torch.float32, device=x.device)
    if rows > 0:
        block, warps = _block(n_cols)
        _bwd_kernel[(programs,)](
            dy,
            ds,
            x2,
            r2,
            weight,
            rstd,
            dx,
            dr,
            dw_partial,
            dy.stride(0),
            ds.stride(0),
            x2.stride(0),
            r2.stride(0),
            dx.stride(0),
            dr.stride(0),
            rows,
            n_cols,
            rows_per_program,
            HAS_DS=grad_sum is not None,
            SEPARATE_DR=separate,
            BLOCK=block,
            num_warps=warps,
        )
    grad_weight = dw_partial.sum(0).to(weight.dtype)
    return dx.view(x.shape), dr.view(residual.shape), grad_weight
