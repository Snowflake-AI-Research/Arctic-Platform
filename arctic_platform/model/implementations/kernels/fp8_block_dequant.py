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
"""Decode block-scaled FP8 weights (HF fine-grained FP8) to BF16 on the GPU.

A matrix ``[rows, cols]`` of ``float8_e4m3fn`` codes carries one FP32
multiply-to-restore scale (``weight_scale_inv``) per ``block x block`` tile,
``[ceil(rows / block), ceil(cols / block)]``. Stacked expert weights add a
leading expert dimension to both.
"""

import torch
import triton
import triton.language as tl

_TILE_ROWS = 32
_TILE_COLS = 256


@triton.jit
def _dequant_fp8_block_kernel(
    code_ptr,
    scale_ptr,
    out_ptr,
    rows,
    cols,
    scale_rows,
    scale_cols,
    out_rows,
    out_row0,
    BLOCK: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    TILE_COLS: tl.constexpr,
):
    expert = tl.program_id(2).to(tl.int64)
    row = (tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)).to(tl.int64)[:, None]
    col = (tl.program_id(1) * TILE_COLS + tl.arange(0, TILE_COLS))[None, :]
    mask = (row < rows) & (col < cols)
    offset = (expert * rows + row) * cols + col
    code = tl.load(code_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    scale_offset = (expert * scale_rows + row // BLOCK) * scale_cols + col // BLOCK
    scale = tl.load(scale_ptr + scale_offset, mask=mask, other=0.0)
    out_offset = (expert * out_rows + out_row0 + row) * cols + col
    tl.store(out_ptr + out_offset, (code * scale).to(tl.bfloat16), mask=mask)


def dequantize_fp8_blocks(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 128,
    out: torch.Tensor | None = None,
    out_row: int = 0,
) -> torch.Tensor:
    """BF16 copy of one CUDA block-FP8 matrix, or of a stack of expert matrices.

    With ``out`` (BF16, same leading and column sizes, at least ``out_row + rows`` rows) the rows are written into
    ``out[..., out_row : out_row + rows, :]`` instead, so several stacks can be decoded side by side for one GEMM.
    """
    if weight.dtype != torch.float8_e4m3fn or weight.ndim not in (2, 3) or not weight.is_cuda:
        raise TypeError(f"expected a CUDA float8_e4m3fn [.., rows, cols] weight, got {weight.dtype} {weight.shape}")
    rows, cols = weight.shape[-2:]
    expected = (*weight.shape[:-2], triton.cdiv(rows, block_size), triton.cdiv(cols, block_size))
    if scale.dtype != torch.float32 or tuple(scale.shape) != expected:
        raise TypeError(f"FP8 block scale must be float32 {expected}, got {scale.dtype} {tuple(scale.shape)}")
    experts = weight.shape[0] if weight.ndim == 3 else 1
    if out is None:
        out = torch.empty(weight.shape, dtype=torch.bfloat16, device=weight.device)
    elif (
        out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or out.shape[:-2] != weight.shape[:-2]
        or out.shape[-1] != cols
        or out.shape[-2] < out_row + rows
    ):
        raise ValueError(f"out {out.dtype} {tuple(out.shape)} cannot hold rows {out_row}..{out_row + rows} of {tuple(weight.shape)}")
    if weight.numel():
        grid = (triton.cdiv(rows, _TILE_ROWS), triton.cdiv(cols, _TILE_COLS), experts)
        with torch.cuda.device(weight.device):
            _dequant_fp8_block_kernel[grid](
                weight.contiguous(),
                scale.contiguous(),
                out,
                rows,
                cols,
                expected[-2],
                expected[-1],
                out.shape[-2],
                out_row,
                BLOCK=block_size,
                TILE_ROWS=_TILE_ROWS,
                TILE_COLS=_TILE_COLS,
            )
    return out
