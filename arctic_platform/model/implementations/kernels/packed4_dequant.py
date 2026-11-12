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
"""Decode 4-bit packed weights (two codes per byte along the input dimension) to BF16 on the GPU.

Both layouts store a ``[rows, cols]`` matrix as ``[rows, cols // 2]`` bytes, the even column in the low nibble:

- **INT4, group-scaled** (W4AFP8 checkpoints, SGLang's ``w4afp8``): signed two's-complement nibbles, one BF16
  multiply-to-restore scale per ``group`` consecutive columns, ``[rows, cols // group]``.
- **NVFP4** (ModelOpt): E2M1 nibbles (sign, 2 exponent bits, 1 mantissa bit: 0, 0.5, 1, 1.5, 2, 3, 4, 6), one
  E4M3 scale per 16 columns, ``[rows, cols // 16]``, times one FP32 scale per matrix (``weight_scale_2``).
- **MXFP4** (OCP microscaling; DeepSeek-V4's routed experts): the same E2M1 nibbles in int8 bytes, one power-of-two
  scale per 32 columns, ``[rows, cols // 32]``: FP32, or the checkpoint's E8M0 exponent bytes (uint8, ``2^(e - 127)``).

Stacked expert weights add a leading expert dimension to every tensor (the NVFP4 per-matrix scale becomes ``[E]``).
"""

import torch
import triton
import triton.language as tl

_TILE_ROWS = 32
_TILE_BYTES = 128


@triton.jit
def _dequant_int4_group_kernel(
    code_ptr,
    scale_ptr,
    out_ptr,
    rows,
    packed_cols,
    scale_cols,
    out_rows,
    out_row0,
    GROUP: tl.constexpr,
    ALIGN: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    TILE_BYTES: tl.constexpr,
):
    expert = tl.program_id(2).to(tl.int64)
    row = (tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)).to(tl.int64)[:, None]
    byte = (tl.program_id(1) * TILE_BYTES + tl.arange(0, TILE_BYTES))[None, :]
    mask = (row < rows) & (byte < packed_cols)
    packed = tl.load(code_ptr + (expert * rows + row) * packed_cols + byte, mask=mask, other=0).to(tl.int32)
    # Sign-extend each nibble: the low one sits in bits 0-3, the high one in bits 4-7 of a sign-extended byte.
    low = (packed << 28) >> 28
    high = packed >> 4
    values = tl.interleave(low.to(tl.float32), high.to(tl.float32))
    cols = packed_cols * 2
    col = (tl.program_id(1) * (2 * TILE_BYTES) + tl.arange(0, 2 * TILE_BYTES))[None, :]
    out_mask = (row < rows) & (col < cols)
    group = col // GROUP
    if ALIGN > 0:
        # The W4A8 layout [E, groups // ALIGN, rows * ALIGN]: ALIGN consecutive groups of a row sit together.
        scale_offset = (expert * (scale_cols // ALIGN) + group // ALIGN) * (rows * ALIGN) + row * ALIGN + group % ALIGN
    else:
        scale_offset = (expert * rows + row) * scale_cols + group
    scale = tl.load(scale_ptr + scale_offset, mask=out_mask, other=0.0)
    out_offset = (expert * out_rows + out_row0 + row) * cols + col
    tl.store(out_ptr + out_offset, (values * scale.to(tl.float32)).to(tl.bfloat16), mask=out_mask)


@triton.jit
def _e2m1_to_float(nibble):
    # Magnitudes 2..7 are normal: 2^(e - 1) * (1 + m / 2) with e = mag >> 1; FP32 bits 0x3F000000 + (mag << 22).
    mag = nibble & 7
    normal = (0x3F000000 + (mag << 22)).to(tl.float32, bitcast=True)
    value = tl.where(mag == 1, 0.5, tl.where(mag == 0, 0.0, normal))
    return tl.where((nibble & 8) != 0, -value, value)


@triton.jit
def _dequant_nvfp4_kernel(
    code_ptr,
    scale_ptr,
    global_ptr,
    out_ptr,
    rows,
    packed_cols,
    scale_cols,
    out_rows,
    out_row0,
    TILE_ROWS: tl.constexpr,
    TILE_BYTES: tl.constexpr,
):
    expert = tl.program_id(2).to(tl.int64)
    row = (tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)).to(tl.int64)[:, None]
    byte = (tl.program_id(1) * TILE_BYTES + tl.arange(0, TILE_BYTES))[None, :]
    mask = (row < rows) & (byte < packed_cols)
    packed = tl.load(code_ptr + (expert * rows + row) * packed_cols + byte, mask=mask, other=0).to(tl.int32)
    values = tl.interleave(_e2m1_to_float(packed & 15), _e2m1_to_float((packed >> 4) & 15))
    cols = packed_cols * 2
    col = (tl.program_id(1) * (2 * TILE_BYTES) + tl.arange(0, 2 * TILE_BYTES))[None, :]
    out_mask = (row < rows) & (col < cols)
    scale = tl.load(scale_ptr + (expert * rows + row) * scale_cols + col // 16, mask=out_mask, other=0.0)
    global_scale = tl.load(global_ptr + expert)
    out_offset = (expert * out_rows + out_row0 + row) * cols + col
    restored = values * (scale.to(tl.float32) * global_scale)
    tl.store(out_ptr + out_offset, restored.to(tl.bfloat16), mask=out_mask)


@triton.jit
def _dequant_mxfp4_kernel(
    code_ptr,
    scale_ptr,
    out_ptr,
    rows,
    packed_cols,
    scale_cols,
    out_rows,
    out_row0,
    E8M0: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    TILE_BYTES: tl.constexpr,
):
    expert = tl.program_id(2).to(tl.int64)
    row = (tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)).to(tl.int64)[:, None]
    byte = (tl.program_id(1) * TILE_BYTES + tl.arange(0, TILE_BYTES))[None, :]
    mask = (row < rows) & (byte < packed_cols)
    packed = tl.load(code_ptr + (expert * rows + row) * packed_cols + byte, mask=mask, other=0).to(tl.int32)
    values = tl.interleave(_e2m1_to_float(packed & 15), _e2m1_to_float((packed >> 4) & 15))
    cols = packed_cols * 2
    col = (tl.program_id(1) * (2 * TILE_BYTES) + tl.arange(0, 2 * TILE_BYTES))[None, :]
    out_mask = (row < rows) & (col < cols)
    scale_offset = (expert * rows + row) * scale_cols + col // 32
    if E8M0:
        exponent = tl.load(scale_ptr + scale_offset, mask=out_mask, other=127).to(tl.int32)
        # 2^(e - 127) is the FP32 with exponent field e; e = 0 is the subnormal 2^-127 (0x00400000).
        scale = tl.where(exponent == 0, 0x00400000, exponent << 23).to(tl.float32, bitcast=True)
    else:
        scale = tl.load(scale_ptr + scale_offset, mask=out_mask, other=0.0)
    out_offset = (expert * out_rows + out_row0 + row) * cols + col
    tl.store(out_ptr + out_offset, (values * scale).to(tl.bfloat16), mask=out_mask)


def _checked_out(weight: torch.Tensor, out: torch.Tensor | None, out_row: int) -> torch.Tensor:
    rows, cols = weight.shape[-2], weight.shape[-1] * 2
    if out is None:
        return torch.empty((*weight.shape[:-2], rows, cols), dtype=torch.bfloat16, device=weight.device)
    if (
        out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or out.shape[:-2] != weight.shape[:-2]
        or out.shape[-1] != cols
        or out.shape[-2] < out_row + rows
    ):
        raise ValueError(
            f"out {out.dtype} {tuple(out.shape)} cannot hold rows {out_row}..{out_row + rows} of packed "
            f"{tuple(weight.shape)}"
        )
    return out


def interleave_int4_scales(scale: torch.Tensor, align: int) -> torch.Tensor:
    """``[E, rows, groups]`` -> the W4A8 layout ``[E, groups // align, rows * align]`` (SGLang's ``interleave_scales``;
    ``align`` is the GEMM tile's K step over the group size, 1 for the training tiles)."""
    experts, rows, groups = scale.shape
    if groups % align:
        raise ValueError(f"interleaving INT4 scales by {align} needs a multiple of {align} groups, got {groups}")
    return (
        scale.reshape(experts, rows, groups // align, align)
        .permute(0, 2, 1, 3)
        .reshape(experts, groups // align, rows * align)
    )


def dequantize_int4_groups(
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor | None = None,
    out_row: int = 0,
    interleaved: int = 0,
) -> torch.Tensor:
    """BF16 copy of a CUDA group-scaled INT4 matrix ``[rows, cols // 2]`` (int8 bytes), or of an expert stack.

    ``scale`` is BF16 ``[.., rows, cols // group]``; the group size follows from the shapes. ``interleaved=a`` (> 0) reads
    an expert stack's scales in the W4A8 kernel's ``[E, groups // a, rows * a]`` layout instead. With ``out`` (BF16, same
    leading and column sizes, at least ``out_row + rows`` rows) the rows are written into
    ``out[..., out_row : out_row + rows, :]``, so several stacks can be decoded side by side for one GEMM.
    """
    if weight.dtype != torch.int8 or weight.ndim not in (2, 3) or not weight.is_cuda:
        raise TypeError(
            f"expected a CUDA int8 [.., rows, cols // 2] packed INT4 weight, got {weight.dtype} {weight.shape}"
        )
    rows, cols = weight.shape[-2], weight.shape[-1] * 2
    if interleaved:
        a = interleaved
        if weight.ndim != 3 or scale.ndim != 3 or scale.shape[0] != weight.shape[0] or scale.shape[2] != rows * a:
            raise TypeError(
                f"interleaved INT4 scales must be [E, groups // {a}, {rows * a}], got {tuple(scale.shape)}"
            )
        groups = scale.shape[1] * a
    else:
        if scale.shape[:-1] != weight.shape[:-1]:
            raise TypeError(f"INT4 group scale must be [.., {rows}, groups], got {tuple(scale.shape)}")
        groups = scale.shape[-1]
    if scale.dtype != torch.bfloat16 or cols % groups:
        raise TypeError(
            f"INT4 group scale must be bfloat16 with groups dividing {cols}, got {scale.dtype} {tuple(scale.shape)}"
        )
    group = cols // groups
    if group % (2 * _TILE_BYTES) and (2 * _TILE_BYTES) % group:
        raise ValueError(f"INT4 group size {group} does not tile the decode's {2 * _TILE_BYTES} columns")
    out = _checked_out(weight, out, out_row)
    experts = weight.shape[0] if weight.ndim == 3 else 1
    if weight.numel():
        grid = (triton.cdiv(rows, _TILE_ROWS), triton.cdiv(weight.shape[-1], _TILE_BYTES), experts)
        with torch.cuda.device(weight.device):
            _dequant_int4_group_kernel[grid](
                weight.contiguous(),
                scale.contiguous(),
                out,
                rows,
                weight.shape[-1],
                groups,
                out.shape[-2],
                out_row,
                GROUP=group,
                ALIGN=interleaved,
                TILE_ROWS=_TILE_ROWS,
                TILE_BYTES=_TILE_BYTES,
            )
    return out


def dequantize_nvfp4(
    weight: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    out: torch.Tensor | None = None,
    out_row: int = 0,
) -> torch.Tensor:
    """BF16 copy of a CUDA NVFP4 matrix ``[rows, cols // 2]`` (uint8 bytes), or of an expert stack.

    ``scale`` is E4M3 ``[.., rows, cols // 16]`` and ``global_scale`` FP32, one per matrix (``[E]`` for a stack).
    ``out`` and ``out_row`` as in ``dequantize_int4_groups``.
    """
    if weight.dtype != torch.uint8 or weight.ndim not in (2, 3) or not weight.is_cuda:
        raise TypeError(f"expected a CUDA uint8 [.., rows, cols // 2] NVFP4 weight, got {weight.dtype} {weight.shape}")
    rows, cols = weight.shape[-2], weight.shape[-1] * 2
    if scale.dtype != torch.float8_e4m3fn or tuple(scale.shape) != (*weight.shape[:-1], cols // 16):
        raise TypeError(
            f"NVFP4 block scale must be float8_e4m3fn [.., {rows}, {cols // 16}], got"
            f" {scale.dtype} {tuple(scale.shape)}"
        )
    experts = weight.shape[0] if weight.ndim == 3 else 1
    if global_scale.dtype != torch.float32 or global_scale.numel() != experts:
        raise TypeError(
            f"NVFP4 global scale must be float32 with {experts} values, got"
            f" {global_scale.dtype} {tuple(global_scale.shape)}"
        )
    out = _checked_out(weight, out, out_row)
    if weight.numel():
        grid = (triton.cdiv(rows, _TILE_ROWS), triton.cdiv(weight.shape[-1], _TILE_BYTES), experts)
        with torch.cuda.device(weight.device):
            _dequant_nvfp4_kernel[grid](
                weight.contiguous(),
                scale.contiguous(),
                global_scale.contiguous(),
                out,
                rows,
                weight.shape[-1],
                scale.shape[-1],
                out.shape[-2],
                out_row,
                TILE_ROWS=_TILE_ROWS,
                TILE_BYTES=_TILE_BYTES,
            )
    return out


MXFP4_SCALE_DTYPES = (torch.float32, torch.uint8)


def dequantize_mxfp4_packed(
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor | None = None,
    out_row: int = 0,
) -> torch.Tensor:
    """BF16 copy of a CUDA MXFP4 matrix ``[rows, cols // 2]`` (int8 bytes), or of an expert stack.

    ``scale`` is ``[.., rows, cols // 32]``: FP32 values or E8M0 exponent bytes (uint8). ``out`` and ``out_row`` as in
    ``dequantize_int4_groups``.
    """
    if weight.dtype != torch.int8 or weight.ndim not in (2, 3) or not weight.is_cuda:
        raise TypeError(f"expected a CUDA int8 [.., rows, cols // 2] MXFP4 weight, got {weight.dtype} {weight.shape}")
    rows, cols = weight.shape[-2], weight.shape[-1] * 2
    if scale.dtype not in MXFP4_SCALE_DTYPES or tuple(scale.shape) != (*weight.shape[:-1], cols // 32):
        raise TypeError(
            f"MXFP4 scale must be float32 or uint8 (E8M0) [.., {rows}, {cols // 32}], got"
            f" {scale.dtype} {tuple(scale.shape)}"
        )
    out = _checked_out(weight, out, out_row)
    experts = weight.shape[0] if weight.ndim == 3 else 1
    if weight.numel():
        grid = (triton.cdiv(rows, _TILE_ROWS), triton.cdiv(weight.shape[-1], _TILE_BYTES), experts)
        with torch.cuda.device(weight.device):
            _dequant_mxfp4_kernel[grid](
                weight.contiguous(),
                scale.contiguous(),
                out,
                rows,
                weight.shape[-1],
                scale.shape[-1],
                out.shape[-2],
                out_row,
                E8M0=scale.dtype == torch.uint8,
                TILE_ROWS=_TILE_ROWS,
                TILE_BYTES=_TILE_BYTES,
            )
    return out


def dequantize_int4_groups_reference(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP32 ``unpack(weight) * scale`` in plain PyTorch (CPU or GPU), for tests and the CPU path."""
    packed = weight.to(torch.int16)
    low = ((packed & 0x0F) ^ 0x08) - 0x08
    high = ((packed >> 4) & 0x0F ^ 0x08) - 0x08
    values = torch.stack((low, high), dim=-1).flatten(-2).float()
    group = values.shape[-1] // scale.shape[-1]
    return values * scale.float().repeat_interleave(group, dim=-1)


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequantize_nvfp4_reference(weight: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """FP32 NVFP4 decode in plain PyTorch (CPU or GPU), for tests and the CPU path."""
    codes = torch.stack((weight & 0x0F, weight >> 4), dim=-1).flatten(-2).long()
    table = torch.tensor(_E2M1, dtype=torch.float32, device=weight.device)
    values = table[codes & 7] * torch.where((codes & 8) != 0, -1.0, 1.0)
    per_matrix = global_scale.float().reshape(-1, 1, 1) if weight.ndim == 3 else global_scale.float().reshape(())
    return values * scale.float().repeat_interleave(16, dim=-1) * per_matrix


def mxfp4_scale_values(scale: torch.Tensor) -> torch.Tensor:
    """FP32 values of MXFP4 scales stored as FP32 or as E8M0 exponent bytes (uint8)."""
    if scale.dtype == torch.uint8:
        return scale.view(torch.float8_e8m0fnu).float()
    return scale.float()


def dequantize_mxfp4_reference(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP32 MXFP4 decode in plain PyTorch (CPU or GPU), for tests and the CPU path."""
    codes = weight.view(torch.uint8)
    codes = torch.stack((codes & 0x0F, codes >> 4), dim=-1).flatten(-2).long()
    table = torch.tensor(_E2M1, dtype=torch.float32, device=weight.device)
    values = table[codes & 7] * torch.where((codes & 8) != 0, -1.0, 1.0)
    return values * mxfp4_scale_values(scale).repeat_interleave(32, dim=-1)
