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
"""Packed MXFP4 (two e2m1 nibbles per byte, one power-of-two scale per 32 values) to BF16 in one pass.

The routed experts of the official DeepSeek-V4/V4.1-Flash checkpoints stay packed, so every expert forward and its
recompute and backward unpack the local experts' weights. Each value is an e2m1 number times a power of two, which
BF16 holds exactly, so this kernel matches the table-lookup reference bit for bit while reading 0.5 + 4/32 bytes and
writing 2 per value, instead of materialising int64 nibble indices and an FP32 copy of the weights and scales.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_VALUES_PER_SCALE = 32


@triton.jit
def _e2m1(nibble):
    # e2m1: sign bit, two exponent bits, one mantissa bit; exponent 0 is subnormal (0 or 0.5).
    sign = (nibble >> 3) & 1
    exponent = ((nibble >> 1) & 3).to(tl.float32)
    mantissa = (nibble & 1).to(tl.float32)
    magnitude = tl.where(exponent == 0, mantissa * 0.5, (1.0 + mantissa * 0.5) * tl.exp2(exponent - 1.0))
    # A product, not a negation, so that nibble 8 decodes to -0.0 as the reference does (0 - x would give +0).
    return magnitude * tl.where(sign == 1, -1.0, 1.0)


@triton.jit
def _mxfp4_to_bf16_kernel(packed_ptr, scale_ptr, out_ptr, num_packed, packed_cols, scale_cols,
                          BLOCK: tl.constexpr, PACKED_PER_SCALE: tl.constexpr):
    start = tl.program_id(0).to(tl.int64) * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < num_packed
    byte = tl.load(packed_ptr + offsets, mask=mask, other=0).to(tl.uint8, bitcast=True)
    row = offsets // packed_cols
    col = offsets - row * packed_cols
    scale = tl.load(scale_ptr + row * scale_cols + col // PACKED_PER_SCALE, mask=mask, other=0.0)
    low = _e2m1(byte & 0xF) * scale
    high = _e2m1(byte >> 4) * scale
    values = tl.interleave(low, high)
    out_offsets = 2 * start + tl.arange(0, 2 * BLOCK)
    tl.store(out_ptr + out_offsets, values.to(tl.bfloat16), mask=out_offsets < 2 * num_packed)


def mxfp4_to_bf16(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``weight`` int8 ``[..., cols / 2]`` packed e2m1, ``scale`` float32 ``[..., cols / 32]`` -> BF16 ``[..., cols]``."""
    packed_cols = weight.shape[-1]
    scale_cols = scale.shape[-1]
    if 2 * packed_cols != _VALUES_PER_SCALE * scale_cols or weight.shape[:-1] != scale.shape[:-1]:
        raise ValueError(f"MXFP4 weight {tuple(weight.shape)} does not match its scales {tuple(scale.shape)}")
    weight = weight.contiguous()
    scale = scale.contiguous()
    out = torch.empty((*weight.shape[:-1], 2 * packed_cols), dtype=torch.bfloat16, device=weight.device)
    num_packed = weight.numel()
    if num_packed:
        grid = (triton.cdiv(num_packed, _BLOCK),)
        _mxfp4_to_bf16_kernel[grid](weight, scale, out, num_packed, packed_cols, scale_cols,
                                    BLOCK=_BLOCK, PACKED_PER_SCALE=_VALUES_PER_SCALE // 2)
    return out
