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
"""Grouped INT4 x FP8 GEMMs for W4AFP8 routed experts on Hopper (SGLang's CUTLASS W4A8 kernel, ported).

For tokens packed by expert, ``y_e = a * (fp8(x_e / a) @ (int4_e * s_e).T)``: activations quantized to E4M3 with one
per-tensor scale ``a`` (the checkpoint's static ``input_scale``), INT4 codes converted to E4M3 exactly inside the
mainloop, per-128-input partial sums accumulated in FP32 and multiplied by their BF16 group scale ``s``. These are the
numerics SGLang serves W4AFP8 checkpoints with. The GEMM runs on FP8 tensor cores, twice the BF16 rate, and reads
the 4-bit codes instead of a decoded BF16 copy.

The kernel reads the scales in its interleaved layout ``[E, groups // a, out * a]`` (``interleave_int4_scales``),
``a = SCALE_ALIGN``: the training tiles step 128 inputs, one group, so ``a = 1``. It is
compiled on first use with ``torch.utils.cpp_extension`` against CUTLASS 4.2 headers: ``ARL_CUTLASS_DIR`` when set,
else the copy DeepGEMM ships in its wheel.
"""

from __future__ import annotations

import os

import torch

ARL_W4A8_EXPERTS_ENV = "ARL_W4A8_EXPERTS"
_TILE_ENV = "ARL_W4A8_TILE"
# K step of every compiled tile over the INT4 group size (csrc/w4a8_moe_mm.cu).
SCALE_ALIGN = 1
_FP8_MAX = 448.0
_module_cache = None
_stride_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def w4a8_experts_requested() -> bool:
    return os.environ.get(ARL_W4A8_EXPERTS_ENV, "0") not in ("", "0", "false", "False")


def _cutlass_include_dirs() -> list[str]:
    root = os.environ.get("ARL_CUTLASS_DIR")
    if root:
        return [os.path.join(root, "include")]
    import deep_gemm  # type: ignore[import-not-found]

    return [os.path.join(os.path.dirname(deep_gemm.__file__), "include")]


def _module():
    global _module_cache
    if _module_cache is None:
        from torch.utils.cpp_extension import load

        here = os.path.dirname(os.path.abspath(__file__))
        _module_cache = load(
            name="arl_w4a8_moe_mm",
            sources=[os.path.join(here, "csrc", "w4a8_moe_mm.cu")],
            extra_include_paths=[os.path.join(here, "csrc"), *_cutlass_include_dirs()],
            extra_cuda_cflags=[
                "-gencode=arch=compute_90a,code=sm_90a",
                "-O3",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                "-DNDEBUG",
                "-Xcompiler=-fno-strict-aliasing",
                "--threads=8",
            ],
            verbose=False,
        )
    return _module_cache


def _strides(experts: int, k: int, n: int, out_row_stride: int, device: torch.device):
    # CUTLASS reads one stride struct per expert; SGLang fills [E, 3] int64 tensors with the row strides.
    key = (experts, k, n, out_row_stride, device)
    if key not in _stride_cache:
        _stride_cache[key] = (
            torch.full((experts, 3), k, dtype=torch.int64, device=device),
            torch.full((experts, 3), out_row_stride, dtype=torch.int64, device=device),
            torch.full((experts, 3), n, dtype=torch.int64, device=device),
        )
    return _stride_cache[key]


def _quantize_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def quantize(x_ptr, scale_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.clamp(x * (1.0 / tl.load(scale_ptr)), -448.0, 448.0)
        tl.store(out_ptr + offs, y.to(tl.float8e4nv), mask=mask)

    return quantize


_quantize = None


def quantize_fp8_per_tensor(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x / scale`` saturated to E4M3 (SGLang's static per-tensor activation quantization), in one pass.

    Written as PyTorch ops it materializes an FP32 copy of ``x`` (6.4 GB for GLM-5.3's gate/up input at 256K over EP8)
    and moves ~20 GB per call; the kernel reads the BF16 once and writes the E4M3 once.
    """
    global _quantize
    if not x.is_cuda:
        return (x.float() / scale).clamp_(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    if _quantize is None:
        _quantize = _quantize_kernel()
    x = x.contiguous()
    out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    n = x.numel()
    if n:
        block = 4096
        _quantize[((n + block - 1) // block,)](x, scale.float().reshape(1), out, n, BLOCK=block)
    return out


def w4a8_grouped_mm_into(
    out: torch.Tensor,
    x_fp8: torch.Tensor,
    weight: torch.Tensor,
    interleaved_scale: torch.Tensor,
    offsets: torch.Tensor,
    act_scale: torch.Tensor,
    tile: int | None = None,
) -> torch.Tensor:
    """``out[rows of e] = act_scale * (x_fp8_e @ dequant(weight_e).T)`` for tokens packed by expert.

    ``x_fp8`` is ``[rows, in]`` E4M3, ``weight`` the ``[E, out, in // 2]`` int8 code stack, ``interleaved_scale`` its
    ``[E, in // 128, out]`` BF16 scales (``SCALE_ALIGN``), ``offsets`` the cumulative row count per expert (int32 ``[E]``) and
    ``act_scale`` an FP32 ``[1]`` tensor. ``out`` is ``[rows, out]`` BF16 and may be a column slice of a wider buffer.
    """
    experts, n, k = weight.shape[0], weight.shape[1], weight.shape[2] * 2
    if x_fp8.dtype != torch.float8_e4m3fn or x_fp8.shape[1] != k:
        raise TypeError(f"W4A8 activations must be E4M3 [rows, {k}], got {x_fp8.dtype} {tuple(x_fp8.shape)}")
    if out.dtype != torch.bfloat16 or out.shape[1] != n or out.stride(1) != 1:
        raise TypeError(f"W4A8 output must be BF16 [rows, {n}] with unit column stride, got {tuple(out.shape)}")
    if k % 128:
        raise ValueError(f"the W4A8 tiles read 128 inputs at a time; in_features={k}")
    counts = torch.diff(offsets, prepend=offsets.new_zeros(1))
    starts = (offsets - counts).to(torch.int32)
    problem_sizes = torch.stack((torch.full_like(counts, n), counts, torch.full_like(counts, k)), dim=1)
    a_strides, d_strides, s_strides = _strides(experts, k, n, out.stride(0), weight.device)
    if tile is None:
        tile = int(os.environ.get(_TILE_ENV, "0"))
    _module().w4a8_moe_mm(
        out,
        x_fp8,
        weight,
        act_scale.float().reshape(1),
        interleaved_scale,
        starts,
        problem_sizes.to(torch.int32).contiguous(),
        a_strides,
        a_strides,
        d_strides,
        s_strides,
        128,
        tile,
    )
    return out
