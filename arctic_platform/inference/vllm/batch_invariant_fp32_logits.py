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

"""fp32 LM-head logits from vLLM's batch-invariant persistent matmul.

The fp32 LM head computes ``torch.mm(hidden, weight.t(), out_dtype=torch.float32)``, a cuBLAS GEMM that
``VLLM_BATCH_INVARIANT=1`` does not replace, so its logits can depend on how many tokens share the call. This
function runs vLLM's batch-invariant persistent kernel on the bf16/fp16 operands with the tile configuration of a
bf16/fp16 output and stores the fp32 accumulator without the final cast. A row's logits then depend only on the
row, and a trainer that calls the same function reproduces them.
"""

from __future__ import annotations

import torch


def _default_config(dtype: torch.dtype, batch_invariant) -> dict:
    if dtype == torch.float16:
        block_n = batch_invariant._fp16_block_size_n
    elif dtype == torch.bfloat16:
        block_n = 128
    else:
        raise TypeError(f"fp32-output batch-invariant matmul takes bf16/fp16 operands, got {dtype}")
    return {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 8,
    }


def matmul_fp32_out(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` (``[M, K] @ [K, N]``, bf16/fp16) as an fp32 ``[M, N]`` by the batch-invariant persistent matmul."""
    import triton
    from vllm.model_executor.determinism import batch_invariant

    if a.shape[1] != b.shape[0] or a.dtype != b.dtype:
        raise ValueError(f"incompatible operands {tuple(a.shape)} {a.dtype} @ {tuple(b.shape)} {b.dtype}")
    num_sms = batch_invariant.num_compute_units(a.device.index)
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    if not batch_invariant._BLOCK_SIZES_RESOLVED:
        batch_invariant._resolve_block_sizes()
    config = batch_invariant._get_matmul_config(M, N, K, a.dtype, _default_config(a.dtype, batch_invariant))

    def grid(meta):
        return (min(num_sms, triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"])),)

    batch_invariant.matmul_kernel_persistent[grid](
        a,
        b,
        c,
        None,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        1,
        NUM_SMS=num_sms,
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=False,
        **config,
    )
    return c


def linear_fp32_out(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` in fp32 for a bf16/fp16 ``x`` and ``weight``, batch-invariant."""
    out = matmul_fp32_out(x.reshape(-1, x.shape[-1]), weight.t())
    return out.view(*x.shape[:-1], weight.shape[0])
