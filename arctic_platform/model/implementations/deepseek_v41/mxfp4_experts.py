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

"""Frozen routed experts kept in the checkpoint's packed MXFP4 layout."""

import torch
from torch import Tensor
from torch import nn

from arctic_platform.model.implementations.deepseek_v4.dequantize import dequantize_weight
from arctic_platform.model.implementations.fp8 import mark_keep_fp32
from arctic_platform.model.implementations.moe.config import DISPATCH_EP_BACKENDS
from arctic_platform.model.implementations.moe.config import EPCommBackend
from arctic_platform.model.implementations.moe.distributed.ep_backend import uses_dispatch_ep
from arctic_platform.model.implementations.moe.layers.moe import _maybe_to_local
from arctic_platform.model.implementations.moe.layers.moe import _run_experts_for_loop_impl
from arctic_platform.model.implementations.moe.layers.moe import _run_experts_grouped_mm_impl

MXFP4_BLOCK = 32
# The unpack makes int64 indices and float32 copies, about 30 bytes per packed byte.
_EXPERTS_PER_DEQUANTIZE = 4


def dequantize_mxfp4(weight: Tensor, scale: Tensor) -> Tensor:
    """``int8`` with two e2m1 nibbles per byte, one scale per 1x32 block, to ``bfloat16``."""
    if weight.dtype != torch.int8:
        raise ValueError(f"MXFP4 expert weight arrived as {weight.dtype}, expected int8")
    if weight.dim() < 3:
        return dequantize_weight(weight, scale)
    out = torch.empty((*weight.shape[:-1], 2 * weight.shape[-1]), dtype=torch.bfloat16, device=weight.device)
    for start in range(0, weight.shape[0], _EXPERTS_PER_DEQUANTIZE):
        end = start + _EXPERTS_PER_DEQUANTIZE
        out[start:end] = dequantize_weight(weight[start:end], scale[start:end])
    return out


def _mxfp4_experts(
    x: Tensor,
    w1: Tensor,
    s1: Tensor,
    w2: Tensor,
    s2: Tensor,
    w3: Tensor,
    s3: Tensor,
    num_tokens_per_expert: Tensor,
    swiglu_limit: float | None,
    use_grouped_mm: bool,
) -> Tensor:
    out_dtype = x.dtype
    if x.dtype != torch.bfloat16:
        x = x.bfloat16()
    restored = (dequantize_mxfp4(w1, s1), dequantize_mxfp4(w2, s2), dequantize_mxfp4(w3, s3))
    if x.is_cuda and use_grouped_mm:
        output = _run_experts_grouped_mm_impl(*restored, x, num_tokens_per_expert, swiglu_limit)
    else:
        output = _run_experts_for_loop_impl(*restored, x, num_tokens_per_expert, swiglu_limit)
    return output if output.dtype == out_dtype else output.to(dtype=out_dtype)


class _Mxfp4ExpertsFn(torch.autograd.Function):
    """Saves only the packed weights; backward dequantizes them again."""

    @staticmethod
    def forward(ctx, x, w1, s1, w2, s2, w3, s3, num_tokens_per_expert, swiglu_limit, use_grouped_mm):
        ctx.save_for_backward(x, w1, s1, w2, s2, w3, s3, num_tokens_per_expert)
        ctx.swiglu_limit = swiglu_limit
        ctx.use_grouped_mm = use_grouped_mm
        return _mxfp4_experts(x, w1, s1, w2, s2, w3, s3, num_tokens_per_expert, swiglu_limit, use_grouped_mm)

    @staticmethod
    def backward(ctx, grad_output):
        x, w1, s1, w2, s2, w3, s3, num_tokens_per_expert = ctx.saved_tensors
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            output = _mxfp4_experts(x_in, w1, s1, w2, s2, w3, s3, num_tokens_per_expert, ctx.swiglu_limit, False)
            (grad_x,) = torch.autograd.grad(output, x_in, grad_output)
        return grad_x, None, None, None, None, None, None, None, None, None


def _scale(num_experts: int, rows: int, cols: int) -> nn.Parameter:
    return mark_keep_fp32(
        nn.Parameter(torch.empty(num_experts, rows, cols // MXFP4_BLOCK, dtype=torch.float32), requires_grad=False)
    )


class Mxfp4GroupedExperts(nn.Module):
    """Routed experts as frozen packed MXFP4. Each forward dequantizes a temporary bf16 copy.

    LoRA does not wrap these parameters.
    """

    mxfp4 = True

    def __init__(self, dim: int, hidden_dim: int, num_experts: int, swiglu_limit: float | None):
        super().__init__()
        if dim % MXFP4_BLOCK or hidden_dim % MXFP4_BLOCK:
            raise ValueError(
                f"MXFP4 experts need hidden sizes divisible by {MXFP4_BLOCK}, got dim={dim} hidden_dim={hidden_dim}"
            )
        self.num_experts = num_experts
        self.swiglu_limit = swiglu_limit
        self.use_grouped_mm = True
        self.ep_comm_backend: EPCommBackend = "deepep"
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim // 2, dtype=torch.int8), requires_grad=False)
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim // 2, dtype=torch.int8), requires_grad=False)
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim // 2, dtype=torch.int8), requires_grad=False)
        self.w1_scale_inv = _scale(num_experts, hidden_dim, dim)
        self.w2_scale_inv = _scale(num_experts, dim, hidden_dim)
        self.w3_scale_inv = _scale(num_experts, hidden_dim, dim)

    def set_ep_comm_backend(self, backend: EPCommBackend) -> None:
        self.ep_comm_backend = backend

    def init_weights(self, init_std: float) -> None:
        del init_std

    def forward(self, x: Tensor, num_tokens_per_expert: Tensor) -> Tensor:
        if not uses_dispatch_ep(self.ep_comm_backend):
            raise NotImplementedError(
                f"EP comm backend must be one of {DISPATCH_EP_BACKENDS}, got {self.ep_comm_backend!r}."
            )
        return _Mxfp4ExpertsFn.apply(
            x,
            _maybe_to_local(self.w1),
            _maybe_to_local(self.w1_scale_inv),
            _maybe_to_local(self.w2),
            _maybe_to_local(self.w2_scale_inv),
            _maybe_to_local(self.w3),
            _maybe_to_local(self.w3_scale_inv),
            num_tokens_per_expert,
            self.swiglu_limit,
            self.use_grouped_mm,
        )
