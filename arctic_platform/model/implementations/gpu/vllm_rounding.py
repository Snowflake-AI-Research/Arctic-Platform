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
"""Operations evaluated in fp32 the way vLLM evaluates them, shared by the Qwen3.5 trainer models.

``FusedAddRMSNorm`` is a residual add followed by a (1 + weight) RMSNorm, both in fp32. ``rope_rotary_slice_fp32``
applies a partial rotary embedding in fp32.
"""

from __future__ import annotations

import torch


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    """Return fp32 for the low-precision training dtypes and ``x``'s own dtype otherwise.

    Keeping fp64 inputs in fp64 lets ``torch.autograd.gradcheck`` exercise :class:`FusedAddRMSNorm`.
    """
    if x.dtype in (torch.bfloat16, torch.float16):
        return torch.float32
    return x.dtype


class FusedAddRMSNorm(torch.autograd.Function):
    """Residual add followed by (1+weight) RMSNorm, both evaluated in fp32.

    This follows vLLM's ``fused_add_rms_norm``: the sum of ``x`` and ``residual`` is formed in fp32, and the
    variance and the normalized value are computed from that fp32 sum rather than from a sum already rounded to
    the input dtype. Both outputs, the normalized value and the sum that becomes the next residual, are returned
    in the input dtype.

    For backward it saves only the inputs as given (``x``, ``residual`` and ``weight``) and the per-row
    ``rstd``, and recomputes the fp32 sum from them, instead of keeping the fp32 sum and the fp32 normalized
    value alive until backward.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
        compute_dtype = _compute_dtype(x)
        summed = x.to(compute_dtype) + residual.to(compute_dtype)
        rstd = torch.rsqrt(summed.pow(2).mean(-1, keepdim=True) + eps)
        out = summed * rstd * (1.0 + weight.to(compute_dtype))
        ctx.save_for_backward(x, residual, weight, rstd)
        return out.to(x.dtype), summed.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_residual: torch.Tensor):
        x, residual, weight, rstd = ctx.saved_tensors
        compute_dtype = _compute_dtype(x)
        summed = x.to(compute_dtype) + residual.to(compute_dtype)
        normed = summed * rstd
        scale = 1.0 + weight.to(compute_dtype)

        grad = grad_out.to(compute_dtype)
        grad_weight = (grad * normed).reshape(-1, weight.shape[-1]).sum(0).to(weight.dtype)

        # Gradient through normed = summed * rstd(summed), with rstd = (mean(summed**2) + eps) ** -0.5.
        grad_normed = grad * scale
        hidden_size = summed.shape[-1]
        grad_summed = rstd * (grad_normed - normed * (grad_normed * normed).sum(-1, keepdim=True) / hidden_size)
        grad_summed = grad_summed + grad_residual.to(compute_dtype)
        return grad_summed.to(x.dtype), grad_summed.to(residual.dtype), grad_weight, None


def rope_rotary_slice_fp32(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply partial RoPE in fp32 and cast the rotated channels back to ``x``'s dtype once.

    ``x`` is ``[batch, heads, seq, head_dim]`` and ``cos``/``sin`` are ``[batch, seq, rotary_dim]``; they are
    unsqueezed on the head axis. Only the first ``rotary_dim`` channels are rotated; the remaining channels are
    returned unchanged. The result is bitwise equal to upcasting all of ``x``, rotating in fp32 and casting back,
    because the pass-through channels survive the round trip through fp32 exactly. Upcasting only the rotary
    slice avoids materializing an fp32 copy of the whole head dimension.
    """
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rot_fp32 = x_rot.float()
    half = rotary_dim // 2
    rotated = torch.cat((-x_rot_fp32[..., half:], x_rot_fp32[..., :half]), dim=-1)
    embed = (x_rot_fp32 * cos.unsqueeze(1).float()) + (rotated * sin.unsqueeze(1).float())
    return torch.cat((embed.to(x.dtype), x_pass), dim=-1)
