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
"""SwiGLU with an optional clamp limit, ``silu(min(gate, L)) * clamp(up, -L, L)``, in one pass each way.

The PyTorch composition launches two clamps, a SiLU and a product, and autograd keeps their intermediates; at the
256K routed-expert batch each is a full pass over ``[rows, hidden]``. Math runs in FP32 per element; gradients pass
the clamps where ``torch.clamp``'s do (bounds inclusive).
"""

import torch
import triton
import triton.language as tl

_BLOCK = 2048


@triton.jit
def _swiglu_fwd_kernel(gate_ptr, up_ptr, out_ptr, n, limit, HAS_LIMIT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_LIMIT:
        g = tl.minimum(g, limit)
        u = tl.minimum(tl.maximum(u, -limit), limit)
    tl.store(out_ptr + offs, (g * tl.sigmoid(g) * u).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_bwd_kernel(
    gate_ptr, up_ptr, grad_ptr, dgate_ptr, dup_ptr, n, limit, HAS_LIMIT: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dh = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_LIMIT:
        pass_g = g <= limit
        pass_u = (u >= -limit) & (u <= limit)
        g = tl.minimum(g, limit)
        u = tl.minimum(tl.maximum(u, -limit), limit)
    s = tl.sigmoid(g)
    dg = dh * u * s * (1.0 + g * (1.0 - s))
    du = dh * g * s
    if HAS_LIMIT:
        dg = tl.where(pass_g, dg, 0.0)
        du = tl.where(pass_u, du, 0.0)
    tl.store(dgate_ptr + offs, dg.to(dgate_ptr.dtype.element_ty), mask=mask)
    tl.store(dup_ptr + offs, du.to(dup_ptr.dtype.element_ty), mask=mask)


class _SwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor, limit: float | None) -> torch.Tensor:
        gate, up = gate.contiguous(), up.contiguous()
        ctx.save_for_backward(gate, up)
        ctx.limit = limit
        out = torch.empty_like(gate)
        n = gate.numel()
        if n:
            _swiglu_fwd_kernel[(triton.cdiv(n, _BLOCK),)](
                gate, up, out, n, 0.0 if limit is None else float(limit), HAS_LIMIT=limit is not None, BLOCK=_BLOCK
            )
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        gate, up = ctx.saved_tensors
        grad = grad.contiguous()
        dgate, dup = torch.empty_like(gate), torch.empty_like(up)
        n = gate.numel()
        if n:
            limit = ctx.limit
            _swiglu_bwd_kernel[(triton.cdiv(n, _BLOCK),)](
                gate,
                up,
                grad,
                dgate,
                dup,
                n,
                0.0 if limit is None else float(limit),
                HAS_LIMIT=limit is not None,
                BLOCK=_BLOCK,
            )
        return dgate, dup, None


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float | None) -> torch.Tensor:
    """``silu(min(gate, limit)) * clamp(up, -limit, limit)`` (no clamps when ``limit`` is None) on CUDA tensors."""
    if gate.shape != up.shape or gate.dtype != up.dtype:
        raise ValueError(f"gate {tuple(gate.shape)} {gate.dtype} and up {tuple(up.shape)} {up.dtype} must match")
    return _SwiGLUFn.apply(gate, up, limit)
