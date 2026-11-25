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


@triton.jit
def _packed_swiglu_fwd_kernel(gu_ptr, out_ptr, D, stride_gu, limit, HAS_LIMIT: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)
        mask = cols < D
        g = tl.load(gu_ptr + row * stride_gu + cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(gu_ptr + row * stride_gu + D + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_LIMIT:
            g = tl.minimum(g, limit)
            u = tl.minimum(tl.maximum(u, -limit), limit)
        tl.store(out_ptr + row * D + cols, (g * tl.sigmoid(g) * u).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _packed_swiglu_bwd_kernel(
    gu_ptr, grad_ptr, dgu_ptr, D, stride_gu, stride_grad, limit, HAS_LIMIT: tl.constexpr, BLOCK_D: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)
        mask = cols < D
        g = tl.load(gu_ptr + row * stride_gu + cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(gu_ptr + row * stride_gu + D + cols, mask=mask, other=0.0).to(tl.float32)
        dh = tl.load(grad_ptr + row * stride_grad + cols, mask=mask, other=0.0).to(tl.float32)
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
        tl.store(dgu_ptr + row * stride_gu + cols, dg.to(dgu_ptr.dtype.element_ty), mask=mask)
        tl.store(dgu_ptr + row * stride_gu + D + cols, du.to(dgu_ptr.dtype.element_ty), mask=mask)


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


_PACKED_BLOCK_D = 1024


class _PackedSwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_up: torch.Tensor, limit: float | None) -> torch.Tensor:
        gate_up = gate_up.contiguous()
        ctx.save_for_backward(gate_up)
        ctx.limit = limit
        hidden = gate_up.shape[1] // 2
        out = torch.empty(gate_up.shape[0], hidden, dtype=gate_up.dtype, device=gate_up.device)
        if out.numel():
            _packed_swiglu_fwd_kernel[(out.shape[0],)](
                gate_up,
                out,
                hidden,
                gate_up.stride(0),
                0.0 if limit is None else float(limit),
                HAS_LIMIT=limit is not None,
                BLOCK_D=_PACKED_BLOCK_D,
            )
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (gate_up,) = ctx.saved_tensors
        grad = grad.contiguous()
        d_gate_up = torch.empty_like(gate_up)
        if grad.numel():
            limit = ctx.limit
            _packed_swiglu_bwd_kernel[(grad.shape[0],)](
                gate_up,
                grad,
                d_gate_up,
                grad.shape[1],
                gate_up.stride(0),
                grad.stride(0),
                0.0 if limit is None else float(limit),
                HAS_LIMIT=limit is not None,
                BLOCK_D=_PACKED_BLOCK_D,
            )
        return d_gate_up, None


def packed_swiglu(gate_up: torch.Tensor, limit: float | None) -> torch.Tensor:
    """``fused_swiglu`` of the halves of a packed ``[rows, 2 * hidden]`` gate|up tensor; its gradient stays packed."""
    if gate_up.ndim != 2 or gate_up.shape[1] % 2:
        raise ValueError(f"gate_up must be [rows, 2 * hidden], got {tuple(gate_up.shape)}")
    return _PackedSwiGLUFn.apply(gate_up, limit)
