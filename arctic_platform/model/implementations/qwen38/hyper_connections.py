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
"""Qwen3.8-Flash-Next hyper-connection stream mix and combine in fp32, rounded once, as vLLM computes them.

``gate_mix`` is ``mean_s(sigmoid(gate_s) * x_s)`` and ``combine`` is ``residual_s + block * 2 * sigmoid(logit_s / hc)``,
with every stream ``s`` of ``[..., hc * hidden]``. vLLM's ``_hc_gate_mix_kernel`` and ``_hc_combine_kernel``
(``qwen4_exp/nvidia/ops/hc.py``) evaluate both in fp32 and store once. On CUDA each direction here is one Triton pass
with the same arithmetic, which saves only its inputs and recomputes the sigmoids in backward; elsewhere PyTorch
evaluates the same fp32 formulas.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs keep the PyTorch path
    triton = None

_BLOCK = 512

if triton is not None:

    @triton.jit
    def _gate_mix_fwd_kernel(x_ptr, g_ptr, out_ptr, HC_DIM, HC: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < HC_DIM
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for stream in tl.static_range(HC):
            offsets = row * (HC * HC_DIM) + stream * HC_DIM + cols
            g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            acc += tl.sigmoid(g.to(tl.float32)) * x.to(tl.float32)
        acc /= HC
        tl.store(out_ptr + row * HC_DIM + cols, acc.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _gate_mix_bwd_kernel(x_ptr, g_ptr, dout_ptr, dx_ptr, dg_ptr, HC_DIM, HC: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < HC_DIM
        dout = tl.load(dout_ptr + row * HC_DIM + cols, mask=mask, other=0.0).to(tl.float32) / HC
        for stream in tl.static_range(HC):
            offsets = row * (HC * HC_DIM) + stream * HC_DIM + cols
            gate = tl.sigmoid(tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32))
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            tl.store(dx_ptr + offsets, (dout * gate).to(dx_ptr.dtype.element_ty), mask=mask)
            tl.store(dg_ptr + offsets, (dout * x * gate * (1.0 - gate)).to(dg_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _combine_fwd_kernel(
        res_ptr, block_ptr, logit_ptr, out_ptr, HC_DIM, HC: tl.constexpr, HC_PAD: tl.constexpr, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        streams = tl.arange(0, HC_PAD)
        offsets = row * (HC * HC_DIM) + streams[:, None] * HC_DIM + cols[None, :]
        mask = (streams[:, None] < HC) & (cols[None, :] < HC_DIM)
        weight = tl.load(logit_ptr + row * HC + streams, mask=streams < HC, other=0.0)
        weight = 2.0 * tl.sigmoid(weight.to(tl.float32) / HC)
        block = tl.load(block_ptr + row * HC_DIM + cols, mask=cols < HC_DIM, other=0.0)
        res = tl.load(res_ptr + offsets, mask=mask, other=0.0)
        block = block.to(tl.float32)[None, :] * weight[:, None]
        out = res.to(tl.float32) + block.to(tl.float32)
        tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _combine_bwd_kernel(
        block_ptr,
        logit_ptr,
        dout_ptr,
        dblock_ptr,
        dlogit_ptr,
        HC_DIM,
        HC: tl.constexpr,
        HC_PAD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        streams = tl.arange(0, HC_PAD)
        sigmoid = tl.load(logit_ptr + row * HC + streams, mask=streams < HC, other=0.0)
        sigmoid = tl.sigmoid(sigmoid.to(tl.float32) / HC)
        weight = 2.0 * sigmoid
        dweight = tl.zeros([HC_PAD], dtype=tl.float32)
        for start in range(0, HC_DIM, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = (streams[:, None] < HC) & (cols[None, :] < HC_DIM)
            dout = tl.load(
                dout_ptr + row * (HC * HC_DIM) + streams[:, None] * HC_DIM + cols[None, :], mask=mask, other=0.0
            ).to(tl.float32)
            block = tl.load(block_ptr + row * HC_DIM + cols, mask=cols < HC_DIM, other=0.0).to(tl.float32)
            dblock = tl.sum(dout * weight[:, None], axis=0)
            tl.store(dblock_ptr + row * HC_DIM + cols, dblock.to(dblock_ptr.dtype.element_ty), mask=cols < HC_DIM)
            dweight += tl.sum(dout * block[None, :], axis=1)
        dlogit = dweight * 2.0 * sigmoid * (1.0 - sigmoid) / HC
        tl.store(dlogit_ptr + row * HC + streams, dlogit.to(dlogit_ptr.dtype.element_ty), mask=streams < HC)


def _fused(*tensors: torch.Tensor) -> bool:
    return triton is not None and all(t.is_cuda for t in tensors)


def _rows(t: torch.Tensor, width: int) -> torch.Tensor:
    return t.reshape(-1, width).contiguous()


def _gate_mix_reference(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    streams = (hc_count, x.shape[-1] // hc_count)
    return (torch.sigmoid(gate.float()).unflatten(-1, streams) * x.unflatten(-1, streams)).mean(dim=-2).to(x.dtype)


def _combine_reference(residual: torch.Tensor, block: torch.Tensor, logits: torch.Tensor, hc_count: int):
    weight = 2 * torch.sigmoid(logits.float() / hc_count)
    return (residual + (block.unsqueeze(-2) * weight.unsqueeze(-1)).flatten(-2)).to(residual.dtype)


class _GateMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
        ctx.save_for_backward(x, gate)
        ctx.hc_count = hc_count
        width = x.shape[-1]
        hc_dim = width // hc_count
        rows_x, rows_g = _rows(x, width), _rows(gate, width)
        out = torch.empty(rows_x.shape[0], hc_dim, dtype=x.dtype, device=x.device)
        if rows_x.shape[0]:
            grid = (rows_x.shape[0], triton.cdiv(hc_dim, _BLOCK))
            _gate_mix_fwd_kernel[grid](rows_x, rows_g, out, hc_dim, HC=hc_count, BLOCK=_BLOCK)
        return out.reshape(*x.shape[:-1], hc_dim)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        x, gate = ctx.saved_tensors
        hc_count = ctx.hc_count
        # Double backward (create_graph) needs differentiable ops, so it keeps the PyTorch path.
        if torch.is_grad_enabled():
            sigmoid = torch.sigmoid(gate.float()).unflatten(-1, (hc_count, -1))
            dout = grad.float().unsqueeze(-2) / hc_count
            dx = (dout * sigmoid).flatten(-2).to(x.dtype)
            dgate = (dout * x.float().unflatten(-1, (hc_count, -1)) * sigmoid * (1 - sigmoid)).flatten(-2)
            return dx, dgate.to(gate.dtype), None
        width = x.shape[-1]
        hc_dim = width // hc_count
        rows_x, rows_g = _rows(x, width), _rows(gate, width)
        dx, dgate = torch.empty_like(rows_x), torch.empty_like(rows_g)
        if rows_x.shape[0]:
            grid = (rows_x.shape[0], triton.cdiv(hc_dim, _BLOCK))
            _gate_mix_bwd_kernel[grid](
                rows_x, rows_g, _rows(grad, hc_dim), dx, dgate, hc_dim, HC=hc_count, BLOCK=_BLOCK
            )
        return dx.reshape(x.shape), dgate.reshape(gate.shape), None


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual: torch.Tensor, block: torch.Tensor, logits: torch.Tensor, hc_count: int):
        ctx.save_for_backward(block, logits)
        ctx.hc_count = hc_count
        width = residual.shape[-1]
        hc_dim = width // hc_count
        rows_res = _rows(residual, width)
        out = torch.empty_like(rows_res)
        if rows_res.shape[0]:
            grid = (rows_res.shape[0], triton.cdiv(hc_dim, _BLOCK))
            _combine_fwd_kernel[grid](
                rows_res,
                _rows(block, hc_dim),
                _rows(logits, hc_count),
                out,
                hc_dim,
                HC=hc_count,
                HC_PAD=triton.next_power_of_2(hc_count),
                BLOCK=_BLOCK,
            )
        return out.reshape(residual.shape)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        block, logits = ctx.saved_tensors
        hc_count = ctx.hc_count
        if torch.is_grad_enabled():
            sigmoid = torch.sigmoid(logits.float() / hc_count)
            dout = grad.float().unflatten(-1, (hc_count, -1))
            dblock = (dout * (2 * sigmoid).unsqueeze(-1)).sum(-2).to(block.dtype)
            dweight = (dout * block.float().unsqueeze(-2)).sum(-1)
            dlogits = (dweight * 2 * sigmoid * (1 - sigmoid) / hc_count).to(logits.dtype)
            return grad, dblock, dlogits, None
        hc_dim = block.shape[-1]
        rows_block, rows_logits = _rows(block, hc_dim), _rows(logits, hc_count)
        dblock, dlogits = torch.empty_like(rows_block), torch.empty_like(rows_logits)
        if rows_block.shape[0]:
            _combine_bwd_kernel[(rows_block.shape[0],)](
                rows_block,
                rows_logits,
                _rows(grad, hc_count * hc_dim),
                dblock,
                dlogits,
                hc_dim,
                HC=hc_count,
                HC_PAD=triton.next_power_of_2(hc_count),
                BLOCK=_BLOCK,
            )
        return grad, dblock.reshape(block.shape), dlogits.reshape(logits.shape), None


def gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    """``mean_s(sigmoid(gate_s) * x_s)`` over the ``hc_count`` streams of ``[..., hc_count * hidden]``, in fp32,
    rounded once to ``x.dtype``."""
    if _fused(x, gate):
        return _GateMix.apply(x, gate, hc_count)
    return _gate_mix_reference(x, gate, hc_count)


def combine(residual: torch.Tensor, block: torch.Tensor, logits: torch.Tensor, hc_count: int) -> torch.Tensor:
    """``residual_s + block * 2 * sigmoid(logits_s / hc_count)`` for every stream ``s`` of ``residual``, in fp32,
    rounded once to ``residual.dtype``."""
    if _fused(residual, block, logits):
        return _Combine.apply(residual, block, logits, hc_count)
    return _combine_reference(residual, block, logits, hc_count)
