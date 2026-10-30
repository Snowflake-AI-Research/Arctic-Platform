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
"""Fused Qwen3.8-Flash-Next hyper connections: ``Qwen4ExpTextGatedResidual`` and the residual injection after it.

Every decoder layer twice mixes its ``hc`` residual streams ``X`` ``[T, hc * H]`` into one block input, and adds the
block output back into each stream with a per-stream weight:

    n = rmsnorm(X per stream of H) * (1 + w)       FP32 inside, BF16 out
    a, l = n @ W_down.T, n @ W_inject.T             one GEMM on the stacked weights
    mixed = mean_g(sigmoid(silu(a / hc) @ W_up.T)_g * n_g)
    inj = 2 * sigmoid(l / hc)
    X' = X + out[:, None, :] * inj[:, :, None]       out = block(mixed)

Hugging Face runs ~10 elementwise passes over ``[T, hc * H]`` tensors per call, most in FP32 (7.6 ms forward, 23.7 ms
forward + backward at 32K tokens x 4 x 2560 on an H200). Here it is three Triton passes each way (norm, mix and
inject forward; inject, mix and norm backward) around the four GEMMs, with only input gradients: callers keep the
module's own forward when a hyper-connection weight trains.

By default the fused path rounds to BF16 at the tensors the Hugging Face module rounds. With ``round_once`` it
evaluates ``silu(a / hc)``, the mix and the injection in FP32 and rounds each once, as vLLM's ``_hc_gate_mix_kernel``
and ``_hc_combine_kernel`` (``qwen4_exp/nvidia/ops/hc.py``) do; ``gated_residual`` then returns the injection logits
``l`` and ``combine`` applies them. ``gate_mix`` and ``combine`` are those two vLLM steps on their own, with weight
gradients, for hyper connections that train.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_BLOCK_H = 512


@triton.jit
def _norm_fwd_kernel(x_ptr, w_ptr, n_ptr, rstd_ptr, H, eps, HC: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)  # one (token, stream) row of H channels
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    x = tl.load(x_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / H + eps)
    w = tl.load(w_ptr + (row % HC) * H + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(n_ptr + row * H + cols, (x * rstd * (1.0 + w)).to(n_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _norm_bwd_kernel(dn_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, H, HC: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    x = tl.load(x_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + (row % HC) * H + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    xhat = x * rstd
    dxhat = tl.load(dn_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32) * (1.0 + w)
    c = tl.sum(dxhat * xhat, axis=0) / H
    tl.store(dx_ptr + row * H + cols, (rstd * (dxhat - xhat * c)).to(dx_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _mix_fwd_kernel(b_ptr, n_ptr, mixed_ptr, H, HC: tl.constexpr, BLOCK_H: tl.constexpr):
    tok = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = cols < H
    acc = tl.zeros([BLOCK_H], tl.float32)
    for g in tl.static_range(HC):
        off = tok * HC * H + g * H + cols
        m = tl.sigmoid(tl.load(b_ptr + off, mask=mask, other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        n = tl.load(n_ptr + off, mask=mask, other=0.0).to(tl.float32)
        acc += (m * n).to(tl.bfloat16).to(tl.float32)
    tl.store(mixed_ptr + tok * H + cols, (acc / HC).to(mixed_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _mix_bwd_kernel(dmixed_ptr, b_ptr, n_ptr, db_ptr, dn_ptr, H, HC: tl.constexpr, BLOCK_H: tl.constexpr):
    tok = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = cols < H
    dp = (tl.load(dmixed_ptr + tok * H + cols, mask=mask, other=0.0).to(tl.float32) / HC).to(tl.bfloat16)
    dp = dp.to(tl.float32)
    for g in tl.static_range(HC):
        off = tok * HC * H + g * H + cols
        m = tl.sigmoid(tl.load(b_ptr + off, mask=mask, other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        n = tl.load(n_ptr + off, mask=mask, other=0.0).to(tl.float32)
        dm = (dp * n).to(tl.bfloat16).to(tl.float32)
        tl.store(db_ptr + off, (dm * m * (1.0 - m)).to(db_ptr.dtype.element_ty), mask=mask)
        tl.store(dn_ptr + off, (dp * m).to(dn_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _inject_fwd_kernel(x_ptr, out_ptr, inj_ptr, y_ptr, H, HC: tl.constexpr, BLOCK_H: tl.constexpr):
    tok = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = cols < H
    o = tl.load(out_ptr + tok * H + cols, mask=mask, other=0.0).to(tl.float32)
    for g in tl.static_range(HC):
        off = tok * HC * H + g * H + cols
        inj = tl.load(inj_ptr + tok * HC + g).to(tl.float32)
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        injection = (o * inj).to(tl.bfloat16).to(tl.float32)
        tl.store(y_ptr + off, (x + injection).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _inject_bwd_kernel(dy_ptr, out_ptr, inj_ptr, dout_ptr, dinj_ptr, H, HC: tl.constexpr, BLOCK_H: tl.constexpr):
    tok = tl.program_id(0).to(tl.int64)
    groups = tl.arange(0, HC)
    inj = tl.load(inj_ptr + tok * HC + groups).to(tl.float32)
    dinj = tl.zeros([HC], tl.float32)
    for start in range(0, H, BLOCK_H):
        cols = start + tl.arange(0, BLOCK_H)
        mask = cols < H
        dy = tl.load(dy_ptr + tok * HC * H + groups[:, None] * H + cols[None, :], mask=mask[None, :], other=0.0)
        dy = dy.to(tl.float32)
        o = tl.load(out_ptr + tok * H + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(dout_ptr + tok * H + cols, tl.sum(dy * inj[:, None], axis=0).to(dout_ptr.dtype.element_ty), mask=mask)
        dinj += tl.sum(dy * o[None, :], axis=1)
    tl.store(dinj_ptr + tok * HC + groups, dinj.to(dinj_ptr.dtype.element_ty))


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


def _stacked_down_inject(module) -> torch.Tensor:
    """``[W_down; W_inject]``, cached on the module while its (frozen) weights are the same tensors."""
    down, inject = module.input_mix_weight_down.weight, module.block_inject_weight.weight
    key = (down.data_ptr(), inject.data_ptr(), down._version, inject._version)
    cached = getattr(module, "_dss_stacked_down_inject", None)
    if cached is None or cached[0] != key:
        cached = (key, torch.cat([down, inject], 0).contiguous())
        module._dss_stacked_down_inject = cached
    return cached[1]


class _GatedResidualFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, norm_weight, w_down_inject, w_up, hc: int, eps: float, round_once: bool):
        rows, width = x.shape
        hidden = width // hc
        block = triton.next_power_of_2(hidden)
        x = x.contiguous()
        n = torch.empty_like(x)
        rstd = torch.empty(rows * hc, dtype=torch.float32, device=x.device)
        _norm_fwd_kernel[(rows * hc,)](x, norm_weight, n, rstd, hidden, eps, HC=hc, BLOCK=block, num_warps=8)
        rank = w_up.shape[1]
        al = n @ w_down_inject.t()
        a_div = al[:, :rank].float() / hc if round_once else al[:, :rank] / hc
        b = F.silu(a_div).to(x.dtype) @ w_up.t()
        mixed = torch.empty(rows, hidden, dtype=x.dtype, device=x.device)
        grid = (rows, triton.cdiv(hidden, _BLOCK_H))
        if round_once:
            _gate_mix_fwd_kernel[grid](n, b, mixed, hidden, HC=hc, BLOCK=_BLOCK_H)
            gate, injection = None, al[:, rank:].contiguous()
        else:
            _mix_fwd_kernel[grid](b, n, mixed, hidden, HC=hc, BLOCK_H=_BLOCK_H)
            gate = torch.sigmoid(al[:, rank:] / hc)
            injection = 2 * gate
        ctx.save_for_backward(x, rstd, n, b, a_div, gate, norm_weight, w_down_inject, w_up)
        ctx.hc, ctx.round_once = hc, round_once
        return mixed, injection

    @staticmethod
    def backward(ctx, dmixed, dinj):
        x, rstd, n, b, a_div, gate, norm_weight, w_down_inject, w_up = ctx.saved_tensors
        hc = ctx.hc
        rows, width = x.shape
        hidden = width // hc
        db = torch.empty_like(b)
        dn = torch.empty_like(n)
        grid = (rows, triton.cdiv(hidden, _BLOCK_H))
        if ctx.round_once:
            _gate_mix_bwd_kernel[grid](n, b, dmixed.contiguous(), dn, db, hidden, HC=hc, BLOCK=_BLOCK_H)
            da = torch.ops.aten.silu_backward((db @ w_up).float(), a_div) / hc
            dl = dinj
        else:
            _mix_bwd_kernel[grid](dmixed.contiguous(), b, n, db, dn, hidden, HC=hc, BLOCK_H=_BLOCK_H)
            da = torch.ops.aten.silu_backward(db @ w_up, a_div) / hc
            dl = (2 * dinj) * gate * (1 - gate) / hc
        dn.addmm_(torch.cat([da.to(dn.dtype), dl.to(dn.dtype)], 1), w_down_inject)
        dx = torch.empty_like(x)
        _norm_bwd_kernel[(rows * hc,)](
            dn, x, norm_weight, rstd, dx, hidden, HC=hc, BLOCK=triton.next_power_of_2(hidden), num_warps=8
        )
        return dx, None, None, None, None, None, None


class _InjectFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out, inj, hc: int):
        rows, width = x.shape
        hidden = width // hc
        y = torch.empty_like(x)
        out, inj = out.contiguous(), inj.contiguous()
        _inject_fwd_kernel[(rows, triton.cdiv(hidden, _BLOCK_H))](
            x.contiguous(), out, inj, y, hidden, HC=hc, BLOCK_H=_BLOCK_H
        )
        ctx.save_for_backward(out, inj)
        ctx.hc = hc
        return y

    @staticmethod
    def backward(ctx, dy):
        out, inj = ctx.saved_tensors
        hc = ctx.hc
        rows, hidden = out.shape
        dy = dy.contiguous()
        dout = torch.empty_like(out)
        dinj = torch.empty_like(inj)
        _inject_bwd_kernel[(rows,)](dy, out, inj, dout, dinj, hidden, HC=hc, BLOCK_H=_BLOCK_H)
        return dy, dout, dinj, None


def fused_hyper_connection_supported(module, x: torch.Tensor) -> bool:
    """The fused path covers frozen BF16 CUDA hyper connections whose stream count is a power of two."""
    hc = module.hc_count
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and module.block_inject_weight is not None
        and hc & (hc - 1) == 0
        and not any(p.requires_grad for p in module.parameters())
    )


def gated_residual(module, hidden_states: torch.Tensor, round_once: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """``(mixed, injection_weights)`` of ``Qwen4ExpTextGatedResidual`` (its hyper input is ``hidden_states``), or
    ``(mixed, injection_logits)`` for ``combine`` with ``round_once``."""
    shape = hidden_states.shape
    mixed, inj = _GatedResidualFn.apply(
        hidden_states.reshape(-1, shape[-1]),
        module.hc_norm.weight,
        _stacked_down_inject(module),
        module.input_mix_weight_up.weight,
        module.hc_count,
        module.hc_norm.eps,
        round_once,
    )
    return mixed.view(*shape[:-1], -1), inj.view(*shape[:-1], -1)


def inject(hyper_input: torch.Tensor, block_output: torch.Tensor, injection_weights: torch.Tensor) -> torch.Tensor:
    """``hyper_input + (block_output[..., None, :] * injection_weights[..., :, None]).flatten(-2)``."""
    shape = hyper_input.shape
    hc = injection_weights.shape[-1]
    y = _InjectFn.apply(
        hyper_input.reshape(-1, shape[-1]),
        block_output.reshape(-1, block_output.shape[-1]),
        injection_weights.reshape(-1, hc),
        hc,
    )
    return y.view(shape)


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
            grid = (rows_x.shape[0], triton.cdiv(hc_dim, _BLOCK_H))
            _gate_mix_fwd_kernel[grid](rows_x, rows_g, out, hc_dim, HC=hc_count, BLOCK=_BLOCK_H)
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
            grid = (rows_x.shape[0], triton.cdiv(hc_dim, _BLOCK_H))
            _gate_mix_bwd_kernel[grid](
                rows_x, rows_g, _rows(grad, hc_dim), dx, dgate, hc_dim, HC=hc_count, BLOCK=_BLOCK_H
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
            grid = (rows_res.shape[0], triton.cdiv(hc_dim, _BLOCK_H))
            _combine_fwd_kernel[grid](
                rows_res,
                _rows(block, hc_dim),
                _rows(logits, hc_count),
                out,
                hc_dim,
                HC=hc_count,
                HC_PAD=triton.next_power_of_2(hc_count),
                BLOCK=_BLOCK_H,
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
                BLOCK=_BLOCK_H,
            )
        return grad, dblock.reshape(block.shape), dlogits.reshape(logits.shape), None


def gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    """``mean_s(sigmoid(gate_s) * x_s)`` over the ``hc_count`` streams of ``[..., hc_count * hidden]``, in fp32,
    rounded once to ``x.dtype``."""
    if x.is_cuda and gate.is_cuda:
        return _GateMix.apply(x, gate, hc_count)
    return _gate_mix_reference(x, gate, hc_count)


def combine(residual: torch.Tensor, block: torch.Tensor, logits: torch.Tensor, hc_count: int) -> torch.Tensor:
    """``residual_s + block * 2 * sigmoid(logits_s / hc_count)`` for every stream ``s`` of ``residual``, in fp32,
    rounded once to ``residual.dtype``."""
    if residual.is_cuda and block.is_cuda and logits.is_cuda:
        return _Combine.apply(residual, block, logits, hc_count)
    return _combine_reference(residual, block, logits, hc_count)
