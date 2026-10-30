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
inject forward; inject, mix and norm backward) around the four GEMMs, rounding to BF16 at the tensors the Hugging
Face module rounds. When the hyper-connection weights train, their gradients come from the same backward: the GEMM
weight gradients from the saved GEMM inputs and ``(1 + w)``'s from the normalized stream.
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


def _stacked_down_inject(module) -> torch.Tensor:
    """``[W_down; W_inject]``: cached on the module while its frozen weights are the same tensors, rebuilt under
    autograd every call when they train."""
    down, inject = module.input_mix_weight_down.weight, module.block_inject_weight.weight
    if down.requires_grad or inject.requires_grad:
        return torch.cat([down, inject], 0)
    key = (down.data_ptr(), inject.data_ptr(), down._version, inject._version)
    cached = getattr(module, "_dss_stacked_down_inject", None)
    if cached is None or cached[0] != key:
        cached = (key, torch.cat([down, inject], 0).contiguous())
        module._dss_stacked_down_inject = cached
    return cached[1]


class _GatedResidualFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, norm_weight, w_down_inject, w_up, hc: int, eps: float):
        rows, width = x.shape
        hidden = width // hc
        block = triton.next_power_of_2(hidden)
        x = x.contiguous()
        n = torch.empty_like(x)
        rstd = torch.empty(rows * hc, dtype=torch.float32, device=x.device)
        _norm_fwd_kernel[(rows * hc,)](x, norm_weight, n, rstd, hidden, eps, HC=hc, BLOCK=block, num_warps=8)
        rank = w_up.shape[1]
        al = n @ w_down_inject.t()
        a_div = al[:, :rank] / hc
        l_div = al[:, rank:] / hc
        b = F.silu(a_div) @ w_up.t()
        mixed = torch.empty(rows, hidden, dtype=x.dtype, device=x.device)
        _mix_fwd_kernel[(rows, triton.cdiv(hidden, _BLOCK_H))](b, n, mixed, hidden, HC=hc, BLOCK_H=_BLOCK_H)
        gate = torch.sigmoid(l_div)
        ctx.save_for_backward(x, rstd, n, b, a_div, gate, norm_weight, w_down_inject, w_up)
        ctx.hc = hc
        return mixed, 2 * gate

    @staticmethod
    def backward(ctx, dmixed, dinj):
        x, rstd, n, b, a_div, gate, norm_weight, w_down_inject, w_up = ctx.saved_tensors
        hc = ctx.hc
        rows, width = x.shape
        hidden = width // hc
        db = torch.empty_like(b)
        dn = torch.empty_like(n)
        _mix_bwd_kernel[(rows, triton.cdiv(hidden, _BLOCK_H))](
            dmixed.contiguous(), b, n, db, dn, hidden, HC=hc, BLOCK_H=_BLOCK_H
        )
        _, need_norm_w, need_down_inject, need_up, _, _ = ctx.needs_input_grad
        da = torch.ops.aten.silu_backward(db @ w_up, a_div) / hc
        dl = (2 * dinj) * gate * (1 - gate) / hc
        dal = torch.cat([da, dl.to(da.dtype)], 1)
        d_up = db.t().mm(F.silu(a_div)) if need_up else None
        d_down_inject = dal.t().mm(n) if need_down_inject else None
        dn.addmm_(dal, w_down_inject)
        d_norm_w = None
        if need_norm_w:
            # n = x_hat * (1 + w) per stream: the weight's gradient is sum over tokens of dn * x_hat, in FP32.
            x_hat = x.view(rows, hc, hidden).float() * rstd.view(rows, hc, 1)
            d_norm_w = (dn.view(rows, hc, hidden).float() * x_hat).sum(0).flatten().to(norm_weight.dtype)
        dx = torch.empty_like(x)
        _norm_bwd_kernel[(rows * hc,)](
            dn, x, norm_weight, rstd, dx, hidden, HC=hc, BLOCK=triton.next_power_of_2(hidden), num_warps=8
        )
        return dx, d_norm_w, d_down_inject, d_up, None, None


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
    """The fused path covers BF16 CUDA hyper connections whose stream count is a power of two, frozen or training."""
    hc = module.hc_count
    return x.is_cuda and x.dtype == torch.bfloat16 and module.block_inject_weight is not None and hc & (hc - 1) == 0


def gated_residual(module, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(mixed, injection_weights)`` of ``Qwen4ExpTextGatedResidual`` (its hyper input is ``hidden_states``)."""
    shape = hidden_states.shape
    mixed, inj = _GatedResidualFn.apply(
        hidden_states.reshape(-1, shape[-1]),
        module.hc_norm.weight,
        _stacked_down_inject(module),
        module.input_mix_weight_up.weight,
        module.hc_count,
        module.hc_norm.eps,
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
