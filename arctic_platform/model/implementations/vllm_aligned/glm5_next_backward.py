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
"""Per-op backward of the vLLM-aligned GLM-5.3-Flash forward (``vllm_aligned_backward="vllm_op"``).

vLLM's kernels have no backward. Each kernel call of the aligned forward runs through :func:`vllm_op`: the forward is
the kernel (the sampler's bits), the backward the vector-Jacobian product of an fp32 reference of the same math,
evaluated at the inputs the kernel was given. FP8 activation quantization is straight-through; discrete choices (the
indexer's top-k rows, the routed expert ids) carry no gradient. The references here state each kernel's math:

* linears, block-FP8 linears (activation quantization straight-through), LoRA ``x A^T B^T`` added to a projection;
* ``SiluAndMulWithClamp``: ``silu(min(gate, limit)) * clamp(up, -limit, limit)``;
* mHC pre (vLLM's ``mhc_pre_torch``: RMS-scaled mixes, sigmoid pre/post, softmax + Sinkhorn comb, the pre-mix) with
  the sublayer's RMSNorm; mHC post ``post * x + comb^T streams``;
* RMSNorm; the gated RMSNorm ``x * rsqrt(mean(x^2) + eps) * w * sigmoid(g)``;
* KDA: the causal conv with SiLU, then the gated delta rule (``fused_recurrent_kda``'s math: l2-normalized q/k, the
  bounded gate ``lb / (1 + exp(-exp(A_log) (g + g_bias)))``, sigmoid beta);
* sparse MLA attention in vLLM's QV form over each token's top-k rows: ``softmax((q_rope k + q_v v) * scale) v``;
* the router: sigmoid scores at the replayed expert ids, renormalized in order and scaled;
* the routed experts: each token's ``sum_k w_k down_e(swiglu(gate_up_e(x)))`` on the dequantized expert weights.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

FP8_BLOCK = 128


class _VllmOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kernel: Callable, reference: Callable, *inputs: torch.Tensor):
        ctx.reference = reference
        ctx.save_for_backward(*inputs)
        with torch.no_grad():
            out = kernel(*inputs)
        ctx.multi = isinstance(out, tuple)
        return out

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        saved = ctx.saved_tensors
        inputs = [t.detach().requires_grad_(t.is_floating_point() and need)
                  for t, need in zip(saved, ctx.needs_input_grad[2:])]  # fmt: skip
        with torch.enable_grad():
            out = ctx.reference(*inputs)
            outs = out if ctx.multi else (out,)
            pairs = [(o.reshape(g.shape), g) for o, g in zip(outs, grads) if g is not None and o.requires_grad]
        if pairs:
            torch.autograd.backward([o for o, _ in pairs], [g.to(o.dtype) for o, g in pairs])
        return (None, None, *[t.grad if t.requires_grad else None for t in inputs])


def vllm_op(kernel: Callable, reference: Callable, *inputs: torch.Tensor):
    """``kernel(*inputs)`` as the value; the gradient of ``reference(*inputs)`` (same math, fp32) at those inputs."""
    return _VllmOp.apply(kernel, reference, *inputs)


class _FakeFp8(torch.autograd.Function):
    """vLLM's per-128-group dynamic FP8 activation quantization (``per_token_group_quant_fp8``: scale ``amax / 448``,
    a power of two with ``ue8m0``), dequantized in fp32, with a straight-through gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, ue8m0: bool) -> torch.Tensor:
        if x.shape[-1] % FP8_BLOCK or x.numel() == 0:
            return x.float()
        groups = x.float().reshape(-1, x.shape[-1] // FP8_BLOCK, FP8_BLOCK)
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = groups.abs().amax(-1, keepdim=True).clamp_min(1e-10) / fp8_max
        if ue8m0:
            scale = torch.exp2(torch.ceil(torch.log2(scale)))
        q = (groups / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        return (q.float() * scale).reshape(x.shape)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad, None


def fake_fp8(x: torch.Tensor, ue8m0: bool) -> torch.Tensor:
    return _FakeFp8.apply(x, ue8m0)


# ---- references (fp32 math, results in the kernel's output dtypes) ----


def ref_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (x.float() @ weight.float().t()).to(x.dtype)


def ref_linear_fp32_out(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.float() @ weight.float().t()


def ref_add_lora(y: torch.Tensor, x: torch.Tensor, *factors: torch.Tensor, slices: tuple[int, ...], **_) -> torch.Tensor:
    """``y`` plus each slice's ``x A^T B^T`` (``factors`` = A0, B0, A1, B1, ...; B narrower than its slice pads)."""
    deltas = []
    for i, width in enumerate(slices):
        a, b = factors[2 * i], factors[2 * i + 1]
        d = (x.float() @ a.float().t()) @ b.float().t()
        if d.shape[-1] < width:
            d = F.pad(d, (0, width - d.shape[-1]))
        deltas.append(d)
    return (y.float() + torch.cat(deltas, dim=-1)).to(y.dtype)


def ref_silu_and_mul_with_clamp(x: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = x.float().chunk(2, dim=-1)
    return (F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)).to(x.dtype)


def ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def ref_gated_rms_norm(x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, eps: float, **_) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight.float() * torch.sigmoid(g.float())
    return y.to(x.dtype)


def ref_mhc_pre(streams, fn, scale, base, norm_weight, rms_eps, pre_eps, sinkhorn_eps, post_mult, iters, norm_eps,
                **_):  # fmt: skip
    """vLLM's ``mhc_pre_torch`` followed by the sublayer's RMSNorm: ``(post [T, n], comb [T, n, n], x [T, H])``."""
    n, hidden = streams.shape[-2:]
    x = streams.reshape(-1, n * hidden).float()
    mixes = (x @ fn.float().t()) * torch.rsqrt(x.square().sum(-1, keepdim=True) / (n * hidden) + rms_eps)
    pre = torch.sigmoid(mixes[:, :n] * scale[0] + base[:n]) + pre_eps
    post = torch.sigmoid(mixes[:, n : 2 * n] * scale[1] + base[n : 2 * n]) * post_mult
    comb = mixes[:, 2 * n :].reshape(-1, n, n) * scale[2] + base[2 * n :].reshape(1, n, n)
    comb = torch.softmax(comb, dim=-1) + sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    layer_input = (pre.unsqueeze(-1) * streams.reshape(-1, n, hidden).float()).sum(1)
    out = ref_rms_norm(layer_input.to(streams.dtype), norm_weight, norm_eps)
    return post, comb, out


def ref_mhc_post(x: torch.Tensor, streams: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    mixed = torch.einsum("tij,tih->tjh", comb.float(), streams.float())
    return (mixed + post.float().reshape(*post.shape[:2], 1) * x.float().unsqueeze(-2)).to(streams.dtype)


def ref_causal_conv_silu(x: torch.Tensor, weight: torch.Tensor, cu_seqlens: torch.Tensor, **_) -> torch.Tensor:
    """Depthwise causal conv (``weight [C, W]``) with SiLU over each packed request, zero initial state: ``x [T, C]``."""
    width = weight.shape[-1]
    out = []
    bounds = cu_seqlens.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        seq = x[start:stop].float().t().unsqueeze(0)
        y = F.conv1d(F.pad(seq, (width - 1, 0)), weight.float().unsqueeze(1), groups=weight.shape[0])
        out.append(F.silu(y[0].t()))
    return torch.cat(out).to(x.dtype)


def _kda_steps(q, k, v, g, beta, h, a_amp, g_bias, lower_bound, scale):
    """One chunk of the gated delta rule (``fused_recurrent_kda``'s per-token math) for a batch of requests from
    states ``h [B, H, V, K]``: ``q/k/g [B, C, H, K]``, ``v [B, C, H, V]``, ``beta [B, C, H]``."""
    outs = []
    for t in range(q.shape[1]):
        qt = q[:, t] / torch.sqrt((q[:, t] * q[:, t]).sum(-1, keepdim=True) + 1e-6) * scale
        kt = k[:, t] / torch.sqrt((k[:, t] * k[:, t]).sum(-1, keepdim=True) + 1e-6)
        gate = lower_bound / (1.0 + torch.exp(-(a_amp * (g[:, t] + g_bias))))
        h = h * torch.exp(gate).unsqueeze(-2)
        vt = (v[:, t] - (h * kt.unsqueeze(-2)).sum(-1)) * torch.sigmoid(beta[:, t]).unsqueeze(-1)
        h = h + vt.unsqueeze(-1) * kt.unsqueeze(-2)
        outs.append((h * qt.unsqueeze(-2)).sum(-1))
    return torch.stack(outs, dim=1), h


def ref_kda_core(q, k, v, g, beta, a_log, dt_bias, cu_seqlens, lower_bound: float, chunk: int = 64, **_):
    """The KDA core over packed requests from a zero state: ``q/k/v/g [(1,) T, H, D]``, ``beta [T, H]`` raw logits,
    ``a_log [H]``, ``dt_bias [H * D]``; returns ``[T, H, D]`` in ``q``'s dtype. The requests step together, padded
    to the longest (a padded position's output is dropped, so it gets no gradient); checkpointed every ``chunk``
    steps."""
    from torch.utils.checkpoint import checkpoint

    q, k, v, g = (t.reshape(t.shape[-3:]) for t in (q, k, v, g))
    heads, dim, dtype = *q.shape[1:], q.dtype
    scale = dim**-0.5
    a_amp = torch.exp(a_log.float()).reshape(heads, 1)
    g_bias = dt_bias.float().reshape(heads, dim)
    starts, lengths = cu_seqlens[:-1].long(), (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    steps = torch.arange(int(lengths.max()), device=q.device)
    valid = steps[None, :] < lengths[:, None]
    index = (starts[:, None] + steps[None, :]).clamp_max(q.shape[0] - 1)

    def padded(t):
        rows = t[index].float()
        return rows * valid.reshape(*valid.shape, *[1] * (rows.dim() - 2))

    q, k, v, g, beta = (padded(t) for t in (q, k, v, g, beta))
    h = q.new_zeros((lengths.numel(), heads, v.shape[-1], dim))
    out = []
    for c0 in range(0, steps.numel(), chunk):
        c1 = c0 + chunk
        o, h = checkpoint(_kda_steps, q[:, c0:c1], k[:, c0:c1], v[:, c0:c1], g[:, c0:c1], beta[:, c0:c1], h,
                          a_amp, g_bias, lower_bound, scale, use_reentrant=False)  # fmt: skip
        out.append(o)
    return torch.cat(out, dim=1)[valid].to(dtype)


def _sparse_mla_chunk(q_rope, q_v, k_all, v_all, idx, mask, softmax_scale):
    k, v = k_all[idx], v_all[idx]
    scores = torch.einsum("tnr,tkr->tnk", q_rope.float(), k) + torch.einsum("tnl,tkl->tnk", q_v.float(), v)
    scores = (scores * softmax_scale).masked_fill(~mask[:, None, :], float("-inf"))
    return torch.einsum("tnk,tkl->tnl", torch.softmax(scores, dim=-1), v)


def ref_sparse_mla(q_rope, k_cache, v_cache, q_v, rows, valid, softmax_scale: float, chunk: int = 128, **_):
    """FA3's sparse QV attention as vLLM calls it: token ``t`` attends to the rows ``rows[t, :valid[t]]`` of the
    one-token pages, ``softmax((q_rope . k + q_v . v) * scale) v``; ``q_rope [T, N, R]``, ``q_v [T, N, L]``,
    ``k_cache [rows, 1, 1, R]``, ``v_cache [rows, 1, 1, L]``. Returns ``[T, N, L]``; checkpointed per token chunk."""
    from torch.utils.checkpoint import checkpoint

    k_all, v_all = k_cache.reshape(k_cache.shape[0], -1).float(), v_cache.reshape(v_cache.shape[0], -1).float()
    out = []
    width = rows.shape[1]
    for c0 in range(0, q_v.shape[0], chunk):
        c1 = min(c0 + chunk, q_v.shape[0])
        idx = rows[c0:c1].clamp_min(0).long()
        mask = torch.arange(width, device=rows.device)[None, :] < valid[c0:c1, None]
        out.append(checkpoint(_sparse_mla_chunk, q_rope[c0:c1], q_v[c0:c1], k_all, v_all, idx, mask, softmax_scale,
                              use_reentrant=False))  # fmt: skip
    return torch.cat(out).to(q_v.dtype)


def ref_router_weights(logits: torch.Tensor, ids: torch.Tensor, renormalize: bool, scaling: float) -> torch.Tensor:
    """The grouped top-k's weights at given expert ids: sigmoid scores, renormalized in column order, scaled."""
    w = torch.sigmoid(logits.float()).gather(1, ids.long())
    if renormalize:
        total = w[:, :1]
        for i in range(1, w.shape[1]):
            total = total + w[:, i : i + 1]
        w = w / total
    return w * scaling


def ref_routed_experts(x, weights, ids, w13, s13, w2, s2, limit: float, block: int = FP8_BLOCK):
    """``sum_k weights[t, k] * down_e(swiglu_clamp(gate_up_e(x_t)))`` for ``e = ids[t, k]`` on the dequantized FP8
    expert weights (``w13 [E, 2I, H]``, ``w2 [E, H, I]``), expert by expert. Both GEMM inputs are quantized as vLLM's
    MoE quantizes them (power-of-two scales), straight-through, so the expert outputs the routing weights' gradient
    sees, and the point where the SwiGLU is differentiated, are the kernel's up to rounding."""
    from torch.utils.checkpoint import checkpoint

    out = x.new_zeros(x.shape, dtype=torch.float32)
    flat_ids = ids.reshape(-1).long()
    token = torch.arange(x.shape[0], device=x.device).repeat_interleave(ids.shape[1])
    flat_w = weights.reshape(-1)
    for e in torch.unique(flat_ids).tolist():
        sel = torch.nonzero(flat_ids == e).flatten()
        rows = token[sel]
        down = checkpoint(_expert, x[rows], w13[e], s13[e], w2[e], s2[e], limit, block, use_reentrant=False)
        out = out.index_add(0, rows, down * flat_w[sel].float().unsqueeze(-1))
    return out.to(x.dtype)


def ref_routed_experts_ep(x, weights, ids, *, moe, tp: int = 1):
    """:func:`ref_routed_experts`' math through the stock MoE's expert-parallel dispatch, so no rank holds every
    expert's weights. The forward's ``tp`` per-rank outputs are summed (``tp_sum``) and so get the same gradient:
    their VJPs add up to the unsharded experts', and each output here carries ``1 / tp`` of them."""
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import (
        stock_routed_experts,
    )

    out = stock_routed_experts(moe, x, ids, weights)
    return tuple(out / tp for _ in range(tp))


def _expert(x, w13, s13, w2, s2, limit, block):
    """One expert on its tokens; checkpointed so only one expert's dequantized weights are alive in the backward."""
    from arctic_platform.model.implementations.fp8 import dequantize_from_fp8_blockwise

    gate_up = fake_fp8(x, True) @ dequantize_from_fp8_blockwise(w13, s13, block, torch.float32).t()
    hidden = ref_silu_and_mul_with_clamp(gate_up.to(x.dtype), limit)
    return fake_fp8(hidden, True) @ dequantize_from_fp8_blockwise(w2, s2, block, torch.float32).t()


def ref_fp8_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, block: int = FP8_BLOCK) -> torch.Tensor:
    """vLLM's block-FP8 linear: the input quantized per 128-group with plain scales (straight-through), the
    dequantized weight."""
    from arctic_platform.model.implementations.fp8 import dequantize_from_fp8_blockwise

    w = dequantize_from_fp8_blockwise(weight, scale, block, torch.float32)
    return (fake_fp8(x, False) @ w.t()).to(x.dtype)


def ref_head_bmm(a: torch.Tensor, b: torch.Tensor, **_) -> torch.Tensor:
    """``a [heads, tokens, i] @ b [heads, i, o]`` as ``[tokens, heads, o]``."""
    return torch.bmm(a.float(), b.float()).transpose(0, 1).to(a.dtype)


def ref_q_kv_rmsnorm(q: torch.Tensor, kv: torch.Tensor, q_weight: torch.Tensor, kv_weight: torch.Tensor,
                     eps: float, **_) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    return ref_rms_norm(q, q_weight, eps), ref_rms_norm(kv, kv_weight, eps)
