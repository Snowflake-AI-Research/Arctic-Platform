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
"""vLLM-aligned logprobs for GLM-5.3-Flash (``glm5_next``). Work in progress: the mixers and the MoE are not yet in.

The target is the sampler :mod:`arctic_platform.inference.vllm.glm5_batch_invariance` makes batch-invariant
(``VLLM_BATCH_INVARIANT=1``, ``ARCTIC_GLM5_BATCH_INVARIANT=1``, enforce_eager, no MTP, no prefix caching, prompts
prefilled whole), at tensor parallelism 1. Every per-token operator of that sampler gives a token the same bits
whatever else the step holds, so the trainer calls those kernels once on its packed tokens; the operators that mix
tokens (KDA, the sparse indexer and sparse attention) replay the sampler's prefill and decode structure.

Each decoder layer is one :class:`_AlignedLayer`: its forward runs the vLLM kernels without autograd on the layer's
input residual streams, and its backward recomputes the trainer's stock layer on those streams and backpropagates
through it, as reentrant activation checkpointing does. The logprobs are then vLLM's, and the gradient is the stock
layer's at the activations the sampler's kernels produced.

vLLM defers a layer's last hyper-connection post into the next layer's fused post + pre-norm. With the fused kernel
off (as the batch-invariant sampler runs it) that op is the standalone post kernel followed by the pre-norm GEMM and
the pre kernel on the same per-token data, so materializing the residual streams at every layer boundary, as the
stock layers carry them, gives the same bits.

Kernels the sampler runs (its kernel inventory, ``layer_digest.kernel_inventory``, on the 8-layer proxy at TP1):

* every unquantized linear (all KDA projections, MLA ``fused_qkv_a``/``q_b``/``kv_b``, the indexer's ``wq_b`` and
  ``wk_weights_proj``, every ``o_proj``): ``linear_batch_invariant``, whose tile is chosen by ``(N, K)``, so the
  trainer passes vLLM's merged weights (KDA's ``[q; k; v; b; f_a; g_a]``, MLA's ``[q_a; kv_a]``);
* dense MLP and shared-expert ``gate_up``/``down``: block-FP8 ``TritonFp8BlockScaledMMKernel`` (activations
  quantized per 128-group, row-major scales, no UE8M0) with ``SiluAndMulWithClamp``;
* routed experts: ``FusedMoEKernel`` with ``TritonExperts`` (clamp limit 10), routing by the batch-invariant
  grouped top-k on fp32 gate logits, ``routed_scaling_factor`` applied to the routed output;
* sparse attention ``FlashAttnMLASparseImpl`` (FA3, one split); KDA prefill FlashKDA, decode ``fused_recurrent_kda``;
* direct ``F.linear``/``torch.mm`` calls in model code (the indexer's kpool gate and head weights) bypass the linear
  layers; the head weights go through the batch-invariant matmul in the sampler patch.
"""

from __future__ import annotations

from typing import Any
from typing import Callable

import torch

from arctic_platform.model.implementations.vllm_aligned import ops

FP8_BLOCK = 128  # block-FP8 weight block (rows and columns)


def _mhc_constants():
    from vllm.models.glm5next.common.model import _MHC_POST_MULT_VALUE

    return _MHC_POST_MULT_VALUE


def hc_pre(streams: torch.Tensor, hc: Any, norm: Any, config: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """vLLM's ``mhc_pre`` (TileLang, one split) with the sublayer's RMSNorm fused: ``(post, comb, layer_input)``.

    ``streams`` is ``[tokens, hc_mult, hidden]`` bf16; ``hc`` the stock ``Glm5NextTextHyperConnection`` and ``norm``
    the RMSNorm that follows it in the stock layer.
    """
    return torch.ops.vllm.mhc_pre_tilelang(
        streams,
        hc.fn.float(),
        hc.scale.float(),
        hc.base.float(),
        config.rms_norm_eps,
        config.hc_eps,
        config.hc_eps,
        _mhc_constants(),
        config.hc_sinkhorn_iters,
        1,
        norm.weight,
        norm.variance_epsilon,
    )


def hc_post(x: torch.Tensor, streams: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """vLLM's ``mhc_post``: ``post_j * x + sum_i comb_ij * streams_i``, the residual streams after a sublayer."""
    return torch.ops.vllm.mhc_post_tilelang(x, streams, post, comb)


class _AlignedLayer(torch.autograd.Function):
    """One decoder layer: the aligned (vLLM-kernel) forward, the stock layer's backward."""

    @staticmethod
    def forward(ctx, streams: torch.Tensor, run_aligned: Callable, run_stock: Callable) -> torch.Tensor:
        ctx.run_stock = run_stock
        ctx.save_for_backward(streams)
        with torch.no_grad():
            return run_aligned(streams)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (streams,) = ctx.saved_tensors
        with torch.enable_grad():
            replay = streams.detach().requires_grad_(streams.requires_grad)
            out = ctx.run_stock(replay)
        torch.autograd.backward(out, grad_out)
        return (replay.grad if streams.requires_grad else None), None, None


KDA_O_NORM_EPS = 1e-5  # vLLM builds the gated output norm with FusedRMSNormGated's default eps, not rms_norm_eps


def _kda_decode(core, qkv, g1, beta, conv_states, recurrent, conv_weight, gate, starts, prompt_lens, lengths, slots,
                heads: int, head_dim: int, config: Any) -> None:  # fmt: skip
    """Write the decoded tokens' KDA core output into ``core``, one decode step at a time over every request still
    decoding, as vLLM's plain decode step runs them: ``causal_conv1d_update`` on the request's conv state slot, then
    ``fused_recurrent_kda`` (gate and beta sigmoid in the kernel) updating its fp32 recurrent state slot in place."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda

    projection = heads * head_dim
    a_log = gate.A_log.float().reshape(1, 1, heads, 1)
    g_bias = gate.dt_bias.float().reshape(-1)
    steps = lengths - prompt_lens
    for step in range(int(steps.max())):
        active = torch.nonzero(steps > step).flatten()
        rows = starts[active] + prompt_lens[active] + step
        batch = rows.numel()
        conv = causal_conv1d_update(qkv[rows], conv_states, conv_weight, None, activation="silu",
                                    conv_state_indices=slots[active])  # fmt: skip
        q, k, v = (t.reshape(1, batch, heads, head_dim) for t in conv.split(projection, dim=-1))
        out, _ = fused_recurrent_kda(
            q=q, k=k, v=v, g=g1[:, rows], beta=beta[rows].unsqueeze(0), initial_state=recurrent,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=torch.arange(batch + 1, dtype=torch.int32, device=qkv.device),
            ssm_state_indices=slots[active], sigmoid_beta=True, a_log=a_log, g_bias=g_bias, compute_gate=True,
            lower_bound=config.linear_lower_bound,
        )  # fmt: skip
        core[:, rows] = out.to(core.dtype)


def kda_prefill(attn: Any, x: torch.Tensor, cu_seqlens: torch.Tensor, config: Any, trace: dict | None = None) -> torch.Tensor:
    """:func:`kda_forward` with every token prefilled."""
    return kda_forward(attn, x, cu_seqlens, config, trace=trace)


def kda_forward(attn: Any, x: torch.Tensor, cu_seqlens: torch.Tensor, config: Any,
                prompt_lens: torch.Tensor | None = None, trace: dict | None = None) -> torch.Tensor:  # fmt: skip
    """vLLM's ``Glm5NextLinearAttention`` on packed requests: each prompt prefilled whole, then the rest decoded.

    ``x`` is ``[tokens, hidden]`` (the layer input after the hyper-connection pre-norm), ``cu_seqlens`` the int32
    request boundaries and ``prompt_lens`` how many leading tokens of each request were prefilled (all by default).
    The merged ``[q; k; v; b; f_a; g_a]`` projection; over each prompt the merged q|k|v causal conv with vLLM's fp32
    conv weights and FlashKDA on the raw gate and beta logits; over each later token, one decode step at a time, the
    conv update and the recurrent kernel from the conv state and the fp32 recurrent state the request has reached;
    then the sigmoid-gated RMSNorm and ``o_proj``. ``trace`` (debug) collects the prefill's intermediate tensors.
    """
    trace = {} if trace is None else trace
    import vllm._flashkda_C  # noqa: F401
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated

    heads, head_dim = attn.num_heads, attn.head_dim
    projection = heads * head_dim
    tokens, num_seqs = x.shape[0], cu_seqlens.numel() - 1
    starts, lengths = cu_seqlens[:-1].long(), (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    prompt_lens = lengths if prompt_lens is None else prompt_lens.long()
    decoded = bool((prompt_lens < lengths).any())
    if decoded:
        seq_of_token = torch.repeat_interleave(torch.arange(num_seqs, device=x.device), lengths)
        in_prompt = torch.arange(tokens, device=x.device) - starts[seq_of_token] < prompt_lens[seq_of_token]
        prompt_rows = torch.nonzero(in_prompt).flatten()
        prompt_cu = torch.cat([cu_seqlens.new_zeros(1), torch.cumsum(prompt_lens, 0).to(cu_seqlens.dtype)])
    else:
        prompt_rows, prompt_cu = None, cu_seqlens
    gate = attn.forget_gate
    weight = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight, attn.b_proj.weight,
                        gate.f_a_proj.weight, attn.g_a_proj.weight])  # fmt: skip
    projected = ops.linear_kernel(x, weight)
    qkv, beta, f_a, g_a = projected.split([3 * projection, heads, head_dim, head_dim], dim=-1)
    g1 = ops.linear_kernel(f_a, gate.f_b_proj.weight).reshape(1, tokens, heads, head_dim)
    g2 = ops.linear_kernel(g_a, attn.g_b_proj.weight).reshape(tokens, heads, head_dim)

    conv_weight = attn.conv1d.weight.float().reshape(3 * projection, -1)
    # State slot 0 is vLLM's null block, which the conv kernel treats as padding, so the slots start at 1.
    slots = torch.arange(1, num_seqs + 1, dtype=torch.int32, device=x.device)
    conv_states = x.new_zeros((num_seqs + 1, 3 * projection, conv_weight.shape[-1] - 1))
    qkv_p = qkv if prompt_rows is None else qkv[prompt_rows]
    g1_p = g1 if prompt_rows is None else g1[:, prompt_rows]
    beta_p = beta if prompt_rows is None else beta[prompt_rows]
    prompt_tokens = qkv_p.shape[0]
    conv = causal_conv1d_fn(
        qkv_p.transpose(0, 1),
        conv_weight,
        None,
        activation="silu",
        conv_states=conv_states,
        has_initial_state=torch.zeros(num_seqs, dtype=torch.bool, device=x.device),
        cache_indices=slots,
        query_start_loc=prompt_cu,
    ).transpose(0, 1)
    q, k, v = (t.reshape(1, prompt_tokens, heads, head_dim).contiguous() for t in conv.split(projection, dim=-1))
    trace.update(conv_in=qkv_p.transpose(0, 1), conv_weight=conv_weight, conv_out=conv, q=q, k=k, v=v, g=g1_p,
                 beta=beta_p.unsqueeze(0))  # fmt: skip

    state = torch.zeros((num_seqs, heads, head_dim, head_dim), dtype=torch.float32, device=x.device)
    final_state = torch.empty_like(state)
    workspace = torch.empty(torch.ops._flashkda_C.get_workspace_size(prompt_tokens, heads, num_seqs),
                            dtype=torch.uint8, device=x.device)  # fmt: skip
    core_p = torch.empty((1, prompt_tokens, heads, head_dim), dtype=x.dtype, device=x.device)
    torch.ops._flashkda_C.fwd(
        q, k, v, g1_p.contiguous(), beta_p.unsqueeze(0), head_dim**-0.5, core_p, workspace,
        gate.A_log.float().view(-1), gate.dt_bias.float().view(-1, head_dim), config.linear_lower_bound,
        state, final_state, prompt_cu.contiguous(), None, None,
    )  # fmt: skip
    if prompt_rows is None:
        core = core_p
    else:
        core = torch.empty((1, tokens, heads, head_dim), dtype=x.dtype, device=x.device)
        core[:, prompt_rows] = core_p
        recurrent = torch.zeros((num_seqs + 1, heads, head_dim, head_dim), dtype=torch.float32, device=x.device)
        recurrent[1:] = final_state
        _kda_decode(core, qkv, g1, beta, conv_states, recurrent, conv_weight, gate, starts, prompt_lens, lengths,
                    slots, heads, head_dim, config)  # fmt: skip
    trace.update(core=core, g2=g2)
    core = rms_norm_gated(core, g2, attn.o_norm.weight, None, "sigmoid", eps=KDA_O_NORM_EPS)
    trace.update(normed=core)
    return ops.linear_kernel(core.reshape(tokens, -1), attn.o_proj.weight)


SPARSE_TOPK_BLOCK = 128  # sparse MLA tiles the top-k index buffer in 128 columns


def topk_buffer_width(config: Any) -> int:
    """vLLM's top-k index buffer width: ``index_topk`` plus the incomplete pool tail, rounded up to 128."""
    width = config.index_topk + (config.index_kpool - 1 if config.index_kpool > 1 else 0)
    return -(-width // SPARSE_TOPK_BLOCK) * SPARSE_TOPK_BLOCK


def causal_topk(positions: torch.Tensor, width: int) -> torch.Tensor:
    """vLLM's short-prefill top-k: every earlier token of the request, ``-1`` past the query's position."""
    columns = torch.arange(width, device=positions.device, dtype=torch.int32)
    rows = columns.expand(positions.shape[0], width).clone()
    rows[columns[None, :] > positions.to(torch.int32)[:, None]] = -1
    return rows


INDEXER_K_NORM_EPS = 1e-6
INDEXER_PADDED_HEADS = 32


def indexer_topk(indexer: Any, x: torch.Tensor, q_c: torch.Tensor, positions: torch.Tensor, cu_seqlens: torch.Tensor,
                 config: Any, width: int, prompt_lens: torch.Tensor | None = None) -> torch.Tensor:  # fmt: skip
    """vLLM's kpool sparse-indexer top-k for whole prompts: ``[tokens, width]`` request-relative token ids, ``-1`` padded.

    A token whose position is below ``index_topk`` gets its causal row (every earlier token). The others score the
    request's complete pools (``(pos + 1) // index_kpool`` of them) as the sampler's prefill does: ``wq_b`` and the
    fused ``[wk; weights_proj]`` projection by the batch-invariant matmul, the fp32 head weights by the bf16-operand
    batch-invariant matmul, the key LayerNorm, the Hadamard-rotated FP8 query, the softmax(gate + ape)-weighted pool
    compression to FP8, DeepGEMM's FP8 MQA logits, the deterministic top ``index_topk // index_kpool`` pools, and
    their expansion to tokens with the incomplete pool's tail appended.
    """
    import torch.nn.functional as F
    from vllm.models.glm5next.nvidia.ops import kpool_compress as kpool_ops
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    from arctic_platform.inference.vllm.batch_invariant_fp32_logits import matmul_fp32_out
    from arctic_platform.inference.vllm.glm5_batch_invariance import deterministic_topk

    topk_tokens, kpool = config.index_topk, config.index_kpool
    out = causal_topk(positions, width)
    long_rows = torch.nonzero(positions >= topk_tokens).flatten()
    if long_rows.numel() == 0:
        return out
    if prompt_lens is not None and bool((positions[long_rows] >= prompt_lens[_sequence_of_tokens(cu_seqlens)[long_rows]]).any()):
        raise NotImplementedError("vLLM-aligned sparse indexer: decoded tokens at or past index_topk (not in yet)")
    if getattr(config, "index_rope_head_dim", 0) or getattr(config, "qk_rope_head_dim", 0):
        raise NotImplementedError("vLLM-aligned sparse indexer covers mla_use_nope (no rotary embedding)")
    tokens = x.shape[0]
    head_dim, n_head = indexer.wk.weight.shape[0], indexer.weights_proj.weight.shape[0]

    q = ops.linear_kernel(q_c, indexer.wq_b.weight).view(tokens, n_head, head_dim)
    kw = ops.linear_kernel(x, torch.cat([indexer.wk.weight, indexer.weights_proj.weight]))
    weights = matmul_fp32_out(x, indexer.weights_proj.weight.t().contiguous())
    k = F.layer_norm(kw[:, :head_dim].float(), (head_dim,), indexer.k_norm.weight.float(),
                     indexer.k_norm.bias.float(), INDEXER_K_NORM_EPS).to(kw.dtype)  # fmt: skip
    q_fp8, q_scale = kpool_ops.fwht128_quant_fp8(q.reshape(-1, head_dim))
    q_fp8, q_scale = q_fp8.view(tokens, n_head, head_dim), q_scale.view(tokens, n_head, 1)
    weights = (weights.unsqueeze(-1) * q_scale * (head_dim**-0.5 * n_head**-0.5)).squeeze(-1)
    gate_score = F.linear(x, indexer.index_kpool_compress_gate)
    if n_head < INDEXER_PADDED_HEADS:
        pad = INDEXER_PADDED_HEADS - n_head
        q_fp8 = torch.cat([q_fp8, q_fp8.new_zeros(tokens, pad, head_dim)], dim=1)
        weights = torch.cat([weights, weights.new_zeros(tokens, pad)], dim=1)

    starts, lengths = cu_seqlens[:-1].long(), (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    pools = lengths // kpool
    pool_starts = torch.cumsum(pools, 0) - pools
    seq = torch.repeat_interleave(torch.arange(pools.numel(), device=x.device), pools)
    first = starts[seq] + kpool * (torch.arange(int(pools.sum()), device=x.device) - pool_starts[seq])
    slots = first[:, None] + torch.arange(kpool, device=x.device)[None, :]
    k_pool, k_scale = kpool_ops.kpool_compress_and_write_cache(
        torch.empty((1, 64, head_dim + 4), dtype=torch.uint8, device=x.device),
        k[slots], gate_score[slots], indexer.index_kpool_compress_ape.float(),
        torch.zeros(slots.shape[0], dtype=torch.int64, device=x.device), pool_size=kpool, head_dim=head_dim,
        round_scale=True, return_compressed=True, write_cache=False,
    )  # fmt: skip

    seq_of_token = torch.repeat_interleave(torch.arange(lengths.numel(), device=x.device), lengths)[long_rows]
    ks = pool_starts[seq_of_token].to(torch.int32)
    ke = ks + ((positions[long_rows] + 1) // kpool).to(torch.int32)
    logits = fp8_fp4_mqa_logits((q_fp8[long_rows], None), (k_pool, k_scale), weights[long_rows], ks, ke,
                                clean_logits=False)  # fmt: skip
    pool_topk = torch.empty((long_rows.numel(), topk_tokens // kpool), dtype=torch.int32, device=x.device)
    deterministic_topk(logits, ks, ke, topk_tokens // kpool, pool_topk)
    expanded = kpool_ops.expand_pools_and_append_tail(pool_topk.to(torch.int64),
                                                      (positions[long_rows] + 1).to(torch.int32), kpool)  # fmt: skip
    rows = torch.full((long_rows.numel(), width), -1, dtype=torch.int32, device=x.device)
    rows[:, : expanded.shape[-1]] = expanded.to(torch.int32)
    out[long_rows] = rows
    return out


def mla_kv_up_weights(kv_b_proj: Any, config: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's absorbed MLA weights from ``kv_b_proj``: ``W_UK_T [N, P, L]`` and ``W_UV [N, L, V]``."""
    heads, latent = config.num_attention_heads, config.kv_lora_rank
    nope, value = config.qk_nope_head_dim, config.v_head_dim
    weight = kv_b_proj.weight.t().reshape(latent, heads, nope + value)
    w_uk, w_uv = weight.split([nope, value], dim=-1)
    return w_uk.permute(1, 2, 0).contiguous(), w_uv.transpose(0, 1).contiguous()


def dequant_fp8_block(weight: torch.Tensor, scale_inv: torch.Tensor, block: int = FP8_BLOCK) -> torch.Tensor:
    """vLLM's load-time dequantization of a block-FP8 weight it keeps in bf16 (``_dequant_fp8_block``)."""
    rows, cols = weight.shape
    padded = torch.nn.functional.pad(weight, (0, (-cols) % block, 0, (-rows) % block))
    scale = scale_inv.to(torch.float32).repeat_interleave(block, 0).repeat_interleave(block, 1)
    return (padded.to(torch.float32) * scale).to(torch.bfloat16)[:rows, :cols].contiguous()


def bf16_weight(linear: Any) -> torch.Tensor:
    """A projection's weight as vLLM serves it in bf16: dequantized if the trainer holds it in block FP8."""
    scale_inv = getattr(linear, "weight_scale_inv", None)
    if linear.weight.element_size() == 1 and scale_inv is not None:
        return dequant_fp8_block(linear.weight, scale_inv)
    return linear.weight


def mla_attention(attn: Any, x: torch.Tensor, positions: torch.Tensor, cu_seqlens: torch.Tensor,
                  config: Any, prompt_lens: torch.Tensor | None = None) -> torch.Tensor:  # fmt: skip
    """vLLM's ``Glm5NextMLAAttention`` (sparse MLA with ``mla_use_nope``) on prompts no longer than ``index_topk``.

    Such prompts take vLLM's short-prefill path: every token attends to all earlier tokens of its request, with no
    indexer scoring. With ``mla_use_nope`` vLLM skips the rotary embedding and pads ``kv_a_proj_with_mqa`` with
    zero rows up to ``kv_lora_rank + qk_rope_head_dim``, so the rope part of every key is zero. The block-FP8
    ``q_a``/``kv_a``/``q_b``/``o`` projections are dequantized to bf16 as vLLM does at load. Then: the merged
    ``[q_a; kv_a]`` projection, the fused q/kv RMSNorm, ``q_b_proj``, the query absorbed into the latent space by
    ``W_UK_T``, FA3 sparse attention over one-token pages of ``[kv_c | k_pe]`` rows (QV form), the value
    up-projection by ``W_UV`` and ``o_proj``.
    """
    from vllm.models.common.ops import fused_q_kv_rmsnorm
    from vllm.v1.attention.backends.mla import flashattn_mla_sparse

    if not getattr(config, "mla_use_nope", True) and config.qk_rope_head_dim:
        raise NotImplementedError("vLLM-aligned sparse MLA covers mla_use_nope (no rotary embedding)")
    starts, lengths = cu_seqlens[:-1], cu_seqlens[1:] - cu_seqlens[:-1]
    heads, latent, rope = config.num_attention_heads, config.kv_lora_rank, config.qk_rope_head_dim
    nope, value = config.qk_nope_head_dim, config.v_head_dim
    tokens = x.shape[0]

    kv_a = bf16_weight(attn.kv_a_proj_with_mqa)
    if kv_a.shape[0] == latent and rope:
        kv_a = torch.cat([kv_a, kv_a.new_zeros((rope, kv_a.shape[1]))])
    qkv_lora = ops.linear_kernel(x, torch.cat([bf16_weight(attn.q_a_proj), kv_a]))
    q_c, kv_lora = qkv_lora.split([config.q_lora_rank, latent + rope], dim=-1)
    kv_c, k_pe = kv_lora.split([latent, rope], dim=-1)
    q_c, kv_c = fused_q_kv_rmsnorm(q_c, kv_c, attn.q_a_layernorm.weight, attn.kv_a_layernorm.weight,
                                   attn.q_a_layernorm.variance_epsilon)  # fmt: skip
    q = ops.linear_kernel(q_c, bf16_weight(attn.q_b_proj)).view(tokens, heads, nope + rope)
    q_nope, q_pe = q.split([nope, rope], dim=-1)

    w_uk_t, w_uv = mla_kv_up_weights(attn.kv_b_proj, config)
    q_latent = q.new_empty((tokens, heads, latent))
    torch.bmm(q_nope.transpose(0, 1), w_uk_t, out=q_latent.transpose(0, 1))

    seq_of_token = torch.repeat_interleave(torch.arange(starts.numel(), device=x.device), lengths)
    topk = indexer_topk(attn.indexer, x, q_c, positions, cu_seqlens, config, topk_buffer_width(config), prompt_lens)
    rows = torch.where(topk >= 0, topk + starts.to(torch.int32)[seq_of_token, None], topk)
    valid = (topk >= 0).sum(-1).to(torch.int32)
    kv_rows = torch.cat([kv_c, k_pe], dim=-1)
    v_cache = kv_rows[:, :latent].unsqueeze(1).unsqueeze(1)
    if rope:
        k_cache, q_rope = kv_rows[:, latent:].unsqueeze(1).unsqueeze(1), q_pe
    else:  # FA3's QV path needs a 64-wide Q/K: a zero query against the first 64 latent columns
        k_cache, q_rope = v_cache[..., :64], q.new_zeros((tokens, heads, 64))
    out = flashattn_mla_sparse.flash_attn_varlen_func(
        q=q_rope,
        k=k_cache,
        v=v_cache,
        q_v=q_latent,
        max_seqlen_q=1,
        cu_seqlens_q=torch.arange(tokens + 1, dtype=torch.int32, device=x.device),
        max_seqlen_k=rows.shape[1],
        seqused_k=valid,
        block_table=rows,
        softmax_scale=(nope + rope) ** -0.5,
        causal=True,
        fa_version=3,
    )
    up = out.new_empty((tokens, heads, value))
    torch.bmm(out.view(-1, heads, latent).transpose(0, 1), w_uv, out=up.transpose(0, 1))
    return ops.linear_kernel(up.view(tokens, heads * value), bf16_weight(attn.o_proj))


def _mixer(layer: Any, x: torch.Tensor, ctx: Any) -> torch.Tensor:
    prompt_lens = getattr(ctx, "prompt_lens", None)
    if layer.block_type == "linear_attention":
        return kda_forward(layer.self_attn, x, ctx.cu_seqlens, ctx.config, prompt_lens)
    return mla_attention(layer.self_attn, x, ctx.positions, ctx.cu_seqlens, ctx.config, prompt_lens)


def fp8_block_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale_inv: torch.Tensor) -> torch.Tensor:
    """vLLM's block-FP8 linear (``TritonFp8BlockScaledMMKernel``): per-128-group activation quant, Triton GEMM."""
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    from vllm.platforms import current_platform

    x2d = x.reshape(-1, x.shape[-1])
    q, scale = fp8_utils.per_token_group_quant_fp8(
        x2d,
        group_size=FP8_BLOCK,
        column_major_scales=False,
        tma_aligned_scales=False,
        dtype=current_platform.fp8_dtype(),
        use_ue8m0=False,
    )
    out = torch.ops.vllm.w8a8_triton_block_scaled_mm_func(
        q, weight, scale, weight_scale_inv, [FP8_BLOCK, FP8_BLOCK], x.dtype
    )
    return out.view(*x.shape[:-1], weight.shape[0])


def silu_and_mul_with_clamp(x: torch.Tensor, limit: float) -> torch.Tensor:
    """vLLM's ``SiluAndMulWithClamp``: ``silu(min(gate, limit)) * clamp(up, -limit, limit)`` on ``[gate | up]``."""
    ops._custom_ops()
    out = torch.empty(*x.shape[:-1], x.shape[-1] // 2, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul_with_clamp(out, x.contiguous(), float(limit), 1.0, 0.0)
    return out


def _merged_fp8(*linears: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's merged column-parallel FP8 weight and block scales: the pieces stacked along the output rows."""
    for linear in linears[:-1]:
        if linear.weight.shape[0] % FP8_BLOCK:
            raise ValueError(f"merging FP8 projections needs {FP8_BLOCK}-row pieces, got {linear.weight.shape[0]}")
    return torch.cat([linear.weight for linear in linears]), torch.cat([linear.weight_scale_inv for linear in linears])


def dense_mlp(mlp: Any, x: torch.Tensor) -> torch.Tensor:
    """vLLM's ``Glm5NextMLP``: merged FP8 ``gate_up``, the clamped SwiGLU, FP8 ``down``."""
    weight, scale = _merged_fp8(mlp.gate_proj, mlp.up_proj)
    hidden = silu_and_mul_with_clamp(fp8_block_linear(x, weight, scale), mlp.swiglu_limit)
    return fp8_block_linear(hidden, mlp.down_proj.weight, mlp.down_proj.weight_scale_inv)


def shared_expert_mlp(ffn: Any, x: torch.Tensor, swiglu_limit: float) -> torch.Tensor:
    """vLLM's shared-expert ``Glm5NextMLP`` on the trainer's ``BCFeedForward`` (``w1`` gate, ``w3`` up, ``w2`` down)."""
    weight = torch.cat([ffn.w1, ffn.w3])
    scale = torch.cat([ffn.w1_scale_inv, ffn.w3_scale_inv])
    if ffn.w1.shape[0] % FP8_BLOCK:
        raise ValueError(f"merging FP8 projections needs {FP8_BLOCK}-row pieces, got {ffn.w1.shape[0]}")
    hidden = silu_and_mul_with_clamp(fp8_block_linear(x, weight, scale), swiglu_limit)
    return fp8_block_linear(hidden, ffn.w2, ffn.w2_scale_inv)


def _full(tensor: torch.Tensor, num_experts: int, ep_group: Any = None) -> torch.Tensor:
    """All ``num_experts`` experts of an expert-parallel weight: a ``DTensor``'s full value, or the ranks' local
    slices (contiguous expert blocks, as DeepSpeed holds them) gathered in rank order over ``ep_group``.

    vLLM at TP 1 sums a token's top-k expert outputs in one kernel; splitting that sum over expert-parallel ranks
    would round differently, so every rank runs all of a token's experts on the gathered weights.
    """
    full_tensor = getattr(tensor, "full_tensor", None)
    if callable(full_tensor):
        return full_tensor()
    if tensor.shape[0] == num_experts:
        return tensor
    if ep_group is None:
        raise ValueError(f"the aligned MoE needs the expert-parallel group to gather {num_experts} experts")
    world = torch.distributed.get_world_size(ep_group)
    if tensor.shape[0] * world != num_experts:
        raise ValueError(f"{tensor.shape[0]} local experts on {world} ranks do not make {num_experts}")
    gathered = tensor.new_empty((num_experts, *tensor.shape[1:]))
    torch.distributed.all_gather_into_tensor(gathered, tensor.contiguous(), group=ep_group)
    return gathered


def _triton_experts(num_experts: int, top_k: int, hidden: int, intermediate: int, swiglu_limit: float, w13_scale,
                    w2_scale, device):  # fmt: skip
    """vLLM's routed-expert kernel for block-FP8 GLM-5.3-Flash at TP 1: ``FusedMoEKernel`` with ``TritonExperts``."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import make_fp8_moe_quant_config
    from vllm.model_executor.layers.fused_moe.all2all_utils import maybe_make_prepare_finalize

    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device=device,
        routing_method=RoutingMethodType.DeepSeekV3,
        swiglu_limit=swiglu_limit,
    )
    quant_config = make_fp8_moe_quant_config(
        fp8_backend=Fp8MoeBackend.TRITON,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        a1_scale=None,
        a2_scale=None,
        block_shape=[FP8_BLOCK, FP8_BLOCK],
        swiglu_limit=swiglu_limit,
        gemm1_alpha=None,
        gemm1_beta=None,
        layer=None,
    )
    prepare = maybe_make_prepare_finalize(
        moe=moe_config, quant_config=quant_config, allow_new_interface=True, use_monolithic=False
    )
    return FusedMoEKernel(prepare, TritonExperts(moe_config, quant_config)), MoEActivation.SILU


def moe(mlp: Any, x: torch.Tensor, config: Any, ep_group: Any = None) -> torch.Tensor:
    """vLLM's ``Glm5NextMoE`` at TP 1 on the trainer's ``MoE``: fp32 router logits by the batch-invariant matmul, the
    batch-invariant grouped top-k (renormalized and scaled by ``routed_scaling_factor`` there, as vLLM's router does
    when the runner does not scale the output), the Triton routed experts, then ``shared + routed`` in bf16."""
    from vllm.v1.worker.workspace import init_workspace_manager
    from vllm.v1.worker.workspace import is_workspace_manager_initialized

    from arctic_platform.inference.vllm.batch_invariant_fp32_logits import linear_fp32_out
    from arctic_platform.inference.vllm.glm5_batch_invariance import grouped_topk

    if not is_workspace_manager_initialized():  # the fused-MoE kernels take their scratch from vLLM's workspace
        init_workspace_manager(x.device)
    logits = linear_fp32_out(x, mlp.router.gate.weight)
    weights, ids = grouped_topk(
        x,
        logits,
        topk=config.num_experts_per_tok,
        renormalize=config.norm_topk_prob,
        num_expert_group=config.n_group,
        topk_group=config.topk_group,
        scoring_func=config.scoring_func,
        routed_scaling_factor=config.routed_scaling_factor,
        e_score_correction_bias=mlp.expert_bias,
    )
    experts = mlp.experts
    total = config.n_routed_experts
    w1, w2, w3 = (_full(t, total, ep_group) for t in (experts.w1, experts.w2, experts.w3))
    s1, s2, s3 = (_full(t, total, ep_group) for t in (experts.w1_scale_inv, experts.w2_scale_inv, experts.w3_scale_inv))
    w13, w13_scale = torch.cat([w1, w3], dim=1), torch.cat([s1, s3], dim=1)
    num_experts, intermediate, hidden = w1.shape
    # The kernel captures the scales in its quant config, so it is rebuilt for every call (cheap, host-side only).
    kernel, activation = _triton_experts(num_experts, config.num_experts_per_tok, hidden, intermediate,
                                         config.swiglu_limit, w13_scale, s2, x.device)  # fmt: skip
    routed = kernel.apply(
        x.contiguous(), w13, w2, weights, ids, activation=activation, global_num_experts=num_experts,
        expert_map=None, apply_router_weight_on_input=False,
    )  # fmt: skip
    return shared_expert_mlp(mlp.shared_expert, x, config.swiglu_limit) + routed


def _mlp(layer: Any, x: torch.Tensor, ctx: Any) -> torch.Tensor:
    if hasattr(layer.mlp, "gate_proj"):
        return dense_mlp(layer.mlp, x)
    return moe(layer.mlp, x, ctx.config, getattr(ctx, "ep_group", None))


def aligned_layer(layer: Any, streams: torch.Tensor, config: Any, ctx: Any) -> torch.Tensor:
    """The sampler's computation of one decoder layer on ``[tokens, hc_mult, hidden]`` residual streams."""
    post, comb, x = hc_pre(streams, layer.attn_hc, layer.input_layernorm, config)
    streams = hc_post(_mixer(layer, x, ctx), streams, post, comb)
    post, comb, x = hc_pre(streams, layer.ffn_hc, layer.post_attention_layernorm, config)
    return hc_post(_mlp(layer, x, ctx), streams, post, comb)


def aligned_hidden(backbone: Any, input_ids: torch.Tensor, config: Any, ctx: Any, run_stock: Callable) -> torch.Tensor:
    """The sampler's final hidden states (after ``hc_contract`` and the final RMSNorm) for packed ``input_ids``.

    ``run_stock(layer, streams)`` runs the stock layer on ``[tokens, hc_mult, hidden]`` streams for the backward.
    """
    embeds = backbone.embed_tokens(input_ids.reshape(-1))
    streams = embeds.unsqueeze(1).expand(-1, config.hc_mult, -1).contiguous()
    for layer in backbone.layers:
        streams = _AlignedLayer.apply(
            streams,
            lambda s, layer=layer: aligned_layer(layer, s, config, ctx),
            lambda s, layer=layer: run_stock(layer, s),
        )
    hidden = streams.mean(dim=1)
    return ops.rms_norm(hidden, backbone.norm.weight, backbone.norm.variance_epsilon, fused=False)


def check_supported_config(config: Any) -> None:
    for name in ("hc_mult", "hc_sinkhorn_iters", "hc_eps", "layer_types", "index_topk", "index_kpool"):
        if getattr(config, name, None) is None:
            raise ValueError(f"the vLLM-aligned GLM-5.3-Flash forward needs config.{name}")
    if getattr(config, "qk_rope_head_dim", 0) and not getattr(config, "mla_use_nope", False):
        raise ValueError("the vLLM-aligned GLM-5.3-Flash forward covers sparse MLA with mla_use_nope")


def _stock_layer_runner(backbone: Any, input_ids: torch.Tensor, position_ids: torch.Tensor, kwargs: dict) -> Callable:
    """``run_stock(layer, streams)``: the stock decoder layer on ``[tokens, hc_mult, hidden]`` streams, packed as the
    stock model forward packs them (one row, the attention-mask mapping and the packing kwargs it passes down)."""
    mask = torch.ones(1, input_ids.numel(), dtype=torch.bool, device=input_ids.device)
    masks = {"deepseek_sparse_attention": mask, "linear_attention": mask}

    def run(layer: Any, streams: torch.Tensor) -> torch.Tensor:
        out, _ = layer(
            streams.unsqueeze(0),
            attention_mask=masks[layer.block_type],
            position_ids=position_ids.reshape(1, -1),
            position_embeddings=None,
            input_ids=input_ids.reshape(1, -1),
            prev_topk_indices=None,
            **kwargs,
        )
        return out.squeeze(0)

    return run


def _sequence_of_tokens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    return torch.repeat_interleave(torch.arange(lengths.numel(), device=cu_seqlens.device), lengths)


def prompt_lengths(trained: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """How many leading tokens of each packed request the sampler prefilled: through its first trained position, whose
    logits come from the prefill (every token when nothing is trained); the tokens after it were decoded."""
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    seq = _sequence_of_tokens(cu_seqlens)
    offset = torch.arange(seq.numel(), device=seq.device) - cu_seqlens[:-1].long()[seq]
    first = torch.where(trained.reshape(-1).bool(), offset, lengths[seq])
    first = torch.full_like(lengths, torch.iinfo(torch.long).max).scatter_reduce(0, seq, first, "amin")
    return torch.minimum(first + 1, lengths)


def aligned_logprobs(model: Any, input_ids: torch.Tensor, position_ids: torch.Tensor, labels: torch.Tensor,
                     trained: torch.Tensor, cu_seqlens: torch.Tensor | None, temperature: float,
                     kwargs: dict) -> torch.Tensor:  # fmt: skip
    """The sampler's logprob of each ``labels`` token (``[tokens]``, fp32): each request's prompt prefilled whole
    through its first ``trained`` position, the tokens after it decoded."""
    import types

    backbone = model.model.language_model
    config = backbone.config
    if position_ids.dim() == 3:
        position_ids = position_ids[0]
    layout = ops.PackedLayout.build(position_ids, block_size=1, cu_seqlens=cu_seqlens)
    cu = layout.cu_seqlens.to(torch.int32)
    ctx = types.SimpleNamespace(config=config, positions=layout.positions, cu_seqlens=cu,
                                prompt_lens=prompt_lengths(trained, cu),
                                ep_group=getattr(model, "_vllm_aligned_ep_group", None))  # fmt: skip
    run_stock = _stock_layer_runner(backbone, input_ids, position_ids, kwargs)
    hidden = aligned_hidden(backbone, input_ids, config, ctx, run_stock)
    return ops.lm_head_logprobs(
        hidden,
        model.lm_head.weight,
        labels.reshape(-1),
        temperature=temperature,
        token_chunk_size=model._vllm_aligned_token_chunk_size,
        fp32=model._vllm_aligned_lm_head_fp32,
    )


def _causal_lm_forward(self, input_ids=None, attention_mask=None, position_ids=None, labels=None, temperature=None,
                       dss_compute_logprobs: bool = False, **kwargs):  # fmt: skip
    from arctic_platform.model.implementations.vllm_aligned.qwen3 import _scalar_temperature
    from arctic_platform.model.implementations.vllm_aligned.qwen3_5 import LOSS_MASK_KWARG
    from arctic_platform.model.implementations.vllm_aligned.turn_plan import PLAN_KWARG

    loss_mask = kwargs.pop(LOSS_MASK_KWARG, None)
    if kwargs.pop(PLAN_KWARG, None) is not None:
        raise NotImplementedError("the vLLM-aligned GLM-5.3-Flash forward covers single-turn rows (no turn plan)")
    if not dss_compute_logprobs:
        return self._vllm_aligned_stock_forward(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels, **kwargs
        )
    if labels is None or input_ids is None:
        raise ValueError("dss_compute_logprobs requires input_ids and labels")
    if position_ids is None:
        position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device).expand_as(input_ids)
    labels = labels.to(input_ids.device)
    trained = loss_mask if loss_mask is not None else labels != -100
    logprobs = aligned_logprobs(
        self,
        input_ids,
        position_ids,
        labels,
        trained.to(input_ids.device),
        kwargs.get("cu_seq_lens_q"),
        _scalar_temperature(temperature),
        kwargs,
    )
    return {"logprobs": logprobs.view(input_ids.shape)}


def apply_vllm_aligned_glm5_next(model: Any, *, token_chunk_size: int, lm_head_fp32: bool = False,
                                 ep_group: Any = None) -> None:  # fmt: skip
    """Make a GLM-5.3-Flash model's ``dss_compute_logprobs`` forward compute the batch-invariant sampler's logprobs.

    Covers prompts prefilled whole at TP 1 (milestone 1); decoded tokens and the indexer for prompts longer than
    ``index_topk`` are not in yet.
    """
    import types

    from arctic_platform.inference.vllm.glm5_batch_invariance import ensure_glm5_batch_invariance

    config = model.model.language_model.config
    check_supported_config(config)
    ops.require_batch_invariant_env()
    import os

    os.environ.setdefault("ARCTIC_GLM5_BATCH_INVARIANT", "1")
    ensure_glm5_batch_invariance()
    model._vllm_aligned_ep_group = ep_group
    model._vllm_aligned_token_chunk_size = int(token_chunk_size)
    model._vllm_aligned_lm_head_fp32 = bool(lm_head_fp32)
    model._vllm_aligned_stock_forward = model.forward
    model.forward = types.MethodType(_causal_lm_forward, model)
    model._vllm_aligned = True
    model._vllm_aligned_reads_loss_mask = True


__all__ = ["aligned_hidden", "aligned_layer", "apply_vllm_aligned_glm5_next", "hc_post", "hc_pre"]
