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


def _mixer(layer: Any, x: torch.Tensor, ctx: Any) -> torch.Tensor:
    if layer.block_type == "linear_attention":
        raise NotImplementedError("vLLM-aligned GLM-5.3-Flash: KDA (FlashKDA prefill, recurrent decode) is not in yet")
    raise NotImplementedError("vLLM-aligned GLM-5.3-Flash: the sparse indexer and sparse MLA are not in yet")


def _mlp(layer: Any, x: torch.Tensor, ctx: Any) -> torch.Tensor:
    raise NotImplementedError("vLLM-aligned GLM-5.3-Flash: the dense MLP and the MoE are not in yet")


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
    for name in ("hc_mult", "hc_sinkhorn_iters", "hc_eps", "layer_types"):
        if getattr(config, name, None) is None:
            raise ValueError(f"the vLLM-aligned GLM-5.3-Flash forward needs config.{name}")


def apply_vllm_aligned_glm5_next(model: Any, **_kwargs) -> None:
    """Placeholder entry point: the aligned forward is incomplete, so this refuses rather than train on it."""
    del model
    raise NotImplementedError("the vLLM-aligned GLM-5.3-Flash forward is work in progress (mixers and MoE missing)")


__all__ = ["aligned_hidden", "aligned_layer", "apply_vllm_aligned_glm5_next", "hc_post", "hc_pre"]
