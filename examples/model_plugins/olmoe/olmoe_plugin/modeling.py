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

"""OLMoE trainer model: a decoder whose every layer has a mixture-of-experts MLP.

Compared with the dense OLMo-2 example, it adds the MoE block (``OlmoeSparseMoeBlock``): it routes through
``route``, so the platform can replay the sampler's routing (router replay) and the forward's own routing on
recompute, and computes experts through ``experts_core``, so the platform can shard them (expert parallelism).
Experts are stacked on dim 0 (``mlp.experts.w1`` is ``[E, I, D]``), while the checkpoint keeps one tensor per expert;
``weights.py`` converts. Attention is pre-norm, with norms on the full query and key projections.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import OlmoeConfig

from arctic_platform.model_plugin import AttentionModule
from arctic_platform.model_plugin import CausalLM
from arctic_platform.model_plugin import ExpertShard
from arctic_platform.model_plugin import InitializesBuffers
from arctic_platform.model_plugin import StackedMoEModule
from arctic_platform.model_plugin import choose_experts
from arctic_platform.model_plugin import mark

FULL_ATTENTION = "full_attention"


class RMSNorm(nn.Module):
    """RMSNorm in fp32, cast back before scaling by the weight, as Hugging Face's ``OlmoeRMSNorm``."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        return self.weight * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)


class RotaryEmbedding(nn.Module, InitializesBuffers):
    """Rotary embedding at each token's position within its packed row; ``inv_freq`` is computed, not loaded."""

    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float):
        super().__init__()
        self.dim, self.theta = dim, theta
        self.register_buffer("inv_freq", self._inv_freq(None), persistent=False)
        mark(self, dtype="fp32")

    def _inv_freq(self, device) -> torch.Tensor:
        return 1.0 / self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)

    def ap_init_buffers(self) -> None:
        self.inv_freq = self._inv_freq(self.inv_freq.device)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        angles = position_ids[0].float()[:, None] * self.inv_freq[None, :]
        cos = torch.cat((angles.cos(), angles.cos()), dim=-1)[None, :, None, :]
        sin = torch.cat((angles.sin(), angles.sin()), dim=-1)[None, :, None, :]
        first, second = x.float().chunk(2, dim=-1)
        return (x.float() * cos + torch.cat((-second, first), dim=-1) * sin).to(x.dtype)


class OlmoeAttention(AttentionModule):
    """Grouped-query attention with q/k norms and rotary positions. The core (``mix``) is the platform's."""

    ap_layer_type = FULL_ATTENTION

    def __init__(self, config: OlmoeConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.num_heads * self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.num_kv_heads * self.head_dim, config.rms_norm_eps)
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_parameters["rope_theta"])

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        q = self.q_norm(self.q_proj(x)).view(1, length, self.num_heads, self.head_dim)
        k = self.k_norm(self.k_proj(x)).view(1, length, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(1, length, self.num_kv_heads, self.head_dim)
        q, k = self.rotary(q, position_ids), self.rotary(k, position_ids)
        return self.o_proj(self.mix(q, k, v, cu_seqlens=cu_seqlens).flatten(-2))


class StackedExperts(nn.Module):
    """Routed experts stacked on dim 0, in the platform's layout: ``w1``/``w3`` gate/up ``[E, I, D]``, ``w2`` down
    ``[E, D, I]``."""

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(num_experts, intermediate_size, hidden_size) * hidden_size**-0.5)
        self.w3 = nn.Parameter(torch.randn(num_experts, intermediate_size, hidden_size) * hidden_size**-0.5)
        self.w2 = nn.Parameter(torch.randn(num_experts, hidden_size, intermediate_size) * intermediate_size**-0.5)


class OlmoeSparseMoeBlock(nn.Module, StackedMoEModule):
    """Softmax over all experts, then the top-k, weighted by their probabilities (renormalized only with
    ``norm_topk_prob``). No shared expert."""

    def __init__(self, config: OlmoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = StackedExperts(self.num_experts, config.hidden_size, config.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.reshape(-1, x.shape[-1])
        selected, weights = self.route(tokens)
        return self.experts_core(tokens, selected, weights).view_as(x)

    def route(self, tokens: torch.Tensor, routed: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Expert ids ``[N, top_k]`` and their weights. ``routed`` is the platform's: the sampler's routing to
        replay, or this forward's own on recompute; :func:`choose_experts` honours it."""
        scores = F.softmax(self.gate(tokens), dim=-1, dtype=torch.float32)
        selected = choose_experts(scores, self.top_k, routed)
        weights = scores.gather(1, selected)
        if self.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True)
        return selected, weights.to(tokens.dtype)

    def experts_core(
        self, tokens: torch.Tensor, selected: torch.Tensor, weights: torch.Tensor, *, shard: ExpertShard | None = None
    ) -> torch.Tensor:
        """Sum over each token's chosen experts that ``shard`` holds; local expert ``i`` is ``shard.start + i``."""
        shard = shard or ExpertShard.full(self.num_experts)
        out = torch.zeros_like(tokens)
        for local in range(shard.per_rank):
            rows, slots = (selected == shard.start + local).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            x = tokens[rows]
            hidden = F.silu(x @ self.experts.w1[local].T) * (x @ self.experts.w3[local].T)
            out = out.index_add(0, rows, (hidden @ self.experts.w2[local].T) * weights[rows, slots, None])
        return out


class OlmoeDecoderLayer(nn.Module):
    def __init__(self, config: OlmoeConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = OlmoeAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = OlmoeSparseMoeBlock(config)

    def forward(self, h: torch.Tensor, position_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        h = h + self.self_attn(self.input_layernorm(h), position_ids, cu_seqlens)
        return h + self.mlp(self.post_attention_layernorm(h))


class OlmoeModel(nn.Module):
    def __init__(self, config: OlmoeConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(OlmoeDecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class OlmoeForCausalLM(nn.Module, CausalLM):
    """Returns final hidden states; the platform computes logits through ``lm_head`` a chunk at a time."""

    def __init__(self, config: OlmoeConfig, options: None = None):
        super().__init__()
        self.config = config
        self.model = OlmoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, *, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h, position_ids, cu_seqlens)
        return self.model.norm(h)
