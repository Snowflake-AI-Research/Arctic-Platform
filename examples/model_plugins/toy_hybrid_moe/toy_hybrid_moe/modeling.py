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

"""ToyHybridMoE trainer model: attention and gated-recurrence layers, dense and mixture-of-experts MLPs.

Plain single-device PyTorch. Nothing here calls ``torch.distributed``: the platform makes the sequence-mixing cores
sequence-parallel through ``mix`` and shards the experts through the plugin's expert layer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from arctic_platform.model_plugin import FULL_SHARD
from arctic_platform.model_plugin import ROUTER_REPLAY_FRESH
from arctic_platform.model_plugin import AttentionModule
from arctic_platform.model_plugin import HeadShard
from arctic_platform.model_plugin import combine_tokens
from arctic_platform.model_plugin import gather_tokens
from arctic_platform.model_plugin import skip_weight_sync
from arctic_platform.model_plugin.attention import segment_ids

from .config import FULL_ATTENTION
from .config import GATED_RECURRENCE
from .config import MOE
from .config import ToyHybridMoEConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(x: torch.Tensor, position_ids: torch.Tensor, theta: float) -> torch.Tensor:
    """Rotary embedding of ``[1, T, H, D]`` at each token's position within its packed row."""
    dim = x.shape[-1]
    inv_freq = theta ** (-torch.arange(0, dim, 2, dtype=torch.float32, device=x.device) / dim)
    angles = position_ids[0].float()[:, None] * inv_freq[None, :]
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)[None, :, None, :]
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)[None, :, None, :]
    return (x.float() * cos + rotate_half(x.float()) * sin).to(x.dtype)


class ToyAttention(AttentionModule):
    """Grouped-query attention with rotary positions. The core is the platform's; this block owns q/k/v and o."""

    ap_layer_type = FULL_ATTENTION

    def __init__(self, config: ToyHybridMoEConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        self.rope_theta = config.rope_theta
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        q = self.q_proj(x).view(1, length, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(1, length, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(1, length, self.num_kv_heads, self.head_dim)
        q = apply_rope(q, position_ids, self.rope_theta)
        k = apply_rope(k, position_ids, self.rope_theta)
        return self.o_proj(self.mix(q, k, v, cu_seqlens=cu_seqlens).flatten(-2))


class ToyRecurrence(nn.Module):
    """A gated linear recurrence: a short causal convolution, then ``h_t = a_t * h_{t-1} + (1 - a_t) * c_t``.

    ``project`` and ``finish`` work token by token. ``mix`` is the sequence-mixing core: it sees whole packed rows,
    restarts its convolution and state at every row boundary in ``cu_seqlens``, and computes only the heads
    ``shard`` selects, slicing its per-head parameters to match.
    """

    ap_layer_type = GATED_RECURRENCE

    def __init__(self, config: ToyHybridMoEConfig):
        super().__init__()
        self.num_heads = config.recurrence_heads
        self.head_dim = config.recurrence_head_dim
        self.kernel = config.conv_kernel
        width = self.num_heads * self.head_dim
        self.in_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.z_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.out_proj = nn.Linear(width, config.hidden_size, bias=False)
        self.conv_weight = nn.Parameter(torch.randn(self.num_heads, self.head_dim, self.kernel) / self.kernel)
        self.decay_bias = nn.Parameter(torch.linspace(-1.0, 1.0, self.num_heads))
        self.norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        u = self.in_proj(x).view(1, length, self.num_heads, self.head_dim)
        g = self.gate_proj(x).view(1, length, self.num_heads, self.head_dim)
        z = self.z_proj(x).view(1, length, self.num_heads, self.head_dim)
        h = self.mix(u, g, cu_seqlens=cu_seqlens)
        return self.out_proj((self.norm(h) * F.silu(z)).flatten(-2))

    def mix(
        self, u: torch.Tensor, g: torch.Tensor, *, cu_seqlens: torch.Tensor, shard: HeadShard = FULL_SHARD
    ) -> torch.Tensor:
        heads = shard.local(self.num_heads)
        conv_weight = self.conv_weight[heads].float()
        decay_bias = self.decay_bias[heads].float()
        length = u.shape[1]
        rows = segment_ids(cu_seqlens, length)
        offset = torch.arange(length, device=u.device) - cu_seqlens.to(u.device)[rows]

        u32 = u.float()
        conv = torch.zeros_like(u32)
        for lag in range(self.kernel):
            shifted = F.pad(u32, (0, 0, 0, 0, lag, 0))[:, :length]
            same_row = (offset >= lag).to(u32.dtype)[None, :, None, None]
            conv = conv + shifted * conv_weight[None, None, :, :, self.kernel - 1 - lag] * same_row
        conv = F.silu(conv)

        decay = torch.sigmoid(g.float() + decay_bias[None, None, :, None])
        state = torch.zeros_like(conv[:, 0])
        outputs = []
        for t in range(length):
            state = state * (offset[t] > 0).to(state.dtype)
            state = decay[:, t] * state + (1 - decay[:, t]) * conv[:, t]
            outputs.append(state)
        return torch.stack(outputs, dim=1).to(u.dtype)


class DenseMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ToyExperts(nn.Module):
    """Routed experts stacked on dim 0, in the platform MoE layer's layout: ``w1``/``w3`` gate/up, ``w2`` down."""

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(num_experts, intermediate_size, hidden_size) * 0.1)
        self.w3 = nn.Parameter(torch.randn(num_experts, intermediate_size, hidden_size) * 0.1)
        self.w2 = nn.Parameter(torch.randn(num_experts, hidden_size, intermediate_size) * 0.1)


class LoadStats(nn.Module):
    """Tokens routed to each expert so far: trainer-only bookkeeping, checkpointed but never sent to the sampler."""

    def __init__(self, num_experts: int):
        super().__init__()
        self.register_buffer("tokens_per_expert", torch.zeros(num_experts))


class ToyMoE(nn.Module):
    """Softmax top-k routing with renormalized weights, routed experts plus one shared expert.

    ``routed`` replays expert ids captured by the sampler; an id of ``ROUTER_REPLAY_FRESH`` falls back to this
    router's own choice. Under expert parallelism ``ep_group`` is set and this rank holds experts
    ``[expert_offset, expert_offset + local experts)``.
    """

    def __init__(self, config: ToyHybridMoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.scale = config.routed_scaling_factor
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = ToyExperts(self.num_experts, config.hidden_size, config.moe_intermediate_size)
        self.shared_expert = DenseMLP(config.hidden_size, config.moe_intermediate_size * config.num_shared_experts)
        self.stats = skip_weight_sync(LoadStats(self.num_experts))
        self.ep_group = None
        self.expert_offset = 0

    def forward(self, x: torch.Tensor, routed: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = x.reshape(-1, x.shape[-1])
        scores = torch.softmax(self.router(tokens).float(), dim=-1)
        own = torch.topk(scores.detach(), self.top_k, dim=-1).indices
        selected = own if routed is None else torch.where(routed == ROUTER_REPLAY_FRESH, own, routed.to(own.dtype))
        weights = scores.gather(1, selected)
        weights = weights / weights.sum(-1, keepdim=True) * self.scale
        if self.training:
            with torch.no_grad():
                self.stats.tokens_per_expert += torch.bincount(selected.flatten(), minlength=self.num_experts)
        out = self._routed(tokens, selected, weights) + self.shared_expert(tokens)
        return out.view_as(x), selected

    def _routed(self, tokens: torch.Tensor, selected: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        group = self.ep_group
        all_tokens = gather_tokens(group, tokens)
        all_selected = gather_tokens(group, selected)
        all_weights = gather_tokens(group, weights)
        partial = torch.zeros_like(all_tokens)
        for local in range(self.experts.w1.shape[0]):
            rows, slots = (all_selected == self.expert_offset + local).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            x = all_tokens[rows]
            hidden = F.silu(x @ self.experts.w1[local].T) * (x @ self.experts.w3[local].T)
            out = (hidden @ self.experts.w2[local].T) * all_weights[rows, slots, None].to(x.dtype)
            partial = partial.index_add(0, rows, out)
        return combine_tokens(group, partial)


class ToyDecoderLayer(nn.Module):
    def __init__(self, config: ToyHybridMoEConfig, layer_idx: int):
        super().__init__()
        self.is_attention = config.layer_types[layer_idx] == FULL_ATTENTION
        self.is_moe = config.mlp_types[layer_idx] == MOE
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if self.is_attention:
            self.self_attn = ToyAttention(config)
        else:
            self.recurrence = ToyRecurrence(config)
        self.mlp = ToyMoE(config) if self.is_moe else DenseMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self, h: torch.Tensor, position_ids: torch.Tensor, cu_seqlens: torch.Tensor, routed: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mixer = self.self_attn if self.is_attention else self.recurrence
        h = h + mixer(self.input_norm(h), position_ids, cu_seqlens)
        if not self.is_moe:
            return h + self.mlp(self.post_norm(h)), None
        out, selected = self.mlp(self.post_norm(h), routed)
        return h + out, selected


class ToyHybridMoEModel(nn.Module):
    def __init__(self, config: ToyHybridMoEConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(ToyDecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class ToyHybridMoEForCausalLM(nn.Module):
    def __init__(self, config: ToyHybridMoEConfig):
        super().__init__()
        self.config = config
        self.model = ToyHybridMoEModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        routed_experts: dict[int, torch.Tensor] | None = None,
        return_routing: bool = False,
    ):
        routed_experts = routed_experts or {}
        h = self.model.embed_tokens(input_ids)
        routing: dict[int, torch.Tensor] = {}
        for index, layer in enumerate(self.model.layers):
            h, selected = layer(h, position_ids, cu_seqlens, routed_experts.get(index))
            if selected is not None:
                routing[index] = selected
        logits = self.lm_head(self.model.norm(h))
        return (logits, routing) if return_routing else logits
