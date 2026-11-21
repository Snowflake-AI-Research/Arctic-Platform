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

"""gpt-oss trainer model: mixture of experts, and attention with learned sinks that the platform's core does not have.

Two things set it apart from the OLMoE example:

- **Its own attention core.** Each head has a learned *sink*, an extra logit in its softmax that absorbs probability
  and is then dropped. The platform's attention core has no sinks, so ``GptOssAttention`` is a plugin layer type
  with its own ``mix``. It runs on the platform's head-parallel policy: under sequence parallelism ``mix`` sees
  whole rows of the heads its ``HeadShard`` holds, and slices the sinks to match. Layers alternate between
  ``sliding_attention`` and ``full_attention`` (``config.layer_types``); both are this core.
- **Experts already stacked.** The checkpoint stores each layer's experts stacked on dim 0
  (``mlp.experts.gate_up_proj`` is ``[E, D, 2I]``), the layout ``StackedExpertLayer`` shards, so the trainer keeps the
  checkpoint's names and layout and needs no converter. The router picks the top-k logits, then softmaxes over them.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import GptOssConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from arctic_platform.model_plugin import FULL_SHARD
from arctic_platform.model_plugin import CausalLM
from arctic_platform.model_plugin import ExpertShard
from arctic_platform.model_plugin import HeadShard
from arctic_platform.model_plugin import InitializesBuffers
from arctic_platform.model_plugin import SequenceMixingModule
from arctic_platform.model_plugin import StackedMoEModule
from arctic_platform.model_plugin import choose_experts
from arctic_platform.model_plugin import mark
from arctic_platform.model_plugin import segment_ids

SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"
SUPPORTED_ROPE_TYPES = ("default", "yarn", "linear", "llama3")


class RMSNorm(nn.Module):
    """RMSNorm in fp32, cast back before scaling by the weight."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        return self.weight * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)


class RotaryEmbedding(nn.Module, InitializesBuffers):
    """Rotary embedding with the config's frequency scaling (YaRN for gpt-oss), at each token's position within its
    packed row. Frequencies come from transformers' own rotary initialization."""

    inv_freq: torch.Tensor

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.config = config
        inv_freq, self.attention_scaling = self._frequencies(None)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        mark(self, dtype="fp32")

    def _frequencies(self, device) -> tuple[torch.Tensor, float]:
        rope_type = self.config.rope_parameters.get("rope_type", "default")
        if rope_type != "default":
            return ROPE_INIT_FUNCTIONS[rope_type](self.config, device)
        dim, theta = self.config.head_dim, self.config.rope_parameters["rope_theta"]
        return 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim), 1.0

    def ap_init_buffers(self) -> None:
        self.inv_freq = self._frequencies(self.inv_freq.device)[0]

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[1, T, H, D]``; the first and second halves of ``D`` rotate together."""
        angles = position_ids[0].float()[:, None] * self.inv_freq[None, :]
        cos = (angles.cos() * self.attention_scaling).to(x.dtype)[None, :, None, :]
        sin = (angles.sin() * self.attention_scaling).to(x.dtype)[None, :, None, :]
        first, second = x.chunk(2, dim=-1)
        return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


def sink_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    sliding_window: int | None,
) -> torch.Tensor:
    """Causal attention within each packed row, with one sink logit per head. ``q`` is ``[1, T, H, D]``, ``k``/``v``
    ``[1, T, Hkv, D]``, ``sinks`` ``[H]``. A sliding window lets a token attend to itself and the ``sliding_window - 1``
    tokens before it."""
    length, heads, kv_heads = q.shape[1], q.shape[2], k.shape[2]
    k = k.repeat_interleave(heads // kv_heads, dim=2)
    v = v.repeat_interleave(heads // kv_heads, dim=2)
    rows = segment_ids(cu_seqlens, length)
    allowed = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
    if sliding_window is not None:
        allowed = allowed.triu(-(sliding_window - 1))
    allowed = allowed & (rows[:, None] == rows[None, :])
    logits = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    logits = logits.masked_fill(~allowed, float("-inf"))
    sink = sinks.to(logits.dtype)[None, :, None, None].expand(-1, -1, length, 1)
    combined = torch.cat([logits, sink], dim=-1)
    probs = torch.softmax(combined - combined.max(dim=-1, keepdim=True).values, dim=-1)[..., :-1]
    return torch.einsum("bhqk,bkhd->bqhd", probs.to(v.dtype), v)


class GptOssAttention(nn.Module, SequenceMixingModule):
    """Grouped-query attention with biases, rotary positions and per-head sinks; ``mix`` is the core."""

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.ap_layer_type = config.layer_types[layer_idx]
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window if self.ap_layer_type == SLIDING_ATTENTION else None
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.sinks = nn.Parameter(torch.randn(self.num_heads))  # loaded from the checkpoint; distinct per head here
        self.rotary = RotaryEmbedding(config)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        q = self.rotary(self.q_proj(x).view(1, length, self.num_heads, self.head_dim), position_ids)
        k = self.rotary(self.k_proj(x).view(1, length, self.num_kv_heads, self.head_dim), position_ids)
        v = self.v_proj(x).view(1, length, self.num_kv_heads, self.head_dim)
        return self.o_proj(self.mix(q, k, v, cu_seqlens=cu_seqlens).flatten(-2))

    def mix(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        shard: HeadShard = FULL_SHARD,
    ) -> torch.Tensor:
        """Sink attention over whole rows, for the heads ``shard`` holds: the sinks are sliced to match."""
        sinks = self.sinks[shard.local(self.num_heads)]
        return sink_attention(q, k, v, sinks, cu_seqlens, self.head_dim**-0.5, self.sliding_window)


class StackedExperts(nn.Module):
    """Routed experts stacked on dim 0, in the checkpoint's layout and names. ``gate_up_proj`` interleaves the gate
    (even columns) and up (odd columns) projections."""

    def __init__(self, config: GptOssConfig):
        super().__init__()
        experts, hidden, inner = config.num_local_experts, config.hidden_size, config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.randn(experts, hidden, 2 * inner) * hidden**-0.5)
        self.gate_up_proj_bias = nn.Parameter(torch.randn(experts, 2 * inner) * 0.1)
        self.down_proj = nn.Parameter(torch.randn(experts, inner, hidden) * inner**-0.5)
        self.down_proj_bias = nn.Parameter(torch.randn(experts, hidden) * 0.1)


class GptOssMoE(nn.Module, StackedMoEModule):
    """Top-k routing by logit, softmax over the chosen experts, and a clamped SwiGLU in each expert."""

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.alpha, self.limit = config.swiglu_alpha, config.swiglu_limit
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=True)
        self.experts = StackedExperts(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.reshape(-1, x.shape[-1])
        selected, weights = self.route(tokens)
        return self.experts_core(tokens, selected, weights).view_as(x)

    def route(self, tokens: torch.Tensor, routed: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Expert ids ``[N, top_k]`` by router logit, and their weights, a softmax over the chosen logits. ``routed``
        is the platform's to supply when it replays routing."""
        logits = self.router(tokens)
        selected = choose_experts(logits, self.top_k, routed)
        return selected, torch.softmax(logits.gather(1, selected), dim=-1)

    def experts_core(
        self, tokens: torch.Tensor, selected: torch.Tensor, weights: torch.Tensor, *, shard: ExpertShard | None = None
    ) -> torch.Tensor:
        """Sum over each token's chosen experts that ``shard`` holds; local expert ``i`` is ``shard.start + i``."""
        shard = shard or ExpertShard.full(self.num_experts)
        experts = self.experts
        out = torch.zeros_like(tokens)
        for local in range(shard.per_rank):
            rows, slots = (selected == shard.start + local).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            gate_up = tokens[rows] @ experts.gate_up_proj[local] + experts.gate_up_proj_bias[local]
            gate = gate_up[..., ::2].clamp(max=self.limit)
            up = gate_up[..., 1::2].clamp(min=-self.limit, max=self.limit)
            hidden = (up + 1) * gate * torch.sigmoid(gate * self.alpha)
            expert_out = hidden @ experts.down_proj[local] + experts.down_proj_bias[local]
            out = out.index_add(0, rows, expert_out * weights[rows, slots, None])
        return out


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GptOssAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = GptOssMoE(config)

    def forward(self, h: torch.Tensor, position_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        h = h + self.self_attn(self.input_layernorm(h), position_ids, cu_seqlens)
        return h + self.mlp(self.post_attention_layernorm(h))


class GptOssModel(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(GptOssDecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class GptOssForCausalLM(nn.Module, CausalLM):
    """Returns final hidden states; the platform computes logits through ``lm_head`` a chunk at a time."""

    def __init__(self, config: GptOssConfig, options: None = None):
        super().__init__()
        self.config = config
        self.model = GptOssModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, *, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h, position_ids, cu_seqlens)
        return self.model.norm(h)
