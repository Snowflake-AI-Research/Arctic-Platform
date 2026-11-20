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

"""LFM2 trainer model: a hybrid of gated short-convolution layers and attention layers, dense, tied embeddings.

``config.layer_types`` gives each decoder layer's kind, ``conv`` or ``full_attention``. The convolution layer is a
new layer type the platform has never seen: ``Lfm2ShortConv.mix`` is its sequence-mixing core, a depthwise causal
convolution that restarts at every packed row. Channels are independent, so the platform's default head-parallel
policy runs it under sequence parallelism with each channel as a "head": ``mix`` computes the channels its
``HeadShard`` holds. Modules are named as in the Hugging Face checkpoint, so it loads with no converter.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Lfm2Config

from arctic_platform.model_plugin import FULL_SHARD
from arctic_platform.model_plugin import AttentionModule
from arctic_platform.model_plugin import CausalLM
from arctic_platform.model_plugin import HeadShard
from arctic_platform.model_plugin import InitializesBuffers
from arctic_platform.model_plugin import SequenceMixingModule
from arctic_platform.model_plugin import mark
from arctic_platform.model_plugin import segment_ids

FULL_ATTENTION = "full_attention"
CONV = "conv"


class RMSNorm(nn.Module):
    """RMSNorm in fp32, cast back before scaling by the weight, as Hugging Face's ``Lfm2RMSNorm``."""

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


class Lfm2Attention(AttentionModule):
    """Grouped-query attention with per-head q/k norms and rotary positions. The core (``mix``) is the platform's."""

    ap_layer_type = FULL_ATTENTION

    def __init__(self, config: Lfm2Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_layernorm = RMSNorm(self.head_dim, config.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, config.norm_eps)
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_parameters["rope_theta"])

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        q = self.q_layernorm(self.q_proj(x).view(1, length, self.num_heads, self.head_dim))
        k = self.k_layernorm(self.k_proj(x).view(1, length, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).view(1, length, self.num_kv_heads, self.head_dim)
        q, k = self.rotary(q, position_ids), self.rotary(k, position_ids)
        return self.out_proj(self.mix(q, k, v, cu_seqlens=cu_seqlens).flatten(-2))


class Lfm2ShortConv(nn.Module, SequenceMixingModule):
    """``out_proj(C * conv(B * x))`` with ``B, C, x = in_proj(h)``: a gated depthwise causal convolution.

    Only ``mix`` sees other tokens. The projections and gates work token by token.
    """

    ap_layer_type = CONV

    def __init__(self, config: Lfm2Config):
        super().__init__()
        self.channels = config.hidden_size
        self.kernel = config.conv_L_cache
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.conv = nn.Conv1d(self.channels, self.channels, self.kernel, groups=self.channels, bias=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        b, c, u = self.in_proj(x).chunk(3, dim=-1)
        mixed = self.mix((b * u).unsqueeze(-1), cu_seqlens=cu_seqlens).squeeze(-1)  # channels as heads, [1, T, C, 1]
        return self.out_proj(c * mixed)

    def mix(self, u: torch.Tensor, *, cu_seqlens: torch.Tensor, shard: HeadShard = FULL_SHARD) -> torch.Tensor:
        """Depthwise causal convolution of ``u`` ``[1, T, C_local, 1]`` over whole rows, restarting at every row in
        ``cu_seqlens``, for the channels ``shard`` holds."""
        weight = self.conv.weight[shard.local(self.channels), 0]  # [C_local, K]
        length = u.shape[1]
        rows = segment_ids(cu_seqlens, length)
        out = torch.zeros_like(u)
        for lag in range(self.kernel):
            shifted = F.pad(u, (0, 0, 0, 0, lag, 0))[:, :length]
            same_row = torch.zeros(length, dtype=torch.bool, device=u.device)
            same_row[lag:] = rows[lag:] == rows[: length - lag]
            out = out + shifted * weight[None, None, :, self.kernel - 1 - lag, None] * same_row[None, :, None, None]
        return out


class Lfm2MLP(nn.Module):
    def __init__(self, config: Lfm2Config):
        super().__init__()
        size = config.intermediate_size
        if config.block_auto_adjust_ff_dim:  # as Hugging Face's Lfm2MLP sizes it
            size = int(2 * size / 3)
            if config.block_ffn_dim_multiplier is not None:
                size = int(config.block_ffn_dim_multiplier * size)
                size = config.block_multiple_of * ((size + config.block_multiple_of - 1) // config.block_multiple_of)
        self.w1 = nn.Linear(config.hidden_size, size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, size, bias=False)
        self.w2 = nn.Linear(size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, config: Lfm2Config, layer_idx: int):
        super().__init__()
        self.is_attention = config.layer_types[layer_idx] == FULL_ATTENTION
        if self.is_attention:
            self.self_attn = Lfm2Attention(config)
        else:
            self.conv = Lfm2ShortConv(config)
        self.operator_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.feed_forward = Lfm2MLP(config)

    def forward(self, h: torch.Tensor, position_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        block = self.self_attn if self.is_attention else self.conv
        h = h + block(self.operator_norm(h), position_ids, cu_seqlens)
        return h + self.feed_forward(self.ffn_norm(h))


class Lfm2Model(nn.Module):
    def __init__(self, config: Lfm2Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # As transformers initializes it. The checkpoint replaces it; with tied embeddings it is also the output head,
        # and PyTorch's default N(0, 1) would give the fixture logits of about 40, where fp32 rounds at about 4e-6.
        nn.init.normal_(self.embed_tokens.weight, std=config.initializer_range)
        self.layers = nn.ModuleList(Lfm2DecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.embedding_norm = RMSNorm(config.hidden_size, config.norm_eps)  # the final norm, despite its name


class Lfm2ForCausalLM(nn.Module, CausalLM):
    """Returns final hidden states. ``lm_head`` shares the embedding's weight when the config ties them, as LFM2's
    released checkpoints do: they store it once, as ``model.embed_tokens.weight``."""

    def __init__(self, config: Lfm2Config, options: None = None):
        super().__init__()
        self.config = config
        self.model = Lfm2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, *, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h, position_ids, cu_seqlens)
        return self.model.embedding_norm(h)
