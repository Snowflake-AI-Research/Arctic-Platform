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

"""OLMo-2 trainer model: a dense decoder with attention on the platform's core.

OLMo-2 differs from Llama in two places: RMSNorm on the full query and key projections before rotary embedding
(``q_norm``, ``k_norm``), and norms after each block instead of before it (``post_attention_layernorm``,
``post_feedforward_layernorm``). Modules are named as in the Hugging Face checkpoint, so it loads with no converter
and vLLM's built-in ``Olmo2ForCausalLM`` loads the weight sync as is.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Olmo2Config

from arctic_platform.model_plugin import AttentionModule
from arctic_platform.model_plugin import CausalLM
from arctic_platform.model_plugin import InitializesBuffers
from arctic_platform.model_plugin import mark

FULL_ATTENTION = "full_attention"


def head_dim(config: Olmo2Config) -> int:
    return getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads


class RMSNorm(nn.Module):
    """RMSNorm in fp32, scaled by the weight before casting back, as Hugging Face's ``Olmo2RMSNorm``."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * normed).to(x.dtype)


class RotaryEmbedding(nn.Module, InitializesBuffers):
    """Rotary embedding at each token's position within its packed row; ``inv_freq`` is computed, not loaded."""

    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float):
        super().__init__()
        self.dim, self.theta = dim, theta
        self.register_buffer("inv_freq", self._inv_freq(None), persistent=False)
        mark(self, dtype="fp32")  # a bf16 build must not round the frequencies

    def _inv_freq(self, device) -> torch.Tensor:
        return 1.0 / self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)

    def ap_init_buffers(self) -> None:
        self.inv_freq = self._inv_freq(self.inv_freq.device)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[1, T, H, D]``."""
        angles = position_ids[0].float()[:, None] * self.inv_freq[None, :]
        cos = torch.cat((angles.cos(), angles.cos()), dim=-1)[None, :, None, :]
        sin = torch.cat((angles.sin(), angles.sin()), dim=-1)[None, :, None, :]
        first, second = x.float().chunk(2, dim=-1)
        return (x.float() * cos + torch.cat((-second, first), dim=-1) * sin).to(x.dtype)


class Olmo2Attention(AttentionModule):
    """Grouped-query attention with q/k norms and rotary positions. The core (``mix``) is the platform's."""

    ap_layer_type = FULL_ATTENTION

    def __init__(self, config: Olmo2Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim(config)
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.num_heads * self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.num_kv_heads * self.head_dim, config.rms_norm_eps)
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        q = self.q_norm(self.q_proj(x)).view(1, length, self.num_heads, self.head_dim)
        k = self.k_norm(self.k_proj(x)).view(1, length, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(1, length, self.num_kv_heads, self.head_dim)
        q, k = self.rotary(q, position_ids), self.rotary(k, position_ids)
        return self.o_proj(self.mix(q, k, v, cu_seqlens=cu_seqlens).flatten(-2))


class Olmo2MLP(nn.Module):
    def __init__(self, config: Olmo2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Olmo2DecoderLayer(nn.Module):
    def __init__(self, config: Olmo2Config):
        super().__init__()
        self.self_attn = Olmo2Attention(config)
        self.mlp = Olmo2MLP(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, h: torch.Tensor, position_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        h = h + self.post_attention_layernorm(self.self_attn(h, position_ids, cu_seqlens))
        return h + self.post_feedforward_layernorm(self.mlp(h))


class Olmo2Model(nn.Module):
    def __init__(self, config: Olmo2Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(Olmo2DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class Olmo2ForCausalLM(nn.Module, CausalLM):
    """Returns final hidden states; the platform computes logits through ``lm_head`` a chunk at a time."""

    def __init__(self, config: Olmo2Config, options: None = None):
        super().__init__()
        self.config = config
        self.model = Olmo2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, *, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h, position_ids, cu_seqlens)
        return self.model.norm(h)
