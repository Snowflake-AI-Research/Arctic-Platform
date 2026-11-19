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

"""ToyHybridMoE configuration. The layer count and the kind of every layer come from here, never from code."""

from __future__ import annotations

from transformers import PretrainedConfig

FULL_ATTENTION = "full_attention"
# transformers validates layer_types against its known kinds, and vLLM counts a hybrid model's stateful layers by
# the name "linear_attention" when it sizes their caches, so the recurrence uses that name.
GATED_RECURRENCE = "linear_attention"
SHORT_CONV = "conv"
VALUE_CONV = "value_conv"
DENSE = "dense"
MOE = "moe"


class ToyHybridMoEConfig(PretrainedConfig):
    model_type = "toy_hybrid_moe"

    def __init__(
        self,
        vocab_size: int = 64,
        hidden_size: int = 32,
        layer_types: list[str] | None = None,
        mlp_types: list[str] | None = None,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 2,
        head_dim: int = 8,
        sliding_window: int | None = None,
        rope_theta: float = 10000.0,
        recurrence_heads: int = 4,
        recurrence_head_dim: int = 8,
        conv_kernel: int = 3,
        conv_channels: int = 32,
        intermediate_size: int = 64,
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
        moe_intermediate_size: int = 16,
        num_shared_experts: int = 1,
        routed_scaling_factor: float = 1.0,
        rms_norm_eps: float = 1e-6,
        attention_value_conv: bool = False,
        final_logit_softcapping: float | None = None,
        **kwargs,
    ):
        layer_types = list(layer_types or [GATED_RECURRENCE, FULL_ATTENTION, SHORT_CONV, FULL_ATTENTION])
        mlp_types = list(mlp_types or [DENSE, MOE, MOE, DENSE])
        if len(layer_types) != len(mlp_types):
            raise ValueError(f"{len(layer_types)} layer_types but {len(mlp_types)} mlp_types")
        kinds = {FULL_ATTENTION, GATED_RECURRENCE, SHORT_CONV}
        unknown = (set(layer_types) - kinds) | (set(mlp_types) - {DENSE, MOE})
        if len(unknown) > 0:
            raise ValueError(f"unknown layer or mlp types: {sorted(unknown)}")
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if not 0 < num_experts_per_tok <= num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layer_types = layer_types
        self.mlp_types = mlp_types
        self.num_hidden_layers = len(layer_types)
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.rope_theta = rope_theta
        self.recurrence_heads = recurrence_heads
        self.recurrence_head_dim = recurrence_head_dim
        self.conv_kernel = conv_kernel
        self.conv_channels = conv_channels
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.num_shared_experts = num_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.rms_norm_eps = rms_norm_eps
        # A short causal convolution on every attention layer's values: a sequence-mixing sub-layer.
        self.attention_value_conv = attention_value_conv
        # Logits become cap * tanh(logits / cap); None leaves them as they are.
        self.final_logit_softcapping = final_logit_softcapping
        kwargs.setdefault("architectures", ["ToyHybridMoEForCausalLM"])
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
