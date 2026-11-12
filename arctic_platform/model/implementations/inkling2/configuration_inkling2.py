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

"""Inkling2 text configuration, read from the checkpoint's ``text_config``."""

from __future__ import annotations

from transformers import PretrainedConfig


class Inkling2TextConfig(PretrainedConfig):
    model_type = "inkling2"

    def __init__(
        self,
        vocab_size=201024,
        unpadded_vocab_size=None,
        hidden_size=4608,
        num_hidden_layers=48,
        num_attention_heads=48,
        num_key_value_heads=8,
        head_dim=128,
        swa_num_attention_heads=None,
        swa_num_key_value_heads=None,
        swa_head_dim=None,
        sliding_window_size=1024,
        local_layer_ids=(),
        d_rel=16,
        rel_extent=1024,
        log_scaling_n_floor=None,
        log_scaling_alpha=0.1,
        rms_norm_eps=1e-6,
        qk_l2_clip_threshold=22.627416997969522,
        rel_l2_clip_threshold=8.0,
        sconv_kernel_size=4,
        n_layers_per_attn_res_block=6,
        attn_res_normalize_values=True,
        n_routed_experts=384,
        num_experts_per_tok=8,
        n_shared_experts=0,
        intermediate_size=2304,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        route_scale=8.0,
        use_gate_bias=True,
        use_value_norm_no_gamma=True,
        q_bias=False,
        o_bias=False,
        pre_norm_pre_affine=False,
        final_norm_pre_affine=False,
        final_logit_softcapping=30.0,
        tie_word_embeddings=False,
        **kwargs,
    ):
        unsupported = {
            "n_shared_experts": n_shared_experts != 0,
            "use_gate_bias": not use_gate_bias,
            "use_value_norm_no_gamma": not use_value_norm_no_gamma,
            "q_bias": q_bias,
            "o_bias": o_bias,
            "pre_norm_pre_affine": pre_norm_pre_affine,
            "final_norm_pre_affine": final_norm_pre_affine,
            "tie_word_embeddings": tie_word_embeddings,
        }
        if any(unsupported.values()):
            names = sorted(name for name, value in unsupported.items() if value)
            raise ValueError(f"Inkling2 training does not support these text_config values: {names}")
        self.vocab_size = vocab_size
        self.unpadded_vocab_size = unpadded_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.swa_num_attention_heads = swa_num_attention_heads or num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads or num_key_value_heads
        self.swa_head_dim = swa_head_dim or head_dim
        self.sliding_window_size = sliding_window_size
        self.local_layer_ids = list(local_layer_ids)
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = log_scaling_alpha
        self.rms_norm_eps = rms_norm_eps
        self.qk_l2_clip_threshold = qk_l2_clip_threshold
        self.rel_l2_clip_threshold = rel_l2_clip_threshold
        self.sconv_kernel_size = sconv_kernel_size
        self.n_layers_per_attn_res_block = n_layers_per_attn_res_block
        self.attn_res_normalize_values = attn_res_normalize_values
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.intermediate_size = intermediate_size
        self.mlp_gate_softcap = mlp_gate_softcap
        self.mlp_up_softcap = mlp_up_softcap
        self.route_scale = route_scale
        self.use_gate_bias = use_gate_bias
        self.use_value_norm_no_gamma = use_value_norm_no_gamma
        self.final_logit_softcapping = final_logit_softcapping
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
