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
"""Fuse Qwen3's per-head ``q_norm`` / ``k_norm`` with the rotary embedding (``kernels/fused_qk_norm_rope.py``).

transformers' ``Qwen3Attention`` (and ``Qwen3MoeAttention``) normalizes each query and key head with an RMSNorm module
and then rotates it, five FP32 passes and a rotation per tensor with an FP32 copy kept for backward. The patch
replaces each such module's ``forward`` with the same computation that calls one fused kernel per tensor instead;
the projections (and any LoRA on them), the attention interface and the output projection are unchanged.
"""

from __future__ import annotations

import types

import torch.nn as nn

from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.patch import register_patch


def _qwen3_attention_classes() -> tuple[type, ...]:
    classes = []
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    classes.append(Qwen3Attention)
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention
    except ImportError:
        pass
    else:
        classes.append(Qwen3MoeAttention)
    return tuple(classes)


def _norm_eps(norm: nn.Module) -> float:
    if getattr(norm, "offset", 0.0):
        raise ValueError(
            f"fused_qk_norm_rope supports a plain RMSNorm weight, got {type(norm).__name__} with an offset"
        )
    eps = getattr(norm, "variance_epsilon", None)
    if eps is None:
        eps = getattr(norm, "eps", None)
    if eps is None or getattr(norm, "weight", None) is None:
        raise ValueError(f"fused_qk_norm_rope cannot read the weight and epsilon of {type(norm).__name__}")
    return float(eps)


def fused_qk_norm_rope_attention_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values=None,
    **kwargs,
):
    """transformers' ``Qwen3Attention.forward`` with ``q_norm`` / ``k_norm`` and the rotary embedding fused."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

    from arctic_platform.model.implementations.kernels.fused_qk_norm_rope import fused_norm_rope

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    cos, sin = position_embeddings
    query_states = fused_norm_rope(
        self.q_proj(hidden_states).view(hidden_shape), self.q_norm.weight, cos, sin, _norm_eps(self.q_norm)
    ).transpose(1, 2)
    key_states = fused_norm_rope(
        self.k_proj(hidden_states).view(hidden_shape), self.k_norm.weight, cos, sin, _norm_eps(self.k_norm)
    ).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


@register_patch("fused_qk_norm_rope")
def apply_fused_qk_norm_rope(model: nn.Module, ctx: LoaderContext) -> None:
    classes = _qwen3_attention_classes()
    patched = 0
    for module in model.modules():
        if type(module) in classes:
            _norm_eps(module.q_norm)
            _norm_eps(module.k_norm)
            module.forward = types.MethodType(fused_qk_norm_rope_attention_forward, module)
            patched += 1
    if patched == 0:
        raise ValueError(
            "fused_qk_norm_rope found no transformers Qwen3Attention / Qwen3MoeAttention module to patch; it applies "
            "only to the Qwen3 and Qwen3-MoE families on the huggingface loader"
        )
