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
"""A training forward for the stock transformers dense Qwen3 that computes vLLM's numbers.

``apply_vllm_aligned`` replaces the forwards of a loaded ``Qwen3ForCausalLM`` with the operator sequence of vLLM's
``Qwen3ForCausalLM`` under ``VLLM_BATCH_INVARIANT=1`` and ``enforce_eager=True``, each operator calling vLLM's
kernel (see ``ops``). Parameters, their names and the state dict are untouched, so loading, checkpointing and the
weight export the sampler receives are those of the stock model.

What the sampler does, and what is reproduced here:

* q/k/v and gate/up run as one matmul each over the concatenated weight, which is how vLLM stores them;
* the first decoder layer's input norm is a norm of one tensor; every later norm, and the final one, is the norm of
  ``fused_add_rms_norm``, whose residual sum is an add rounded to the model dtype, which a plain add reproduces;
* q and k are normalized per head, then rotated with a cos/sin cache built as vLLM builds it;
* attention reads a paged KV cache through the sampler's FlashAttention entry point with ``num_splits=1``;
* the log-probability of each label comes from the LM-head matmul over vLLM's padded vocabulary and Model Runner
  V2's token-logprob kernel.

This holds for one sampler rank: a sampler with tensor parallelism splits and all-reduces the matmuls differently.
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn

from arctic_platform.model.implementations.vllm_aligned import ops

SUPPORTED_MODEL_TYPES = frozenset({"qwen3"})


def _text_config(model: nn.Module):
    config = model.config
    return config.get_text_config() if hasattr(config, "get_text_config") else config


def _rope_theta(config) -> float | int:
    parameters = getattr(config, "rope_parameters", None) or {}
    rope_type = parameters.get("rope_type", "default")
    if rope_type != "default":
        raise ValueError(
            f"the vLLM-aligned Qwen3 forward supports the default rotary embedding only, got {rope_type!r}"
        )
    if parameters.get("partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("the vLLM-aligned Qwen3 forward does not support a partial rotary embedding")
    theta = parameters.get("rope_theta", getattr(config, "rope_theta", None))
    if theta is None:
        raise ValueError("the Qwen3 config has no rope_theta")
    return theta


def check_supported_config(config) -> None:
    """Fail unless ``config`` describes a model this forward reproduces."""
    model_type = getattr(config, "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"the vLLM-aligned forward supports model types {sorted(SUPPORTED_MODEL_TYPES)}, got {model_type!r}"
        )
    if getattr(config, "use_sliding_window", False):
        raise ValueError("the vLLM-aligned Qwen3 forward does not support sliding-window attention")
    if getattr(config, "hidden_act", "silu") != "silu":
        raise ValueError(f"the vLLM-aligned Qwen3 forward supports hidden_act='silu', got {config.hidden_act!r}")
    _rope_theta(config)


class _Geometry:
    def __init__(self, config, *, kv_block_size: int):
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", None) or config.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.eps = float(config.rms_norm_eps)
        self.rope_theta = _rope_theta(config)
        self.max_position = int(config.max_position_embeddings)
        self.kv_block_size = int(kv_block_size)
        self._cos_sin_cache: torch.Tensor | None = None
        self._fa_version: int | None = None

    def cos_sin_cache(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        cache = self._cos_sin_cache
        if cache is None or cache.dtype != dtype or cache.device != device:
            cache = ops.rotary_cos_sin_cache(
                rotary_dim=self.head_dim,
                base=self.rope_theta,
                max_position=self.max_position,
                dtype=dtype,
                device=device,
            )
            self._cos_sin_cache = cache
        return cache

    def fa_version(self) -> int:
        if self._fa_version is None:
            self._fa_version = ops.flash_attn_version(self.head_dim)
        return self._fa_version


def _concatenated(*tensors: torch.Tensor | None) -> torch.Tensor | None:
    if all(tensor is None for tensor in tensors):
        return None
    if any(tensor is None for tensor in tensors):
        raise ValueError("either every fused projection has a bias or none does")
    return torch.cat(tensors, dim=0)


def _decoder_layer_forward(
    self, hidden_states: torch.Tensor, aligned_layout: ops.PackedLayout | None = None, **kwargs
):
    """One decoder layer on ``[tokens, hidden]``; returns the residual stream after the layer."""
    if aligned_layout is None:
        raise RuntimeError("the vLLM-aligned decoder layer needs the packed layout from the aligned Qwen3 model")
    geometry: _Geometry = self._vllm_aligned_geometry
    attn = self.self_attn
    mlp = self.mlp
    num_tokens = hidden_states.shape[0]

    x = ops.rms_norm(hidden_states, self.input_layernorm.weight, geometry.eps, fused=self._vllm_aligned_index > 0)
    qkv = ops.linear(
        x,
        torch.cat((attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight), dim=0),
        _concatenated(attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias),
    )
    q, k, v = qkv.split((geometry.q_size, geometry.kv_size, geometry.kv_size), dim=-1)
    q = ops.rms_norm(
        q.view(num_tokens, geometry.num_heads, geometry.head_dim), attn.q_norm.weight, geometry.eps, fused=False
    ).view(num_tokens, geometry.q_size)
    k = ops.rms_norm(
        k.view(num_tokens, geometry.num_kv_heads, geometry.head_dim), attn.k_norm.weight, geometry.eps, fused=False
    ).view(num_tokens, geometry.kv_size)
    q, k = ops.rotary(
        aligned_layout.positions,
        q,
        k,
        geometry.cos_sin_cache(q.dtype, q.device),
        geometry.head_dim,
        is_neox=True,
    )
    attn_out = ops.paged_attention(
        q.view(num_tokens, geometry.num_heads, geometry.head_dim),
        k.view(num_tokens, geometry.num_kv_heads, geometry.head_dim),
        v.view(num_tokens, geometry.num_kv_heads, geometry.head_dim),
        aligned_layout,
        scale=geometry.scale,
        fa_version=geometry.fa_version(),
    )
    attn_out = ops.linear(attn_out.reshape(num_tokens, geometry.q_size), attn.o_proj.weight, attn.o_proj.bias)
    residual = hidden_states + attn_out

    x = ops.rms_norm(residual, self.post_attention_layernorm.weight, geometry.eps, fused=True)
    gate_up = ops.linear(x, torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight), dim=0))
    mlp_out = ops.linear(ops.silu_and_mul(gate_up), mlp.down_proj.weight)
    return residual + mlp_out


def _model_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    cu_seq_lens_q: torch.Tensor | None = None,
    **kwargs,
):
    from transformers.modeling_outputs import BaseModelOutputWithPast

    if use_cache or past_key_values is not None:
        raise ValueError("the vLLM-aligned Qwen3 forward is a training forward and does not use a KV cache")
    if attention_mask is not None and not bool(attention_mask.bool().all().item()):
        raise ValueError(
            "the vLLM-aligned Qwen3 forward takes packed sequences delimited by position_ids, not a padding mask"
        )
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("pass exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    batch_size, seq_len, hidden_size = inputs_embeds.shape
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).expand(batch_size, seq_len)

    geometry: _Geometry = self._vllm_aligned_geometry
    layout = ops.PackedLayout.build(position_ids, block_size=geometry.kv_block_size, cu_seqlens=cu_seq_lens_q)
    if int(layout.positions.max().item()) >= geometry.max_position:
        raise ValueError(f"a position reaches max_position_embeddings={geometry.max_position}")

    hidden_states = inputs_embeds.reshape(batch_size * seq_len, hidden_size)
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(hidden_states, aligned_layout=layout)
    hidden_states = ops.rms_norm(hidden_states, self.norm.weight, geometry.eps, fused=True)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states.view(batch_size, seq_len, hidden_size))


def _scalar_temperature(temperature) -> float:
    if temperature is None:
        return 1.0
    if isinstance(temperature, bool):
        raise ValueError("temperature must be a real scalar, not bool")
    if torch.is_tensor(temperature):
        values = temperature.detach().to(torch.float64).reshape(-1)
        if values.numel() == 0:
            return 1.0
        first = float(values[0].item())
        if not bool((values == first).all().item()):
            raise ValueError("the vLLM-aligned logprobs take one temperature per call, not one per token")
        temperature = first
    temperature = float(temperature)
    return 1.0 if temperature == 0.0 else temperature


def _causal_lm_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    logits_to_keep: int | torch.Tensor = 0,
    temperature=None,
    action_masks=None,
    dss_compute_logprobs: bool = False,
    group_token_ids=None,
    **kwargs,
):
    from transformers.modeling_outputs import CausalLMOutputWithPast

    from arctic_platform.model.implementations.gpu.lm_head import inherit_lm_head_target_validation
    from arctic_platform.model.implementations.gpu.lm_head import validate_lm_head_targets

    if action_masks is not None or group_token_ids is not None:
        raise ValueError("the vLLM-aligned Qwen3 forward does not support action_masks or group_token_ids")
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )
    hidden_states = outputs.last_hidden_state
    batch_size, seq_len = hidden_states.shape[:2]
    if isinstance(logits_to_keep, int):
        slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
    else:
        slice_indices = logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    if dss_compute_logprobs:
        if labels is None:
            raise ValueError("dss_compute_logprobs requires labels")
        if labels.dim() != 2:
            labels = inherit_lm_head_target_validation(labels, labels.reshape(batch_size, seq_len))
        labels = inherit_lm_head_target_validation(labels, labels[:, slice_indices])
        kept = hidden_states.shape[1]
        labels_flat = inherit_lm_head_target_validation(labels, labels.reshape(-1))
        labels_flat = inherit_lm_head_target_validation(labels_flat, labels_flat.to(hidden_states.device))
        validate_lm_head_targets(labels_flat, vocab_size=int(self.lm_head.weight.shape[0]))
        logprobs = ops.lm_head_logprobs(
            hidden_states.reshape(batch_size * kept, -1),
            self.lm_head.weight,
            labels_flat,
            temperature=_scalar_temperature(temperature),
            token_chunk_size=self._vllm_aligned_token_chunk_size,
        )
        return {"logprobs": logprobs.view(batch_size, kept)}

    logits = ops.linear(hidden_states, self.lm_head.weight)
    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
    return CausalLMOutputWithPast(loss=loss, logits=logits)


def apply_vllm_aligned(model: nn.Module, *, kv_block_size: int = 16, token_chunk_size: int = 1024) -> None:
    """Make a loaded transformers ``Qwen3ForCausalLM`` compute vLLM's numbers; see the module docstring.

    ``kv_block_size`` is the sampler's KV-cache page size. ``token_chunk_size`` bounds the rows of the LM-head
    logits held at once when the forward returns log-probabilities.
    """
    config = _text_config(model)
    check_supported_config(config)
    backbone = getattr(model, "model", None)
    if backbone is None or getattr(model, "lm_head", None) is None:
        raise ValueError("the vLLM-aligned forward needs a causal LM with .model and .lm_head")
    if kv_block_size < 1 or token_chunk_size < 1:
        raise ValueError("kv_block_size and token_chunk_size must be positive")
    ops.require_batch_invariant_env()

    geometry = _Geometry(config, kv_block_size=kv_block_size)
    backbone._vllm_aligned_geometry = geometry
    backbone.forward = types.MethodType(_model_forward, backbone)
    for index, layer in enumerate(backbone.layers):
        layer._vllm_aligned_geometry = geometry
        layer._vllm_aligned_index = index
        layer.forward = types.MethodType(_decoder_layer_forward, layer)
    model._vllm_aligned_token_chunk_size = int(token_chunk_size)
    model.forward = types.MethodType(_causal_lm_forward, model)
    model._vllm_aligned = True
