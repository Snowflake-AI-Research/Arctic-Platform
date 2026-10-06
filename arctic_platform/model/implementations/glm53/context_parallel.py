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

"""Sparse-MLA and head-parallel KDA training path for GLM-5.3-Flash."""

from __future__ import annotations

import math
import types

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.nn.functional as F
from torch import nn

from arctic_platform.model.implementations.glm52.models.kernels.sparse_mla_flashmla import sparse_mla_flashmla_apply
from arctic_platform.model.implementations.gpu.packing import cu_seqlens_from_position_ids
from arctic_platform.model.implementations.gpu.sp.collectives import sequence_head_all_to_all
from arctic_platform.model.implementations.gpu.sp.gated_delta_net import _fallback_depthwise_causal_convolution
from arctic_platform.model.implementations.gpu.sp.gated_delta_net import _pack_and_exchange_head_tensors
from arctic_platform.model.implementations.gpu.sp.gated_delta_net import _shard_convolution_parameter
from arctic_platform.model.implementations.moe.logging_utils import get_logger
from arctic_platform.model.implementations.moe.vlm import get_language_model

_GLOBAL_POSITION_IDS = "_ap_glm53_global_position_ids"
_GLOBAL_CU_SEQLENS = "_ap_glm53_global_cu_seqlens"


def _gather_sequence(tensor: torch.Tensor, group) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return tensor
    return torch.cat(dist_nn.all_gather(tensor.contiguous(), group=group), dim=1)


@torch.no_grad()
def _gather_sequence_no_grad(tensor: torch.Tensor, group) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return torch.cat(gathered, dim=1)


@torch.no_grad()
def _select_sparse_indices(
    indexer: nn.Module,
    hidden_states: torch.Tensor,
    q_resid: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    cp_group,
    cp_rank: int,
    cp_world_size: int,
) -> torch.Tensor:
    batch_size, local_length = hidden_states.shape[:2]
    hidden_shape = (batch_size, local_length, -1, indexer.head_dim)
    query = indexer.wq_b(q_resid).view(hidden_shape)
    key = indexer.k_norm(indexer.wk(hidden_states)).view(hidden_shape).squeeze(2)
    gate_scores = F.linear(hidden_states, indexer.index_kpool_compress_gate)
    packed_local = torch.cat(
        [key, gate_scores, attention_mask.to(key.dtype)[..., None]],
        dim=-1,
    )
    packed_states = _gather_sequence_no_grad(packed_local, cp_group)
    valid_keys = packed_states[..., -1].bool()
    global_length = packed_states.shape[1]

    query_positions = cp_rank * local_length + torch.arange(local_length, device=hidden_states.device)
    key_positions = torch.arange(global_length, device=hidden_states.device)
    visible_tokens = (key_positions[None, None, :] <= query_positions[None, :, None]) & valid_keys[:, None, :]

    pool_keys, pool_indices, pool_valid = indexer.get_pooled_states(packed_states=packed_states)
    scores = torch.matmul(query.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1))
    scores = F.relu(scores * indexer.softmax_scale)
    weights = indexer.weights_proj(hidden_states.to(indexer.weights_proj.weight.dtype)).float() * (
        indexer.n_heads**-0.5
    )
    index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

    pool_end = pool_indices[..., -1].clamp(0, global_length - 1)
    pool_visible = visible_tokens.gather(
        dim=-1,
        index=pool_end[:, None, :].expand(batch_size, local_length, -1),
    )
    valid_candidates = pool_visible & pool_valid[:, None]
    index_scores.masked_fill_(~valid_candidates, torch.finfo(index_scores.dtype).min)

    select_k = min(
        indexer.index_topk // indexer.index_kpool,
        index_scores.shape[-1],
    )
    selected = index_scores.topk(select_k, dim=-1).indices
    batch_idx = torch.arange(batch_size, device=hidden_states.device)[:, None, None]
    selected_valid = valid_candidates.gather(-1, selected)
    selected_indices = pool_indices[batch_idx, selected]
    topk_indices = selected_indices.flatten(-2)
    topk_indices.masked_fill_(
        ~selected_valid[..., None].expand_as(selected_indices).flatten(-2),
        -1,
    )

    output_width = indexer.index_topk
    if indexer.index_kpool_always_select_tail:
        topk_indices = indexer.append_visible_tail(topk_indices, visible_tokens, valid_keys)
        output_width += indexer.index_kpool - 1
    topk_indices = F.pad(
        topk_indices,
        (0, output_width - topk_indices.shape[-1]),
        value=-1,
    )[..., :output_width]
    topk_indices.masked_fill_(~attention_mask[..., None], -1)
    return topk_indices.to(torch.int32)


def _reference_sparse_mla(
    query: torch.Tensor,
    key_value: torch.Tensor,
    indices: torch.Tensor,
    *,
    scale: float,
    value_dim: int,
) -> torch.Tensor:
    kv_length = key_value.shape[1] - 1
    valid = (indices >= 0) & (indices < kv_length)
    safe = indices.long().clamp(0, kv_length)
    batch = torch.arange(query.shape[0], device=query.device)[:, None, None]
    gathered = key_value[batch, safe, 0]
    scores = torch.einsum("bqhd,bqkd->bqhk", query.float(), gathered.float())
    scores.mul_(scale).masked_fill_(~valid[:, :, None], float("-inf"))
    empty = ~valid.any(dim=-1)
    scores.masked_fill_(empty[:, :, None, None], 0)
    probabilities = torch.softmax(scores, dim=-1).to(query.dtype)
    probabilities = probabilities.masked_fill(empty[:, :, None, None], 0)
    return torch.einsum(
        "bqhk,bqkd->bqhd",
        probabilities,
        gathered[..., :value_dim],
    )


def _pad_sparse_indices(indices: torch.Tensor, multiple: int = 128) -> torch.Tensor:
    width = indices.shape[-1]
    padded_width = math.ceil(width / multiple) * multiple
    if padded_width == width:
        return indices
    return F.pad(indices, (0, padded_width - width), value=-1)


def _sparse_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    prev_topk_indices: torch.Tensor | None = None,
    **_kwargs,
):
    if past_key_values is not None:
        raise NotImplementedError("GLM-5.3 sparse-MLA training does not support KV-cache decoding")
    if hidden_states.shape[0] != 1:
        raise NotImplementedError("GLM-5.3 context parallelism currently requires one packed row")
    if attention_mask is None:
        attention_mask = torch.ones(
            hidden_states.shape[:2],
            dtype=torch.bool,
            device=hidden_states.device,
        )

    batch_size, local_length = hidden_states.shape[:2]
    query_shape = (batch_size, local_length, -1, self.qk_head_dim)
    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    query_states = self.q_b_proj(q_resid).view(query_shape)
    q_nope, q_position = query_states.split(
        [self.qk_nope_head_dim, self.qk_rope_head_dim],
        dim=-1,
    )

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    kv_pass, k_position = compressed_kv.split(
        [self.kv_lora_rank, self.qk_rope_head_dim],
        dim=-1,
    )
    kv_pass = self.kv_a_layernorm(kv_pass)
    kv_pass_full = _gather_sequence(kv_pass, self._cp_group)
    k_position_full = _gather_sequence(k_position, self._cp_group)

    if self.indexer is not None:
        topk_indices = _select_sparse_indices(
            self.indexer,
            hidden_states,
            q_resid,
            attention_mask,
            cp_group=self._cp_group,
            cp_rank=self._cp_rank,
            cp_world_size=self._cp_world_size,
        )
    elif prev_topk_indices is None:
        raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
    else:
        topk_indices = prev_topk_indices

    kv_b_weight = self.kv_b_proj.weight.view(
        self.num_heads,
        self.qk_nope_head_dim + self.v_head_dim,
        self.kv_lora_rank,
    )
    key_weight = kv_b_weight[:, : self.qk_nope_head_dim]
    value_weight = kv_b_weight[:, self.qk_nope_head_dim :]
    query_absorbed = torch.einsum("bqhd,hdk->bqhk", q_nope, key_weight)
    sparse_query = torch.cat([query_absorbed, q_position], dim=-1)
    sparse_kv = torch.cat([kv_pass_full, k_position_full], dim=-1).unsqueeze(2)
    sparse_kv = torch.cat(
        [
            sparse_kv,
            sparse_kv.new_zeros(batch_size, 1, 1, sparse_kv.shape[-1]),
        ],
        dim=1,
    )

    kernel_indices = _pad_sparse_indices(topk_indices)
    if sparse_query.is_cuda and sparse_query.shape[-1] == 576:
        absorbed_output = sparse_mla_flashmla_apply(
            sparse_query,
            sparse_kv,
            kernel_indices.unsqueeze(2),
            self.scaling,
        )
    else:
        absorbed_output = _reference_sparse_mla(
            sparse_query,
            sparse_kv,
            kernel_indices,
            scale=self.scaling,
            value_dim=self.kv_lora_rank,
        )
    attention_output = torch.einsum("bqhk,hdk->bqhd", absorbed_output, value_weight)
    attention_output = attention_output.reshape(batch_size, local_length, -1)
    attention_output = self.o_proj(attention_output)
    return (
        attention_output,
        None,
        topk_indices if self.next_skip_topk else None,
    )


def _linear_attention_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    attention_mask: torch.Tensor | None = None,
    **_kwargs,
):
    if cache_params is not None:
        raise NotImplementedError("GLM-5.3 head-parallel KDA is a training-only path")
    if hidden_states.shape[0] != 1:
        raise NotImplementedError("GLM-5.3 head-parallel KDA currently requires one packed row")
    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask[..., None].to(hidden_states.dtype)

    batch_size, local_length = hidden_states.shape[:2]
    shape = (batch_size, local_length, self.num_heads, self.head_dim)
    query = self.q_proj(hidden_states).view(shape)
    key = self.k_proj(hidden_states).view(shape)
    value = self.v_proj(hidden_states).view(shape)
    decay = self.forget_gate(hidden_states)
    beta = torch.sigmoid(self.b_proj(hidden_states))
    exchanged = _pack_and_exchange_head_tensors(
        {
            "query": query,
            "key": key,
            "value": value,
            "decay": decay,
            "beta": beta,
        },
        process_group=self._cp_group,
    )

    world_size = self._cp_world_size
    local_heads = self.num_heads // world_size
    global_length = local_length * world_size
    conv_weight = _shard_convolution_parameter(
        self.conv1d.weight.squeeze(1),
        name="conv1d.weight",
        num_key_heads=self.num_heads,
        num_value_heads=self.num_heads,
        key_head_dim=self.head_dim,
        value_head_dim=self.head_dim,
        process_group=self._cp_group,
    )
    mixed_qkv = torch.cat(
        [
            exchanged["query"].reshape(batch_size, global_length, -1),
            exchanged["key"].reshape(batch_size, global_length, -1),
            exchanged["value"].reshape(batch_size, global_length, -1),
        ],
        dim=-1,
    ).transpose(1, 2)
    mixed_qkv = _fallback_depthwise_causal_convolution(
        mixed_qkv,
        weight=conv_weight,
        bias=None,
        activation=self.activation,
        global_cu_seqlens=getattr(self, _GLOBAL_CU_SEQLENS),
    ).transpose(1, 2)
    local_width = local_heads * self.head_dim
    query, key, value = torch.split(
        mixed_qkv,
        [local_width, local_width, local_width],
        dim=-1,
    )
    query = query.view(batch_size, global_length, local_heads, self.head_dim)
    key = key.view(batch_size, global_length, local_heads, self.head_dim)
    value = value.view(batch_size, global_length, local_heads, self.head_dim)

    from transformers.models.glm5_next.modeling_glm5_next import chunk_kimi_delta_attention

    output, final_state = chunk_kimi_delta_attention(
        query,
        key,
        value,
        g=exchanged["decay"],
        beta=exchanged["beta"],
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=getattr(self, _GLOBAL_CU_SEQLENS),
    )
    if final_state is not None:
        raise RuntimeError("GLM-5.3 KDA unexpectedly returned recurrent state")
    output = sequence_head_all_to_all(
        self._cp_group,
        output,
        scatter_dim=1,
        gather_dim=2,
    )
    gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(shape)
    output = self.o_norm(output, gate).reshape(batch_size, local_length, -1)
    return self.o_proj(output)


def _wrap_backbone_forward(backbone: nn.Module, parallel_modules: list[nn.Module]):
    original_forward = backbone.forward

    def forward(self, *args, **kwargs):
        position_ids = kwargs.get("position_ids")
        if not torch.is_tensor(position_ids):
            raise ValueError("GLM-5.3 context parallelism requires position_ids for packed boundaries")
        global_position_ids = _gather_sequence_no_grad(
            position_ids,
            self._cp_group,
        )
        cu_seqlens = cu_seqlens_from_position_ids(global_position_ids)
        for module in parallel_modules:
            setattr(module, _GLOBAL_POSITION_IDS, global_position_ids)
            setattr(module, _GLOBAL_CU_SEQLENS, cu_seqlens)
        return original_forward(*args, **kwargs)

    backbone.forward = types.MethodType(forward, backbone)


def apply_context_parallelism(model: nn.Module, cp_size: int, cp_group) -> None:
    backbone = get_language_model(model)
    if cp_size > 1:
        if cp_group is None:
            raise ValueError("GLM-5.3 context parallelism requires an SP process group")
        cp_world_size = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)
        if cp_world_size != cp_size:
            raise ValueError(f"GLM-5.3 CP group size ({cp_world_size}) does not match configured size ({cp_size})")
    else:
        cp_world_size = 1
        cp_rank = 0

    backbone._cp_group = cp_group
    backbone._cp_rank = cp_rank
    backbone._cp_world_size = cp_world_size
    sparse_layers = 0
    linear_layers = 0
    parallel_modules: list[nn.Module] = []
    for layer in backbone.layers:
        attention = layer.self_attn
        attention._cp_group = cp_group
        attention._cp_rank = cp_rank
        attention._cp_world_size = cp_world_size
        if layer.block_type == "deepseek_sparse_attention":
            attention.forward = types.MethodType(
                _sparse_attention_forward,
                attention,
            )
            sparse_layers += 1
        elif cp_size > 1 and layer.block_type == "linear_attention":
            if attention.num_heads % cp_world_size:
                raise ValueError(
                    f"GLM-5.3 KDA heads ({attention.num_heads}) must be divisible by CP size ({cp_world_size})"
                )
            attention.forward = types.MethodType(
                _linear_attention_forward,
                attention,
            )
            parallel_modules.append(attention)
            linear_layers += 1

    if not sparse_layers:
        raise TypeError("GLM-5.3 model exposes no sparse-attention layers")
    if cp_size > 1:
        _wrap_backbone_forward(backbone, parallel_modules)
    get_logger().info(
        "Applied GLM-5.3 sparse MLA and context parallelism (cp_size=%d, sparse_layers=%d, KDA_layers=%d)",
        cp_world_size,
        sparse_layers,
        linear_layers,
    )


__all__ = [
    "_pad_sparse_indices",
    "_reference_sparse_mla",
    "apply_context_parallelism",
]
