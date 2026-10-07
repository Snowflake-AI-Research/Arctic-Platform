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

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA NeMo AutoModel commit
# 3914f200a4c782d44b58ee7a01b4685e4158e19c.

"""FlexAttention backend for Qwen3.8-Flash-Next QSA."""

from __future__ import annotations

import functools
import math
import types
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import flex_attention

_BACKEND = "qsa_flex"
_QSA_SCORE_BUDGET_BYTES = 256 * 1024 * 1024
_QSA_ATTENTION_CHUNK_ROWS = 128


def _cp_info(module) -> tuple[object | None, int, int]:
    group = getattr(module, "_cp_group", None)
    if group is None:
        return None, 0, 1
    return group, dist.get_rank(group), dist.get_world_size(group)


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


def _compact_qsa_mask(*, attention_mask=None, **_kwargs):
    if attention_mask is not None and attention_mask.ndim != 2:
        raise NotImplementedError("Qwen3.8 QSA FlexAttention supports only a 2D right-padded attention mask")
    return attention_mask


def _unpatched_qsa_attention(*_args, **_kwargs):
    raise RuntimeError("Qwen3.8 QSA FlexAttention was selected before its model patch was applied")


def register_qsa_flex_backend() -> None:
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(_BACKEND, _unpatched_qsa_attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register(_BACKEND, _compact_qsa_mask)


def _right_padded_lengths(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    if attention_mask is None:
        return torch.full((batch_size,), sequence_length, dtype=torch.long, device=device)
    if attention_mask.shape != (batch_size, sequence_length):
        raise NotImplementedError(
            "Qwen3.8 QSA FlexAttention requires a [batch, sequence] right-padded mask; "
            f"got {tuple(attention_mask.shape)}"
        )
    valid = attention_mask.to(device=device).bool()
    lengths = valid.sum(dim=-1, dtype=torch.long)
    expected = torch.arange(sequence_length, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    if not torch.equal(valid, expected):
        raise NotImplementedError("Qwen3.8 QSA FlexAttention does not yet support packed or left-padded batches")
    return lengths


def _query_chunk_rows(*, num_heads: int, num_blocks: int, rows: int) -> int:
    bytes_per_row = max(num_heads * num_blocks * 4, 1)
    return max(1, min(rows, _QSA_SCORE_BUDGET_BYTES // bytes_per_row))


@torch.no_grad()
def select_qsa_token_ids(
    index_queries: torch.Tensor,
    compressed_keys: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    token_budget: int,
    compress_ratio: int,
    query_offset: int = 0,
) -> torch.Tensor:
    batch_size, sequence_length, num_heads, head_dim = index_queries.shape
    block_budget = token_budget // compress_ratio
    route_width = token_budget + compress_ratio - 1
    selected_tokens = torch.full(
        (batch_size, sequence_length, route_width),
        -1,
        dtype=torch.int32,
        device=index_queries.device,
    )
    block_offsets = torch.arange(compress_ratio, device=index_queries.device)
    tail_offsets = torch.arange(compress_ratio - 1, device=index_queries.device)
    first_sparse_position = (block_budget + 1) * compress_ratio - 1

    for batch_idx, logical_length_tensor in enumerate(sequence_lengths):
        logical_length = int(logical_length_tensor)
        available_blocks = logical_length // compress_ratio
        local_valid_length = min(max(logical_length - query_offset, 0), sequence_length)
        if local_valid_length == 0:
            continue
        keys = compressed_keys[batch_idx, :available_blocks, 0].float()
        topk_width = min(block_budget, available_blocks)
        chunk_rows = _query_chunk_rows(
            num_heads=num_heads,
            num_blocks=max(available_blocks, 1),
            rows=local_valid_length,
        )
        for query_start in range(0, local_valid_length, chunk_rows):
            query_end = min(query_start + chunk_rows, local_valid_length)
            rows = query_end - query_start
            query_positions = query_offset + torch.arange(query_start, query_end, device=index_queries.device)
            visible_blocks = torch.div(query_positions + 1, compress_ratio, rounding_mode="floor")
            result = torch.full(
                (rows, route_width),
                -1,
                dtype=torch.int32,
                device=index_queries.device,
            )

            if topk_width:
                candidate_blocks = torch.arange(topk_width, device=index_queries.device)
                top_blocks = candidate_blocks.unsqueeze(0).expand(rows, -1)
                valid_blocks = candidate_blocks.unsqueeze(0) < visible_blocks.unsqueeze(1)
                sparse_start = min(max(first_sparse_position - query_offset - query_start, 0), rows)
                if sparse_start < rows:
                    queries = index_queries[batch_idx, query_start + sparse_start : query_end].float()
                    scores = torch.einsum("qhd,pd->qhp", queries, keys)
                    scores = torch.relu(scores).sum(dim=1) / math.sqrt(head_dim)
                    block_ids = torch.arange(available_blocks, device=index_queries.device)
                    scores.masked_fill_(
                        block_ids.unsqueeze(0) >= visible_blocks[sparse_start:, None],
                        -torch.inf,
                    )
                    sparse_top_blocks = torch.topk(scores, k=block_budget, dim=-1).indices
                    top_blocks = torch.cat([top_blocks[:sparse_start], sparse_top_blocks], dim=0)
                    valid_blocks = torch.cat(
                        [
                            valid_blocks[:sparse_start],
                            torch.ones_like(valid_blocks[sparse_start:]),
                        ],
                        dim=0,
                    )
                expanded = top_blocks.unsqueeze(-1) * compress_ratio + block_offsets
                expanded = torch.where(valid_blocks.unsqueeze(-1), expanded, -1)
                result[:, : topk_width * compress_ratio] = expanded.reshape(rows, -1).to(torch.int32)

            tail_start = visible_blocks * compress_ratio
            tail_count = query_positions + 1 - tail_start
            valid_block_count = torch.clamp(visible_blocks, max=block_budget)
            tail_values = tail_start[:, None] + tail_offsets[None, :]
            tail_values = torch.where(tail_offsets[None, :] < tail_count[:, None], tail_values, -1)
            destinations = valid_block_count[:, None] * compress_ratio + tail_offsets[None, :]
            result.scatter_(1, destinations, tail_values.to(torch.int32))
            selected_tokens[batch_idx, query_start:query_end] = result

    return selected_tokens


def _qsa_indexer_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
) -> torch.Tensor:
    if past_key_values is not None and past_key_values.get_seq_length() > 0:
        raise NotImplementedError("Qwen3.8 QSA FlexAttention cache decoding is handled by the inference backend")
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    batch_size, local_sequence_length, _ = hidden_states.shape
    cp_group, cp_rank, cp_world_size = _cp_info(self)
    global_attention_mask = (
        _gather_sequence_no_grad(attention_mask, cp_group) if attention_mask is not None else None
    )
    global_sequence_length = local_sequence_length * cp_world_size
    lengths = _right_padded_lengths(
        global_attention_mask,
        batch_size=batch_size,
        sequence_length=global_sequence_length,
        device=hidden_states.device,
    )
    local_cos, local_sin = position_embeddings
    projected = self.index_qk_proj(hidden_states)
    query_width = self.index_n_heads * self.index_head_dim
    raw_query, raw_key = torch.split(projected, [query_width, self.index_head_dim], dim=-1)
    query = raw_query.unflatten(-1, (self.index_n_heads, self.index_head_dim))
    raw_key = raw_key.unflatten(-1, (1, self.index_head_dim))
    query = apply_rotary_pos_emb(
        self.q_layernorm(query),
        cos=local_cos[:, -local_sequence_length:],
        sin=local_sin[:, -local_sequence_length:],
        unsqueeze_dim=2,
    )

    num_blocks = local_sequence_length // self.compress_ratio
    grouped_key = raw_key[:, : num_blocks * self.compress_ratio].unflatten(
        1, (num_blocks, self.compress_ratio)
    )
    compressed_key = grouped_key.float().mean(dim=2).to(raw_key.dtype)
    compressed_key = apply_rotary_pos_emb(
        self.k_layernorm(compressed_key),
        cos=local_cos[:, : num_blocks * self.compress_ratio : self.compress_ratio],
        sin=local_sin[:, : num_blocks * self.compress_ratio : self.compress_ratio],
        unsqueeze_dim=2,
    )
    compressed_key = _gather_sequence_no_grad(compressed_key, cp_group)
    return select_qsa_token_ids(
        query,
        compressed_key,
        lengths,
        token_budget=self.token_budget,
        compress_ratio=self.compress_ratio,
        query_offset=cp_rank * local_sequence_length,
    )


@functools.cache
def _compiled_flex() -> Callable:
    return torch.compile(flex_attention, dynamic=False)


def _routes_to_block_mask(
    selected_token_ids: torch.Tensor,
    kv_length: int,
) -> tuple[object, torch.Tensor]:
    batch_size, query_length, _ = selected_token_ids.shape
    valid = (selected_token_ids >= 0) & (selected_token_ids < kv_length)
    hit_counts = torch.zeros(
        batch_size,
        query_length,
        kv_length,
        dtype=torch.int32,
        device=selected_token_ids.device,
    )
    safe_ids = selected_token_ids.long().clamp(min=0, max=kv_length - 1)
    hit_counts.scatter_add_(-1, safe_ids, valid.to(torch.int32))
    membership = hit_counts > 0
    has_routes = valid.any(dim=-1)
    membership[..., 0] |= ~has_routes
    membership_flat = membership.reshape(-1)

    def mask_mod(batch_idx, _head_idx, query_idx, kv_idx):
        in_range = (query_idx < query_length) & (kv_idx < kv_length)
        safe_query = torch.clamp(query_idx, max=query_length - 1)
        safe_kv = torch.clamp(kv_idx, max=kv_length - 1)
        flat_query = batch_idx.to(torch.int64) * query_length + safe_query.to(torch.int64)
        offset = flat_query * kv_length + safe_kv.to(torch.int64)
        return in_range & membership_flat[offset]

    block_mask = create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=query_length,
        KV_LEN=kv_length,
        device=str(selected_token_ids.device),
    )
    return block_mask, has_routes


def _sparse_gqa_from_selected(
    query: torch.Tensor,
    selected_key: torch.Tensor,
    selected_value: torch.Tensor,
    valid: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    batch_size, query_length, num_query_heads, head_dim = query.shape
    num_kv_heads = selected_key.shape[3]
    if num_query_heads % num_kv_heads:
        raise ValueError(
            f"Qwen3.8 QSA query heads ({num_query_heads}) must be divisible by KV heads ({num_kv_heads})"
        )

    groups = num_query_heads // num_kv_heads
    grouped_query = query.view(batch_size, query_length, num_kv_heads, groups, head_dim)
    scores = torch.einsum(
        "bqhgd,bqrhd->bqhgr",
        grouped_query.float(),
        selected_key.float(),
    )
    scores.mul_(softmax_scale)
    scores.masked_fill_(~valid[:, :, None, None], -torch.inf)
    empty = ~valid.any(dim=-1)
    scores.masked_fill_(empty[:, :, None, None, None], 0)
    probabilities = torch.softmax(scores, dim=-1).to(selected_value.dtype)
    probabilities.masked_fill_(~valid[:, :, None, None], 0)
    output = torch.einsum(
        "bqhgr,bqrhd->bqhgd",
        probabilities,
        selected_value,
    ).reshape(batch_size, query_length, num_query_heads, head_dim)
    output.masked_fill_(empty[:, :, None, None], 0)
    return output


def _sparse_gqa_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_token_ids: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = (selected_token_ids >= 0) & (selected_token_ids < key.shape[1])
    safe_ids = selected_token_ids.long().clamp(min=0, max=key.shape[1] - 1)
    batch_indices = torch.arange(query.shape[0], device=query.device)[:, None, None]
    selected_key = key[batch_indices, safe_ids]
    selected_value = value[batch_indices, safe_ids]
    output = _sparse_gqa_from_selected(
        query,
        selected_key,
        selected_value,
        valid,
        softmax_scale,
    )
    return output, safe_ids, selected_key, selected_value


class _SparseGQAAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, selected_token_ids, softmax_scale, chunk_rows):
        ctx.softmax_scale = float(softmax_scale)
        ctx.chunk_rows = int(chunk_rows)
        ctx.save_for_backward(query, key, value, selected_token_ids)
        outputs = []
        for start in range(0, query.shape[1], ctx.chunk_rows):
            stop = min(start + ctx.chunk_rows, query.shape[1])
            output, *_ = _sparse_gqa_chunk(
                query[:, start:stop],
                key,
                value,
                selected_token_ids[:, start:stop],
                ctx.softmax_scale,
            )
            outputs.append(output)
        return torch.cat(outputs, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, selected_token_ids = ctx.saved_tensors
        grad_query = torch.empty_like(query)
        grad_key = torch.zeros_like(key)
        grad_value = torch.zeros_like(value)
        flattened_grad_key = grad_key.flatten(2)
        flattened_grad_value = grad_value.flatten(2)
        kv_width = flattened_grad_key.shape[-1]

        for start in range(0, query.shape[1], ctx.chunk_rows):
            stop = min(start + ctx.chunk_rows, query.shape[1])
            routes = selected_token_ids[:, start:stop]
            valid = (routes >= 0) & (routes < key.shape[1])
            with torch.enable_grad():
                local_query = query[:, start:stop].detach().requires_grad_(True)
                safe_ids = routes.long().clamp(min=0, max=key.shape[1] - 1)
                batch_indices = torch.arange(query.shape[0], device=query.device)[:, None, None]
                selected_key = key.detach()[batch_indices, safe_ids]
                selected_value = value.detach()[batch_indices, safe_ids]
                selected_key = selected_key.detach().requires_grad_(True)
                selected_value = selected_value.detach().requires_grad_(True)
                output = _sparse_gqa_from_selected(
                    local_query,
                    selected_key,
                    selected_value,
                    valid,
                    ctx.softmax_scale,
                )
                local_grad_query, local_grad_key, local_grad_value = torch.autograd.grad(
                    output,
                    (local_query, selected_key, selected_value),
                    grad_output[:, start:stop],
                )

            grad_query[:, start:stop] = local_grad_query
            scatter_indices = safe_ids.reshape(safe_ids.shape[0], -1, 1).expand(-1, -1, kv_width)
            valid_values = valid[..., None, None]
            flattened_grad_key.scatter_add_(
                1,
                scatter_indices,
                (local_grad_key * valid_values).reshape(safe_ids.shape[0], -1, kv_width),
            )
            flattened_grad_value.scatter_add_(
                1,
                scatter_indices,
                (local_grad_value * valid_values).reshape(safe_ids.shape[0], -1, kv_width),
            )

        return grad_query, grad_key, grad_value, None, None, None


def flex_sparse_gqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_token_ids: torch.Tensor,
    *,
    softmax_scale: float,
) -> torch.Tensor:
    if not query.is_cuda or any(tensor.dtype != torch.bfloat16 for tensor in (query, key, value)):
        raise RuntimeError("Qwen3.8 QSA FlexAttention requires CUDA BF16 Q/K/V tensors")
    return _SparseGQAAttention.apply(
        query,
        key,
        value,
        selected_token_ids,
        softmax_scale,
        _QSA_ATTENTION_CHUNK_ROWS,
    )


def _qsa_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    **_kwargs,
):
    if self.training and self.attention_dropout:
        raise ValueError("Qwen3.8 QSA FlexAttention does not support attention dropout")
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    selected_token_ids = self.indexer(hidden_states, position_embeddings, attention_mask, past_key_values)
    position_embeddings = tuple(value[:, -hidden_states.shape[1] :] for value in position_embeddings)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
        2,
        dim=-1,
    )
    gate = gate.reshape(*input_shape, -1)
    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        *position_embeddings,
    )
    if past_key_values is not None and past_key_values.get_seq_length() > 0:
        raise NotImplementedError("Qwen3.8 QSA FlexAttention cache decoding is handled by the inference backend")
    cp_group, _, _ = _cp_info(self)
    key_states = _gather_sequence(key_states.transpose(1, 2), cp_group)
    value_states = _gather_sequence(value_states.transpose(1, 2), cp_group)
    attention_output = flex_sparse_gqa_attention(
        query_states.transpose(1, 2),
        key_states,
        value_states,
        selected_token_ids,
        softmax_scale=self.scaling,
    )
    attention_output = attention_output.reshape(*input_shape, -1).contiguous()
    attention_output = attention_output * torch.sigmoid(gate)
    return self.o_proj(attention_output), None


def apply_qsa_flex(model: nn.Module) -> None:
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextAttention

    register_qsa_flex_backend()
    patched = 0
    for module in model.modules():
        if not isinstance(module, Qwen4ExpTextAttention):
            continue
        module.indexer.requires_grad_(False)
        module.indexer.forward = types.MethodType(_qsa_indexer_forward, module.indexer)
        module.forward = types.MethodType(_qsa_attention_forward, module)
        patched += 1
    if not patched:
        raise TypeError("Qwen3.8 model exposes no QSA attention layers")


__all__ = [
    "apply_qsa_flex",
    "flex_sparse_gqa_attention",
    "register_qsa_flex_backend",
    "select_qsa_token_ids",
]
