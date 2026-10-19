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

import functools
import importlib.util
import math
import os
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
from arctic_platform.model.implementations.gpu.sp.gated_delta_net import _packed_sequence_indices
from arctic_platform.model.implementations.gpu.sp.gated_delta_net import _shard_convolution_parameter
from arctic_platform.model.implementations.moe.logging_utils import get_logger
from arctic_platform.model.implementations.moe.vlm import get_language_model

_GLOBAL_POSITION_IDS = "_ap_glm53_global_position_ids"
_GLOBAL_CU_SEQLENS = "_ap_glm53_global_cu_seqlens"
# Bound one fp32 gathered-key tile while recomputing reference sparse-MLA backward.
_REFERENCE_SPARSE_MLA_TILE_BYTES = 1 << 30
_INDEXER_SCORE_CHUNK_BYTES = 512 * 1024 * 1024


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


def _normalize_cu_seqlens(
    cu_seqlens: torch.Tensor,
    *,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    if not torch.is_tensor(cu_seqlens) or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("GLM-5.3 packed boundaries must be a one-dimensional tensor with at least two entries")
    cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int32)
    if int(cu_seqlens[0]) != 0:
        raise ValueError("GLM-5.3 packed boundaries must start at zero")
    if bool((cu_seqlens[1:] <= cu_seqlens[:-1]).any()):
        raise ValueError("GLM-5.3 packed boundaries must be strictly increasing")
    real_length = int(cu_seqlens[-1])
    if real_length > sequence_length:
        raise ValueError(f"GLM-5.3 packed boundaries end at {real_length}, beyond sequence length {sequence_length}")
    if real_length < sequence_length:
        cu_seqlens = torch.cat(
            [
                cu_seqlens,
                torch.arange(
                    real_length + 1,
                    sequence_length + 1,
                    device=device,
                    dtype=torch.int32,
                ),
            ]
        )
    return cu_seqlens


@torch.no_grad()
def _select_sparse_indices(
    indexer: nn.Module,
    hidden_states: torch.Tensor,
    q_resid: torch.Tensor,
    attention_mask: torch.Tensor,
    global_cu_seqlens: torch.Tensor,
    *,
    cp_group,
    cp_rank: int,
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
    weights = indexer.weights_proj(hidden_states.to(indexer.weights_proj.weight.dtype)).float() * (
        indexer.n_heads**-0.5
    )
    output_width = indexer.index_topk
    if indexer.index_kpool_always_select_tail:
        output_width += indexer.index_kpool - 1
    topk_indices = torch.full(
        (batch_size, local_length, output_width),
        -1,
        dtype=torch.int32,
        device=hidden_states.device,
    )
    query_offset = cp_rank * local_length
    for start_tensor, end_tensor in zip(global_cu_seqlens[:-1], global_cu_seqlens[1:], strict=True):
        start = int(start_tensor)
        end = int(end_tensor)
        local_start = max(start, query_offset)
        local_end = min(end, query_offset + local_length)
        if local_start >= local_end:
            continue

        segment_states = packed_states[:, start:end]
        valid_keys = segment_states[..., -1].bool()
        segment_length = end - start
        key_positions = torch.arange(segment_length, device=hidden_states.device)
        pool_keys, pool_indices, pool_valid = indexer.get_pooled_states(packed_states=segment_states)
        pool_keys_transposed = pool_keys.transpose(-1, -2).float().unsqueeze(1)
        pool_end = pool_indices[..., -1].clamp(0, segment_length - 1)
        batch_idx = torch.arange(batch_size, device=hidden_states.device)[:, None, None]

        # Score every pooled key, but bound the temporary by processing query rows in chunks.
        score_bytes_per_query = batch_size * indexer.n_heads * pool_keys.shape[-2] * 4
        query_chunk_size = max(1, _INDEXER_SCORE_CHUNK_BYTES // max(1, score_bytes_per_query))
        for chunk_start in range(local_start, local_end, query_chunk_size):
            chunk_end = min(chunk_start + query_chunk_size, local_end)
            query_slice = slice(chunk_start - query_offset, chunk_end - query_offset)
            query_positions = torch.arange(chunk_start - start, chunk_end - start, device=hidden_states.device)
            visible_tokens = (key_positions[None, None, :] <= query_positions[None, :, None]) & valid_keys[:, None]
            scores = torch.matmul(query[:, query_slice].float(), pool_keys_transposed)
            scores.mul_(indexer.softmax_scale).relu_()
            index_scores = torch.matmul(weights[:, query_slice].unsqueeze(-2), scores).squeeze(-2)
            del scores

            pool_visible = visible_tokens.gather(
                dim=-1,
                index=pool_end[:, None].expand(batch_size, chunk_end - chunk_start, -1),
            )
            valid_candidates = pool_visible & pool_valid[:, None]
            index_scores.masked_fill_(~valid_candidates, torch.finfo(index_scores.dtype).min)

            select_k = min(indexer.index_topk // indexer.index_kpool, index_scores.shape[-1])
            selected = index_scores.topk(select_k, dim=-1).indices
            selected_valid = valid_candidates.gather(-1, selected)
            selected_indices = pool_indices[batch_idx, selected]
            selected_indices.masked_fill_(
                ~selected_valid[..., None].expand_as(selected_indices),
                -1,
            )
            segment_topk = selected_indices.flatten(-2)

            if indexer.index_kpool_always_select_tail:
                segment_topk = indexer.append_visible_tail(segment_topk, visible_tokens, valid_keys)
            segment_topk = F.pad(
                segment_topk,
                (0, output_width - segment_topk.shape[-1]),
                value=-1,
            )[..., :output_width]
            segment_topk = torch.where(segment_topk >= 0, segment_topk + start, segment_topk)
            segment_topk.masked_fill_(~attention_mask[:, query_slice, None], -1)
            topk_indices[:, query_slice] = segment_topk.to(torch.int32)
    return topk_indices


def _reference_sparse_mla_math(
    query: torch.Tensor,
    key_value: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    value_dim: int,
) -> torch.Tensor:
    kv_length = key_value.shape[1] - 1
    valid = (indices >= 0) & (indices < kv_length)
    safe = indices.long().clamp(0, kv_length)
    batch = torch.arange(query.shape[0], device=query.device)[:, None, None]
    gathered = key_value[batch, safe, 0].float()
    scores = torch.einsum("bqhd,bqkd->bqhk", query.float(), gathered)
    scores.mul_(scale).masked_fill_(~valid[:, :, None], float("-inf"))
    empty = ~valid.any(dim=-1)
    scores.masked_fill_(empty[:, :, None, None], 0)
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = probabilities.masked_fill(empty[:, :, None, None], 0)
    return torch.einsum(
        "bqhk,bqkd->bqhd",
        probabilities,
        gathered[..., :value_dim],
    ).to(query.dtype)


def _sparse_mla_query_tile(query_len: int, topk: int, dim: int) -> int:
    row_bytes = topk * dim * 4
    if row_bytes <= 0:
        return max(query_len, 1)
    return max(1, min(query_len, _REFERENCE_SPARSE_MLA_TILE_BYTES // row_bytes))


def _reference_sparse_mla_tiled(
    query: torch.Tensor,
    key_value: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    value_dim: int,
) -> torch.Tensor:
    query_len = query.shape[1]
    output = query.new_empty(query.shape[0], query_len, query.shape[2], value_dim)
    tile = _sparse_mla_query_tile(query_len, indices.shape[-1], key_value.shape[-1])
    for start in range(0, query_len, tile):
        end = min(start + tile, query_len)
        output[:, start:end] = _reference_sparse_mla_math(
            query[:, start:end],
            key_value,
            indices[:, start:end],
            scale,
            value_dim,
        )
    return output


# Hopper MLA scheduler keeps at most this many work items. One query token is one request.
_FLASHINFER_MAX_REQUESTS = 16384
_FLASHINFER_WRAPPERS: dict[torch.device, object] = {}


@functools.cache
def _flashinfer_available() -> bool:
    return importlib.util.find_spec("flashinfer") is not None


def _use_flashinfer(query: torch.Tensor, key_value: torch.Tensor) -> bool:
    return (
        os.environ.get("ARCTIC_GLM53_FLASHINFER") == "1"
        and query.is_cuda
        and query.dtype in (torch.bfloat16, torch.float16)
        and query.shape[-1] == 512
        and key_value.shape[-1] == 512
        and _flashinfer_available()
        and torch.cuda.get_device_capability(query.device)[0] == 9
    )


def _use_flashmla(query: torch.Tensor, value_dim: int) -> bool:
    return (
        query.is_cuda
        and value_dim == 512
        and query.shape[-1] in (512, 576)
        and torch.cuda.get_device_capability(query.device)[0] in (9, 10)
    )


def _flashinfer_wrapper(device: torch.device):
    import flashinfer

    key = torch.device(device)
    wrapper = _FLASHINFER_WRAPPERS.get(key)
    if wrapper is None:
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=key)
        wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="fa3")
        _FLASHINFER_WRAPPERS[key] = wrapper
    return wrapper


def _flashinfer_sparse_mla_requests(
    query: torch.Tensor,
    key_value: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    value_dim: int,
) -> torch.Tensor:
    """One FlashInfer plan. ``query`` must stay within ``_FLASHINFER_MAX_REQUESTS``."""
    kv_len = key_value.shape[1] - 1
    rows = indices[0]
    valid = (rows >= 0) & (rows < kv_len)
    counts = valid.sum(dim=-1).to(torch.int32)
    packed = rows.masked_select(valid).to(torch.int32).contiguous()
    if packed.numel() == 0:
        packed = rows.new_zeros(1, dtype=torch.int32)
    query_len, heads, _dim = query.shape[1:]
    qo_indptr = torch.arange(query_len + 1, device=query.device, dtype=torch.int32)
    kv_indptr = torch.zeros(query_len + 1, device=query.device, dtype=torch.int32)
    kv_indptr[1:] = counts.cumsum(0)
    wrapper = _flashinfer_wrapper(query.device)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        packed,
        counts,
        heads,
        query.shape[-1],
        0,
        1,
        False,
        scale,
        query.dtype,
        key_value.dtype,
    )
    absorbed = wrapper.run(
        query[0].contiguous(),
        query.new_empty(query_len, heads, 0),
        key_value[0, :kv_len].contiguous(),
        key_value.new_empty(kv_len, 1, 0),
    )
    # Empty sparse rows have zero output even if the backend leaves them undefined.
    absorbed.masked_fill_(counts[:, None, None] == 0, 0)
    return absorbed.unsqueeze(0)[..., :value_dim]


def _flashinfer_sparse_mla(
    query: torch.Tensor,
    key_value: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    value_dim: int,
) -> torch.Tensor:
    """Hopper FlashInfer MLA. Each query row is one page-size-1 request over its valid keys."""
    if query.shape[0] != 1:
        raise NotImplementedError("FlashInfer sparse MLA is wired for one packed row")
    query_len = query.shape[1]
    if query_len <= _FLASHINFER_MAX_REQUESTS:
        return _flashinfer_sparse_mla_requests(query, key_value, indices, scale, value_dim)
    output = query.new_empty(1, query_len, query.shape[2], value_dim)
    for start in range(0, query_len, _FLASHINFER_MAX_REQUESTS):
        end = min(start + _FLASHINFER_MAX_REQUESTS, query_len)
        output[:, start:end] = _flashinfer_sparse_mla_requests(
            query[:, start:end],
            key_value,
            indices[:, start:end],
            scale,
            value_dim,
        )
    return output


class _TiledSparseMLA(torch.autograd.Function):
    """Sparse MLA forward through FlashInfer or the tiled reference.

    Backward always recomputes the reference math one query tile at a time.
    """

    @staticmethod
    def forward(ctx, query, key_value, indices, scale, value_dim, use_flashinfer):
        ctx.scale = float(scale)
        ctx.value_dim = int(value_dim)
        ctx.save_for_backward(query, key_value, indices)
        with torch.no_grad():
            if use_flashinfer:
                return _flashinfer_sparse_mla(query, key_value, indices, ctx.scale, ctx.value_dim)
            return _reference_sparse_mla_tiled(query, key_value, indices, ctx.scale, ctx.value_dim)

    @staticmethod
    def backward(ctx, grad_output):
        query, key_value, indices = ctx.saved_tensors
        need_query, need_kv = ctx.needs_input_grad[:2]
        grad_query = torch.zeros_like(query) if need_query else None
        grad_kv = torch.zeros_like(key_value, dtype=torch.float32) if need_kv else None
        if need_query or need_kv:
            # FP32 leaves keep tile derivatives unrounded until the final KV-gradient cast.
            kv_source = key_value.detach().float() if need_kv else key_value.detach()
            query_len = query.shape[1]
            tile = _sparse_mla_query_tile(query_len, indices.shape[-1], key_value.shape[-1])
            grad_output = grad_output.contiguous()
            for start in range(0, query_len, tile):
                end = min(start + tile, query_len)
                query_tile = query[:, start:end].detach()
                kv_tile = kv_source.detach()
                if need_query:
                    query_tile = query_tile.requires_grad_(True)
                if need_kv:
                    kv_tile = kv_tile.requires_grad_(True)
                with torch.enable_grad():
                    tile_out = _reference_sparse_mla_math(
                        query_tile,
                        kv_tile,
                        indices[:, start:end],
                        ctx.scale,
                        ctx.value_dim,
                    )
                    torch.autograd.backward(tile_out, grad_output[:, start:end])
                if need_query:
                    grad_query[:, start:end] = query_tile.grad
                if need_kv:
                    grad_kv.add_(kv_tile.grad)
        if need_kv:
            grad_kv = grad_kv.to(key_value.dtype)
        return grad_query, grad_kv, None, None, None, None


def _reference_sparse_mla(
    query: torch.Tensor,
    key_value: torch.Tensor,
    indices: torch.Tensor,
    *,
    scale: float,
    value_dim: int,
) -> torch.Tensor:
    return _TiledSparseMLA.apply(query, key_value, indices, float(scale), int(value_dim), False)


def _flashinfer_sparse_mla_apply(
    query: torch.Tensor,
    key_value: torch.Tensor,
    indices: torch.Tensor,
    *,
    scale: float,
    value_dim: int,
) -> torch.Tensor:
    return _TiledSparseMLA.apply(query, key_value, indices, float(scale), int(value_dim), True)


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
        global_cu_seqlens = getattr(self, _GLOBAL_CU_SEQLENS, None)
        if global_cu_seqlens is None:
            global_cu_seqlens = torch.tensor(
                [0, local_length * self._cp_world_size],
                dtype=torch.int32,
                device=hidden_states.device,
            )
        topk_indices = _select_sparse_indices(
            self.indexer,
            hidden_states,
            q_resid,
            attention_mask,
            global_cu_seqlens,
            cp_group=self._cp_group,
            cp_rank=self._cp_rank,
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
    absorbed_output = None
    if (
        self.kv_lora_rank == 512
        and not getattr(self, "_ap_flashinfer_failed", False)
        and _use_flashinfer(sparse_query, sparse_kv)
    ):
        try:
            absorbed_output = _flashinfer_sparse_mla_apply(
                sparse_query,
                sparse_kv,
                kernel_indices,
                scale=self.scaling,
                value_dim=self.kv_lora_rank,
            )
        except torch.OutOfMemoryError:
            # Capacity failures must still fail the trial, not select another backend.
            raise
        except Exception as error:
            if isinstance(error, RuntimeError) and str(error).startswith("CUDA error"):
                raise
            self._ap_flashinfer_failed = True
            get_logger().warning(
                "GLM53_FLASHINFER_FALLBACK rank=%s error=%s; disabling FlashInfer for this module",
                dist.get_rank() if dist.is_initialized() else 0,
                error,
            )
        else:
            if not getattr(self, "_ap_flashinfer_dispatch_reported", False):
                get_logger().info(
                    "GLM53_FLASHINFER_EXECUTED rank=%s query_width=512 backward=reference",
                    dist.get_rank() if dist.is_initialized() else 0,
                )
                self._ap_flashinfer_dispatch_reported = True
    if absorbed_output is None and _use_flashmla(sparse_query, self.kv_lora_rank):
        # The rope-free checkpoint keeps the same dot products with zero-padded kernel channels.
        padding = 576 - sparse_query.shape[-1]
        if padding:
            sparse_query = F.pad(sparse_query, (0, padding))
            sparse_kv = F.pad(sparse_kv, (0, padding))
        absorbed_output = sparse_mla_flashmla_apply(
            sparse_query,
            sparse_kv,
            kernel_indices.unsqueeze(2),
            self.scaling,
        )
        if not getattr(self, "_ap_flashmla_dispatch_reported", False):
            get_logger().info(
                "GLM53_FLASHMLA_EXECUTED rank=%s query_width=%s kernel_width=576 backward=reference",
                dist.get_rank() if dist.is_initialized() else 0,
                576 - padding,
            )
            self._ap_flashmla_dispatch_reported = True
    elif absorbed_output is None:
        absorbed_output = _reference_sparse_mla(
            sparse_query,
            sparse_kv,
            kernel_indices,
            scale=self.scaling,
            value_dim=self.kv_lora_rank,
        )
        if not getattr(self, "_ap_reference_dispatch_reported", False):
            get_logger().info(
                "GLM53_REFERENCE_EXECUTED rank=%s kv_lora_rank=%s query_width=%s is_cuda=%s backward=tiled",
                dist.get_rank() if dist.is_initialized() else 0,
                self.kv_lora_rank,
                sparse_query.shape[-1],
                sparse_query.is_cuda,
            )
            self._ap_reference_dispatch_reported = True
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


def _local_linear_attention_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    attention_mask: torch.Tensor | None = None,
    **_kwargs,
):
    if cache_params is not None:
        raise NotImplementedError("GLM-5.3 packed KDA is a training-only path")
    if hidden_states.shape[0] != 1:
        raise NotImplementedError("GLM-5.3 packed KDA currently requires one packed row")
    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask[..., None].to(hidden_states.dtype)

    batch_size, sequence_length = hidden_states.shape[:2]
    global_cu_seqlens = getattr(self, _GLOBAL_CU_SEQLENS, None)
    if global_cu_seqlens is None:
        global_cu_seqlens = torch.tensor(
            [0, sequence_length],
            dtype=torch.int32,
            device=hidden_states.device,
        )
    shape = (batch_size, sequence_length, self.num_heads, self.head_dim)
    mixed_qkv = torch.cat(
        [
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        ],
        dim=-1,
    ).transpose(1, 2)

    from causal_conv1d import causal_conv1d_fn

    mixed_qkv = causal_conv1d_fn(
        mixed_qkv,
        weight=self.conv1d.weight.squeeze(1),
        bias=self.conv1d.bias,
        activation=self.activation,
        seq_idx=_packed_sequence_indices(global_cu_seqlens),
    ).transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.qkv_dim] * 3, dim=-1)
    query = query.view(shape)
    key = key.view(shape)
    value = value.view(shape)
    decay = self.forget_gate(hidden_states)
    beta = torch.sigmoid(self.b_proj(hidden_states))

    from fla.ops.kda import chunk_kda

    output, final_state = chunk_kda(
        query,
        key,
        value,
        g=decay,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=global_cu_seqlens,
    )
    if final_state is not None:
        raise RuntimeError("GLM-5.3 KDA unexpectedly returned recurrent state")
    gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(shape)
    output = self.o_norm(output, gate).reshape(batch_size, sequence_length, -1)
    return self.o_proj(output)


def _wrap_backbone_forward(backbone: nn.Module, parallel_modules: list[nn.Module]):
    original_forward = backbone.forward

    def forward(self, *args, **kwargs):
        position_ids = kwargs.get("position_ids")
        if position_ids is None and len(args) >= 3:
            position_ids = args[2]
        input_tensor = kwargs.get("input_ids")
        if input_tensor is None:
            input_tensor = kwargs.get("inputs_embeds")
        if input_tensor is None and args:
            input_tensor = args[0]
        if not torch.is_tensor(input_tensor) or input_tensor.ndim < 2:
            raise ValueError("GLM-5.3 needs input_ids or inputs_embeds to determine the sequence length")
        global_sequence_length = int(input_tensor.shape[1]) * self._cp_world_size
        global_position_ids = (
            _gather_sequence_no_grad(position_ids, self._cp_group) if torch.is_tensor(position_ids) else None
        )
        cu_seqlens = kwargs.get("cu_seq_lens_q")
        if cu_seqlens is None:
            cu_seqlens = (
                cu_seqlens_from_position_ids(global_position_ids)
                if global_position_ids is not None
                else torch.tensor(
                    [0, global_sequence_length],
                    dtype=torch.int32,
                    device=input_tensor.device,
                )
            )
        cu_seqlens = _normalize_cu_seqlens(
            cu_seqlens,
            sequence_length=global_sequence_length,
            device=input_tensor.device,
        )
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
            parallel_modules.append(attention)
            sparse_layers += 1
        elif layer.block_type == "linear_attention":
            if attention.num_heads % cp_world_size:
                raise ValueError(
                    f"GLM-5.3 KDA heads ({attention.num_heads}) must be divisible by CP size ({cp_world_size})"
                )
            attention.forward = types.MethodType(
                _linear_attention_forward if cp_size > 1 else _local_linear_attention_forward,
                attention,
            )
            parallel_modules.append(attention)
            linear_layers += 1

    if not sparse_layers:
        raise TypeError("GLM-5.3 model exposes no sparse-attention layers")
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
