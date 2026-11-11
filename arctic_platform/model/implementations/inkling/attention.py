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

"""Query-chunked FlexAttention with Inkling's learned relative bias."""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.distributed as dist
import torch.nn.functional as F
from packaging.version import Version
from torch.fx.experimental import _config as fx_config
from torch.nn.attention.flex_attention import AuxRequest
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention
from torch.utils.checkpoint import checkpoint
from transformers.models.inkling.modeling_inkling import InklingAttention as NativeInklingAttention
from transformers.models.inkling.modeling_inkling import InklingTextModel as NativeInklingTextModel

from arctic_platform.model.implementations.gpu.sp.collectives import gather_sequence
from arctic_platform.model.implementations.gpu.sp.collectives import sequence_head_all_to_all

_BLOCK = 128


@lru_cache(maxsize=32)
def _compiled_attention(geometry):
    def run(*args, **kwargs):
        return flex_attention(*args, **kwargs)

    # Separate Dynamo caches for dtype, head geometry, and mask structure.
    run.__code__ = run.__code__.replace(co_name=f"inkling_flex_{geometry}")
    compiled = torch.compile(run, fullgraph=True, dynamic=True)

    # Chunk and split lengths vary freely; size-1 and equal-size specializations each cost a cache entry.
    def call(*args, **kwargs):
        with torch._dynamo.config.patch(recompile_limit=64), fx_config.patch(use_duck_shape=False):
            return compiled(*args, **kwargs)

    return call


def relative_attention(
    query,
    key,
    value,
    relative,
    projection,
    *,
    query_offset,
    key_offset=0,
    padding_mask=None,
    sequence_ids=None,
    window=None,
    scaling=None,
    bias_scale=None,
    return_lse=False,
    mask_cache=None,
):
    """One query chunk; no token-by-token attention or bias matrix is allocated.

    Keys at least the bias extent behind every query of the chunk carry no bias. They run through a kernel without
    a score_mod, whose result merges with the biased near keys by log-sum-exp. Both select their keys by block
    rather than by slicing: sliced keys lose the batch-stride relation and compile to a slower kernel.

    With ``return_lse``, also returns the FP32 log-sum-exp of the scores, ``[batch, heads, q_len]``. A
    ``mask_cache`` dict reuses block masks across calls with the same chunk geometry and the same padding and
    sequence ids.
    """
    extent = projection.shape[1]
    bank = (relative @ projection).transpose(1, 2).contiguous()
    if bias_scale is not None:
        bank = (bank.float() * bias_scale[None, None, :, None]).to(bank.dtype)
    split = 0
    if window is None or window > extent:
        split = max(query_offset - extent + 1 - key_offset, 0) // _BLOCK * _BLOCK
    masks = dict(
        padding_mask=padding_mask, sequence_ids=sequence_ids, window=window, scaling=scaling, mask_cache=mask_cache
    )
    near, near_lse = _flex_chunk(query, key, value, bank, query_offset, key_offset, (split, None), **masks)
    if split == 0:
        return (near, near_lse) if return_lse else near
    far, far_lse = _flex_chunk(query, key, value, None, query_offset, key_offset, (0, split), **masks)
    top = torch.maximum(near_lse, far_lse)
    top = torch.where(torch.isfinite(top), top, 0).detach()
    near_weight, far_weight = (near_lse - top).exp(), (far_lse - top).exp()
    total = near_weight + far_weight
    # Rows with every key masked keep a zero output and -inf log-sum-exp without NaN gradients.
    safe = torch.where(total > 0, total, 1)
    output = (near.float() * near_weight[..., None] + far.float() * far_weight[..., None]) / safe[..., None]
    output = output.to(query.dtype)
    if not return_lse:
        return output
    return output, torch.where(total > 0, top + safe.log(), float("-inf"))


def _block_mask(batch, q_len, kv_len, query_offset, key_offset, key_range, padding_mask, sequence_ids, window, device):
    q_offset = torch.full((), query_offset, dtype=torch.int64, device=device)
    k_offset = torch.full((), key_offset, dtype=torch.int64, device=device)

    def mask_mod(b, h, qi, ki):
        q_pos, k_pos = qi + q_offset, ki + k_offset
        valid = q_pos >= k_pos
        if window is not None:
            valid = valid & (q_pos - k_pos < window)
        if padding_mask is not None:
            valid = valid & padding_mask[b, k_pos.clamp(max=padding_mask.shape[1] - 1)]
        if sequence_ids is not None:
            valid = valid & (
                sequence_ids[b, q_pos.clamp(max=sequence_ids.shape[1] - 1)]
                == sequence_ids[b, k_pos.clamp(max=sequence_ids.shape[1] - 1)]
            )
        return valid

    mask = BlockMask.from_kv_blocks(
        *_block_mask_tensors(
            batch, q_len, kv_len, query_offset, key_offset, key_range, padding_mask, sequence_ids, window, device
        ),
        BLOCK_SIZE=_BLOCK,
        mask_mod=mask_mod,
        seq_lengths=(q_len, kv_len),
    )
    return mask, q_offset, k_offset


def _block_mask_tensors(
    batch, q_len, kv_len, query_offset, key_offset, key_range, padding_mask, sequence_ids, window, device
):
    """Partial and full KV block lists. Blocks with no valid pair are skipped and fully valid blocks skip mask_mod;
    ``sequence_ids`` are nondecreasing, so a block's first and last ids bound the sequences it holds."""
    q_blocks, kv_blocks = -(-q_len // _BLOCK), -(-kv_len // _BLOCK)
    q_first = torch.arange(q_blocks, device=device) * _BLOCK + query_offset
    q_last = (q_first + _BLOCK - 1).clamp(max=query_offset + q_len - 1)
    k_first = torch.arange(kv_blocks, device=device) * _BLOCK + key_offset
    k_last = (k_first + _BLOCK - 1).clamp(max=key_offset + kv_len - 1)
    first, last = key_range
    selected = torch.arange(kv_blocks, device=device) >= first // _BLOCK
    if last is not None:
        selected = selected & (torch.arange(kv_blocks, device=device) < last // _BLOCK)
    some = (k_first[None] <= q_last[:, None]) & selected
    full = (k_last[None] <= q_first[:, None]) & selected
    if window is not None:
        some = some & (q_first[:, None] - k_last[None] < window)
        full = full & (q_last[:, None] - k_first[None] < window)
    some, full = some.expand(batch, -1, -1), full.expand(batch, -1, -1)
    if sequence_ids is not None:
        cap = sequence_ids.shape[1] - 1
        q_lo, q_hi = sequence_ids[:, q_first.clamp(max=cap)], sequence_ids[:, q_last.clamp(max=cap)]
        k_lo, k_hi = sequence_ids[:, k_first.clamp(max=cap)], sequence_ids[:, k_last.clamp(max=cap)]
        some = some & (k_lo[:, None] <= q_hi[..., None]) & (k_hi[:, None] >= q_lo[..., None])
        full = full & ((q_lo == q_hi)[..., None] & (k_lo == k_hi)[:, None] & (q_lo[..., None] == k_lo[:, None]))
    if padding_mask is not None:
        keys = padding_mask[:, key_offset : key_offset + kv_len].to(torch.int8)
        keys = F.pad(keys, (0, kv_blocks * _BLOCK - kv_len)).view(batch, kv_blocks, _BLOCK).bool()
        some = some & keys.any(-1)[:, None]
        full = full & keys.all(-1)[:, None]

    def ordered(dense):
        counts = dense.sum(-1, dtype=torch.int32)
        indices = torch.argsort(dense.to(torch.int8), dim=-1, descending=True, stable=True).to(torch.int32)
        return counts[:, None].contiguous(), indices[:, None].contiguous()

    return (*ordered(some & ~full), *ordered(full))


def _flex_chunk(
    query,
    key,
    value,
    bank,
    query_offset,
    key_offset,
    key_range,
    *,
    padding_mask,
    sequence_ids,
    window,
    scaling,
    mask_cache,
):
    batch, heads, q_len, dim = query.shape
    kv_len = key.shape[2]
    extent = 0 if bank is None else bank.shape[-1]
    cache_key = (batch, q_len, kv_len, query_offset, key_offset, key_range, window)
    cached = None if mask_cache is None else mask_cache.get(cache_key)
    if cached is None:
        cached = _block_mask(
            batch, q_len, kv_len, query_offset, key_offset, key_range, padding_mask, sequence_ids, window, query.device
        )
        if mask_cache is not None:
            mask_cache[cache_key] = cached
    mask, q_offset, k_offset = cached

    def score_mod(score, b, h, qi, ki):
        distance = qi + q_offset - ki - k_offset
        bias = bank[b, h, qi, distance.clamp(0, extent - 1)]
        return score + torch.where((distance >= 0) & (distance < extent), bias, 0)

    geometry = (
        query.dtype,
        heads,
        key.shape[1],
        dim,
        window,
        padding_mask is not None,
        sequence_ids is not None,
        bank is not None,
    )
    output, aux = _compiled_attention(geometry)(
        query,
        key,
        value,
        score_mod=score_mod if bank is not None else None,
        block_mask=mask,
        enable_gqa=heads != key.shape[1],
        scale=scaling if scaling is not None else 1 / dim,
        kernel_options={"BACKEND": "TRITON"},
        return_aux=AuxRequest(lse=True),
    )
    return output, aux.lse


class InklingFlexAttention(NativeInklingAttention):
    def forward(self, hidden_states, attention_mask=None, conv_mask=None, past_key_values=None, **kwargs):
        if past_key_values is not None:
            raise ValueError("Inkling FlexAttention training does not support KV caching")
        if self.training and self.attention_dropout:
            raise ValueError("Inkling FlexAttention requires attention_dropout=0")
        batch, length, _ = hidden_states.shape
        chunk_size = self.config.inkling_attention_query_chunk_size

        def query_projection(hidden):
            query = self.q_proj(hidden).view(batch, hidden.shape[1], self.num_heads, self.head_dim)
            return self.q_norm(query).transpose(1, 2).contiguous()

        def key_projection(hidden):
            key = self.k_sconv(self.k_proj(hidden), conv_mask=conv_mask)
            return self.k_norm(key.view(batch, hidden.shape[1], self.num_key_value_heads, self.head_dim))

        def value_projection(hidden):
            return self.v_sconv(self.v_proj(hidden), conv_mask=conv_mask)

        def project(fn, hidden):
            return checkpoint(fn, hidden, use_reentrant=False) if torch.is_grad_enabled() else fn(hidden)

        query = torch.cat(
            [project(query_projection, hidden) for hidden in hidden_states.split(chunk_size, dim=1)], dim=2
        )
        key, value = (project(key_projection, hidden_states), project(value_projection, hidden_states))
        key = key.transpose(1, 2).contiguous()
        value = value.view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2).contiguous()
        relative = self.r_proj(hidden_states).view(batch, length, self.num_heads, -1)
        group = getattr(self, "_sp_group", None)
        if group is not None:
            size = dist.get_world_size(group)
            query = sequence_head_all_to_all(group, query.transpose(1, 2), scatter_dim=2, gather_dim=1)
            relative = sequence_head_all_to_all(group, relative, scatter_dim=2, gather_dim=1)
            key, value = key.transpose(1, 2), value.transpose(1, 2)
            if size > self.num_key_value_heads:
                repeats = size // self.num_key_value_heads
                key, value = key.repeat_interleave(repeats, dim=2), value.repeat_interleave(repeats, dim=2)
            key = sequence_head_all_to_all(group, key, scatter_dim=2, gather_dim=1).transpose(1, 2).contiguous()
            value = sequence_head_all_to_all(group, value, scatter_dim=2, gather_dim=1).transpose(1, 2).contiguous()
            query = query.transpose(1, 2).contiguous()
            length = query.shape[2]
        padding_mask, sequence_ids = attention_mask if attention_mask is not None else (None, None)
        chunks = []
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            key_start = max(0, start - self.sliding_window + 1) if self.is_sliding else 0

            def attend(q, k, v, r, projection, start=start, key_start=key_start):
                tau = None
                if not self.is_sliding and self.config.log_scaling_n_floor is not None:
                    positions = torch.arange(q.shape[2], device=q.device) + start + 1
                    tau = 1 + self.config.log_scaling_alpha * torch.log(
                        (positions.float() / self.config.log_scaling_n_floor).clamp(min=1)
                    )
                    q = (q.float() * tau[None, None, :, None]).to(q.dtype)
                return relative_attention(
                    q,
                    k,
                    v,
                    r,
                    projection,
                    query_offset=start,
                    key_offset=key_start,
                    padding_mask=padding_mask,
                    sequence_ids=sequence_ids,
                    window=self.sliding_window,
                    scaling=self.scaling,
                    bias_scale=tau,
                )

            args = (
                query[:, :, start:end].contiguous(),
                key[:, :, key_start:end].contiguous() if self.is_sliding else key,
                value[:, :, key_start:end].contiguous() if self.is_sliding else value,
                relative[:, start:end].contiguous(),
                self.rel_logits_proj.proj,
            )
            if torch.is_grad_enabled():
                output = checkpoint(attend, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                output = attend(*args)
            chunks.append(output.transpose(1, 2))
        output = torch.cat(chunks, dim=1)
        if group is not None:
            output = sequence_head_all_to_all(group, output, scatter_dim=1, gather_dim=2)
        output = output.reshape(batch, hidden_states.shape[1], -1)
        return self.o_proj(output), None


class InklingFlexTextModel(NativeInklingTextModel):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        if past_key_values is not None or use_cache:
            raise ValueError("Inkling FlexAttention training requires use_cache=False")
        if attention_mask is not None and (not torch.is_tensor(attention_mask) or attention_mask.ndim != 2):
            raise ValueError("Inkling FlexAttention requires a two-dimensional padding mask")
        padding = attention_mask.bool() if attention_mask is not None else None
        group = getattr(self, "_sp_group", None)
        global_padding = gather_sequence(padding, group) if padding is not None else None
        global_positions = gather_sequence(position_ids, group) if position_ids is not None else None
        sequence_ids = None
        if global_positions is not None:
            resets = global_positions[:, 1:] != global_positions[:, :-1] + 1
            sequence_ids = torch.nn.functional.pad(resets.to(torch.int64).cumsum(-1), (1, 0))
            if sequence_ids.shape[0] == 1:
                size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
                sequence_ids = sequence_ids.expand(size, -1)
        masks = {
            "full_attention": (global_padding, sequence_ids),
            "sliding_attention": (global_padding, sequence_ids),
            "linear_attention": padding,
        }
        return super().forward(
            input_ids=input_ids,
            attention_mask=masks,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            **kwargs,
        )


def enable_flex_attention(model, *, query_chunk_size=2048):
    if Version(torch.__version__).release < (2, 13):
        raise ValueError("Inkling FlexAttention requires PyTorch 2.13+ for differentiable relative-position bias")
    model.config.inkling_attention_query_chunk_size = query_chunk_size
    model.model.__class__ = InklingFlexTextModel
    for layer in model.model.layers:
        layer.self_attn.__class__ = InklingFlexAttention
