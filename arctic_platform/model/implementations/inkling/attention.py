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
from packaging.version import Version
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention
from torch.utils.checkpoint import checkpoint
from transformers.models.inkling.modeling_inkling import InklingAttention as NativeInklingAttention
from transformers.models.inkling.modeling_inkling import InklingTextModel as NativeInklingTextModel


@lru_cache(maxsize=32)
def _compiled_attention(geometry):
    def run(*args, **kwargs):
        return flex_attention(*args, **kwargs)

    # Separate Dynamo caches for dtype, head geometry, and mask structure.
    run.__code__ = run.__code__.replace(co_name=f"inkling_flex_{geometry}")
    return torch.compile(run, fullgraph=True, dynamic=True)


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
):
    """One query chunk; no token-by-token attention or bias matrix is allocated."""
    batch, heads, q_len, dim = query.shape
    kv_len = key.shape[2]
    extent = projection.shape[1]
    bank = (relative @ projection).transpose(1, 2).contiguous()
    if bias_scale is not None:
        bank = (bank.float() * bias_scale[None, None, :, None]).to(bank.dtype)
    q_offset = query.new_tensor(query_offset, dtype=torch.int64)
    k_offset = query.new_tensor(key_offset, dtype=torch.int64)

    def score_mod(score, b, h, qi, ki):
        distance = qi + q_offset - ki - k_offset
        bias = bank[b, h, qi, distance.clamp(0, extent - 1)]
        return score + torch.where((distance >= 0) & (distance < extent), bias, 0)

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

    block = 128
    q_blocks, kv_blocks = (q_len + block - 1) // block, (kv_len + block - 1) // block
    first_query = torch.arange(q_blocks, device=query.device) * block + q_offset
    last_query = torch.minimum(first_query + block - 1, q_offset + q_len - 1)
    first_key_block = torch.zeros_like(first_query)
    if window is not None:
        first_key_block = (first_query - window + 1 - k_offset).clamp(min=0) // block
    last_key_block = ((last_query - k_offset) // block).clamp(max=kv_blocks - 1)
    counts = (last_key_block - first_key_block + 1).clamp(min=0).to(torch.int32)
    indices = first_key_block[:, None] + torch.arange(kv_blocks, device=query.device)
    indices = indices.clamp(max=kv_blocks - 1).to(torch.int32)
    mask = BlockMask.from_kv_blocks(
        counts[None, None].expand(batch, 1, -1).contiguous(),
        indices[None, None].expand(batch, 1, -1, -1).contiguous(),
        BLOCK_SIZE=block,
        mask_mod=mask_mod,
        seq_lengths=(q_len, kv_len),
    )
    geometry = (query.dtype, heads, key.shape[1], dim, window, padding_mask is not None, sequence_ids is not None)
    return _compiled_attention(geometry)(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=mask,
        enable_gqa=heads != key.shape[1],
        scale=scaling if scaling is not None else 1 / dim,
        kernel_options={"BACKEND": "TRITON"},
    )


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
            return self.k_norm(key.view(batch, length, self.num_key_value_heads, self.head_dim))

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
        output = torch.cat(chunks, dim=1).reshape(batch, length, -1)
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
        sequence_ids = None
        if position_ids is not None:
            resets = position_ids[:, 1:] != position_ids[:, :-1] + 1
            sequence_ids = torch.nn.functional.pad(resets.to(torch.int64).cumsum(-1), (1, 0))
            if sequence_ids.shape[0] == 1:
                size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
                sequence_ids = sequence_ids.expand(size, -1)
        masks = {
            "full_attention": (padding, sequence_ids),
            "sliding_attention": (padding, sequence_ids),
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
