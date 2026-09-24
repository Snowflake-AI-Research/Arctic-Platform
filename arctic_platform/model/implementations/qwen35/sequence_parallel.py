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
"""Sequence parallelism (SP) for the carved-out Qwen3.5 MoE model.

Wires SP onto the model's native varlen flash-attention layout (FA2/FA3/FA4; not the ArcticTraining ALST
monkey-patch, which this hand-rolled model bypasses):

* Full-attention layers get a Ulysses all-to-all wrapper: project q/k/v on the local sequence shard, all-to-
  all to swap the sequence shard for a head shard, run varlen flash-attn over the full sequence, all-to-all
  back.
* Linear-attention (GDN) layers use one all-to-all to exchange native Q/K/V and gate projections from
  local-sequence/all-heads to full-sequence/local-heads, then run both convolution and recurrence there.

Both share one SP process group (kept orthogonal to EP by ``jobs/gpu/sp.setup_parallel_groups``).
The varlen path needs the global ``cu_seqlens`` after the gather, so the backbone forward is wrapped to
all-gather ``position_ids`` and recompute the boundaries. Assumes a single packed row (``B == 1``).
"""

from __future__ import annotations

import types

import torch
import torch.distributed as dist
from torch import nn

from arctic_platform.model.implementations.gpu.packing import cu_seqlens_from_position_ids
from arctic_platform.model.implementations.gpu.sp.collectives import sequence_head_all_to_all

from arctic_platform.model.implementations.moe.logging_utils import get_logger
from arctic_platform.model.implementations.moe.vlm import get_language_model

_FA_IMPLS = ("flash_attention_2", "flash_attention_3", "flash_attention_4")


def _ulysses_attn_forward(self, hidden_states, position_embeddings, cu_seqlens=None, max_seqlen=None):
    """Drop-in replacement for ``Qwen3_5MoeGatedFlashAttention.forward`` that runs attention over the full
    (SP-gathered) sequence with a head shard."""
    group = self._sp_group
    if hidden_states.size(0) != 1:
        raise NotImplementedError(
            f"Ulysses varlen path assumes a single packed row (B==1), got "
            f"B={hidden_states.size(0)}"
        )

    # Local projections; RoPE uses the shard's true positions.
    query_states, key_states, value_states, gate = self.attn_projections(
        hidden_states, position_embeddings
    )

    # GQA KV-head replication for sp > num_kv_heads: the all-to-all scatters KV heads across sp ranks, so it
    # needs sp % num_kv_heads == 0. num_kv_heads is read from the tensor at runtime (e.g. 2 for the
    # Qwen3.6-35B-A3B model used here); replicate each KV head sp/num_kv_heads times so rank r receives the
    # KV head its q-head shard attends to (the kv_replication_factor trick; autograd sums grads back).
    sp_world_size = dist.get_world_size(group)
    num_kv_heads = key_states.size(2)
    if sp_world_size > num_kv_heads:
        if sp_world_size % num_kv_heads != 0:
            raise NotImplementedError(
                f"Ulysses KV replication needs sp_world_size ({sp_world_size}) to be a "
                f"multiple of num_kv_heads ({num_kv_heads})"
            )
        kv_replication = sp_world_size // num_kv_heads
        key_states = key_states.repeat_interleave(kv_replication, dim=2)
        value_states = value_states.repeat_interleave(kv_replication, dim=2)

    # [B, S_local, n_heads, hd] -> [B, S_full, n_heads/sp, hd]
    query_states = sequence_head_all_to_all(
        group, query_states, scatter_dim=2, gather_dim=1
    )
    key_states = sequence_head_all_to_all(
        group, key_states, scatter_dim=2, gather_dim=1
    )
    value_states = sequence_head_all_to_all(
        group, value_states, scatter_dim=2, gather_dim=1
    )

    # Global boundaries stashed by the backbone wrapper; fall back to the passed-in locals.
    full_cu_seqlens = getattr(self, "_sp_cu_seqlens", None)
    full_max_seqlen = getattr(self, "_sp_max_seqlen", None)
    if full_cu_seqlens is None:
        full_cu_seqlens = cu_seqlens
        full_max_seqlen = max_seqlen

    # varlen flash-attn (FA2/FA3/FA4) over the full sequence -> [S_full, n_heads/sp, hd]
    attn_output = self._attention_core(
        query_states,
        key_states,
        value_states,
        cu_seqlens=full_cu_seqlens,
        max_seqlen=full_max_seqlen,
    )

    # All-to-all back to [B=1, S_local, n_heads, hd], then drop batch for output_proj's 3-D path.
    attn_output = attn_output.unsqueeze(0)
    attn_output = sequence_head_all_to_all(
        group, attn_output, scatter_dim=1, gather_dim=2
    )
    attn_output = attn_output.squeeze(0)

    return self.output_proj(attn_output, gate), None


def _make_backbone_sp_forward(original_forward, sp_group, full_attn_modules, linear_attn_modules=()):
    """Wrap the backbone forward to rebuild the global ``cu_seqlens`` from SP-gathered ``position_ids`` and
    stash it on the attention modules before the layer loop.

    Full attention needs it for the varlen kernel after the all-to-all gather. GatedDeltaNet needs it for a
    different reason: both the full recurrence and the convolution left-pad must reset at each packed sequence
    boundary, and a shard-local boundary list cannot say whether a sequence continues on the next rank."""
    sp_world_size = dist.get_world_size(sp_group)

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, routed_experts=None):
        if (
            position_ids is not None
            and self.config._attn_implementation in _FA_IMPLS
            and (full_attn_modules or linear_attn_modules)
        ):
            gathered_position_ids = [torch.empty_like(position_ids) for _ in range(sp_world_size)]
            dist.all_gather(gathered_position_ids, position_ids.contiguous(), group=sp_group)
            # SP shards are contiguous in sp_rank order, so concatenation rebuilds global order.
            global_position_ids = torch.cat(gathered_position_ids, dim=1)
            cu_seqlens = cu_seqlens_from_position_ids(global_position_ids)
            max_seqlen = int(cu_seqlens.diff().max().item())
            for module in full_attn_modules:
                module._sp_cu_seqlens = cu_seqlens
                module._sp_max_seqlen = max_seqlen
            for module in linear_attn_modules:
                module._dss_sp_global_cu_seqlens = cu_seqlens

        return original_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            routed_experts=routed_experts,
        )

    return forward


def apply_sequence_parallelism(model: nn.Module, sp_size: int, sp_group) -> None:
    """Install Ulysses SP on ``model`` in place. Must run before activation-checkpoint wrapping (so
    ``self_attn`` and ``linear_attn`` submodules are still directly addressable)."""
    if sp_size == 1 or sp_group is None:
        return

    logger = get_logger()
    backbone = get_language_model(model)
    sp_world_size = dist.get_world_size(sp_group)

    full_attn_modules = []
    linear_attn_modules = []
    for layer in backbone.layers:
        layer_type = getattr(layer, "layer_type", None)
        if layer_type == "linear_attention" and hasattr(layer, "linear_attn"):
            linear_attn = layer.linear_attn
            linear_attn.cp_group = sp_group
            linear_attn_modules.append(linear_attn)
        elif layer_type == "full_attention" and hasattr(layer, "self_attn"):
            self_attn = layer.self_attn
            self_attn._sp_group = sp_group
            self_attn._sp_world_size = sp_world_size
            self_attn.forward = types.MethodType(_ulysses_attn_forward, self_attn)
            full_attn_modules.append(self_attn)

    backbone.forward = types.MethodType(
        _make_backbone_sp_forward(backbone.forward, sp_group, full_attn_modules, linear_attn_modules),
        backbone,
    )

    logger.info(
        "Applied Ulysses SP (sp_size=%d): wrapped %d full-attention layers, "
        "%d linear-attention (full temporal-core head-parallel GDN) layers",
        sp_world_size,
        len(full_attn_modules),
        len(linear_attn_modules),
    )
