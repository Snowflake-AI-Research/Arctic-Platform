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

"""Ulysses attention and causal convolution halos for Inkling text training."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from transformers.models.inkling.modeling_inkling import InklingShortConvolution

from arctic_platform.model.implementations.gpu.sp.collectives import gather_sequence
from arctic_platform.model.implementations.inkling.attention import InklingFlexTextModel


class InklingSequenceParallelConvolution(InklingShortConvolution):
    def forward(self, hidden_states, past_key_values=None, conv_mask=None, **kwargs):
        if past_key_values is not None:
            raise ValueError("Inkling sequence parallelism requires use_cache=False")
        width = self.conv_kernel_size - 1
        if not width:
            return super().forward(hidden_states, conv_mask=conv_mask, **kwargs)
        group = self._sp_group
        rank = dist.get_rank(group)
        halo_width = min(width, hidden_states.shape[1])
        tail = hidden_states[:, -halo_width:].float()
        tails = gather_sequence(tail, group)
        end = rank * halo_width
        previous = tails[:, max(0, end - width) : end]
        # Retain a zero dependency so rank zero joins the gather backward.
        zeros = tails[:, :1] * 0
        previous = torch.cat((zeros.expand(-1, width - previous.shape[1], -1), previous), dim=1)
        extended = torch.cat((previous, hidden_states.float()), dim=1)
        mask = None
        if conv_mask is not None:
            masks = gather_sequence(conv_mask[:, -halo_width:], group)
            previous_mask = masks[:, max(0, end - width) : end]
            previous_mask = torch.nn.functional.pad(previous_mask, (width - previous_mask.shape[1], 0))
            mask = torch.cat((previous_mask, conv_mask), dim=1)
        output = super().forward(extended, conv_mask=mask, **kwargs)
        return output[:, width:].to(hidden_states.dtype)


def apply_sequence_parallelism(model, sp_size, sp_group):
    if sp_size == 1:
        return
    if sp_group is None or dist.get_world_size(sp_group) != sp_size:
        raise ValueError("Inkling SP group size must match sp_size")
    backbone = model.model
    if not isinstance(backbone, InklingFlexTextModel):
        raise ValueError("Inkling sequence parallelism requires flex_attention")
    for layer in backbone.layers:
        attention = layer.self_attn
        heads, kv_heads = attention.num_heads, attention.num_key_value_heads
        if heads % sp_size or not (kv_heads % sp_size == 0 or sp_size % kv_heads == 0):
            raise ValueError(
                "Inkling SP must divide both attention head counts; SP and KV heads must divide one another"
            )
    # Finish asynchronous expert backward before entering SP collectives.
    os.environ.setdefault("ARCTIC_UCCLEP_BACKWARD_BARRIER", "1")
    backbone._sp_group = sp_group
    for layer in backbone.layers:
        layer.self_attn._sp_group = sp_group
        for conv in (layer.self_attn.k_sconv, layer.self_attn.v_sconv, layer.attn_sconv, layer.mlp_sconv):
            conv.__class__ = InklingSequenceParallelConvolution
            conv._sp_group = sp_group
