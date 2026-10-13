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
"""Sequence-parallel degree checks that do not depend on a training service."""

from __future__ import annotations

from arctic_platform.model.loader import ModelParallelismMetadata


def validate_sequence_parallel(
    sp_size: int,
    world_size: int,
    metadata: ModelParallelismMetadata,
    *,
    skip_head_exchange: bool = False,
) -> None:
    """Reject a sequence-parallel degree the head exchanges cannot run.

    ``sp_size == 1`` is a no-op. Otherwise ``sp_size`` must divide ``world_size``.
    Ulysses and head-parallel GatedDeltaNet checks run unless the caller is using
    sequence-only context parallelism.
    """
    if sp_size == 1:
        return
    if world_size % sp_size != 0:
        raise ValueError(f"world_size ({world_size}) must be divisible by sp_size ({sp_size})")
    if skip_head_exchange:
        return
    if metadata.num_attention_heads % sp_size != 0:
        raise ValueError(
            f"sp_size ({sp_size}) must divide num_attention_heads "
            f"({metadata.num_attention_heads}): Ulysses SP scatters the query heads across "
            "SP ranks, so the head count must be divisible by sp (max sp = "
            f"num_attention_heads = {metadata.num_attention_heads})."
        )
    num_key_value_heads = metadata.num_key_value_heads
    if num_key_value_heads is not None and not (
        num_key_value_heads % sp_size == 0 or sp_size % num_key_value_heads == 0
    ):
        raise ValueError(
            f"sp_size ({sp_size}) and num_key_value_heads ({num_key_value_heads}) must divide one another: "
            "Ulysses must either scatter KV heads evenly across SP ranks or replicate each KV head to an "
            "integer number of SP ranks."
        )
    if not metadata.has_linear_attention:
        return
    num_gdn_key_heads = metadata.linear_num_key_heads
    num_gdn_value_heads = metadata.linear_num_value_heads
    if num_gdn_key_heads is None or num_gdn_value_heads is None:
        raise ValueError(
            "linear-attention model config must define positive linear_num_key_heads and linear_num_value_heads"
        )
    if num_gdn_key_heads % sp_size:
        raise ValueError(
            f"sp_size ({sp_size}) must divide linear_num_key_heads "
            f"({num_gdn_key_heads}) for head-parallel GatedDeltaNet"
        )
    if num_gdn_value_heads % sp_size:
        raise ValueError(
            f"sp_size ({sp_size}) must divide linear_num_value_heads "
            f"({num_gdn_value_heads}) for head-parallel GatedDeltaNet"
        )
    if num_gdn_value_heads % num_gdn_key_heads:
        raise ValueError(
            f"linear_num_value_heads ({num_gdn_value_heads}) must be divisible "
            f"by linear_num_key_heads ({num_gdn_key_heads}) for GatedDeltaNet GVA"
        )
