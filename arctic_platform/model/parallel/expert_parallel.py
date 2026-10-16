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
"""Expert-parallel checks and the sequence-parallel-1 process-group setup."""

from __future__ import annotations


def ep_group_name(ep_size: int) -> str:
    """DeepSpeed expert-group registry key for an expert-parallel degree."""
    return f"ep_size_{ep_size}"


def validate_expert_parallel(
    ep_size: int,
    world_size: int,
    *,
    zero_stage: int | None = None,
) -> None:
    """Reject an expert-parallel degree DeepSpeed cannot run.

    ``ep_size == 1`` is a no-op. Otherwise the degree must divide ``world_size``,
    and ZeRO stage 3 is rejected because it does not support expert-data-parallel
    groups.
    """
    if ep_size == 1:
        return
    if zero_stage == 3:
        raise ValueError(
            f"ep_size={ep_size} requires ZeRO stage 1 or 2; stage 3 does not support MoE expert-DP groups"
        )
    if world_size % ep_size != 0:
        raise ValueError(f"ep_size ({ep_size}) must divide world_size ({world_size})")


def setup_replicated_expert_groups(ep_size: int) -> None:
    """Register expert and expert-data-parallel groups when sequence parallelism is 1.

    ``ep_size == 1`` creates nothing. Sequence-parallel topologies are not created here.
    """
    if ep_size > 1:
        _create_expert_and_data_parallel(ep_size)


def expert_parallel_group(ep_size: int):
    """Return the DeepSpeed expert group registered for this degree."""
    return _expert_parallel_group(ep_group_name(ep_size))


def _create_expert_and_data_parallel(ep_size: int) -> None:
    import deepspeed.utils.groups as ds_groups

    ds_groups._create_expert_and_data_parallel(ep_size)


def split_expert_param_groups(param_groups: list) -> list:
    """Name optimizer groups and split MoE parameters onto the expert-data-parallel group.

    DeepSpeed only does this split when it builds the optimizer itself. A caller-supplied
    optimizer has to hand it groups that already separate expert parameters.
    """
    from deepspeed.moe.utils import configure_moe_param_groups

    named = []
    for index, group in enumerate(param_groups):
        group.setdefault("name", f"group_{index}")
        named.append(group)
    return configure_moe_param_groups(named)


def _expert_parallel_group(group_name: str):
    import deepspeed.utils.groups as ds_groups

    return ds_groups._get_expert_parallel_group(group_name)
