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
"""Sequence-parallel rank layouts and the sequence-parallel-1 group entry point."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import arctic_platform.model.parallel.groups as parallel_groups
from arctic_platform.model.parallel.groups import SPGroups
from arctic_platform.model.parallel.groups import setup_parallel_groups
from arctic_platform.model.parallel.topology import orthogonal_edp_groups
from arctic_platform.model.parallel.topology import resolve_ep_group_topology


def test_spgroups_unpacks_positionally():
    groups = SPGroups("mpu", "group", 8, 3)
    assert groups == ("mpu", "group", 8, 3)
    assert groups.sp_world_size == 8 and groups.sp_rank == 3


def test_setup_parallel_groups_sp1_registers_replicated_experts(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "arctic_platform.model.parallel.groups.setup_replicated_expert_groups",
        lambda ep_size: calls.append(ep_size),
    )
    assert setup_parallel_groups(1, 4) == SPGroups(None, None, 1, 0)
    assert calls == [4]


def test_setup_parallel_groups_above_one_requires_gpus_per_node():
    with pytest.raises(ValueError, match="gpus_per_node"):
        setup_parallel_groups(2, 1, world_size=4)


@pytest.mark.parametrize(
    "layout,world,sp,ep,gpus,expected",
    [
        ("auto", 32, 8, 2, 8, "node-local"),
        ("auto", 32, 32, 8, 8, "decoupled"),
        ("auto", 32, 2, 16, 8, "host-complete"),
        ("auto", 8, 4, 2, 8, "orthogonal"),
        ("packed", 8, 4, 2, 8, "node-local"),
        ("mesh", 8, 4, 2, 8, "orthogonal"),
    ],
)
def test_resolve_ep_group_topology(layout, world, sp, ep, gpus, expected):
    assert resolve_ep_group_topology(world, sp, ep, gpus, layout) == expected


def test_orthogonal_expert_data_parallel_groups_span_both_axes():
    assert orthogonal_edp_groups(8, 2, 2) == [[0, 1, 4, 5], [2, 3, 6, 7]]
    assert orthogonal_edp_groups(8, 2, 4) == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_orthogonal_setup_registers_every_replica(monkeypatch):
    world_size, sp_size, ep_size, rank = 8, 2, 2, 0
    expert_parallel_groups = []
    for offset in range(sp_size):
        dp_ranks = list(range(offset, world_size, sp_size))
        expert_parallel_groups.extend(dp_ranks[start : start + ep_size] for start in range(0, len(dp_ranks), ep_size))
    positions = {peer: position for group in expert_parallel_groups for position, peer in enumerate(group)}
    ds_groups = SimpleNamespace(
        _EXPERT_PARALLEL_GROUP={},
        _EXPERT_PARALLEL_GROUP_RANKS={},
        _EXPERT_DATA_PARALLEL_GROUP={},
        _EXPERT_DATA_PARALLEL_GROUP_RANKS={},
    )

    def new_group(ranks):
        return SimpleNamespace(ranks=list(ranks))

    def create_expert_groups(degree, mpu):
        name = f"ep_size_{degree}"
        for ranks in expert_parallel_groups:
            if rank in ranks:
                ds_groups._EXPERT_PARALLEL_GROUP_RANKS[name] = list(ranks)
        ds_groups._EXPERT_DATA_PARALLEL_GROUP_RANKS[name] = [rank]

    ds_groups._create_expert_data_and_model_parallel = create_expert_groups
    monkeypatch.setattr(
        parallel_groups,
        "dist",
        SimpleNamespace(
            get_rank=lambda: rank,
            get_world_size=lambda: world_size,
            new_group=new_group,
            all_gather_object=lambda output, obj: output.__setitem__(
                slice(None), [positions[peer] for peer in range(world_size)]
            ),
        ),
    )
    monkeypatch.setattr(parallel_groups, "ds_groups", ds_groups)
    parallel_groups.setup_orthogonal_expert_groups(
        SimpleNamespace(get_sequence_parallel_world_size=lambda: sp_size),
        ep_size,
    )
    assert ds_groups._EXPERT_DATA_PARALLEL_GROUP_RANKS["ep_size_2"] == [0, 1, 4, 5]
