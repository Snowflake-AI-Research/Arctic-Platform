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

"""Native SP and EP groups must cover every rank in each logical replica."""

from types import SimpleNamespace

import pytest
from deepspeed.runtime.sequence_parallel import parallel_state_sp
from deepspeed.utils import groups

from arctic_platform.common import deepspeed_worker


class Transport:
    def __init__(self, world, rank):
        self.world = world
        self.rank = rank
        self.groups = []

    def is_initialized(self):
        return True

    def get_world_size(self, group=None):
        return self.world if group is None else len(group)

    def get_rank(self, group=None):
        return self.rank if group is None else group.index(self.rank)

    def new_group(self, ranks):
        group = tuple(ranks)
        self.groups.append(group)
        return group


def topology(monkeypatch, world, rank, sp, ep):
    transport = Transport(world, rank)
    monkeypatch.setattr(deepspeed_worker, "dist", transport)
    monkeypatch.setattr(parallel_state_sp, "dist", transport)
    monkeypatch.setattr(parallel_state_sp, "_SEQUENCE_PARALLEL_GROUP", None)
    monkeypatch.setattr(parallel_state_sp, "_SEQUENCE_DATA_PARALLEL_GROUP", None)
    for name in (
        "_EXPERT_PARALLEL_GROUP",
        "_EXPERT_PARALLEL_GROUP_RANKS",
        "_EXPERT_DATA_PARALLEL_GROUP",
        "_EXPERT_DATA_PARALLEL_GROUP_RANKS",
    ):
        monkeypatch.setattr(groups, name, {})
    monkeypatch.setattr(groups, "mpu", None)
    spec = SimpleNamespace(parallelism=SimpleNamespace(sequence_parallel=sp, expert_parallel=ep))
    return transport, spec


@pytest.mark.parametrize("world", [8, 16, 32])
def test_sp8_expert_groups_cover_every_rank(monkeypatch, world):
    for rank in range(world):
        with monkeypatch.context() as context:
            transport, spec = topology(context, world, rank, sp=8, ep=4)
            result = deepspeed_worker._setup_model_parallel_groups(spec)

            assert result["sp_group"] == tuple(range(rank // 8 * 8, rank // 8 * 8 + 8))
            assert result["ep_group"] == tuple(range(rank // 4 * 4, rank // 4 * 4 + 4))
            assert groups._EXPERT_DATA_PARALLEL_GROUP["ep_size_4"] == tuple(range(rank % 4, world, 4))
            assert parallel_state_sp.get_sequence_data_parallel_group() == tuple(range(world))
            assert parallel_state_sp.get_sequence_parallel_rank() == rank % 8
            assert parallel_state_sp.get_sequence_parallel_world_size() == 8
            assert result["mpu"] is parallel_state_sp and groups.mpu is parallel_state_sp
            assert len(transport.groups) == world // 8 + 1 + world // 4 + 4


@pytest.mark.parametrize("world,sp,ep", [(12, 8, 4), (8, 16, 4), (8, 8, 3)])
def test_invalid_topology_rejected_before_collectives(monkeypatch, world, sp, ep):
    transport, spec = topology(monkeypatch, world, rank=0, sp=sp, ep=ep)
    with pytest.raises(ValueError):
        deepspeed_worker._setup_model_parallel_groups(spec)
    assert transport.groups == []


def test_dense_data_parallel_does_not_create_groups(monkeypatch):
    transport, spec = topology(monkeypatch, world=16, rank=0, sp=1, ep=1)
    assert deepspeed_worker._setup_model_parallel_groups(spec) == {
        "ep_group": None,
        "sp_group": None,
        "mpu": None,
    }
    assert transport.groups == []
