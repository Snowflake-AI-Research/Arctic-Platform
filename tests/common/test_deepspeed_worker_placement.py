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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Ray scheduling must not split node-local expert groups across hosts."""

import asyncio
from types import SimpleNamespace

import pytest

from arctic_platform.common import deepspeed_worker


class RemoteMethod:
    def __init__(self, function):
        self.function = function

    async def remote(self, *args):
        return self.function(*args)


class WorkerFactory:
    def __init__(self, hosts, config, *, fail_initialize=False, fail_rank=False):
        self.hosts = hosts
        self.config = config
        self.fail_initialize = fail_initialize
        self.fail_rank = fail_rank
        self.workers = []
        self.options_seen = []
        self.initialized = []
        self.destroyed = []

    def options(self, **options):
        self.options_seen.append(options)
        return self

    def remote(self, rank, world_size, port):
        worker = SimpleNamespace(rank=rank, host=self.hosts[rank], original_rank=rank, engine=None)

        def set_rank(new_rank):
            if self.fail_rank:
                raise RuntimeError("rank assignment failed")
            worker.rank = new_rank

        def initialize(master, config):
            self.initialized.append(worker)
            assert config is self.config
            assert master == self.hosts[0]
            expert_parallel = config.get("ds_worker_config", {}).get("ep_size", 1)
            ipc_group_size = min(expert_parallel, 8)
            group_start = worker.rank // ipc_group_size * ipc_group_size
            if expert_parallel > 1:
                group = [actor for actor in self.workers if group_start <= actor.rank < group_start + ipc_group_size]
                assert len({actor.host for actor in group}) == 1, "expert group attempts cross-host CUDA IPC"
            if self.fail_initialize:
                raise RuntimeError("initialization failed")

        worker.get_ip = RemoteMethod(lambda: worker.host)
        worker.set_rank = RemoteMethod(set_rank)
        worker.initialize = RemoteMethod(initialize)
        worker.destroy = RemoteMethod(lambda: self.destroyed.append(worker))
        self.workers.append(worker)
        return worker


def spawn(monkeypatch, hosts, ep_size=4, **kwargs):
    config = {"ds_worker_config": {"ep_size": ep_size}}
    factory = WorkerFactory(hosts, config, **kwargs)
    monkeypatch.setattr(deepspeed_worker, "DeepSpeedWorker", factory)
    return factory, deepspeed_worker.spawn_and_initialize_workers(
        len(hosts), 29500, config, lambda rank: {"num_gpus": 1, "bundle": rank}
    )


@pytest.mark.parametrize("ep_size", [4, 8, 16])
def test_interleaved_sixteen_rank_placement_keeps_expert_groups_node_local(monkeypatch, ep_size):
    hosts = ["head"] * 7 + ["peer", "peer", "head"] + ["peer"] * 6
    factory, pending = spawn(monkeypatch, hosts, ep_size=ep_size)
    workers = asyncio.run(pending)

    assert [worker.rank for worker in workers] == list(range(16))
    assert [worker.host for worker in workers] == ["head"] * 8 + ["peer"] * 8
    assert [worker.original_rank for worker in workers] == [0, 1, 2, 3, 4, 5, 6, 9, 7, 8, 10, 11, 12, 13, 14, 15]
    assert factory.options_seen == [{"num_gpus": 1, "bundle": rank} for rank in range(16)]
    assert not factory.destroyed


@pytest.mark.parametrize("hosts", [["head"] * 8, ["z-node"] * 8 + ["a-node"] * 8])
def test_contiguous_placement_preserves_actor_and_bundle_order(monkeypatch, hosts):
    factory, pending = spawn(monkeypatch, hosts)
    workers = asyncio.run(pending)

    assert workers == factory.workers
    assert [worker.original_rank for worker in workers] == list(range(len(hosts)))
    assert [worker.rank for worker in workers] == list(range(len(hosts)))


def test_dense_workers_do_not_require_node_local_blocks(monkeypatch):
    hosts = ["head", "peer"] * 3
    factory, pending = spawn(monkeypatch, hosts, ep_size=1)
    assert asyncio.run(pending) == factory.workers


def test_infeasible_node_local_expert_group_fails_before_initialization(monkeypatch):
    factory, pending = spawn(monkeypatch, ["head"] * 3 + ["peer"] * 5)
    with pytest.raises(ValueError, match="expert.*host|node.local"):
        asyncio.run(pending)
    assert not factory.initialized
    assert len(factory.destroyed) == 8


@pytest.mark.parametrize("failure", ["initialize", "rank"])
def test_spawn_failure_destroys_every_actor(monkeypatch, failure):
    hosts = ["head", "peer"] * 4
    factory, pending = spawn(monkeypatch, hosts, fail_initialize=failure == "initialize", fail_rank=failure == "rank")
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(pending)
    assert len(factory.destroyed) == len(hosts)


def test_actor_rank_assignment_rejects_initialized_engine():
    cls = deepspeed_worker.DeepSpeedWorker.__ray_metadata__.modified_class
    worker = object.__new__(cls)
    worker.engine = None
    worker.rank = 5
    worker.set_rank(2)
    assert worker.rank == 2
    worker.engine = object()
    with pytest.raises(RuntimeError, match="initializ"):
        worker.set_rank(1)
    assert worker.rank == 2
