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

"""A replica whose actor or engine died is taken out of rotation; its requests finish on the live replicas."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import ray
from vllm.v1.engine.exceptions import EngineDeadError

from arctic_platform.inference.server.replica_pool import ReplicaPool
from arctic_platform.inference.server.scheduler import Scheduler
from arctic_platform.inference.server.testing import RemoteMock
from arctic_platform.inference.server.worker import InferenceWorker, WorkerLifecycleState

DEATHS = [ray.exceptions.ActorDiedError, EngineDeadError]
pytestmark = pytest.mark.asyncio


def _worker(death: type[Exception] | None = None) -> MagicMock:
    worker = MagicMock()
    worker.generate.remote = RemoteMock(side_effect=death or (lambda prompt, params: {"text": prompt}))
    return worker


@pytest.mark.parametrize("death", DEATHS, ids=lambda d: d.__name__)
async def test_requests_routed_to_a_dead_replica_finish_on_a_live_one(death):
    scheduler = Scheduler([_worker(death), _worker()], capacity=1)
    scheduler._keys["group"] = 0
    futures = [scheduler.submit(f"p{i}", {}, routing_key="group") for i in range(4)]
    assert len(scheduler._workers[0].waiting) == 2
    assert await asyncio.wait_for(asyncio.gather(*futures), timeout=1) == [{"text": f"p{i}"} for i in range(4)]
    assert await asyncio.wait_for(scheduler.submit("next", {}, routing_key="group"), 1) == {"text": "next"}
    assert [scheduler.is_worker_available(i) for i in range(2)] == [False, True]
    assert all(ws.active_requests == 0 and not ws.inflight for ws in scheduler._workers)
    scheduler.mark_worker_unavailable(1)
    queued = scheduler.submit("during scale-down", {})
    assert not queued.done()
    scheduler.remove_last_worker()
    with pytest.raises(death):
        queued.result()


@pytest.mark.parametrize("death", DEATHS, ids=lambda d: d.__name__)
async def test_dead_pool_fails_current_and_later_requests(death):
    scheduler = Scheduler([_worker(death), _worker(death)], capacity=64)
    futures = [scheduler.submit(f"p{i}", {}) for i in range(4)]
    results = await asyncio.wait_for(asyncio.gather(*futures, return_exceptions=True), timeout=1)
    assert all(isinstance(result, death) for result in results)
    assert not any(scheduler.is_worker_available(i) for i in range(2))
    with pytest.raises(death):
        scheduler.submit("later", {}).result()


async def test_a_request_that_kills_every_engine_is_rerouted_only_once():
    workers = [_worker(EngineDeadError) for _ in range(3)]
    scheduler = Scheduler(workers, capacity=64)
    with pytest.raises(EngineDeadError):
        await asyncio.wait_for(scheduler.submit("p", {}), timeout=1)
    assert sum(worker.generate.remote.call_count for worker in workers) == 2
    assert sum(ws.quarantined for ws in scheduler._workers) == 2


async def test_late_death_does_not_quarantine_a_replacement():
    scheduler = Scheduler([_worker(EngineDeadError)], capacity=64)
    future = scheduler.submit("p", {})
    # Replace the handle before the old RPC's completion callback can run.
    scheduler.add_worker(_worker())
    scheduler.mark_worker_unavailable(0)
    scheduler.update_worker_handle(0, _worker())
    scheduler.mark_worker_available(0)
    assert await asyncio.wait_for(future, timeout=1) == {"text": "p"}
    assert scheduler.is_worker_available(0)


async def test_health_monitor_removes_a_dead_engine_before_generate(monkeypatch):
    replica = InferenceWorker.__ray_metadata__.modified_class()
    replica.state = WorkerLifecycleState.READY
    replica.llm = MagicMock()
    replica.llm.check_health = AsyncMock(side_effect=EngineDeadError)
    dead, live = _worker(), _worker()
    pool = ReplicaPool()
    pool._workers = [dead, live]
    pool._scheduler = Scheduler(pool._workers, capacity=1)

    def finish_poll():
        pool._stop_monitoring = True
        return True

    dead.is_healthy.remote = RemoteMock(side_effect=replica.is_healthy)
    live.is_healthy.remote = RemoteMock(side_effect=finish_poll)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    await asyncio.wait_for(pool._monitor_health(), 1)
    pool._scheduler._keys["group"] = 0
    assert await pool.generate([1], {}, routing_key="group") == [{"text": [1]}]
    dead.generate.remote.assert_not_called()
    pool._scheduler.remove_last_worker()
    with pytest.raises(EngineDeadError):
        await asyncio.wait_for(pool.generate([1], {}), 1)
