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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import ray
from vllm.v1.engine.exceptions import EngineDeadError

from arctic_platform.inference.server.scheduler import Scheduler

DEATHS = [ray.exceptions.ActorDiedError, EngineDeadError]
pytestmark = pytest.mark.asyncio


def _remote(generate):
    return lambda *args: SimpleNamespace(as_future=lambda: asyncio.ensure_future(generate(*args)))


def _worker(death: type[Exception] | None = None) -> MagicMock:
    async def generate(prompt, params):
        if death is not None:
            raise death()
        return {"text": prompt}

    worker = MagicMock()
    worker.generate.remote.side_effect = _remote(generate)
    return worker


@pytest.mark.parametrize("death", DEATHS, ids=lambda d: d.__name__)
async def test_requests_routed_to_a_dead_replica_finish_on_a_live_one(death):
    scheduler = Scheduler([_worker(death), _worker()], capacity=1)
    scheduler._keys["group"] = 0
    futures = [scheduler.submit(f"p{i}", {}, routing_key="group") for i in range(4)]
    assert len(scheduler._workers[0].waiting) == 2
    assert await asyncio.wait_for(asyncio.gather(*futures), timeout=1) == [{"text": f"p{i}"} for i in range(4)]
    assert await scheduler.submit("next", {}, routing_key="group") == {"text": "next"}
    assert [scheduler.is_worker_available(i) for i in range(2)] == [False, True]
    assert all(ws.active_requests == 0 and not ws.inflight for ws in scheduler._workers)


@pytest.mark.parametrize("death", DEATHS, ids=lambda d: d.__name__)
async def test_all_dead_replicas_fail_inflight_requests_and_waiting_requests_use_a_replacement(death):
    scheduler = Scheduler([_worker(death), _worker(death)], capacity=64)
    futures = [scheduler.submit(f"p{i}", {}) for i in range(4)]
    results = await asyncio.wait_for(asyncio.gather(*futures, return_exceptions=True), timeout=1)
    assert all(isinstance(result, death) for result in results)
    later = scheduler.submit("later", {})
    assert not later.done()
    scheduler.update_worker_handle(0, _worker())
    scheduler.mark_worker_available(0)
    assert await asyncio.wait_for(later, timeout=1) == {"text": "later"}


async def test_a_request_that_kills_every_engine_is_rerouted_only_once():
    workers = [_worker(EngineDeadError) for _ in range(3)]
    scheduler = Scheduler(workers, capacity=64)
    with pytest.raises(EngineDeadError):
        await asyncio.wait_for(scheduler.submit("p", {}), timeout=1)
    assert sum(worker.generate.remote.call_count for worker in workers) == 2
    assert sum(ws.quarantined for ws in scheduler._workers) == 2


async def test_late_death_does_not_quarantine_a_replacement():
    started, replaced = asyncio.Event(), asyncio.Event()

    async def generate(prompt, params):
        started.set()
        await replaced.wait()
        raise EngineDeadError()

    old = _worker()
    old.generate.remote.side_effect = _remote(generate)
    scheduler = Scheduler([old], capacity=64)
    future = scheduler.submit("p", {})
    await started.wait()
    scheduler.mark_worker_unavailable(0)
    scheduler.update_worker_handle(0, _worker())
    scheduler.mark_worker_available(0)
    replaced.set()
    assert await asyncio.wait_for(future, timeout=1) == {"text": "p"}
    assert scheduler.is_worker_available(0)
