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
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# CPU-only environments: stub the runtime imports the scheduler module needs.
try:
    import ray
except ModuleNotFoundError:
    ray = types.ModuleType("ray")
    ray.actor = SimpleNamespace(ActorHandle=object)
    ray.exceptions = SimpleNamespace(ActorDiedError=type("ActorDiedError", (Exception,), {}))
    sys.modules["ray"] = ray
    for name in ("vllm", "vllm.config", "vllm.v1", "vllm.v1.metrics", "vllm.v1.metrics.loggers", "vllm.v1.metrics.stats"):
        sys.modules.setdefault(name, MagicMock())
    sys.modules.setdefault(
        "vllm.v1.engine.exceptions",
        SimpleNamespace(EngineDeadError=type("EngineDeadError", (Exception,), {})),
    )
    import arctic_platform.inference

    server = types.ModuleType("arctic_platform.inference.server")
    server.__path__ = [str(Path(arctic_platform.inference.__file__).parent / "server")]
    sys.modules["arctic_platform.inference.server"] = server

from vllm.v1.engine.exceptions import EngineDeadError

from arctic_platform.inference.server.scheduler import Scheduler

DEATHS = [ray.exceptions.ActorDiedError, EngineDeadError]


def _remote(generate):
    """An actor method's ``.remote`` whose calls return ObjectRef stand-ins."""
    return lambda *args: SimpleNamespace(as_future=lambda: asyncio.ensure_future(generate(*args)))


def _worker(death: type[Exception] | None) -> MagicMock:
    async def generate(prompt, params):
        await asyncio.sleep(0.01)
        if death is not None:
            raise death()
        return {"text": prompt}

    worker = MagicMock()
    worker.generate.remote = _remote(generate)
    return worker


@pytest.mark.parametrize("death", DEATHS, ids=lambda d: d.__name__)
def test_requests_routed_to_a_dead_replica_finish_on_a_live_one(death):
    async def run():
        scheduler = Scheduler([_worker(None), _worker(death), _worker(None)], capacity=64)
        scheduler._keys.update(group0=0, group1=1, group2=2)  # each replica holds one sticky key
        futures = [scheduler.submit(f"p{i}", {}, routing_key=f"group{i % 3}") for i in range(30)]
        results = await asyncio.wait_for(asyncio.gather(*futures, return_exceptions=True), timeout=10)
        assert await scheduler.submit("next", {}, routing_key="group1") == {"text": "next"}
        assert all(ws.active_requests == 0 for ws in scheduler._workers)
        assert not any(ws.inflight for ws in scheduler._workers)
        return scheduler, results

    scheduler, results = asyncio.run(run())
    assert results == [{"text": f"p{i}"} for i in range(30)]
    assert [scheduler.is_worker_available(i) for i in range(3)] == [True, False, True]


@pytest.mark.parametrize("death", DEATHS, ids=lambda d: d.__name__)
def test_when_every_replica_died_the_error_reaches_the_caller_and_new_requests_wait_for_a_replacement(death):
    async def run():
        scheduler = Scheduler([_worker(death), _worker(death)], capacity=64)
        futures = [scheduler.submit(f"p{i}", {}) for i in range(4)]
        results = await asyncio.wait_for(asyncio.gather(*futures, return_exceptions=True), timeout=1)
        later = scheduler.submit("later", {})
        await asyncio.sleep(0.05)
        assert not later.done()
        scheduler.update_worker_handle(0, _worker(None))
        scheduler.mark_worker_available(0)
        assert await asyncio.wait_for(later, timeout=1) == {"text": "later"}
        return results

    results = asyncio.run(run())
    assert all(isinstance(result, death) for result in results)


def test_a_request_that_kills_every_engine_it_reaches_takes_down_at_most_two_replicas():
    sent = []

    def poisoned(idx: int) -> MagicMock:
        async def generate(prompt, params):
            sent.append(idx)
            await asyncio.sleep(0.01)
            raise EngineDeadError()

        worker = MagicMock()
        worker.generate.remote = _remote(generate)
        return worker

    async def run():
        scheduler = Scheduler([poisoned(i) for i in range(5)], capacity=64)
        result = await asyncio.wait_for(asyncio.gather(scheduler.submit("p", {}), return_exceptions=True), timeout=10)
        return scheduler, result

    scheduler, (result,) = asyncio.run(run())
    assert isinstance(result, EngineDeadError)
    assert len(sent) <= 2
    assert sum(not scheduler.is_worker_available(i) for i in range(5)) <= 2


def test_late_death_does_not_quarantine_a_replacement():
    async def run():
        started, replaced = asyncio.Event(), asyncio.Event()

        async def generate(prompt, params):
            started.set()
            await replaced.wait()
            raise EngineDeadError()

        old = _worker(None)
        old.generate.remote = _remote(generate)
        scheduler = Scheduler([old], capacity=64)
        future = scheduler.submit("p", {})
        await started.wait()
        scheduler.mark_worker_unavailable(0)
        scheduler.update_worker_handle(0, _worker(None))
        scheduler.mark_worker_available(0)
        replaced.set()
        assert await asyncio.wait_for(future, timeout=1) == {"text": "p"}
        assert scheduler.is_worker_available(0)

    asyncio.run(run())
