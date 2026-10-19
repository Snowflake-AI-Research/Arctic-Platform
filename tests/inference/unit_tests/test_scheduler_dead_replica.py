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


def _worker(death: type[Exception] | None) -> MagicMock:
    async def generate(prompt, params):
        await asyncio.sleep(0.01)  # in flight together, so least-loaded routing spreads them across replicas
        if death is not None:
            raise death()
        return {"text": prompt}

    worker = MagicMock()
    worker.generate.remote = generate
    return worker


@pytest.mark.parametrize("death", DEATHS, ids=lambda d: d.__name__)
def test_requests_routed_to_a_dead_replica_finish_on_a_live_one(death):
    async def run():
        scheduler = Scheduler([_worker(None), _worker(death), _worker(None)])
        futures = [scheduler.submit(f"p{i}", {}) for i in range(30)]
        return scheduler, await asyncio.wait_for(asyncio.gather(*futures, return_exceptions=True), timeout=10)

    scheduler, results = asyncio.run(run())
    assert results == [{"text": f"p{i}"} for i in range(30)]
    assert [scheduler.is_worker_available(i) for i in range(3)] == [True, False, True]


@pytest.mark.parametrize("death", DEATHS, ids=lambda d: d.__name__)
def test_when_every_replica_died_the_error_reaches_the_caller(death):
    async def run():
        scheduler = Scheduler([_worker(death), _worker(death)])
        futures = [scheduler.submit(f"p{i}", {}) for i in range(4)]
        return await asyncio.wait_for(asyncio.gather(*futures, return_exceptions=True), timeout=10)

    assert all(isinstance(result, death) for result in asyncio.run(run()))


def test_a_request_that_kills_every_engine_it_reaches_takes_down_at_most_two_replicas():
    sent = []

    def poisoned(idx: int) -> MagicMock:
        async def generate(prompt, params):
            sent.append(idx)
            await asyncio.sleep(0.01)
            raise EngineDeadError()

        worker = MagicMock()
        worker.generate.remote = generate
        return worker

    async def run():
        scheduler = Scheduler([poisoned(i) for i in range(5)])
        result = await asyncio.wait_for(asyncio.gather(scheduler.submit("p", {}), return_exceptions=True), timeout=10)
        return scheduler, result

    scheduler, (result,) = asyncio.run(run())
    assert isinstance(result, EngineDeadError)
    assert len(sent) <= 2
    assert sum(not scheduler.is_worker_available(i) for i in range(5)) <= 2
