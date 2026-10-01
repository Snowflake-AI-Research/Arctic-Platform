import asyncio
import logging
import time
from types import SimpleNamespace

import pytest

from arctic_inference.server import replica_pool as replica_pool_mod
from arctic_inference.server.replica_pool import ReplicaPool


class _RemoteMethod:
    def __init__(self, func):
        self._func = func

    def remote(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class _FakeActor:
    def __init__(self, index: int, fail_init: bool) -> None:
        self.index = index
        self.initialize = _RemoteMethod(self._initialize)
        self.shutdown = _RemoteMethod(self._shutdown)
        self.shutdown_called = False
        self._fail_init = fail_init

    async def _initialize(self, *args, **kwargs):
        if self._fail_init:
            raise RuntimeError("worker init failed")

    def _shutdown(self):
        self.shutdown_called = True
        return f"shutdown-{self.index}"


class _FakeWorkerClass:
    def __init__(self) -> None:
        self.actors: list[_FakeActor] = []

    def options(self, **kwargs):
        return self

    def remote(self):
        index = len(self.actors)
        actor = _FakeActor(index=index, fail_init=(index == 1))
        self.actors.append(actor)
        return actor


def test_initialize_cleans_created_workers_on_init_failure(monkeypatch):
    worker_cls = _FakeWorkerClass()
    killed = []

    monkeypatch.setattr(
        replica_pool_mod.ray,
        "kill",
        lambda actor: killed.append(actor.index),
    )

    pool = ReplicaPool(worker_cls=worker_cls)
    config = SimpleNamespace(
        tensor_parallel_size=1,
        ray_num_gpus=None,
        extra_env={},
        to_engine_kwargs=lambda: {"model": "test-model"},
    )

    with pytest.raises(RuntimeError, match="worker init failed"):
        asyncio.run(pool.initialize(config, model_id="job-1", num_replicas=2))

    assert [actor.shutdown_called for actor in worker_cls.actors] == [True, True]
    assert killed == [0, 1]
    assert pool._workers == []
    assert pool._config is None
    assert pool._model_id is None


def test_shutdown_starts_all_workers_before_waiting(monkeypatch):
    n = 160
    started: list[int] = []
    killed: list[int] = []

    async def run():
        all_started = asyncio.Event()
        release = asyncio.Event()

        class Actor:
            def __init__(self, index: int) -> None:
                self.index = index
                self.shutdown = _RemoteMethod(self._shutdown)

            async def _shutdown(self):
                started.append(self.index)
                if len(started) == n:
                    all_started.set()
                await release.wait()
                return {"status": "ok", "index": self.index}

        pool = ReplicaPool()
        pool._workers = [Actor(index) for index in range(n)]

        task = asyncio.create_task(pool.shutdown())
        await asyncio.wait_for(all_started.wait(), timeout=1.0)
        assert sorted(started) == list(range(n))

        release.set()
        await asyncio.wait_for(task, timeout=1.0)

    monkeypatch.setattr(
        replica_pool_mod.ray,
        "kill",
        lambda actor: killed.append(actor.index),
    )
    monkeypatch.setenv("ARCTIC_WORKER_SHUTDOWN_CONCURRENCY", str(n))

    asyncio.run(run())

    assert sorted(killed) == list(range(n))


def test_shutdown_default_concurrency_is_capped(monkeypatch):
    cap = replica_pool_mod._DEFAULT_WORKER_SHUTDOWN_CONCURRENCY
    n = cap + 2
    started: list[int] = []
    killed: list[int] = []

    async def run():
        first_wave_started = asyncio.Event()
        release = asyncio.Event()

        class Actor:
            def __init__(self, index: int) -> None:
                self.index = index
                self.shutdown = _RemoteMethod(self._shutdown)

            async def _shutdown(self):
                started.append(self.index)
                if len(started) == cap:
                    first_wave_started.set()
                await release.wait()
                return {"status": "ok", "index": self.index}

        pool = ReplicaPool()
        pool._workers = [Actor(index) for index in range(n)]

        task = asyncio.create_task(pool.shutdown())
        await asyncio.wait_for(first_wave_started.wait(), timeout=1.0)
        await asyncio.sleep(0.05)
        assert len(started) == cap

        release.set()
        await asyncio.wait_for(task, timeout=1.0)

    monkeypatch.setattr(
        replica_pool_mod.ray,
        "kill",
        lambda actor: killed.append(actor.index),
    )

    asyncio.run(run())

    assert sorted(killed) == list(range(n))


def test_shutdown_times_out_hung_workers_and_kills_all(monkeypatch, caplog):
    n = 3
    killed: list[int] = []

    async def run():
        never = asyncio.Event()

        class Actor:
            def __init__(self, index: int) -> None:
                self.index = index
                self.shutdown = _RemoteMethod(self._shutdown)

            async def _shutdown(self):
                if self.index == 0:
                    await never.wait()
                return {"status": "ok", "index": self.index}

        pool = ReplicaPool()
        pool._workers = [Actor(index) for index in range(n)]

        start = time.monotonic()
        await pool.shutdown()
        elapsed = time.monotonic() - start

        assert elapsed < 1.0
        assert pool._workers == []
        assert pool._config is None
        assert pool._model_id is None

    monkeypatch.setenv("ARCTIC_WORKER_SHUTDOWN_TIMEOUT_S", "0.05")
    monkeypatch.setenv("ARCTIC_WORKER_SHUTDOWN_CONCURRENCY", "1")
    monkeypatch.setattr(
        replica_pool_mod.ray,
        "kill",
        lambda actor: killed.append(actor.index),
    )
    caplog.set_level(logging.WARNING, logger="arctic_inference.server")

    asyncio.run(run())

    assert sorted(killed) == list(range(n))
    assert "shutdown RPCs still running for workers [0]" in caplog.text
    assert "shutdown RPCs not yet started for workers [1, 2]" in caplog.text
