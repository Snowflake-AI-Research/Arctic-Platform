from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

runtime_stubbed = False
try:
    import ray  # noqa: F401
except ModuleNotFoundError:
    runtime_stubbed = True
    ray_module = types.ModuleType("ray")
    ray_module.remote = lambda obj=None, **kwargs: (
        obj if obj is not None else (lambda inner: inner)
    )
    ray_module.init = lambda *args, **kwargs: None
    ray_module.nodes = lambda: []
    ray_module.kill = lambda *args, **kwargs: None
    ray_module.cancel = lambda *args, **kwargs: None
    ray_module.actor = SimpleNamespace(ActorHandle=object)
    ray_util = types.ModuleType("ray.util")
    ray_scheduling = types.ModuleType("ray.util.scheduling_strategies")
    ray_scheduling.PlacementGroupSchedulingStrategy = object
    ray_module.util = ray_util
    sys.modules["ray"] = ray_module
    sys.modules["ray.util"] = ray_util
    sys.modules["ray.util.scheduling_strategies"] = ray_scheduling

try:
    import vllm  # noqa: F401
except ModuleNotFoundError:
    runtime_stubbed = True
    vllm_module = types.ModuleType("vllm")
    vllm_config = types.ModuleType("vllm.config")
    vllm_config.VllmConfig = object
    vllm_v1 = types.ModuleType("vllm.v1")
    vllm_metrics = types.ModuleType("vllm.v1.metrics")
    vllm_loggers = types.ModuleType("vllm.v1.metrics.loggers")
    vllm_loggers.StatLoggerBase = object
    vllm_stats = types.ModuleType("vllm.v1.metrics.stats")
    vllm_stats.IterationStats = object
    vllm_stats.SchedulerStats = object
    vllm_stats.MultiModalCacheStats = object
    sys.modules["vllm"] = vllm_module
    sys.modules["vllm.config"] = vllm_config
    sys.modules["vllm.v1"] = vllm_v1
    sys.modules["vllm.v1.metrics"] = vllm_metrics
    sys.modules["vllm.v1.metrics.loggers"] = vllm_loggers
    sys.modules["vllm.v1.metrics.stats"] = vllm_stats

import arctic_inference

if runtime_stubbed and "arctic_inference.server" not in sys.modules:
    server_package = types.ModuleType("arctic_inference.server")
    server_package.__path__ = [
        str(Path(arctic_inference.__file__).parent / "server")
    ]
    sys.modules["arctic_inference.server"] = server_package
    arctic_inference.server = server_package

from arctic_inference.server.replica_pool import ReplicaPool
from arctic_inference.server.scheduler import Scheduler


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    async def remote(self, *args, **kwargs):
        return await self._fn(*args, **kwargs)


class _RecordingWorker:
    def __init__(self, events, index, replay_result):
        self.events = events
        self.index = index
        self.replay_result = replay_result
        self.send_router_replay = _RemoteMethod(self._send_router_replay)

    async def _send_router_replay(self):
        self.events.append(f"send_router_replay:{self.index}")
        if isinstance(self.replay_result, BaseException):
            raise self.replay_result
        return self.replay_result


class _RecordingScheduler:
    def __init__(self, events):
        self.events = events
        self.paused = False

    def pause(self):
        self.paused = True
        self.events.append("scheduler.pause")

    def resume(self):
        self.paused = False
        self.events.append("scheduler.resume")

    async def drain(self):
        self.events.append("scheduler.drain")

    async def abort_streams(self):
        self.events.append("scheduler.abort_streams")

    def cancel_worker_inflight(self, index):
        self.events.append(f"scheduler.cancel_worker_inflight:{index}")


def _recording_pool(events, *replay_results):
    pool = ReplicaPool()
    pool._workers = [
        _RecordingWorker(events, index, result)
        for index, result in enumerate(replay_results)
    ]
    pool._scheduler = _RecordingScheduler(events)
    return pool


class _RemoteEventIterator:
    def __init__(self, release: asyncio.Event):
        self._release = release
        self._events = iter([
            {
                "type": "choice_finished",
                "choice_index": 0,
                "finish_reason": "stop",
                "sequence": 0,
                "version": 1,
            },
            {
                "type": "usage",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "sequence": 1,
                "version": 1,
            },
            {
                "type": "completed",
                "sequence": 2,
                "version": 1,
            },
        ])
        self._first = True

    async def __anext__(self):
        if self._first:
            self._first = False
            await self._release.wait()
        try:
            event = next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None

        async def resolve():
            return event

        return resolve()


def _scheduler_pool(
    *,
    generate,
    send_router_replay,
    stream_release: asyncio.Event,
):
    worker = MagicMock()
    worker.generate.remote = AsyncMock(side_effect=generate)
    worker.send_router_replay.remote = AsyncMock(side_effect=send_router_replay)
    worker.start_stream.remote = AsyncMock(
        return_value={"status": "registered"}
    )
    worker.stream_events.remote = MagicMock(
        side_effect=lambda *_: _RemoteEventIterator(stream_release)
    )
    worker.acknowledge_stream.remote = AsyncMock(return_value=True)
    worker.abort_stream.remote = AsyncMock(return_value={"status": "aborted"})

    pool = ReplicaPool()
    pool._workers = [worker]
    pool._scheduler = Scheduler(
        [worker],
        dynamic_concurrency=False,
        initial_concurrency=4,
    )
    pool._scheduler._ensure_background_tasks = lambda: None
    return pool, worker


async def _wait_until(predicate, timeout: float = 1.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise TimeoutError("condition was not met before the deadline")
        await asyncio.sleep(0.001)


async def _collect_stream(stream):
    return [event async for event in stream]


@pytest.mark.parametrize(
    ("replay_result", "expected_worker"),
    [
        ({"status": "ok"}, {"status": "ok"}),
        (
            RuntimeError("worker replay failed"),
            {"status": "error", "message": "worker replay failed"},
        ),
    ],
)
def test_router_replay_pauses_drains_sends_and_resumes(
    replay_result, expected_worker,
):
    async def run():
        events = []
        pool = _recording_pool(events, {"status": "ok"}, replay_result)

        result = await pool.send_router_replay()

        assert result == {
            "n_replicas": 2,
            "workers": [{"status": "ok"}, expected_worker],
        }
        assert events == [
            "scheduler.pause",
            "scheduler.drain",
            "send_router_replay:0",
            "send_router_replay:1",
            "scheduler.resume",
        ]
        assert "scheduler.abort_streams" not in events
        assert not any("cancel_worker_inflight" in event for event in events)

    asyncio.run(run())


def test_router_replay_preserves_preexisting_scheduler_pause():
    async def run():
        events = []
        pool = _recording_pool(events, {"status": "ok"})
        pool._scheduler.pause()
        events.clear()

        await pool.send_router_replay()

        assert pool._scheduler.paused
        assert events == [
            "scheduler.pause",
            "scheduler.drain",
            "send_router_replay:0",
        ]

    asyncio.run(run())


def test_router_replay_drains_unary_and_streaming_at_pause_boundary():
    async def run():
        unary_started = asyncio.Event()
        unary_release = asyncio.Event()
        stream_release = asyncio.Event()
        replay_started = asyncio.Event()

        async def generate(*_):
            unary_started.set()
            await unary_release.wait()
            return {"text": "done"}

        async def send_router_replay():
            replay_started.set()
            return {"status": "ok"}

        pool, worker = _scheduler_pool(
            generate=generate,
            send_router_replay=send_router_replay,
            stream_release=stream_release,
        )
        scheduler = pool._scheduler
        abort_streams = AsyncMock(wraps=scheduler.abort_streams)
        cancel_worker_inflight = MagicMock(
            wraps=scheduler.cancel_worker_inflight
        )
        scheduler.abort_streams = abort_streams
        scheduler.cancel_worker_inflight = cancel_worker_inflight
        pause_active_requests = []
        original_pause = scheduler.pause

        def record_pause_boundary():
            pause_active_requests.append(
                scheduler._workers[0].active_requests
            )
            original_pause()

        scheduler.pause = record_pause_boundary

        unary = pool.submit_generate_futures("unary")[0]
        stream = pool.stream_generate("stream", "streaming")
        stream_result = asyncio.create_task(_collect_stream(stream))
        replay = asyncio.create_task(pool.send_router_replay())
        try:
            await _wait_until(lambda: scheduler.paused)
            assert pause_active_requests == [2]
            assert unary_started.is_set()
            assert stream.worker is scheduler._workers[0]
            assert stream.registration is not None
            await asyncio.sleep(0.02)
            assert not replay_started.is_set()

            unary_release.set()
            assert await unary == {"text": "done"}
            assert scheduler._workers[0].active_requests == 1
            await asyncio.sleep(0.12)
            assert not replay_started.is_set()

            stream_release.set()
            events = await stream_result
            assert events[-1]["type"] == "completed"
            result = await asyncio.wait_for(replay, 1)

            assert result["workers"] == [{"status": "ok"}]
            assert scheduler._workers[0].active_requests == 0
            assert not scheduler.paused
            abort_streams.assert_not_awaited()
            cancel_worker_inflight.assert_not_called()
            worker.abort_stream.remote.assert_not_awaited()
        finally:
            unary_release.set()
            stream_release.set()
            await asyncio.gather(
                unary,
                stream_result,
                replay,
                return_exceptions=True,
            )
            await scheduler.shutdown()

    asyncio.run(run())


def test_router_replay_queues_after_pause_unary_and_streaming_requests():
    async def run():
        replay_started = asyncio.Event()
        replay_release = asyncio.Event()
        stream_release = asyncio.Event()

        async def generate(*_):
            return {"text": "queued request completed"}

        async def send_router_replay():
            replay_started.set()
            await replay_release.wait()
            return {"status": "ok"}

        pool, worker = _scheduler_pool(
            generate=generate,
            send_router_replay=send_router_replay,
            stream_release=stream_release,
        )
        scheduler = pool._scheduler
        replay = asyncio.create_task(pool.send_router_replay())
        await replay_started.wait()

        unary = pool.submit_generate_futures("queued unary")[0]
        stream = pool.stream_generate("queued-stream", "queued streaming")
        stream_result = asyncio.create_task(_collect_stream(stream))
        try:
            await asyncio.sleep(0.02)
            worker.generate.remote.assert_not_awaited()
            worker.start_stream.remote.assert_not_awaited()
            assert scheduler._workers[0].active_requests == 0

            replay_release.set()
            await asyncio.wait_for(replay, 1)

            assert await asyncio.wait_for(unary, 1) == {
                "text": "queued request completed"
            }
            stream_release.set()
            events = await asyncio.wait_for(stream_result, 1)
            assert events[-1]["type"] == "completed"
            worker.generate.remote.assert_awaited_once()
            worker.start_stream.remote.assert_awaited_once()
            assert not scheduler.paused
        finally:
            replay_release.set()
            stream_release.set()
            await asyncio.gather(replay, stream_result, return_exceptions=True)
            await scheduler.shutdown()

    asyncio.run(run())


def test_router_replay_cancellation_waits_for_unsettled_collective():
    async def run():
        replay_started = asyncio.Event()
        replay_release = asyncio.Event()
        stream_release = asyncio.Event()
        send_cancelled = asyncio.Event()
        lifecycle_entered = asyncio.Event()

        async def generate(*_):
            return {"text": "done"}

        async def send_router_replay():
            replay_started.set()
            try:
                await replay_release.wait()
            except asyncio.CancelledError:
                send_cancelled.set()
                raise
            return {"status": "ok"}

        pool, worker = _scheduler_pool(
            generate=generate,
            send_router_replay=send_router_replay,
            stream_release=stream_release,
        )
        scheduler = pool._scheduler
        replay = asyncio.create_task(pool.send_router_replay())
        lifecycle = None
        try:
            await replay_started.wait()
            replay.cancel()
            await asyncio.sleep(0.02)

            assert scheduler.paused
            assert pool._lock.locked()
            assert not replay.done()
            assert not send_cancelled.is_set()

            async def lifecycle_operation():
                async with pool._lock:
                    lifecycle_entered.set()

            lifecycle = asyncio.create_task(lifecycle_operation())
            await asyncio.sleep(0.02)
            assert not lifecycle_entered.is_set()

            replay_release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(replay, 1)
            await asyncio.wait_for(lifecycle, 1)

            assert lifecycle_entered.is_set()
            assert not scheduler.paused
            assert not pool._lock.locked()
            assert not send_cancelled.is_set()
            worker.abort_stream.remote.assert_not_awaited()
        finally:
            replay_release.set()
            await asyncio.gather(
                replay,
                *([lifecycle] if lifecycle is not None else []),
                return_exceptions=True,
            )
            await scheduler.shutdown()

    asyncio.run(run())
