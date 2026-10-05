"""Regressions for streaming admission, cleanup, and replica lifecycle review."""

import asyncio
import time
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import ray

from test_library_streaming import FakeEngine, FakeEngineWorker, runtime as runtime
from arctic_platform.inference.server.multi_model import BaseExceptionGroup, Driver
from arctic_platform.inference.server.replica_pool import ReplicaPool
from arctic_platform.inference.server.scheduler import Scheduler
from arctic_platform.inference.server.streaming import MAX_WORKER_STREAMS, StreamError, StreamLimits
from arctic_platform.inference.server.worker import InferenceWorker, WorkerLifecycleState


def local_worker():
    worker = InferenceWorker.__ray_metadata__.modified_class()
    worker.state = WorkerLifecycleState.READY
    worker.llm = FakeEngine()
    worker._stream_sampling_params = lambda params: params
    return worker


@pytest.mark.parametrize("clock_offset", [-3600, 3600])
def test_worker_deadline_uses_local_monotonic_clock(monkeypatch, clock_offset):
    async def scenario():
        import arctic_platform.inference.server.streaming as streaming

        worker = local_worker()
        limits = StreamLimits(timeout_s=0.2, stall_timeout_s=10)
        monkeypatch.setattr(streaming, "time", SimpleNamespace(
            time=lambda: time.time() + clock_offset, monotonic=time.monotonic
        ))
        before = time.monotonic()
        worker.start_stream("skew", "blocked", {}, 0.2, asdict(limits))
        session = worker._engine_streams["skew"]
        try:
            assert before < session.expires_at <= time.monotonic() + 0.2
            await asyncio.wait_for(session.watchdog, 2)
            assert session.buffer.error.code == "deadline_exceeded"
            assert not worker._engine_streams
            assert not worker.llm.active
        finally:
            await worker.abort_all_streams()

    asyncio.run(scenario())


@pytest.mark.parametrize("abort_fails", [False, True])
def test_cancellation_has_one_engine_abort_owner(abort_fails):
    async def scenario():
        worker = local_worker()
        worker.llm.abort = AsyncMock(
            side_effect=[RuntimeError("first abort failed"), None]
            if abort_fails else None
        )
        worker.start_stream(
            "single-abort", "blocked", {}, 20, asdict(StreamLimits())
        )
        session = worker._engine_streams["single-abort"]
        await session.buffer.get()
        result = await worker.abort_stream("single-abort")
        assert worker.llm.abort.await_count == 1
        assert result["status"] == ("cleanup_unconfirmed" if abort_fails else "aborted")
        assert bool(getattr(worker, "_stream_cleanup_failed", False)) == abort_fails
        assert not worker._engine_streams

    asyncio.run(scenario())


@pytest.mark.parametrize("model_id", [None, "first"])
def test_driver_shutdown_clears_failed_pools_and_attempts_all(model_id):
    async def scenario():
        driver = Driver()
        first = SimpleNamespace(shutdown=AsyncMock(side_effect=RuntimeError("cleanup failed")))
        second = SimpleNamespace(shutdown=AsyncMock())
        driver._pools.update(first=first, second=second)
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await driver.shutdown(model_id)
        first.shutdown.assert_awaited_once()
        if model_id is None:
            second.shutdown.assert_awaited_once()
            assert not driver._pools
        else:
            second.shutdown.assert_not_awaited()
            assert driver._pools == {"second": second}

    asyncio.run(scenario())


@pytest.mark.parametrize("clock_offset", [-3600, 3600])
def test_client_clock_skew_and_queue_time_budget(monkeypatch, clock_offset):
    async def scenario():
        import arctic_platform.inference.server.streaming as streaming

        actor = FakeEngineWorker.remote()
        await actor.stats.remote()
        scheduler = Scheduler([actor])
        reader = None
        try:
            scheduler._paused = True
            stream = scheduler.stream_generate(
                "clock", "blocked", limits=StreamLimits(timeout_s=5)
            )
            monkeypatch.setattr(streaming, "time", SimpleNamespace(
                time=lambda: time.time() + clock_offset, monotonic=time.monotonic
            ))
            reader = asyncio.create_task(anext(stream))
            await asyncio.sleep(0.1)
            assert not reader.done()
            scheduler._paused = False
            assert (await asyncio.wait_for(reader, 5))["type"] == "delta"
            assert 0 < (await actor.stats.remote())["remaining_s"] < 4.95
            await stream.aclose()
        finally:
            await scheduler.shutdown()
            if reader is not None:
                await asyncio.gather(reader, return_exceptions=True)
            ray.kill(actor)

    asyncio.run(asyncio.wait_for(scenario(), 20))


def test_failed_single_abort_quarantines_scheduler_replica():
    async def scenario():
        actor = FakeEngineWorker.remote()
        await actor.fail_first_abort.remote()
        scheduler = Scheduler([actor])
        try:
            stream = scheduler.stream_generate("failed-abort", "blocked")
            await anext(stream)
            assert (await stream.abort())["status"] == "cleanup_unconfirmed"
            assert not scheduler._workers[0].available
            stats = await actor.stats.remote()
            assert stats["cleanup_failed"]
            assert stats["abort_calls"] == 1
            assert stats["sessions"] == 0
        finally:
            await scheduler.shutdown()
            ray.kill(actor)

    asyncio.run(asyncio.wait_for(scenario(), 20))


def test_cancellation_reuses_inflight_engine_cleanup():
    async def scenario():
        worker = local_worker()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def abort(request_id):
            entered.set()
            await release.wait()

        worker.llm.abort = AsyncMock(side_effect=abort)
        worker.start_stream("error-abort", "failure", {}, 20, asdict(StreamLimits()))
        session = worker._engine_streams["error-abort"]
        await asyncio.wait_for(entered.wait(), 2)
        stopping = asyncio.create_task(worker.abort_stream("error-abort"))
        try:
            await asyncio.sleep(0.02)
            assert not stopping.done()
            assert worker.llm.abort.await_count == 1
            release.set()
            assert (await asyncio.wait_for(stopping, 2))["status"] == "aborted"
            assert session.cleanup_confirmed
            assert worker.llm.abort.await_count == 1
        finally:
            release.set()
            await worker.abort_all_streams()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancelled", [False, True])
def test_driver_shutdown_collects_multiple_failures(cancelled):
    async def scenario():
        driver = Driver()
        errors = [RuntimeError("first"), asyncio.CancelledError() if cancelled else ValueError("second")]
        pools = [SimpleNamespace(shutdown=AsyncMock(side_effect=error)) for error in errors]
        pools.append(SimpleNamespace(shutdown=AsyncMock()))
        driver._pools.update({str(index): pool for index, pool in enumerate(pools)})
        with pytest.raises(BaseExceptionGroup) as caught:
            await driver.shutdown()
        assert list(caught.value.exceptions) == errors
        assert not driver._pools
        for pool in pools:
            pool.shutdown.assert_awaited_once()

    asyncio.run(scenario())


def test_streams_wait_for_configured_worker_capacity():
    async def scenario():
        shared_slots = 4
        actor = FakeEngineWorker.options(max_concurrency=128).remote()
        await actor.stats.remote()
        scheduler = Scheduler([actor], initial_concurrency=shared_slots)
        reader = None
        try:
            streams = [
                scheduler.stream_generate(f"stream-{index}", "blocked")
                for index in range(shared_slots + 1)
            ]
            await asyncio.gather(*(anext(stream) for stream in streams[:shared_slots]))
            reader = asyncio.create_task(anext(streams[shared_slots]))
            await asyncio.sleep(0.1)
            assert not reader.done(), "Streaming must respect the configured shared limit"
            assert scheduler._workers[0].available
            assert (await actor.stats.remote())["sessions"] == shared_slots
            await streams[0].aclose()
            assert (await asyncio.wait_for(reader, 5))["type"] == "delta"
            assert scheduler._workers[0].available
        finally:
            if reader is not None:
                await asyncio.gather(reader, return_exceptions=True)
            await scheduler.shutdown()
            ray.kill(actor)

    asyncio.run(asyncio.wait_for(scenario(), 25))


@pytest.mark.parametrize("choices", [1, 8])
@pytest.mark.parametrize("replicas", [1, 2])
def test_stream_capacity_scales_with_replicas_and_queues_overflow(monkeypatch, choices, replicas):
    monkeypatch.delenv("ARCTIC_WORKER_CONCURRENCY_LIMIT", raising=False)

    async def scenario():
        assert MAX_WORKER_STREAMS == 128
        actors = [FakeEngineWorker.options(max_concurrency=256).remote() for _ in range(replicas)]
        await asyncio.gather(*(actor.stats.remote() for actor in actors))
        pool = ReplicaPool()
        pool._scheduler = pool._make_scheduler(actors)
        scheduler = pool._scheduler
        reader = None
        try:
            streams = [
                pool.stream_generate(
                    f"stream-{index}", "blocked", {"n": choices, "max_tokens": 2}
                )
                for index in range(replicas * MAX_WORKER_STREAMS)
            ]
            events = await asyncio.gather(*(anext(stream) for stream in streams))
            assert all(event["type"] == "delta" for event in events)
            for state, actor in zip(scheduler._workers, actors):
                assert state.active_requests == MAX_WORKER_STREAMS
                assert state.streaming_requests == MAX_WORKER_STREAMS
                assert (await actor.stats.remote())["sessions"] == MAX_WORKER_STREAMS
            queued = pool.stream_generate("over-capacity", "blocked", {"n": choices, "max_tokens": 2})
            reader = asyncio.create_task(anext(queued))
            await asyncio.sleep(0.1)
            assert not reader.done()
            assert queued.worker is None
            assert len(scheduler._streams) == replicas * MAX_WORKER_STREAMS + 1
            await streams[0].aclose()
            assert (await asyncio.wait_for(reader, 5))["type"] == "delta"
            assert queued.worker is streams[0].worker
            assert all(state.active_requests == MAX_WORKER_STREAMS for state in scheduler._workers)
            await scheduler.shutdown()
            for actor in actors:
                stats = await actor.stats.remote()
                assert stats["sessions"] == 0
                assert stats["active"] == 0
        finally:
            await scheduler.shutdown()
            if reader is not None:
                await asyncio.gather(reader, return_exceptions=True)
            for actor in actors:
                ray.kill(actor)

    asyncio.run(asyncio.wait_for(scenario(), 45))


@pytest.mark.parametrize("shared_slots", [1, 3])
def test_streams_consume_legacy_capacity_until_closed(shared_slots):
    async def scenario():
        actor = FakeEngineWorker.remote()
        await actor.stats.remote()
        scheduler = Scheduler(
            [actor], initial_concurrency=shared_slots, dynamic_concurrency=False
        )
        legacy = None
        try:
            streams = [
                scheduler.stream_generate(f"shared-{index}", "blocked")
                for index in range(shared_slots)
            ]
            await asyncio.gather(*(anext(stream) for stream in streams))
            worker = scheduler._workers[0]
            assert worker.active_requests == shared_slots
            assert worker.streaming_requests == shared_slots
            routed = asyncio.Event()
            original_routing = scheduler._routing_fn

            def observed_routing(request, workers):
                routed.set()
                return original_routing(request, workers)

            scheduler._routing_fn = observed_routing
            legacy = scheduler.submit("legacy", {"max_tokens": 1})
            await asyncio.wait_for(routed.wait(), 5)
            assert not legacy.done()
            assert (await actor.stats.remote())["legacy_calls"] == 0
            await streams[0].aclose()
            for attempt in range(100):
                if (await actor.stats.remote())["legacy_calls"] == 1:
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("Legacy generation did not acquire the released slot")
            assert worker.active_requests == shared_slots
            assert worker.streaming_requests == shared_slots - 1
            await actor.release_legacy.remote()
            assert (await asyncio.wait_for(legacy, 5))["text"] == "legacy result"
            assert worker.active_requests == shared_slots - 1
            await scheduler.abort_streams()
            assert worker.active_requests == worker.streaming_requests == 0
        finally:
            await actor.release_legacy.remote()
            await scheduler.shutdown()
            if legacy is not None:
                await asyncio.wait_for(asyncio.gather(legacy, return_exceptions=True), 5)
            ray.kill(actor)

    asyncio.run(asyncio.wait_for(scenario(), 25))


def test_legacy_generation_consumes_stream_capacity_until_finished():
    async def scenario():
        actor = FakeEngineWorker.remote()
        await actor.stats.remote()
        scheduler = Scheduler([actor], initial_concurrency=1, dynamic_concurrency=False)
        legacy = None
        reader = None
        try:
            legacy = scheduler.submit("legacy", {"max_tokens": 1})
            for attempt in range(100):
                if (await actor.stats.remote())["legacy_calls"] == 1:
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("Legacy generation did not start")
            stream = scheduler.stream_generate("waiting-stream", "blocked")
            routed = asyncio.Event()
            original_routing = scheduler._routing_fn

            def observed_routing(request, workers):
                routed.set()
                return original_routing(request, workers)

            scheduler._routing_fn = observed_routing
            reader = asyncio.create_task(anext(stream))
            await asyncio.wait_for(routed.wait(), 5)
            assert not reader.done()
            worker = scheduler._workers[0]
            assert worker.active_requests == 1
            assert worker.streaming_requests == 0
            assert (await actor.stats.remote())["sessions"] == 0
            await actor.release_legacy.remote()
            await asyncio.wait_for(legacy, 5)
            assert (await asyncio.wait_for(reader, 5))["type"] == "delta"
            assert worker.active_requests == worker.streaming_requests == 1
            await stream.aclose()
            assert worker.active_requests == worker.streaming_requests == 0
        finally:
            await actor.release_legacy.remote()
            await scheduler.shutdown()
            pending = [future for future in (legacy, reader) if future is not None]
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), 5)
            ray.kill(actor)

    asyncio.run(asyncio.wait_for(scenario(), 25))


def test_review_capacity_rejection_does_not_quarantine():
    async def scenario():
        actor = FakeEngineWorker.options(max_concurrency=128).remote()
        await actor.stats.remote()
        scheduler = Scheduler([actor], initial_concurrency=128)
        try:
            for index in range(MAX_WORKER_STREAMS):
                await actor.start_stream.remote(
                    f"external-{index}",
                    "blocked",
                    {},
                    20,
                    asdict(StreamLimits()),
                )
            stream = scheduler.stream_generate("rejected", "blocked")
            with pytest.raises((StreamError, ray.exceptions.RayTaskError)):
                await anext(stream)
            assert scheduler._workers[0].available
            assert scheduler._workers[0].active_requests == 0
        finally:
            await scheduler.shutdown()
            ray.kill(actor)

    asyncio.run(asyncio.wait_for(scenario(), 25))


def test_review_shutdown_kills_workers_after_cleanup_failure(monkeypatch):
    async def scenario():
        worker = MagicMock()
        worker.shutdown.remote.side_effect = RuntimeError("graceful worker failure")
        pool = ReplicaPool()
        pool._workers = [worker]
        scheduler = Scheduler([worker])
        pool._scheduler = scheduler
        stream = SimpleNamespace(
            abort=AsyncMock(return_value={"status": "cleanup_unconfirmed"})
        )
        scheduler._streams["stuck"] = stream
        scheduler._poll_task = asyncio.create_task(asyncio.sleep(60))
        scheduler._adjust_task = asyncio.create_task(asyncio.sleep(60))
        killed = []
        monkeypatch.setattr(ray, "kill", killed.append)
        with pytest.raises(RuntimeError, match="cleanup"):
            await pool.shutdown()
        assert killed == [worker]
        assert scheduler._poll_task.done()
        assert scheduler._adjust_task.done()
        assert pool._scheduler is None
        assert pool._workers == []

    asyncio.run(scenario())


def test_review_retired_cache_evicts_instead_of_blocking_admission():
    async def scenario():
        scheduler = Scheduler([MagicMock()])
        now = time.monotonic()
        scheduler._stream_retired.update(
            (f"old-{index}", now) for index in range(100000)
        )
        try:
            stream = scheduler.stream_generate("new", "prompt")
            await stream.aclose()
            assert len(scheduler._stream_retired) <= 100000
            assert "old-0" not in scheduler._stream_retired
            with pytest.raises(ValueError, match="Duplicate"):
                scheduler.stream_generate("new", "prompt")
        finally:
            await scheduler.shutdown()

    asyncio.run(scenario())


def test_retired_ids_expire_without_changing_active_admission():
    async def scenario():
        scheduler = Scheduler([MagicMock()])
        scheduler._stream_retired["expired"] = time.monotonic() - 3601
        scheduler._stream_retired["recent"] = time.monotonic()
        try:
            assert (await scheduler.abort("expired"))["status"] == "not_found"
            assert (await scheduler.abort("recent"))["status"] == "already_terminal"
            stream = scheduler.stream_generate("expired", "prompt")
            with pytest.raises(ValueError, match="Duplicate"):
                scheduler.stream_generate("expired", "prompt")
            await stream.aclose()
        finally:
            await scheduler.shutdown()

    asyncio.run(scenario())


def test_registration_validation_error_keeps_worker_available():
    async def scenario():
        actor = FakeEngineWorker.remote()
        await actor.stats.remote()
        scheduler = Scheduler([actor])
        try:
            stream = scheduler.stream_generate("invalid-deadline", "prompt")
            stream.expires_at += 3600
            with pytest.raises(ray.exceptions.RayTaskError):
                await anext(stream)
            assert scheduler._workers[0].available
            assert scheduler._workers[0].streaming_requests == 0
            assert (await actor.stats.remote())["sessions"] == 0
        finally:
            await scheduler.shutdown()
            ray.kill(actor)

    asyncio.run(asyncio.wait_for(scenario(), 15))


def test_review_abort_before_reader_releases_session_immediately():
    async def scenario():
        worker = local_worker()
        for index in range(40):
            attempt = f"attempt-{index}"
            worker.start_stream(
                attempt, "blocked", {}, 20, asdict(StreamLimits())
            )
            session = worker._engine_streams[attempt]
            assert (await worker.abort_stream(attempt))["status"] == "aborted"
            assert not worker._engine_streams
            await asyncio.sleep(0)
            assert session.watchdog.done()
            assert not worker.llm.active

    asyncio.run(scenario())


@pytest.mark.parametrize("close_fails", [False, True])
def test_review_success_waits_for_engine_iterator_cleanup(close_fails):
    async def scenario():
        worker = local_worker()
        entered = asyncio.Event()
        release = asyncio.Event()

        class ControlledOutput:
            def __init__(self):
                self.sent = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.sent:
                    raise StopAsyncIteration
                self.sent = True
                return SimpleNamespace(
                    prompt_token_ids=[1],
                    outputs=[
                        SimpleNamespace(
                            index=0, text="x", token_ids=[2], finish_reason="stop"
                        )
                    ],
                )

            async def aclose(self):
                entered.set()
                await release.wait()
                if close_fails:
                    raise RuntimeError("close failed")

        worker.llm.generate = lambda *args, **kwargs: ControlledOutput()
        worker.start_stream(
            "cleanup", "prompt", {}, 20, asdict(StreamLimits())
        )
        session = worker._engine_streams["cleanup"]
        try:
            await entered.wait()
            assert not any(
                event["type"] in {"usage", "completed"}
                for event, size in session.buffer.events
            )
            release.set()
            await session.pump
            if close_fails:
                assert session.buffer.error.code == "cleanup_unconfirmed"
                assert not any(
                    event["type"] == "completed"
                    for event, size in session.buffer.events
                )
            else:
                assert session.buffer.events[-1][0]["type"] == "completed"
        finally:
            release.set()
            await worker.abort_stream("cleanup")
            session.watchdog.cancel()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["cleanup_unconfirmed", "exception", "cancelled"])
def test_scale_down_force_removes_target_after_cleanup_failure(monkeypatch, failure):
    async def scenario():
        actors = [MagicMock(), MagicMock()]
        pool = ReplicaPool()
        pool._workers = list(actors)
        scheduler = Scheduler(actors)
        pool._scheduler = scheduler
        retained = SimpleNamespace(worker=scheduler._workers[0], abort=AsyncMock())
        removed = SimpleNamespace(
            worker=scheduler._workers[1],
            abort=AsyncMock(return_value={"status": "cleanup_unconfirmed"}),
        )
        if failure == "exception":
            removed.abort.side_effect = RuntimeError("abort RPC failed")
        scheduler._streams.update(retained=retained, removed=removed)
        if failure == "cancelled":
            scheduler.abort_streams = AsyncMock(side_effect=asyncio.CancelledError)
        killed = []
        monkeypatch.setattr(ray, "kill", killed.append)
        expected = asyncio.CancelledError if failure == "cancelled" else RuntimeError
        with pytest.raises(expected):
            await pool.scale_down(1)
        assert killed == [actors[1]]
        assert pool._workers == [actors[0]]
        assert len(scheduler._workers) == 1
        assert scheduler._workers[0].available
        retained.abort.assert_not_awaited()
        assert not pool._lock.locked()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure", ["none", "abort_failure", "already_unhealthy", "exception"]
)
def test_bulk_abort_releases_sessions_and_watchdogs(failure):
    async def scenario():
        worker = local_worker()
        for index in range(3):
            worker.start_stream(
                f"bulk-{index}", "blocked", {}, 20, asdict(StreamLimits())
            )
        sessions = list(worker._engine_streams.values())
        await asyncio.sleep(0.02)
        if failure == "abort_failure":
            worker.llm.abort = AsyncMock(
                side_effect=RuntimeError("engine abort failed")
            )
        elif failure == "already_unhealthy":
            worker._stream_cleanup_failed = True
        elif failure == "exception":
            await sessions[0].stop("test_setup")
            sessions[0].stop = AsyncMock(side_effect=RuntimeError("stop failed"))
        try:
            if failure == "none":
                await worker.abort_all_streams()
            else:
                with pytest.raises(RuntimeError, match="cleanup"):
                    await worker.abort_all_streams()
            assert not worker._engine_streams
            await asyncio.sleep(0)
            assert all(session.watchdog.done() for session in sessions)
            assert not worker.llm.active
            if failure == "none":
                await worker.abort_all_streams()
                worker.start_stream(
                    "next", "blocked", {}, 20, asdict(StreamLimits())
                )
                await worker.abort_all_streams()
                assert not worker._engine_streams
        finally:
            for session in sessions:
                session.watchdog.cancel()
                if not session.pump.done():
                    session.pump.cancel()
            await asyncio.gather(
                *(session.pump for session in sessions), return_exceptions=True
            )

    asyncio.run(scenario())


def test_review_scale_down_preserves_retained_replica_streams():
    async def scenario():
        actors = [FakeEngineWorker.remote(), FakeEngineWorker.remote()]
        await asyncio.gather(*(actor.stats.remote() for actor in actors))
        pool = ReplicaPool()
        pool._workers = list(actors)
        scheduler = Scheduler(actors)
        pool._scheduler = scheduler
        driver = Driver()
        driver._pools["model"] = pool
        try:
            retained = driver.stream_generate(
                "model", "retained", "blocked", {"max_tokens": 2}
            )
            removed = driver.stream_generate(
                "model", "removed", "blocked", {"max_tokens": 2}
            )
            await anext(retained)
            await anext(removed)
            assert retained.request.worker_idx == 0
            assert removed.request.worker_idx == 1
            await pool.scale_down(1)
            assert not retained.closed
            assert removed.closed
            await actors[0].release.remote()
            events = [event async for event in retained]
            assert events[-1]["type"] == "completed"
        finally:
            await scheduler.shutdown()
            for actor in actors:
                ray.kill(actor)

    asyncio.run(asyncio.wait_for(scenario(), 25))
