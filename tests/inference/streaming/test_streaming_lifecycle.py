"""Regression coverage for streaming cleanup, admission, and lifecycle safety."""

import asyncio
import time
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from test_review_regressions import local_worker
from arctic_platform.inference.server.config import ModelConfig
from arctic_platform.inference.server.replica_pool import ReplicaPool
from arctic_platform.inference.server.scheduler import (
    Scheduler,
    least_loaded_routing,
    prefix_affinity_routing,
    strict_affinity_routing,
)
from arctic_platform.inference.server.streaming import StreamLimits


WEIGHT_METHODS = (
    "sync_weights",
    "sync_weights_broadcast",
    "sync_lora_weights_broadcast",
    "sync_spec_weights",
)
MUTATIONS = [("sleep", None)] + [
    (method, strategy)
    for method in WEIGHT_METHODS
    for strategy in ("pause", "drain", "skip", "hotswap")
    if method != "sync_spec_weights" or strategy != "pause"
]


def make_pool():
    handle = MagicMock()
    for method in (*WEIGHT_METHODS, "sleep", "wake_up", "pause_generation",
                   "resume_generation", "abort_stream"):
        getattr(handle, method).remote = AsyncMock(return_value={"status": "aborted"})
    handle.generate.remote = AsyncMock(return_value={"text": "legacy", "token_ids": [1]})
    pool = ReplicaPool()
    pool._config = ModelConfig(model="fake")
    pool._workers = [handle]
    pool._scheduler = Scheduler([handle], dynamic_concurrency=False)
    pool._scheduler._ensure_background_tasks = lambda: None
    pool._reset_prefix_cache_after_weight_sync = AsyncMock(return_value={"status": "ok"})
    return pool, handle


async def mutate(pool, method, strategy):
    if method == "sleep":
        return await pool.sleep()
    kwargs = dict(master_addr="127.0.0.1", master_port=12345, strategy=strategy)
    if method == "sync_lora_weights_broadcast":
        kwargs["lora_config"] = {}
    return await getattr(pool, method)(**kwargs)


@pytest.mark.parametrize("ignore_cancellation", [False, True])
def test_cancel_during_iterator_close_blocks_mutation(ignore_cancellation):
    async def scenario():
        worker = local_worker()
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        release = asyncio.Event()
        close_tasks = []

        class ControlledOutput:
            sent = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.sent:
                    raise StopAsyncIteration
                self.sent = True
                return SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(
                    index=0, text="x", token_ids=[2], finish_reason="stop"
                )])

            async def aclose(self):
                close_tasks.append(asyncio.current_task())
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    if ignore_cancellation:
                        await release.wait()
                    else:
                        raise

        worker.llm.generate = lambda *args, **kwargs: ControlledOutput()
        worker.llm.pause_generation = AsyncMock()
        worker.start_stream(
            "closing", "prompt", {"max_tokens": 1}, 20,
            asdict(StreamLimits(cleanup_timeout_s=0.05)),
        )
        session = worker._engine_streams["closing"]
        try:
            await asyncio.wait_for(entered.wait(), 1)
            result = await asyncio.wait_for(worker.abort_stream("closing"), 1)
            assert result["status"] == "cleanup_unconfirmed"
            assert cancelled.is_set()
            if not ignore_cancellation:
                assert close_tasks[0].done()
            with pytest.raises(RuntimeError, match="cleanup unconfirmed"):
                await worker.pause_generation()
            worker.llm.pause_generation.assert_not_awaited()
        finally:
            release.set()
            await asyncio.gather(session.pump, *close_tasks, return_exceptions=True)
            session.watchdog.cancel()

    asyncio.run(scenario())


@pytest.mark.parametrize("routing", [
    least_loaded_routing, prefix_affinity_routing, strict_affinity_routing, None,
])
def test_quarantine_blocks_legacy_dispatch_until_worker_replacement(routing):
    async def scenario():
        pool, handle = make_pool()
        scheduler = pool._scheduler
        if routing is not None:
            scheduler._routing_fn = routing
        stream = scheduler.stream_generate("failed", "prompt")
        stream.worker = scheduler._workers[0]
        stream.worker.active_requests = stream.worker.streaming_requests = 1
        handle.abort_stream.remote.return_value = {"status": "cleanup_unconfirmed"}
        await stream.abort()
        scheduler.mark_worker_available(0)
        try:
            assert not scheduler.is_worker_available(0)
            stream.worker.available = True
            future = scheduler.submit("prompt", {}, routing_key="key" if routing is None else None)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(future), 0.02)
            handle.generate.remote.assert_not_awaited()
            replacement = MagicMock()
            replacement.generate.remote = AsyncMock(return_value={"text": "recovered"})
            scheduler.update_worker_handle(0, replacement)
            scheduler.mark_worker_available(0)
            assert (await asyncio.wait_for(future, 1))["text"] == "recovered"
            replacement.generate.remote.assert_awaited_once()
        finally:
            await scheduler.shutdown()

    asyncio.run(scenario())


def test_generator_cancel_failure_quarantines_worker(monkeypatch):
    async def scenario():
        pool, _ = make_pool()
        scheduler = pool._scheduler
        stream = scheduler.stream_generate("cancel", "prompt")
        stream.worker = scheduler._workers[0]
        stream.worker.active_requests = stream.worker.streaming_requests = 1
        stream.remote_stream = object()
        monkeypatch.setattr("ray.cancel", MagicMock(side_effect=RuntimeError("cancel failed")))
        result = await stream.abort()
        assert result["status"] == "cleanup_unconfirmed"
        scheduler.mark_worker_available(0)
        assert not scheduler.is_worker_available(0)
        await scheduler.shutdown()

    asyncio.run(scenario())


def test_worker_rejects_legacy_generation_after_uncertain_cleanup():
    async def scenario():
        worker = local_worker()
        worker._stream_cleanup_failed = True
        with pytest.raises(RuntimeError, match="cleanup unconfirmed"):
            await worker.generate("prompt", {})
        assert not worker.llm.calls

    asyncio.run(scenario())


@pytest.mark.parametrize("method", WEIGHT_METHODS)
@pytest.mark.parametrize("failed", [False, True])
def test_skip_sync_does_not_clear_quarantine(method, failed):
    async def scenario():
        pool, handle = make_pool()
        scheduler = pool._scheduler
        state = scheduler._workers[0]
        state.quarantined = True
        state.available = False
        if failed:
            getattr(handle, method).remote.side_effect = RuntimeError("sync failed")
        try:
            await mutate(pool, method, "skip")
        except RuntimeError:
            assert failed
        assert not scheduler.is_worker_available(0)
        await scheduler.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("method,strategy", MUTATIONS)
@pytest.mark.parametrize("state", ["unread", "queued", "running"])
def test_mutation_aborts_all_scheduler_streams_before_engine_work(method, strategy, state):
    async def scenario():
        pool, handle = make_pool()
        scheduler = pool._scheduler
        scheduler.pause()
        stream = pool.stream_generate("existing", "prompt")
        reader = None
        if state == "queued":
            reader = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
        elif state == "running":
            stream.worker = scheduler._workers[0]
            stream.worker.active_requests = stream.worker.streaming_requests = 1

        async def engine_mutation(*args, **kwargs):
            assert stream.closed
            assert not scheduler._streams
            with pytest.raises(RuntimeError, match="not available"):
                pool.stream_generate("during-mutation", "prompt")
            return {"status": "ok"}

        getattr(handle, method).remote.side_effect = engine_mutation
        try:
            await asyncio.wait_for(mutate(pool, method, strategy), 1)
            assert stream.closed
            assert stream.error == "lifecycle_change"
            getattr(handle, method).remote.assert_awaited_once()
            if method == "sleep":
                await pool.wake_up()
            follow_up = pool.stream_generate("after-mutation", "prompt")
            await follow_up.aclose()
        finally:
            await stream.abort()
            if reader is not None:
                await asyncio.gather(reader, return_exceptions=True)
            await scheduler.shutdown()

    asyncio.run(scenario())


def test_scale_up_keeps_stream_admission_open():
    async def scenario():
        pool, _ = make_pool()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def initialize(*args):
            entered.set()
            await release.wait()

        new_handle = MagicMock()
        new_handle.initialize.remote = AsyncMock(side_effect=initialize)
        pool._make_worker = lambda: new_handle
        pool._engine_kwargs = lambda: {}
        scaling = asyncio.create_task(pool.scale_up(2))
        try:
            await entered.wait()
            stream = pool.stream_generate("during-scale-up", "prompt")
            assert not stream.closed
            assert (await pool.generate("prompt", {}))[0]["text"] == "legacy"
            await stream.aclose()
        finally:
            release.set()
            await scaling
            await pool._scheduler.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("method,strategy", MUTATIONS)
def test_uncertain_stream_cleanup_prevents_engine_mutation(method, strategy):
    async def scenario():
        pool, handle = make_pool()
        scheduler = pool._scheduler
        stream = pool.stream_generate("uncertain", "prompt")
        stream.worker = scheduler._workers[0]
        stream.worker.active_requests = stream.worker.streaming_requests = 1
        handle.abort_stream.remote.return_value = {"status": "cleanup_unconfirmed"}
        healthy = MagicMock()
        healthy.generate.remote = AsyncMock(return_value={"text": "healthy"})
        getattr(healthy, method).remote = AsyncMock()
        healthy.pause_generation.remote = AsyncMock()
        pool._workers.append(healthy)
        scheduler.add_worker(healthy)
        try:
            with pytest.raises(RuntimeError, match="cleanup unconfirmed"):
                await mutate(pool, method, strategy)
            for worker in (handle, healthy):
                getattr(worker, method).remote.assert_not_awaited()
                worker.pause_generation.remote.assert_not_awaited()
            assert scheduler._workers[0].quarantined
            assert not scheduler.is_worker_available(0)
            assert scheduler.is_worker_available(1)
            assert not scheduler._paused
            assert not pool._stream_admission_blocked
            assert not pool.sleeping
            assert await pool.wake_up() == {"status": "already_ready"}
            result = await asyncio.wait_for(scheduler.submit("prompt", {}), 1)
            assert result["text"] == "healthy"
            healthy.generate.remote.assert_awaited_once()
            handle.generate.remote.assert_not_awaited()
            follow_up = pool.stream_generate("after-failed-mutation", "prompt")
            await follow_up.aclose()
        finally:
            await scheduler.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("method,strategy,phase", [
    ("sleep", None, "abort_streams"),
    ("sleep", None, "drain"),
] + [
    (method, strategy, phase)
    for method, strategy in MUTATIONS[1:]
    for phase in ("abort_streams", "drain")
    if phase == "abort_streams" or strategy == "drain"
])
@pytest.mark.parametrize("initially_paused", [False, True])
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_lifecycle_preparation_failure_restores_previous_state(
    method, strategy, phase, initially_paused, failure,
):
    async def scenario():
        pool, handle = make_pool()
        scheduler = pool._scheduler
        unavailable = MagicMock()
        pool._workers.append(unavailable)
        scheduler.add_worker(unavailable)
        scheduler.mark_worker_unavailable(1)
        if initially_paused:
            scheduler.pause()
        original_phase = getattr(scheduler, phase)
        setattr(scheduler, phase, AsyncMock(side_effect=failure("preparation failed")))
        try:
            with pytest.raises(failure, match="preparation failed"):
                await mutate(pool, method, strategy)
            assert scheduler._paused == initially_paused
            assert scheduler._pause_event.is_set() == (not initially_paused)
            assert scheduler.is_worker_available(0)
            assert not scheduler.is_worker_available(1)
            assert not pool._stream_admission_blocked
            assert not pool.sleeping
            getattr(handle, method).remote.assert_not_awaited()
        finally:
            setattr(scheduler, phase, original_phase)
            await scheduler.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("method,strategy", MUTATIONS[1:])
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_weight_sync_preparation_failure_does_not_wake_sleeping_pool(method, strategy, failure):
    async def scenario():
        pool, handle = make_pool()
        scheduler = pool._scheduler
        await pool.sleep()
        original_abort = scheduler.abort_streams
        scheduler.abort_streams = AsyncMock(side_effect=failure("preparation failed"))
        try:
            with pytest.raises(failure, match="preparation failed"):
                await mutate(pool, method, strategy)
            assert pool.sleeping
            assert scheduler.paused
            assert not scheduler._pause_event.is_set()
            getattr(handle, method).remote.assert_not_awaited()
            handle.pause_generation.remote.assert_not_awaited()
        finally:
            scheduler.abort_streams = original_abort
            await scheduler.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("method,strategy", [
    (method, strategy) for method, strategy in MUTATIONS if strategy != "hotswap"
])
def test_cancellation_after_engine_mutation_starts_does_not_restore_scheduling(method, strategy):
    async def scenario():
        pool, handle = make_pool()
        scheduler = pool._scheduler
        entered = asyncio.Event()
        release = asyncio.Event()

        async def engine_mutation(*args, **kwargs):
            entered.set()
            await release.wait()

        getattr(handle, method).remote.side_effect = engine_mutation
        mutation = asyncio.create_task(mutate(pool, method, strategy))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            mutation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await mutation
            if strategy == "skip":
                assert not scheduler.is_worker_available(0)
            else:
                assert scheduler._paused
            assert not pool._stream_admission_blocked
        finally:
            release.set()
            await asyncio.gather(mutation, return_exceptions=True)
            await scheduler.shutdown()

    asyncio.run(scenario())


def test_completed_but_undelivered_abort_reports_already_terminal():
    async def scenario():
        worker = local_worker()
        pool, handle = make_pool()
        handle.abort_stream.remote = AsyncMock(side_effect=worker.abort_stream)
        scheduler = pool._scheduler
        stream = scheduler.stream_generate("public", "prompt", {"max_tokens": 1})
        stream.worker = scheduler._workers[0]
        stream.worker.active_requests = stream.worker.streaming_requests = 1
        worker.start_stream(stream.attempt_id, "prompt", {"max_tokens": 1}, 20, asdict(StreamLimits()))
        events = worker.stream_events(stream.attempt_id)
        try:
            while True:
                batch = await anext(events)
                if batch[-1]["type"] == "completed":
                    break
                worker.acknowledge_stream(stream.attempt_id, batch[-1]["sequence"])
            assert "public" in scheduler._streams
            assert stream.attempt_id not in worker._engine_streams
            assert (await scheduler.abort("public"))["status"] == "already_terminal"
            assert (await scheduler.abort("unknown"))["status"] == "not_found"
        finally:
            await events.aclose()
            await scheduler.shutdown()

    asyncio.run(scenario())


class _OneEvent:
    def __init__(self, event):
        self.event = event

    async def __anext__(self):
        async def ref():
            return self.event
        return ref()


def _stream_with_terminal_error(scheduler, request_id):
    stream = scheduler.stream_generate(request_id, "prompt")
    stream.worker = scheduler._workers[0]
    stream.worker.active_requests = stream.worker.streaming_requests = 1
    stream.remote_stream = _OneEvent({
        "type": "terminal_error", "code": "context_length_exceeded",
        "context_limit_source": "prompt", "sequence": 0, "version": 1,
    })
    return stream


def test_unconfirmed_cleanup_overrides_terminal_error_and_survives_retirement(monkeypatch):
    async def scenario():
        monkeypatch.setattr("ray.cancel", MagicMock())
        pool, handle = make_pool()
        handle.abort_stream.remote.return_value = {"status": "cleanup_unconfirmed"}
        scheduler = pool._scheduler
        stream = _stream_with_terminal_error(scheduler, "ctx")
        try:
            event = await anext(stream)
            assert event["type"] == "terminal_error"
            assert event["code"] == "cleanup_unconfirmed"
            assert "context_limit_source" not in event
            assert "ctx" not in scheduler._streams
            # The retired record must not turn "unconfirmed" into "already_terminal".
            for _ in range(2):
                assert (await scheduler.abort("ctx"))["status"] == "cleanup_unconfirmed"
                assert (await stream.abort())["status"] == "cleanup_unconfirmed"
            scheduler.mark_worker_available(0)
            assert not scheduler.is_worker_available(0)
        finally:
            await scheduler.shutdown()

    asyncio.run(scenario())


def test_confirmed_cleanup_keeps_terminal_error(monkeypatch):
    async def scenario():
        monkeypatch.setattr("ray.cancel", MagicMock())
        pool, _ = make_pool()
        scheduler = pool._scheduler
        stream = _stream_with_terminal_error(scheduler, "ctx")
        try:
            event = await anext(stream)
            assert event["code"] == "context_length_exceeded"
            assert event["context_limit_source"] == "prompt"
            assert (await scheduler.abort("ctx"))["status"] == "already_terminal"
            assert (await stream.abort())["status"] == "already_terminal"
        finally:
            await scheduler.shutdown()

    asyncio.run(scenario())


def test_unconfirmed_retirement_is_pruned_with_its_record():
    async def scenario():
        pool, _ = make_pool()
        scheduler = pool._scheduler
        scheduler._retire_stream("old", cleanup_unconfirmed=True)
        scheduler._stream_retired["old"] = time.monotonic() - 3601
        try:
            assert (await scheduler.abort("old"))["status"] == "not_found"
            assert "old" not in scheduler._stream_retired_unconfirmed
        finally:
            await scheduler.shutdown()

    asyncio.run(scenario())
