"""Actual Driver/Pool/Scheduler/Worker stream path; only model output is fake."""

import asyncio
from dataclasses import asdict
import os
from types import SimpleNamespace

import pytest
import ray

from cpu_support import load_library

if os.environ.get("ARCTIC_RUN_GPU_TESTS") == "1":
    pytest.skip(
        "CPU fake-engine harness must run separately from GPU tests",
        allow_module_level=True,
    )

load_library()
from arctic_platform.inference.server.multi_model import Driver
from arctic_platform.inference.server.replica_pool import ReplicaPool
from arctic_platform.inference.server.scheduler import Scheduler
from arctic_platform.inference.server.streaming import (
    EventBuffer,
    StreamError,
    StreamLimits,
    validate_request,
)
from arctic_platform.inference.server.worker import InferenceWorker


class FakeEngine:
    def __init__(self):
        self.active = set()
        self.aborted = []
        self.gate = asyncio.Event()
        self.calls = []
        self.model_config = SimpleNamespace(max_model_len=131072)

    async def generate(self, prompt, params, request_id, **kwargs):
        self.calls.append((prompt, params, kwargs))
        self.active.add(request_id)
        try:
            if isinstance(prompt, str) and prompt in {
                "context-error",
                "legacy-context-error",
                "validation-error",
            }:
                from vllm.exceptions import VLLMValidationError

                if prompt == "context-error":
                    raise VLLMValidationError(
                        "This model's maximum context length is 8 tokens. However, "
                        "you requested 0 output tokens and your prompt contains at "
                        "least 9 input tokens, for a total of at least 9 tokens. "
                        "Please reduce the length of the input prompt or the number "
                        "of requested output tokens.",
                        parameter="input_tokens",
                        value=9,
                    )
                if prompt == "legacy-context-error":
                    raise VLLMValidationError(
                        "The decoder prompt (length 9) is longer than the maximum "
                        "model length of 8."
                    )
                raise VLLMValidationError(
                    "sensitive unsupported sampling parameter"
                )
            for step in range(params["max_tokens"]):
                if step == 1 and prompt == "blocked":
                    await self.gate.wait()
                if step == 1 and prompt == "failure":
                    raise RuntimeError("sensitive engine details")
                if step == 1 and prompt == "truncated":
                    return
                await asyncio.sleep(0.002)
                choices = [
                    SimpleNamespace(
                        index=index,
                        text="x",
                        token_ids=[step],
                        finish_reason="length"
                        if step
                        == (
                            index
                            if prompt == "different-lengths"
                            else params["max_tokens"] - 1
                        )
                        else None,
                    )
                    for index in range(params["n"])
                    if prompt != "different-lengths" or step <= index
                ]
                yield SimpleNamespace(prompt_token_ids=[1, 2], outputs=choices)
        finally:
            self.active.discard(request_id)

    async def abort(self, request_id):
        self.aborted.append(request_id)


@ray.remote(num_cpus=0, max_concurrency=16)
class FakeEngineWorker:
    def __init__(self):
        load_library()
        from arctic_platform.inference.server.worker import InferenceWorker, WorkerLifecycleState

        worker_type = InferenceWorker.__ray_metadata__.modified_class
        self.worker = worker_type()
        self.worker.state = WorkerLifecycleState.READY
        self.worker.llm = FakeEngine()
        self.worker._stream_sampling_params = lambda params: params
        self.registration_gate = asyncio.Event()
        self.registration_started = asyncio.Event()
        self.delay_registration = False
        self.legacy_gate = asyncio.Event()
        self.legacy_calls = 0
        self.registration_remaining_s = None
        self.abort_calls = 0
        self.ack_delay_s = 0

    def fail_first_abort(self):
        async def abort(request_id):
            self.abort_calls += 1
            if self.abort_calls == 1:
                raise RuntimeError("engine abort failed")

        self.worker.llm.abort = abort

    async def generate(self, prompt, sampling_params):
        self.legacy_calls += 1
        await self.legacy_gate.wait()
        return {"text": "legacy result", "token_ids": [1]}

    def release_legacy(self):
        self.legacy_gate.set()

    def set_replica_id(self, index):
        return None

    async def start_stream(self, *args):
        self.registration_remaining_s = args[3]
        self.registration_started.set()
        if self.delay_registration:
            await self.registration_gate.wait()
        return self.worker.start_stream(*args)

    def delay_start(self):
        self.delay_registration = True

    async def wait_registration(self):
        await self.registration_started.wait()

    def release_registration(self):
        self.registration_gate.set()

    async def stream_events(self, attempt_id):
        async for event in self.worker.stream_events(attempt_id):
            yield event

    async def acknowledge_stream(self, *args):
        if self.ack_delay_s:
            await asyncio.sleep(self.ack_delay_s)
        return self.worker.acknowledge_stream(*args)

    def set_ack_delay(self, seconds):
        self.ack_delay_s = seconds

    async def abort_stream(self, *args):
        return await self.worker.abort_stream(*args)

    async def lifecycle_change(self):
        await self.worker.abort_all_streams()

    def release(self):
        self.worker.llm.gate.set()

    def stats(self):
        return {
            "active": len(self.worker.llm.active),
            "aborted": len(self.worker.llm.aborted),
            "sessions": len(getattr(self.worker, "_engine_streams", {})),
            "calls": self.worker.llm.calls,
            "legacy_calls": self.legacy_calls,
            "remaining_s": self.registration_remaining_s,
            "abort_calls": self.abort_calls,
            "cleanup_failed": getattr(self.worker, "_stream_cleanup_failed", False),
        }


@pytest.fixture(scope="module", autouse=True)
def runtime():
    ray.init(
        address="local",
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        _node_ip_address="127.0.0.1",
        object_store_memory=80 * 1024 * 1024,
    )
    yield
    ray.shutdown()


async def exercise(check):
    actor = FakeEngineWorker.remote()
    await actor.stats.remote()
    pool = ReplicaPool()
    pool._scheduler = Scheduler([actor], dynamic_concurrency=False)
    driver = Driver()
    driver._pools["model"] = pool
    try:
        await asyncio.wait_for(check(driver, pool, actor), 30)
    finally:
        await pool._scheduler.shutdown()
        ray.kill(actor)


@pytest.mark.parametrize("choices", [1, 2, 8])
def test_complete_library_path(choices):
    async def check(driver, pool, actor):
        events = [
            event
            async for event in driver.stream_generate(
                "model", "request", [10, 20], {"n": choices, "max_tokens": 3}
            )
        ]
        assert events[-1]["type"] == "completed"
        assert events[-2]["completion_tokens"] == choices * 3
        assert events[-2]["prompt_tokens"] == 2
        assert events[-2]["total_tokens"] == 2 + choices * 3
        assert [event["sequence"] for event in events] == list(range(len(events)))
        assert {
            event["choice_index"]
            for event in events
            if event["type"] == "choice_finished"
        } == set(range(choices))
        assert pool._scheduler._workers[0].active_requests == 0
        assert (await driver.abort("model", "request"))["status"] == "already_terminal"
        assert (await actor.stats.remote())["calls"][0][0] == {
            "prompt_token_ids": [10, 20]
        }
        [record] = pool._scheduler._request_records.drain()
        assert record.streaming is True
        assert record.first_delta_time is not None
        assert record.arrival_time <= record.submitted_time
        assert record.submitted_time <= record.first_delta_time
        assert record.first_delta_time <= record.completion_time
        assert record.prompt_len == 2
        assert record.generation_len == choices * 3

    asyncio.run(exercise(check))


def test_early_delivery_abort_and_unrelated_request():
    async def check(driver, pool, actor):
        stream = driver.stream_generate(
            "model", "blocked", "blocked", {"max_tokens": 8, "n": 2}
        )
        assert (await anext(stream))["text"] == "x"
        assert (await actor.stats.remote())["active"] == 1
        other = [
            event
            async for event in driver.stream_generate(
                "model", "other", "normal", {"max_tokens": 2}
            )
        ]
        assert other[-1]["type"] == "completed"
        assert (await driver.abort("model", "blocked"))["status"] == "aborted"
        assert (await actor.stats.remote())["active"] == 0
        assert (await actor.stats.remote())["aborted"] > 0
        with pytest.raises(StreamError):
            await anext(stream)
        with pytest.raises(ValueError, match="Duplicate"):
            driver.stream_generate("model", "blocked", "normal")

    asyncio.run(exercise(check))


def test_cancel_before_dispatch():
    async def check(driver, pool, actor):
        stream = driver.stream_generate("model", "queued", "blocked")
        await driver.abort("model", "queued")
        assert (await actor.stats.remote())["calls"] == []
        with pytest.raises(StreamError):
            await anext(stream)

    asyncio.run(exercise(check))


@pytest.mark.parametrize(
    "prompt,code,context_limit_source",
    [
        ("failure", "engine_error", None),
        ("truncated", "incomplete_engine_output", None),
        ("context-error", "context_length_exceeded", "prompt"),
        ("legacy-context-error", "context_length_exceeded", "prompt"),
        ("validation-error", "engine_error", None),
    ],
)
def test_engine_errors_are_terminal_and_sanitized(
    prompt, code, context_limit_source
):
    async def check(driver, pool, actor):
        stream = driver.stream_generate("model", "error", prompt, {"max_tokens": 3})
        events = []
        async for event in stream:
            events.append(event)
            if event["type"] == "terminal_error":
                break
        assert events[-1]["code"] == code
        assert events[-1].get("context_limit_source") == context_limit_source
        assert not any(event["type"] == "completed" for event in events)
        assert "sensitive" not in str(events)

    asyncio.run(exercise(check))


def test_requested_output_must_fit_model_context():
    async def check(driver, pool, actor):
        stream = driver.stream_generate(
            "model", "context", "normal", {"max_tokens": 131071}
        )
        events = [event async for event in stream]
        assert events[-1]["type"] == "terminal_error"
        assert events[-1]["code"] == "context_length_exceeded"
        assert events[-1]["context_limit_source"] == "completion_budget"
        assert not any(event["type"] == "delta" for event in events)

    asyncio.run(exercise(check))


@pytest.mark.parametrize(
    "code,context_limit_source",
    [
        ("context_length_exceeded", None),
        ("context_length_exceeded", "future_source"),
        ("engine_error", "prompt"),
    ],
)
def test_context_limit_source_validation(code, context_limit_source):
    with pytest.raises(ValueError):
        StreamError(code, context_limit_source=context_limit_source)


@pytest.mark.parametrize(
    "mode", ["stall", "deadline", "overflow", "lifecycle", "shutdown", "death"]
)
def test_cleanup_paths(mode):
    async def check(driver, pool, actor):
        limits = StreamLimits(
            timeout_s=0.7 if mode == "deadline" else 10,
            stall_timeout_s=0.7 if mode == "stall" else 10,
            max_buffer_events=4 if mode == "overflow" else 128,
        )
        stream = driver.stream_generate(
            "model",
            mode,
            "normal" if mode == "overflow" else "blocked",
            {"max_tokens": 64},
            limits=limits,
        )
        await anext(stream)
        if mode == "lifecycle":
            await actor.lifecycle_change.remote()
        elif mode == "shutdown":
            await pool._scheduler.shutdown()
        elif mode == "death":
            ray.kill(actor)
        else:
            await asyncio.sleep(1)
        events = []
        try:
            async for event in stream:
                events.append(event)
                if event["type"] == "terminal_error":
                    break
        except (StreamError, ray.exceptions.RayError):
            pass
        assert not any(event["type"] == "completed" for event in events)
        if mode != "death":
            assert (await actor.stats.remote())["active"] == 0
        assert pool._scheduler._workers[0].active_requests == 0

    asyncio.run(exercise(check))


@pytest.mark.parametrize(
    "params",
    [
        {"n": 0},
        {"n": 9},
        {"n": True},
        {"max_tokens": None},
        {"top_p": 0},
        {"temperature": float("nan")},
        {"frequency_penalty": -2.1},
        {"frequency_penalty": True},
        {"presence_penalty": 2.1},
        {"presence_penalty": float("nan")},
        {"stop": []},
        {"stop": ["a"] * 5},
        {"stop": ""},
        {"messages": []},
        {"logprobs": 2},
    ],
)
def test_parameter_rejection(params):
    with pytest.raises(ValueError):
        validate_request("prompt", params)


def test_backlog_is_delivered_in_one_batch():
    async def check():
        worker = FakeEngineWorker.__ray_metadata__.modified_class()
        limits = StreamLimits()
        worker.worker.start_stream("attempt", [1, 2], {"max_tokens": 20}, 20, asdict(limits))
        session = worker.worker._engine_streams["attempt"]
        await asyncio.wait_for(session.pump, 5)  # engine done before anyone reads
        events = worker.worker.stream_events("attempt")
        batch = await anext(events)
        # 20 deltas, choice_finished, usage, completed: one hand-over, not 23.
        assert [event["sequence"] for event in batch] == list(range(23))
        assert batch[-1]["type"] == "completed"
        await events.aclose()

    asyncio.run(check())


def test_batch_is_acknowledged_by_its_last_sequence():
    async def check():
        worker = FakeEngineWorker.__ray_metadata__.modified_class()
        worker.worker.start_stream(
            "attempt", [1, 2], {"max_tokens": 4}, 20, asdict(StreamLimits())
        )
        session = worker.worker._engine_streams["attempt"]
        while len(session.buffer.events) < 3:
            await asyncio.sleep(0.001)
        events = worker.worker.stream_events("attempt")
        batch = await anext(events)
        assert len(batch) >= 3
        with pytest.raises(ValueError, match="Unexpected stream acknowledgement"):
            worker.worker.acknowledge_stream("attempt", batch[0]["sequence"])
        assert worker.worker.acknowledge_stream("attempt", batch[-1]["sequence"])
        await events.aclose()

    asyncio.run(check())


def test_slow_round_trips_do_not_overflow_a_small_buffer():
    # The engine makes a token every 2 ms and each acknowledgement takes 6 ms.
    # Delivered one event per round trip, a 16-event buffer overflows within
    # ~25 tokens; delivered in batches, each round trip takes the backlog.
    async def check(driver, pool, actor):
        await actor.set_ack_delay.remote(0.006)
        limits = StreamLimits(max_buffer_events=16)
        events = [
            event
            async for event in driver.stream_generate(
                "model", "slow-consumer", [1, 2], {"max_tokens": 100}, limits=limits
            )
        ]
        assert events[-1]["type"] == "completed", events[-1]
        assert events[-2]["completion_tokens"] == 100
        assert [event["sequence"] for event in events] == list(range(len(events)))
        await actor.set_ack_delay.remote(0)

    asyncio.run(exercise(check))


def test_bounded_buffer():
    async def check():
        limits = StreamLimits(
            max_buffer_events=2, max_buffer_bytes=1024, max_event_bytes=512
        )
        buffer = EventBuffer(limits)
        buffer.put({"text": "a"})
        buffer.put({"text": "b"})
        with pytest.raises(StreamError, match="buffer_overflow"):
            buffer.put({"text": "c"})
        assert buffer.peak_events == 2
        assert buffer.peak_bytes <= limits.max_buffer_bytes
        buffer.fail("cancelled")
        assert buffer.bytes == 0
        with pytest.raises(StreamError, match="cancelled"):
            await buffer.get()

    asyncio.run(check())


def test_abort_during_registration():
    async def check(driver, pool, actor):
        await actor.delay_start.remote()
        stream = driver.stream_generate("model", "race", "blocked")
        reader = asyncio.create_task(anext(stream))
        await actor.wait_registration.remote()
        abort = asyncio.create_task(driver.abort("model", "race"))
        await asyncio.sleep(0)
        await actor.release_registration.remote()
        assert (await abort)["status"] == "aborted"
        with pytest.raises(StreamError):
            await reader
        assert (await actor.stats.remote())["active"] == 0

    asyncio.run(exercise(check))


def test_queued_abort_does_not_dispatch():
    async def check(driver, pool, actor):
        pool._scheduler.pause()
        stream = driver.stream_generate("model", "queued", "blocked")
        reader = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.05)
        await driver.abort("model", "queued")
        pool._scheduler.resume()
        with pytest.raises(StreamError):
            await reader
        assert (await actor.stats.remote())["calls"] == []

    asyncio.run(exercise(check))


def test_early_context_exit_closes_generation():
    async def check(driver, pool, actor):
        async with driver.stream_generate("model", "close", "blocked") as stream:
            await anext(stream)
        assert (await actor.stats.remote())["active"] == 0
        assert not pool._scheduler._streams

    asyncio.run(exercise(check))


def test_model_scoping_and_wrong_model_abort():
    async def check(driver, pool, actor):
        stream = driver.stream_generate("model", "shared-id", "blocked")
        await anext(stream)
        assert (await driver.abort("wrong-model", "shared-id"))["status"] == "not_found"
        assert (await actor.stats.remote())["active"] == 1
        await stream.aclose()

    asyncio.run(exercise(check))


def test_worker_reclaims_stream_without_reader():
    async def check(driver, pool, actor):
        from dataclasses import asdict

        limits = StreamLimits(stall_timeout_s=0.3)
        await actor.start_stream.remote(
            "orphan", "blocked", {"max_tokens": 8}, 20, asdict(limits)
        )
        await asyncio.sleep(0.8)
        stats = await actor.stats.remote()
        assert stats["active"] == 0
        assert stats["sessions"] == 0
        assert stats["aborted"] > 0

    asyncio.run(exercise(check))


def test_byte_and_event_size_limits():
    async def check():
        limits = StreamLimits(max_buffer_bytes=100, max_event_bytes=80)
        buffer = EventBuffer(limits)
        with pytest.raises(StreamError, match="event_too_large"):
            buffer.put({"text": "z" * 100})
        buffer.put({"text": "z" * 50})
        with pytest.raises(StreamError, match="buffer_overflow"):
            buffer.put({"text": "z" * 50})
        assert buffer.peak_bytes <= 100

    asyncio.run(check())


def test_independent_choice_finishes():
    async def check(driver, pool, actor):
        events = [
            event
            async for event in driver.stream_generate(
                "model", "different", "different-lengths", {"n": 2, "max_tokens": 3}
            )
        ]
        assert events[-1]["type"] == "completed"
        assert events[-2]["completion_tokens"] == 3
        assert (
            "".join(
                event["text"]
                for event in events
                if event["type"] == "delta" and event["choice_index"] == 0
            )
            == "x"
        )
        assert (
            "".join(
                event["text"]
                for event in events
                if event["type"] == "delta" and event["choice_index"] == 1
            )
            == "xx"
        )

    asyncio.run(exercise(check))


def test_engine_parameters_and_adapter_are_preserved():
    async def check():
        from arctic_platform.inference.server.worker import WorkerLifecycleState

        worker = InferenceWorker.__ray_metadata__.modified_class()
        worker.state = WorkerLifecycleState.READY
        worker.llm = FakeEngine()
        worker._stream_sampling_params = lambda params: params
        worker._active_lora_request = lambda: "active-adapter"
        from dataclasses import asdict

        params = {
            "n": 2,
            "temperature": 0.7,
            "top_p": 0.8,
            "frequency_penalty": -0.2,
            "presence_penalty": 0.3,
            "max_tokens": 3,
            "stop": ["END"],
            "seed": 12,
        }
        worker.start_stream(
            "attempt", [1, 2], params, 20, asdict(StreamLimits())
        )
        events = worker.stream_events("attempt")
        async for batch in events:
            if batch[-1]["type"] != "completed":
                worker.acknowledge_stream("attempt", batch[-1]["sequence"])
        prepared, received, kwargs = worker.llm.calls[0]
        assert prepared == {"prompt_token_ids": [1, 2]}
        assert received == params
        assert kwargs["lora_request"] == "active-adapter"

    asyncio.run(check())


def test_legacy_worker_still_returns_final_output():
    async def check():
        worker = InferenceWorker.__ray_metadata__.modified_class()
        worker.llm = FakeEngine()
        output = await worker._generate_once(
            "normal", {"n": 1, "max_tokens": 3}, "legacy"
        )
        assert output.outputs[0].token_ids == [2]
        assert output.outputs[0].finish_reason == "length"

    asyncio.run(check())


def test_lifecycle_mutation_blocks_new_stream_registration():
    async def check():
        from dataclasses import asdict
        from arctic_platform.inference.server.worker import WorkerLifecycleState

        worker = InferenceWorker.__ray_metadata__.modified_class()
        worker.state = WorkerLifecycleState.READY
        worker.llm = FakeEngine()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def collective(*args, **kwargs):
            entered.set()
            await release.wait()
            return [{}]

        worker.llm.collective_rpc = collective
        mutation = asyncio.create_task(worker.sync_weights("localhost", 1, 0, 1))
        await entered.wait()
        with pytest.raises(RuntimeError, match="not ready"):
            worker.start_stream(
                "during-update", "normal", {}, 20, asdict(StreamLimits())
            )
        release.set()
        await mutation
        assert worker._stream_lifecycle_depth == 0

    asyncio.run(check())


def test_failed_abort_quarantines_worker():
    async def check():
        from dataclasses import asdict
        from arctic_platform.inference.server.worker import WorkerLifecycleState

        worker = InferenceWorker.__ray_metadata__.modified_class()
        worker.state = WorkerLifecycleState.READY
        worker.llm = FakeEngine()
        worker._stream_sampling_params = lambda params: params

        async def broken_abort(request_id):
            raise RuntimeError("engine unreachable")

        worker.llm.abort = broken_abort
        worker.start_stream(
            "broken", "blocked", {}, 20, asdict(StreamLimits())
        )
        result = await worker.abort_stream("broken")
        assert result["status"] == "cleanup_unconfirmed"
        with pytest.raises(RuntimeError, match="not ready"):
            worker.start_stream(
                "next", "normal", {}, 20, asdict(StreamLimits())
            )
        assert not worker._engine_streams

    asyncio.run(check())
