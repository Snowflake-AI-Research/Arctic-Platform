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
        self.ack_count = 0

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
        self.ack_count += 1
        if self.ack_delay_s:
            await asyncio.sleep(self.ack_delay_s)
        return self.worker.acknowledge_stream(*args)

    def set_ack_delay(self, seconds):
        self.ack_delay_s = seconds

    def acks(self):
        return self.ack_count

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
        # Deltas from one choice merge while the reader is behind, so overflow
        # needs more open choices than buffer slots.
        stream = driver.stream_generate(
            "model",
            mode,
            "normal" if mode == "overflow" else "blocked",
            {"max_tokens": 64, "n": 8 if mode == "overflow" else 1},
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
        # One hand-over. The 20 undelivered deltas merged into one event.
        assert [event["type"] for event in batch] == [
            "delta", "choice_finished", "usage", "completed"
        ]
        assert batch[0]["text"] == "x" * 20
        assert [event["sequence"] for event in batch] == [0, 1, 2, 3]
        await events.aclose()

    asyncio.run(check())


def test_batch_is_acknowledged_by_its_last_sequence():
    async def check():
        worker = FakeEngineWorker.__ray_metadata__.modified_class()
        # Three choices, engine paused after the first token: three separate
        # deltas (choices never merge) and a stream that is still open.
        worker.worker.start_stream(
            "attempt", "blocked", {"max_tokens": 4, "n": 3}, 20, asdict(StreamLimits())
        )
        session = worker.worker._engine_streams["attempt"]
        while len(session.buffer.events) < 3:
            await asyncio.sleep(0.001)
        events = worker.worker.stream_events("attempt")
        batch = await anext(events)
        assert [event["choice_index"] for event in batch] == [0, 1, 2]
        with pytest.raises(ValueError, match="Unexpected stream acknowledgement"):
            worker.worker.acknowledge_stream("attempt", batch[0]["sequence"])
        assert worker.worker.acknowledge_stream("attempt", batch[-1]["sequence"])
        worker.worker.llm.gate.set()
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


def test_read_buffered_hands_over_the_rest_of_a_batch_without_a_round_trip():
    async def check(driver, pool, actor):
        await actor.set_ack_delay.remote(0.006)  # let a backlog form
        stream = driver.stream_generate("model", "batched-reader", [1, 2], {"max_tokens": 60})
        events = [await anext(stream)]
        saw_batch = False
        while events[-1]["type"] != "completed":
            acks = await actor.acks.remote()
            more = await stream.read_buffered(1000)
            assert await actor.acks.remote() == acks  # no round trip
            saw_batch = saw_batch or bool(more)
            events.extend(more)
            if events[-1]["type"] != "completed":
                events.append(await anext(stream))
        assert saw_batch
        assert [event["sequence"] for event in events] == list(range(len(events)))
        assert events[-2]["completion_tokens"] == 60
        assert await stream.read_buffered(10) == []
        await actor.set_ack_delay.remote(0)

    asyncio.run(exercise(check))


def test_read_buffered_respects_its_limit():
    async def check(driver, pool, actor):
        await actor.set_ack_delay.remote(0.006)
        stream = driver.stream_generate("model", "limited-reader", [1, 2], {"max_tokens": 60})
        await anext(stream)
        while True:
            more = await stream.read_buffered(2)
            assert len(more) <= 2
            if more and more[-1]["type"] == "completed":
                break
            if not more:
                if (await anext(stream))["type"] == "completed":
                    break
        await actor.set_ack_delay.remote(0)

    asyncio.run(exercise(check))


def _delta(index, text):
    return {"type": "delta", "choice_index": index, "text": text}


def _drain_texts(buffer):
    return [(e["type"], e.get("choice_index"), e.get("text")) for e in buffer.drain()]


def test_waiting_deltas_of_one_choice_merge():
    buffer = EventBuffer(StreamLimits())
    for text in ("The", " cat", " sat"):
        buffer.put(_delta(0, text))
    assert _drain_texts(buffer) == [("delta", 0, "The cat sat")]
    assert buffer.bytes == 0


def test_merged_deltas_keep_every_token_id():
    # Usage counts tokens, so a merge that kept only the text would under-report them.
    buffer = EventBuffer(StreamLimits())
    for text, token_ids in (("The", [791]), (" cat", [8415]), (" sat", [7731, 13])):
        buffer.put({**_delta(0, text), "token_ids": token_ids})
    [event] = buffer.drain()
    assert event["text"] == "The cat sat"
    assert event["token_ids"] == [791, 8415, 7731, 13]


def test_choices_never_merge_with_each_other():
    buffer = EventBuffer(StreamLimits())
    for text in ("a", "b"):
        buffer.put(_delta(0, text))
        buffer.put(_delta(1, text.upper()))
    assert _drain_texts(buffer) == [("delta", 0, "ab"), ("delta", 1, "AB")]


def test_delivered_delta_is_not_reopened():
    buffer = EventBuffer(StreamLimits())
    buffer.put(_delta(0, "The"))
    assert _drain_texts(buffer) == [("delta", 0, "The")]
    buffer.put(_delta(0, " cat"))
    assert _drain_texts(buffer) == [("delta", 0, " cat")]


def test_merging_never_crosses_a_later_event():
    buffer = EventBuffer(StreamLimits())
    buffer.put(_delta(0, "a"))
    buffer.put({"type": "choice_finished", "choice_index": 0, "finish_reason": "stop"})
    buffer.put(_delta(0, "b"))
    buffer.put(_delta(1, "c"))
    buffer.put({"type": "usage", "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})
    buffer.put(_delta(1, "d"))
    assert [(t, i, x) for t, i, x in _drain_texts(buffer)] == [
        ("delta", 0, "a"),
        ("choice_finished", 0, None),
        ("delta", 0, "b"),
        ("delta", 1, "c"),
        ("usage", None, None),
        ("delta", 1, "d"),
    ]


def test_merge_that_would_exceed_event_limit_starts_a_new_event():
    limits = StreamLimits(max_event_bytes=120, max_buffer_bytes=4096)
    buffer = EventBuffer(limits)
    buffer.put(_delta(0, "x" * 40))
    buffer.put(_delta(0, "y" * 40))
    events = buffer.drain()
    assert [e["text"] for e in events] == ["x" * 40, "y" * 40]


def test_buffer_bytes_track_merged_sizes():
    from arctic_platform.inference.server.streaming import event_size

    buffer = EventBuffer(StreamLimits())
    for text in ("The", " cat", " sat"):
        buffer.put(_delta(0, text))
    buffer.put(_delta(1, "hi"))
    assert buffer.bytes == event_size(_delta(0, "The cat sat")) + event_size(_delta(1, "hi"))


def test_slow_reader_no_longer_overflows_on_a_long_answer():
    # Before merging, 1,000 undelivered tokens needed 1,000 slots.
    buffer = EventBuffer(StreamLimits(max_buffer_events=2))
    for _ in range(1000):
        buffer.put(_delta(0, "x"))
    assert _drain_texts(buffer) == [("delta", 0, "x" * 1000)]


def test_many_choices_behind_a_slow_reader_complete():
    # 8 choices x 200 tokens is 1,600 deltas; with slow acks a 20-event buffer
    # overflowed. Merged, each choice needs one slot while behind; the end of
    # the stream needs 2n + 2 = 18 (finish events never merge).
    async def check(driver, pool, actor):
        await actor.set_ack_delay.remote(0.006)
        events = [
            event
            async for event in driver.stream_generate(
                "model",
                "many-choices",
                [1, 2],
                {"max_tokens": 200, "n": 8},
                limits=StreamLimits(max_buffer_events=20),
            )
        ]
        assert events[-1]["type"] == "completed", events[-1]
        assert events[-2]["completion_tokens"] == 1600
        text = {}
        for event in events:
            if event["type"] == "delta":
                text[event["choice_index"]] = text.get(event["choice_index"], "") + event["text"]
        assert text == {index: "x" * 200 for index in range(8)}
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
