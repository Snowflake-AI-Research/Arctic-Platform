"""Opt-in real-engine Driver streaming acceptance tests."""

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("ARCTIC_RUN_GPU_TESTS") != "1",
        reason="Set ARCTIC_RUN_GPU_TESTS=1 in an authorized GPU environment",
    ),
]


def model_directory():
    model_path = os.environ.get("ARCTIC_TEST_MODEL_PATH")
    assert model_path, "Set ARCTIC_TEST_MODEL_PATH to pre-downloaded model weights"
    assert Path(model_path).is_absolute() and Path(model_path).is_dir()
    return model_path


async def with_driver(check):
    from arctic_platform.inference.server.config import ModelConfig
    from arctic_platform.inference.server.multi_model import Driver
    import ray
    import torch

    assert torch.cuda.is_available(), "A supported CUDA GPU is required"
    assert not ray.is_initialized(), "Run GPU tests in a dedicated process"
    ray.init(address="local", include_dashboard=False)
    driver = Driver()
    try:
        config = ModelConfig(
            model=model_directory(),
            tensor_parallel_size=1,
            max_model_len=512,
            max_num_seqs=4,
            gpu_memory_utilization=0.5,
            trust_remote_code=False,
        )
        await asyncio.wait_for(
            driver.initialize(config, model_id="stream-test", num_replicas=1),
            timeout=300,
        )
        await asyncio.wait_for(check(driver), timeout=120)
    finally:
        try:
            await asyncio.wait_for(driver.shutdown(), timeout=60)
        finally:
            ray.shutdown()


def test_legacy_driver_generate():
    async def check(driver):
        results = await driver.generate(
            "The capital of France is",
            {"temperature": 0.0, "max_tokens": 16},
            model_id="stream-test",
        )
        assert len(results) == 1
        assert isinstance(results[0]["text"], str)
        assert 0 < len(results[0]["token_ids"]) <= 16
        assert results[0]["finish_reason"] in {"stop", "length"}

    asyncio.run(with_driver(check))


async def collect(driver, prompt, params):
    return [
        event
        async for event in driver.stream_generate(
            "stream-test", uuid4().hex, prompt, params
        )
    ]


def test_logit_bias_forces_and_bans_a_token():
    async def check(driver):
        prompt = "The capital of France is"
        greedy = {"temperature": 0.0, "max_tokens": 1}

        def first_token(events):
            assert events[-1]["type"] == "completed", events[-1]
            return next(e for e in events if e["type"] == "delta")["token_ids"][0]

        baseline = first_token(await collect(driver, prompt, greedy))
        other = first_token(await collect(driver, "Hello", greedy))
        if other == baseline:
            other = first_token(await collect(driver, "1, 2, 3,", greedy))
        assert other != baseline
        forced = await collect(
            driver, prompt, {**greedy, "logit_bias": {str(other): 100}}
        )
        assert first_token(forced) == other
        banned = await collect(driver, prompt, {**greedy, "logit_bias": {baseline: -100}})
        assert first_token(banned) != baseline
        out_of_vocab = await collect(
            driver, prompt, {**greedy, "logit_bias": {2**31 - 1: 1}}
        )
        assert out_of_vocab[-1]["type"] == "terminal_error"
        assert out_of_vocab[-1]["code"] == "invalid_sampling_params"

    asyncio.run(with_driver(check))


@pytest.mark.parametrize("choice_count", [1, 2])
def test_driver_stream_contract(choice_count):
    from arctic_platform.inference.server.multi_model import Driver

    assert callable(getattr(Driver, "stream_generate", None)), (
        "Streaming API not implemented"
    )
    assert callable(getattr(Driver, "abort", None)), (
        "Explicit abort API not implemented"
    )

    async def check(driver):
        request_id = uuid4().hex
        events = []
        try:
            async for event in driver.stream_generate(
                model_id="stream-test",
                request_id=request_id,
                prompt="The capital of France is",
                sampling_params={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.1,
                    "max_tokens": 16,
                    "n": choice_count,
                },
            ):
                events.append(event)
        finally:
            await driver.abort(model_id="stream-test", request_id=request_id)
        assert events
        assert events[-1]["type"] == "completed"
        assert sum(event["type"] == "completed" for event in events) == 1
        assert not any(event["type"] == "terminal_error" for event in events)
        finishes = [event for event in events if event["type"] == "choice_finished"]
        assert len(finishes) == choice_count
        assert {event["choice_index"] for event in finishes} == set(range(choice_count))
        assert all(event["finish_reason"] in {"stop", "length"} for event in finishes)
        assert sum(event["type"] == "usage" for event in events) == 1
        assert events[-2]["type"] == "usage"
        assert any(event["type"] == "delta" and event["text"] for event in events)
        usage = events[-2]
        assert usage["completion_tokens"] == sum(
            len(event["token_ids"]) for event in events if event["type"] == "delta"
        )
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )
        metrics = await driver.drain_metrics(model_id="stream-test")
        [record] = metrics["requests"]
        assert record["streaming"] is True
        assert record["first_delta_time"] is not None
        assert record["submitted_time"] <= record["first_delta_time"]
        assert record["first_delta_time"] <= record["completion_time"]
        assert record["prompt_len"] == usage["prompt_tokens"]
        assert record["generation_len"] == usage["completion_tokens"]

    asyncio.run(with_driver(check))


def test_driver_abort_reclaims_engine_requests():
    async def check(driver):
        stream = driver.stream_generate(
            "stream-test",
            uuid4().hex,
            "Count from one upwards:",
            {"max_tokens": 256, "n": 2},
        )
        first = await anext(stream)
        assert first["type"] == "delta"
        result = await driver.abort("stream-test", stream.request_id)
        assert result["status"] == "aborted"
        worker = driver._get_pool("stream-test")._workers[0]
        for attempt in range(100):
            stats = await worker.streaming_status.remote()
            if stats["engine_unfinished_requests"] == 0:
                break
            await asyncio.sleep(0.1)
        assert stats["engine_unfinished_requests"] == 0
        assert not stats["cleanup_failed"]
        followup = [
            event
            async for event in driver.stream_generate(
                "stream-test", uuid4().hex, "Hello", {"max_tokens": 8}
            )
        ]
        assert followup[-1]["type"] == "completed"

    asyncio.run(with_driver(check))


def test_structured_output_always_matches_the_schema():
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "population": {"type": "integer"},
        },
        "required": ["city", "population"],
        "additionalProperties": False,
    }

    async def check(driver):
        runs = await asyncio.gather(
            *(
                collect(
                    driver,
                    "Describe a city as JSON:",
                    {
                        "temperature": 0.7,
                        "max_tokens": 256,
                        "seed": seed,
                        "structured_output": {"json": schema},
                    },
                )
                for seed in range(20)
            )
        )
        for events in runs:
            assert events[-1]["type"] == "completed", events[-1]
            finish = next(e for e in events if e["type"] == "choice_finished")
            assert finish["finish_reason"] == "stop"
            value = json.loads(
                "".join(e["text"] for e in events if e["type"] == "delta")
            )
            assert value.keys() == {"city", "population"}
            assert isinstance(value["city"], str)
            assert type(value["population"]) is int

    asyncio.run(with_driver(check))


def test_slow_consumer_is_bounded_and_reclaimed():
    async def check(driver):
        from arctic_platform.inference.server.streaming import StreamError, StreamLimits

        limits = StreamLimits(stall_timeout_s=1, max_buffer_events=4)
        stream = driver.stream_generate(
            "stream-test",
            uuid4().hex,
            "Count from one to a thousand:",
            {"max_tokens": 256, "n": 2},
            limits=limits,
        )
        first = await anext(stream)
        assert first["type"] in {"delta", "terminal_error"}
        await asyncio.sleep(2)
        worker = driver._get_pool("stream-test")._workers[0]
        stats = await worker.streaming_status.remote()
        assert stats["buffered_events"] <= limits.max_buffer_events
        assert stats["buffered_bytes"] <= limits.max_buffer_bytes
        try:
            remaining = [event async for event in stream]
            assert not any(event["type"] == "completed" for event in remaining)
        except StreamError:
            pass
        await stream.aclose()
        stats = await worker.streaming_status.remote()
        assert stats["engine_unfinished_requests"] == 0

    asyncio.run(with_driver(check))


def test_driver_stop_text_is_excluded():
    async def check(driver):
        prompt = "The capital of France is"
        params = {"temperature": 0.0, "max_tokens": 32, "seed": 1}
        baseline = [
            event
            async for event in driver.stream_generate(
                "stream-test", uuid4().hex, prompt, params
            )
        ]
        text = "".join(event["text"] for event in baseline if event["type"] == "delta")
        assert len(text) >= 4, "Use a model that produces at least four characters"
        stop = text[:4]
        events = [
            event
            async for event in driver.stream_generate(
                "stream-test", uuid4().hex, prompt, {**params, "stop": stop}
            )
        ]
        output = "".join(event["text"] for event in events if event["type"] == "delta")
        assert stop not in output
        assert events[-1]["type"] == "completed"
        assert (
            next(event for event in events if event["type"] == "choice_finished")[
                "finish_reason"
            ]
            == "stop"
        )

    asyncio.run(with_driver(check))


def test_context_limit_errors_are_classified():
    async def check(driver):
        cases = [
            ([1] * 513, 1, "prompt"),
            ([1] * 500, 16, "completion_budget"),
        ]
        for prompt, max_tokens, context_limit_source in cases:
            events = [
                event
                async for event in driver.stream_generate(
                    "stream-test",
                    uuid4().hex,
                    prompt,
                    {"temperature": 0.0, "max_tokens": max_tokens},
                )
            ]
            assert events[-1]["type"] == "terminal_error"
            assert events[-1]["code"] == "context_length_exceeded"
            assert events[-1]["context_limit_source"] == context_limit_source
            assert not any(event["type"] == "delta" for event in events)
            assert not any(event["type"] == "completed" for event in events)

        followup = [
            event
            async for event in driver.stream_generate(
                "stream-test", uuid4().hex, "Hello", {"max_tokens": 8}
            )
        ]
        assert followup[-1]["type"] == "completed"

    asyncio.run(with_driver(check))
