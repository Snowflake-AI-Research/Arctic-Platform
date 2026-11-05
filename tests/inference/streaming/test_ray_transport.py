"""Real Ray transport probes, not ArcticInference streaming acceptance tests."""

import asyncio
import os

import pytest
import ray


@ray.remote(num_cpus=0, max_concurrency=8)
class ControlledProducer:
    def __init__(self):
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.aborted = False

    async def stream(self, fail=False):
        self.started.set()
        try:
            yield {"text": "Hello", "pid": os.getpid()}
            await self.release.wait()
            if self.aborted:
                return
            if fail:
                raise RuntimeError("controlled producer failure")
            yield {"text": " world", "pid": os.getpid()}
        finally:
            self.finished.set()

    async def release_output(self):
        self.release.set()

    async def abort(self):
        self.aborted = True
        self.release.set()

    async def wait_started(self):
        await self.started.wait()
        return True

    async def wait_finished(self):
        await self.finished.wait()
        return True

    async def is_finished(self):
        return self.finished.is_set()


@pytest.fixture(scope="module", autouse=True)
def local_ray():
    assert not ray.is_initialized(), "Run these probes in a separate pytest process"
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


@pytest.fixture
def producer():
    actor = ControlledProducer.remote()
    yield actor
    ray.kill(actor)


async def next_event(stream):
    reference = await asyncio.wait_for(stream.__anext__(), timeout=15)
    return await asyncio.wait_for(reference, timeout=15)


def test_delta_crosses_process_boundary_before_completion(producer):
    async def scenario():
        stream = producer.stream.remote()
        first = await next_event(stream)
        assert first["text"] == "Hello"
        assert first["pid"] != os.getpid()
        assert not await producer.is_finished.remote()
        await producer.release_output.remote()
        second = await next_event(stream)
        assert first["text"] + second["text"] == "Hello world"
        with pytest.raises(StopAsyncIteration):
            await next_event(stream)

    asyncio.run(asyncio.wait_for(scenario(), timeout=40))


def test_abort_rpc_remains_responsive_while_stream_waits(producer):
    async def scenario():
        stream = producer.stream.remote()
        await next_event(stream)
        await asyncio.wait_for(producer.abort.remote(), timeout=10)
        assert await asyncio.wait_for(producer.wait_finished.remote(), timeout=10)
        with pytest.raises(StopAsyncIteration):
            await next_event(stream)

    asyncio.run(asyncio.wait_for(scenario(), timeout=40))


def test_error_after_first_delta_reaches_consumer(producer):
    async def scenario():
        stream = producer.stream.remote(fail=True)
        assert (await next_event(stream))["text"] == "Hello"
        await producer.release_output.remote()
        with pytest.raises(
            ray.exceptions.RayTaskError, match="controlled producer failure"
        ):
            await next_event(stream)

    asyncio.run(asyncio.wait_for(scenario(), timeout=40))


def test_actor_death_does_not_look_like_success(producer):
    async def scenario():
        stream = producer.stream.remote()
        await next_event(stream)
        ray.kill(producer)
        with pytest.raises(ray.exceptions.RayActorError):
            await next_event(stream)

    asyncio.run(asyncio.wait_for(scenario(), timeout=40))


def test_generator_executes_before_consumer_reads(producer):
    async def scenario():
        stream = producer.stream.remote()
        assert await asyncio.wait_for(producer.wait_started.remote(), timeout=15)
        await producer.abort.remote()
        assert await asyncio.wait_for(producer.wait_finished.remote(), timeout=15)
        assert (await next_event(stream))["text"] == "Hello"
        with pytest.raises(StopAsyncIteration):
            await next_event(stream)

    asyncio.run(asyncio.wait_for(scenario(), timeout=40))
