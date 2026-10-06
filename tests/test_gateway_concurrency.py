"""The gateway must sample as many agents at once as it was sized for.

The failure this guards against is invisible from the outside: requests all
succeed, they just queue. ``asyncio.to_thread`` dispatches to the event loop's
default executor, which caps at ``min(32, cpu + 4)`` threads, so a blocking
sampling call silently limits the whole loop to 32 concurrent rollouts however
many replicas the job is running. The symptom is a throughput plateau that
looks like a slow sampler.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from arctic_platform.integrations.harbor.openai_gateway import _ClientBackedPool


class _BlockingClient:
    """A sampler that blocks until released, and records peak overlap."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.inflight = 0
        self.peak = 0
        self._lock = threading.Lock()

    def generate(self, prompts, sampling_params):  # noqa: ARG002
        with self._lock:
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
        self.release.wait(timeout=10)
        with self._lock:
            self.inflight -= 1
        return [{"text": "ok", "token_ids": [1], "finish_reason": "stop"}]


async def _drive(pool: _ClientBackedPool, client: _BlockingClient, n: int) -> int:
    tasks = [
        asyncio.create_task(pool.generate(["p"], {"n": 1}))
        for _ in range(n)
    ]
    # Let them all reach the blocking call before measuring.
    deadline = time.time() + 5
    while client.peak < n and time.time() < deadline:
        await asyncio.sleep(0.01)
    peak = client.peak
    client.release.set()
    await asyncio.gather(*tasks)
    return peak


@pytest.mark.asyncio
async def test_concurrency_is_not_capped_by_the_default_executor() -> None:
    """64 agents must actually sample 64 at a time.

    64 is above the default executor's 32-thread ceiling on any machine, so
    this fails if the call ever goes back to ``asyncio.to_thread``.
    """
    client = _BlockingClient()
    pool = _ClientBackedPool(client, max_inflight=64)

    assert await _drive(pool, client, 64) == 64


@pytest.mark.asyncio
async def test_pool_size_is_the_limit_that_applies() -> None:
    """Sizing the pool below the offered load is what should throttle it, so
    that the bound is the one the caller chose rather than an inherited
    default."""
    client = _BlockingClient()
    pool = _ClientBackedPool(client, max_inflight=4)

    assert await _drive(pool, client, 16) == 4
