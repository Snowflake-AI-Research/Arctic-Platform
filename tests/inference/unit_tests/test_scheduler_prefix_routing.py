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

"""Tests for affinity routing in the scheduler."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from arctic_platform.inference.server.scheduler import (
    Scheduler,
    _compute_prefix_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scheduler(n: int, capacity: int = 64, prefix_routing: bool = True) -> Scheduler:
    return Scheduler([MagicMock() for _ in range(n)], capacity, prefix_routing=prefix_routing)


def _route(sch: Scheduler, prompt="hello", routing_key=None, strict=False) -> int | None:
    """Route one request and count it against its worker, as ``_take`` does."""
    req = sch._request(prompt, {}, None, routing_key, strict)
    ws = sch._take(req)
    return None if ws is None else req.worker_idx


# ---------------------------------------------------------------------------
# _compute_prefix_hash
# ---------------------------------------------------------------------------

class TestComputePrefixHash:
    def test_identical_strings_same_hash(self):
        assert _compute_prefix_hash("abc") == _compute_prefix_hash("abc")

    def test_identical_token_lists_same_hash(self):
        assert _compute_prefix_hash([1, 2, 3]) == _compute_prefix_hash([1, 2, 3])

    def test_full_hash_differs_with_different_suffix(self):
        base = "A" * 100
        assert _compute_prefix_hash(base + "X") != _compute_prefix_hash(base + "Y")

    def test_different_prefix_different_hash(self):
        assert _compute_prefix_hash("hello world") != _compute_prefix_hash("goodbye world")

    def test_token_full_hash_differs_with_different_suffix(self):
        base = list(range(100))
        assert _compute_prefix_hash(base + [999]) != _compute_prefix_hash(base + [888])

    def test_empty_string(self):
        assert isinstance(_compute_prefix_hash(""), int)

    def test_single_token(self):
        assert isinstance(_compute_prefix_hash([42]), int)


# ---------------------------------------------------------------------------
# Scheduler._route
# ---------------------------------------------------------------------------

class TestRoute:
    def test_unkeyed_goes_to_least_loaded(self):
        sch = _scheduler(3, prefix_routing=False)
        sch._workers[0].active_requests = 5
        sch._workers[1].active_requests = 2
        sch._workers[2].active_requests = 7
        assert _route(sch, "a") == 1

    def test_idle_new_keys_spread(self):
        sch = _scheduler(3)
        for ws in sch._workers:
            ws.active_requests = 0
        assigned = []
        for idx in range(6):
            assigned.append(_route(sch, routing_key=f"group-{idx}"))
            for ws in sch._workers:
                ws.active_requests = 0
        assert assigned == [0, 1, 2, 0, 1, 2]

    def test_grpo_group_colocates(self):
        sch = _scheduler(4)
        prompt = "System: You are a math tutor.\nUser: Solve x+2=5"
        assert len({_route(sch, prompt) for _ in range(8)}) == 1

    def test_different_prompts_spread(self):
        sch = _scheduler(8)
        assert len({_route(sch, f"System prompt variant {i}") for i in range(20)}) > 1

    def test_prefix_key_spills_at_capacity(self):
        sch = _scheduler(2, capacity=2)
        first = _route(sch, "p")
        assert _route(sch, "p") == first
        assert _route(sch, "p") != first

    def test_strict_prefix_key_never_spills(self):
        sch = _scheduler(2, capacity=1)
        first = _route(sch, "p", strict=True)
        assert all(_route(sch, "p", strict=True) == first for _ in range(5))

    def test_routing_key_never_spills(self):
        sch = _scheduler(2, capacity=1, prefix_routing=False)
        first = _route(sch, "turn-1", routing_key="g")
        assert _route(sch, "turn-2 very different prompt", routing_key="g") == first

    def test_unavailable_worker_is_reassigned(self):
        sch = _scheduler(3)
        first = _route(sch, routing_key="g")
        sch._workers[first].available = False
        again = _route(sch, routing_key="g")
        assert again != first
        assert _route(sch, routing_key="g") == again

    def test_paused_routes_nothing(self):
        sch = _scheduler(2)
        sch.pause()
        assert _route(sch, "p") is None


# ---------------------------------------------------------------------------
# Scheduler.submit with routing_key
# ---------------------------------------------------------------------------


def _capture_requests(sch: Scheduler) -> list:
    """Replace _process_request with a no-op capture so we can inspect the
    _Request objects submit() builds, without racing real worker dispatch."""
    captured: list = []

    async def fake_process(req):
        captured.append(req)

    sch._process_request = fake_process  # type: ignore[method-assign]
    return captured


class TestSchedulerRoutingKey:
    def test_replica_pool_scheduler_capacity_is_max_num_seqs(self):
        from arctic_platform.inference.server.replica_pool import ReplicaPool

        scheduler = ReplicaPool()._make_scheduler([MagicMock(), MagicMock()], 256)
        assert scheduler._capacity == 256
        assert scheduler._prefix_routing is True

    def test_shutdown_fails_waiting_requests(self):
        async def run() -> None:
            sch = _scheduler(1)
            sch.pause()
            future = sch.submit("p", {})
            await asyncio.sleep(0)
            assert not future.done()
            await sch.shutdown()
            with pytest.raises(RuntimeError, match="stopped"):
                await asyncio.wait_for(future, 1.0)

        asyncio.run(run())

    def test_routing_key_replaces_prompt_hash(self):
        async def run() -> None:
            sch = _scheduler(2)
            captured = _capture_requests(sch)
            sch.submit("turn-1 prompt", {}, routing_key="g42")
            sch.submit("turn-2 very different prompt", {}, routing_key="g42")
            await asyncio.sleep(0)
            assert [(r.key, r.sticky) for r in captured] == [("g42", True), ("g42", True)]
            await sch.shutdown()

        asyncio.run(run())

    def test_no_routing_key_uses_prompt_hash(self):
        async def run() -> None:
            sch = _scheduler(2)
            captured = _capture_requests(sch)
            sch.submit("abc", {})
            await asyncio.sleep(0)
            assert captured[0].key == _compute_prefix_hash("abc")
            assert captured[0].sticky is False
            await sch.shutdown()

        asyncio.run(run())

    def test_no_key_without_prefix_routing(self):
        async def run() -> None:
            sch = _scheduler(2, prefix_routing=False)
            captured = _capture_requests(sch)
            sch.submit("abc", {})
            await asyncio.sleep(0)
            assert captured[0].key is None
            await sch.shutdown()

        asyncio.run(run())

    def test_strict_flag_makes_request_sticky(self):
        async def run() -> None:
            sch = _scheduler(2)
            captured = _capture_requests(sch)
            sch.submit("p", {}, strict=True)
            await asyncio.sleep(0)
            assert captured[0].sticky is True
            assert captured[0].key == _compute_prefix_hash("p")
            await sch.shutdown()

        asyncio.run(run())

    def test_submit_batch_accepts_per_prompt_sampling_params(self):
        async def run() -> None:
            sch = _scheduler(2)
            submitted = []

            def fake_submit(prompt, sampling_params, routing_key=None, strict=False):
                submitted.append((prompt, sampling_params, routing_key, strict))
                future = asyncio.get_running_loop().create_future()
                future.set_result({"text": prompt})
                return future

            sch.submit = fake_submit  # type: ignore[method-assign]
            result = await sch.submit_batch(
                ["a", "b"],
                [{"max_tokens": 1}, None],
                routing_key=["k0", None],
                strict=True,
            )
            await sch.shutdown()

            assert result == [{"text": "a"}, {"text": "b"}]
            assert submitted == [
                ("a", {"max_tokens": 1}, "k0", True),
                ("b", {}, None, True),
            ]

        asyncio.run(run())
