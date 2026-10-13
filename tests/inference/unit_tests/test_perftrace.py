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

"""Tests for the opt-in generate-path perftrace."""
from __future__ import annotations

import asyncio
import collections
import gc
from types import SimpleNamespace

import pytest

from arctic_platform.inference.server import perftrace
from arctic_platform.inference.server.scheduler import Scheduler


@pytest.fixture(autouse=True)
def _fresh_perftrace(monkeypatch):
    for key in [k for k in __import__("os").environ if "PERFTRACE" in k]:
        monkeypatch.delenv(key)
    monkeypatch.setattr(perftrace, "_enabled", False)
    monkeypatch.setattr(perftrace, "_buf", collections.deque(maxlen=1000))
    monkeypatch.setattr(perftrace, "_role", "")
    monkeypatch.setattr(perftrace, "_settings", {})
    monkeypatch.setattr(perftrace, "_counters", {})
    monkeypatch.setattr(perftrace, "_gauges", [])
    monkeypatch.setattr(perftrace, "_sampler_started", False)
    monkeypatch.setattr(perftrace, "_engine", perftrace._EngineStats())
    monkeypatch.setattr(perftrace, "_engine_error_logged", False)


def _points(source):
    return [m[1] for m in source["marks"]]


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return asyncio.ensure_future(self._fn(*args, **kwargs))


class _FakeWorker:
    def __init__(self):
        self.calls = []
        self.generate = _Remote(self._generate)
        self.set_replica_id = _Remote(self._noop)
        self.get_stats = _Remote(self._stats)
        self.perftrace_drain = _Remote(self._drain)

    async def _generate(self, prompt, params, **kwargs):
        self.calls.append(kwargs)
        result = {"text": "x", "prompt_len": len(prompt), "generation_len": 1}
        if kwargs.get("perftrace_id") is not None:
            result["_perftrace"] = [("wk_enter", 1, None), ("wk_result", 2, {"prompt_len": len(prompt)})]
        return result

    async def _noop(self, *args):
        return None

    async def _stats(self):
        return {"gpu_cache_usage": 0.0, "num_requests_running": 0, "num_requests_waiting": 0}

    async def _drain(self):
        return {"src": "worker", "pid": 1, "host": "h", "ring_full": False, "marks": [[None, "sample", 3, {}]]}


def test_off_by_default_records_nothing():
    assert perftrace.configure(role="scheduler") is False
    perftrace.mark("r", "p")
    assert perftrace.current_trace_id() is None
    assert perftrace.drain()["marks"] == []


def test_dss_prefix_enables_and_arctic_prefix_wins():
    env = {"DSS_PERFTRACE": "1", "DSS_PERFTRACE_LAG_MS": "20", "ARCTIC_PERFTRACE_LAG_MS": "5"}
    assert perftrace.configure(env, role="scheduler") is True
    assert perftrace._settings["lag_ms"] == 5.0
    meta = perftrace.drain()["marks"][0]
    assert meta[1] == "meta" and meta[3]["role"] == "scheduler"


def test_trace_context_reaches_created_tasks():
    perftrace.configure({"ARCTIC_PERFTRACE": "1"}, role="scheduler")

    async def main():
        with perftrace.trace_context("rid-1"):
            task = asyncio.create_task(asyncio.sleep(0, result=perftrace.current_trace_id()))
        assert perftrace.current_trace_id() is None
        return await task

    assert asyncio.run(main()) == "rid-1"


def test_scheduler_traces_request_and_folds_worker_marks():
    perftrace.configure({"ARCTIC_PERFTRACE": "1"}, role="scheduler")
    perftrace.drain()
    worker = _FakeWorker()

    async def main():
        scheduler = Scheduler([worker], dynamic_concurrency=False)
        untraced = await scheduler.submit([1, 2, 3], {})
        with perftrace.trace_context("rid-7"):
            fut = scheduler.submit([1, 2, 3, 4], {})
        traced = await fut
        await asyncio.sleep(0)
        sources = await scheduler.perftrace_drain()
        scheduler._stopped = True
        return untraced, traced, sources

    untraced, traced, sources = asyncio.run(main())
    assert worker.calls == [{}, {"perftrace_id": "rid-7"}]
    assert "_perftrace" not in traced
    assert "_perftrace" not in untraced
    local, remote = sources
    marks = [m for m in local["marks"] if m[0] == "rid-7"]
    assert [m[1] for m in marks] == [
        "sch_submit_enter", "sch_submit", "sch_task_start", "sch_dispatch",
        "sch_sent", "sch_worker_ret", "wk_enter", "wk_result", "sch_task_end",
    ]
    wk_result = marks[7]
    assert wk_result[2] == 2 and wk_result[3] == {"worker": 0, "prompt_len": 4}
    assert remote["replica"] == 0 and _points(remote) == ["sample"]


def test_scheduler_gauges_are_weak_and_reported():
    perftrace.configure({"ARCTIC_PERFTRACE": "1"}, role="scheduler")

    async def main():
        scheduler = Scheduler([_FakeWorker(), _FakeWorker()], dynamic_concurrency=False)
        gauges = perftrace.collect_gauges()
        del scheduler
        gc.collect()
        return gauges

    gauges = asyncio.run(main())
    assert gauges["sch_active"] == [0, 0] and gauges["sch_submitted"] == 0
    assert perftrace.collect_gauges() == {}


def test_engine_stats_aggregate_and_never_raise():
    perftrace.configure({"ARCTIC_PERFTRACE": "1"}, role="worker")
    sched = SimpleNamespace(
        num_running_reqs=3, num_waiting_reqs=5, kv_cache_usage=0.4,
        prefix_cache_stats=SimpleNamespace(queries=100, hits=90),
    )
    finished = SimpleNamespace(
        e2e_latency=0.5, queued_time=0.1, prefill_time=0.3, inference_time=0.4, num_prompt_tokens=10,
    )
    it = SimpleNamespace(
        num_generation_tokens=2, num_prompt_tokens=10, num_preempted_reqs=0,
        time_to_first_tokens_iter=[0.45], finished_requests=[finished],
    )
    perftrace.record_engine_stats(sched, it)
    perftrace.record_engine_stats(SimpleNamespace(num_running_reqs="bad"), None)
    out = perftrace._engine.pop()
    assert out["steps"] == 2 and out["running"] == [3, 3] and out["prefix"] == [100, 90]
    assert out["queued_ms"] == [1, 100.0, 100.0, 100.0] and out["finished_prompt_tokens"] == 10
    assert "engine_error" in _points(perftrace.drain())
