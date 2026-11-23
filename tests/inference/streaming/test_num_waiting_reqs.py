"""Per-replica num_waiting_reqs: recorded every engine step, None for a failed replica."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from cpu_support import load_library

load_library()
from arctic_platform.inference.server.metrics import WorkerMetricsCollector
from arctic_platform.inference.server.replica_pool import ReplicaPool


def _stats(num_waiting_reqs):
    return SimpleNamespace(
        num_waiting_reqs=num_waiting_reqs,
        num_running_reqs=0,
        kv_cache_usage=0.0,
        spec_decoding_stats=None,
    )


def _worker(**remote):
    worker = MagicMock()
    worker.num_waiting_reqs.remote = AsyncMock(**remote)
    return worker


class TestNumWaitingReqs:
    def test_collector_records_throttled_steps_and_survives_drain(self):
        """The count updates on steps the snapshot throttle skips and is not cleared by a drain."""
        collector = WorkerMetricsCollector(min_interval_s=3600)
        collector.push_snapshot(_stats(3), None)
        collector.push_snapshot(_stats(7), None)
        assert len(collector.drain_snapshots()) == 1
        assert collector.num_waiting_reqs == 7

    def test_pool_returns_none_for_failed_replica(self):
        """A replica whose call raises reports None instead of failing the whole call."""
        pool = ReplicaPool()
        pool._workers = [_worker(return_value=4), _worker(side_effect=RuntimeError("dead"))]
        assert asyncio.run(pool.num_waiting_reqs()) == [4, None]

    def test_pool_rejects_other_model_id(self):
        """A model_id that does not match the pool's model raises."""
        pool = ReplicaPool()
        pool._model_id = "m"
        with pytest.raises(ValueError, match="model_id mismatch"):
            asyncio.run(pool.num_waiting_reqs("other"))
