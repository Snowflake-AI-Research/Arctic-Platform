from unittest.mock import MagicMock

from arctic_platform.inference.server.replica_pool import ReplicaPool


def _pool_with_scheduler(n_workers: int) -> ReplicaPool:
    pool = ReplicaPool()
    pool._scheduler = pool._make_scheduler([MagicMock() for _ in range(n_workers)])
    return pool


def test_concurrency_limit_is_zero_before_initialize():
    assert ReplicaPool().concurrency_limit == 0


def test_concurrency_limit_defaults_to_128_per_replica(monkeypatch):
    monkeypatch.delenv("ARCTIC_WORKER_CONCURRENCY_LIMIT", raising=False)
    assert _pool_with_scheduler(2).concurrency_limit == 256


def test_concurrency_limit_follows_the_env_override(monkeypatch):
    monkeypatch.setenv("ARCTIC_WORKER_CONCURRENCY_LIMIT", "32")
    assert _pool_with_scheduler(3).concurrency_limit == 96


def test_concurrency_limit_reports_the_current_limits(monkeypatch):
    monkeypatch.setenv("ARCTIC_WORKER_CONCURRENCY_LIMIT", "32")
    pool = _pool_with_scheduler(2)
    workers = pool._scheduler._workers
    workers[0].concurrency_limit = 200  # as dynamic concurrency would adapt it
    assert pool.concurrency_limit == 232
    workers[1].quarantine()  # an unschedulable replica takes no requests
    assert pool.concurrency_limit == 200
