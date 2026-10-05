import logging
import threading
import time
from collections import OrderedDict

import pytest
import torch

from arctic_platform.inference.server.router_replay import (
    RouterReplayCacheTX,
    RouterReplayDuplicateError,
)
from arctic_platform.inference.server.router_replay.all2all import (
    _compute_plan,
    _PerRankManifest,
)


def _tensor(seq_len: int = 4, n_layers: int = 2, top_k: int = 2, base: int = 1) -> torch.Tensor:
    values = torch.arange(seq_len * n_layers * top_k, dtype=torch.int32) + base
    return values.reshape(seq_len, n_layers, top_k)


def test_tx_cache_default_ttl_is_24_hours(monkeypatch):
    monkeypatch.delenv("ARCTIC_ROUTER_REPLAY_TX_TTL_S", raising=False)

    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)

    assert cache.ttl_s == 24 * 60 * 60


def test_tx_cache_evicts_lru_when_full(monkeypatch):
    monkeypatch.setenv("ARCTIC_ROUTER_REPLAY_TX_EVICT_MIN_AGE_S", "0")
    one = _tensor(seq_len=4)
    bytes_one = one.numel()
    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=bytes_one)

    cache.put("sid-1", one)
    cache.put("sid-2", one)

    assert "sid-1" not in cache
    assert "sid-2" in cache
    assert cache.stats()["bytes_in_use"] == bytes_one
    assert cache.stats()["n_evicted_lru"] == 1


def test_tx_cache_logs_lru_eviction(monkeypatch, caplog):
    monkeypatch.setenv("ARCTIC_ROUTER_REPLAY_TX_EVICT_MIN_AGE_S", "0")
    one = _tensor(seq_len=4)
    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=one.numel())
    caplog.set_level(logging.WARNING, logger="arctic_platform.inference.server.router_replay.cache")

    cache.put("sid-1", one)
    cache.put("sid-2", one)

    messages = [record.getMessage() for record in caplog.records]
    assert any("reason=lru" in message for message in messages)
    assert any("new_sample_id=sid-2" in message for message in messages)
    assert any("evicted_count=1" in message for message in messages)
    assert any("evicted_sample_ids_head=['sid-1']" in message for message in messages)
    assert any(f"max_bytes={one.numel()}" in message for message in messages)


def test_tx_cache_logs_ttl_eviction(monkeypatch, caplog):
    monkeypatch.setenv("ARCTIC_ROUTER_REPLAY_TX_TTL_S", "0.001")
    one = _tensor(seq_len=4)
    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)
    caplog.set_level(logging.WARNING, logger="arctic_platform.inference.server.router_replay.cache")

    cache.put("sid-old", one)
    time.sleep(0.01)
    cache.put("sid-new", one)

    messages = [record.getMessage() for record in caplog.records]
    assert any("reason=ttl" in message for message in messages)
    assert any("new_sample_id=sid-new" in message for message in messages)
    assert any("evicted_count=1" in message for message in messages)
    assert any("evicted_sample_ids_head=['sid-old']" in message for message in messages)


def test_tx_cache_snapshot_keeps_tensor_ref_after_discard():
    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)
    tensor = _tensor(seq_len=4)
    cache.put("sid-1", tensor)

    snapshot = cache.snapshot()
    removed = cache.discard(["sid-1"])

    assert removed == 1
    assert "sid-1" not in cache
    assert torch.equal(snapshot["sid-1"], tensor.to(torch.uint8))


def test_cache_peak_bytes_in_use_survives_removal_and_eviction(monkeypatch):
    monkeypatch.setenv("ARCTIC_ROUTER_REPLAY_TX_EVICT_MIN_AGE_S", "0")
    tensor = _tensor(seq_len=4)
    entry_bytes = tensor.numel()
    cache = RouterReplayCacheTX(
        device=torch.device("cpu"),
        max_bytes=3 * entry_bytes,
    )

    cache.put("sid-1", tensor)
    cache.put("sid-2", tensor)
    cache.put("sid-3", tensor)
    peak_bytes = 3 * entry_bytes

    cache.pop("sid-1")
    assert cache.stats()["peak_bytes_in_use"] == peak_bytes

    cache.discard(["sid-2"])
    assert cache.stats()["peak_bytes_in_use"] == peak_bytes

    cache.put("sid-4", tensor)
    cache.put("sid-5", tensor)
    cache.put("sid-6", tensor)
    assert cache.stats()["n_evicted_lru"] > 0
    assert cache.stats()["peak_bytes_in_use"] == peak_bytes

    cache.clear()
    stats = cache.stats()
    assert stats["bytes_in_use"] == 0
    assert stats["peak_bytes_in_use"] == peak_bytes


def test_tx_cache_put_new_rejects_duplicate_exact_id_without_overwrite():
    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)
    original = _tensor(seq_len=4, base=1)
    replacement = _tensor(seq_len=4, base=20)
    cache.put_new("rr1:attempt-1", original)

    with pytest.raises(RouterReplayDuplicateError, match="rr1:attempt-1"):
        cache.put_new("rr1:attempt-1", replacement)

    assert torch.equal(cache.get("rr1:attempt-1"), original.to(torch.uint8))
    assert cache.stats()["n_duplicate_rejected"] == 1


def test_tx_cache_absent_exact_discard_suppresses_late_put_new():
    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)

    assert cache.discard(["rr1:late"]) == 0
    cache.put_new("rr1:late", _tensor())

    assert "rr1:late" not in cache
    assert cache.stats()["bytes_in_use"] == 0
    assert cache.stats()["n_put"] == 0


def test_tx_cache_tombstone_precheck_skips_materialization(monkeypatch):
    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)
    cache.discard(["rr1:late"])
    coerce_calls = []

    def record_coerce(value):
        coerce_calls.append(value)
        return value

    monkeypatch.setattr(cache, "_coerce", record_coerce)

    cache.put_new("rr1:late", _tensor())

    assert coerce_calls == []
    assert "rr1:late" not in cache


def test_tx_cache_discard_cannot_complete_during_same_id_materialization(
    monkeypatch,
):
    class TrackingLock:
        def __init__(self):
            self.lock = threading.Lock()
            self.waiting = threading.Event()

        def acquire(self):
            if self.lock.locked():
                self.waiting.set()
            return self.lock.acquire()

        def release(self):
            self.lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *_args):
            self.release()

    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)
    sample_id = "rr1:race"
    lock_index = cache._tombstone_order_lock_index(sample_id)
    tracking_lock = TrackingLock()
    cache._tombstone_order_locks[lock_index] = tracking_lock
    coerce_started = threading.Event()
    allow_coerce = threading.Event()
    events = []
    errors = []
    original_coerce = cache._coerce

    def blocking_coerce(value):
        coerce_started.set()
        allow_coerce.wait()
        events.append("materialized")
        return original_coerce(value)

    def put():
        try:
            cache.put_new(sample_id, _tensor())
            events.append("put_done")
        except BaseException as exc:
            errors.append(exc)

    def discard():
        try:
            cache.discard([sample_id])
            events.append("discard_done")
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(cache, "_coerce", blocking_coerce)
    put_thread = threading.Thread(target=put)
    discard_thread = threading.Thread(target=discard)
    put_thread.start()
    assert coerce_started.wait(timeout=1)
    discard_thread.start()
    assert tracking_lock.waiting.wait(timeout=1)
    allow_coerce.set()
    put_thread.join(timeout=1)
    discard_thread.join(timeout=1)

    assert errors == []
    assert not put_thread.is_alive()
    assert not discard_thread.is_alive()
    assert events == ["materialized", "put_done", "discard_done"]
    assert sample_id not in cache
    assert cache.stats()["n_evicted_lru"] == 0


def test_tx_cache_unrelated_exact_ids_materialize_concurrently(monkeypatch):
    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)
    first_id = "rr1:first"
    first_lock = cache._tombstone_order_lock(first_id)
    second_id = next(
        f"rr1:second-{index}"
        for index in range(1_000)
        if cache._tombstone_order_lock(f"rr1:second-{index}") is not first_lock
    )
    first_started = threading.Event()
    allow_first = threading.Event()
    second_done = threading.Event()
    errors = []
    original_coerce = cache._coerce
    calls = 0
    calls_lock = threading.Lock()

    def blocking_first_coerce(value):
        nonlocal calls
        with calls_lock:
            calls += 1
            call = calls
        if call == 1:
            first_started.set()
            allow_first.wait()
        return original_coerce(value)

    def put(sample_id, done=None):
        try:
            cache.put_new(sample_id, _tensor())
            if done is not None:
                done.set()
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(cache, "_coerce", blocking_first_coerce)
    first_thread = threading.Thread(target=put, args=(first_id,))
    second_thread = threading.Thread(target=put, args=(second_id, second_done))
    first_thread.start()
    assert first_started.wait(timeout=1)
    second_thread.start()
    assert second_done.wait(timeout=1)
    allow_first.set()
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)

    assert errors == []
    assert first_id in cache
    assert second_id in cache


def test_tx_cache_late_put_new_does_not_evict_replacement(monkeypatch):
    monkeypatch.setenv("ARCTIC_ROUTER_REPLAY_TX_EVICT_MIN_AGE_S", "0")
    tensor = _tensor()
    cache = RouterReplayCacheTX(
        device=torch.device("cpu"),
        max_bytes=tensor.numel(),
    )

    cache.discard(["rr1:late"])
    cache.put_new("rr1:replacement", tensor)
    cache.put_new("rr1:late", _tensor(base=20))

    assert "rr1:late" not in cache
    assert torch.equal(cache.get("rr1:replacement"), tensor.to(torch.uint8))
    assert cache.stats()["n_evicted_lru"] == 0


def test_tx_cache_exact_tombstone_expires_with_cache_ttl(monkeypatch):
    now = [100.0]
    monkeypatch.setenv("ARCTIC_ROUTER_REPLAY_TX_TTL_S", "10")
    monkeypatch.setattr(
        "arctic_platform.inference.server.router_replay.cache.time.monotonic",
        lambda: now[0],
    )
    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)

    cache.discard(["rr1:late"])
    now[0] += cache.ttl_s
    cache.put_new("rr1:late", _tensor())

    assert "rr1:late" in cache


def test_tx_cache_repeated_discard_refreshes_tombstone_ttl(monkeypatch):
    now = [100.0]
    monkeypatch.setenv("ARCTIC_ROUTER_REPLAY_TX_TTL_S", "10")
    monkeypatch.setattr(
        "arctic_platform.inference.server.router_replay.cache.time.monotonic",
        lambda: now[0],
    )
    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)

    cache.discard(["rr1:late"])
    now[0] += 5
    cache.discard(["rr1:late"])
    assert len(cache._tombstones) == 1
    now[0] += 5
    cache.put_new("rr1:late", _tensor())
    assert "rr1:late" not in cache

    now[0] += 5
    cache.put_new("rr1:late", _tensor())
    assert "rr1:late" in cache


def test_tx_cache_tombstone_expiry_work_is_bounded_by_expired_prefix(monkeypatch):
    class CountingOrderedDict(OrderedDict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.items_yielded = 0

        def items(self):
            for item in super().items():
                self.items_yielded += 1
                yield item

    now = [100.0]
    monkeypatch.setenv("ARCTIC_ROUTER_REPLAY_TX_TTL_S", "10")
    monkeypatch.setattr(
        "arctic_platform.inference.server.router_replay.cache.time.monotonic",
        lambda: now[0],
    )
    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)
    cache.discard([f"rr1:retained-{index}" for index in range(10_000)])
    tombstones = CountingOrderedDict(cache._tombstones)
    cache._tombstones = tombstones

    cache.put_new("rr1:new", _tensor())

    assert tombstones.items_yielded == 2
    assert "rr1:new" in cache

    tombstones.items_yielded = 0
    cache.discard(["rr1:another"])
    assert tombstones.items_yielded == 1


def test_tx_cache_clear_removes_exact_tombstones():
    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)

    cache.discard(["rr1:late"])
    assert cache.clear() == 0
    cache.put_new("rr1:late", _tensor())

    assert "rr1:late" in cache


def test_tx_cache_absent_legacy_discard_preserves_overwrite_behavior():
    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)
    original = _tensor(base=1)
    replacement = _tensor(base=20)

    assert cache.discard(["trajectory-1"]) == 0
    cache.put("trajectory-1", original)
    cache.put("trajectory-1", replacement)

    assert torch.equal(cache.get("trajectory-1"), replacement.to(torch.uint8))
    assert cache.stats()["n_overwrite"] == 1


def test_all2all_rejects_duplicate_exact_id_across_senders():
    manifests = [
        _PerRankManifest(
            role="sender",
            rank=2,
            held=["rr1:attempt-1"],
            shapes={"rr1:attempt-1": [4, 2, 2]},
        ),
        _PerRankManifest(
            role="sender",
            rank=3,
            held=["rr1:attempt-1"],
            shapes={"rr1:attempt-1": [4, 2, 2]},
        ),
        _PerRankManifest(role="receiver", rank=0, needed=["rr1:attempt-1"]),
    ]

    with pytest.raises(RouterReplayDuplicateError) as exc_info:
        _compute_plan(manifests)

    assert exc_info.value.owners == {"rr1:attempt-1": [2, 3]}


def test_all2all_legacy_duplicate_uses_lowest_rank_with_warning(caplog):
    caplog.set_level(logging.WARNING, logger="arctic_platform.inference.server.router_replay.all2all")
    manifests = [
        _PerRankManifest(
            role="sender",
            rank=3,
            held=["trajectory-1"],
            shapes={"trajectory-1": [4, 2, 2]},
        ),
        _PerRankManifest(
            role="sender",
            rank=2,
            held=["trajectory-1"],
            shapes={"trajectory-1": [4, 2, 2]},
        ),
        _PerRankManifest(role="receiver", rank=0, needed=["trajectory-1"]),
    ]

    plan, missing = _compute_plan(manifests)

    assert missing == set()
    assert plan[0].sender_rank == 2
    assert "legacy duplicate ids use lowest-rank owner count=1" in caplog.text
