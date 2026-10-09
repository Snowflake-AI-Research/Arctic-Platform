"""GPU-resident caches for router-replay tensors.

TX (sampling side, overwrite-on-put) and RX (training side, pop-on-read)
share a common base. Tensors are ``torch.uint8`` on the worker's device,
shape ``[seq_len, num_layers, topk]``. Byte counts back ``max_bytes``
backpressure (raises ``RouterReplayCacheFull``).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field

import numpy as np
import torch

logger = logging.getLogger(__name__)

ROUTER_REPLAY_CACHE_DTYPE = torch.uint8
EXACT_REPLAY_ID_PREFIX = "rr1:"
_TOMBSTONE_ORDER_LOCK_STRIPES = 64


def is_exact_replay_id(sample_id: str) -> bool:
    return sample_id.startswith(EXACT_REPLAY_ID_PREFIX)


class RouterReplayMissingError(RuntimeError):
    """Training side requested sample_ids that no sampling rank holds.

    Raised identically on every rank (deterministic missing-set compute)
    so the second collective never runs and no rank blocks, unless
    tolerated via ``allow_missing``. The training
    job actor catches it and returns HTTP 410 to the trainer, which drops
    the step.
    """

    def __init__(self, missing_sample_ids: list[str]) -> None:
        self.missing_sample_ids = list(missing_sample_ids)
        super().__init__(
            f"router-replay missing for {len(self.missing_sample_ids)} sample_id(s): {self.missing_sample_ids[:8]}"
            + ("..." if len(self.missing_sample_ids) > 8 else "")
        )


class RouterReplayCacheFull(RuntimeError):
    """Raised by ``put`` when adding the entry would exceed ``max_bytes``."""


class RouterReplayDuplicateError(RuntimeError):
    """Raised when an exact replay id is advertised or inserted twice."""

    def __init__(self, duplicate_sample_ids: list[str], *, owners: dict[str, list[int]] | None = None) -> None:
        self.duplicate_sample_ids = list(duplicate_sample_ids)
        self.owners = dict(owners or {})
        owner_summary = {sample_id: self.owners.get(sample_id, []) for sample_id in self.duplicate_sample_ids[:8]}
        super().__init__(
            f"router-replay duplicate exact id(s): count={len(self.duplicate_sample_ids)} "
            f"ids={self.duplicate_sample_ids[:8]} owners={owner_summary}"
        )


@dataclass
class _EvictionSummary:
    reason: str
    projected_before: int
    projected_after: int
    target_bytes: int | None = None
    evicted_count: int = 0
    evicted_bytes: int = 0
    evicted_sample_ids_head: list[str] = field(default_factory=list)

    @property
    def evicted(self) -> bool:
        return self.evicted_count > 0

    def record(self, sample_id: str, removed_bytes: int, projected: int) -> int:
        self.evicted_count += 1
        self.evicted_bytes += int(removed_bytes)
        if len(self.evicted_sample_ids_head) < 16:
            self.evicted_sample_ids_head.append(sample_id)
        self.projected_after = int(projected)
        return projected


class _RouterReplayCacheBase:
    """Shared put / get / pop / discard / stats backbone.

    A side dict tracks per-entry byte counts for O(1) ``max_bytes`` accounting.
    A single ``threading.Lock`` guards the dicts (vLLM async loop + per-step
    NCCL task; contention is low). Tensor copy runs outside the lock.
    """

    def __init__(self, device: torch.device, max_bytes: int) -> None:
        self.device = torch.device(device)
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._data: dict[str, torch.Tensor] = {}
        self._sizes: dict[str, int] = {}
        self._created_at: dict[str, float] = {}
        self._last_touched: dict[str, float] = {}
        self._bytes_in_use: int = 0
        self._peak_bytes_in_use: int = 0
        # Lifetime counters for observability.
        self._n_put = 0
        self._n_overwrite = 0
        self._n_duplicate_rejected = 0
        self._n_pop = 0
        self._n_discard = 0
        self._n_evicted_lru = 0
        self._n_evicted_ttl = 0

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def put(self, sample_id: str, value: torch.Tensor | np.ndarray) -> None:
        """Insert or replace the entry; coerces to uint8 on self.device.

        Overwrite drops the prior tensor BEFORE the max_bytes check so a
        multi-turn rollout replacing its own prior turn does not double-count.
        """
        self._put(sample_id, value, replace=True)

    def put_new(self, sample_id: str, value: torch.Tensor | np.ndarray) -> None:
        """Insert an entry exactly once.

        Exact replay ids identify one backend attempt. Seeing the same id
        twice means a stale or duplicated request could overwrite accepted
        routing, so reject it atomically instead.
        """
        self._put(sample_id, value, replace=False)

    def _put(
        self,
        sample_id: str,
        value: torch.Tensor | np.ndarray,
        *,
        replace: bool,
    ) -> None:
        tensor = self._coerce(value)
        new_bytes = tensor.element_size() * tensor.numel()
        new_shape = list(tensor.shape)
        now = time.monotonic()
        with self._lock:
            if not replace and self._suppress_put_new_locked(sample_id, now):
                return
            if not replace and sample_id in self._data:
                self._n_duplicate_rejected += 1
                raise RouterReplayDuplicateError([sample_id])
            old_bytes = self._sizes.get(sample_id, 0)
            bytes_before = self._bytes_in_use
            projected = self._bytes_in_use - old_bytes + new_bytes
            projected_before_eviction = projected
            projected, evictions = self._evict_for_put_locked(
                sample_id=sample_id,
                projected=projected,
                new_bytes=new_bytes,
                now=now,
            )
            if projected > self.max_bytes:
                raise RouterReplayCacheFull(
                    "router-replay cache would exceed max_bytes "
                    f"({projected} > {self.max_bytes}) on put({sample_id!r}, "
                    f"shape={list(tensor.shape)}, dtype={tensor.dtype})"
                )
            if sample_id in self._data:
                self._n_overwrite += 1
            else:
                self._created_at[sample_id] = now
            self._data[sample_id] = tensor
            self._sizes[sample_id] = new_bytes
            self._last_touched[sample_id] = now
            self._bytes_in_use = projected
            self._peak_bytes_in_use = max(self._peak_bytes_in_use, projected)
            self._n_put += 1
            if evictions:
                self._log_evictions_locked(
                    evictions,
                    sample_id=sample_id,
                    new_shape=new_shape,
                    new_bytes=new_bytes,
                    bytes_before=bytes_before,
                    projected_before_eviction=projected_before_eviction,
                )

    def pop(self, sample_id: str) -> torch.Tensor:
        """Remove and return the entry; raises KeyError if absent."""
        with self._lock:
            tensor = self._data.pop(sample_id)
            self._remove_metadata_locked(sample_id)
            self._n_pop += 1
            return tensor

    def get(self, sample_id: str) -> torch.Tensor:
        """Return the entry without removing it; raises KeyError if absent."""
        with self._lock:
            tensor = self._data[sample_id]
            self._last_touched[sample_id] = time.monotonic()
            return tensor

    def discard(self, sample_ids: Iterable[str]) -> int:
        """Remove listed sample_ids (missing ids ignored); return count removed.

        Idempotent so it can safely run after a partial NCCL transfer.
        """
        removed = 0
        with self._lock:
            for sid in sample_ids:
                if sid in self._data:
                    self._data.pop(sid)
                    self._remove_metadata_locked(sid)
                    removed += 1
            self._n_discard += removed
        return removed

    def clear(self) -> int:
        """Remove every entry; return the count removed.

        Called at the top of every RX recv to guard against (recv-ok,
        fwd-bwd-crashed) skew from the prior step.
        """
        with self._lock:
            removed = len(self._data)
            self._data.clear()
            self._sizes.clear()
            self._created_at.clear()
            self._last_touched.clear()
            self._bytes_in_use = 0
        return removed

    # ------------------------------------------------------------------
    # Read-only inspection
    # ------------------------------------------------------------------

    def __contains__(self, sample_id: str) -> bool:
        with self._lock:
            return sample_id in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def held_sample_ids(self) -> list[str]:
        """Snapshot of current keys; NCCL send-side advertises this in
        the all_gather phase."""
        with self._lock:
            return list(self._data)

    def peek(self, sample_id: str) -> dict | None:
        """Return ``{shape, dtype, bytes}`` for the entry without touching
        the tensor. Returns ``None`` if absent."""
        with self._lock:
            t = self._data.get(sample_id)
            if t is None:
                return None
            return {
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "bytes": self._sizes[sample_id],
            }

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._data),
                "bytes_in_use": self._bytes_in_use,
                "peak_bytes_in_use": self._peak_bytes_in_use,
                "max_bytes": self.max_bytes,
                "n_put": self._n_put,
                "n_overwrite": self._n_overwrite,
                "n_duplicate_rejected": self._n_duplicate_rejected,
                "n_pop": self._n_pop,
                "n_discard": self._n_discard,
                "n_evicted_lru": self._n_evicted_lru,
                "n_evicted_ttl": self._n_evicted_ttl,
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _coerce(self, value: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(value, np.ndarray):
            t = torch.from_numpy(value)
        elif isinstance(value, torch.Tensor):
            t = value
        else:
            raise TypeError(f"router-replay cache.put expects torch.Tensor or np.ndarray, got {type(value).__name__}")
        if t.dtype != ROUTER_REPLAY_CACHE_DTYPE:
            t = t.to(ROUTER_REPLAY_CACHE_DTYPE)
        if t.device != self.device:
            t = t.to(self.device, non_blocking=True)
        return t.contiguous()

    def _evict_for_put_locked(
        self,
        *,
        sample_id: str,
        projected: int,
        new_bytes: int,
        now: float,
    ) -> tuple[int, list[_EvictionSummary]]:
        return projected, []

    def _suppress_put_new_locked(self, sample_id: str, now: float) -> bool:
        return False

    def _log_evictions_locked(
        self,
        evictions: list[_EvictionSummary],
        *,
        sample_id: str,
        new_shape: list[int],
        new_bytes: int,
        bytes_before: int,
        projected_before_eviction: int,
    ) -> None:
        stats_after = {
            "entries": len(self._data),
            "bytes_in_use": self._bytes_in_use,
            "peak_bytes_in_use": self._peak_bytes_in_use,
            "max_bytes": self.max_bytes,
            "n_put": self._n_put,
            "n_overwrite": self._n_overwrite,
            "n_duplicate_rejected": self._n_duplicate_rejected,
            "n_pop": self._n_pop,
            "n_discard": self._n_discard,
            "n_evicted_lru": self._n_evicted_lru,
            "n_evicted_ttl": self._n_evicted_ttl,
        }
        for event in evictions:
            logger.warning(
                "router-replay TX cache evicted entries reason=%s "
                "new_sample_id=%s new_shape=%s new_bytes=%d max_bytes=%d "
                "bytes_before=%d projected_before=%d projected_after=%d "
                "target_bytes=%s evicted_count=%d evicted_bytes=%d "
                "evicted_sample_ids_head=%s stats_after=%s",
                event.reason,
                sample_id,
                new_shape,
                new_bytes,
                self.max_bytes,
                bytes_before,
                projected_before_eviction,
                event.projected_after,
                event.target_bytes,
                event.evicted_count,
                event.evicted_bytes,
                event.evicted_sample_ids_head,
                stats_after,
            )

    def _remove_metadata_locked(self, sample_id: str) -> int:
        removed_bytes = self._sizes.pop(sample_id, 0)
        self._created_at.pop(sample_id, None)
        self._last_touched.pop(sample_id, None)
        self._bytes_in_use -= removed_bytes
        return removed_bytes


class RouterReplayCacheTX(_RouterReplayCacheBase):
    """Sampling-side cache. Overwrite-on-put.

    Each turn of a multi-turn rollout emits a full capture covering the
    conversation up to that turn, so the latest put is authoritative.
    Group-affinity routing keeps the same worker per sample_id; failover
    is benign because the new replica re-prefills its own complete capture.
    """

    def __init__(self, device: torch.device, max_bytes: int) -> None:
        super().__init__(device=device, max_bytes=max_bytes)
        self._tombstones: OrderedDict[str, float] = OrderedDict()
        self._tombstone_order_locks = [threading.Lock() for _ in range(_TOMBSTONE_ORDER_LOCK_STRIPES)]
        self.ttl_s = max(0.0, float(os.environ.get("ARCTIC_ROUTER_REPLAY_TX_TTL_S", "86400")))
        self.evict_min_age_s = max(
            0.0,
            float(os.environ.get("ARCTIC_ROUTER_REPLAY_TX_EVICT_MIN_AGE_S", "120")),
        )
        self.low_watermark_ratio = min(
            1.0,
            max(0.1, float(os.environ.get("ARCTIC_ROUTER_REPLAY_TX_LOW_WATERMARK_RATIO", "0.85"))),
        )

    def put_new(self, sample_id: str, value: torch.Tensor | np.ndarray) -> None:
        if not is_exact_replay_id(sample_id):
            super().put_new(sample_id, value)
            return
        with self._tombstone_order_lock(sample_id):
            now = time.monotonic()
            with self._lock:
                if self._suppress_put_new_locked(sample_id, now):
                    return
            super().put_new(sample_id, value)

    def snapshot(self) -> dict[str, torch.Tensor]:
        """Return strong tensor refs for a race-safe router-replay send.

        The cache may evict/discard entries while the NCCL send is preparing.
        Holding these refs ensures a manifest-advertised tensor remains alive
        until the send either completes or fails.
        """
        now = time.monotonic()
        with self._lock:
            for sid in self._data:
                self._last_touched[sid] = now
            return dict(self._data)

    def discard(self, sample_ids: Iterable[str]) -> int:
        """Remove entries and suppress late inserts for absent exact ids."""
        sample_ids = list(sample_ids)
        lock_indexes = sorted({self._tombstone_order_lock_index(sid) for sid in sample_ids if is_exact_replay_id(sid)})
        order_locks = [self._tombstone_order_locks[index] for index in lock_indexes]
        for lock in order_locks:
            lock.acquire()
        removed = 0
        try:
            now = time.monotonic()
            with self._lock:
                self._expire_tombstones_locked(now)
                for sid in sample_ids:
                    if sid in self._data:
                        self._data.pop(sid)
                        self._remove_metadata_locked(sid)
                        removed += 1
                    elif is_exact_replay_id(sid):
                        self._tombstones.pop(sid, None)
                        self._tombstones[sid] = now
                self._n_discard += removed
        finally:
            for lock in reversed(order_locks):
                lock.release()
        return removed

    def _tombstone_order_lock_index(self, sample_id: str) -> int:
        return hash(sample_id) % len(self._tombstone_order_locks)

    def _tombstone_order_lock(self, sample_id: str):
        return self._tombstone_order_locks[self._tombstone_order_lock_index(sample_id)]

    def clear(self) -> int:
        with self._lock:
            removed = len(self._data)
            self._data.clear()
            self._sizes.clear()
            self._created_at.clear()
            self._last_touched.clear()
            self._tombstones.clear()
            self._bytes_in_use = 0
        return removed

    def _suppress_put_new_locked(self, sample_id: str, now: float) -> bool:
        self._expire_tombstones_locked(now)
        return is_exact_replay_id(sample_id) and sample_id in self._tombstones

    def _expire_tombstones_locked(self, now: float) -> None:
        if self.ttl_s <= 0:
            return
        while self._tombstones:
            _, created_at = next(iter(self._tombstones.items()))
            if now - created_at < self.ttl_s:
                return
            self._tombstones.popitem(last=False)

    def _evict_for_put_locked(
        self,
        *,
        sample_id: str,
        projected: int,
        new_bytes: int,
        now: float,
    ) -> tuple[int, list[_EvictionSummary]]:
        events: list[_EvictionSummary] = []
        if self.ttl_s > 0:
            ttl_event = _EvictionSummary(
                reason="ttl",
                projected_before=projected,
                projected_after=projected,
            )
            expired = [
                sid
                for sid, created_at in self._created_at.items()
                if sid != sample_id and now - created_at >= self.ttl_s
            ]
            for sid in expired:
                if sid in self._data:
                    self._data.pop(sid)
                    removed_bytes = self._remove_metadata_locked(sid)
                    projected -= removed_bytes
                    ttl_event.record(sid, removed_bytes, projected)
                    self._n_evicted_ttl += 1
            if ttl_event.evicted:
                events.append(ttl_event)

        if projected <= self.max_bytes:
            return projected, events

        target_bytes = int(self.max_bytes * self.low_watermark_ratio)
        target_bytes = min(target_bytes, self.max_bytes)
        projected, lru_event = self._evict_lru_locked(
            sample_id=sample_id,
            projected=projected,
            target_bytes=target_bytes,
            now=now,
            respect_min_age=True,
        )
        if lru_event.evicted:
            events.append(lru_event)
        if projected <= self.max_bytes:
            return projected, events
        projected, lru_event = self._evict_lru_locked(
            sample_id=sample_id,
            projected=projected,
            target_bytes=self.max_bytes,
            now=now,
            respect_min_age=False,
        )
        if lru_event.evicted:
            events.append(lru_event)
        return projected, events

    def _evict_lru_locked(
        self,
        *,
        sample_id: str,
        projected: int,
        target_bytes: int,
        now: float,
        respect_min_age: bool,
    ) -> tuple[int, _EvictionSummary]:
        event = _EvictionSummary(
            reason="lru",
            projected_before=projected,
            projected_after=projected,
            target_bytes=target_bytes,
        )
        candidates = sorted(
            (
                (self._last_touched.get(sid, self._created_at.get(sid, now)), sid)
                for sid in self._data
                if sid != sample_id
            ),
            key=lambda item: item[0],
        )
        for touched_at, sid in candidates:
            if projected <= target_bytes:
                break
            if respect_min_age and now - self._created_at.get(sid, touched_at) < self.evict_min_age_s:
                continue
            self._data.pop(sid, None)
            removed_bytes = self._remove_metadata_locked(sid)
            projected -= removed_bytes
            event.record(sid, removed_bytes, projected)
            self._n_evicted_lru += 1
        return projected, event


class RouterReplayCacheRX(_RouterReplayCacheBase):
    """Training-side cache. Pop-on-read.

    Filled by ``recv_router_replay`` and drained by the per-rank
    ``fwd_bwd``. Empty after each successful step (asserted in tests).
    """
