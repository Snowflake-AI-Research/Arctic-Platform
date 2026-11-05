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
"""Bounded pinned host staging cache for activation CPU offload."""

from __future__ import annotations

from collections import Counter
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import fields
from typing import Dict
from typing import Optional
from typing import Protocol
from typing import Tuple

import torch

_AUTO_CAP_SHRINK_STEPS = 2


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _empty_host_pin_cache() -> None:
    """Return freed pinned host blocks from PyTorch's host caching allocator to the OS."""
    empty_cache = getattr(getattr(torch, "cuda", None), "memory", None)
    if empty_cache is not None and hasattr(empty_cache, "empty_host_cache"):
        empty_cache.empty_host_cache()
        return
    host_empty_cache = getattr(getattr(torch, "_C", None), "_host_emptyCache", None)
    if host_empty_cache is not None:
        host_empty_cache()


@dataclass(frozen=True)
class PoolKey:
    """Cache pool identity: contiguous buffers share dtype buckets; strided views use exact layout."""

    is_contiguous: bool
    dtype: torch.dtype
    shape: Optional[tuple[int, ...]] = None
    stride: Optional[tuple[int, ...]] = None

    @classmethod
    def from_slot(cls, slot: _PinSlot) -> PoolKey:
        if slot.is_contiguous:
            return cls(is_contiguous=True, dtype=slot.dtype)
        return cls(
            is_contiguous=False,
            dtype=slot.dtype,
            shape=tuple(slot.shape),
            stride=slot.stride,
        )


@dataclass
class _StepPinRecord:
    entry: _PinnedEntry
    needed_bytes: int


@dataclass(eq=False)
class _PinnedEntry:
    key: PoolKey
    tensor: Optional[torch.Tensor]
    capacity_bytes: int
    event: Optional[torch.cuda.Event] = None


class _PinSlot(Protocol):
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int
    is_contiguous: bool
    pinned_entry: Optional[_PinnedEntry]


@dataclass
class _PinnedCacheStats:
    pinned_allocations: int = 0
    pinned_allocated_bytes: int = 0
    pinned_reuses: int = 0
    pinned_reused_bytes: int = 0
    pinned_evictions: int = 0
    pinned_evicted_bytes: int = 0

    def reset(self) -> None:
        for field in fields(self):
            setattr(self, field.name, 0)


class _PinnedStagingCache:
    """Reuse pinned CPU buffers across variable-sequence-length activation offload steps."""

    def __init__(self, stats: _PinnedCacheStats) -> None:
        self._stats = stats
        self.enabled = True
        self.bucket_size_bytes = 64 << 20
        self.hard_max_size_bytes: Optional[int] = None
        self.auto_max_size_bytes: Optional[int] = None
        self._auto_shrink_streak = 0
        self._pinned_free: Dict[PoolKey, list[_PinnedEntry]] = {}
        self._pinned_fifo: OrderedDict[int, _PinnedEntry] = OrderedDict()
        self._cache_tensors = 0
        self._cache_bytes = 0
        self._step_records: list[_StepPinRecord] = []
        self._retained_capacity_counts: Dict[PoolKey, Counter] = {}
        self._host_flush_pending = False

    @property
    def retained_capacity_counts(self) -> Dict[PoolKey, Counter]:
        return self._retained_capacity_counts

    @property
    def fifo_entries(self) -> list[_PinnedEntry]:
        return list(self._pinned_fifo.values())

    def pin_memory_policy_snapshot(self) -> tuple[bool, int, Optional[int]]:
        return self.enabled, self.bucket_size_bytes, self.hard_max_size_bytes

    def configure(
        self,
        *,
        enabled: bool,
        bucket_size_bytes: int,
        hard_max_size_bytes: Optional[int],
    ) -> bool:
        """Apply pin-memory policy; return True when the retained cache must be cleared."""
        policy_changed = (
            self.enabled != enabled
            or self.bucket_size_bytes != bucket_size_bytes
            or self.hard_max_size_bytes != hard_max_size_bytes
        )
        self.enabled = enabled
        self.bucket_size_bytes = bucket_size_bytes
        self.hard_max_size_bytes = hard_max_size_bytes
        if policy_changed:
            self.clear()
            self.reset_learned_state()
        self._evict_if_needed(allow_auto=True)
        return policy_changed

    def clear(self) -> None:
        for entry in list(self._pinned_fifo.values()):
            self._evict_and_release(entry)
        self._pinned_free.clear()
        self._pinned_fifo.clear()
        self._cache_tensors = 0
        self._cache_bytes = 0
        self._step_records.clear()
        self._flush_host_cache()

    def reset_learned_state(self) -> None:
        self.auto_max_size_bytes = None
        self._auto_shrink_streak = 0
        self._retained_capacity_counts.clear()

    def cache_size(self) -> Tuple[int, int]:
        return self._cache_tensors, self._cache_bytes

    def acquire(
        self,
        slot: _PinSlot,
        *,
        offload_stream: Optional[torch.cuda.Stream],
        use_streams: bool,
    ) -> torch.Tensor:
        entry = self._pop_entry(slot)
        if entry is not None:
            if entry.event is not None and use_streams and offload_stream is not None:
                offload_stream.wait_event(entry.event)
            self._stats.pinned_reuses += 1
            self._stats.pinned_reused_bytes += slot.nbytes
        else:
            entry = self._allocate_entry(slot)
            self._stats.pinned_allocations += 1
            self._stats.pinned_allocated_bytes += entry.capacity_bytes
        slot.pinned_entry = entry
        return self._entry_view(entry, slot)

    def release(self, slot: _PinSlot, event: Optional[torch.cuda.Event]) -> None:
        entry = slot.pinned_entry
        slot.pinned_entry = None
        if entry is None:
            return
        entry.event = event
        self._pinned_free.setdefault(entry.key, []).append(entry)
        self._pinned_fifo[id(entry)] = entry
        self._cache_tensors += 1
        self._cache_bytes += entry.capacity_bytes
        self._step_records.append(_StepPinRecord(entry=entry, needed_bytes=self._needed_bytes_for_slot(slot)))
        self._evict_if_needed()

    def finalize_step(self) -> None:
        self._record_step_capacity_multisets()
        step_bytes = self._step_footprint_bytes()
        shrank = self._maybe_shrink_auto_cap(self._step_needed_footprint_bytes())
        if self.hard_max_size_bytes is None and not shrank:
            current_auto = self.auto_max_size_bytes
            if step_bytes > 0 and (current_auto is None or step_bytes > current_auto):
                self.auto_max_size_bytes = step_bytes
                self._auto_shrink_streak = 0
        self._prune_excess_cached_entries()
        if self.hard_max_size_bytes is None:
            if shrank:
                self._evict_down_to_auto_cap()
            else:
                self._evict_if_needed(allow_auto=True)
        self._step_records.clear()
        self._flush_host_cache()

    def _needed_bytes_for_slot(self, slot: _PinSlot) -> int:
        if slot.is_contiguous:
            return _round_up(slot.nbytes, self.bucket_size_bytes)
        return slot.nbytes

    def _entry_view(self, entry: _PinnedEntry, slot: _PinSlot) -> torch.Tensor:
        assert entry.tensor is not None
        if slot.is_contiguous:
            return entry.tensor.as_strided(slot.shape, slot.stride)
        return entry.tensor

    def _storage_nbytes(self, tensor: torch.Tensor) -> int:
        return tensor.untyped_storage().nbytes()

    def _schedule_host_flush(self) -> None:
        self._host_flush_pending = True

    def _flush_host_cache(self) -> None:
        if not self._host_flush_pending:
            return
        _empty_host_pin_cache()
        self._host_flush_pending = False

    def _evict_and_release(self, entry: _PinnedEntry, *, sync_event: bool = True) -> None:
        pool = self._pinned_free.get(entry.key)
        if pool is not None:
            try:
                pool.remove(entry)
            except ValueError:
                pass
            if not pool:
                self._pinned_free.pop(entry.key, None)
        self._pinned_fifo.pop(id(entry), None)
        self._cache_tensors -= 1
        self._cache_bytes -= entry.capacity_bytes
        if sync_event and entry.event is not None:
            entry.event.synchronize()
        self._step_records = [record for record in self._step_records if record.entry is not entry]
        entry.tensor = None
        self._stats.pinned_evictions += 1
        self._stats.pinned_evicted_bytes += entry.capacity_bytes
        self._schedule_host_flush()

    def _effective_max_size_bytes(self, allow_auto: bool) -> Optional[int]:
        if self.hard_max_size_bytes is not None:
            return self.hard_max_size_bytes
        if allow_auto:
            return self.auto_max_size_bytes
        return None

    def _merged_retained_counts(self) -> Counter:
        merged: Counter = Counter()
        for counts in self._retained_capacity_counts.values():
            merged.update(counts)
        return merged

    def _cached_counts_by_capacity(self) -> Counter:
        return Counter(entry.capacity_bytes for entry in self._pinned_fifo.values())

    def _pick_cap_eviction_entry(self) -> Optional[_PinnedEntry]:
        retained = self._merged_retained_counts()
        cached = self._cached_counts_by_capacity()
        for cap in sorted(cached):
            excess = cached[cap] - retained.get(cap, 0)
            if excess <= 0:
                continue
            for entry in self._pinned_fifo.values():
                if entry.capacity_bytes == cap:
                    return entry
        return None

    def _evict_if_needed(self, allow_auto: bool = False) -> None:
        max_size_bytes = self._effective_max_size_bytes(allow_auto=allow_auto)
        if max_size_bytes is None:
            return
        if max_size_bytes <= 0:
            while self._pinned_fifo:
                oldest = next(iter(self._pinned_fifo.values()))
                self._evict_and_release(oldest)
            self._flush_host_cache()
            return
        while self._cache_bytes > max_size_bytes and self._pinned_fifo:
            entry = self._pick_cap_eviction_entry()
            if entry is None:
                if allow_auto and self.hard_max_size_bytes is None and not self._auto_shrink_streak:
                    self.auto_max_size_bytes = self._cache_bytes
                break
            self._evict_and_release(entry)
        self._flush_host_cache()

    def _evict_down_to_auto_cap(self) -> None:
        max_size_bytes = self.auto_max_size_bytes
        if max_size_bytes is None:
            return
        while self._cache_bytes > max_size_bytes and self._pinned_fifo:
            entry = max(self._pinned_fifo.values(), key=lambda item: item.capacity_bytes)
            self._evict_and_release(entry)
        self._flush_host_cache()

    def _step_capacity_multiset_by_key(self) -> Dict[PoolKey, Counter]:
        by_key: Dict[PoolKey, Counter] = {}
        for record in self._step_records:
            by_key.setdefault(record.entry.key, Counter())[record.entry.capacity_bytes] += 1
        return by_key

    def _step_needed_footprint_bytes(self) -> int:
        needed_by_entry: Dict[int, int] = {}
        for record in self._step_records:
            needed_by_entry[id(record.entry)] = max(
                needed_by_entry.get(id(record.entry), 0),
                record.needed_bytes,
            )
        return sum(needed_by_entry.values())

    def _step_footprint_bytes(self) -> int:
        return sum(
            cap * count for counts in self._step_capacity_multiset_by_key().values() for cap, count in counts.items()
        )

    def _record_step_capacity_multisets(self) -> None:
        for key, step_counts in self._step_capacity_multiset_by_key().items():
            retained = self._retained_capacity_counts.setdefault(key, Counter())
            for cap, count in step_counts.items():
                retained[cap] = max(retained[cap], count)

    def _maybe_shrink_auto_cap(self, step_needed_bytes: int) -> bool:
        if self.hard_max_size_bytes is not None or step_needed_bytes <= 0:
            return False
        auto_max = self.auto_max_size_bytes
        if auto_max is not None and step_needed_bytes < auto_max:
            self._auto_shrink_streak += 1
            if self._auto_shrink_streak >= _AUTO_CAP_SHRINK_STEPS:
                self.auto_max_size_bytes = step_needed_bytes
                self._auto_shrink_streak = 0
                return True
        elif auto_max is None or step_needed_bytes >= auto_max:
            self._auto_shrink_streak = 0
        return False

    def _contiguous_smaller_tier_excess(
        self,
        cap: int,
        by_cap: Counter,
        retained: Counter,
        step_counts: Counter,
    ) -> int:
        if cap >= max(retained.keys(), default=0):
            return 0
        step_need = step_counts.get(cap, 0)
        direct_excess = by_cap[cap] - max(retained.get(cap, 0), step_need)
        if direct_excess > 0:
            return direct_excess
        if by_cap[cap] <= 0 or retained.get(cap, 0) <= step_need:
            return 0
        larger_available = sum(count for larger, count in by_cap.items() if larger > cap)
        if larger_available >= retained[cap]:
            return by_cap[cap]
        return 0

    def _prune_excess_cached_entries(self) -> None:
        fifo_set = set(self._pinned_fifo.values())
        step_by_key = self._step_capacity_multiset_by_key()
        for key, pool in list(self._pinned_free.items()):
            retained = self._retained_capacity_counts.get(key)
            if not retained:
                continue
            step_counts = step_by_key.get(key, Counter())
            entries = [entry for entry in pool if entry in fifo_set]
            by_cap = Counter(entry.capacity_bytes for entry in entries)
            for cap in sorted(by_cap):
                if key.is_contiguous:
                    excess = self._contiguous_smaller_tier_excess(cap, by_cap, retained, step_counts)
                else:
                    excess = by_cap[cap] - retained.get(cap, 0)
                if excess <= 0:
                    continue
                evicted = 0
                for entry in list(entries):
                    if entry.capacity_bytes != cap:
                        continue
                    if evicted >= excess:
                        break
                    self._evict_and_release(entry)
                    evicted += 1
                    by_cap[cap] -= 1
        self._flush_host_cache()

    def _pop_entry(self, slot: _PinSlot) -> Optional[_PinnedEntry]:
        key = PoolKey.from_slot(slot)
        pool = self._pinned_free.get(key)
        if not pool:
            return None
        if slot.is_contiguous:
            best_idx = None
            best_capacity = None
            for idx, entry in enumerate(pool):
                if entry.capacity_bytes < slot.nbytes:
                    continue
                if best_capacity is None or entry.capacity_bytes < best_capacity:
                    best_idx = idx
                    best_capacity = entry.capacity_bytes
            if best_idx is None:
                return None
            entry = pool.pop(best_idx)
        else:
            entry = pool.pop()
        if not pool:
            self._pinned_free.pop(key, None)
        self._pinned_fifo.pop(id(entry), None)
        self._cache_tensors -= 1
        self._cache_bytes -= entry.capacity_bytes
        return entry

    def _allocate_entry(self, slot: _PinSlot) -> _PinnedEntry:
        key = PoolKey.from_slot(slot)
        if slot.is_contiguous:
            element_size = torch.empty((), dtype=slot.dtype).element_size()
            capacity_bytes = _round_up(slot.nbytes, self.bucket_size_bytes)
            capacity_numel = _round_up(capacity_bytes, element_size) // element_size
            tensor = torch.empty(capacity_numel, dtype=slot.dtype, device="cpu", pin_memory=True)
        else:
            tensor = torch.empty_strided(slot.shape, slot.stride, dtype=slot.dtype, device="cpu", pin_memory=True)
        return _PinnedEntry(key=key, tensor=tensor, capacity_bytes=self._storage_nbytes(tensor))
