"""Buddy-allocator for fractional GPU slots.

A singleton ``Slots`` class hands out level-based slots from a pool of GPUs.
Each GPU starts as one whole slot (level 1) and can be recursively split in
halves: level 2 = half a GPU, level 3 = a quarter, etc.  A level-L slot
covers ``1 / 2**(L-1)`` of a GPU.

The bookkeeping is a classic Knuth/Knowlton buddy allocator stored as
free-lists per ``(gpu_id, level)``.  No explicit tree is materialised; the
implicit tree is only walked by :meth:`Slots.status` for printing.

See ``slots_DESIGN.md`` for the full design rationale.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import semip_logging
from abstract import SlotsBase

log = semip_logging.slots()


@dataclass(frozen=True)
class Slot:
    gpu_id: int
    level: int
    index: int


@dataclass
class _Waiter:
    level: int
    gpu: int | None  # None means coldest-first auto-pick


def _buddy_index(i: int) -> int:
    return i ^ 1


def _split(s: Slot) -> tuple[Slot, Slot]:
    return (Slot(s.gpu_id, s.level + 1, 2 * s.index),
            Slot(s.gpu_id, s.level + 1, 2 * s.index + 1))


def _parent(s: Slot) -> Slot:
    return Slot(s.gpu_id, s.level - 1, s.index // 2)


class Slots(SlotsBase):
    """Singleton buddy-allocator.  All methods are class-level."""

    _lock: threading.Lock = threading.Lock()
    _cv: threading.Condition = threading.Condition(_lock)
    _pools: dict[tuple[int, int], deque[Slot]] = {}
    _live: set[Slot] = set()
    _last_used: dict[int, float] = {}
    _waiters: "deque[_Waiter]" = deque()
    _draining: set[int] = set()
    _inited: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def init(cls, gpu_ids: list[int]) -> None:
        with cls._cv:
            assert not cls._inited, "Slots already initialised"
            cls._pools.clear()
            cls._live.clear()
            cls._last_used.clear()
            cls._waiters.clear()
            cls._draining.clear()
            for g in gpu_ids:
                cls._pools[(g, 1)] = deque([Slot(g, 1, 0)])
                cls._last_used[g] = 0.0
            cls._inited = True

    @classmethod
    def remove(cls) -> None:
        with cls._cv:
            assert cls._inited, "Slots not initialised"
            assert not cls._live, f"leaked slots: {sorted(cls._live)}"
            for g in cls._last_used:
                root_pool = cls._pools.get((g, 1))
                assert root_pool and len(root_pool) == 1, (
                    f"GPU {g} root pool not whole: {list(root_pool or [])}")
            for (g, L), pool in cls._pools.items():
                if L != 1:
                    assert not pool, f"non-empty pool ({g}, {L}): {list(pool)}"
            cls._pools.clear()
            cls._live.clear()
            cls._last_used.clear()
            cls._waiters.clear()
            cls._draining.clear()
            cls._inited = False

    @classmethod
    def add(cls, gpu: int) -> bool:
        """Add *gpu* to the pool with a single whole-GPU root slot.

        Returns True if the GPU was added, False if it was already in
        the pool (idempotent).  Wakes FIFO waiters that may now be
        satisfiable on the new GPU.
        """
        with cls._cv:
            assert cls._inited, "Slots not initialised"
            if gpu in cls._last_used:
                return False
            cls._pools[(gpu, 1)] = deque([Slot(gpu, 1, 0)])
            cls._last_used[gpu] = 0.0
            cls._draining.discard(gpu)
            cls._cv.notify_all()
            log.info("add GPU %d", gpu)
            return True

    @classmethod
    def pop(cls, gpu: int) -> None:
        """Remove a fully-idle *gpu* from the pool.

        Asserts no live slot or non-root-free pool entry references the
        GPU.  Caller (e.g. ``Orchestrator._sub_sync``) is
        responsible for marking the GPU draining and ensuring residents
        have moved off before invoking this.
        """
        with cls._cv:
            assert cls._inited, "Slots not initialised"
            assert not any(s.gpu_id == gpu for s in cls._live), (
                f"GPU {gpu} still has live slots: "
                f"{sorted(s for s in cls._live if s.gpu_id == gpu)}")
            root = cls._pools.get((gpu, 1))
            assert root and len(root) == 1, (
                f"GPU {gpu} root pool not whole: {list(root or [])}")
            for key in [k for k in cls._pools if k[0] == gpu]:
                pool = cls._pools[key]
                if key[1] != 1:
                    assert not pool, (
                        f"non-empty pool {key}: {list(pool)}")
                cls._pools.pop(key)
            cls._last_used.pop(gpu, None)
            cls._draining.discard(gpu)
            cls._cv.notify_all()
            log.info("pop GPU %d", gpu)

    # ------------------------------------------------------------------
    # Allocation core (private, lock held)
    # ------------------------------------------------------------------

    @classmethod
    def _pop_free_in_subtree(cls, gpu: int, level: int) -> Slot | None:
        for L in range(level, 0, -1):
            pool = cls._pools.get((gpu, L))
            if pool:
                s = pool.popleft()
                while s.level < level:
                    a, b = _split(s)
                    cls._pools.setdefault((gpu, b.level), deque()).append(b)
                    s = a
                return s
        return None

    @classmethod
    def _try_allocate(cls, level: int, gpu: int | None) -> Slot | None:
        if gpu is not None:
            if gpu in cls._draining:
                return None
            return cls._pop_free_in_subtree(gpu, level)
        for g in sorted(cls._last_used,
                        key=lambda g: (cls._last_used[g], g)):
            if g in cls._draining:
                continue
            s = cls._pop_free_in_subtree(g, level)
            if s is not None:
                return s
        return None

    # ------------------------------------------------------------------
    # Public allocate / deallocate
    # ------------------------------------------------------------------

    @classmethod
    def allocate(cls, level: int, gpu: int | None = None,
                 on_block: Callable[[], None] | None = None) -> Slot:
        """Block until a level-*level* slot is available, then return it.

        If *gpu* is ``None``, picks the coldest GPU that can satisfy the
        request.  Strict head-of-line FIFO among waiters.

        If *on_block* is provided, it is invoked at most once -- the
        first time this call has to actually wait on the condition
        variable (i.e. the slot was not immediately available).  When
        the request is satisfied on entry, *on_block* never fires.
        Callers can use this to publish a transient "waiting for slot"
        state only when there's real waiting.  The callback runs while
        holding the Slots condition variable, so it must be cheap and
        must not call back into ``Slots``.
        """
        assert level >= 1
        with cls._cv:
            assert cls._inited, "Slots not initialised"
            log.info("allocate L%d gpu=%s (waiters=%d)",
                     level, gpu, len(cls._waiters))
            me = _Waiter(level, gpu)
            cls._waiters.append(me)
            blocked_announced = False
            while True:
                if cls._waiters[0] is me:
                    s = cls._try_allocate(level, gpu)
                    if s is not None:
                        cls._waiters.popleft()
                        cls._live.add(s)
                        cls._cv.notify_all()
                        log.info("allocate L%d gpu=%s -> %s", level, gpu, s)
                        return s
                if on_block is not None and not blocked_announced:
                    blocked_announced = True
                    log.info("allocate L%d gpu=%s blocked", level, gpu)
                    on_block()
                cls._cv.wait()

    @classmethod
    def try_allocate(cls, level: int, gpu: int | None = None) -> Slot | None:
        """Non-blocking allocate.  Returns ``None`` if not satisfiable now.

        Bypasses the FIFO waiter queue.  Intended for opportunistic
        fast-path acquisition (e.g. a sleeper that lost its slot trying
        to re-acquire the same GPU); callers that must respect FIFO
        ordering should call :meth:`allocate`.
        """
        assert level >= 1
        with cls._cv:
            assert cls._inited, "Slots not initialised"
            s = cls._try_allocate(level, gpu)
            if s is not None:
                cls._live.add(s)
                cls._cv.notify_all()
                log.info("try_allocate L%d gpu=%s -> %s", level, gpu, s)
            else:
                log.debug("try_allocate L%d gpu=%s -> None", level, gpu)
            return s

    @classmethod
    def deallocate(cls, slot: Slot) -> None:
        """Release *slot* and coalesce buddies upward."""
        with cls._cv:
            assert cls._inited, "Slots not initialised"
            assert slot in cls._live, f"unknown slot: {slot}"
            cls._live.remove(slot)
            s = slot
            while s.level > 1:
                pool = cls._pools.get((s.gpu_id, s.level))
                if not pool:
                    break
                bi = _buddy_index(s.index)
                buddy = next((x for x in pool if x.index == bi), None)
                if buddy is None:
                    break
                pool.remove(buddy)
                s = _parent(s)
            cls._pools.setdefault((s.gpu_id, s.level), deque()).append(s)
            log.info("deallocate %s -> coalesced %s", slot, s)
            if s.level == 1:
                cls._last_used[s.gpu_id] = time.perf_counter()
            cls._cv.notify_all()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @classmethod
    def status(cls) -> None:
        """Print a tree-style view of every GPU's slot state."""
        with cls._cv:
            if not cls._inited:
                print("Slots: <not initialised>")
                return
            pool_set = {s for pool in cls._pools.values() for s in pool}
            live_set = set(cls._live)
            last_used = dict(cls._last_used)
            waiters = list(cls._waiters)

        def label(node: Slot) -> str:
            if node in pool_set:
                return "FREE"
            if node in live_set:
                return "ALLOC"
            return "SPLIT"

        def render(node: Slot, prefix: str, is_last: bool) -> None:
            connector = "└── " if is_last else "├── "
            tag = label(node)
            print(f"{prefix}{connector}L{node.level}[{node.index}] {tag}")
            if tag == "SPLIT":
                a, b = _split(node)
                new_prefix = prefix + ("    " if is_last else "│   ")
                render(a, new_prefix, False)
                render(b, new_prefix, True)

        now = time.perf_counter()

        def fmt_last_used(t: float) -> str:
            if t == 0.0:
                return "never"
            return f"{now - t:.2f}s ago"

        print("Slots:")
        for g in sorted(last_used):
            print(f"  GPU {g} (last_used={fmt_last_used(last_used[g])})")
            root = Slot(g, 1, 0)
            tag = label(root)
            print(f"    L1[0] {tag}")
            if tag == "SPLIT":
                a, b = _split(root)
                render(a, "    ", False)
                render(b, "    ", True)

        print(f"  Waiters: {len(waiters)}")
        for i, w in enumerate(waiters):
            head = "head" if i == 0 else f"[{i}] "
            print(f"    {head}: level={w.level}, gpu={w.gpu}")
