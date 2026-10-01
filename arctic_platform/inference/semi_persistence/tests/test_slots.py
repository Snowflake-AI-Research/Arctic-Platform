"""Unit tests for the Slots buddy allocator.

Run from the package directory:

    cd arctic_inference/semi_persistence
    python -m pytest tests/test_slots.py -v

Or directly:

    python tests/test_slots.py
"""
from __future__ import annotations

import os
import sys
import threading
import time

# Make `slots` importable when running directly: tests/ is one level
# below the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slots import Slot, Slots


def _reset() -> None:
    """Force-clear Slots state between tests (bypasses leak assertions)."""
    with Slots._cv:
        Slots._pools.clear()
        Slots._live.clear()
        Slots._last_used.clear()
        Slots._waiters.clear()
        Slots._inited = False


def test_try_allocate_returns_none_when_busy() -> None:
    _reset()
    Slots.init([0, 1])
    s0 = Slots.allocate(level=1, gpu=0)
    s1 = Slots.allocate(level=1, gpu=1)

    # Both GPUs fully held -> try_allocate must return None.
    assert Slots.try_allocate(level=1) is None
    assert Slots.try_allocate(level=1, gpu=0) is None
    assert Slots.try_allocate(level=1, gpu=1) is None

    Slots.deallocate(s0)
    # Now GPU 0 is free; try_allocate(gpu=0) should succeed.
    s_again = Slots.try_allocate(level=1, gpu=0)
    assert s_again is not None
    assert s_again.gpu_id == 0

    Slots.deallocate(s_again)
    Slots.deallocate(s1)
    _reset()


def test_try_allocate_specific_gpu() -> None:
    _reset()
    Slots.init([0, 1])
    s0 = Slots.allocate(level=1, gpu=0)

    # GPU 0 is busy, but GPU 1 is free -- specific request for 0
    # should fail without poaching GPU 1.
    assert Slots.try_allocate(level=1, gpu=0) is None

    s1 = Slots.try_allocate(level=1, gpu=1)
    assert s1 is not None
    assert s1.gpu_id == 1

    Slots.deallocate(s0)
    Slots.deallocate(s1)
    _reset()


def test_fifo_blocking_allocate() -> None:
    """Concurrent waiters get served in FIFO order."""
    _reset()
    Slots.init([0])
    holder = Slots.allocate(level=1, gpu=0)

    # Spawn three waiters on the only GPU; they queue up.
    order: list[int] = []
    barrier = threading.Barrier(4)  # main + 3 waiters

    def waiter(tag: int) -> None:
        barrier.wait()
        time.sleep(0.01 * tag)  # stagger arrival to make FIFO order well-defined
        s = Slots.allocate(level=1, gpu=0)
        order.append(tag)
        time.sleep(0.05)
        Slots.deallocate(s)

    threads = [threading.Thread(target=waiter, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    barrier.wait()
    time.sleep(0.1)  # let all three enqueue
    Slots.deallocate(holder)

    for t in threads:
        t.join(timeout=5)
    assert order == [0, 1, 2], f"FIFO violated: {order}"
    _reset()


def test_l1_l2_mixed() -> None:
    _reset()
    Slots.init([0])

    # Two L2 allocs fill one GPU.
    a = Slots.allocate(level=2, gpu=0)
    b = Slots.allocate(level=2, gpu=0)
    assert a.level == 2 and b.level == 2

    # Now an L1 try_allocate must fail (both halves taken).
    assert Slots.try_allocate(level=1, gpu=0) is None
    assert Slots.try_allocate(level=1) is None

    # And another L2 also fails (no free pool entries).
    assert Slots.try_allocate(level=2, gpu=0) is None

    # Free one half -> still no L1 (other half held).
    Slots.deallocate(a)
    assert Slots.try_allocate(level=1, gpu=0) is None
    # An L2 succeeds though.
    c = Slots.try_allocate(level=2, gpu=0)
    assert c is not None and c.level == 2

    # Free both halves -> coalesce -> L1 available.
    Slots.deallocate(b)
    Slots.deallocate(c)
    big = Slots.try_allocate(level=1, gpu=0)
    assert big is not None and big.level == 1
    Slots.deallocate(big)
    _reset()


def test_l1_waiter_unblocked_by_l2_coalesce() -> None:
    _reset()
    Slots.init([0])
    a = Slots.allocate(level=2, gpu=0)
    b = Slots.allocate(level=2, gpu=0)

    result: list[Slot] = []

    def waiter() -> None:
        result.append(Slots.allocate(level=1, gpu=0))

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.05)  # waiter is at FIFO head, blocked on L1
    assert not result

    # Free one half -> coalesce not yet (other half held) -> still blocked.
    Slots.deallocate(a)
    time.sleep(0.05)
    assert not result

    # Free other half -> coalesce -> L1 available -> waiter unblocks.
    Slots.deallocate(b)
    t.join(timeout=5)
    assert len(result) == 1
    assert result[0].level == 1
    Slots.deallocate(result[0])
    _reset()


def test_coldest_first_auto_pick() -> None:
    _reset()
    Slots.init([0, 1, 2])

    # Free GPUs are ordered by (_last_used, gpu_id) ascending. Initially
    # all are 0.0 so GPU 0 is picked first.
    s0 = Slots.allocate(level=1)
    assert s0.gpu_id == 0
    s1 = Slots.allocate(level=1)
    assert s1.gpu_id == 1
    s2 = Slots.allocate(level=1)
    assert s2.gpu_id == 2

    # Release GPU 1 first, then GPU 0. GPU 1 has a smaller _last_used,
    # so the next allocate should prefer GPU 1.
    Slots.deallocate(s1)
    time.sleep(0.001)
    Slots.deallocate(s0)
    s_next = Slots.allocate(level=1)
    assert s_next.gpu_id == 1, f"expected coldest=1, got {s_next.gpu_id}"

    Slots.deallocate(s_next)
    Slots.deallocate(s2)
    _reset()


if __name__ == "__main__":
    tests = [
        test_try_allocate_returns_none_when_busy,
        test_try_allocate_specific_gpu,
        test_fifo_blocking_allocate,
        test_l1_l2_mixed,
        test_l1_waiter_unblocked_by_l2_coalesce,
        test_coldest_first_auto_pick,
    ]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc!r}")
    sys.exit(1 if failures else 0)
