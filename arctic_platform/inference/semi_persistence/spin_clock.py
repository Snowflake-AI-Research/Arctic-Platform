"""Time vLLM's shm_broadcast idle spin on the wall clock, so it survives a restore.

A TP worker waiting for its next RPC sits in ``SpinCondition.wait``
(``vllm/distributed/device_communicators/shm_broadcast.py``), which yields the
CPU while ``time.monotonic() <= last_read + busy_loop_s`` and otherwise blocks
in its zmq poller. CLOCK_MONOTONIC counts from the host's boot, and a restore
cannot carry it over -- that needs a time namespace, which needs CAP_SYS_ADMIN,
and the pod runs with none. So on a restore host that booted later than the
dump host, ``last_read`` lies in the future, and every rank busy-yields until
its next read. The quiescence wait before ``cuda_restore`` sees them in state R
and runs out its whole budget: 30 s of the 54.8 s ``cuda_restore`` measured on
GLM-5.3-Flash-BF16, whose image was dumped on the pool's oldest node.

``install()`` makes the readers time that window on CLOCK_REALTIME instead,
which is NTP-synced across nodes. The dump-to-restore gap is then real elapsed
time, and a restored rank is blocked in the poller (state S) from its first
wait. It is installed in every TP worker before the dump and never withdrawn,
so the CRIU image carries it. The clock only chooses between yielding and
blocking: a wall-clock step can at worst lengthen or shorten one busy window.
"""
from __future__ import annotations

import math
import time

_state = {"active": False, "err": None}


def install() -> dict:
    """Patch ``SpinCondition`` in place. Idempotent; returns the state."""
    if _state["active"]:
        return dict(_state)
    try:
        from vllm.distributed.device_communicators import shm_broadcast
    except Exception as exc:  # noqa: BLE001 - never load-bearing for the engine
        _state["err"] = f"{type(exc).__name__}: {exc}"
        return dict(_state)
    cls = getattr(shm_broadcast, "SpinCondition", None)
    if cls is None or not all(hasattr(cls, name) for name in ("record_read", "wait")):
        _state["err"] = "shm_broadcast has no SpinCondition.record_read/wait"
        return dict(_state)
    upstream_wait = cls.wait

    def record_read(self):
        self.last_read = time.time()

    def wait(self, timeout_ms=None):
        assert self.is_reader, "Only readers can wait"
        if time.time() <= self.last_read + self.busy_loop_s:
            shm_broadcast.sched_yield()
            return
        # Upstream's idle branch, reached by making its own busy test fail.
        busy_loop_s, self.busy_loop_s = self.busy_loop_s, -math.inf
        try:
            upstream_wait(self, timeout_ms)
        finally:
            self.busy_loop_s = busy_loop_s

    cls.record_read = record_read
    cls.wait = wait
    _state.update(active=True, err=None)
    return dict(_state)
