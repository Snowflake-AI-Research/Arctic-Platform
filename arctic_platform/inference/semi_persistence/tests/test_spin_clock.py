"""A restored TP rank must be idle, not spinning, whichever host it lands on.

vLLM's ``SpinCondition`` busy-yields while ``time.monotonic() <= last_read +
busy_loop_s``. CLOCK_MONOTONIC is the host's uptime and a restore cannot carry
it over, so a rank dumped on a host up 87 days and restored on one up 29 holds
a ``last_read`` 58 days in the future: it spins in state R, and the quiescence
wait before ``cuda_restore`` burns its full 30 s. That is the 54.8 s
``cuda_restore`` measured on GLM-5.3-Flash-BF16 against 14 s on images dumped
on younger hosts, and the node boot times matched on every restore.

``spin_clock`` times the window on the wall clock instead. The upstream class
is replaced here by a double with the same busy test and the same two
branches, and the two clocks are driven by hand: a restore moves the
monotonic clock backwards and the wall clock forwards.

Run from the package directory::

    cd arctic_inference/semi_persistence
    python -m pytest tests/test_spin_clock.py -v
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import os
import sys
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                      # .../semi_persistence
_SPIN_CLOCK = os.path.join(_PKG, "spin_clock.py")
_WORKER = os.path.join(_PKG, "_semip_worker.py")

_DAY = 86400.0
_DUMP_UPTIME = 87 * _DAY
_RESTORE_UPTIME = 29 * _DAY
_WALL = 1.79e9


class _Clocks:
    def __init__(self):
        self.mono = _DUMP_UPTIME
        self.wall = _WALL

    def restore(self, gap_s=600.0):
        self.mono = _RESTORE_UPTIME
        self.wall += gap_s


def _fake_shm_broadcast(clocks, with_class=True):
    """``shm_broadcast`` with an upstream-shaped ``SpinCondition`` (0.32.0)."""
    mod = types.ModuleType("vllm.distributed.device_communicators.shm_broadcast")
    mod.time = types.SimpleNamespace(monotonic=lambda: clocks.mono)
    mod.calls = []
    mod.sched_yield = lambda: mod.calls.append("yield")

    class SpinCondition:
        def __init__(self, is_reader=True, busy_loop_s=1):
            self.is_reader = is_reader
            self.busy_loop_s = busy_loop_s if is_reader else 0
            self.last_read = mod.time.monotonic() if is_reader else 0

        def record_read(self):
            self.last_read = mod.time.monotonic()

        def wait(self, timeout_ms=None):
            assert self.is_reader, "Only readers can wait"
            current_time = mod.time.monotonic()
            if current_time <= self.last_read + self.busy_loop_s:
                mod.sched_yield()
            else:
                mod.calls.append(("poll", timeout_ms))

    if with_class:
        mod.SpinCondition = SpinCondition
    return mod


class _NotifySocket:
    def __init__(self, calls):
        self.calls = calls
        self.pending = 0

    def poll(self, timeout=None):
        self.calls.append(("notify_poll", timeout))
        return self.pending

    def recv(self, flags=0, copy=True):
        self.calls.append(("notify_recv", flags))
        self.pending = 0


def _fake_shm_broadcast_033(clocks):
    """``shm_broadcast`` at 0.33.0, whose busy readers poll the notify socket
    (vllm#63165)."""
    mod = _fake_shm_broadcast(clocks)
    mod.SHM_NOTIFY_POLL_INTERVAL_S = 2.0
    mod.zmq = types.SimpleNamespace(NOBLOCK=1)
    base = mod.SpinCondition

    class SpinCondition(base):
        def __init__(self, is_reader=True, busy_loop_s=1):
            super().__init__(is_reader, busy_loop_s)
            self.last_notify_poll = self.last_read
            self.local_notify_socket = _NotifySocket(mod.calls)

        def record_read(self):
            self.last_read = mod.time.monotonic()
            if self.last_read - self.last_notify_poll >= mod.SHM_NOTIFY_POLL_INTERVAL_S:
                self.last_notify_poll = self.last_read
                if self.local_notify_socket.poll(timeout=0):
                    self.local_notify_socket.recv(flags=mod.zmq.NOBLOCK, copy=False)

    mod.SpinCondition = SpinCondition
    return mod


@contextlib.contextmanager
def _vllm(shm_broadcast):
    """Register ``vllm.distributed.device_communicators.shm_broadcast``."""
    names = ["vllm", "vllm.distributed", "vllm.distributed.device_communicators"]
    mods = [types.ModuleType(n) for n in names] + [shm_broadcast]
    for parent, child in zip(mods, mods[1:]):
        setattr(parent, child.__name__.rsplit(".", 1)[1], child)
    saved = {m.__name__: sys.modules.get(m.__name__) for m in mods}
    sys.modules.update({m.__name__: m for m in mods})
    try:
        yield
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


@pytest.fixture
def clocks():
    return _Clocks()


@pytest.fixture
def spin_clock(clocks):
    """A fresh module per test: the installed state is module-level."""
    spec = importlib.util.spec_from_file_location("spin_clock_under_test",
                                                  _SPIN_CLOCK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.time = types.SimpleNamespace(time=lambda: clocks.wall)
    return mod


def test_upstream_double_spins_after_restore_to_a_younger_host(clocks):
    sb = _fake_shm_broadcast(clocks)
    cond = sb.SpinCondition()
    cond.record_read()
    clocks.restore()
    cond.wait(timeout_ms=100)
    assert sb.calls == ["yield"]


def test_installed_rank_is_idle_after_restore_to_a_younger_host(clocks, spin_clock):
    sb = _fake_shm_broadcast(clocks)
    with _vllm(sb):
        assert spin_clock.install() == {"active": True, "err": None}
    cond = sb.SpinCondition()
    cond.record_read()
    clocks.restore()
    cond.wait(timeout_ms=100)
    assert sb.calls == [("poll", 100)]
    assert cond.busy_loop_s == 1


def test_installed_rank_still_spins_right_after_a_read(clocks, spin_clock):
    sb = _fake_shm_broadcast(clocks)
    with _vllm(sb):
        spin_clock.install()
    cond = sb.SpinCondition()
    cond.record_read()
    clocks.wall += 0.5
    cond.wait()
    clocks.wall += 1.0
    cond.wait()
    assert sb.calls == ["yield", ("poll", None)]


def test_reader_created_before_install_goes_idle(clocks, spin_clock):
    sb = _fake_shm_broadcast(clocks)
    cond = sb.SpinCondition()
    with _vllm(sb):
        spin_clock.install()
    cond.wait()
    assert sb.calls == [("poll", None)]


def test_idle_branch_restores_busy_window_when_upstream_raises(clocks, spin_clock):
    sb = _fake_shm_broadcast(clocks)

    def boom(self, timeout_ms=None):
        raise RuntimeError("cancelled")

    sb.SpinCondition.wait = boom
    with _vllm(sb):
        spin_clock.install()
    cond = sb.SpinCondition()
    cond.record_read()
    clocks.restore()
    with pytest.raises(RuntimeError):
        cond.wait()
    assert cond.busy_loop_s == 1


def _notify_polls(sb):
    return [c for c in sb.calls if c[0] == "notify_poll"]


def test_installed_busy_reader_polls_the_notify_socket_every_interval(clocks, spin_clock):
    sb = _fake_shm_broadcast_033(clocks)
    with _vllm(sb):
        spin_clock.install()
    cond = sb.SpinCondition()
    cond.record_read()              # last_notify_poll still on the monotonic clock
    assert _notify_polls(sb) == [("notify_poll", 0)]
    for _ in range(3):
        clocks.wall += 0.5
        cond.record_read()
    assert len(_notify_polls(sb)) == 1
    clocks.wall += 0.5
    cond.local_notify_socket.pending = 1
    cond.record_read()
    assert len(_notify_polls(sb)) == 2
    assert sb.calls[-1] == ("notify_recv", 1)
    assert cond.last_notify_poll == cond.last_read == clocks.wall


def test_installed_busy_reader_keeps_polling_after_restore(clocks, spin_clock):
    sb = _fake_shm_broadcast_033(clocks)
    with _vllm(sb):
        spin_clock.install()
    cond = sb.SpinCondition()
    cond.record_read()
    clocks.restore()
    cond.record_read()
    assert len(_notify_polls(sb)) == 2
    cond.wait()
    assert sb.calls[-1] == "yield"


def test_wall_clock_step_back_polls_instead_of_stalling(clocks, spin_clock):
    sb = _fake_shm_broadcast_033(clocks)
    with _vllm(sb):
        spin_clock.install()
    cond = sb.SpinCondition()
    cond.record_read()
    clocks.wall -= 3600.0
    cond.record_read()
    assert len(_notify_polls(sb)) == 2


def test_install_is_idempotent(clocks, spin_clock):
    sb = _fake_shm_broadcast(clocks)
    with _vllm(sb):
        spin_clock.install()
        wait = sb.SpinCondition.wait
        assert spin_clock.install()["active"] is True
    assert sb.SpinCondition.wait is wait


def test_missing_spin_condition_is_reported_not_raised(clocks, spin_clock):
    with _vllm(_fake_shm_broadcast(clocks, with_class=False)):
        info = spin_clock.install()
    assert info["active"] is False
    assert "SpinCondition" in info["err"]


def test_missing_vllm_is_reported_not_raised(spin_clock):
    saved = {n: sys.modules.pop(n) for n in list(sys.modules)
             if n == "vllm" or n.startswith("vllm.")}
    sys.modules["vllm"] = None
    try:
        info = spin_clock.install()
    finally:
        del sys.modules["vllm"]
        sys.modules.update(saved)
    assert info["active"] is False
    assert info["err"]


def test_worker_installs_in_init_device():
    tree = ast.parse(open(_WORKER).read())
    worker = next(n for n in tree.body
                  if isinstance(n, ast.ClassDef) and n.name == "SemipGPUWorker")
    methods = {n.name: n for n in worker.body if isinstance(n, ast.FunctionDef)}

    def calls(fn):
        return {n.func.attr for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}

    assert "_semip_install_spin_clock" in calls(methods["init_device"])
    assert "install" in calls(methods["_semip_install_spin_clock"])
