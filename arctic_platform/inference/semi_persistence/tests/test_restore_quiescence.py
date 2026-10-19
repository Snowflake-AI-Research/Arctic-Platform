"""Unit tests for the post-CRIU quiescence wait and its timeout budgets.

This is the guard on the bug that produced every restore hang we have seen:
``cuda_restore`` waited for the restored pids to settle, and when they did not
it **skipped the CUDA restore and logged ``OK`` anyway**. The ranks came back
holding zero ``/dev/nvidia`` fds, ``reinit_nccl`` deadlocked in futex, and the
job sat idle for its full timeout with nothing in the log to say why. Measured
across eighteen restores, every hang had the skip and no non-hang did.

Two properties matter and both are covered here: the wait must be against a
single deadline for the whole set (the per-pid form cost ``timeout x pids``,
so a TP=8 tree burned 45 s to report what a TP=2 tree reported in 10 s), and a
pid that never settles must be *returned*, so the caller raises rather than
carrying on.

No cluster, no GPU, no ``/proc`` and no Linux: ``_proc_state`` is the seam, and
the device scan reads a glob pattern held in a module attribute.

Run from the package directory:

    cd arctic_inference/semi_persistence
    python -m pytest tests/test_restore_quiescence.py -v
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                      # .../semi_persistence
_WORKER = os.path.join(_PKG, "worker.py")


def _load_worker():
    """Import worker.py without pulling in NVML, torch or the log plumbing.

    ``worker.py`` imports all three at module scope for the runtime paths;
    none of them is reachable from the helpers under test, so a stub each is
    enough and keeps this hermetic. If worker.py grows a fourth module-scope
    dependency, this bootstrap is what will notice.

    The stubs are withdrawn from ``sys.modules`` afterwards. Leaving them
    there is not harmless: ``test_slots.py`` imports the *real*
    ``semip_logging`` and fails on the stub, and which of us runs first is
    just collection order. The loaded module keeps its own references, so
    nothing here depends on them staying registered.
    """
    stubs = {}
    torch = types.ModuleType("torch")
    torch.multiprocessing = types.ModuleType("torch.multiprocessing")
    stubs["pynvml"] = types.ModuleType("pynvml")
    stubs["torch"] = torch
    stubs["torch.multiprocessing"] = torch.multiprocessing
    stubs["semip_logging"] = types.ModuleType("semip_logging")

    added = [name for name in stubs if name not in sys.modules]
    for name in added:
        sys.modules[name] = stubs[name]

    sys.path.insert(0, _PKG)
    try:
        spec = importlib.util.spec_from_file_location("worker_under_test",
                                                      _WORKER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(_PKG)
        for name in added:
            sys.modules.pop(name, None)
    return module


w = _load_worker()


class _FakeProc:
    """Scripted ``/proc/<pid>/status`` states, one step per poll."""

    def __init__(self, timelines):
        # pid -> list of states; the last entry repeats forever.
        self.timelines = {pid: list(states) for pid, states in timelines.items()}
        self.calls = {pid: 0 for pid in timelines}

    def __call__(self, pid):
        states = self.timelines[pid]
        i = min(self.calls[pid], len(states) - 1)
        self.calls[pid] += 1
        return states[i]


def _with_proc(monkey_states):
    fake = _FakeProc(monkey_states)
    w._proc_state = fake
    return fake


# --------------------------------------------------------------------------
# _wait_for_quiescence
# --------------------------------------------------------------------------


def test_all_sleeping_returns_immediately():
    _with_proc({1: ["S (sleeping)"], 2: ["S (sleeping)"]})
    unsettled, states = w._wait_for_quiescence([1, 2], timeout_s=5)
    assert unsettled == []
    assert states == {1: "S (sleeping)", 2: "S (sleeping)"}


def test_stopped_counts_as_settled():
    """'T' is what a CRIU-restored, still-frozen task reports."""
    _with_proc({1: ["T (stopped)"]})
    unsettled, _ = w._wait_for_quiescence([1], timeout_s=5)
    assert unsettled == []


def test_pid_that_exits_counts_as_settled():
    """Nothing left to restore for a pid that is gone -- not a failure."""
    _with_proc({1: [None], 2: ["S (sleeping)"]})
    unsettled, states = w._wait_for_quiescence([1, 2], timeout_s=5)
    assert unsettled == []
    assert 1 not in states


def test_running_pid_is_returned_with_its_state():
    """The caller raises on this, and the state is the only clue to why."""
    _with_proc({1: ["S (sleeping)"], 2: ["R (running)"]})
    unsettled, states = w._wait_for_quiescence([1, 2], timeout_s=0.2)
    assert unsettled == [2]
    assert states[2] == "R (running)"


def test_pid_that_settles_late_is_waited_for():
    _with_proc({1: ["R (running)", "R (running)", "S (sleeping)"]})
    unsettled, _ = w._wait_for_quiescence([1], timeout_s=5)
    assert unsettled == []


def test_deadline_is_shared_across_pids_not_per_pid():
    """The regression that made a TP=8 failure cost 45 s to report.

    Three pids that never settle must cost one timeout between them, not
    three. Generous upper bound so this does not flake on a loaded laptop;
    the old per-pid form would take 3x the budget and blow even that.
    """
    _with_proc({p: ["R (running)"] for p in (1, 2, 3)})
    budget = 0.3
    t0 = time.monotonic()
    unsettled, _ = w._wait_for_quiescence([1, 2, 3], timeout_s=budget)
    elapsed = time.monotonic() - t0
    assert sorted(unsettled) == [1, 2, 3]
    assert elapsed < budget * 2.5, f"took {elapsed:.2f}s against a {budget}s budget"


def test_settled_pids_are_not_repolled():
    """Once a pid is sleeping it drops out of the loop."""
    fake = _with_proc({1: ["S (sleeping)"], 2: ["R (running)"]})
    w._wait_for_quiescence([1, 2], timeout_s=0.2)
    assert fake.calls[1] == 1
    assert fake.calls[2] > 1


# --------------------------------------------------------------------------
# _is_userspace_spin
# --------------------------------------------------------------------------
#
# The predicate that decides whether a pid which missed the window is a failure
# or merely spinning. Root-caused 2026-09-22: vLLM's SpinCondition busy-loop runs
# against a CLOCK_MONOTONIC the restore could not preserve, so when the restore
# host booted later than the dump host the loop never ends and no budget outlasts
# it. Such a rank has no CUDA context yet, so it cannot be doing GPU work and is
# safe to checkpoint. A rank blocked in the kernel is not, and must still fail.


def _with_proc_files(wchan=None, syscall=None):
    """Stub the two /proc reads. None means the file could not be read."""
    w._proc_wchan = lambda _pid: wchan
    w._proc_syscall = lambda _pid: syscall


def test_userspace_spin_is_recognised():
    """wchan=0 and syscall=running: not in the kernel, not in a syscall.

    Exactly what the in-pod sampler recorded for every stuck rank, alongside
    100% of one core sustained.
    """
    _with_proc_files(wchan="0", syscall="running")
    assert w._is_userspace_spin(1234) is True


def test_a_pid_blocked_in_the_kernel_is_not_a_spin():
    _with_proc_files(wchan="futex_wait_queue", syscall="202")
    assert w._is_userspace_spin(1234) is False


def test_a_pid_inside_a_syscall_is_not_a_spin():
    """wchan can read 0 while the task is still in a syscall; both must agree."""
    _with_proc_files(wchan="0", syscall="202")
    assert w._is_userspace_spin(1234) is False


def test_unreadable_proc_files_are_not_a_spin():
    """Conservative on error: an unreadable /proc keeps the old failure."""
    for wchan, syscall in (("0", None), (None, "running"), (None, None)):
        _with_proc_files(wchan=wchan, syscall=syscall)
        assert w._is_userspace_spin(1234) is False, (wchan, syscall)


# --------------------------------------------------------------------------
# timeout budgets
# --------------------------------------------------------------------------


def _env(name, value):
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def test_settle_timeout_defaults_when_unset():
    _env(w._SETTLE_TIMEOUT_ENV, None)
    assert w._settle_timeout_s() == w._SETTLE_TIMEOUT_DEFAULT_S


def test_settle_timeout_honours_the_environment():
    """This process is the Ray actor, not a restored one, so extra_env lands."""
    _env(w._SETTLE_TIMEOUT_ENV, "12.5")
    try:
        assert w._settle_timeout_s() == 12.5
    finally:
        _env(w._SETTLE_TIMEOUT_ENV, None)


def test_unusable_budgets_fall_back_rather_than_raise():
    """A typo in a diagnostics knob must not be what fails a restore."""
    for bad in ("", "abc", "0", "-1"):
        _env(w._SETTLE_TIMEOUT_ENV, bad)
        try:
            assert w._settle_timeout_s() == w._SETTLE_TIMEOUT_DEFAULT_S, bad
        finally:
            _env(w._SETTLE_TIMEOUT_ENV, None)


def test_reinit_timeout_is_separately_tunable():
    _env(w._REINIT_TIMEOUT_ENV, "45")
    try:
        assert w._reinit_timeout_s() == 45
        assert w._settle_timeout_s() == w._SETTLE_TIMEOUT_DEFAULT_S
    finally:
        _env(w._REINIT_TIMEOUT_ENV, None)


# --------------------------------------------------------------------------
# _visible_device_nodes
# --------------------------------------------------------------------------


def test_device_nodes_are_basenames_and_sorted():
    with tempfile.TemporaryDirectory() as d:
        for name in ("nvidia3", "nvidia0", "nvidia11"):
            open(os.path.join(d, name), "w").close()
        saved = w._DEVICE_GLOBS
        w._DEVICE_GLOBS = (os.path.join(d, "nvidia[0-9]*"),)
        try:
            assert w._visible_device_nodes() == ["nvidia0", "nvidia11", "nvidia3"]
        finally:
            w._DEVICE_GLOBS = saved


def test_device_nodes_empty_when_nothing_is_allocated():
    with tempfile.TemporaryDirectory() as d:
        saved = w._DEVICE_GLOBS
        w._DEVICE_GLOBS = (os.path.join(d, "nvidia[0-9]*"),)
        try:
            assert w._visible_device_nodes() == []
        finally:
            w._DEVICE_GLOBS = saved


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
