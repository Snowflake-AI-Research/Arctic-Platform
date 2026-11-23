"""``worker._get_descendant_pids`` leaves zombies out, so ``cuda_checkpoint``
never locks one.

An Inkling-2-Small TP=8 cold start failed its dump on exactly that. Two
short-lived helpers the vLLM child forked before the TP workers were dead but
unreaped when ``_worker_checkpoint`` walked the tree, and the child only reaps
them later, in prepare_criu_dump:

    cuda_checkpoint FAILED: RuntimeError: Lock(100404) failed with CUresult=3
    prepare_criu_dump ... reaped_strays=[100402, 100404]

On the previous pod the same pair had been orphaned to the worker instead,
outside the child's tree, and the dump passed: a race any cold start can lose.

No ``/proc``, no psutil and no GPU: the tree comes from a stub psutil, the
states from ``stat`` files written to a temporary directory, and the driver is
a stub that answers CUresult=3 for a dead task the way the real one did.

Run from the package directory:
    cd arctic_platform/inference/semi_persistence
    python -m pytest tests/test_descendant_zombies.py -v
"""
from __future__ import annotations

import functools
import os
import sys
import types

import pytest

from test_restore_quiescence import _load_worker

w = _load_worker()

_CHILD = 100016
_ZOMBIES = (100402, 100404)
_TRACKER = 100410
_RANKS = tuple(range(100700, 100708))
_CUDA_ERROR_NOT_INITIALIZED = 3


def _stat(proc, pid, state, comm="python3", ppid=_CHILD):
    """One ``/proc/<pid>/stat`` line, fields 1-7, as the kernel formats it."""
    os.makedirs(os.path.join(proc, str(pid)), exist_ok=True)
    with open(os.path.join(proc, str(pid), "stat"), "w") as handle:
        handle.write(f"{pid} ({comm}) {state} {ppid} {_CHILD} {_CHILD} 0 -1\n")


def _fake_tree(proc):
    """The failing pod's tree: two zombie helpers, the resource tracker, and
    eight live ranks, in the order psutil walks them."""
    order = list(_ZOMBIES) + [_TRACKER] + list(_RANKS)
    for pid in _ZOMBIES:
        _stat(proc, pid, "Z")
    _stat(proc, _TRACKER, "S")
    for pid in _RANKS:
        _stat(proc, pid, "S", comm="VLLM::Worker_TP0")
    return order


def _fake_psutil(order, cmdlines):
    psutil = types.ModuleType("psutil")

    class Error(Exception):
        pass

    class NoSuchProcess(Error):
        pass

    class ZombieProcess(NoSuchProcess):
        pass

    class AccessDenied(Error):
        pass

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def children(self, recursive=False):
            assert self.pid == _CHILD and recursive
            return [Process(p) for p in order]

        def cmdline(self):
            return cmdlines.get(self.pid, ["python3"])

    for cls in (Error, NoSuchProcess, ZombieProcess, AccessDenied, Process):
        setattr(psutil, cls.__name__, cls)
    return psutil


@pytest.fixture
def tree(tmp_path, monkeypatch):
    proc = str(tmp_path)
    order = _fake_tree(proc)
    cmdlines = {_TRACKER: ["python3", "-c",
                           "from multiprocessing.resource_tracker import main"]}
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(order, cmdlines))
    monkeypatch.setattr(w, "_proc_is_zombie",
                        functools.partial(w._proc_is_zombie, proc=proc))
    return proc


# --------------------------------------------------------------------------
# _proc_is_zombie
# --------------------------------------------------------------------------


def test_state_z_is_a_zombie(tmp_path):
    _stat(str(tmp_path), 100404, "Z")
    assert w._proc_is_zombie(100404, proc=str(tmp_path)) is True


@pytest.mark.parametrize("state", ["S", "R", "D", "T", "t", "I"])
def test_every_other_state_is_not(tmp_path, state):
    _stat(str(tmp_path), 100700, state)
    assert w._proc_is_zombie(100700, proc=str(tmp_path)) is False


def test_the_state_is_read_after_the_last_paren(tmp_path):
    """A comm may spell ``) Z (`` itself; only the field after it counts."""
    _stat(str(tmp_path), 100700, "S", comm="evil) Z (x")
    assert w._proc_is_zombie(100700, proc=str(tmp_path)) is False
    _stat(str(tmp_path), 100701, "Z", comm="a (b) c")
    assert w._proc_is_zombie(100701, proc=str(tmp_path)) is True


def test_a_pid_that_is_gone_or_garbled_is_not_a_zombie(tmp_path):
    assert w._proc_is_zombie(4242, proc=str(tmp_path)) is False
    os.makedirs(tmp_path / "4243")
    (tmp_path / "4243" / "stat").write_text("4243 (truncated")
    assert w._proc_is_zombie(4243, proc=str(tmp_path)) is False


# --------------------------------------------------------------------------
# _get_descendant_pids
# --------------------------------------------------------------------------


def test_descendants_leave_out_zombies_and_the_tracker(tree):
    assert w._get_descendant_pids(_CHILD) == list(reversed(_RANKS))


def test_descendants_keep_a_live_helper(tree):
    """Only the dead are dropped: a live helper is still signalled and
    scanned, and the lock decides what to make of it."""
    _stat(tree, _ZOMBIES[0], "S")
    pids = w._get_descendant_pids(_CHILD)
    assert _ZOMBIES[0] in pids and _ZOMBIES[1] not in pids
    assert pids == list(reversed(_RANKS)) + [_ZOMBIES[0]]


# --------------------------------------------------------------------------
# _worker_checkpoint
# --------------------------------------------------------------------------


class _FakeDriver:
    """cuCheckpointProcess* that, like the real driver, cannot lock a task
    that has exited: CUDA_ERROR_NOT_INITIALIZED."""

    def __init__(self, dead):
        self.dead = set(dead)
        self.locked = []
        self.checkpointed = []

    def cuCheckpointProcessLock(self, pid, _args):
        if pid in self.dead:
            return _CUDA_ERROR_NOT_INITIALIZED
        self.locked.append(pid)
        return 0

    def cuCheckpointProcessCheckpoint(self, pid, _args):
        self.checkpointed.append(pid)
        return 0

    def cuCheckpointProcessUnlock(self, pid, _args):
        return 0


def test_checkpoint_locks_every_live_task_and_no_zombie(tree, monkeypatch):
    driver = _FakeDriver(dead=_ZOMBIES)
    monkeypatch.setattr(w, "_get_cu", lambda: driver)
    pids = w._worker_checkpoint(_CHILD, unlock=False)
    assert pids == list(reversed(_RANKS)) + [_CHILD]
    assert driver.locked == pids
    assert driver.checkpointed == pids


def test_without_the_skip_the_fake_fails_like_the_pod(tree, monkeypatch):
    """Guards the guard: were zombies still walked, this is the pod's error."""
    driver = _FakeDriver(dead=_ZOMBIES)
    monkeypatch.setattr(w, "_get_cu", lambda: driver)
    monkeypatch.setattr(w, "_proc_is_zombie", lambda pid: False)
    with pytest.raises(RuntimeError,
                       match=r"Lock\(100404\) failed with CUresult=3"):
        w._worker_checkpoint(_CHILD, unlock=False)
