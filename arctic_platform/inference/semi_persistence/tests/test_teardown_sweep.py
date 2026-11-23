"""Teardown's orphan sweep outlasts a TP=8 tree's exit, and still stops early.

A second restore of an Inkling-2-Small TP=8 image in the same pod failed:

    criu restore failed ... Can't fork for 105600: File exists

The first restore's teardown SIGKILLed the tree and swept for 5 s:

    orphan sweep gave up after 5s with a live descendant left
    reaped 3 orphaned descendant(s): 110453, 105600, 106280

The other seven ranks were still tearing down their CUDA contexts. They
outlived the worker, went to a PID 1 that never reaps, and as zombies with sid
105600 held the dumped leader's id against every later restore.

No processes and no waiting: ``waitpid`` and the clock are stubs, and the
leftover listing reads ``stat`` files written to a temporary directory.

Run from the package directory:
    cd arctic_platform/inference/semi_persistence
    python -m pytest tests/test_teardown_sweep.py -v
"""
from __future__ import annotations

import ast
import os
import types

import pytest

from test_restore_quiescence import _load_worker

w = _load_worker()

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_WORKER = os.path.join(_PKG, "worker.py")
_INSTANCE = os.path.join(_PKG, "instance.py")

_LEADER = 105600
_RANKS = (106269, 106276, 106277, 106278, 106279, 106281, 106282)


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = 0

    def monotonic(self):
        return self.now

    def sleep(self, dt):
        self.sleeps += 1
        self.now += dt


class _Tree:
    """``waitpid(-1, WNOHANG)`` over children that exit at scripted times.

    ``exits`` maps pid -> the clock reading it exits at; a pid in ``wedged``
    never exits.
    """

    def __init__(self, clock, exits=None, wedged=()):
        self.clock = clock
        self.pending = dict(exits or {})
        self.wedged = set(wedged)

    def waitpid(self, pid, options):
        assert pid == -1 and options == os.WNOHANG
        for p, t in sorted(self.pending.items(), key=lambda kv: kv[1]):
            if t <= self.clock.now:
                del self.pending[p]
                return p, 9
        if self.pending or self.wedged:
            return 0, 0
        raise ChildProcessError(10, "No child processes")


class _Log:
    def __init__(self):
        self.info_lines = []
        self.warnings = []

    def info(self, msg, *args):
        self.info_lines.append(msg % args)

    def warning(self, msg, *args):
        self.warnings.append(msg % args)


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(w, "time", types.SimpleNamespace(
        monotonic=c.monotonic, sleep=c.sleep))
    return c


def _children(monkeypatch, clock, **tree):
    t = _Tree(clock, **tree)
    real_os = os

    class _Os:
        def __getattr__(self, name):
            return getattr(real_os, name)

        waitpid = staticmethod(t.waitpid)

    monkeypatch.setattr(w, "os", _Os())
    return t


# --------------------------------------------------------------------------
# _reap_orphaned_descendants
# --------------------------------------------------------------------------


def test_nothing_left_returns_at_once(clock, monkeypatch):
    """The fast path: a tree already gone costs one waitpid, no sleep."""
    _children(monkeypatch, clock)
    log = _Log()
    assert w._reap_orphaned_descendants(
        log, timeout_s=w._TEARDOWN_SWEEP_TIMEOUT_S) == []
    assert clock.now == 0.0 and clock.sleeps == 0
    assert log.info_lines == [] and log.warnings == []


def test_ranks_slower_than_the_old_bound_are_all_reaped(clock, monkeypatch):
    """The pod's tree: the leader at once, the ranks 8-20 s later."""
    exits = {_LEADER: 0.0}
    exits.update({p: 8.0 + 2 * i for i, p in enumerate(_RANKS)})
    _children(monkeypatch, clock, exits=exits)
    log = _Log()
    reaped = w._reap_orphaned_descendants(
        log, timeout_s=w._TEARDOWN_SWEEP_TIMEOUT_S)
    assert sorted(reaped) == sorted(exits)
    assert log.warnings == []
    assert log.info_lines == [
        "  reaped 8 orphaned descendant(s) in 20.0s: "
        + ", ".join(str(p) for p in [_LEADER, *_RANKS])]


def test_the_sweep_stops_as_soon_as_the_last_one_is_reaped(clock, monkeypatch):
    _children(monkeypatch, clock, exits={p: 12.0 for p in _RANKS})
    w._reap_orphaned_descendants(_Log(), timeout_s=w._TEARDOWN_SWEEP_TIMEOUT_S)
    assert 12.0 <= clock.now < 12.1


def test_the_old_five_second_bound_left_them_behind(clock, monkeypatch):
    """What the pod logged, with the leftovers now named."""
    exits = {_LEADER: 0.0}
    exits.update({p: 8.0 for p in _RANKS})
    _children(monkeypatch, clock, exits=exits)
    monkeypatch.setattr(w, "_proc_children",
                        lambda ppid: [(p, "D") for p in _RANKS])
    log = _Log()
    assert w._reap_orphaned_descendants(log, timeout_s=5.0) == [_LEADER]
    assert len(log.warnings) == 1
    assert log.warnings[0].startswith(
        "  orphan sweep gave up after 5.0s (reaped 1) with live "
        "descendant(s) left: 106269:D, 106276:D,")


def test_a_wedged_task_still_ends_the_sweep_at_the_bound(clock, monkeypatch):
    _children(monkeypatch, clock, exits={_LEADER: 0.0}, wedged={_RANKS[0]})
    monkeypatch.setattr(w, "_proc_children", lambda ppid: [(_RANKS[0], "D")])
    log = _Log()
    reaped = w._reap_orphaned_descendants(
        log, timeout_s=w._TEARDOWN_SWEEP_TIMEOUT_S)
    assert reaped == [_LEADER]
    assert w._TEARDOWN_SWEEP_TIMEOUT_S <= clock.now < (
        w._TEARDOWN_SWEEP_TIMEOUT_S + 0.1)
    assert log.warnings == [
        f"  orphan sweep gave up after {w._TEARDOWN_SWEEP_TIMEOUT_S:.1f}s "
        "(reaped 1) with live descendant(s) left: 106269:D; they may still "
        "reference recorded ids"]


def test_a_sweep_that_cannot_list_the_leftovers_still_warns(clock, monkeypatch):
    _children(monkeypatch, clock, wedged={_RANKS[0]})
    monkeypatch.setattr(w, "_proc_children", lambda ppid: [])
    log = _Log()
    assert w._reap_orphaned_descendants(log, timeout_s=1.0) == []
    assert "live descendant(s) left: ?;" in log.warnings[0]


# --------------------------------------------------------------------------
# _proc_children
# --------------------------------------------------------------------------


def _stat(proc, pid, state, ppid, comm="python3"):
    os.makedirs(os.path.join(proc, str(pid)), exist_ok=True)
    with open(os.path.join(proc, str(pid), "stat"), "w") as handle:
        handle.write(f"{pid} ({comm}) {state} {ppid} {_LEADER} {_LEADER} 0 -1\n")


def test_children_are_read_from_stat(tmp_path):
    proc = str(tmp_path)
    worker = 4000
    _stat(proc, 106282, "D", worker, comm="VLLM::Worker_TP7")
    _stat(proc, 106269, "Z", worker, comm="a) S 1 (b")
    _stat(proc, 106270, "S", 106269)
    _stat(proc, 1, "S", 0, comm="sleep")
    os.makedirs(os.path.join(proc, "self"))
    os.makedirs(os.path.join(proc, "106290"))     # exited while we walked it
    assert w._proc_children(worker, proc=proc) == [(106269, "Z"), (106282, "D")]


def test_children_of_an_unreadable_proc_are_none(tmp_path):
    assert w._proc_children(4000, proc=str(tmp_path / "missing")) == []


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------


def _function(path, name, cls=None):
    tree = ast.parse(open(path).read())
    body = tree.body
    if cls is not None:
        body = next(n for n in body
                    if isinstance(n, ast.ClassDef) and n.name == cls).body
    return next(n for n in body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def _sweep_bounds(fn):
    """The ``timeout_s`` of every ``_reap_orphaned_descendants`` call in *fn*."""
    bounds = []
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "_reap_orphaned_descendants"):
            kw = {k.arg: k.value for k in node.keywords}
            bounds.append(ast.unparse(kw["timeout_s"]) if "timeout_s" in kw
                          else None)
    return bounds


def test_teardown_and_exit_sweep_with_the_teardown_bound():
    assert w._TEARDOWN_SWEEP_TIMEOUT_S >= 60.0
    assert _sweep_bounds(_function(_WORKER, "worker_loop")) == [
        "_TEARDOWN_SWEEP_TIMEOUT_S", "_TEARDOWN_SWEEP_TIMEOUT_S"]


def test_the_instance_does_not_kill_its_worker_mid_sweep():
    """The teardown result reaches the Instance before the worker sweeps, and
    _reset used to SIGKILL the worker after 10 s -- inside a 60 s sweep."""
    fn = _function(_INSTANCE, "_reset", cls="Instance")
    joins = [ast.unparse(k.value) for node in ast.walk(fn)
             if isinstance(node, ast.Call)
             and getattr(node.func, "attr", None) == "join"
             for k in node.keywords if k.arg == "timeout"]
    assert joins[0] == "_TEARDOWN_SWEEP_TIMEOUT_S + 10"
    assert "from worker import _TEARDOWN_SWEEP_TIMEOUT_S" in open(_INSTANCE).read()
