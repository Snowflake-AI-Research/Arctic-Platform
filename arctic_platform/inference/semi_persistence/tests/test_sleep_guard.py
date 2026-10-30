"""The semi-p worker never checkpoints an engine whose sleep failed.

An engine that did not sleep still holds its weights and KV cache, so a CUDA
checkpoint of it captures all of that, and the image it leads to is tens of
times larger and is later restored as a hit. ``_child_thread`` runs the
commands; it is driven here with a scripted pipe in place of the child, and
the checkpoint itself recorded instead of run.

    cd arctic_platform/inference/semi_persistence
    python -m pytest tests/test_sleep_guard.py -v
"""
from __future__ import annotations

import contextlib
import importlib.util
import logging
import os
import queue
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_WORKER = os.path.join(_PKG, "worker.py")


def _load_worker():
    """Import worker.py with NVML, torch and the log plumbing stubbed, as
    ``test_restore_quiescence`` does, withdrawing the stubs afterwards."""
    torch = types.ModuleType("torch")
    torch.multiprocessing = types.ModuleType("torch.multiprocessing")
    stubs = {"pynvml": types.ModuleType("pynvml"), "torch": torch,
             "torch.multiprocessing": torch.multiprocessing,
             "semip_logging": types.ModuleType("semip_logging")}
    added = [name for name in stubs if name not in sys.modules]
    for name in added:
        sys.modules[name] = stubs[name]
    sys.path.insert(0, _PKG)
    try:
        spec = importlib.util.spec_from_file_location("worker_sleep_guard", _WORKER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(_PKG)
        for name in added:
            sys.modules.pop(name, None)
    return module


w = _load_worker()


class _Pipe:
    """The child's end: answers each synchronous command with the next scripted error."""

    def __init__(self, errors):
        self.errors = dict(errors)
        self.sent = []
        self._replies = []

    def send(self, message):
        cmd, _ = message
        self.sent.append(cmd)
        errors = self.errors.get(cmd) or [None]
        self._replies.append((cmd, 0.0, errors.pop(0) if len(errors) > 1 else errors[0], {}))

    def recv(self):
        return self._replies.pop(0)

    def poll(self, timeout=0):
        return False

    def close(self):
        pass


def _run(commands, errors, monkeypatch):
    checkpoints = []
    monkeypatch.setattr(w, "semip_logging", types.SimpleNamespace(
        worker=lambda *a: logging.getLogger("sleep_guard")), raising=False)
    monkeypatch.setattr(w, "_worker_checkpoint",
                        lambda pid, unlock=False: checkpoints.append(pid) or [pid])
    monkeypatch.setattr(w, "_kill_process_tree", lambda pid: None)
    commands_q, results_q = queue.Queue(), queue.Queue()
    for cmd in [*commands, "teardown"]:
        commands_q.put((cmd, {}))
    counter = types.SimpleNamespace(value=0, get_lock=contextlib.nullcontext)
    pipe = _Pipe(errors)
    w._child_thread("t", [0], 4242, pipe, commands_q, results_q, counter)
    results = []
    while not results_q.empty():
        cmd, _, error, _ = results_q.get()
        results.append((cmd, error))
    return results, checkpoints, pipe.sent


def test_a_failed_sleep_refuses_the_checkpoint_and_the_dump(monkeypatch):
    results, checkpoints, sent = _run(
        ["sleep", "cuda_checkpoint", "criu_dump"],
        {"sleep": ["AuxOutput connector refused the reset"]}, monkeypatch)
    assert checkpoints == []
    assert "prepare_criu_dump" not in sent and "get_pipe_fd" not in sent
    by_cmd = dict(results)
    assert "failed to sleep" in by_cmd["cuda_checkpoint"]
    assert "AuxOutput" in by_cmd["criu_dump"]


def test_a_later_sleep_that_succeeds_lifts_the_refusal(monkeypatch):
    results, checkpoints, _ = _run(
        ["sleep", "cuda_checkpoint", "sleep", "cuda_checkpoint"],
        {"sleep": ["boom", None]}, monkeypatch)
    assert checkpoints == [4242]
    assert [e for c, e in results if c == "cuda_checkpoint"][1] is None


def test_a_checkpoint_without_a_sleep_is_not_refused(monkeypatch):
    """A follower node's worker sees no sleep: the leader's executor sleeps its ranks."""
    results, checkpoints, _ = _run(["cuda_checkpoint"], {}, monkeypatch)
    assert checkpoints == [4242]
    assert dict(results)["cuda_checkpoint"] is None
