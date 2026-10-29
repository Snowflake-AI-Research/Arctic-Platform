"""``worker._dump_slot``: at most SEMIP_CONCURRENT_DUMPS criu dumps per pod.

Run from the package directory:
    cd arctic_inference/semi_persistence
    python -m pytest tests/test_dump_slots.py -v
"""
from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import threading
import time

from test_restore_quiescence import _load_worker

w = _load_worker()


def _hold(path):
    """An flock on ``path`` from another open file, as another worker holds it."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def test_a_third_dump_waits_for_one_of_two_slots(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(w, "_DUMP_SLOTS_DIR", str(tmp_path))
    monkeypatch.setenv("SEMIP_CONCURRENT_DUMPS", "2")
    held = [_hold(tmp_path / "slot0"), _hold(tmp_path / "slot1")]
    threading.Timer(1.2, os.close, args=(held[1],)).start()
    t0 = time.monotonic()
    with w._dump_slot():
        waited = time.monotonic() - t0
        # The freed slot is now ours, so nobody else gets it.
        fd = os.open(tmp_path / "slot1", os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise AssertionError("slot1 should be held")
        except BlockingIOError:
            pass
        finally:
            os.close(fd)
    os.close(held[0])
    assert 1.0 <= waited < 3.0
    assert "waited" in capsys.readouterr().out
    _hold(tmp_path / "slot1")  # released on exit


def test_a_dead_holder_frees_its_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "_DUMP_SLOTS_DIR", str(tmp_path))
    monkeypatch.setenv("SEMIP_CONCURRENT_DUMPS", "1")
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import fcntl, os, sys, time\n"
         f"fd = os.open({str(tmp_path / 'slot0')!r}, os.O_RDWR | os.O_CREAT)\n"
         "fcntl.flock(fd, fcntl.LOCK_EX)\n"
         "print('held', flush=True)\n"
         "time.sleep(60)\n"],
        stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "held"
    holder.kill()
    holder.wait()
    t0 = time.monotonic()
    with w._dump_slot():
        assert time.monotonic() - t0 < 1.0


def test_zero_means_no_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "_DUMP_SLOTS_DIR", str(tmp_path / "slots"))
    monkeypatch.setenv("SEMIP_CONCURRENT_DUMPS", "0")
    with w._dump_slot():
        pass
    assert not (tmp_path / "slots").exists()


def test_no_slot_directory_dumps_without_a_limit(tmp_path, monkeypatch, capsys):
    (tmp_path / "file").write_text("")
    monkeypatch.setattr(w, "_DUMP_SLOTS_DIR", str(tmp_path / "file" / "slots"))
    monkeypatch.delenv("SEMIP_CONCURRENT_DUMPS", raising=False)
    with w._dump_slot():
        pass
    assert "dumping without a limit" in capsys.readouterr().out
