"""``worker._shm_ghost_limit``: criu's ``--ghost-limit`` covers the tree's own
deleted ``/dev/shm`` mappings.

The dump unlinks the tree's ``sem.*`` and ``psm_*`` and keeps the mappings, so
criu carries each as a ghost file, refused above the limit (1 MiB unless told
otherwise). A DeepSeek-V4.1-Flash TP=8 cold start with router replay failed
exactly there, on the engine child's view of a rank's response ring:

    Handling VMA ... 7f09c8fff000-7f09d8000000 rw-s 00000000 00:67 31  /dev/shm/psm_1f1d89c7 (deleted)
    Error (criu/files-reg.c:1031): Can't dump ghost file /dev/shm/psm_1f1d89c7 of 1228800 size, increase limit

No ``/proc`` and no Linux: the maps are written to a temporary directory.

Run from the package directory:
    cd arctic_platform/inference/semi_persistence
    python -m pytest tests/test_ghost_limit.py -v
"""
from __future__ import annotations

import os
import subprocess

import pytest

from test_restore_quiescence import _load_worker

w = _load_worker()

_MIB = 1 << 20
# vLLM 0.32 MessageQueue(1, 1): 10 x (24 MiB + 2 B of flags), page-rounded.
_RESPONSE_RING = "7f09c8fff000-7f09d8000000"
_RESPONSE_RING_BYTES = 0x7f09d8000000 - 0x7f09c8fff000


def _maps(proc, pid, *lines):
    os.makedirs(os.path.join(proc, str(pid)), exist_ok=True)
    with open(os.path.join(proc, str(pid), "maps"), "w") as handle:
        for line in lines:
            handle.write(line + "\n")
        handle.write("7fff0000-7fff1000 r-xp 00000000 00:00 0 \n")


def test_the_engine_childs_response_ring_fits(tmp_path):
    proc = str(tmp_path)
    _maps(proc, 100016,
          "7f09c4000000-7f09c4021000 rw-p 00000000 00:00 0 ",
          f"{_RESPONSE_RING} rw-s 00000000 00:67 31"
          "                         /dev/shm/psm_1f1d89c7 (deleted)",
          "7f09d8000000-7f09d8001000 rw-s 00000000 00:67 7"
          "                          /dev/shm/sem.wZMmgP (deleted)")
    limit = w._shm_ghost_limit([100016], proc=proc)
    assert _RESPONSE_RING_BYTES == 240 * _MIB + 4096
    assert limit == 242 * _MIB
    assert limit >= 1228800


def test_the_largest_extent_across_the_tree_wins(tmp_path):
    """Each rank maps its own ring and the broadcast ring; the child maps all."""
    proc = str(tmp_path)
    _maps(proc, 100200,
          "7f0000000000-7f000a001000 rw-s 00000000 00:67 40"
          "  /dev/shm/psm_broadcast (deleted)")
    _maps(proc, 100016,
          f"{_RESPONSE_RING} rw-s 00000000 00:67 31  /dev/shm/psm_r0 (deleted)")
    assert w._shm_ghost_limit([100200, 100016], proc=proc) == 242 * _MIB


def test_the_file_offset_counts_toward_the_extent(tmp_path):
    proc = str(tmp_path)
    _maps(proc, 100,
          "7f0000000000-7f0000100000 rw-s 00300000 00:67 9"
          "  /dev/shm/psm_tail (deleted)")
    assert w._shm_ghost_limit([100], proc=proc) == 4 * _MIB


def test_only_deleted_dev_shm_mappings_raise_it(tmp_path):
    """A linked segment is reopened by path, not ghosted, and a deleted file
    outside /dev/shm is not one the dump unlinked: neither moves the limit."""
    proc = str(tmp_path)
    _maps(proc, 100,
          "7f0000000000-7f0040000000 rw-s 00000000 00:67 9  /dev/shm/psm_live",
          "7f1000000000-7f1040000000 r--p 00000000 08:01 5"
          "  /tmp/torchinductor_root/big.so (deleted)")
    assert w._shm_ghost_limit([100], proc=proc) == w._CRIU_DEFAULT_GHOST_LIMIT


def test_a_one_page_ghost_rounds_up_to_one_huge_page(tmp_path):
    proc = str(tmp_path)
    _maps(proc, 100,
          "7f0000000000-7f0000001000 rw-s 00000000 00:67 9"
          "  /dev/shm/sem.abc (deleted)")
    assert w._shm_ghost_limit([100, 4242], proc=proc) == 2 * _MIB


def test_no_ghosts_or_no_pids_keep_the_default(tmp_path):
    assert w._shm_ghost_limit([4242], proc=str(tmp_path)) == _MIB
    assert w._shm_ghost_limit([], proc=str(tmp_path)) == _MIB


def test_the_dump_passes_the_limit_for_the_whole_tree(tmp_path, monkeypatch):
    seen = {}

    def fake_limit(pids):
        seen["pids"] = list(pids)
        return 242 * _MIB

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 1, "", "stop here")

    monkeypatch.setattr(w, "_get_descendant_pids", lambda pid: [100201, 100200])
    monkeypatch.setattr(w, "_resolve_fd_resource", lambda pid, fd: "pipe:[1]")
    monkeypatch.setattr(w, "_shm_ghost_limit", fake_limit)
    monkeypatch.setattr(w.subprocess, "run", fake_run)
    monkeypatch.setenv("SEMIP_CONCURRENT_DUMPS", "0")
    monkeypatch.delenv("SEMIP_UNPRIVILEGED", raising=False)
    with pytest.raises(RuntimeError, match="criu dump failed"):
        w._worker_criu_save(100016, str(tmp_path / "img"), -1, None, [0])
    cmd = seen["cmd"]
    assert cmd[1] == "dump"
    assert sorted(seen["pids"]) == [100016, 100200, 100201]
    assert cmd[cmd.index("--ghost-limit") + 1] == str(242 * _MIB)
