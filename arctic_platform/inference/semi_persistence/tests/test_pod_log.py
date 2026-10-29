"""Where the tree's stdout goes, and which /dev/shm files a dump may unlink.

``semip_logging`` is stdlib-only and imports directly. ``vllm_child`` imports
torch at module scope, so its two pure helpers are lifted out of the source
instead -- the same treatment ``test_layout_names`` gives the layout constants.
"""
from __future__ import annotations

import ast
import logging
import os
import select
import stat
import sys
import tempfile
import threading

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
sys.path.insert(0, _PKG)

import semip_logging  # noqa: E402


def _with_target(path, body):
    saved = os.environ.get("SEMIP_LOG_TARGET")
    os.environ["SEMIP_LOG_TARGET"] = path
    semip_logging._warned_target = False
    try:
        return body()
    finally:
        if saved is None:
            os.environ.pop("SEMIP_LOG_TARGET", None)
        else:
            os.environ["SEMIP_LOG_TARGET"] = saved


def _is_devnull(fd):
    return os.fstat(fd).st_rdev == os.stat(os.devnull).st_rdev and \
        stat.S_ISCHR(os.fstat(fd).st_mode)


# --------------------------------------------------------------------------
# open_pod_log
# --------------------------------------------------------------------------

def test_a_pipe_with_a_reader_is_the_pod_log():
    with tempfile.TemporaryDirectory() as tmp:
        fifo = os.path.join(tmp, "stdout")
        os.mkfifo(fifo)
        reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        try:
            fd = _with_target(fifo, semip_logging.open_pod_log)
            try:
                assert stat.S_ISFIFO(os.fstat(fd).st_mode)
                # Blocking again: a non-blocking stdout drops lines on EAGAIN.
                import fcntl
                assert not fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_NONBLOCK
                os.write(fd, b"line\n")
                assert os.read(reader, 16) == b"line\n"
            finally:
                os.close(fd)
        finally:
            os.close(reader)


def test_a_pipe_with_no_reader_falls_back_instead_of_hanging():
    with tempfile.TemporaryDirectory() as tmp:
        fifo = os.path.join(tmp, "stdout")
        os.mkfifo(fifo)
        fd = _with_target(fifo, semip_logging.open_pod_log)
        try:
            assert _is_devnull(fd)
        finally:
            os.close(fd)


def test_a_regular_file_is_not_accepted():
    """CRIU would reopen it by path and re-validate it; only a pipe inherits."""
    with tempfile.NamedTemporaryFile() as handle:
        fd = _with_target(handle.name, semip_logging.open_pod_log)
        try:
            assert _is_devnull(fd)
        finally:
            os.close(fd)


def test_a_missing_target_falls_back():
    fd = _with_target("/nonexistent/semip/stdout", semip_logging.open_pod_log)
    try:
        assert _is_devnull(fd)
    finally:
        os.close(fd)


def test_the_default_target_is_pid_1_stdout():
    saved = os.environ.pop("SEMIP_LOG_TARGET", None)
    try:
        assert semip_logging.pod_log_target() == "/proc/1/fd/1"
    finally:
        if saved is not None:
            os.environ["SEMIP_LOG_TARGET"] = saved


# --------------------------------------------------------------------------
# The copy to Ray (tee_tree_to_ray)
# --------------------------------------------------------------------------

def _copy(chunks, dests, monkeypatch):
    """Run _copy_lines over *chunks* to EOF; return its writes per destination."""
    writes = []
    monkeypatch.setattr(semip_logging, "_write_all",
                        lambda fd, data: writes.append((fd, bytes(data))))
    r, w = os.pipe()

    def feed():  # in a thread: more than a pipe's capacity would block here
        for chunk in chunks:
            os.write(w, chunk)
        os.close(w)

    feeder = threading.Thread(target=feed)
    feeder.start()
    semip_logging._copy_lines(r, dests)
    feeder.join()
    return writes


def test_a_line_written_in_pieces_is_copied_whole(monkeypatch):
    """One write per whole line is what keeps replicas apart in the pod log."""
    writes = _copy([b"ab", b"c\nde", b"f\n"], [7, 8], monkeypatch)
    assert writes == [(7, b"abc\n"), (8, b"abc\n"), (7, b"def\n"), (8, b"def\n")]


def test_a_trailing_partial_line_is_delivered_at_eof(monkeypatch):
    writes = _copy([b"a\nno newline"], [7], monkeypatch)
    assert writes == [(7, b"a\n"), (7, b"no newline")]


def test_a_line_that_never_ends_is_passed_on_in_pieces(monkeypatch):
    writes = _copy([b"x" * 70000 + b"\n"], [7], monkeypatch)
    assert [len(data) for _, data in writes] == [65536, 70001 - 65536]


def test_a_dead_destination_is_dropped_and_the_other_keeps_every_line():
    dead_r, dead_w = os.pipe()
    os.close(dead_r)                     # writes now fail with EPIPE
    live_r, live_w = os.pipe()
    r, w = os.pipe()
    os.write(w, b"one\ntwo\n")
    os.close(w)
    try:
        semip_logging._copy_lines(r, [dead_w, live_w])
        os.close(live_w)
        out = b""
        while chunk := os.read(live_r, 4096):
            out += chunk
    finally:
        os.close(dead_w)
        os.close(live_r)
    assert out.endswith(b"one\ntwo\n")
    assert b"dropped it" in out


def test_the_off_switch_leaves_the_target_alone(monkeypatch):
    monkeypatch.setenv("SEMIP_LOG_TO_RAY", "0")
    monkeypatch.setenv("SEMIP_LOG_TARGET", "/proc/1/fd/1")
    semip_logging.tee_tree_to_ray()
    assert os.environ["SEMIP_LOG_TARGET"] == "/proc/1/fd/1"
    assert semip_logging._tee_write_fd is None


def _read_within(fd, timeout=5.0):
    ready, _, _ = select.select([fd], [], [], timeout)
    return os.read(fd, 65536) if ready else b""


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="opens the pipe through /proc")
def test_the_tree_and_semip_loggers_reach_both_and_the_engine_logger_once(
        monkeypatch):
    monkeypatch.delenv("SEMIP_LOG_TO_RAY", raising=False)
    names = ("semip.inst.tee_test", semip_logging._ENGINE_LOGGER + ".tee_test")
    with tempfile.TemporaryDirectory() as tmp:
        fifo = os.path.join(tmp, "stdout")
        os.mkfifo(fifo)
        pod_r = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        ray_r, ray_w = os.pipe()
        monkeypatch.setenv("SEMIP_LOG_TARGET", fifo)
        semip_logging._warned_target = False
        try:
            semip_logging.tee_tree_to_ray(ray_fd=ray_w)
            target = os.environ["SEMIP_LOG_TARGET"]
            assert target.startswith(f"/proc/{os.getpid()}/fd/")

            # What the worker and the child do at startup.
            fd = semip_logging.open_pod_log()
            assert stat.S_ISFIFO(os.fstat(fd).st_mode)
            os.write(fd, b"from the tree\n")
            os.close(fd)
            assert _read_within(ray_r) == b"from the tree\n"
            assert _read_within(pod_r) == b"from the tree\n"

            semip_logging.attach_pod_log(list(names))
            logging.getLogger(names[0]).info("semip record")
            assert b"semip record" in _read_within(ray_r)
            assert b"semip record" in _read_within(pod_r)

            logging.getLogger(names[1]).info("engine record")
            assert b"engine record" in _read_within(pod_r)
            assert _read_within(ray_r, timeout=0.5) == b""
        finally:
            for name in names:
                logger = logging.getLogger(name)
                logger.handlers[:] = [h for h in logger.handlers
                                      if not getattr(h, "_semip_pod_log", False)]
            semip_logging._pod_log_stream = None
            semip_logging._direct_stream = None
            semip_logging._tee_write_fd = None
            semip_logging._tee_pod_log_fd = None
            os.close(pod_r)
            os.close(ray_r)
            os.close(ray_w)


# --------------------------------------------------------------------------
# The line tag
# --------------------------------------------------------------------------

def _format(name, replica):
    saved = os.environ.get("SEMIP_REPLICA_ID")
    os.environ["SEMIP_REPLICA_ID"] = replica
    try:
        record = logging.LogRecord(name, logging.INFO, "f.py", 7, "hello",
                                   None, None)
        semip_logging._FILTER.filter(record)
        return semip_logging._FORMATTER.format(record)
    finally:
        if saved is None:
            os.environ.pop("SEMIP_REPLICA_ID", None)
        else:
            os.environ["SEMIP_REPLICA_ID"] = saved


def test_every_line_names_its_replica_and_role():
    """Several replicas share one pod log; the tag is what separates them."""
    assert "[r3 child]" in _format("semip.inst.0.child", "3")
    assert "[r3 worker]" in _format("semip.inst.0.worker", "3")
    assert "[r0 engine]" in _format("arctic_platform.inference.server.semip", "0")


# --------------------------------------------------------------------------
# vllm_child's /dev/shm scoping
# --------------------------------------------------------------------------

def _lift(*names):
    path = os.path.join(_PKG, "vllm_child.py")
    with open(path) as handle:
        tree = ast.parse(handle.read(), filename=path)
    funcs = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {f.name for f in funcs} == set(names), names
    namespace = {"os": os}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), path, "exec"),
         namespace)
    return [namespace[n] for n in names]


_own_shm_paths, _tree_pids = _lift("_own_shm_paths", "_tree_pids")


def _fake_proc(tmp, pid, *, maps=(), fds=(), children=()):
    base = os.path.join(tmp, str(pid))
    os.makedirs(os.path.join(base, "fd"), exist_ok=True)
    os.makedirs(os.path.join(base, "task", str(pid)), exist_ok=True)
    with open(os.path.join(base, "maps"), "w") as handle:
        for i, path in enumerate(maps):
            handle.write(f"7f{i:04x}000-7f{i:04x}fff rw-s 00000000 00:1a 99 "
                         f"           {path}\n")
        handle.write("7fff0000-7fff1000 r-xp 00000000 00:00 0 \n")
    for i, target in enumerate(fds):
        os.symlink(target, os.path.join(base, "fd", str(10 + i)))
    with open(os.path.join(base, "task", str(pid), "children"), "w") as handle:
        handle.write(" ".join(str(c) for c in children))


def test_only_this_trees_segments_are_named():
    """A sibling replica's psm_ segment must survive this replica's dump."""
    with tempfile.TemporaryDirectory() as proc, \
            tempfile.TemporaryDirectory() as shm:
        _fake_proc(proc, 100, maps=["/dev/shm/psm_mine", "/usr/lib/x.so"],
                   fds=["/dev/shm/psm_open", "socket:[5]"], children=[101])
        _fake_proc(proc, 101, maps=["/dev/shm/sem.mp-child"])
        _fake_proc(proc, 200, maps=["/dev/shm/psm_sibling"])
        assert _tree_pids(100, proc=proc) == [100, 101]
        assert _own_shm_paths([100], ("/dev/shm/psm_",), proc=proc,
                              shm_dir=shm) == [
            "/dev/shm/psm_mine", "/dev/shm/psm_open"]
        assert _own_shm_paths(_tree_pids(100, proc=proc),
                              ("/dev/shm/sem.",), proc=proc, shm_dir=shm) == [
            "/dev/shm/sem.mp-child"]


def test_already_unlinked_segments_are_skipped():
    with tempfile.TemporaryDirectory() as proc, \
            tempfile.TemporaryDirectory() as shm:
        _fake_proc(proc, 100, maps=["/dev/shm/psm_gone (deleted)"])
        assert _own_shm_paths([100], ("/dev/shm/psm_",), proc=proc,
                              shm_dir=shm) == []


def test_a_semaphore_mapped_under_its_temporary_name_is_found_by_inode():
    """glibc's sem_open maps sem.XXXXXX, links it to the real name, unlinks it.

    ``maps`` then shows only the deleted temporary, and the real name has to be
    found by inode -- left linked, CRIU link-remaps it into the dumping pod's
    /dev/shm and the image is restorable nowhere else. A sibling replica's
    semaphore, a different inode, must not be touched.
    """
    with tempfile.TemporaryDirectory() as proc, \
            tempfile.TemporaryDirectory() as shm:
        mine = os.path.join(shm, "sem.loky-100128-abcd")
        sibling = os.path.join(shm, "sem.loky-200128-efgh")
        for path in (mine, sibling):
            with open(path, "w") as handle:
                handle.write("x" * 32)
        st = os.stat(mine)
        dev = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}"
        base = os.path.join(proc, "100")
        os.makedirs(os.path.join(base, "fd"))
        os.makedirs(os.path.join(base, "task", "100"))
        open(os.path.join(base, "task", "100", "children"), "w").close()
        with open(os.path.join(base, "maps"), "w") as handle:
            handle.write(f"7fc308aae000-7fc308aaf000 rw-s 00000000 {dev} "
                         f"{st.st_ino}        /dev/shm/sem.wZMmgP (deleted)\n")
        assert _own_shm_paths([100], ("/dev/shm/sem.",), proc=proc,
                              shm_dir=shm) == [mine]


def test_a_vanished_process_is_not_an_error():
    with tempfile.TemporaryDirectory() as proc:
        assert _own_shm_paths([4242], ("/dev/shm/psm_",), proc=proc,
                              shm_dir=proc) == []
        assert _tree_pids(4242, proc=proc) == [4242]
