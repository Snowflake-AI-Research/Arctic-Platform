"""Unit tests for Complication 15: the dump-side listener close and its report.

Three halves by now, for the three ways this went wrong.

The **classifier** half checks that ``_loopback_listeners`` picks exactly the
orphaned-rendezvous shape and nothing else, and that it is pure -- a census that
quietly closed sockets would corrupt the very dump it exists to explain, so it is
asserted directly rather than reasoned about. This half needs ``/proc`` and so
only runs on Linux; it must also run *in-process*, because ``socket(fileno=)``
resolves fds in the calling process and an out-of-process dry run returns EBADF
for everything and looks like total failure.

The **call-site** half asserts where the close may and may not be reached.
``_close_loopback_listeners`` is a TP=1 tool: there the seven loopback listeners
are torch's and NCCL runs no RAS subsystem for a single rank. At TP>1 the set is
inverted -- torch retires its own before the dump and what survives is RAS's --
so ``_prepare_worker_dump`` must census and never close. That is checked against
the source, because the alternative needs a live TP>1 tree.

The **report** half checks ``_is_port_collision`` / ``_port_collision_report``
against the verbatim criu excerpt from the failure that started all of this
(job ``6964ec1c3fd7``). It is portable: no ``/proc``, no Linux, no sockets.

One invariant guards a trap rather than a behaviour: the wait list must not grow
to include ``pt_gloo_runloop``. Gloo's listener lives for the whole process, so
waiting on that thread would turn a fast poll into a guaranteed 2.5 s timeout
plus a warning on every dump. It belongs in the census list only.

Both modules import torch / pynvml at module scope, so the functions under test
are lifted out of the source by AST and exec'd. That keeps the test honest: it
runs the shipped text rather than a copy that can drift.

Run from the package directory::

    cd arctic_inference/semi_persistence
    python -m pytest tests/test_listener_close.py -v

Or directly::

    python tests/test_listener_close.py
"""
from __future__ import annotations

import ast
import os
import socket
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                      # .../semi_persistence
_WORKER = os.path.join(_PKG, "worker.py")
_CHILD = os.path.join(_PKG, "vllm_child.py")

_ON_LINUX = sys.platform.startswith("linux")


def _lift(path, names, ns=None, assigns=()):
    """Exec the named top-level defs (and assignments) out of *path*.

    Neither module is importable here -- ``worker.py`` pulls in pynvml and
    ``vllm_child.py`` pulls in torch -- so the functions are compiled straight
    out of the source tree. ``assigns`` names module-level constants the lifted
    functions close over.
    """
    tree = ast.parse(open(path).read())
    ns = {"os": os, "sys": sys} if ns is None else ns
    found = set()
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") in assigns):
            exec(compile(ast.Module([node], []), path, "exec"), ns)
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in names):
            exec(compile(ast.Module([node], []), path, "exec"), ns)
            found.add(node.name)
    missing = set(names) - found
    assert not missing, f"{path}: could not lift {missing}"
    return ns


# --------------------------------------------------------------------------
# The classifier
# --------------------------------------------------------------------------

def _child_ns():
    return _lift(_CHILD,
                 {"_loopback_listeners", "_close_loopback_listeners",
                  "_live_store_threads", "_ephemeral_port_range"},
                 assigns={"_WAIT_STORE_THREADS", "_CENSUS_STORE_THREADS"})


def _census_tags(ns):
    """The ``addr:port`` set the census selects as closeable (ephemeral)."""
    return {tag for _, tag in ns["_loopback_listeners"]()[0]}


def _fixed_loopback_port():
    """A loopback port below this host's ephemeral range, or None.

    Anything the kernel will not hand out by accident stands in for NCCL's RAS
    listener, which is the shape that has to survive.
    """
    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range") as f:
            lo = int(f.read().split()[0])
    except (OSError, ValueError):
        return None
    for port in range(lo - 1, lo - 400, -1):
        if port < 1024:
            return None
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", port))
            s.listen(1)
            return s
        except OSError:
            s.close()
    return None


class _Shapes:
    """One socket per shape the classifier has to tell apart.

    Kept as instance attributes so nothing is garbage-collected mid-test: a
    collected wrapper closes its fd, which would look like the classifier having
    closed it.
    """

    def __init__(self):
        self.should_close = {}
        self.should_spare = {}
        self._keep = []

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        self._keep.append(s)
        self.should_close[s.fileno()] = "loopback v4 listener"

        try:
            s6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            s6.bind(("::1", 0))
            s6.listen(1)
            self._keep.append(s6)
            self.should_close[s6.fileno()] = "loopback v6 listener"
        except OSError:
            pass

        # A wildcard listener is reachable from off-host, so it is not the
        # orphaned-rendezvous shape -- and it is how the API server's recorded
        # `::`:8000 survives the close on purpose.
        w = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        w.bind(("0.0.0.0", 0))
        w.listen(1)
        self._keep.append(w)
        self.should_spare[w.fileno()] = "wildcard listener (not loopback)"

        # Established connections are criu's --tcp-close problem, not ours.
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cli.connect(srv.getsockname())
        acc, _ = srv.accept()
        self._keep += [srv, cli, acc]
        self.should_close[srv.fileno()] = "loopback listener with a live peer"
        self.should_spare[cli.fileno()] = "connected client (not listening)"
        self.should_spare[acc.fileno()] = "accepted peer (not listening)"

        # AF_UNIX is the worker pipe's family; it must never be touched.
        self._ux_path = f"/tmp/test-listener-close-{os.getpid()}.sock"
        ux = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        ux.bind(self._ux_path)
        ux.listen(1)
        self._keep.append(ux)
        self.should_spare[ux.fileno()] = "unix listener (not inet)"

    @property
    def all_fds(self):
        return list(self.should_close) + list(self.should_spare)

    def close(self):
        for s in self._keep:
            try:
                s.close()
            except OSError:
                pass
        try:
            os.unlink(self._ux_path)
        except OSError:
            pass


def _tag(fd):
    """``addr:port`` for *fd*, read through a dup so the original is untouched."""
    dup = os.dup(fd)
    s = socket.socket(fileno=dup)
    try:
        if s.family == socket.AF_UNIX:
            return None
        addr = s.getsockname()
        return f"{addr[0]}:{addr[1]}"
    finally:
        s.close()          # closes the dup only


def test_census_selects_the_loopback_listeners_and_closes_nothing():
    if not _ON_LINUX:
        return
    ns = _child_ns()
    shapes = _Shapes()
    try:
        matched = _census_tags(ns)
        for fd, what in shapes.should_close.items():
            assert _tag(fd) in matched, f"census missed {what}"
        # The property this mode exists for: every inspected fd must survive.
        # ``detach()`` in the finally is what makes that true, and the census
        # must not reach ``os.close()`` at all. A census that quietly closed
        # sockets would corrupt the dump it exists to explain.
        for fd in shapes.all_fds:
            assert os.path.exists(f"/proc/{os.getpid()}/fd/{fd}"), (
                f"census closed fd={fd}")
    finally:
        shapes.close()


def test_census_spares_the_shapes_that_are_distinguishable_by_address():
    """Wildcard and AF_UNIX must not be selected.

    The accepted peer is deliberately not checked here: an accepted socket
    reports the *listener's* local ``addr:port``, so it is indistinguishable
    from it in the census output and can only be told apart by fd. That is what
    ``test_close_actually_closes_exactly_the_selected_fds`` does.
    """
    if not _ON_LINUX:
        return
    ns = _child_ns()
    shapes = _Shapes()
    try:
        matched = _census_tags(ns)
        listener_tags = {_tag(fd) for fd in shapes.should_close}
        for fd, what in shapes.should_spare.items():
            tag = _tag(fd)
            if tag is None or tag in listener_tags:
                continue
            assert tag not in matched, f"census selected {what}"
    finally:
        shapes.close()


def test_close_actually_closes_exactly_the_selected_fds():
    if not _ON_LINUX:
        return
    ns = _child_ns()
    shapes = _Shapes()
    try:
        pid = os.getpid()
        before = set(shapes.should_close)
        ns["_close_loopback_listeners"]()[0]
        for fd, what in shapes.should_close.items():
            assert not os.path.exists(f"/proc/{pid}/fd/{fd}"), (
                f"close left {what} open")
        for fd, what in shapes.should_spare.items():
            assert os.path.exists(f"/proc/{pid}/fd/{fd}"), (
                f"close destroyed {what}")
        assert before, "test built no closeable sockets"
    finally:
        shapes.close()


def test_a_fixed_port_loopback_listener_is_never_closed():
    """The regression test for the NCCL RAS failure.

    NCCL's RAS listener sits on ``localhost:28028`` behind a thread no torch
    teardown stops.  Closing it produced a restored process spinning
    ``accept()`` on a dead fd -- 39.7 GB of ``EBADF`` warnings in five minutes
    and a dead tree -- and the census that was supposed to verify the fix
    reported the image as *cleaner* for it, because a socket removed from the
    image looks identical whether or not something still needed it.

    Every socket shape in the original suite was ephemeral-loopback, wildcard,
    non-listening or AF_UNIX, so the bug shipped through a green run on Linux.
    """
    if not _ON_LINUX:
        return
    ns = _child_ns()
    lo, hi = ns["_ephemeral_port_range"]()
    assert lo is not None, "cannot read ip_local_port_range"
    fixed = _fixed_loopback_port()
    if fixed is None:
        return          # no bindable fixed port here; nothing to assert
    shapes = _Shapes()
    try:
        port = fixed.getsockname()[1]
        assert port < lo, "test bound inside the ephemeral range"
        tag = f"127.0.0.1:{port}"

        matched = _census_tags(ns)
        assert tag not in matched, "census selected a fixed-port listener"

        closed, skipped = ns["_close_loopback_listeners"]()
        assert tag not in closed, "closed a fixed-port listener"
        assert tag in skipped, "fixed-port listener was not reported as skipped"
        assert os.path.exists(f"/proc/{os.getpid()}/fd/{fixed.fileno()}"), (
            "fixed-port listener's fd was closed")
        # And the ephemeral ones still go, or the fix has thrown out the cure
        # along with the disease.
        assert closed, "no ephemeral listener was closed"
        for c in closed:
            assert lo <= int(c.rsplit(":", 1)[1]) <= hi
    finally:
        fixed.close()
        shapes.close()


def test_the_tp_gt_1_worker_path_never_closes_a_listener():
    """The regression test for the second RAS failure, asserted structurally.

    At TP>1 torch retires its own rendezvous listeners before the dump -- 28
    across two ranks during init, 2 by dump time -- so the ephemeral loopback
    listener that survives into ``_prepare_worker_dump`` is NCCL RAS's. Closing
    it left RAS calling ``accept()`` on a dead fd after restore, spinning
    ``EBADF`` with no backoff at ~143 MB/s while the job reported ``RUNNING``.

    ``e326442`` exempted RAS's *fixed* ``::1:28028`` listener, which removed one
    of the two failure signatures (``ras/client_support.cc:203`` went to zero)
    and left the other (``misc/socket.cc:458``) untouched, because RAS's second
    listener is ephemeral and so still matched the predicate.

    Asserted against the source rather than by running the function, because
    ``_prepare_worker_dump`` needs a live vLLM worker and the property worth
    protecting is simply that this path cannot reach a close. A behavioural test
    would need the very TP>1 tree that makes this expensive to reproduce.
    """
    tree = ast.parse(open(_CHILD).read())
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef)
               and n.name == "_prepare_worker_dump"), None)
    assert fn is not None, "could not find _prepare_worker_dump"
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_close_loopback_listeners" not in called, (
        "_prepare_worker_dump calls _close_loopback_listeners; at TP>1 the only "
        "ephemeral loopback listener left is NCCL RAS's, and closing it spins "
        "accept() on EBADF after restore. Use _loopback_listeners to census.")
    assert "_loopback_listeners" in called, (
        "_prepare_worker_dump no longer censuses its listeners; the record of "
        "what the image carries is the only dump-side evidence there is")


def test_the_census_helper_cannot_close_anything():
    """``_loopback_listeners`` must not contain a close, by construction.

    The TP>1 path's safety rests entirely on this function being pure, so it is
    asserted rather than left to review. ``detach()`` is what keeps the inspected
    fds alive; an ``os.close`` reintroduced here would be invisible until a
    restore failed.
    """
    tree = ast.parse(open(_CHILD).read())
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef)
               and n.name == "_loopback_listeners"), None)
    assert fn is not None, "could not find _loopback_listeners"
    closes = [n for n in ast.walk(fn)
              if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute)
              and n.func.attr == "close"
              and isinstance(n.func.value, ast.Name)
              and n.func.value.id == "os"]
    assert not closes, "_loopback_listeners calls os.close; it must be pure"


def test_gloo_is_censused_but_never_waited_on():
    """The trap, asserted: waiting on gloo's runloop never terminates.

    Gloo keeps its listening socket for the life of the process, so folding
    ``pt_gloo_runloop`` into the wait list would make every dump pay the full
    poll budget and log a warning. It must appear only in the census.
    """
    ns = _child_ns()
    wait = ns["_WAIT_STORE_THREADS"]
    census = ns["_CENSUS_STORE_THREADS"]
    assert "pt_gloo_runloop" not in wait
    assert "pt_gloo_runloop" in census
    assert set(wait).issubset(set(census))
    for name in ("pt_tcpstore", "pt_nccl_watchdg", "pt_nccl_heartbt"):
        assert name in wait, f"{name} dropped from the wait list"


def test_live_store_threads_reads_this_process():
    if not _ON_LINUX:
        return
    ns = _child_ns()
    # No torch here, so the real answer is the empty list; what is being tested
    # is that it reads /proc without raising and honours the name filter.
    assert ns["_live_store_threads"]() == []
    assert ns["_live_store_threads"](names=("python", "pytest", "")) != []


# --------------------------------------------------------------------------
# The restore-side report
# --------------------------------------------------------------------------

# Verbatim from the failing pod: job 6964ec1c3fd7, /tmp/inst0.log.
EXCERPT = """
(00.119380) 100018: \t\tCreate fd for 69
(00.119380) 100018: inet: \tRestore: family AF_INET    type SOCK_STREAM    proto IPPROTO_TCP      port 42513 state TCP_LISTEN       src_addr 127.0.0.1
(00.119389) 100018: Error (criu/sk-inet.c:1059): inet: Can't bind inet socket (id 464): Address already in use
(00.119396) 100018: Error (criu/files.c:1221): Unable to open fd=70 id=0x1d0
(00.119568) Error (criu/cr-restore.c:2331): Restoring FAILED.
"""

PID_COLLISION = """
(00.011) Error (criu/cr-restore.c:1234): Can't fork for 100018: File exists
"""


def _worker_ns():
    return _lift(
        _WORKER,
        {"_is_port_collision", "_port_collision_report",
         "_tcp_listen_holder", "_ephemeral_port_range"},
        ns={"os": os, "sys": sys,
            "_PORT_COLLISION_MARKER": "semi_p listening-port collision"},
        assigns={"_PORT_COLLISION_SIGNATURES"})


def test_recognises_a_port_collision_and_only_that():
    ns = _worker_ns()
    assert ns["_is_port_collision"](EXCERPT) is True
    # Complication 8's signature must not match, or a live socket would reach a
    # retry gate that cannot help it.
    assert ns["_is_port_collision"](PID_COLLISION) is False
    assert ns["_is_port_collision"]("") is False


def test_report_names_the_port_and_refuses_to_promise_a_retry():
    ns = _worker_ns()
    rep = ns["_port_collision_report"](EXCERPT)
    assert "42513" in rep
    assert "Retrying" in rep
    assert ns["_port_collision_report"]("nothing") is None


def test_report_survives_an_excerpt_with_only_the_socket_id():
    ns = _worker_ns()
    idonly = ("Error: inet: Can't bind inet socket (id 464): "
              "Address already in use")
    rep = ns["_port_collision_report"](idonly)
    assert "464" in rep


def test_report_picks_the_port_that_actually_failed():
    """criu logs every restored socket, so the last LISTEN is the failing one."""
    ns = _worker_ns()
    multi = ("port 34817 state TCP_LISTEN\nport 45383 state TCP_LISTEN\n"
             "port 42513 state TCP_LISTEN\nCan't bind inet socket (id 464): "
             "Address already in use\n")
    rep = ns["_port_collision_report"](multi)
    assert "42513" in rep
    assert "34817" not in rep


if __name__ == "__main__":
    fails = []
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  [ok  ] {name}")
        except AssertionError as exc:
            fails.append(f"{name}: {exc}")
            print(f"  [FAIL] {name}: {exc}")
    if not _ON_LINUX:
        print("  (socket/proc tests skipped: not Linux)")
    print()
    print("FAILURES:", fails if fails else "none")
    raise SystemExit(1 if fails else 0)
