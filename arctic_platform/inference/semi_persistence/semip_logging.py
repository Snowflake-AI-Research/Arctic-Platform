"""Unified logging for the semi-persistence system.

Five role-tagged loggers, all routed through one ``StreamHandler`` per
process.  No file handlers (CRIU-friendly), no cross-process queues.

Logger name hierarchy::

    semip
    |- semip.orch                      (parent process)
    |- semip.slots                     (parent process)
    `- semip.inst.<N>
       |- semip.inst.<N>.instance      (parent process)
       |- semip.inst.<N>.worker        (worker subprocess)
       `- semip.inst.<N>.child         (vLLM subprocess)

Each process calls :func:`init_process` once at startup, then obtains a
scoped logger via the factory functions :func:`orch`, :func:`slots`,
:func:`instance`, :func:`worker`, or :func:`child`.

The per-instance factories return a :class:`_Bound` adapter exposing
:meth:`_Bound.set_gpu` so the GPU column tracks live migrations without
recreating the adapter.

**Where the bytes go.**  The worker and vLLM child point their fd 1 and 2 at
the pod log -- PID 1's stdout, a pipe whose reader is the container runtime --
via :func:`redirect_stdio_to_pod_log`, and the parent attaches a handler on the
same pipe with :func:`attach_pod_log`.  Every ``print``, traceback and vLLM
line from the child and its TP ranks therefore reaches ``kubectl logs``
directly.  CRIU dumps that fd as an external pipe; the restore hands the new
pod's pipe back with ``--inherit-fd`` (see ``worker._worker_criu_save``), so a
restored tree keeps logging with no rebinding step.

In a Ray actor, :func:`tee_tree_to_ray` puts a pipe of the actor's own in
between: the tree writes to it, and a thread in the actor copies each line to
the pod log and to the actor's stdout, which is the file Ray forwards to the
driver (``ct tui``, ``ct download-log``).  It is still a pipe, so dump and
restore are unchanged.

Several replicas share one pod log, so every line carries ``r<K>`` (the
replica slot from ``SEMIP_REPLICA_ID``) and the role of the process.
"""
import logging
import os
import stat
import sys
import threading
import time

_FMT = ("%(asctime)s.%(msecs)03d [r%(replica)s %(role)s] %(levelname)-4s "
        "%(loc)s pid=%(pidn)d %(message)s")
_DATEFMT = "%H:%M:%S"

# The pod log.  PID 1's stdout is the container's log stream; opening it through
# /proc gives a new write end on the same pipe.  Overridable so an experiment run
# by hand can point the tree at its own terminal pipe instead.
_POD_LOG_ENV = "SEMIP_LOG_TARGET"
_POD_LOG_DEFAULT = "/proc/1/fd/1"

# "0" keeps the tree's output out of the Ray actor's stdout (tee_tree_to_ray).
_TO_RAY_ENV = "SEMIP_LOG_TO_RAY"

_REPLICA_ENV = "SEMIP_REPLICA_ID"

# Last component of a semip logger name -> role.  Records from other loggers
# take the role the process declared in init_process.
_ROLES = ("orch", "slots", "instance", "worker", "child")
_ENGINE_LOGGER = "arctic_platform.inference.server.semip"

_process_role = "py"


class _Ctx(logging.Filter):
    def filter(self, record):
        record.pidn = os.getpid()
        record.loc = f"{record.filename}:{record.lineno}"
        record.replica = os.environ.get(_REPLICA_ENV) or "0"
        last = record.name.rsplit(".", 1)[-1]
        if record.name.startswith("semip.") and last in _ROLES:
            record.role = last
        elif record.name.startswith(_ENGINE_LOGGER):
            record.role = "engine"
        else:
            record.role = _process_role
        if not hasattr(record, "scope"):
            record.scope = ""
        return True


_FORMATTER = logging.Formatter(_FMT, _DATEFMT)
_FILTER = _Ctx()
_root_handler: logging.StreamHandler | None = None

# Opened at most once per process: the parent attaches one handler per Instance,
# and a fresh fd each time would leak one per restore.
_pod_log_stream = None
_warned_target = False

# Set once by tee_tree_to_ray: the write end the tree opens through /proc (held
# open for the actor's life, or the path stops resolving), and the real pod log.
_tee_write_fd = None
_tee_pod_log_fd = None
_direct_stream = None


def init_process(level: int = logging.INFO, role: str | None = None) -> None:
    """Install one root ``StreamHandler(stdout)`` for this process.

    Idempotent: re-configures the root logger's handlers each call.
    Should be called as early as possible in every process that uses
    semip logging (parent, worker subprocess, vLLM child subprocess).
    ``role`` tags records from non-semip loggers routed through the root.
    """
    global _root_handler, _process_role
    if role:
        _process_role = role
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_FORMATTER)
    h.addFilter(_FILTER)
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(level)
    _root_handler = h


def rebind_stdout() -> None:
    """Re-point the root ``StreamHandler`` at the current ``sys.stdout``."""
    if _root_handler is not None:
        _root_handler.setStream(sys.stdout)


# ---------------------------------------------------------------------------
# The pod log
# ---------------------------------------------------------------------------

def pod_log_target() -> str:
    """The path the pod log is opened from."""
    return os.environ.get(_POD_LOG_ENV) or _POD_LOG_DEFAULT


def _warn_once(msg: str) -> None:
    global _warned_target
    if _warned_target:
        return
    _warned_target = True
    print(f"[semip] WARNING: {msg}", file=sys.stderr, flush=True)


def open_pod_log() -> int:
    """Open a write end on the pod log, or on ``/dev/null`` if there is none.

    Returns a new fd with ``O_CLOEXEC`` set.  Only a pipe is accepted: CRIU
    reopens a regular file by path and re-validates it at restore, whereas a pipe
    is handed back with ``--inherit-fd``, which is the whole mechanism.  Outside
    a pod (no readable PID 1, a terminal, ``/dev/null``) the tree logs to
    ``/dev/null`` and says so once, rather than capturing a tty CRIU could not
    reattach.

    ``O_NONBLOCK`` for the open only: a FIFO opened for writing with no reader
    blocks forever without it, and fails with ``ENXIO`` with it.  It is cleared
    straight after, because a non-blocking stdout drops lines with ``EAGAIN``
    whenever the reader falls behind.
    """
    path = pod_log_target()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as exc:
        fd = None
        reason = f"cannot open {path} ({exc.strerror})"
    else:
        if stat.S_ISFIFO(os.fstat(fd).st_mode):
            import fcntl
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
            return fd
        os.close(fd)
        fd = None
        reason = f"{path} is not a pipe"
    _warn_once(f"{reason}; semi-p output goes to /dev/null. Set "
               f"{_POD_LOG_ENV} to a pipe (e.g. /proc/<shell pid>/fd/1) to "
               f"see it.")
    return os.open(os.devnull, os.O_WRONLY | os.O_CLOEXEC)


def redirect_stdio_to_pod_log() -> str:
    """Point this process's fd 1 and 2 at the pod log, line-buffered.

    Used at the top of the worker and the vLLM child, so every byte they emit
    -- ``log.info`` records, ``print`` calls, tracebacks, vLLM's own logging,
    and the TP ranks the child spawns, which inherit both fds -- lands in the
    pod log.  Writes below ``PIPE_BUF`` (4 KiB) are atomic, so whole lines from
    concurrent replicas do not interleave.

    Returns what fd 1 now refers to (``pipe:[N]`` or ``/dev/null``).
    """
    fd = open_pod_log()
    for std_fd in (1, 2):
        os.dup2(fd, std_fd)
    os.close(fd)
    # Without line buffering third-party writes (vLLM banners, stray prints)
    # sit in an 8 KiB block buffer until the process exits or crashes.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, OSError):
            pass
    rebind_stdout()
    try:
        return os.readlink("/proc/self/fd/1")
    except OSError:
        return "?"


def _write_all(fd, data) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _copy_lines(read_fd, dest_fds) -> None:
    """Copy *read_fd* to every fd in *dest_fds*, line by line, until EOF.

    One write per line, so a line up to ``PIPE_BUF`` (4 KiB) lands whole in a
    pod log other replicas write to as well.  The read is capped so a line that
    never ends (a ``\\r`` progress bar) is still passed on.  A destination that
    fails is dropped and the others keep going.  Nothing else ends the loop:
    if it stopped reading, the pipe would fill and every writer in the tree
    would block.
    """
    src = os.fdopen(read_fd, "rb", buffering=1 << 20)
    live = list(dest_fds)
    while True:
        try:
            line = src.readline(1 << 16)
            if not line:
                return
            for fd in list(live):
                try:
                    _write_all(fd, line)
                except OSError as exc:
                    live.remove(fd)
                    note = (f"[semip] WARNING: log copy to fd {fd} failed "
                            f"({exc.strerror}); dropped it\n").encode()
                    for other in live:
                        try:
                            _write_all(other, note)
                        except OSError:
                            pass
        except Exception:  # noqa: BLE001
            time.sleep(0.1)


def _set_pipe_size(fd, size) -> None:
    import fcntl
    op = getattr(fcntl, "F_SETPIPE_SZ", None)  # Linux only
    if op is None:
        return
    try:
        fcntl.fcntl(fd, op, size)
    except OSError:
        pass


def tee_tree_to_ray(ray_fd: int = 1) -> None:
    """Make the tree's output reach this Ray actor's stdout as well as the pod log.

    For the Ray actor, before it creates its first ``Instance``.  The actor's
    stdout (``ray_fd``) is the file Ray forwards to the driver; the tree's fd 1
    and 2 point at the pod log instead, so without this none of the worker's,
    the child's or the TP ranks' lines reach ``ct tui``.  The tree cannot write
    to Ray's file itself: CRIU reopens a regular file by its path, and that
    path differs in every job.

    So the actor makes a pipe and points ``SEMIP_LOG_TARGET`` at its write end
    through ``/proc``.  Everything that opens the pod log -- the worker, the
    child at startup, ``_stdout_inherit`` at restore, and the ``semip.*``
    handlers :func:`attach_pod_log` adds -- then gets this pipe, and a thread
    copies each line to both the real pod log and Ray's file.

    Idempotent.  Does nothing if ``SEMIP_LOG_TO_RAY=0``.  Never raises: on any
    failure the target is left alone and the tree logs to the pod log only,
    as before.
    """
    global _tee_write_fd, _tee_pod_log_fd
    if _tee_write_fd is not None or os.environ.get(_TO_RAY_ENV) == "0":
        return
    opened = []
    try:
        pod = open_pod_log()
        opened.append(pod)
        ray = os.dup(ray_fd)
        opened.append(ray)
        r, w = os.pipe()
        opened += [r, w]
        _set_pipe_size(w, 1 << 20)
        threading.Thread(target=_copy_lines, args=(r, [ray, pod]),
                         name="semip-log-tee", daemon=True).start()
    except Exception as exc:  # noqa: BLE001
        for fd in opened:
            try:
                os.close(fd)
            except OSError:
                pass
        print(f"[semip] WARNING: could not send the tree's output to Ray: "
              f"{exc!r}", file=sys.stderr, flush=True)
        return
    _tee_write_fd, _tee_pod_log_fd = w, pod
    os.environ[_POD_LOG_ENV] = f"/proc/{os.getpid()}/fd/{w}"


def _direct_pod_log_stream():
    """The pod log itself, bypassing the tee when there is one."""
    global _direct_stream
    if _tee_pod_log_fd is None:
        return _pod_log_stream
    if _direct_stream is None:
        _direct_stream = os.fdopen(os.dup(_tee_pod_log_fd), "w", buffering=1)
    return _direct_stream


def attach_pod_log(logger_names) -> None:
    """Send the named loggers to the pod log from this (parent) process.

    For the Ray actor that drives an ``Instance``: its own stdout goes to Ray's
    per-worker file, not to the pod log.  ``semip.*`` loggers stop propagating,
    since the actor's root logger has no sink of its own; any other logger keeps
    propagating, so a sink it already has (``arctic_platform.inference``'s, which Ray
    forwards to the driver) still sees the record.

    With :func:`tee_tree_to_ray` active, which has to run first, ``semip.*``
    records go through the tee and so reach Ray too.  Other loggers write to
    the pod log directly: they reach Ray already, by propagating, and through
    the tee they would show up there twice.

    Idempotent per logger.  Never raises: losing the log costs evidence, not
    correctness, and this runs in ``Instance.__init__``.
    """
    global _pod_log_stream
    try:
        if _pod_log_stream is None:
            _pod_log_stream = os.fdopen(open_pod_log(), "w", buffering=1)
        for name in logger_names:
            logger = logging.getLogger(name)
            if any(getattr(h, "_semip_pod_log", False) for h in logger.handlers):
                continue
            h = logging.StreamHandler(_pod_log_stream if name.startswith("semip")
                                      else _direct_pod_log_stream())
            h._semip_pod_log = True
            h.setFormatter(_FORMATTER)
            h.addFilter(_FILTER)
            logger.addHandler(h)
            if logger.level == logging.NOTSET:
                logger.setLevel(logging.INFO)
            if name.startswith("semip"):
                logger.propagate = False
    except Exception as exc:  # noqa: BLE001
        print(f"[semip] WARNING: could not attach the pod log: {exc!r}",
              file=sys.stderr, flush=True)


def _scope(inst_id, role: str, gpu) -> str:
    """Build the scope string for the per-instance roles.

    The role word itself is omitted because the file name (rendered by
    the formatter as ``filename:lineno``) already identifies it.

    The worker process itself runs on the host CPU; its ``gpu`` reflects
    the GPU its child is currently bound to.  Render that as
    ``child=gpuN`` so it isn't mistaken for the worker's own device.

    The formatter pads the field to a fixed width so the trailing
    ``pid=`` column stays aligned across roles (and across the empty
    scopes used by orch / slots).
    """
    if role == "worker":
        return f"i{inst_id} child=gpu{gpu}"
    return f"i{inst_id} gpu{gpu}"


class _Bound(logging.LoggerAdapter):
    """LoggerAdapter exposing a mutable ``scope`` (and ``set_gpu``).

    The ``scope`` field is injected via ``extra`` and rendered by the
    formatter.  For per-instance roles the scope encodes inst id, role
    name, and current GPU; the GPU portion is mutated in place by
    :meth:`set_gpu` so subsequent log calls reflect a live migration.
    """

    def __init__(self, logger, scope: str, *,
                 inst_id=None, role: str | None = None, gpu=None):
        super().__init__(logger, {"scope": scope})
        self._inst_id = inst_id
        self._role = role
        self._gpu = gpu

    def process(self, msg, kwargs):
        kwargs.setdefault("extra", {})["scope"] = self.extra["scope"]
        return msg, kwargs

    def set_gpu(self, gpu) -> None:
        if self._role is None or self._inst_id is None:
            return
        self._gpu = gpu
        self.extra["scope"] = _scope(self._inst_id, self._role, gpu)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def orch() -> _Bound:
    return _Bound(logging.getLogger("semip.orch"), scope="")


def slots() -> _Bound:
    return _Bound(logging.getLogger("semip.slots"), scope="")


def instance(inst_id, gpu) -> _Bound:
    return _Bound(
        logging.getLogger(f"semip.inst.{inst_id}.instance"),
        scope=_scope(inst_id, "instance", gpu),
        inst_id=inst_id, role="instance", gpu=gpu,
    )


def worker(inst_id, gpu) -> _Bound:
    return _Bound(
        logging.getLogger(f"semip.inst.{inst_id}.worker"),
        scope=_scope(inst_id, "worker", gpu),
        inst_id=inst_id, role="worker", gpu=gpu,
    )


def child(inst_id, gpu) -> _Bound:
    return _Bound(
        logging.getLogger(f"semip.inst.{inst_id}.child"),
        scope=_scope(inst_id, "child", gpu),
        inst_id=inst_id, role="child", gpu=gpu,
    )
