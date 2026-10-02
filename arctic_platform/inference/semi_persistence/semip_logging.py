"""Unified logging for the semi-persistence system.

Five role-tagged loggers, all routed through one ``StreamHandler`` per
process to inherited stdout.  No file handlers (CRIU-friendly), no
cross-process queues.

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
"""
import logging
import os
import sys

_FMT_FILE = ("%(asctime)s.%(msecs)03d %(loc)-20s "
             "%(levelname)-4s pid=%(pidn)-7d %(message)s")
_FMT_TERMINAL = ("%(asctime)s.%(msecs)03d %(loc)-20s "
                 "%(levelname)-4s %(message)s")
_DATEFMT = "%H:%M:%S"


class _Ctx(logging.Filter):
    def filter(self, record):
        record.pidn = os.getpid()
        record.loc = f"{record.filename}:{record.lineno}"
        if not hasattr(record, "scope"):
            record.scope = ""
        return True


_FORMATTER_FILE = logging.Formatter(_FMT_FILE, _DATEFMT)
_FORMATTER_TERMINAL = logging.Formatter(_FMT_TERMINAL, _DATEFMT)
_FILTER = _Ctx()
_root_handler: logging.StreamHandler | None = None


def init_process(level: int = logging.INFO) -> None:
    """Install one root ``StreamHandler(stdout)`` for this process.

    Idempotent: re-configures the root logger's handlers each call.
    Should be called as early as possible in every process that uses
    semip logging (parent, worker subprocess, vLLM child subprocess).
    """
    global _root_handler
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_FORMATTER_TERMINAL)
    h.addFilter(_FILTER)
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(level)
    _root_handler = h


def rebind_stdout() -> None:
    """Re-point the root ``StreamHandler`` at the current ``sys.stdout``.

    Used by :mod:`vllm_child` after a CRIU restore where the inherited
    stdout fd is broken and the process re-assigns ``sys.stdout`` to a
    fallback file handle.  Without this call, ``logging`` would keep
    writing to the cached (broken) stream.
    """
    if _root_handler is not None:
        _root_handler.setStream(sys.stdout)


# ---------------------------------------------------------------------------
# Per-instance file plumbing
# ---------------------------------------------------------------------------

_INSTANCE_LOG_TEMPLATE = "/tmp/inst{}.log"


def instance_log_path(inst_id) -> str:
    """Return the canonical per-instance log file path."""
    return _INSTANCE_LOG_TEMPLATE.format(inst_id)


def truncate_instance_file(inst_id) -> str:
    """Truncate the per-instance log file (best-effort)."""
    path = instance_log_path(inst_id)
    try:
        open(path, "w").close()
    except OSError:
        pass
    return path


def attach_instance_file(inst_id) -> str:
    """Attach a ``FileHandler`` to ``semip.inst.<N>`` and silence terminal.

    Used in the parent process so that ``instance.N`` records land in
    ``/tmp/inst{N}.log`` only (not in the terminal, which keeps the
    ``orch`` / ``slots`` lines uncluttered).  Worker and child
    subprocesses route into the same file via ``dup2``'d stdout.

    Idempotent.
    """
    path = instance_log_path(inst_id)
    logger = logging.getLogger(f"semip.inst.{inst_id}")
    for h in logger.handlers:
        if (isinstance(h, logging.FileHandler)
                and getattr(h, "_semip_inst_path", None) == path):
            return path
    h = logging.FileHandler(path, mode="a")
    h._semip_inst_path = path
    h.setFormatter(_FORMATTER_FILE)
    h.addFilter(_FILTER)
    logger.addHandler(h)
    logger.propagate = False
    return path


def redirect_stdio_to_instance_file(inst_id) -> str:
    """Redirect this process's ``sys.stdout`` / ``sys.stderr`` to the
    per-instance log file via ``dup2``, and rebind the root
    ``StreamHandler`` to follow.

    Used at the top of worker and vLLM child subprocesses so every byte
    they emit -- ``log.info`` records, ``print`` calls, tracebacks,
    vLLM's own ``logging`` -- lands in the same per-instance file.

    The fd is opened with ``O_APPEND`` so concurrent writes from the
    parent (FileHandler) and from this subprocess interleave atomically
    for lines below the kernel's ``PIPE_BUF`` (4 KiB on Linux).
    """
    path = instance_log_path(inst_id)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
    for std_fd in (1, 2):
        os.dup2(fd, std_fd)
    os.close(fd)
    # Force line buffering so third-party stdout writes (vLLM banners,
    # transformers warnings, stray prints) reach the file promptly
    # instead of sitting in Python's default ~8 KiB block buffer until
    # the process exits or crashes.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, OSError):
            pass
    rebind_stdout()
    if _root_handler is not None:
        _root_handler.setFormatter(_FORMATTER_FILE)
    return path


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
