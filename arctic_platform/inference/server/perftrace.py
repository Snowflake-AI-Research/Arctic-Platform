"""Opt-in performance trace for the generate path: scheduler and workers.

Off unless the process environment, or the mapping passed to :func:`configure`,
sets ``ARCTIC_PERFTRACE=1``. Every setting is also read under the
``DSS_PERFTRACE`` prefix, so the one ``extra_env`` switch a DSS job sets turns
on both layers; the ``ARCTIC_`` spelling wins when both are present.

When on, the process keeps an in-memory ring of records
``(request_id, point, wall_ns, extra)``, the same shape as
``dss.ray_dss.perftrace``:

* the scheduler marks each traced request as it is submitted, dispatched to a
  worker and resolved, and re-records the marks the worker returns under the
  same id;
* :func:`start_sampler` adds, once a second, a ``sample`` of the process (CPU
  per process and thread, optionally across the process tree so a worker
  covers its vLLM EngineCore, event-loop lag, RSS and registered gauges) and,
  in a worker, an ``engine`` record aggregating vLLM's per-step stats. A host
  process that already samples itself can report :func:`collect_gauges`
  instead of starting a second sampler.

The caller owns transport: :func:`drain` returns the ring as one source dict.

Settings (read once, at the first :func:`configure` that enables):

``*_PERFTRACE``           ``1`` to enable.
``*_PERFTRACE_LAG_MS``    loop-lag probe period (default 10).
``*_PERFTRACE_PYSPY``     ``start_s:duration_s[,...]`` py-spy captures, timed
                          from enable; ``*_PERFTRACE_PYSPY_ARGS`` adds flags.
``*_PERFTRACE_RING``      ring capacity in records (default 1,000,000).
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import contextvars
import ctypes
import inspect
import logging
import os
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

logger = logging.getLogger("arctic_platform.inference.server")

_PREFIXES = ("ARCTIC_PERFTRACE", "DSS_PERFTRACE")
_TRUE = frozenset({"1", "true", "yes", "on"})
_DEFAULT_RING = 1_000_000

_enabled = False
_buf: "collections.deque[tuple]" = collections.deque(maxlen=_DEFAULT_RING)
_role = ""
_settings: dict[str, Any] = {}
_configure_lock = threading.Lock()
_counters: dict[str, int] = {}
_gauges: list[Callable[[], Callable[[], dict[str, Any]] | None]] = []
_sampler_started = False

_trace_id: "contextvars.ContextVar[str | None]" = contextvars.ContextVar(
    "arctic_perftrace_trace_id", default=None
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _setting(env: Mapping[str, Any], suffix: str, default: str = "") -> str:
    for prefix in _PREFIXES:
        value = env.get(prefix + suffix)
        if value is not None:
            return str(value)
    return default


def _float_setting(env: Mapping[str, Any], suffix: str, default: float) -> float:
    try:
        return float(_setting(env, suffix, str(default)))
    except ValueError:
        return default


def configure(env: Mapping[str, Any] | None = None, *, role: str) -> bool:
    """Enable tracing in this process if ``env`` (over ``os.environ``) asks for it.

    Returns whether tracing is on. The first call that enables fixes the role
    and settings for the life of the process; later calls are no-ops.
    """
    global _enabled, _buf, _role, _settings
    merged: dict[str, Any] = dict(os.environ)
    merged.update(env or {})
    if _setting(merged, "").strip().lower() not in _TRUE:
        return _enabled
    with _configure_lock:
        if _enabled:
            return True
        ring = int(_float_setting(merged, "_RING", _DEFAULT_RING))
        _settings = {
            "lag_ms": _float_setting(merged, "_LAG_MS", 10.0),
            "pyspy": _setting(merged, "_PYSPY"),
            "pyspy_args": _setting(merged, "_PYSPY_ARGS"),
            "ring": max(ring, 1000),
        }
        _buf = collections.deque(maxlen=_settings["ring"])
        _role = role
        _enabled = True
    event("meta", {
        "role": role,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "clock_mono_ns": time.monotonic_ns(),
        **_settings,
    })
    logger.info("arctic perftrace enabled: role=%s settings=%s", role, _settings)
    return True


def enabled() -> bool:
    return _enabled


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def mark(request_id: str | None, point: str, extra: dict[str, Any] | None = None) -> None:
    if _enabled:
        _buf.append((request_id, point, time.time_ns(), extra))


def event(point: str, extra: dict[str, Any] | None = None) -> None:
    mark(None, point, extra)


def extend(records: Iterable[tuple]) -> None:
    """Append ready-made ``(request_id, point, wall_ns, extra)`` records."""
    if _enabled:
        _buf.extend(records)


def counter_add(name: str, delta: int = 1) -> None:
    """Bump a counter reported in every ``sample``. Event-loop thread only."""
    if _enabled:
        _counters[name] = _counters.get(name, 0) + delta


def drain() -> dict[str, Any]:
    """Return and clear this process's ring as one source dict."""
    buf = _buf
    marks: list[tuple] = []
    full = len(buf) == buf.maxlen
    try:
        while True:
            marks.append(buf.popleft())
    except IndexError:
        pass
    return {
        "src": _role,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "ring_full": full,
        "marks": marks,
    }


def current_trace_id() -> str | None:
    """The trace id the caller bound with :func:`trace_context`, if tracing is on."""
    return _trace_id.get() if _enabled else None


@contextlib.contextmanager
def trace_context(trace_id: str | None) -> Iterator[None]:
    """Bind ``trace_id`` to work started in this context, tasks it creates included."""
    token = _trace_id.set(trace_id)
    try:
        yield
    finally:
        _trace_id.reset(token)


# ---------------------------------------------------------------------------
# vLLM engine stats (worker)
# ---------------------------------------------------------------------------


class _EngineStats:
    """Per-second aggregate of what vLLM's stat logger reports each step."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self.steps = 0
        self.gen_tokens = 0
        self.prompt_tokens = 0
        self.preempted = 0
        self.running_max = 0
        self.waiting_max = 0
        self.running_last = 0
        self.waiting_last = 0
        self.kv_max = 0.0
        self.prefix_queries = 0
        self.prefix_hits = 0
        self.ttft: list[float] = []
        self.finished: list[tuple[float, float, float, float, int]] = []

    def record(self, scheduler_stats: Any, iteration_stats: Any) -> None:
        # Field names vary across vLLM releases; anything missing reads as 0.
        with self._lock:
            self.steps += 1
            if scheduler_stats is not None:
                running = int(getattr(scheduler_stats, "num_running_reqs", 0))
                waiting = int(getattr(scheduler_stats, "num_waiting_reqs", 0))
                self.running_last, self.waiting_last = running, waiting
                self.running_max = max(self.running_max, running)
                self.waiting_max = max(self.waiting_max, waiting)
                self.kv_max = max(self.kv_max, float(getattr(scheduler_stats, "kv_cache_usage", 0.0)))
                pcs = getattr(scheduler_stats, "prefix_cache_stats", None)
                if pcs is not None:
                    self.prefix_queries += int(getattr(pcs, "queries", 0))
                    self.prefix_hits += int(getattr(pcs, "hits", 0))
            if iteration_stats is not None:
                self.gen_tokens += int(getattr(iteration_stats, "num_generation_tokens", 0))
                pts = getattr(iteration_stats, "prompt_token_stats", None)
                if pts is not None and hasattr(pts, "computed"):
                    self.prompt_tokens += int(pts.computed)
                else:
                    self.prompt_tokens += int(getattr(iteration_stats, "num_prompt_tokens", 0))
                self.preempted += int(getattr(iteration_stats, "num_preempted_reqs", 0))
                self.ttft.extend(getattr(iteration_stats, "time_to_first_tokens_iter", None) or ())
                for f in getattr(iteration_stats, "finished_requests", None) or ():
                    self.finished.append((
                        float(getattr(f, "e2e_latency", 0.0)),
                        float(getattr(f, "queued_time", 0.0)),
                        float(getattr(f, "prefill_time", 0.0)),
                        float(getattr(f, "inference_time", 0.0)),
                        int(getattr(f, "num_prompt_tokens", 0)),
                    ))

    def pop(self) -> dict[str, Any] | None:
        with self._lock:
            if self.steps == 0:
                return None
            out: dict[str, Any] = {
                "steps": self.steps,
                "gen_tokens": self.gen_tokens,
                "prompt_tokens": self.prompt_tokens,
                "preempted": self.preempted,
                "running": [self.running_last, self.running_max],
                "waiting": [self.waiting_last, self.waiting_max],
                "kv_max": round(self.kv_max, 4),
                "prefix": [self.prefix_queries, self.prefix_hits],
                "ttft_ms": _dist(self.ttft),
                "finished": len(self.finished),
            }
            if self.finished:
                cols = list(zip(*self.finished))
                out["e2e_ms"] = _dist(cols[0])
                out["queued_ms"] = _dist(cols[1])
                out["prefill_ms"] = _dist(cols[2])
                out["inference_ms"] = _dist(cols[3])
                out["finished_prompt_tokens"] = sum(cols[4])
            self._reset()
            return out


def _dist(values: Iterable[float]) -> list[float] | None:
    """``[n, p50, p90, max]`` in milliseconds for values in seconds."""
    v = sorted(values)
    if not v:
        return None
    n = len(v)
    return [n, round(v[n // 2] * 1e3, 2), round(v[min(n - 1, int(n * 0.9))] * 1e3, 2), round(v[-1] * 1e3, 2)]


_engine = _EngineStats()


_engine_error_logged = False


def record_engine_stats(scheduler_stats: Any, iteration_stats: Any) -> None:
    """Fold one vLLM stat-logger callback into the per-second ``engine`` record.

    Never raises: it runs inside vLLM's output loop.
    """
    global _engine_error_logged
    if not _enabled:
        return
    try:
        _engine.record(scheduler_stats, iteration_stats)
    except Exception as exc:
        if not _engine_error_logged:
            _engine_error_logged = True
            logger.exception("arctic perftrace: engine stats unreadable")
            event("engine_error", {"error": repr(exc)[:500]})


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


def register_gauges(fn: Callable[[], dict[str, Any]]) -> None:
    """Add gauges to every ``sample``. ``fn`` runs on the sampler thread.

    Bound methods are held weakly, so registering does not keep their object alive.
    """
    ref = weakref.WeakMethod(fn) if inspect.ismethod(fn) else (lambda: fn)
    _gauges.append(ref)


def collect_gauges() -> dict[str, Any]:
    """Registered gauges plus counters, for a sampler outside this module to report."""
    out: dict[str, Any] = {}
    for ref in list(_gauges):
        fn = ref()
        if fn is None:
            _gauges.remove(ref)
            continue
        try:
            out.update(fn())
        except Exception as exc:
            out.setdefault("errors", []).append(repr(exc))
    if _counters:
        out["counters"] = dict(_counters)
    return out


_CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _read_stat(path: str) -> tuple[str, int] | None:
    """``(comm, utime+stime ticks)`` from a ``/proc/.../stat`` file."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read().decode(errors="replace")
    except OSError:
        return None
    lpar, rpar = raw.find("("), raw.rfind(")")
    fields = raw[rpar + 2:].split()
    return raw[lpar + 1:rpar], int(fields[11]) + int(fields[12])


def _children(pid: int) -> list[int]:
    out: list[int] = []
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for tid in tids:
        try:
            with open(f"/proc/{pid}/task/{tid}/children") as fh:
                out.extend(int(c) for c in fh.read().split())
        except OSError:
            continue
    return out


def _process_tree(root: int, depth: int = 3) -> list[int]:
    pids, frontier = [root], [root]
    for _ in range(depth):
        frontier = [c for p in frontier for c in _children(p)]
        pids.extend(frontier)
    return pids


class _LoopLag:
    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self._bucket = [0, 0.0, 0.0, 0]  # n, sum, max, >50ms

    async def run(self) -> None:
        while True:
            start = time.monotonic()
            await asyncio.sleep(self.interval_s)
            lag = time.monotonic() - start - self.interval_s
            b = self._bucket
            b[0] += 1
            b[1] += lag
            if lag > b[2]:
                b[2] = lag
            if lag > 0.050:
                b[3] += 1

    def pop(self) -> dict[str, Any]:
        b, self._bucket = self._bucket, [0, 0.0, 0.0, 0]
        n = b[0]
        return {
            "n": n,
            "mean_ms": round(b[1] / n * 1e3, 3) if n else None,
            "max_ms": round(b[2] * 1e3, 3),
            "gt50": b[3],
        }


class _Sampler:
    def __init__(self, process_tree: bool, lag: _LoopLag | None) -> None:
        self.process_tree = process_tree
        self.lag = lag
        self._prev: dict[tuple[int, int], int] = {}
        self._prev_proc: dict[int, int] = {}
        self._prev_t = time.monotonic()

    def run(self) -> None:
        while True:
            time.sleep(1.0)
            try:
                self._sample()
            except Exception:
                logger.exception("arctic perftrace sampler failed")

    def _sample(self) -> None:
        now = time.monotonic()
        dt = max(now - self._prev_t, 1e-6)
        self._prev_t = now
        me = os.getpid()
        pids = _process_tree(me) if self.process_tree else [me]
        names = {t.native_id: t.name for t in threading.enumerate() if t.native_id is not None}
        procs: list[list[Any]] = []
        threads: list[list[Any]] = []
        cur: dict[tuple[int, int], int] = {}
        cur_proc: dict[int, int] = {}
        for pid in pids:
            stat = _read_stat(f"/proc/{pid}/stat")
            if stat is None:
                continue
            cur_proc[pid] = stat[1]
            prev = self._prev_proc.get(pid)
            if prev is not None:
                procs.append([pid, stat[0], round((stat[1] - prev) / _CLK_TCK / dt, 3)])
            try:
                tids = os.listdir(f"/proc/{pid}/task")
            except OSError:
                continue
            for tid_s in tids:
                tstat = _read_stat(f"/proc/{pid}/task/{tid_s}/stat")
                if tstat is None:
                    continue
                key = (pid, int(tid_s))
                cur[key] = tstat[1]
                tprev = self._prev.get(key)
                if tprev is None:
                    continue
                busy = (tstat[1] - tprev) / _CLK_TCK / dt
                if busy >= 0.005:
                    name = names.get(key[1], tstat[0]) if pid == me else tstat[0]
                    threads.append([pid, key[1], name, round(busy, 3)])
        self._prev, self._prev_proc = cur, cur_proc
        threads.sort(key=lambda t: -t[3])
        rss = None
        try:
            with open("/proc/self/statm") as fh:
                rss = int(fh.read().split()[1]) * _PAGE
        except OSError:
            pass
        data: dict[str, Any] = {
            "procs": procs,
            "threads": threads[:24],
            "rss": rss,
        }
        if self.lag is not None:
            data["loop"] = self.lag.pop()
        gauges = collect_gauges()
        if gauges:
            data["gauges"] = gauges
        event("sample", data)
        engine = _engine.pop()
        if engine is not None:
            event("engine", engine)


def start_sampler(*, process_tree: bool = False, loop_lag: bool = True) -> None:
    """Start the once-a-second sampler (and py-spy schedule). Once per process.

    ``process_tree`` extends the CPU census to child processes, e.g. a
    worker's vLLM EngineCore and its executor workers. Call from the event-loop
    thread when ``loop_lag`` is set so the probe runs on that loop.
    """
    global _sampler_started
    if not _enabled or _sampler_started:
        return
    _sampler_started = True
    lag: _LoopLag | None = None
    if loop_lag:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            lag = _LoopLag(max(_settings.get("lag_ms", 10.0), 1.0) / 1e3)
            loop.create_task(lag.run(), name="arctic-perftrace-loop-lag")
    sampler = _Sampler(process_tree, lag)
    threading.Thread(target=sampler.run, name="arctic-perftrace-sampler", daemon=True).start()
    if _settings.get("pyspy"):
        _schedule_pyspy(_settings["pyspy"], _settings.get("pyspy_args", ""), subprocesses=process_tree)


# ---------------------------------------------------------------------------
# py-spy captures
# ---------------------------------------------------------------------------


def _parse_pyspy(spec: str) -> list[tuple[float, float]]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        start, _, dur = part.partition(":")
        try:
            out.append((float(start), float(dur or 30)))
        except ValueError:
            logger.warning("arctic perftrace: ignoring bad py-spy spec %r", part)
    return out


def _allow_ptrace_by_any() -> None:
    """Let a same-uid process trace this one under Yama ptrace_scope=1."""
    pr_set_ptracer = 0x59616D61
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(pr_set_ptracer, ctypes.c_ulong(-1 & 0xFFFFFFFFFFFFFFFF), 0, 0, 0)


def run_pyspy(duration_s: float, extra_args: str = "", *, subprocesses: bool = False) -> None:
    """Record this process with py-spy and store the collapsed stacks as a ``pyspy`` event."""
    exe = shutil.which("py-spy")
    start_ns = time.time_ns()
    if exe is None:
        event("pyspy", {"start_ns": start_ns, "error": "py-spy not on PATH"})
        return
    try:
        _allow_ptrace_by_any()
    except Exception as exc:
        logger.warning("arctic perftrace: prctl(PR_SET_PTRACER) failed: %r", exc)
    out = Path(f"/tmp/arctic-perftrace-pyspy-{os.getpid()}-{start_ns}.txt")
    cmd = [
        exe, "record", "--pid", str(os.getpid()), "--duration", str(int(duration_s)),
        "--rate", "100", "--format", "raw", "--output", str(out), "--nonblocking",
    ]
    if subprocesses:
        cmd.append("--subprocesses")
    cmd += shlex.split(extra_args)
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=duration_s + 120)
        collapsed = out.read_text(errors="replace") if out.exists() else ""
        event("pyspy", {
            "start_ns": start_ns,
            "duration_s": duration_s,
            "rc": proc.returncode,
            "stderr": proc.stderr.decode(errors="replace")[-4000:],
            "collapsed": collapsed,
        })
    except Exception as exc:
        event("pyspy", {"start_ns": start_ns, "error": repr(exc)})
    finally:
        out.unlink(missing_ok=True)


def _schedule_pyspy(spec: str, extra_args: str, *, subprocesses: bool) -> None:
    for start_s, dur_s in _parse_pyspy(spec):
        timer = threading.Timer(
            start_s, run_pyspy, args=(dur_s, extra_args), kwargs={"subprocesses": subprocesses}
        )
        timer.daemon = True
        timer.name = "arctic-perftrace-pyspy"
        timer.start()
