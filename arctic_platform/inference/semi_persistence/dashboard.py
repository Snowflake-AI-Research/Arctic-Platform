#!/usr/bin/env python3
"""Terminal dashboard for the Orchestrator.

Polls GET /state from the orchestrator's embedded HTTP server and renders
a three-tier curses display:

    Top tier    -- one column per GPU, showing models in up/running state
    Middle tier -- models in checkpoint (or wait) state
    Bottom tier -- models in saved state

Usage:
    python dashboard.py [--host HOST] [--port PORT] [--interval SECS]
"""
from __future__ import annotations

import argparse
import curses
import json
import threading
import time
import urllib.request
import urllib.error

import pynvml


# ---------------------------------------------------------------------------
# Colour pairs (initialised in _main)
# ---------------------------------------------------------------------------
C_TITLE = 1
C_GPU_HEADER = 2
C_RUNNING = 3
C_UP = 4
C_WAIT = 5
C_CHECKPOINT = 6
C_SAVED = 7
C_STATUS_OK = 8
C_STATUS_ERR = 9
C_BORDER = 10
C_SECTION = 11
C_SLEEP = 12
C_AGE_OLD = 13
C_ERROR = 14

def _init_colours():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(C_TITLE, curses.COLOR_WHITE, -1)
    curses.init_pair(C_GPU_HEADER, curses.COLOR_CYAN, -1)
    curses.init_pair(C_RUNNING, curses.COLOR_GREEN, -1)
    curses.init_pair(C_UP, curses.COLOR_CYAN, -1)
    curses.init_pair(C_WAIT, curses.COLOR_YELLOW, -1)
    curses.init_pair(C_CHECKPOINT, curses.COLOR_YELLOW, -1)
    curses.init_pair(C_SAVED, curses.COLOR_WHITE, -1)
    curses.init_pair(C_STATUS_OK, curses.COLOR_GREEN, -1)
    curses.init_pair(C_STATUS_ERR, curses.COLOR_RED, -1)
    curses.init_pair(C_BORDER, curses.COLOR_WHITE, -1)
    curses.init_pair(C_SECTION, curses.COLOR_WHITE, -1)
    curses.init_pair(C_SLEEP, curses.COLOR_MAGENTA, -1)
    # Maroon for stale ages: prefer xterm-256 maroon (88), fall back to red+dim.
    if curses.COLORS >= 256:
        curses.init_pair(C_AGE_OLD, 88, -1)
    else:
        curses.init_pair(C_AGE_OLD, curses.COLOR_RED, -1)
    # Maroon (same xterm-256 88 hue) for errored requests so they stand
    # out from completed-OK (dim white) and waiting/generating
    # (yellow/green).  Fallback to plain red on 8-colour terminals.
    if curses.COLORS >= 256:
        curses.init_pair(C_ERROR, 88, -1)
    else:
        curses.init_pair(C_ERROR, curses.COLOR_RED, -1)

# ---------------------------------------------------------------------------
# Local memory queries (CPU /proc/meminfo + NVML for GPU)
# ---------------------------------------------------------------------------

# Sentinel returned by NVML when per-process memory accounting is unavailable
# (e.g. driver lacks support, or the running user has no permission).
_NVML_NOT_AVAILABLE = 0xFFFFFFFFFFFFFFFF


def _query_cpu_memory() -> tuple[float, float]:
    """Return (used_gib, total_gib) from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", 0)
        used = total - avail
        return used / (1 << 20), total / (1 << 20)
    except Exception:
        return 0.0, 0.0


def _query_gpu_memory() -> dict[int, dict]:
    """Return {gpu_index: {"used_mib": int, "total_mib": int}} via NVML."""
    try:
        pynvml.nvmlInit()
        result: dict[int, dict] = {}
        for idx in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            m = pynvml.nvmlDeviceGetMemoryInfo(h)
            result[idx] = {
                "used_mib": int(m.used // (1 << 20)),
                "total_mib": int(m.total // (1 << 20)),
            }
        return result
    except Exception:
        return {}


def _query_process_gpu_memory() -> dict[int, int]:
    """Return {pid: used_mib} aggregated across GPUs via NVML."""
    try:
        pynvml.nvmlInit()
        result: dict[int, int] = {}
        for idx in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            except Exception:
                continue
            for p in procs:
                used = getattr(p, "usedGpuMemory", None)
                if used is None or used == _NVML_NOT_AVAILABLE:
                    continue
                result[p.pid] = result.get(p.pid, 0) + int(used // (1 << 20))
        return result
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Delta recording / replay helpers
# ---------------------------------------------------------------------------
#
# Each live session is streamed to a JSONL file as a baseline + patch log so
# the on-disk size is O(N + R) rather than O(N·R) for N polls / R requests.
# The orchestrator's /state payload is append-mostly for ``requests`` and
# update-in-place for ``models``, which makes a small hand-written diff
# scheme far simpler than a generic patch library.
#
# File format
# -----------
# Line 1:        {"t": 0.0, "state": <full snapshot>}
# Line 2..N:     {"t": <rel_s>, "patch": <patch>}  # only what changed
#
# Patch format
# ------------
# {
#   "elapsed_s": <float>,            # if changed
#   "gpu_pool":  [ids],              # if changed
#   "gpu_mem":   {...},              # emitted whole if any value changed
#   "cpu_mem":   {...},              #           "
#   "models":    {mid: {...changed keys...}, ...},  # only changed mids
#   "requests":  {"new":     [records...],  # req_ids never seen before
#                 "updated": [records...]}, # req_ids whose fields changed
# }


def _compute_patch(prev: dict, cur: dict) -> dict:
    """Return a minimal dict describing how *cur* differs from *prev*.

    See the module header for the exact patch shape.  An empty dict means
    nothing changed.
    """
    patch: dict = {}

    for key in ("elapsed_s", "gpu_pool", "image_cache", "gpu_ids",
                "slot_waiters"):
        if prev.get(key) != cur.get(key):
            patch[key] = cur.get(key)

    for key in ("gpu_mem", "cpu_mem", "slots"):
        if prev.get(key) != cur.get(key):
            patch[key] = cur.get(key)

    prev_models = prev.get("models") or {}
    cur_models = cur.get("models") or {}
    models_patch: dict = {}
    for mid, cur_info in cur_models.items():
        prev_info = prev_models.get(mid)
        if prev_info is None:
            models_patch[mid] = cur_info
            continue
        inner: dict = {}
        for k, v in cur_info.items():
            if prev_info.get(k) != v:
                inner[k] = v
        if inner:
            models_patch[mid] = inner
    if models_patch:
        patch["models"] = models_patch

    prev_reqs = {r["req_id"]: r for r in (prev.get("requests") or [])}
    cur_reqs_list = cur.get("requests") or []
    new_reqs: list = []
    updated_reqs: list = []
    for r in cur_reqs_list:
        rid = r["req_id"]
        if rid not in prev_reqs:
            new_reqs.append(r)
        elif prev_reqs[rid] != r:
            updated_reqs.append(r)
    if new_reqs or updated_reqs:
        reqs_patch: dict = {}
        if new_reqs:
            reqs_patch["new"] = new_reqs
        if updated_reqs:
            reqs_patch["updated"] = updated_reqs
        patch["requests"] = reqs_patch

    return patch


def _apply_patch(base: dict, patch: dict) -> dict:
    """Apply *patch* to *base* and return the resulting full-snapshot dict.

    *base* is not mutated; a shallow-new dict is returned with nested
    structures (``models``, ``requests``) rebuilt where needed.
    """
    out = dict(base)

    for key in ("elapsed_s", "gpu_pool", "image_cache", "gpu_ids",
                "gpu_mem", "cpu_mem", "slots", "slot_waiters"):
        if key in patch:
            out[key] = patch[key]

    if "models" in patch:
        merged = {mid: dict(info) for mid, info in (base.get("models") or {}).items()}
        for mid, inner in patch["models"].items():
            if mid not in merged:
                merged[mid] = dict(inner)
            else:
                merged[mid].update(inner)
        out["models"] = merged

    if "requests" in patch:
        req_patch = patch["requests"]
        req_by_id = {r["req_id"]: dict(r)
                     for r in (base.get("requests") or [])}
        for r in req_patch.get("updated", []):
            req_by_id[r["req_id"]] = dict(r)
        for r in req_patch.get("new", []):
            req_by_id[r["req_id"]] = dict(r)
        out["requests"] = sorted(req_by_id.values(), key=lambda r: r["req_id"])

    return out


# ---------------------------------------------------------------------------
# State poller (background thread)
# ---------------------------------------------------------------------------

class Poller:
    def __init__(self, url: str, interval: float, temp_path: str):
        self._url = url
        self._interval = interval
        self._state: dict | None = None
        self._connected = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last_poll_ts: float | None = None
        self._rate_ema: float | None = None
        # EMA of per-phase durations (seconds) within each poll.  Used to
        # surface which subsystem (http vs nvml) is dominating the actual
        # poll interval.
        self._phase_ema: dict[str, float] = {}
        # Set when the first consecutive failed poll is observed after a
        # prior success; cleared on the next successful poll.
        self._disconnected_since: float | None = None
        self._ever_connected: bool = False

        # Always-on delta recorder.  We open lazily on the first successful
        # poll so a failed start-up leaves no stray file on disk.
        self._temp_path: str = temp_path
        self._temp_file = None
        self._rec_t0: float | None = None
        self._rec_count: int = 0
        # Last full snapshot written to disk (baseline + accumulated patches)
        # kept as the diffing reference.
        self._prev_written: dict | None = None

    def start(self):
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._stop.set()
        if self._temp_file is not None:
            try:
                self._temp_file.close()
            except OSError:
                pass
            self._temp_file = None

    def snapshot(self) -> tuple[dict | None, bool]:
        with self._lock:
            return self._state, self._connected

    def poll_stats(self) -> tuple[float | None, float | None]:
        """Return ``(rate_hz, age_s)`` describing actual poll throughput.

        ``rate_hz`` is an EMA of successful-poll frequency; ``age_s`` is the
        wall time since the last successful poll completed.  Both are
        ``None`` until the first successful poll.
        """
        with self._lock:
            rate = self._rate_ema
            last = self._last_poll_ts
        age = (time.monotonic() - last) if last is not None else None
        return rate, age

    def phase_breakdown(self) -> dict[str, float]:
        """Return EMA of per-phase durations (seconds) for the latest poll.

        Keys: ``http``, ``nvml``, ``cpu``.  Empty until the first poll.
        """
        with self._lock:
            return dict(self._phase_ema)

    def disconnect_age(self) -> float | None:
        """Seconds since the first consecutive failed poll, or ``None``.

        Returns ``None`` if currently connected, or if no failed poll has
        been observed yet.
        """
        with self._lock:
            ds = self._disconnected_since
        return (time.monotonic() - ds) if ds is not None else None

    def ever_connected(self) -> bool:
        with self._lock:
            return self._ever_connected

    def recorded_count(self) -> int:
        """Number of lines written so far to the always-on temp recorder."""
        with self._lock:
            return self._rec_count

    def temp_path(self) -> str:
        return self._temp_path

    def finalize(self, save_path: str | None) -> int | None:
        """Close the temp recorder and either rename it to *save_path* or
        delete it.

        Returns the number of lines in the finalized file if saved, else
        ``None``.  Safe to call even if no polls have been recorded yet
        (in which case the temp file may never have been created).
        """
        with self._lock:
            f = self._temp_file
            self._temp_file = None
            count = self._rec_count
            path = self._temp_path
        if f is not None:
            try:
                f.close()
            except OSError:
                pass
        import os as _os
        if not _os.path.exists(path):
            return count if save_path else None
        if save_path is None:
            try:
                _os.remove(path)
            except OSError:
                pass
            return None
        dest = (save_path if _os.path.isabs(save_path)
                else _os.path.abspath(save_path))
        # shutil.move falls back to copy+unlink across filesystems, which
        # os.replace cannot do (raises EXDEV when /tmp is a separate mount).
        import shutil as _shutil
        _shutil.move(path, dest)
        return count

    def _loop(self):
        while not self._stop.is_set():
            try:
                t_phase = time.monotonic()
                req = urllib.request.Request(self._url, method="GET")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    data = json.loads(resp.read())
                t_after_http = time.monotonic()
                gpu_mem = _query_gpu_memory()
                pid_mem = _query_process_gpu_memory()
                t_after_nvml = time.monotonic()
                cpu_used, cpu_total = _query_cpu_memory()
                t_after_cpu = time.monotonic()
                phase_durations = {
                    "http": t_after_http - t_phase,
                    "nvml": t_after_nvml - t_after_http,
                    "cpu": t_after_cpu - t_after_nvml,
                }
                data["gpu_mem"] = gpu_mem
                data["cpu_mem"] = {"used_gib": cpu_used, "total_gib": cpu_total}
                for _mid, info in data.get("models", {}).items():
                    pid = info.get("pid")
                    info["gpu_mem_mib"] = pid_mem.get(pid, 0) if pid else 0

                # Always-on delta record to the temp file.  Write a full
                # baseline on the first successful poll; deltas thereafter.
                now_mono = time.monotonic()
                if self._temp_file is None:
                    self._temp_file = open(self._temp_path, "w")
                    self._rec_t0 = now_mono
                    t_rel = 0.0
                    self._temp_file.write(
                        json.dumps({"t": round(t_rel, 3), "state": data}) + "\n"
                    )
                    self._temp_file.flush()
                    self._prev_written = data
                    self._rec_count = 1
                else:
                    t_rel = now_mono - (self._rec_t0 or now_mono)
                    patch = _compute_patch(self._prev_written or {}, data)
                    if patch:
                        self._temp_file.write(
                            json.dumps({"t": round(t_rel, 3), "patch": patch})
                            + "\n"
                        )
                        self._temp_file.flush()
                        self._prev_written = data
                        self._rec_count += 1

                now = now_mono
                with self._lock:
                    self._state = data
                    self._connected = True
                    self._ever_connected = True
                    self._disconnected_since = None
                    if self._last_poll_ts is not None:
                        dt = now - self._last_poll_ts
                        if dt > 0:
                            inst_rate = 1.0 / dt
                            alpha = 0.3
                            self._rate_ema = (
                                inst_rate if self._rate_ema is None
                                else alpha * inst_rate + (1 - alpha) * self._rate_ema
                            )
                    self._last_poll_ts = now
                    alpha_p = 0.3
                    for k, v in phase_durations.items():
                        prev = self._phase_ema.get(k)
                        self._phase_ema[k] = (
                            v if prev is None else alpha_p * v + (1 - alpha_p) * prev
                        )
            except Exception:
                with self._lock:
                    self._connected = False
                    if self._disconnected_since is None:
                        self._disconnected_since = time.monotonic()
            self._stop.wait(self._interval)


class ReplayPoller:
    """Replays a recorded JSONL file, mimicking the Poller interface.

    The file is expected in the delta format produced by ``Poller``: the
    first line carries a full ``state``, subsequent lines carry ``patch``
    values applied sequentially.  We eagerly reconstruct the full snapshot
    for each tick so ``snapshot()`` is O(1) at render time.
    """

    def __init__(self, path: str, speed: float = 1.0):
        self._entries: list[tuple[float, dict]] = []
        current: dict | None = None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "state" in rec:
                    current = rec["state"]
                elif "patch" in rec and current is not None:
                    current = _apply_patch(current, rec["patch"])
                else:
                    continue
                self._entries.append((rec["t"], current))
        self._idx = 0
        self._wall_t0: float | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # Playback speed multiplier: 2.0 => play twice as fast, etc.
        self._speed = max(1e-6, float(speed))

    def start(self):
        self._wall_t0 = time.monotonic()
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._stop.set()

    def snapshot(self) -> tuple[dict | None, bool]:
        with self._lock:
            if not self._entries:
                return None, False
            return self._entries[self._idx][1], True

    def poll_stats(self) -> tuple[float | None, float | None]:
        return None, None

    def phase_breakdown(self) -> dict[str, float]:
        return {}

    def disconnect_age(self) -> float | None:
        return None

    def ever_connected(self) -> bool:
        return True

    def recorded_count(self) -> int:
        return 0

    def temp_path(self) -> str:
        return ""

    def finalize(self, save_path: str | None) -> int | None:
        return None

    def is_done(self) -> bool:
        """True once the replay has reached (and stopped at) the final tick."""
        with self._lock:
            return bool(self._entries) and self._idx >= len(self._entries) - 1

    def current_interval(self) -> float | None:
        """Wall-clock gap between the current tick and its predecessor.

        Returns ``t[idx] - t[idx-1]`` in *recorded* seconds divided by
        the playback speed, i.e. the effective wall-clock interval at
        which this particular frame is being shown.  ``None`` when the
        replay is still on its very first tick.
        """
        with self._lock:
            if self._idx < 1 or self._idx >= len(self._entries):
                return None
            dt = self._entries[self._idx][0] - self._entries[self._idx - 1][0]
        if dt <= 0:
            return None
        return dt / self._speed

    def _loop(self):
        while not self._stop.is_set():
            if self._wall_t0 is not None and self._entries:
                elapsed = (time.monotonic() - self._wall_t0) * self._speed
                with self._lock:
                    while (self._idx < len(self._entries) - 1
                           and self._entries[self._idx + 1][0] <= elapsed):
                        self._idx += 1
            self._stop.wait(0.05)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_SPINNER = "|/-\\"


def _safe_addstr(win, y: int, x: int, text: str, attr=0, max_x: int | None = None):
    """Write text clipped to window bounds, avoiding the curses bottom-right corner bug."""
    h, w = win.getmaxyx()
    if max_x is not None:
        w = min(w, max_x)
    if y < 0 or y >= h or x >= w:
        return
    avail = w - x
    if avail <= 0:
        return
    text = text[:avail]
    try:
        win.addstr(y, x, text, attr)
    except curses.error:
        pass


def _hline(win, y: int, x: int, width: int, char="─", attr=0):
    for i in range(width):
        _safe_addstr(win, y, x + i, char, attr)


def _vline(win, y: int, x: int, height: int, char="│", attr=0):
    for i in range(height):
        _safe_addstr(win, y + i, x, char, attr)


def _level_suffix(slot_level: int | None) -> str:
    """``, L<n>`` when a slot of level ``n`` is held; empty otherwise."""
    return f", L{slot_level}" if slot_level else ""


def _pinned_str(pinned_cpu_bytes: int, slot_level: int | None = None) -> str:
    if pinned_cpu_bytes <= 0:
        return ""
    gib = pinned_cpu_bytes / (1 << 30)
    return f" ({gib:.1f}G{_level_suffix(slot_level)})"


def _stars_for_level(level: int | None, max_level: int) -> str:
    """Render *level* as ``*``-stars, scaled so the deepest level shown
    (``max_level``, the smallest allocation) gets a single star and each
    step shallower doubles the count.

    With *max_level* = 3:  L1 -> ``****``, L2 -> ``**``, L3 -> ``*``.
    With *max_level* = 2:  L1 -> ``**``,   L2 -> ``*``.

    Returns the empty string when the level is unknown or when there's
    nothing to render against (no max_level).
    """
    if not level or max_level <= 0:
        return ""
    depth = max_level - int(level)
    if depth < 0:
        depth = 0
    return "*" * (1 << depth)


def _pinned_stars_chunks(info: dict, primary_attr: int, max_level: int
                         ) -> list[tuple[str, int]]:
    """Build ``[(text, attr), ...]`` chunks rendering ``" (X.XG, ****)"``
    for the image-cache tier.

    *primary_attr* styles the size text and the surrounding ``( )`` /
    ``,`` punctuation.  The stars themselves render dim (matching free
    cells in the slot bar) since the image-cache tier shows the
    model's intrinsic level rather than a live slot allocation.  The
    slot's GPU id is *not* shown here -- the image cache is a
    location-agnostic catalogue of resident weights, so cluttering
    each row with a per-slot locator would muddy that view; the GPU
    and CPU tiers above already display the locator on the rows where
    it's actionable.

    Returns ``[]`` when there's nothing to render (no pinned bytes).
    """
    pinned_cpu_bytes = info.get("pinned_cpu_bytes", 0) or 0
    if pinned_cpu_bytes <= 0:
        return []
    gib = pinned_cpu_bytes / (1 << 30)
    stars = _stars_for_level(info.get("level"), max_level)
    dim_gray = curses.color_pair(C_TITLE) | curses.A_DIM
    if not stars:
        return [(f" ({gib:.1f}G)", primary_attr)]
    return [
        (f" ({gib:.1f}G, ", primary_attr),
        (stars, dim_gray),
        (")", primary_attr),
    ]


def _gpu_mem_str(info: dict) -> str:
    """Format GPU memory for a model: prefer actual gpu_mem, fall back to pinned.

    Appends ``, L<level>`` when the model currently holds a slot.
    """
    slot_level = info.get("slot_level")
    gpu_mib = info.get("gpu_mem_mib", 0)
    if gpu_mib > 0:
        return f" ({gpu_mib / 1024:.1f}G{_level_suffix(slot_level)})"
    return _pinned_str(info.get("pinned_cpu_bytes", 0), slot_level)


def _mem_chunks(info: dict, primary_attr: int, max_level: int,
                *, prefer_gpu_mem: bool = False,
                slot_gpu: int | None = None
                ) -> list[tuple[str, int]]:
    """Build ``" (X.XG, **)"`` suffix chunks where the stars are sized
    by the model's intrinsic ``level`` (scaled against *max_level*) and
    coloured by allocation status.

    When the model holds a slot (``slot_level`` is set), the stars
    render in the slot-chart's ``ALLOC`` style (running-green, bold);
    otherwise they render dim, matching free cells in the slot bar.

    *prefer_gpu_mem* picks ``gpu_mem_mib`` over ``pinned_cpu_bytes`` when
    the model still has a GPU assigned -- the GPU-tier convention,
    keyed off ``info["gpu"]`` rather than the live ``gpu_mem_mib``
    reading so that transient 0-readings (NVML poll lag, pre-restore
    slotless sleep, mid-transition windows) don't flip the display
    into the much-larger pinned-bytes value.  The CPU tier should
    leave the flag at the default to mirror :func:`_pinned_str`.

    *slot_gpu* is the id of the GPU whose buddy-allocator currently
    holds this model's slot (resolved by the caller from the
    authoritative ``slots`` snapshot rather than ``info["gpu"]``,
    which is the model's *current* GPU assignment and can diverge
    mid-transition).  When provided, it's appended directly after the
    stars (e.g. ``****2``) in dim gray so the suffix reads as a
    single ``stars+gpu`` token while remaining distinguishable from
    the allocation glyph.

    Returns ``[]`` when there's nothing to show (no GPU, no pinned
    bytes), matching :func:`_gpu_mem_str` / :func:`_pinned_str`.
    """
    size_str: str | None = None
    if prefer_gpu_mem and info.get("gpu") is not None:
        gpu_mib = info.get("gpu_mem_mib", 0)
        size_str = f"{gpu_mib / 1024:.1f}G"
    if size_str is None:
        pinned = info.get("pinned_cpu_bytes", 0)
        if pinned > 0:
            size_str = f"{pinned / (1 << 30):.1f}G"
    if size_str is None:
        return []

    stars = _stars_for_level(info.get("level"), max_level)
    is_alloc = info.get("slot_level") is not None
    gpu_str = str(slot_gpu) if slot_gpu is not None else ""
    dim_gray = curses.color_pair(C_TITLE) | curses.A_DIM
    if not stars:
        if gpu_str:
            return [
                (f" ({size_str}, ", primary_attr),
                (gpu_str, dim_gray),
                (")", primary_attr),
            ]
        return [(f" ({size_str})", primary_attr)]

    if is_alloc:
        star_attr = curses.color_pair(C_RUNNING) | curses.A_BOLD
    else:
        star_attr = dim_gray
    chunks: list[tuple[str, int]] = [
        (f" ({size_str}, ", primary_attr),
        (stars, star_attr),
    ]
    if gpu_str:
        chunks.append((gpu_str, dim_gray))
    chunks.append((")", primary_attr))
    return chunks


def _fmt_duration_compact(seconds: float) -> str:
    """Format a duration as a compact string like ``12s``, ``3m24s``, ``1h05m``."""
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    m = total // 60
    s = total % 60
    if m < 60:
        return f"{m}m{s:02d}s"
    h = m // 60
    m = m % 60
    return f"{h}h{m:02d}m"


def _state_age_chunk(info: dict, elapsed_s: float | None,
                     default_attr: int,
                     is_waiting_target: bool = False,
                     waiter_submit_rel_s: float | None = None,
                     ) -> tuple[str, int]:
    """Return ``(text, attr)`` for the age suffix; maroon when age > 10 s.

    Only shown for "active" models -- ``wait``, ``running``, any
    ``paused`` model, or any model that is currently the target of a
    request in ``waiting`` state (``is_waiting_target=True``).  All
    other states (``up``, ``init``, ``checkpoint``, ``sleep``,
    ``saved``, ...) suppress the age string entirely.

    When *is_waiting_target* is True and *waiter_submit_rel_s* is
    provided, the anchor used for the age is
    ``max(state_since_rel_s, waiter_submit_rel_s)`` -- i.e. whichever
    happened more recently.  This keeps the chip aligned with the
    request timeline: when a generate lands on a model that has been
    parked in (e.g.) ``sleep`` for 30 minutes, the chip resets to ~0
    instead of continuing to show the model's pre-existing tenure.
    Once the model transitions, ``state_since`` overtakes the waiter
    anchor and the chip tracks the new state's age normally.
    """
    state = info.get("state")
    if (state not in ("wait", "running")
            and not info.get("paused")
            and not is_waiting_target):
        return ("", default_attr)
    if elapsed_s is None:
        return ("", default_attr)
    since = info.get("state_since_rel_s")
    if is_waiting_target and waiter_submit_rel_s is not None:
        since = max(since or 0.0, float(waiter_submit_rel_s))
    if since is None:
        return ("", default_attr)
    age = elapsed_s - since
    if age < 1:
        return ("", default_attr)
    text = f" {_fmt_duration_compact(age)}"
    if age > 10:
        return (text, curses.color_pair(C_AGE_OLD))
    return (text, default_attr)


def _fmt_rel_t0_hms(seconds: float) -> str:
    """Elapsed seconds since orchestrator ``t=0`` as ``h:mm:ss`` (minutes and seconds zero-padded)."""
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}:{m:02d}:{s:02d}"


def _fmt_request_time_log(req: dict) -> str:
    """Submission time since orchestrator ``t=0`` as ``h:mm:ss``."""
    rs = req.get("submit_rel_s")
    if rs is None:
        return ""
    return _fmt_rel_t0_hms(float(rs))


_START_STATE_SHORT = {
    "saved": "saved",
    "checkpoint": "ckpt",
    "sleep": "sleep",
    "up": "up",
    "running": "running",
    "wait": "wait",
}


def _fmt_model_with_start_state(model_id: str, start_state: str | None) -> str:
    """``model_id (ckpt)``-style label using model ladder state at generate submit."""
    if not start_state:
        return model_id
    short = _START_STATE_SHORT.get(start_state, start_state)
    return f"{model_id} ({short})"


def _build_state_timeline(req: dict,
                          now_rel_s: float | None
                          ) -> list[tuple[str, bool]]:
    """Derive ``[(segment_text, is_current), ...]`` for a request row.

    ``is_current`` flags the last segment (bright); earlier segments
    are all dim. A few states are suppressed because they'd otherwise
    be redundant or uninteresting:

    * ``running`` is dropped — the ``gen N.Ns`` column on the right
      already carries that duration.
    * A leading ``up`` is dropped — if the request found the model
      already up, there's no interesting ladder to show.

    Falls back to the legacy ``start_state`` label if the server
    didn't include a ``state_log`` (e.g. older recording).
    """
    log = req.get("state_log") or []
    if not log:
        ss = req.get("start_state")
        if ss and ss not in ("running", "up"):
            return [(_START_STATE_SHORT.get(ss, ss), True)]
        return []

    # Right edge of the last log entry: request end if done, else "now".
    done_rel = req.get("done_rel_s")
    if done_rel is not None:
        end_t = float(done_rel)
    elif now_rel_s is not None:
        end_t = float(now_rel_s)
    else:
        end_t = float(log[-1][0])

    # First compute every segment's duration using the raw log so that
    # a pre-``running`` state (usually ``up``) correctly reports the
    # time it spent before generation started, even though we don't
    # render the ``running`` segment itself.
    raw: list[tuple[str, float]] = []
    n = len(log)
    for i, entry in enumerate(log):
        t = float(entry[0])
        s = entry[1]
        next_t = float(log[i + 1][0]) if i + 1 < n else end_t
        raw.append((s, max(0.0, next_t - t)))

    # Hide states that don't add signal to the headline timeline:
    #   running   -- redundant with the ``gen N.Ns`` column
    #   sleep     -- transient idle state, noise in the ladder
    #   checkpoint-- intermediate loading step, mostly noise too
    # Their durations are still reported in the per-request suffix
    # (:func:`_build_state_durations`).
    _HIDDEN = {"running", "sleep", "checkpoint"}
    visible = [(s, dur) for (s, dur) in raw if s not in _HIDDEN]
    # Drop a leading ``up`` — if the request started with the model
    # already up, the timeline has nothing useful to show on the left.
    if visible and visible[0][0] == "up":
        visible = visible[1:]
    if not visible:
        return []
    segments: list[tuple[str, bool]] = []
    last = len(visible) - 1
    for i, (s, _dur) in enumerate(visible):
        short = _START_STATE_SHORT.get(s, s)
        segments.append((short, i == last))
    return segments


def _build_state_durations(req: dict,
                           now_rel_s: float | None,
                           *,
                           paused_since_rel_s: float | None = None,
                           ) -> list[tuple[str, bool]]:
    """Return ``[(segment_text, is_active), ...]`` for the right-aligned
    durations tail.

    Segments are labelled by the raw orchestrator state name (only
    ``checkpoint`` is abbreviated to ``ckpt`` via
    :data:`_START_STATE_SHORT`), so the bracket reads as a step-by-step
    log of the climb -- e.g. a cold rebound generate shows
    ``[saved Xs, ckpt Ys, sleep Zs, wait Ws, ckpt Vs, sleep Us]``.

    The *last visible* segment is flagged ``is_active=True`` while the
    request is still climbing (before the model reaches ``running``).
    Once generation has begun, or the request is done, every segment
    is ``is_active=False`` -- the live signal is carried by the
    ``gen N.Ns`` column on the left.

    ``running`` is filtered (redundant with the ``gen`` column) and a
    leading ``up`` is dropped.  A pseudo-entry for ``start_state`` is
    prepended when the recorded ``state_log`` doesn't already cover
    the window from ``t_submit`` to the first transition, so that
    e.g. the long CRIU-load phase of a cold start appears as the
    ``saved Xs`` segment on the left of the bracket.

    For *piggy-back* requests the bracket collapses to a dim
    ``[wait #N]`` where ``#N`` is the lead request.  Two flavours
    are merged into the same field by ``state_server``, both
    classified once and persisted for R's lifetime so the bracket
    keeps reading ``wait #L+1`` even after the lead has started /
    finished:

    * climb-piggyback -- submitted while an earlier-submitted request
      for the same model was still climbing toward ``running``, so
      the climb work was done by the lead.

    * pause-piggyback -- submitted while the model was already paused
      (orchestrator's one-shot ``paused_at_submit`` marker on the
      request record) with an earlier earlier-submit request L on
      the same model that was alive at R's submit time.  R cannot
      make progress on its own until the next ``resume`` re-prefills
      L and unfreezes the engine (which also drives R), and the
      classification persists past resume since both inputs
      (``paused_at_submit``, L's submit/done timestamps) are
      immutable.

    In both cases the ``wait X.Ys`` / ``paused X.Ys`` metric on the
    left already shows how long this request sat waiting, and the
    dim styling makes clear no real work is happening here.
    ``req["piggyback"]`` is ``None`` when the request drove its own
    climb (and was not parked behind a pause at submit), otherwise
    the lead ``req_id``.
    """
    log = list(req.get("state_log") or [])
    done_rel = req.get("done_rel_s")
    is_in_flight = done_rel is None
    # ``up`` and ``running`` are both settled states (model is loaded,
    # generation either active or paused) -- neither counts as "still
    # climbing", so the trailing visible segment shouldn't be
    # highlighted bright once the model has reached either.
    is_climbing = (is_in_flight
                   and (not log or log[-1][1] not in ("running", "up")))

    lead = req.get("piggyback")
    if lead is not None:
        return [(f"wait #{int(lead) + 1}", False)]

    submit_rel = req.get("submit_rel_s")
    start_state = req.get("start_state")
    if (start_state and submit_rel is not None
            and start_state not in ("running", "up")
            and (not log or log[0][1] != start_state)):
        log = [(float(submit_rel), start_state)] + log

    if not log:
        return []

    if done_rel is not None:
        end_t = float(done_rel)
    elif now_rel_s is not None:
        end_t = float(now_rel_s)
    else:
        end_t = float(log[-1][0])

    raw: list[tuple[str, float, float]] = []
    n = len(log)
    for i, entry in enumerate(log):
        t = float(entry[0])
        s = entry[1]
        next_t = float(log[i + 1][0]) if i + 1 < n else end_t
        raw.append((s, max(0.0, next_t - t), t))

    # ``running`` is redundant with the ``gen N.Ns`` column on the left,
    # and ``up`` is the brief sub-state between ``sleep -> running``
    # (typically renders as a noisy ``up 0.0s`` segment immediately
    # before the running flip, or as a tail entry after running ends);
    # it carries no useful timing signal so we drop it entirely.
    visible = [(s, dur, t) for (s, dur, t) in raw
               if s not in ("running", "up")]
    if not visible:
        return []

    # Drop any ``checkpoint`` segment that is immediately followed by
    # ``wait``.  The orchestrator's only path into ``wait`` is through
    # ``_step_up(checkpoint -> sleep)``'s ``on_block`` callback, which
    # fires after the previous step has already announced
    # ``checkpoint``.  That intervening ``checkpoint`` entry is always a
    # tombstone (microseconds wide) -- the model never observably
    # occupied it -- so suppressing it cleans up brackets like
    # ``[saved Xs, ckpt 0.0s, wait Ys, ...]`` without losing real time.
    # If anyone ever adds a second ``_set_state(model_id, "wait")`` call
    # site preceded by a genuinely-occupied ``checkpoint``, this rule
    # would silently swallow that interval -- revisit then.
    filtered: list[tuple[str, float, float]] = []
    n_vis = len(visible)
    for i, (s, dur, t) in enumerate(visible):
        if s == "checkpoint" and i + 1 < n_vis and visible[i + 1][0] == "wait":
            continue
        filtered.append((s, dur, t))
    visible = filtered
    if not visible:
        return []

    last_idx = len(visible) - 1
    segments: list[tuple[str, bool]] = []
    for i, (s, dur, t) in enumerate(visible):
        label = _START_STATE_SHORT.get(s, s)
        is_active = is_climbing and (i == last_idx)
        # Segments that started at or after the model's pause time
        # would otherwise tick a meaningless ``end_t -> now`` duration
        # (waiting requests stuck behind a paused model) -- render
        # the label without the duration so the bracket freezes
        # visibly at pause time.  Segments that started before pause
        # are settled and still show their durations.
        suppress = (paused_since_rel_s is not None
                    and t >= paused_since_rel_s)
        if suppress:
            segments.append((label, is_active))
        else:
            segments.append((f"{label} {dur:.1f}s", is_active))
    return segments


def _write_chunks(win, y: int, x: int,
                  chunks: list[tuple[str, int]],
                  max_x: int) -> int:
    """Write ``(text, attr)`` chunks left-to-right, clipping at ``max_x``.

    Returns the final x position.
    """
    for text, attr in chunks:
        if x >= max_x:
            break
        remaining = max_x - x
        piece = text[:remaining]
        if piece:
            _safe_addstr(win, y, x, piece, attr, max_x=max_x)
        x += len(text)
    return x


def _slot_sort_key(info: dict) -> tuple[int, float]:
    """Sort key matching the left-to-right slot-bar order.

    Returns ``(bucket, secondary)`` so slotted models land before slotless
    ones (e.g. slotless-sleep residents).  Slotted models are ordered by
    the leaf's left position in the buddy tree -- a leaf at level *L* with
    index *I* covers the normalized range
    ``[I << (MAX-L), (I+1) << (MAX-L))``, so sorting by ``I << (MAX-L)``
    is equivalent to a DFS pre-order of the buddy tree.

    Slotless residents are ordered most-recently-transitioned first /
    oldest at the bottom: ``-state_since_rel_s`` puts larger (more recent)
    timestamps earlier.  Missing timestamps sink past every dated entry.
    """
    level = info.get("slot_level")
    index = info.get("slot_index")
    if level is None or index is None:
        since = info.get("state_since_rel_s")
        return (1, -since if since is not None else float("inf"))
    MAX = 32  # well beyond any practical buddy depth
    return (0, float(int(index) << max(0, MAX - int(level))))


def _slot_bar_chunks(leaves: list[dict],
                     max_level: int = 0
                     ) -> tuple[list[tuple[str, int]], int]:
    """Return ``(chunks, total_width)`` rendering a buddy-allocator bar.

    *leaves* is the per-GPU list emitted by ``_snapshot_slots`` in
    ``state_server`` -- a left-to-right DFS over the implicit buddy tree
    where each entry covers ``1 / 2**(level-1)`` of the GPU.

    *max_level* is the deepest level designated anywhere in the image
    cache (i.e. the smallest allocation any model in the system would
    take).  The bar is sized to ``2 ** (max_level - 1)`` cells, padded
    with ``.`` when the GPU's actual buddy tree is shallower, so every
    GPU's bar shares a common scale.  When *max_level* is 0 (e.g. old
    recordings), falls back to the deepest level present in *leaves*.

    Cells within a leaf are packed with no spacing (``...``); leaves
    are separated by ``|``.  Free cells render as ``.``; allocated
    cells as ``*``.

    Empty when no leaves are reported (recordings that predate slot
    publishing).
    """
    if not leaves:
        return [], 0
    leaf_max = max(int(leaf.get("level", 1)) for leaf in leaves)
    deepest = max(int(max_level), leaf_max, 1)
    border_attr = curses.color_pair(C_BORDER) | curses.A_DIM
    alloc_attr = curses.color_pair(C_RUNNING) | curses.A_BOLD
    free_attr = curses.color_pair(C_TITLE) | curses.A_DIM
    chunks: list[tuple[str, int]] = [("[", border_attr)]
    n_leaves = len(leaves)
    for i, leaf in enumerate(leaves):
        cells = 1 << (deepest - int(leaf.get("level", 1)))
        glyph = "*" if leaf.get("alloc") else "."
        attr = alloc_attr if leaf.get("alloc") else free_attr
        chunks.append((glyph * cells, attr))
        if i < n_leaves - 1:
            chunks.append(("|", border_attr))
    chunks.append(("]", border_attr))
    width = 2 + (1 << (deepest - 1)) + max(0, n_leaves - 1)
    return chunks, width


def _gpu_mem_bar_chunks(models_on_gpu: list[tuple[str, dict]],
                        used_mib: float, total_mib: float,
                        *, width: int = 10
                        ) -> tuple[list[tuple[str, int]], int]:
    """Return ``(chunks, total_width)`` for a per-state GPU memory bar.

    Splits the used portion of the GPU's HBM into colored segments by
    the residing model's state -- ``running`` (green), ``up`` (cyan),
    and a ``sleep+other`` bucket (blue) that lumps sleeping vLLM models
    together with any remaining tracked-process bytes that don't fit
    those buckets (``init`` / ``wait`` transients, plus non-vLLM CUDA
    processes that show up in ``used_mib`` but not in ``models_on_gpu``).
    Free memory is rendered as ``░`` in dim white.

    Cell widths are assigned via cumulative integer rounding so the
    bar always sums to exactly *width* cells regardless of how the
    real-valued shares fall.  Returns ``([], 0)`` when *total_mib* is
    non-positive (no signal to plot).
    """
    if total_mib <= 0 or width <= 0:
        return [], 0
    # Per-model HBM accounting.  Each ``running`` / ``up`` model emits
    # two adjacent segments in its hue: weights (BOLD; equal to
    # ``pinned_cpu_bytes`` since vLLM mirrors weights between pinned
    # CPU and HBM) and kv_cache (DIM; the rest -- activations + KV
    # cache).  Models are ordered to match the slot-bar / row order so
    # the visual matches the per-GPU column underneath.  ``sleep`` and
    # untracked HBM tenants share a single bucket since they cannot be
    # split into weights + kv.
    state_colors = {"running": C_RUNNING, "up": C_UP}
    per_model_segments: list[tuple[float, str, int, int]] = []
    # ``segment_model[i]`` = which logical model produced segment i (or
    # -1 for non-model segments like sleep / other / free).  Used at
    # render time to drop a thin ``┊`` separator between adjacent
    # per-model segments belonging to different models.  The
    # separator does not consume a cell from ``width``.
    segment_model: list[int] = []
    sleep_mib = 0.0
    tracked_mib = 0.0
    sortable = sorted(models_on_gpu, key=lambda x: _slot_sort_key(x[1]))
    for model_idx, (_mid, info) in enumerate(sortable):
        m = float(info.get("gpu_mem_mib") or 0)
        if m <= 0:
            continue
        s = info.get("state")
        if s in state_colors:
            color = state_colors[s]
            w = min(m, float(info.get("pinned_cpu_bytes") or 0) / (1 << 20))
            kv = max(0.0, m - w)
            if w > 0:
                per_model_segments.append((w, "\u2588", color, curses.A_BOLD))
                segment_model.append(model_idx)
            if kv > 0:
                per_model_segments.append((kv, "\u2588", color, curses.A_DIM))
                segment_model.append(model_idx)
        elif s == "sleep":
            sleep_mib += m
        tracked_mib += m
    other_mib = max(0.0, float(used_mib) - tracked_mib)
    free_mib = max(0.0, float(total_mib) - float(used_mib))
    # (mib, glyph, color, attr).  ``sleep`` and untracked HBM tenants
    # (non-vLLM CUDA processes, init/wait transients, accounting
    # overhead) share one magenta bucket: they're both "evictable but
    # not currently a per-model citizen", and the combined view
    # avoids a redundant divider between two same-purpose segments.
    n_per_model = len(per_model_segments)
    segments = [
        *per_model_segments,
        (sleep_mib + other_mib, "\u2588", C_SLEEP, curses.A_BOLD),
        (free_mib,              "\u2591", C_TITLE, curses.A_DIM),
    ]
    # Largest-remainder cell allocation so each segment gets a fair
    # share, with a min-1 guarantee for every non-zero per-model
    # weights segment.  Without the guarantee, a small model whose
    # weights round to <0.5 cells loses its bright cell and only the
    # dim kv tail shows -- visually misleading (looks like "all kv").
    n = len(segments)
    ideals = [(seg[0] / total_mib * width) if total_mib > 0 else 0.0
              for seg in segments]
    cells = [int(v) for v in ideals]
    leftover = width - sum(cells)
    order = sorted(range(n), key=lambda i: -(ideals[i] - cells[i]))
    for k in range(max(0, leftover)):
        cells[order[k % n]] += 1
    # Force min-1 only for per-model weights (bold) -- those are the
    # primary signal of "this model is here, even if tiny".  Sleep
    # (magenta), other, and free are NOT protected: they appear /
    # disappear naturally based on proportional rounding, so a sleeper
    # only shows magenta when its footprint is actually worth a cell.
    # Donor preference (most expendable first):
    #   1. magenta sleep -- but only if it has >1 cells, so we never
    #      pop a 1-cell sleep just to satisfy a tiny weights segment.
    #   2. dim cells (other models' kv tails, ``other``, or free).
    #   3. bold weights of other models (last resort; same >1 guard).
    protected_idxs: set[int] = {
        i for i in range(n_per_model)
        if segments[i][3] == curses.A_BOLD and segments[i][0] > 0
    }

    def _donor_rank(j: int) -> tuple[int, int]:
        _mib, _g, color, mod = segments[j]
        if color == C_SLEEP:
            tier = 0
        elif mod == curses.A_DIM:
            tier = 1
        else:
            tier = 2
        return (tier, -cells[j])

    def _can_donate(j: int) -> bool:
        if cells[j] <= 0:
            return False
        # Protect bold weights from being drained to 0 by another
        # bold's promotion.  Also avoid eviscerating a 1-cell sleep
        # just to satisfy a tiny weights segment -- that would feel
        # worse than the tiny model losing its forced cell.
        _mib, _g, color, _mod = segments[j]
        if (j in protected_idxs or color == C_SLEEP) and cells[j] <= 1:
            return False
        return True

    for i in sorted(protected_idxs):
        if cells[i] > 0:
            continue
        donors = sorted(
            (j for j in range(n) if j != i and _can_donate(j)),
            key=_donor_rank,
        )
        if donors:
            cells[donors[0]] -= 1
            cells[i] = 1
    # Group id per segment: each model is one group; the magenta
    # sleep+other bucket and free share one trailing group (their
    # color/glyph difference is already distinguishing).  A ``┊``
    # separator is inserted between consecutive rendered segments
    # belonging to different groups (model-model, model-tail).
    # Separators don't consume cells from ``width``; they extend the
    # bar's horizontal extent.
    group_id: list[int] = []
    for i in range(n):
        if i < n_per_model:
            group_id.append(segment_model[i])
        else:
            group_id.append(-1)  # sleep + free share one tail group
    border_attr = curses.color_pair(C_BORDER) | curses.A_DIM
    chunks: list[tuple[str, int]] = [("[", border_attr)]
    sep_chars = 0
    last_group: int | None = None
    for i, ((mib, glyph, color, mod), c) in enumerate(zip(segments, cells)):
        if c <= 0:
            continue
        cur_group = group_id[i]
        if last_group is not None and cur_group != last_group:
            chunks.append(("\u250a", border_attr))  # ┊
            sep_chars += 1
        chunks.append((glyph * c, curses.color_pair(color) | mod))
        last_group = cur_group
    chunks.append(("]", border_attr))
    return chunks, width + 2 + sep_chars


def _cpu_mem_bar_chunks(used_gib: float, total_gib: float, weights_gib: float,
                        *, width: int = 20,
                        ) -> tuple[list[tuple[str, int]], int]:
    """Return ``(chunks, total_width)`` for a CPU memory bar split into
    weights / other / free.

    *weights_gib* is the portion of *used_gib* attributable to pinned
    model weights (sum of ``pinned_cpu_bytes`` for non-``saved`` models).
    The remaining ``used_gib - weights_gib`` is rendered as ``other``.
    Cells are assigned with cumulative integer rounding so the bar
    always sums to exactly *width* cells.  Returns ``([], 0)`` when
    *total_gib* is non-positive.
    """
    if total_gib <= 0 or width <= 0:
        return [], 0
    weights = max(0.0, min(float(weights_gib), float(used_gib)))
    other = max(0.0, float(used_gib) - weights)
    free = max(0.0, float(total_gib) - float(used_gib))
    segments = [
        (weights, "\u2588", C_CHECKPOINT, curses.A_DIM),
        (other,   "\u2588", C_TITLE,      curses.A_DIM),
        (free,    "\u2591", C_TITLE,      curses.A_DIM),
    ]
    border_attr = curses.color_pair(C_BORDER) | curses.A_DIM
    chunks: list[tuple[str, int]] = [("[", border_attr)]
    cumulative = 0.0
    cumulative_cells = 0
    last_idx = len(segments) - 1
    for i, (val, glyph, color, mod) in enumerate(segments):
        cumulative += val
        if i == last_idx:
            target = width
        else:
            target = int(round(cumulative / total_gib * width))
            target = max(0, min(target, width))
        cells = target - cumulative_cells
        if cells <= 0:
            continue
        cumulative_cells = target
        chunks.append((glyph * cells, curses.color_pair(color) | mod))
    chunks.append(("]", border_attr))
    return chunks, width + 2


def _mem_bar_chunks(used: float, total: float, *, width: int = 20,
                    used_color: int = C_UP,
                    used_mod: int | None = None,
                    ) -> tuple[list[tuple[str, int]], int]:
    """Return ``(chunks, total_width)`` for a used/free memory bar.

    Renders ``[████░░░░...]`` with ``█`` (U+2588) for the used portion
    and ``░`` (U+2591) for free, framed by ``[`` ``]`` in the same dim
    border style as :func:`_slot_bar_chunks`.  *used_color* is the
    curses color-pair index applied to the filled cells; *used_mod* is
    the curses attribute modifier (e.g. ``A_BOLD`` or ``A_DIM``) and
    defaults to ``A_BOLD``.  Returns ``([], 0)`` when *total* is
    non-positive.
    """
    if total <= 0 or width <= 0:
        return [], 0
    if used_mod is None:
        used_mod = curses.A_BOLD
    frac = min(1.0, max(0.0, used / total))
    filled = int(round(frac * width))
    free = width - filled
    border_attr = curses.color_pair(C_BORDER) | curses.A_DIM
    used_attr = curses.color_pair(used_color) | used_mod
    free_attr = curses.color_pair(C_TITLE) | curses.A_DIM
    chunks: list[tuple[str, int]] = [("[", border_attr)]
    if filled:
        chunks.append(("\u2588" * filled, used_attr))
    if free:
        chunks.append(("\u2591" * free, free_attr))
    chunks.append(("]", border_attr))
    return chunks, width + 2


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def _render(win, state: dict | None, connected: bool, tick: int,
            req_scroll: int = 0, follow_newest: bool = True,
            poll_interval: float = 0.5,
            poll_rate_hz: float | None = None,
            poll_age_s: float | None = None,
            phase_breakdown: dict[str, float] | None = None,
            replay_speed: float | None = None,
            replay_done: bool = False) -> tuple[int, int, int]:
    """Render the dashboard.

    Returns ``(total_requests, avail_rows, effective_req_scroll)``.
    Requests are shown **oldest first** (top to bottom).  When *follow_newest*
    is true, the window is scrolled so the **newest** requests appear at the
    bottom of the panel (tail-follow).
    """
    win.erase()
    h, w = win.getmaxyx()
    if h < 6 or w < 30:
        _safe_addstr(win, 0, 0, "Terminal too small", curses.A_BOLD)
        win.noutrefresh()
        return 0, 0, 0

    border = curses.color_pair(C_BORDER) | curses.A_DIM

    # -- Title bar --
    title = " Orchestrator Dashboard "
    _safe_addstr(win, 0, 0, "╔" + "═" * (w - 2) + "╗", border)
    _safe_addstr(win, 0, 2, title, curses.color_pair(C_TITLE) | curses.A_BOLD)

    elapsed_s = state.get("elapsed_s") if state else None
    title_tail_x = 2 + len(title)
    if elapsed_s is not None:
        t0_tag = f" t={_fmt_rel_t0_hms(elapsed_s)} "
        _safe_addstr(win, 0, title_tail_x, t0_tag,
                     curses.color_pair(C_TITLE) | curses.A_DIM)
        title_tail_x += len(t0_tag)
    if replay_speed is not None:
        # Format as "1x", "1.5x", "0.5x" etc. so it's obvious we're not
        # looking at a live orchestrator.
        speed_str = (f"{replay_speed:g}x" if replay_speed != int(replay_speed)
                     else f"{int(replay_speed)}x")
        replay_tag = f" replay {speed_str} "
        _safe_addstr(win, 0, title_tail_x, replay_tag,
                     curses.color_pair(C_WAIT) | curses.A_BOLD)
        title_tail_x += len(replay_tag)
        if replay_done:
            done_tag = "[done] "
            _safe_addstr(win, 0, title_tail_x, done_tag,
                         curses.color_pair(C_STATUS_ERR) | curses.A_BOLD)
            title_tail_x += len(done_tag)

    # Report the configured poll interval alongside the *actual* achieved
    # interval — these diverge when /state or NVML queries are slow,
    # which matters for interpreting how fresh the on-screen state is.
    # Append the dominant sub-phase so the user can tell *why* it's slow.
    if poll_rate_hz is not None and poll_rate_hz > 0:
        actual_interval_s = 1.0 / poll_rate_hz
        phase_suffix = ""
        if phase_breakdown:
            name, dur = max(phase_breakdown.items(), key=lambda kv: kv[1])
            if dur >= 0.01:
                phase_suffix = f" ({name} {dur:.2f}s)"
        interval_tag = (f" poll {poll_interval:.2f}s -> "
                        f"{actual_interval_s:.2f}s{phase_suffix} ")
    else:
        interval_tag = f" poll {poll_interval:.2f}s "
    if connected:
        tag = " connected "
        combined = interval_tag + tag
        _safe_addstr(win, 0, max(0, w - len(combined) - 4), interval_tag,
                     curses.color_pair(C_TITLE) | curses.A_DIM)
        _safe_addstr(win, 0, max(0, w - len(tag) - 4), tag,
                     curses.color_pair(C_STATUS_OK) | curses.A_BOLD)
        _safe_addstr(win, 0, max(0, w - 3), "● ", curses.color_pair(C_STATUS_OK))
    else:
        spin = _SPINNER[tick % len(_SPINNER)]
        tag = f" connecting {spin} "
        combined = interval_tag + tag
        _safe_addstr(win, 0, max(0, w - len(combined) - 2), interval_tag,
                     curses.color_pair(C_TITLE) | curses.A_DIM)
        _safe_addstr(win, 0, max(0, w - len(tag) - 2), tag,
                     curses.color_pair(C_STATUS_ERR) | curses.A_BOLD)

    if state is None:
        _safe_addstr(win, 2, 2, "Waiting for orchestrator...",
                     curses.color_pair(C_STATUS_ERR))
        win.noutrefresh()
        return 0, 0, 0

    # Sort by id so columns render left-to-right in ascending GPU order;
    # the orchestrator may report ids in arrival order (e.g. after
    # ``add``/``sub``), but the dashboard layout should be stable.
    gpu_ids: list[int] = sorted(state.get("gpu_ids", []))
    gpu_mem: dict = state.get("gpu_mem", {})
    models: dict = state.get("models", {})
    # Buddy-allocator slot tree, keyed by gpu id.  JSON keys are strings
    # over the wire, so accept either form when looking up a GPU.
    slots_by_gpu: dict = state.get("slots", {}) or {}

    # Reverse map ``mid -> gpu_id`` over the slot tree, so renderers can
    # show the GPU that *owns* the slot rather than the model's current
    # ``gpu`` field.  Those two normally agree, but during transitions
    # (move, drain, mid-restore) the model's ``gpu`` may briefly differ
    # from where its buddy-allocator slot lives.  The slot snapshot is
    # the authoritative source for "which GPU has the allocation".
    mid_to_slot_gpu: dict[str, int] = {}
    for gpu_key, leaves in slots_by_gpu.items():
        try:
            gpu_id_int = int(gpu_key)
        except (TypeError, ValueError):
            continue
        for leaf in (leaves or []):
            if leaf.get("alloc") and leaf.get("mid") is not None:
                mid_to_slot_gpu[leaf["mid"]] = gpu_id_int

    # Deepest level designated anywhere in the image cache: the smallest
    # allocation any model would take.  Used to size the per-GPU slot
    # bar in the GPU header row and to scale the ``*``-stars suffix in
    # the image-cache tier so both views share a single global scale.
    ic_max_level = max(
        (info.get("level") or 0 for info in models.values()),
        default=0,
    )

    # Classify models by tier
    gpu_active: dict[int, list[tuple[str, dict]]] = {g: [] for g in gpu_ids}
    gpu_sleep: dict[int, list[tuple[str, dict]]] = {g: [] for g in gpu_ids}
    checkpoint_models: list[tuple[str, dict]] = []
    saved_models: list[tuple[str, dict]] = []

    for mid, info in models.items():
        s = info.get("state")
        gpu = info.get("gpu")
        if s in ("up", "running", "init") and gpu is not None and gpu in gpu_active:
            gpu_active[gpu].append((mid, info))
        elif s == "sleep" and gpu is not None and gpu in gpu_sleep:
            gpu_sleep[gpu].append((mid, info))
        elif s in ("checkpoint", "wait"):
            checkpoint_models.append((mid, info))
        elif s == "saved":
            saved_models.append((mid, info))
        else:
            checkpoint_models.append((mid, info))

    # Set of model_ids that have a request still waiting.  Used to
    # highlight the model's row in the GPU/CPU/image-cache tiers --
    # i.e. wherever it shows up.  Computed once here so all three
    # renderers can reuse the lookup.
    waiting_mids: set[str] = {
        r.get("model_id")
        for r in (state.get("requests") or [])
        if r.get("state") == "waiting" and r.get("model_id") is not None
    }

    # Set of model_ids whose orchestrator entry currently has the
    # ``paused`` flag set.  Requests against these models are
    # highlighted cyan in the request panel since the model is not
    # actively progressing them (they're suspended at the engine).
    paused_mids: set[str] = {
        mid for mid, info in models.items() if info.get("paused")
    }

    # For each model with at least one waiting request, the
    # *earliest* ``submit_rel_s`` among those waiters.  ``_state_age_chunk``
    # uses this as a secondary anchor so the per-model age chip resets
    # to ~0 the moment a generate lands on a long-tenure state (e.g.
    # ``sleep`` for 30 min), keeping the chip aligned with the
    # request-row timeline rather than reporting the model's
    # pre-existing tenure.
    waiting_submit_by_mid: dict[str, float] = {}
    for r in (state.get("requests") or []):
        if r.get("state") != "waiting":
            continue
        mid = r.get("model_id")
        sub = r.get("submit_rel_s")
        if mid is None or sub is None:
            continue
        prev = waiting_submit_by_mid.get(mid)
        if prev is None or sub < prev:
            waiting_submit_by_mid[mid] = float(sub)

    n_gpus = max(len(gpu_ids), 1)
    inner_w = w - 2
    col_w = max(inner_w // n_gpus, 12)

    # -- GPU tier --
    row = 1
    _safe_addstr(win, row, 0, "╠" + "═" * (inner_w) + "╣", border)
    row += 1

    # GPU headers span two rows:
    #   row 1: " GPU N   [slot bar]"
    #   row 2: "   XX.X/YY.YG  [mem bar]"
    # so the per-GPU column has more horizontal room for both bars.
    hdr_row = row
    mem_row = row + 1
    for i, gpu_id in enumerate(gpu_ids):
        cx = 1 + i * col_w
        max_col = 1 + (i + 1) * col_w
        header = f" GPU {gpu_id}"
        _safe_addstr(win, hdr_row, cx, header,
                     curses.color_pair(C_GPU_HEADER) | curses.A_BOLD,
                     max_x=max_col)
        # Slot / allocation bar on row 1, right after the GPU label.
        leaves = (slots_by_gpu.get(gpu_id)
                  or slots_by_gpu.get(str(gpu_id)) or [])
        slot_chunks, _ = _slot_bar_chunks(leaves, max_level=ic_max_level)
        if slot_chunks:
            bar_x = cx + len(header) + 1
            _write_chunks(win, hdr_row, bar_x, slot_chunks, max_x=max_col)
        # Memory readout + memory bar on row 2, indented under "GPU N".
        gi = gpu_mem.get(gpu_id, gpu_mem.get(str(gpu_id), {}))
        used = gi.get("used_mib", 0)
        total = gi.get("total_mib", 0)
        rest_parts: list[str] = []
        if total:
            used_gib = used / 1024
            rest_parts.append(f"{used_gib:.1f}/{total / 1024:.1f}G")
        if not rest_parts:
            rest_parts.append("—")
        rest = " " + "  ".join(rest_parts)
        rest_x = cx
        _safe_addstr(
            win, mem_row, rest_x, rest,
            curses.color_pair(C_GPU_HEADER) | curses.A_DIM,
            max_x=max_col,
        )
        next_x = rest_x + len(rest)
        # GPU-memory bar with per-model weights (bold) + kv (dim)
        # segments, sleep+other (magenta), free (dim).
        if total:
            gpu_models = [(mid, info) for mid, info in models.items()
                          if info.get("gpu") == gpu_id]
            mem_chunks, mem_bar_w = _gpu_mem_bar_chunks(
                gpu_models, used, total, width=12)
            if mem_chunks:
                bar_x = next_x + 1
                _write_chunks(win, mem_row, bar_x, mem_chunks, max_x=max_col)
                next_x = bar_x + mem_bar_w
    row = mem_row + 1

    # Compute tier heights per GPU column
    max_active = max((len(v) for v in gpu_active.values()), default=0)
    max_sleep = max((len(v) for v in gpu_sleep.values()), default=0)
    active_height = max(max_active, 1)
    sleep_height = max(max_sleep, 0)
    # 1 row for separator between active and sleep (if any sleep models exist)
    has_sleep = max_sleep > 0
    gpu_tier_height = active_height + (1 + sleep_height if has_sleep else 0)

    gpu_tier_start = row
    gray_attr = curses.color_pair(C_TITLE) | curses.A_DIM

    # -- Active (up/running) models --
    for i, gpu_id in enumerate(gpu_ids):
        cx = 1 + i * col_w
        max_col = 1 + (i + 1) * col_w - 1
        mlist = sorted(gpu_active[gpu_id],
                       key=lambda x: _slot_sort_key(x[1]))
        for j, (mid, info) in enumerate(mlist):
            s = info.get("state", "")
            if s == "running":
                indicator = "▶ "
                colour = curses.color_pair(C_RUNNING) | curses.A_BOLD
            elif s == "init":
                indicator = _SPINNER[tick % len(_SPINNER)] + " "
                colour = curses.color_pair(C_WAIT)
            else:
                indicator = "● "
                colour = curses.color_pair(C_UP)
            if mid in waiting_mids and s != "running":
                colour |= curses.A_BOLD
            paused_mark = "*" if info.get("paused") else ""
            head_label = indicator + paused_mark + mid
            mem_chunks = _mem_chunks(info, colour, ic_max_level,
                                     prefer_gpu_mem=True,
                                     slot_gpu=mid_to_slot_gpu.get(mid))
            age_chunk = _state_age_chunk(
                info, elapsed_s, gray_attr,
                is_waiting_target=mid in waiting_mids,
                waiter_submit_rel_s=waiting_submit_by_mid.get(mid))
            chunks: list[tuple[str, int]] = [(head_label, colour)]
            chunks.extend(mem_chunks)
            chunks.append(age_chunk)
            _write_chunks(win, gpu_tier_start + j, cx + 1, chunks,
                          max_x=max_col)
        if not mlist:
            _safe_addstr(win, gpu_tier_start, cx + 1, "—",
                         curses.A_DIM, max_x=max_col)

    # -- Thin separator + sleep models --
    if has_sleep:
        sep_y = gpu_tier_start + active_height
        for i, gpu_id in enumerate(gpu_ids):
            cx = 1 + i * col_w
            _hline(win, sep_y, cx, col_w - 1, char="·", attr=curses.A_DIM)
        sleep_start = sep_y + 1
        for i, gpu_id in enumerate(gpu_ids):
            cx = 1 + i * col_w
            max_col = 1 + (i + 1) * col_w - 1
            # Newest-transitioned first regardless of slot status; entries
            # missing ``state_since_rel_s`` sink to the bottom of the column.
            sleep_sorted = sorted(
                gpu_sleep[gpu_id],
                key=lambda x: -(x[1].get("state_since_rel_s")
                                if x[1].get("state_since_rel_s") is not None
                                else float("-inf")),
            )
            for j, (mid, info) in enumerate(sleep_sorted):
                row_attr = curses.color_pair(C_UP)
                if mid in waiting_mids:
                    row_attr |= curses.A_BOLD
                paused_mark = "*" if info.get("paused") else ""
                head_label = "○ " + paused_mark + mid
                mem_chunks = _mem_chunks(info, row_attr, ic_max_level,
                                         prefer_gpu_mem=True,
                                         slot_gpu=mid_to_slot_gpu.get(mid))
                age_chunk = _state_age_chunk(
                    info, elapsed_s, gray_attr,
                    is_waiting_target=mid in waiting_mids,
                    waiter_submit_rel_s=waiting_submit_by_mid.get(mid))
                chunks = [(head_label, row_attr)]
                chunks.extend(mem_chunks)
                chunks.append(age_chunk)
                _write_chunks(win, sleep_start + j, cx + 1, chunks,
                              max_x=max_col)

    # Vertical separators between GPU columns
    for i in range(1, n_gpus):
        sep_x = 1 + i * col_w - 1
        if sep_x < w - 1:
            for dy in range(gpu_tier_height):
                _safe_addstr(win, gpu_tier_start + dy, sep_x, "│", border)

    row = gpu_tier_start + gpu_tier_height

    # -- Checkpoint tier --
    _safe_addstr(win, row, 0, "╠" + "═" * inner_w + "╣", border)
    row += 1
    cpu = state.get("cpu_mem", {})
    cpu_used = cpu.get("used_gib", 0)
    cpu_total = cpu.get("total_gib", 0)
    # Weights = pinned CPU bytes of every model that's actually paying
    # for it (i.e. anything past the ``saved`` state).  Saved models
    # carry ``pinned_cpu_bytes`` in their metadata as the recorded
    # weight size on disk, but do not currently hold pinned memory.
    cpu_resident_models = [
        info for info in models.values()
        if info.get("state") != "saved"
    ]
    n_cpu_models = len(cpu_resident_models)
    weights_bytes = sum(
        info.get("pinned_cpu_bytes", 0) or 0
        for info in cpu_resident_models
    )
    weights_gib = weights_bytes / (1 << 30)
    section_label = " CPU "
    _safe_addstr(win, row, 1, section_label,
                 curses.color_pair(C_SECTION) | curses.A_BOLD)
    label_x = 1 + len(section_label)
    count_label = f"({n_cpu_models} models)"
    _safe_addstr(win, row, label_x, count_label,
                 curses.color_pair(C_SECTION) | curses.A_DIM)
    label_x += len(count_label)
    if cpu_total:
        mem_label = f" {cpu_used:.1f}/{cpu_total:.1f}G"
        _safe_addstr(win, row, label_x, mem_label,
                     curses.color_pair(C_SECTION) | curses.A_DIM)
        label_x += len(mem_label)
        bar_chunks, bar_w = _cpu_mem_bar_chunks(cpu_used, cpu_total,
                                                weights_gib)
        if bar_chunks:
            bar_x = label_x + 1
            _write_chunks(win, row, bar_x, bar_chunks, max_x=inner_w + 1)
            label_x = bar_x + bar_w
        if weights_gib > 0:
            breakdown = f"  w={weights_gib:.1f}G"
            _safe_addstr(win, row, label_x, breakdown,
                         curses.color_pair(C_SECTION) | curses.A_DIM,
                         max_x=inner_w + 1)
            label_x += len(breakdown)
    row += 1

    # Oldest -> newest: smaller state_since_rel_s means the transition
    # happened earlier, i.e. the model has been on CPU longer.  Models
    # without a recorded state_since sink to the bottom.  Split the CPU
    # tier into idle (checkpoint) models and the allocation queue
    # (wait), each independently sorted oldest-first.
    def _cpu_sort_key(item):
        return (
            item[1].get("state_since_rel_s")
            if item[1].get("state_since_rel_s") is not None
            else float("inf")
        )

    cpu_idle = sorted(
        (mi for mi in checkpoint_models if mi[1].get("state") != "wait"),
        key=_cpu_sort_key,
    )
    cpu_waiting = sorted(
        (mi for mi in checkpoint_models if mi[1].get("state") == "wait"),
        key=_cpu_sort_key,
    )
    waiting_count = len(cpu_waiting)
    has_waiting = waiting_count > 0
    cpu_models = cpu_idle + cpu_waiting

    # -- Compute bottom-pinned tiers first so CPU can absorb the slack --
    # Layout (top to bottom):
    #   [GPU + CPU header drawn so far]
    #   CPU data (fills down, capped at req_header_y)
    #   ── Requests separator (req_header_y)
    #     Requests header
    #     ...request data rows...
    #   ── Image Cache separator (saved_start)
    #     Image Cache header
    #     ...two-column model list...
    #   ╚═══...═══╝   bottom border
    requests: list[dict] = state.get("requests", [])
    total_requests = len(requests)

    all_models = list(models.items())
    all_models.sort(key=lambda x: x[1].get("pinned_cpu_bytes", 0), reverse=True)

    # Two-column saved layout: ceil(n/2) data rows.  Surrounding frame
    # is separator + header + data + bottom border.
    n_saved_data = (max(len(all_models), 1) + 1) // 2
    saved_tier_rows = 1 + 1 + n_saved_data + 1
    saved_start = max(0, h - saved_tier_rows)

    # Request tier sits between the CPU bottom and the saved tier.
    # CPU gets its natural row count first (one row per model, or one
    # em-dash row when empty), capped by the remaining slack.  Requests
    # take whatever is left between the CPU bottom and the saved tier
    # (minus 2 for the requests separator + header).
    cpu_natural = max(len(cpu_models) + (1 if has_waiting else 0), 1)
    slack_total = max(saved_start - row, 0)
    cpu_rows = min(cpu_natural, slack_total)
    post_cpu_row = row + cpu_rows
    avail_rows = max(saved_start - post_cpu_row - 2, 0)
    req_header_y = post_cpu_row
    req_data_y = req_header_y + 2

    # -- CPU data (fills from current row to just above requests separator) --
    def _draw_cpu_row(y: int, mid: str, info: dict) -> None:
        s = info.get("state", "checkpoint")
        paused_mark = "*" if info.get("paused") else ""
        if s == "wait":
            spin = _SPINNER[tick % len(_SPINNER)]
            head_label = f"{spin} {paused_mark}{mid}"
            colour = curses.color_pair(C_WAIT)
        elif s in ("checkpoint",):
            head_label = paused_mark + mid
            colour = curses.color_pair(C_CHECKPOINT)
        else:
            head_label = paused_mark + mid
            colour = curses.color_pair(C_CHECKPOINT) | curses.A_DIM
        if mid in waiting_mids and s != "wait":
            colour |= curses.A_BOLD
        mem_chunks = _mem_chunks(info, colour, ic_max_level,
                                 slot_gpu=mid_to_slot_gpu.get(mid))
        age_chunk = _state_age_chunk(
            info, elapsed_s, gray_attr,
            is_waiting_target=mid in waiting_mids,
            waiter_submit_rel_s=waiting_submit_by_mid.get(mid))
        chunks: list[tuple[str, int]] = [(head_label, colour)]
        chunks.extend(mem_chunks)
        chunks.append(age_chunk)
        _write_chunks(win, y, 2, chunks, max_x=w)

    if cpu_idle or cpu_waiting:
        for mid, info in cpu_idle:
            if row >= req_header_y:
                break
            _draw_cpu_row(row, mid, info)
            row += 1
        if has_waiting and row < req_header_y:
            queue_label = f" ({waiting_count} waiting) "
            _safe_addstr(win, row, 1, queue_label,
                         curses.color_pair(C_SECTION) | curses.A_DIM)
            sep_x = 1 + len(queue_label)
            sep_w = max(inner_w - len(queue_label), 0)
            if sep_w > 0:
                _hline(win, row, sep_x, sep_w, char="·",
                       attr=curses.A_DIM)
            row += 1
        for mid, info in cpu_waiting:
            if row >= req_header_y:
                break
            _draw_cpu_row(row, mid, info)
            row += 1
    elif row < req_header_y:
        _safe_addstr(win, row, 2, "—", curses.A_DIM)
        row += 1

    # -- Requests separator + header (pinned position) --
    # Skip entirely when there isn't room for both the separator line and
    # the header line above the image-cache tier; otherwise they would
    # collide with the image-cache separator/header.
    has_request_tier = 0 <= req_header_y and req_header_y + 1 < saved_start
    if has_request_tier:
        _safe_addstr(win, req_header_y, 0,
                     "╠" + "═" * inner_w + "╣", border)
        active_requests = sum(
            1 for r in requests
            if r.get("state") in ("waiting", "generating")
        )
        generating_requests = sum(
            1 for r in requests if r.get("state") == "generating"
        )
        req_label = " Requests "
        _safe_addstr(win, req_header_y + 1, 1, req_label,
                     curses.color_pair(C_SECTION) | curses.A_BOLD)
        req_header_extra = 0
        if active_requests:
            active_label = (f" ({active_requests} active "
                            f"/ {generating_requests} generating)")
            _safe_addstr(win, req_header_y + 1, 1 + len(req_label),
                         active_label,
                         curses.color_pair(C_SECTION) | curses.A_DIM)
            req_header_extra = len(active_label)
        tail_tag = "  tail" if follow_newest else "  scroll"
        _safe_addstr(win, req_header_y + 1,
                     1 + len(req_label) + req_header_extra, tail_tag,
                     curses.color_pair(C_SECTION) | curses.A_DIM)

    # -- Image Cache tier (pinned to absolute bottom) --
    _safe_addstr(win, saved_start, 0,
                 "╠" + "═" * inner_w + "╣", border)
    saved_label = " Image Cache "
    _safe_addstr(win, saved_start + 1, 1, saved_label,
                 curses.color_pair(C_SECTION) | curses.A_BOLD)
    image_cache = state.get("image_cache", "")
    if image_cache:
        _safe_addstr(win, saved_start + 1, 1 + len(saved_label),
                     f" {image_cache}",
                     curses.color_pair(C_SECTION) | curses.A_DIM)

    srow = saved_start + 2
    if all_models:
        # Two-column layout, column-major order.
        left_x = 2
        gap = 2
        half = max((inner_w - left_x - gap) // 2, 12)
        right_x = left_x + half + gap
        n = len(all_models)
        rows_n = (n + 1) // 2
        # Stars scale relative to ``ic_max_level`` (computed at the top
        # of this render): the deepest level designated in the image
        # cache becomes a single ``*`` and shallower levels expand from
        # there.  Shares a scale with the per-GPU slot bar.
        for r in range(rows_n):
            left_idx = r
            right_idx = r + rows_n
            pair: list[tuple[int, tuple[str, dict]]] = [
                (0, all_models[left_idx])]
            if right_idx < n:
                pair.append((1, all_models[right_idx]))
            for col_idx, (mid, info) in pair:
                vc = info.get("vllm_config", {})
                model_path = vc.get("model", "")
                is_saved = info.get("state") == "saved"
                # Image cache shows the model's *intrinsic* level (a
                # property derived from gpu_memory_utilization), since
                # saved/cached models do not currently hold a slot.
                # When the model *does* hold a slot, the GPU id is
                # appended after the stars (e.g. ``****2``) in dim
                # gray so the reader can locate it without
                # cross-referencing the per-GPU column.
                if is_saved:
                    colour = curses.color_pair(C_SAVED)
                else:
                    colour = curses.color_pair(C_SAVED) | curses.A_DIM
                # Restrict the image-cache bold to saved models: for
                # any other state the model is already shown bold in
                # its own (CPU or GPU) tier.
                if is_saved and mid in waiting_mids:
                    colour |= curses.A_BOLD
                if col_idx == 0:
                    x = left_x
                    max_x = right_x - gap + 1
                else:
                    x = right_x
                    max_x = inner_w + 1
                chunks: list[tuple[str, int]] = [(mid, colour)]
                chunks.extend(_pinned_stars_chunks(
                    info, colour, ic_max_level))
                chunks.append((f"  {model_path}", colour))
                _write_chunks(win, srow, x, chunks, max_x=max_x)
            srow += 1
    else:
        _safe_addstr(win, srow, 2, "—", curses.A_DIM)
        srow += 1

    if srow < h:
        _safe_addstr(win, srow, 0, "╚" + "═" * inner_w + "╝", border)

    # -- Request data rows (pinned position, between the two separators) --
    row = req_data_y
    eff_scroll = 0

    if requests and avail_rows > 0:
        if follow_newest:
            eff_scroll = max(0, total_requests - avail_rows)
        else:
            eff_scroll = min(req_scroll, max(0, total_requests - avail_rows))
        visible = requests[eff_scroll:eff_scroll + avail_rows]
        # Pre-compute column widths so middle_text aligns after model_id
        # across every visible row.  Each component (``#N``, ``tlog``,
        # ``model_id``) is padded to its max observed width.
        _id_w = max(
            (len(f"#{r.get('req_id', 0) + 1}") for r in visible),
            default=0)
        _tlog_w = max(
            (len(_fmt_request_time_log(r)) for r in visible),
            default=0)
        _mid_w = max(
            (len(r.get("model_id", "?")) for r in visible),
            default=0)
        for req in visible:
            if row >= saved_start:
                break
            rid = req.get("req_id", 0)
            mid = req.get("model_id", "?")
            rstate = req.get("state", "?")
            wait_s = req.get("wait_s")
            gen_s = req.get("gen_s")
            paused_s = req.get("paused_s") or 0.0
            ptok = req.get("prompt_tokens")
            ctok = req.get("completion_tokens")
            ttft_s = req.get("ttft_s")
            tpot_ms = req.get("tpot_ms")

            tlog = _fmt_request_time_log(req)

            # -- Per-row colour --
            if rstate == "waiting":
                primary = curses.color_pair(C_WAIT)
            elif rstate == "generating":
                primary = curses.color_pair(C_RUNNING)
            elif rstate == "error":
                # Maroon for errored generates (e.g.
                # ``_fail_all_inflight`` after a worker cmd OOM).
                # Distinct from the dim-white ``done`` treatment so
                # failures don't visually blend with successes.
                primary = curses.color_pair(C_ERROR)
            else:
                primary = curses.color_pair(C_SAVED) | curses.A_DIM
            # In-flight requests against a paused model recolor cyan
            # (matching the C_UP hue used elsewhere for paused/up
            # models) so it's visually obvious why they're not making
            # progress.  Done / errored requests keep their own
            # styling (dim-white / maroon respectively).
            if mid in paused_mids and rstate in ("waiting", "generating"):
                primary = curses.color_pair(C_UP)

            # -- Middle portion (wait/gen/tokens), flows left --
            middle_parts: list[str] = []
            if rstate == "waiting":
                spin = _SPINNER[tick % len(_SPINNER)]
                middle_parts.append(
                    f"{spin} wait {wait_s:.1f}s"
                    if wait_s is not None else f"{spin} wait")
            elif rstate == "generating":
                middle_parts.append(
                    f"wait {wait_s:.1f}s"
                    if wait_s is not None else "wait")
                spin = _SPINNER[tick % len(_SPINNER)]
                # Spinner sits next to the segment that's actively
                # advancing: ``gen`` when running, ``paused`` when the
                # model is currently paused (``gen`` is frozen then).
                # ``paused N.Ns`` stays visible after resume too -- a
                # sticky record of accumulated pause time, just
                # without the spinner.
                is_paused_now = mid in paused_mids
                gen_spin = "" if is_paused_now else f"{spin} "
                middle_parts.append(
                    f"{gen_spin}gen {gen_s:.1f}s"
                    if gen_s is not None else f"{gen_spin}gen")
                if paused_s > 0:
                    paused_spin = f"{spin} " if is_paused_now else ""
                    middle_parts.append(
                        f"{paused_spin}paused {paused_s:.1f}s")
            else:
                wait_str = f"wait {wait_s:.1f}s" if wait_s is not None else ""
                gen_str = f"gen {gen_s:.1f}s" if gen_s is not None else ""
                if wait_str:
                    middle_parts.append(wait_str)
                if gen_str:
                    middle_parts.append(gen_str)
                if paused_s > 0:
                    middle_parts.append(f"paused {paused_s:.1f}s")
                if ptok is not None or ctok is not None:
                    # Match ``cl.requests()`` table format: ``out/in``
                    # (completion / prompt), with ``ctok or 0`` coercing
                    # a missing completion count to ``0`` when only the
                    # prompt total is known.  Bare slash form keeps the
                    # row compact -- consistent with the client's
                    # dedicated ``tokens`` column header.
                    if ptok is not None:
                        middle_parts.append(f"{ctok or 0}/{ptok} tok")
                    else:
                        middle_parts.append(f"{ctok} tok")
                if ttft_s is not None:
                    if ttft_s >= 1.0:
                        middle_parts.append(f"ttft {ttft_s:.2f}s")
                    else:
                        middle_parts.append(f"ttft {ttft_s * 1000:.0f}ms")
                if tpot_ms is not None:
                    middle_parts.append(f"tpot {tpot_ms:.1f}ms")

            prefix_parts = [f"#{rid + 1}".ljust(_id_w)]
            if _tlog_w:
                prefix_parts.append(tlog.ljust(_tlog_w))
            prefix_parts.append(mid.ljust(_mid_w))
            prefix = " ".join(prefix_parts)
            middle_text = " ".join(p for p in middle_parts if p)
            left_text = prefix if not middle_text else f"{prefix}  {middle_text}"

            # -- Right tail: state-duration bracket, painted per-segment.
            # Active (current) segment is bright white; finished ones are
            # dim.  Populated incrementally on every row, even while the
            # request is still in flight.
            # When the model is paused, the model's ``state_since_rel_s``
            # is the pause-start time (``pause()`` forces a
            # ``_set_state(.., "up")`` which updates ``state_since``,
            # and no further transitions happen while paused).  We
            # forward that as the cutoff so segments that started
            # after pause render without their ticking duration.
            paused_since_rel_s = None
            if mid in paused_mids:
                paused_since_rel_s = models.get(mid, {}).get(
                    "state_since_rel_s")
            durs_segs = _build_state_durations(
                req, elapsed_s,
                paused_since_rel_s=paused_since_rel_s)

            def _build_durs(start_idx: int) -> tuple[list[tuple[str, int]], int]:
                """Build bracket chunks showing ``durs_segs[start_idx:]``.
                A ``..., `` head segment is inserted whenever
                *start_idx* > 0 so the dropped (oldest) entries are
                visibly indicated.
                """
                if not durs_segs:
                    return [], 0
                dim_attr = curses.color_pair(C_TITLE) | curses.A_DIM
                active_attr = curses.color_pair(C_TITLE) | curses.A_BOLD
                chunks: list[tuple[str, int]] = [("[", dim_attr)]
                if start_idx > 0:
                    chunks.append(("..., ", dim_attr))
                for i in range(start_idx, len(durs_segs)):
                    if i > start_idx:
                        chunks.append((", ", dim_attr))
                    text, is_active = durs_segs[i]
                    chunks.append(
                        (text, active_attr if is_active else dim_attr))
                chunks.append(("]", dim_attr))
                return chunks, sum(len(t) for t, _ in chunks)

            durs_chunks, tail_len = _build_durs(0)

            # -- Render: left_text takes priority, flowing from col 2
            # up to the right border.  The bracket right-aligns when
            # there's room; otherwise it drops oldest segments from
            # its left (replaced with a leading ``..., ``) until it
            # fits in the slack past the left text.  At least the
            # most recent segment is kept whenever any bracket
            # renders -- that's the bit users actually need to see.
            left_start = 2
            if durs_chunks:
                _safe_addstr(win, row, left_start, left_text, primary,
                             max_x=w - 1)
                left_end = left_start + len(left_text)
                right_border = w - 1
                GAP = 2  # min spaces between left text and bracket
                avail = max(0, right_border - left_end - GAP)
                if tail_len > avail:
                    for start_idx in range(1, len(durs_segs)):
                        durs_chunks, tail_len = _build_durs(start_idx)
                        if tail_len <= avail:
                            break
                bracket_x = max(left_end + GAP, right_border - tail_len)
                x = bracket_x
                for text, attr in durs_chunks:
                    if x >= w - 1:
                        break
                    _safe_addstr(win, row, x, text, attr, max_x=w - 1)
                    x += len(text)
            else:
                _safe_addstr(win, row, left_start, left_text, primary,
                             max_x=w - 1)
            row += 1
    elif req_data_y < saved_start:
        _safe_addstr(win, req_data_y, 2, "—", curses.A_DIM)

    # -- Left/right borders (full height up to the bottom border) --
    border_bottom = min(srow, h - 1)
    for y in range(1, border_bottom):
        _safe_addstr(win, y, 0, "║", border)
        _safe_addstr(win, y, w - 1, "║", border)

    win.noutrefresh()
    return total_requests, avail_rows, eff_scroll


# ---------------------------------------------------------------------------
# Disconnect overlay (in-curses save prompt)
# ---------------------------------------------------------------------------


def _render_disconnect_overlay(win, dc_mode: str, dc_filename_buf: str,
                               dc_save_target: str | None,
                               dc_message: str,
                               poller) -> None:
    """Draw a centered modal on top of the (frozen) dashboard.

    Content depends on the current save-prompt step (*dc_mode*).  Must be
    followed by ``stdscr.noutrefresh()`` and ``curses.doupdate()`` to take
    effect since ``_render`` already called ``noutrefresh`` once.
    """
    import os as _os

    h, w = win.getmaxyx()

    title = " Disconnected "
    cursor_line_idx: int | None = None
    cursor_col_offset: int = 0

    if dc_mode == "ask_save":
        n = poller.recorded_count()
        lines = [
            "",
            "Orchestrator disconnected.",
            f"Recorded {n} ticks  ({poller.temp_path()}).",
            "",
            "Save session log?  [y] yes   [n/Esc] discard",
            "",
        ]
    elif dc_mode == "enter_filename":
        lines = [
            "",
            "Filename (relative to",
            f"  {_os.getcwd()}):",
            "",
            f"> {dc_filename_buf}",
            "",
            "[Enter] save   [Esc] cancel",
            "",
        ]
        cursor_line_idx = 4
        cursor_col_offset = len("> ") + len(dc_filename_buf)
    elif dc_mode == "confirm_overwrite":
        lines = [
            "",
            f"{dc_save_target}",
            "already exists.",
            "",
            "Overwrite?  [y] yes   [n] back   [Esc] cancel",
            "",
        ]
    elif dc_mode == "done":
        title = " Done "
        lines = ["", dc_message, "", "Press any key to exit.", ""]
    else:
        return

    content_w = max((len(s) for s in lines), default=0) + 4
    content_w = max(content_w, len(title) + 4)
    content_w = min(content_w, max(w - 4, 10))
    content_h = len(lines) + 2

    y0 = max(0, (h - content_h) // 2)
    x0 = max(0, (w - content_w) // 2)

    border_attr = curses.color_pair(C_BORDER) | curses.A_BOLD
    _safe_addstr(win, y0, x0,
                 "┌" + "─" * (content_w - 2) + "┐", border_attr)
    for i in range(1, content_h - 1):
        _safe_addstr(win, y0 + i, x0, "│", border_attr)
        _safe_addstr(win, y0 + i, x0 + content_w - 1, "│", border_attr)
        # Blank interior so the frozen dashboard doesn't bleed through.
        _safe_addstr(win, y0 + i, x0 + 1,
                     " " * (content_w - 2), curses.A_NORMAL)
    _safe_addstr(win, y0 + content_h - 1, x0,
                 "└" + "─" * (content_w - 2) + "┘", border_attr)

    _safe_addstr(win, y0, x0 + 2, title,
                 curses.color_pair(C_TITLE) | curses.A_BOLD)

    for i, line in enumerate(lines):
        _safe_addstr(win, y0 + 1 + i, x0 + 2, line, curses.A_NORMAL)

    if cursor_line_idx is not None:
        cy = y0 + 1 + cursor_line_idx
        cx = x0 + 2 + cursor_col_offset
        try:
            win.move(cy, min(cx, x0 + content_w - 2))
        except curses.error:
            pass


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _main(stdscr, args, poller, exit_reason: dict):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)
    _init_colours()
    curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)

    import os as _os

    tick = 0
    req_scroll = 0
    prev_total_requests = 0
    follow_newest = True
    SCROLL_STEP = 3
    # Grace period before treating a disconnect as terminal, to tolerate
    # transient HTTP blips without bailing out.
    DISCONNECT_GRACE_S = 2.0
    is_live = not args.replay

    # Save-prompt state machine.  ``None`` == normal dashboard interaction;
    # other states drive a modal overlay on top of the (now frozen)
    # dashboard and steal keystrokes until the user resolves the prompt.
    dc_mode: str | None = None
    dc_filename_buf: str = ""
    dc_save_target: str | None = None
    dc_message: str = ""

    # Tracks whether we've already painted the final "[done]" replay
    # frame so we don't keep redrawing it every iteration.
    replay_frozen_rendered = False

    while True:
        try:
            key = stdscr.getch()
        except KeyboardInterrupt:
            # Treat Ctrl+C the same as the user hitting 'q': offer to
            # save if we have something to save, otherwise quit.  When a
            # modal is already open we fall through to the dc_mode
            # branches as if Esc had been pressed, so the user can break
            # out without corrupting the save flow.
            if dc_mode is None:
                if is_live and poller.recorded_count() > 0:
                    dc_mode = "ask_save"
                    key = -1
                else:
                    break
            else:
                key = 27

        if dc_mode is None:
            # Normal dashboard interaction.
            if key == ord("q") or key == ord("Q") or key == 27:
                # Offer to save before exiting, same flow as disconnect.
                # Skip the prompt in replay mode or when nothing's been
                # recorded yet — nothing to save, so just quit.
                if is_live and poller.recorded_count() > 0:
                    dc_mode = "ask_save"
                else:
                    break
            elif key in (ord("t"), ord("T")):
                follow_newest = True
            elif key == curses.KEY_RESIZE:
                curses.update_lines_cols()
            if key == curses.KEY_MOUSE:
                try:
                    _, _, _, _, bstate = curses.getmouse()
                    if bstate & curses.BUTTON4_PRESSED:
                        follow_newest = False
                        req_scroll = max(0, req_scroll - SCROLL_STEP)
                    elif bstate & curses.BUTTON5_PRESSED:
                        follow_newest = False
                        req_scroll += SCROLL_STEP
                except curses.error:
                    pass

        elif dc_mode == "ask_save":
            if key in (ord("y"), ord("Y")):
                dc_filename_buf = ""
                dc_mode = "enter_filename"
                curses.curs_set(1)
            elif key in (ord("n"), ord("N"), 27):
                poller.finalize(None)
                dc_message = "Session discarded."
                dc_mode = "done"

        elif dc_mode == "enter_filename":
            if key == 27:
                poller.finalize(None)
                dc_message = "Session discarded."
                dc_mode = "done"
                curses.curs_set(0)
            elif key in (10, 13, curses.KEY_ENTER):
                if dc_filename_buf:
                    path = (dc_filename_buf
                            if _os.path.isabs(dc_filename_buf)
                            else _os.path.abspath(dc_filename_buf))
                    if _os.path.exists(path):
                        dc_save_target = path
                        dc_mode = "confirm_overwrite"
                        curses.curs_set(0)
                    else:
                        try:
                            written = poller.finalize(path)
                            dc_message = (f"Saved {written} ticks to "
                                          f"{path}.")
                        except OSError as exc:
                            dc_message = f"save failed: {exc}"
                        dc_mode = "done"
                        curses.curs_set(0)
            elif key in (curses.KEY_BACKSPACE, 127, 8):
                dc_filename_buf = dc_filename_buf[:-1]
            elif 32 <= key < 127:
                dc_filename_buf += chr(key)

        elif dc_mode == "confirm_overwrite":
            if key in (ord("y"), ord("Y")):
                try:
                    written = poller.finalize(dc_save_target)
                    dc_message = (f"Saved {written} ticks to "
                                  f"{dc_save_target}.")
                except OSError as exc:
                    dc_message = f"save failed: {exc}"
                dc_mode = "done"
            elif key in (ord("n"), ord("N")):
                dc_mode = "enter_filename"
                curses.curs_set(1)
            elif key == 27:
                poller.finalize(None)
                dc_message = "Session discarded."
                dc_mode = "done"

        elif dc_mode == "done":
            if key != -1:
                break

        # Once a replay has played out all of its frames, freeze the
        # display: stop advancing the spinner tick and skip the render
        # pass entirely.  The last frame stays painted on screen so the
        # user can read it at their leisure; only keyboard input is
        # still processed.
        replay_done = (not is_live
                       and hasattr(poller, "is_done")
                       and poller.is_done())

        # Pick the interval shown in the title bar.  For a live session
        # this is the --interval CLI arg.  For replay it's meaningless
        # (you can't "poll" a file), so we report the *effective*
        # playback cadence instead: the average recorded gap divided by
        # the speed multiplier.
        if is_live:
            shown_interval = args.interval
        else:
            # Instantaneous gap (already scaled by replay speed).  Falls
            # back to the CLI value on the first tick before we have a
            # predecessor to compare against.
            cur = (poller.current_interval()
                   if hasattr(poller, "current_interval") else None)
            shown_interval = cur if cur is not None else args.interval

        if not replay_done:
            state, connected = poller.snapshot()
            poll_rate_hz, poll_age_s = poller.poll_stats()
            phase_breakdown = poller.phase_breakdown()
            total, _avail, eff = _render(
                stdscr, state, connected, tick, req_scroll,
                follow_newest=follow_newest,
                poll_interval=shown_interval,
                poll_rate_hz=poll_rate_hz,
                poll_age_s=poll_age_s,
                phase_breakdown=phase_breakdown,
                replay_speed=(getattr(args, "replay_speed", None)
                              if args.replay else None),
                replay_done=False,
            )
            if total > prev_total_requests:
                follow_newest = True
            prev_total_requests = total
            req_scroll = eff
        elif not replay_frozen_rendered:
            # First tick after the replay finishes — repaint one last
            # time with the "[done]" tag, then stop touching the screen.
            state, connected = poller.snapshot()
            _render(
                stdscr, state, connected, tick, req_scroll,
                follow_newest=follow_newest,
                poll_interval=shown_interval,
                replay_speed=getattr(args, "replay_speed", None),
                replay_done=True,
            )
            replay_frozen_rendered = True

        if dc_mode is not None:
            _render_disconnect_overlay(
                stdscr, dc_mode, dc_filename_buf, dc_save_target,
                dc_message, poller,
            )
            stdscr.noutrefresh()

        curses.doupdate()
        if not replay_done:
            tick += 1
        time.sleep(0.1)

        # Open the save prompt once a live poller has been disconnected
        # for longer than the grace period.  The dashboard stays rendered
        # behind the modal so the user still sees the last state.
        if (dc_mode is None
                and is_live
                and poller.ever_connected()):
            age = poller.disconnect_age()
            if age is not None and age >= DISCONNECT_GRACE_S:
                exit_reason["disconnected"] = True
                if poller.recorded_count() == 0:
                    poller.finalize(None)
                    dc_message = "Disconnected (no polls recorded)."
                    dc_mode = "done"
                else:
                    dc_mode = "ask_save"


def _prompt_startup_mode() -> tuple[str | None, float]:
    """Ask the user at startup whether to replay a file or start fresh.

    Returns ``(replay_path, speed)``.  ``replay_path`` is ``None`` for a
    new live session (in which case *speed* is ignored).  *speed* is a
    multiplier applied to the replay wall-clock: 2.0 plays twice as fast,
    0.5 at half speed.  Defaults to 1.0.
    """
    import os as _os

    try:
        ans = input("Replay from a saved session file? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(0)
    if ans not in ("y", "yes"):
        return None, 1.0

    path: str
    while True:
        try:
            p = input("Replay file path: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(0)
        if not p:
            print("  (empty path, starting a new live session instead)")
            return None, 1.0
        resolved = p if _os.path.isabs(p) else _os.path.abspath(p)
        if not _os.path.isfile(resolved):
            print(f"  no such file: {resolved}")
            continue
        path = resolved
        break

    while True:
        try:
            raw = input("Replay speed multiplier [1.0]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(0)
        if not raw:
            return path, 1.0
        try:
            speed = float(raw)
        except ValueError:
            print(f"  not a number: {raw}")
            continue
        if speed <= 0:
            print(f"  speed must be > 0 (got {speed})")
            continue
        return path, speed


def main():
    import os as _os

    parser = argparse.ArgumentParser(description="Orchestrator dashboard")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8157)
    parser.add_argument("--interval", type=float, default=0.5,
                        help="polling interval in seconds")
    args = parser.parse_args()
    args.replay = None  # kept on args for _main's is_live check

    replay_path, replay_speed = _prompt_startup_mode()
    args.replay = replay_path
    args.replay_speed = replay_speed

    if replay_path:
        poller = ReplayPoller(replay_path, speed=replay_speed)
    else:
        url = f"http://{args.host}:{args.port}/state"
        # Always stream to a temp file; on disconnect we offer to keep it.
        temp_path = _os.path.join(
            "/tmp", f"dashboard-session-{_os.getpid()}.jsonl")
        poller = Poller(url, args.interval, temp_path=temp_path)
    poller.start()

    exit_reason: dict = {"disconnected": False}
    try:
        curses.wrapper(lambda stdscr: _main(stdscr, args, poller, exit_reason))
    finally:
        poller.stop()

    if not replay_path:
        # Save / discard was already handled inside curses on disconnect.
        # This is a safety net for the clean-quit path and a no-op if
        # finalize() already ran.
        poller.finalize(None)


if __name__ == "__main__":
    main()
