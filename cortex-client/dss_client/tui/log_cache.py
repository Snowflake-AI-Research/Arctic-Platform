"""Client-side persistent cache for the read-only log TUI.

Reopening a job should not re-retrieve logs already fetched — that is wasted
load on the (deliberately stateless) Zone Manager. The transport is already
cursor-based (``tail_logs(cursor) -> {entries, next_cursor, eof}``), so we cache,
per ``(job_id, source_id)`` on the *client*:

  * the opaque ``next_cursor`` — lets the next open resume with a delta-only fetch
  * the retrieved entries (append-only JSONL) — lets the next open repaint instantly

On reopen the TUI replays the local cache, then resumes ``tail_logs`` from the
saved cursor. Nothing is added to the server.

Pure stdlib, NO ``textual`` import, so it is unit-testable like ``format.py``.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple

# Cache dir resolution mirrors dss_neutrino_cli._login_state_path: an explicit
# override wins, else XDG_CACHE_HOME, else ~/.cache, namespaced under the app.
_OVERRIDE_ENV = "NEUTRINO_TUI_CACHE_DIR"
_XDG_CACHE_ENV = "XDG_CACHE_HOME"
_NAMESPACE = "neutrino-tui"

# Backoff defaults, matching NeutrinoClient.poll_backoff_multiplier /
# poll_max_interval. Kept module-level so a test/mock client need not carry them.
_POLL_BACKOFF_MULTIPLIER = 1.25
_POLL_MAX_INTERVAL = 6.0

_ERROR_LOG = os.path.expanduser("~/.neutrino-tui-errors.log")
_persist_warned = False

# Filename-safe byte set; everything else (incl. ':' '/' '%') is %XX-escaped so
# a source_id collapses to a single, reversible path component.
_SAFE = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_."
)
_MAX_NAME = 200


def cache_root() -> Path:
    """Base cache directory (not created here; writers mkdir lazily)."""
    override = os.environ.get(_OVERRIDE_ENV)
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get(_XDG_CACHE_ENV)
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / _NAMESPACE


def safe_source_name(source_id: str) -> str:
    """Encode an arbitrary source_id into one filesystem-safe path component.

    Reversible via :func:`unsafe_source_name` for ids under ``_MAX_NAME`` bytes;
    longer ids get a sha1 suffix and are not reversible (correctness comes from
    the cursors.json id->cursor map, not the filename).
    """
    out = []
    for b in source_id.encode("utf-8"):
        c = chr(b)
        out.append(c if c in _SAFE else "%%%02X" % b)
    name = "".join(out)
    if name in (".", ".."):
        name = name.replace(".", "%2E")
    if len(name) > _MAX_NAME:
        import hashlib

        digest = hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:12]
        name = name[: _MAX_NAME - 13] + "-" + digest
    return name


def unsafe_source_name(name: str) -> str:
    """Inverse of :func:`safe_source_name` (for tests / introspection)."""
    raw = bytearray()
    i = 0
    while i < len(name):
        ch = name[i]
        if ch == "%" and i + 2 < len(name) + 1 and i + 3 <= len(name):
            raw.append(int(name[i + 1 : i + 3], 16))
            i += 3
        else:
            raw.append(ord(ch))
            i += 1
    return raw.decode("utf-8")


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text via same-dir temp + fsync + os.replace (atomic on one fs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".swp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class LogCache:
    """Per-job on-disk cache of log entries + resume cursors.

    Layout::

        <cache_root>/jobs/<safe(job_id)>/cursors.json
        <cache_root>/jobs/<safe(job_id)>/logs/<safe(source_id)>.jsonl
    """

    DEFAULT_MAX_BYTES = 50 * 1024 * 1024
    DEFAULT_MAX_LINES = 100_000

    def __init__(
        self,
        job_id: str,
        *,
        root: Optional[Path] = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_lines: int = DEFAULT_MAX_LINES,
    ) -> None:
        base = root if root is not None else cache_root()
        self._dir = base / "jobs" / safe_source_name(job_id)
        self._max_bytes = max_bytes
        self._max_lines = max_lines

    # ---- paths ----
    def _logs_dir(self) -> Path:
        return self._dir / "logs"

    def _jsonl_path(self, source_id: str) -> Path:
        return self._logs_dir() / (safe_source_name(source_id) + ".jsonl")

    def _cursors_path(self) -> Path:
        return self._dir / "cursors.json"

    # ---- cursors ----
    def _read_cursors(self) -> dict:
        try:
            doc = json.loads(self._cursors_path().read_text(encoding="utf-8"))
            if isinstance(doc, dict):
                return doc
        except (OSError, ValueError):
            pass
        return {"version": 1, "sources": {}}

    def get_cursor(self, source_id: str) -> Optional[str]:
        entry = self._read_cursors().get("sources", {}).get(source_id)
        if isinstance(entry, dict):
            cur = entry.get("cursor")
            return cur if isinstance(cur, str) else None
        return None

    def set_cursor(self, source_id: str, cursor: Optional[str]) -> None:
        doc = self._read_cursors()
        sources = doc.setdefault("sources", {})
        sources[source_id] = {
            "cursor": cursor,
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        doc["version"] = 1
        _atomic_write_text(
            self._cursors_path(), json.dumps(doc, sort_keys=True) + "\n"
        )

    # ---- entries ----
    def append_entries(self, source_id: str, entries: Iterable[Any]) -> int:
        path = self._jsonl_path(source_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(path, "a", encoding="utf-8") as f:
            for e in entries:
                try:
                    line = json.dumps(e, separators=(",", ":"))
                except (TypeError, ValueError):
                    line = json.dumps({"_raw": str(e)}, separators=(",", ":"))
                f.write(line + "\n")
                n += 1
            f.flush()
        if n:
            self._rotate_if_needed(source_id)
        return n

    def load_entries(self, source_id: str, *, limit: Optional[int] = None) -> list:
        path = self._jsonl_path(source_id)
        if not path.exists():
            return []
        sink: Any = deque(maxlen=limit) if limit else []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sink.append(json.loads(line))
                    except ValueError:
                        continue  # torn last line / corruption: skip
        except OSError:
            return []
        return list(sink)

    def load(
        self, source_id: str, *, limit: Optional[int] = None
    ) -> Tuple[list, Optional[str]]:
        return self.load_entries(source_id, limit=limit), self.get_cursor(source_id)

    def cached_sources(self) -> list:
        """Source ids that have cached log data on disk for this job. Lets the
        TUI show previously-viewed sources (and replay them) even when a live
        ``list_log_sources`` call fails."""
        d = self._logs_dir()
        if not d.exists():
            return []
        out = []
        for p in sorted(d.glob("*.jsonl")):
            try:
                out.append(unsafe_source_name(p.stem))
            except Exception:  # noqa: BLE001 - skip an undecodable (hashed) name
                continue
        return out

    # ---- maintenance ----
    def _rotate_if_needed(self, source_id: str) -> None:
        path = self._jsonl_path(source_id)
        try:
            if path.stat().st_size <= self._max_bytes:
                return
        except OSError:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return
        # Keep the newest max_lines, then drop from the front until within bytes.
        kept = deque(lines, maxlen=self._max_lines)
        kept = list(kept)
        budget = self._max_bytes
        total = sum(len(s.encode("utf-8")) for s in kept)
        while kept and total > budget:
            total -= len(kept[0].encode("utf-8"))
            kept.pop(0)
        # The cursor is intentionally NOT touched: history is bounded but the
        # resume position is preserved, so reopen still delta-fetches correctly.
        _atomic_write_text(path, "".join(kept))


def _note_persist_error(exc: BaseException) -> None:
    global _persist_warned
    if _persist_warned:
        return
    _persist_warned = True
    try:
        with open(_ERROR_LOG, "a", encoding="utf-8") as f:
            f.write("log cache persistence disabled (%s: %s)\n" % (type(exc).__name__, exc))
    except OSError:
        pass


def _safe_append(cache: LogCache, source_id: str, entries: list) -> None:
    try:
        cache.append_entries(source_id, entries)
    except OSError as exc:  # disk full / unwritable: degrade to live-only
        _note_persist_error(exc)


def _safe_set_cursor(cache: LogCache, source_id: str, cursor: Optional[str]) -> None:
    try:
        cache.set_cursor(source_id, cursor)
    except OSError as exc:
        _note_persist_error(exc)


def cached_pages(
    cache: LogCache,
    source_id: str,
    fetch: Callable[[Optional[str]], dict],
    *,
    entries_key: str,
    follow: bool,
    poll_interval: float,
    live: bool = True,
    sleep: Callable[[float], None] = time.sleep,
    replay_limit: Optional[int] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    backoff_multiplier: float = _POLL_BACKOFF_MULTIPLIER,
    max_interval: float = _POLL_MAX_INTERVAL,
) -> Iterator[Tuple[str, list]]:
    """Replay the cached entries, then (when ``live``) resume ``fetch`` from the
    saved cursor.

    Yields ``(kind, entries)`` where ``kind`` is ``"cache"`` for the replayed
    batch and ``"live"`` for each newly fetched page. Entries are yielded a
    whole page (or the whole cached batch) at a time so the consumer can render
    them in one shot — essential for high-velocity logs, where per-line
    rendering would saturate the UI thread. New entries are appended to disk and
    the advancing cursor persisted; on a crash between the two we re-fetch a few
    lines next time rather than skip any.

    ``live=False`` replays the cache and stops without ever calling ``fetch`` —
    used for terminal jobs whose zone is gone, so the operation API (which would
    500) is never hit.
    """
    cancelled = is_cancelled or (lambda: False)

    cached, cursor = cache.load(source_id, limit=replay_limit)
    if cached and not cancelled():
        yield ("cache", cached)

    if not live:
        return  # cache-only (terminal job): never touch the operation API

    delay = poll_interval
    while True:
        if cancelled():
            return
        page = fetch(cursor)
        new_cursor = page.get("next_cursor", cursor)
        entries = page.get(entries_key) or []
        if entries:
            _safe_append(cache, source_id, entries)  # entries first...
            _safe_set_cursor(cache, source_id, new_cursor)  # ...then cursor
            cursor = new_cursor
            if cancelled():
                return
            yield ("live", entries)  # whole page in one batch
            # Throttle floor: even when the source is busy, wait at least
            # poll_interval before the next request. Biases toward ZMD
            # reliability over log freshness — caps the rate at ~1 req/s per
            # source per viewer regardless of how chatty the source is.
            delay = poll_interval
            sleep(delay)
            continue
        if new_cursor != cursor:
            _safe_set_cursor(cache, source_id, new_cursor)
            cursor = new_cursor
        if not follow:
            # An empty page means we've drained everything available right now.
            # Stop rather than poll forever — independent of the server's eof
            # flag (the pod-logs backend never sets eof=True). This is the
            # "terminal job: replay + drain once, then stop" path.
            return
        sleep(delay)
        delay = min(delay * backoff_multiplier, max_interval)


def cached_log_pages(
    cache: LogCache,
    client: Any,
    job_id: str,
    sub_job_id: str,
    *,
    follow: bool,
    poll_interval: float,
    live: bool = True,
    replay_limit: Optional[int] = 2000,
    tail_lines: Optional[int] = 2000,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Iterator[Tuple[str, list]]:
    """``cached_pages`` wired to ``client.tail_logs`` for a sub-job's logs. The
    cache is keyed by ``sub_job_id``.

    First open (no cached cursor) tails the last ``tail_lines`` lines via
    ``max_lines`` instead of streaming the whole pod log from the beginning — a
    long-running job can have hours of backlog, and reading it all is a slow,
    heavy catch-up storm on the ZMD/kubelet. Reopens resume from the saved
    cursor (delta-only), so ``max_lines`` is sent only on that first request.
    """

    def fetch(cursor: Optional[str]) -> dict:
        kw: dict = {"cursor": cursor, "sub_job_id": sub_job_id}
        if cursor is None and tail_lines:
            kw["max_lines"] = tail_lines  # first fetch: tail recent, not from start
        return client.tail_logs(job_id, **kw)

    return cached_pages(
        cache,
        sub_job_id,
        fetch,
        entries_key="entries",
        follow=follow,
        poll_interval=poll_interval,
        live=live,
        replay_limit=replay_limit,
        is_cancelled=is_cancelled,
    )
