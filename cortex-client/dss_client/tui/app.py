"""Read-only Textual TUI for Neutrino logs and ZMD scheduling events.

Flow: open → a list of jobs with their status (via the SDK's list_jobs) →
pick one → its log sources (zone server / job orchestration / Ray head /
scheduling events) → live tail. Pass a job_id on the command line to jump
straight to that job's logs and skip the picker.

Imports ``textual`` at module load, so it is only importable with the optional
``dss-client[tui]`` extra. Strictly read-only — it never mutates state.
"""

from __future__ import annotations

import os
import time
import traceback

from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, Log, Static
from textual.worker import get_current_worker

from dss_client.tui.format import (
    entry_at_level,
    entry_matches,
    format_job_config,
    format_job_row,
    format_log_entry,
    is_active_status,
    job_matches,
    job_status_color,
    sort_jobs,
    source_label,
    wrap_log_line,
)
from dss_client.tui.log_cache import LogCache, cached_log_pages

_ERROR_LOG = "~/.neutrino-tui-errors.log"


class JobListScreen(Screen):
    """Pick a job. Lists jobs + status via the SDK's ``list_jobs``."""

    BINDINGS = [
        ("/", "filter", "Filter"),
        ("escape", "clear_filter", "Clear filter"),
        ("r", "refresh", "Refresh"),
        ("q", "quit", "Quit"),
    ]
    _REFRESH_SECONDS = 20

    def __init__(self, client, *, sub_job_id=None, poll_interval=1.0):
        super().__init__()
        self._client = client
        self._sub_job_id = sub_job_id
        self._poll_interval = poll_interval
        self._job_by_item: dict[str, str] = {}
        self._status_by_item: dict[str, str] = {}
        self._all_jobs: list = []
        self._filter = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield ListView(id="jobs")
        yield Input(placeholder="filter jobs…  (/ focus · Esc clear)", id="jobfilter")
        yield Footer()

    def on_mount(self) -> None:
        self.app.sub_title = "select a job (↑/↓, Enter · / filter · r refresh)"
        self._load_jobs()
        # Gentle auto-refresh so status changes appear without pressing r.
        self.set_interval(self._REFRESH_SECONDS, self._auto_refresh)

    def _auto_refresh(self) -> None:
        # Only refetch while the picker is the active screen — don't poll GS in
        # the background while the user is down in the log view.
        if self.app.screen is self:
            self._load_jobs()

    def action_refresh(self) -> None:
        self._load_jobs()

    def action_filter(self) -> None:
        self.set_focus(self.query_one("#jobfilter", Input))

    def action_clear_filter(self) -> None:
        inp = self.query_one("#jobfilter", Input)
        if self.focused is inp or self._filter:
            inp.value = ""  # fires on_input_changed → clears the filter
            self.set_focus(self.query_one("#jobs", ListView))

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "jobfilter":
            self._filter = event.value.strip()
            await self._render_jobs()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "jobfilter":
            self.set_focus(self.query_one("#jobs", ListView))

    @work(thread=True, exclusive=True, group="jobs")
    def _load_jobs(self) -> None:
        worker = get_current_worker()
        try:
            jobs = self._client.list_jobs()
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.app.call_from_thread(self._on_jobs, None, f"list_jobs failed: {type(exc).__name__}: {exc}")
            return
        if worker.is_cancelled:
            return
        self.app.call_from_thread(self._on_jobs, jobs, None)

    async def _on_jobs(self, jobs, error) -> None:
        if error:
            lv = self.query_one("#jobs", ListView)
            await lv.clear()
            self._all_jobs = []
            self._job_by_item = {}
            self._status_by_item = {}
            lv.append(ListItem(Label(f"[error] {error}")))
            return
        self._all_jobs = sort_jobs(jobs)
        await self._render_jobs()

    async def _render_jobs(self) -> None:
        lv = self.query_one("#jobs", ListView)
        prev = lv.index
        # await clear() so old items (job-0, …) are gone before re-appending
        # ones with the same IDs.
        await lv.clear()
        self._job_by_item = {}
        self._status_by_item = {}
        visible = [j for j in self._all_jobs if job_matches(j, self._filter)]
        if not visible:
            lv.append(ListItem(Label("(no matching jobs)" if self._filter else "(no jobs found)")))
            return
        for i, job in enumerate(visible):
            jid = job.get("job_id") or job.get("id")
            if not jid:
                continue
            item_id = f"job-{i}"
            self._job_by_item[item_id] = jid
            self._status_by_item[item_id] = job.get("status")
            # Tint only the leading status word (subtle); the rest stays default.
            plain = format_job_row(job)
            status_disp = (str(job.get("status", "")).strip() or "?").removeprefix("JOB_STATE_")
            color = job_status_color(job.get("status"))
            if color and plain.startswith(status_disp):
                text = f"[{color}]{escape(status_disp)}[/]{escape(plain[len(status_disp):])}"
            else:
                text = escape(plain)
            lv.append(ListItem(Label(text), id=item_id))
        # Keep the cursor roughly where it was so auto-refresh doesn't jump it.
        if prev is not None and self._job_by_item:
            lv.index = min(prev, len(self._job_by_item) - 1)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        jid = self._job_by_item.get(item_id)
        if jid:
            self.app.push_screen(
                LogScreen(self._client, jid, sub_job_id=self._sub_job_id,
                          poll_interval=self._poll_interval,
                          job_status=self._status_by_item.get(item_id))
            )


class LogScreen(Screen):
    """Source list + live tail for one job. Read-only."""

    BINDINGS = [
        ("b", "app.pop_screen", "Back"),
        ("escape", "dismiss_or_back", "Back/clear"),
        ("/", "filter", "Filter"),
        ("L", "cycle_level", "Level"),
        ("p", "toggle_pause", "Pause"),
        ("s", "save_log", "Save"),
        ("y", "copy_log", "Copy all"),
        ("c", "copy_text", "Copy sel"),
        ("r", "refresh_sources", "Refresh"),
        ("[", "shrink_sources", "Narrower"),
        ("]", "grow_sources", "Wider"),
        ("q", "quit", "Quit"),
    ]

    _SOURCES_MIN = 24
    _SOURCES_MAX = 140
    _SOURCES_STEP = 8
    _LEVEL_CYCLE = [None, "INFO", "WARNING", "ERROR"]

    def __init__(self, client, job_id, *, sub_job_id=None, poll_interval=1.0,
                 job_status=None):
        super().__init__()
        self._client = client
        self._job_id = job_id
        self._sub_job_id = sub_job_id
        self._poll_interval = poll_interval
        self._job_status = job_status
        self._job_reason = None             # failure/cancel reason, when terminal
        self._sources_width = 48
        self._cache = LogCache(job_id)
        self._source_by_item: dict[str, str] = {}
        self._logview: Log | None = None
        self._filter = ""           # grep substring for the log pane
        self._min_level = None      # None | INFO | WARNING | ERROR
        self._paused = False        # auto-scroll paused?
        self._current_source = None  # source currently being tailed
        self._shown_lines: list[str] = []   # unwrapped lines on screen (for re-wrap)
        self._wrap_width = -1               # content width the buffer is wrapped to
        self._last_update = None            # local HH:MM:SS of the last live line

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="summary")
        with Horizontal():
            yield ListView(id="sources")
            # Log (not RichLog): it supports native text selection + copy. It
            # does not soft-wrap, so we pre-wrap each line to the pane width.
            yield Log(id="logview", highlight=False, max_lines=20000)
        yield Input(placeholder="filter logs…  (/ focus · Esc clear)", id="logfilter")
        yield Footer()

    def on_mount(self) -> None:
        self._logview = self.query_one("#logview", Log)
        # We drive scrolling ourselves (scroll to the tail on every write unless
        # paused), so disable the widget's blanket auto-scroll.
        self._logview.auto_scroll = False
        self._apply_sources_width()
        self._update_subtitle()
        self._load_sources()

    def on_unmount(self) -> None:
        # Stop the tail worker when leaving this job (it polls follow=True).
        self.workers.cancel_all()

    def _apply_sources_width(self) -> None:
        try:
            self.query_one("#sources", ListView).styles.width = self._sources_width
        except Exception:  # noqa: BLE001 - not mounted yet
            pass

    def action_grow_sources(self) -> None:
        self._sources_width = min(self._SOURCES_MAX, self._sources_width + self._SOURCES_STEP)
        self._apply_sources_width()
        self.call_after_refresh(self._rewrap)  # log pane width changed

    def action_shrink_sources(self) -> None:
        self._sources_width = max(self._SOURCES_MIN, self._sources_width - self._SOURCES_STEP)
        self._apply_sources_width()
        self.call_after_refresh(self._rewrap)  # log pane width changed

    def on_resize(self, event) -> None:
        # Terminal resize changes the log pane width → re-wrap the buffer.
        self.call_after_refresh(self._rewrap)

    def action_refresh_sources(self) -> None:
        self._load_sources()

    def _update_subtitle(self) -> None:
        # Dynamic status line: everything that changes over the session lives
        # here, leaving the #summary line for static config only.
        bits = [self._job_id]
        st = (self._job_status or "").strip()
        if st:
            bits.append(st.removeprefix("JOB_STATE_"))
        if self._job_reason:
            bits.append(f"reason: {self._job_reason}")
        if self._filter:
            bits.append(f"filter:{self._filter}")
        if self._min_level:
            bits.append(f"≥{self._min_level}")
        if self._paused:
            bits.append("PAUSED")
        if self._last_update:
            bits.append(f"updated {self._last_update}")
        self.app.sub_title = "  ·  ".join(bits)

    def _set_summary(self, text: str) -> None:
        try:
            self.query_one("#summary", Static).update(text)
        except Exception:  # noqa: BLE001 - not mounted yet
            pass

    # ─── filter / level / pause ──────────────────────────────────────────
    def action_filter(self) -> None:
        self.set_focus(self.query_one("#logfilter", Input))

    def action_dismiss_or_back(self) -> None:
        # Esc clears an active filter (and unfocuses the input); otherwise backs out.
        inp = self.query_one("#logfilter", Input)
        if self.focused is inp or self._filter:
            inp.value = ""  # fires on_input_changed → clears the filter
            self.set_focus(self.query_one("#sources", ListView))
        else:
            self.app.pop_screen()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "logfilter":
            self._filter = event.value.strip()
            self._update_subtitle()
            self._refilter()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "logfilter":
            self.set_focus(self.query_one("#sources", ListView))

    def action_cycle_level(self) -> None:
        i = self._LEVEL_CYCLE.index(self._min_level) if self._min_level in self._LEVEL_CYCLE else 0
        self._min_level = self._LEVEL_CYCLE[(i + 1) % len(self._LEVEL_CYCLE)]
        self._update_subtitle()
        self._refilter()

    def action_toggle_pause(self) -> None:
        self._paused = not self._paused
        # Resuming jumps to the end and follows again; pausing just stops
        # following (the user can scroll freely).
        if not self._paused and self._logview is not None:
            self._logview.scroll_end(animate=False)
        self._update_subtitle()

    def _refilter(self) -> None:
        # Re-run the tail so the filter/level apply to the replayed cache too.
        if self._current_source:
            self._start_tail(self._current_source)

    # ─── export / copy ───────────────────────────────────────────────────
    @work(thread=True, exclusive=True, group="export")
    def action_save_log(self) -> None:
        src = self._current_source
        if not src:
            return
        entries = self._cache.load_entries(src)
        text = "\n".join(format_log_entry(e) for e in entries)
        safe = src.replace("/", "_").replace(":", "-")
        path = os.path.expanduser(f"~/neutrino-{self._job_id[:8]}-{safe}.log")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text + ("\n" if text else ""))
            msg = f"[saved] {len(entries)} line(s) → {path}"
        except OSError as exc:  # noqa: BLE001
            msg = f"[error] save failed: {exc}"
        self.app.call_from_thread(self._write_line, msg)

    @work(thread=True, exclusive=True, group="copy")
    def action_copy_log(self) -> None:
        src = self._current_source
        if not src:
            return
        entries = self._cache.load_entries(src)
        text = "\n".join(format_log_entry(e) for e in entries)
        self.app.call_from_thread(self.app.copy_to_clipboard, text)
        self.app.call_from_thread(
            self._write_line, f"[copied] {len(entries)} line(s) to clipboard"
        )

    @work(thread=True, exclusive=True, group="sources")
    def _load_sources(self) -> None:
        worker = get_current_worker()
        # Fetch the job once (GS metadata — survives after the zone is gone) for
        # the summary header, status (drives follow/cache), and the sub-job list.
        try:
            job = self._client.get_job(self._job_id) or {}
        except Exception:  # noqa: BLE001 - best-effort
            job = {}
        if job.get("status"):
            self._job_status = job.get("status")
        if job:
            self._job_reason = job.get("reason")  # clears a stale reason on refresh
            # The summary line is reserved for static config; status/reason go
            # on the subtitle line instead.
            self._post(self._set_summary, format_job_config(job))
        self._post(self._update_subtitle)

        # Each sub-job is a tailable source (its zone-manager / Ray-head pod,
        # whose stdout already includes worker output via Ray log_to_driver).
        sources = []
        for sj in (job.get("sub_jobs") or []):
            if isinstance(sj, dict):
                sid = sj.get("sub_job_id") or sj.get("id")
                if sid:
                    sources.append({"sub_job_id": sid, "job_type": sj.get("job_type") or sj.get("type")})
        if not sources:
            # get_job failed/empty (e.g. offline): fall back to cached sub-jobs.
            sources = [{"sub_job_id": sid} for sid in self._cache.cached_sources()]
        if not sources:
            self._post(
                self._write_line,
                "[no logs] this job has no sub-jobs and no cached logs are available",
            )
        if worker.is_cancelled:
            return
        self._post(self._populate_sources, sources)

    async def _populate_sources(self, sources) -> None:
        lv = self.query_one("#sources", ListView)
        await lv.clear()  # await removal before re-appending src-* ids (see _populate)
        self._source_by_item = {}
        items = [s for s in sources if s.get("sub_job_id")]
        for i, s in enumerate(items):
            sid = s["sub_job_id"]
            item_id = f"src-{i}"
            self._source_by_item[item_id] = sid
            lv.append(ListItem(Label(source_label(s)), id=item_id))
        if items:
            lv.index = 0
            self._start_tail(items[0]["sub_job_id"])

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        sid = self._source_by_item.get(event.item.id or "")
        # Re-clicking the source you're already viewing shouldn't reload/reset
        # the pane (that's what caused the jump-to-bottom on every click).
        if sid and sid != self._current_source:
            self._start_tail(sid)

    def _post(self, fn, *args, **kwargs) -> bool:
        """Run ``fn`` on the UI thread from a worker; return ``False`` instead of
        raising once the app/event loop is gone (teardown), so background workers
        stop cleanly rather than surfacing a NoActiveAppError as a WorkerError.

        Only teardown-class errors (``RuntimeError`` — NoActiveAppError and
        "event loop is closed" both derive from it) are swallowed; a real bug in
        ``fn`` (Attribute/Type/Key/…) still propagates to ``_tail``'s handler so
        it gets logged rather than silently stopping the tail."""
        try:
            self.app.call_from_thread(fn, *args, **kwargs)
            return True
        except RuntimeError:
            return False

    def _start_tail(self, sub_job_id: str) -> None:
        self._current_source = sub_job_id
        self._shown_lines = []
        # A fresh tail context (new source, or a re-tail after a filter change)
        # starts following the tail — otherwise a pause left over from a prior
        # source would strand this one at the top, the very bug we're fixing.
        self._paused = False
        self._update_subtitle()
        if self._logview is not None:
            self._logview.clear()
        self._write_line(f"— tailing {sub_job_id} —")
        self._tail(sub_job_id)

    @work(thread=True, exclusive=True, group="tail")
    def _tail(self, sub_job_id: str) -> None:
        worker = get_current_worker()
        # A terminal job's zone is gone, so the operation API would error: serve
        # cache only (no live fetch). A running/unknown job follows live.
        active = is_active_status(self._job_status)
        try:
            # Replay the local cache instantly, then (only for a live job) resume
            # the tail from the saved cursor so only new lines are fetched. Each
            # yield is a whole page (or the cache batch); render it in one shot
            # so high-velocity logs don't saturate the UI thread per line.
            saw_live = False
            for kind, entries in cached_log_pages(
                self._cache, self._client, self._job_id, sub_job_id,
                follow=active, live=active,
                poll_interval=self._poll_interval, replay_limit=2000,
                is_cancelled=lambda: worker.is_cancelled,
            ):
                if worker.is_cancelled:
                    return
                # Apply the grep filter + min-level to the whole batch.
                lines = [
                    format_log_entry(e) for e in entries
                    if entry_matches(e, self._filter) and entry_at_level(e, self._min_level)
                ]
                if not lines:
                    continue
                if kind == "live" and not saw_live:
                    saw_live = True
                    lines = ["— live —"] + lines
                # Every batch sticks to the tail (unless paused), so the view
                # opens on the newest cached lines and follows live from there.
                # _post fails closed if the app is tearing down, so the worker
                # stops cleanly.
                if not self._post(self._write_lines, lines):
                    return
                if kind == "live":
                    # Stamp the freshness indicator on each live batch.
                    self._last_update = time.strftime("%H:%M:%S")
                    if not self._post(self._update_subtitle):
                        return
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            if not worker.is_cancelled:
                try:
                    with open(os.path.expanduser(_ERROR_LOG), "a", encoding="utf-8") as f:
                        f.write(f"tail {sub_job_id} failed:\n{traceback.format_exc()}\n")
                except Exception:  # noqa: BLE001
                    pass
                self._post(
                    self._write_line,
                    f"[error] tail {sub_job_id} failed: {type(exc).__name__}: {exc}"
                    f"  (full traceback: {_ERROR_LOG})",
                )

    def _content_width(self) -> int:
        """Usable text columns of the log pane (accounts for padding/scrollbar)."""
        lv = self._logview
        if lv is None:
            return 0
        try:
            w = lv.scrollable_content_region.width
        except Exception:  # noqa: BLE001 - not laid out yet
            w = 0
        if w <= 0:
            w = getattr(lv.size, "width", 0)
        return max(0, w)

    def _write_line(self, line: str) -> None:
        self._write_lines([line])

    def _write_lines(self, lines: list[str]) -> None:
        lv = self._logview
        if lv is None or not lines:
            return
        self._shown_lines.extend(lines)
        # Bound the re-wrap buffer to the widget's own max_lines budget.
        if len(self._shown_lines) > 20000:
            del self._shown_lines[: len(self._shown_lines) - 20000]
        # Always stick to the tail so every viewer sees the newest lines on
        # open and as they stream — cache replay and live alike. Pause is the
        # one escape hatch: it stops following so the user can scroll back.
        follow = not self._paused
        w = self._content_width()
        wrapped: list[str] = []
        for line in lines:
            wrapped.extend(wrap_log_line(line, w))
        # One write_lines call per page keeps the UI responsive under
        # high-velocity logs (vs. a cross-thread hop + write per line).
        lv.write_lines(wrapped, scroll_end=follow)

    def _rewrap(self) -> None:
        """Re-wrap the on-screen buffer after the pane width changes."""
        lv = self._logview
        if lv is None:
            return
        w = self._content_width()
        if w <= 0 or w == self._wrap_width:
            return
        self._wrap_width = w
        lv.clear()
        for line in self._shown_lines:
            for piece in wrap_log_line(line, w):
                lv.write_line(piece)
        # Re-wrapping clears + repaints (resetting scroll to the top), so snap
        # back to the tail unless the user has paused to read history.
        if not self._paused:
            lv.scroll_end(animate=False)


class NeutrinoLogTUI(App):
    """Entry app: show the job picker, or jump straight to a job's logs when a
    job_id is supplied."""

    CSS = """
    #jobs { padding: 0 1; }
    #sources { width: 48; border-right: solid $panel; }
    #logview { padding: 0 1; }
    #summary { color: $text-muted; padding: 0 1; }
    #logfilter { border: tall $panel; }
    #jobfilter { border: tall $panel; }
    /* Text-selection highlight: reuse the picker's subtle "blurred cursor"
       shade so it's visible but not a heavy saturated block. */
    LogScreen > .screen--selection {
        background: $block-cursor-blurred-background;
        color: $block-cursor-blurred-foreground;
    }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, client, job_id=None, *, sub_job_id=None, poll_interval=1.0):
        super().__init__()
        self._client = client
        self._job_id = job_id
        self._sub_job_id = sub_job_id
        self._poll_interval = poll_interval
        self.title = "neutrino logs"

    def on_mount(self) -> None:
        if self._job_id:
            self.push_screen(LogScreen(
                self._client, self._job_id, sub_job_id=self._sub_job_id,
                poll_interval=self._poll_interval))
        else:
            self.push_screen(JobListScreen(
                self._client, sub_job_id=self._sub_job_id,
                poll_interval=self._poll_interval))
