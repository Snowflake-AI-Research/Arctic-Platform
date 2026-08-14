"""Pure formatting helpers for the read-only log/event TUI.

Kept free of any ``textual`` import so they can be unit-tested without the
optional TUI dependency installed.
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from typing import Any

EVENTS_SOURCE_ID = "__events__"
EVENTS_LABEL = "▸ scheduling events"


def format_log_entry(entry: Any) -> str:
    """Render one log entry (a parsed JSONL object or ``{"_raw": ...}``)."""
    if not isinstance(entry, dict):
        return str(entry)
    if "_raw" in entry:
        text = str(entry["_raw"])
        return text + " …[truncated]" if entry.get("_truncated") else text
    ts = str(entry.get("ts", "")).strip()
    level = str(entry.get("level", "")).strip()
    logger = str(entry.get("logger", "")).strip()
    msg = entry.get("msg")
    if msg is None:
        msg = entry.get("message")
    if msg is None:
        # Unknown shape: show the raw JSON so nothing is silently dropped.
        msg = json.dumps(entry, sort_keys=True)
    parts = []
    if ts:
        parts.append(ts)
    if level:
        parts.append(f"{level:<5}")
    if logger:
        parts.append(f"{logger}:")
    parts.append(str(msg))
    line = " ".join(parts)
    if entry.get("_truncated"):
        line += " …[truncated]"
    return line


def wrap_log_line(text: Any, width: int) -> list[str]:
    """Soft-wrap one log line to ``width`` columns for display.

    The ``Log`` widget (unlike ``RichLog``) does not wrap, so we pre-wrap. Word
    boundaries are preferred but long tokens (URLs, JSON blobs) are hard-broken
    so nothing overflows. Embedded newlines split into separate lines first, and
    leading whitespace (indentation) is preserved. ``width <= 0`` (not laid out
    yet) returns the line unwrapped.
    """
    out: list[str] = []
    for logical in str(text).split("\n"):
        if width <= 0 or len(logical) <= width:
            out.append(logical)
            continue
        pieces = textwrap.wrap(
            logical,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        out.extend(pieces or [""])
    return out or [""]


def format_event(ev: Any) -> str:
    """Render one ZMD scheduling/zone event."""
    if not isinstance(ev, dict):
        return str(ev)
    parts = []
    ts = str(ev.get("ts", "")).strip()
    if ts:
        parts.append(ts)
    parts.append(str(ev.get("kind", "event")))
    if zone := ev.get("zone"):
        parts.append(f"zone={zone}")
    frm, to = ev.get("from"), ev.get("to")
    if frm or to:
        parts.append(f"{frm or '?'}→{to or '?'}")
    if sub := ev.get("sub_job_id"):
        parts.append(f"sub_job={sub}")
    if detail := ev.get("detail"):
        parts.append(str(detail))
    return " ".join(parts)


def source_label(source: Any) -> str:
    """Side-panel label for a sub-job source.

    A source is a sub-job: ``{"sub_job_id", "job_type"}`` (or just a sub_job_id
    string). Rendered as ``"<type> #<index>"`` — the type comes from job_type
    when present, else the middle field of the ``<uuid>:<type>:<index>`` id.
    """
    sjid = ""
    job_type = None
    if isinstance(source, dict):
        sjid = str(source.get("sub_job_id") or source.get("id") or source.get("source_id") or "")
        job_type = source.get("job_type") or source.get("type")
    else:
        sjid = str(source)
    parts = sjid.split(":")
    if not job_type and len(parts) >= 2:
        job_type = parts[1]  # <uuid>:<type>:<index>
    idx = parts[2] if len(parts) >= 3 else ""
    if job_type:
        return f"{job_type} #{idx}" if idx != "" else str(job_type)
    if isinstance(source, dict) and source.get("label"):
        return str(source["label"])
    return sjid or str(source)


_ACTIVE_STATUSES = {
    "running", "initializing", "pending", "placing", "queued", "creating",
}

_TERMINAL_STATUSES = {
    "failed", "cancelled", "canceled", "terminated", "completed", "succeeded",
    "done", "error",
}


def is_active_status(status: Any) -> bool:
    """Whether a job is still live, i.e. its log tail should keep following.

    Missing/unknown statuses default to active so a running-but-unclassified job
    never stops tailing; only a recognized terminal status stops the follow.
    """
    if not status:
        return True
    s = str(status).lower().removeprefix("job_state_")
    if s in _ACTIVE_STATUSES:
        return True
    if s in _TERMINAL_STATUSES:
        return False
    return True


def format_job_row(job: Any) -> str:
    """One job-picker row with │-separated columns: status │ created │ id │ types."""
    if not isinstance(job, dict):
        return str(job)
    jid = job.get("job_id") or job.get("id") or "?"
    status = (str(job.get("status", "")).strip() or "?")
    status = status.removeprefix("JOB_STATE_")
    created = format_created_at(job.get("created_at")) or "—"
    # Fixed-width columns + separators so fields stay visually distinct.
    row = f"{status:<10} │ {created:<16} │ {jid}"
    subs = job.get("sub_jobs") or []
    types = [s.get("job_type") or s.get("type") for s in subs if isinstance(s, dict)]
    types = [t for t in types if t]
    if not types and job.get("job_type"):
        types = [job["job_type"]]
    if types:
        row += f" │ {', '.join(map(str, types))}"
    return row


def job_status_color(status: Any) -> str:
    """Subtle accent for a job's *status word only* (empty = default color).

    Deliberately minimal: a gentle green marks live jobs so you can spot the one
    you're running; every other status (failed/cancelled/terminated/unknown)
    keeps the default foreground — no alarm colors.
    """
    s = str(status or "").strip().lower().removeprefix("job_state_")
    if s in _ACTIVE_STATUSES:
        return "green"
    return ""


def format_created_at(raw: Any) -> str:
    """Render an ISO-8601 created_at (e.g. ``2026-06-13T18:49:54Z``) as local
    ``YYYY-MM-DD HH:MM``. Returns ``""`` for missing/unparseable values."""
    if not raw:
        return ""
    s = str(raw).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime("%Y-%m-%d %H:%M")


def _created_epoch(raw: Any) -> float:
    """Parse a created_at into epoch seconds for sorting. Missing/unparseable
    values sort oldest (``-inf``)."""
    if not raw:
        return float("-inf")
    s = str(raw).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return float("-inf")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def sort_jobs(jobs: Any) -> list:
    """Order jobs for the picker newest-first by created_at. Ties keep their
    original order (stable); jobs without a parseable created_at sort last."""
    if not isinstance(jobs, list):
        return []
    return sorted(
        [j for j in jobs if isinstance(j, dict)],
        key=lambda job: _created_epoch(job.get("created_at")),
        reverse=True,
    )


# ─── Job summary (whitelisted create-job params) ─────────────────────────────

# Only the parameters a customer passes to SubJobConfig.training_job /
# sampling_job are surfaced — short label + the config key it reads.
_SUMMARY_CONFIG_KEYS = [
    ("n_gpus", "gpus"),
    ("max_seq_len", "seq"),
    ("train_batch_size", "bs"),
    ("gradient_clipping", "clip"),
]
_SUMMARY_SUBJOB_KEYS = [
    ("global_batch_size", "gbs"),
    ("dtype", "dtype"),
    ("seed", "seed"),
]


def _num(v: Any) -> str:
    """Render a value compactly: integral floats lose the .0 (n_gpus 1.0 -> 1)."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _subjob_config(sj: dict) -> dict:
    for key in ("training_config", "inference_config", "sampling_config"):
        cfg = sj.get(key)
        if isinstance(cfg, dict):
            return cfg
    return {}


def _job_config_segments(job: dict) -> list:
    """Per-sub-job static config segments (type / model / whitelisted params)."""
    segs = []
    for sj in (job.get("sub_jobs") or []):
        if not isinstance(sj, dict):
            continue
        seg = []
        jt = str(sj.get("job_type") or sj.get("type") or "").lower()
        if jt:
            seg.append(jt)
        if sj.get("model_name"):
            seg.append(str(sj["model_name"]))
        cfg = _subjob_config(sj)
        for key, label in _SUMMARY_CONFIG_KEYS:
            if cfg.get(key) is not None:
                seg.append(f"{label}={_num(cfg[key])}")
        opt = cfg.get("optimizer")
        if isinstance(opt, dict):
            o = [str(opt["name"])] if opt.get("name") else []
            if opt.get("lr") is not None:
                o.append(f"lr={_num(opt['lr'])}")
            if o:
                seg.append("opt=" + "/".join(o))
        for key, label in _SUMMARY_SUBJOB_KEYS:
            if sj.get(key) is not None:
                seg.append(f"{label}={_num(sj[key])}")
        if seg:
            segs.append(" ".join(seg))
    return segs


def format_job_config(job: Any) -> str:
    """Static create-time config only (no status / reason) — for the line we
    reserve to describe the job. Shows only whitelisted params that are present,
    so it stays compact."""
    if not isinstance(job, dict):
        return ""
    return "  ·  ".join(_job_config_segments(job))


def format_job_summary(job: Any) -> str:
    """One-line summary of a job's create-time config (whitelisted) + status.

    Shows only the parameters the customer passed to create_job that are
    present, so it stays compact. Failure/cancel ``reason`` is appended.
    """
    if not isinstance(job, dict):
        return ""
    status = (str(job.get("status", "")).strip() or "?").removeprefix("JOB_STATE_")
    parts = [status, *_job_config_segments(job)]
    if job.get("reason"):
        parts.append(f"reason: {job['reason']}")
    return "  ·  ".join(p for p in parts if p)


# ─── Search / filter helpers ─────────────────────────────────────────────────

def job_matches(job: Any, query: str) -> bool:
    """Case-insensitive match of query against a job's id, status, sub-job
    types and model names. Empty query matches everything."""
    if not query:
        return True
    if not isinstance(job, dict):
        return False
    hay = [str(job.get("job_id") or job.get("id") or ""), str(job.get("status") or "")]
    for sj in (job.get("sub_jobs") or []):
        if isinstance(sj, dict):
            hay.append(str(sj.get("job_type") or sj.get("type") or ""))
            hay.append(str(sj.get("model_name") or ""))
    return query.lower() in " ".join(hay).lower()


def entry_matches(entry: Any, query: str) -> bool:
    """Whether a log entry's rendered text contains query (case-insensitive)."""
    if not query:
        return True
    return query.lower() in format_log_entry(entry).lower()


_LEVELS = {
    "DEBUG": 10, "INFO": 20, "WARNING": 30, "WARN": 30,
    "ERROR": 40, "CRITICAL": 50, "FATAL": 50,
}


def entry_at_level(entry: Any, min_level: Any) -> bool:
    """Whether a structured entry is at or above min_level. Entries without a
    parseable level (raw stdout lines) always pass, so the filter only trims
    lower-severity *structured* logs and never hides raw output."""
    if not min_level:
        return True
    floor = _LEVELS.get(str(min_level).upper())
    if floor is None:
        return True
    lvl = entry.get("level") if isinstance(entry, dict) else None
    if not lvl:
        return True
    return _LEVELS.get(str(lvl).upper(), 0) >= floor
