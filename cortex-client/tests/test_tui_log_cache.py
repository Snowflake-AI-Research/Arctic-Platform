"""Unit tests for the TUI log cache (no ``textual`` dependency)."""

import json
from pathlib import Path

import pytest

from dss_client.tui.log_cache import (
    LogCache,
    cache_root,
    cached_log_pages,
    cached_pages,
    safe_source_name,
    unsafe_source_name,
)


# ─── cache dir resolution ────────────────────────────────────────────────

def test_cache_root_respects_xdg_cache_home(monkeypatch, tmp_path):
    monkeypatch.delenv("NEUTRINO_TUI_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert cache_root() == tmp_path / "neutrino-tui"


def test_cache_root_falls_back_to_home_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("NEUTRINO_TUI_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert cache_root() == tmp_path / ".cache" / "neutrino-tui"


def test_override_env_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("NEUTRINO_TUI_CACHE_DIR", str(tmp_path / "override"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_root() == tmp_path / "override"


# ─── source-id <-> filename encoding ─────────────────────────────────────

@pytest.mark.parametrize("sid", [
    "head:server",
    "head:job:42",
    "pod:dz-abc-t0-zone-manager-x/zone-manager",
    "node:host-1:zone.log",
    "unícode:π/x",
    "has%percent",
])
def test_safe_source_name_roundtrip(sid):
    enc = safe_source_name(sid)
    assert "/" not in enc and ":" not in enc  # single path component
    assert unsafe_source_name(enc) == sid


def test_safe_source_name_handles_dot_components():
    assert safe_source_name(".") not in (".", "..")
    assert safe_source_name("..") not in (".", "..")


# ─── entries: append / load / limit / torn line ──────────────────────────

def _cache(tmp_path, **kw):
    return LogCache("job-123", root=tmp_path, **kw)


def test_append_and_load_roundtrip(tmp_path):
    c = _cache(tmp_path)
    entries = [{"ts": "t1", "msg": "a"}, {"msg": "b"}]
    assert c.append_entries("head:server", entries) == 2
    assert c.load_entries("head:server") == entries


def test_load_entries_limit_keeps_last_n(tmp_path):
    c = _cache(tmp_path)
    c.append_entries("s", [{"i": i} for i in range(10)])
    assert c.load_entries("s", limit=3) == [{"i": 7}, {"i": 8}, {"i": 9}]


def test_cached_sources_lists_sources_with_data(tmp_path):
    c = _cache(tmp_path)
    assert c.cached_sources() == []  # nothing cached yet
    c.append_entries("pod:dz-x-t0-zone-manager-a/zone-manager", [{"n": 1}])
    c.append_entries("head:job:abc:training:0", [{"n": 2}])
    got = c.cached_sources()
    assert "pod:dz-x-t0-zone-manager-a/zone-manager" in got  # round-trips through the filename
    assert "head:job:abc:training:0" in got


def test_load_tolerates_torn_last_line(tmp_path):
    c = _cache(tmp_path)
    c.append_entries("s", [{"msg": "ok1"}, {"msg": "ok2"}])
    # Simulate a crash mid-append: a partial trailing line with no newline.
    with open(c._jsonl_path("s"), "a", encoding="utf-8") as f:
        f.write('{"msg": "trunc')
    assert c.load_entries("s") == [{"msg": "ok1"}, {"msg": "ok2"}]


def test_append_falls_back_for_unserializable(tmp_path):
    c = _cache(tmp_path)
    c.append_entries("s", [{"bad": {1, 2, 3}}])  # set is not JSON-serializable
    loaded = c.load_entries("s")
    assert len(loaded) == 1 and "_raw" in loaded[0]


# ─── cursors: persistence, atomicity, corruption tolerance ───────────────

def test_cursor_persists_across_instances(tmp_path):
    _cache(tmp_path).set_cursor("s", "c1")
    assert _cache(tmp_path).get_cursor("s") == "c1"


def test_cursor_overwrite_and_no_temp_leftovers(tmp_path):
    c = _cache(tmp_path)
    c.set_cursor("s", "c1")
    c.set_cursor("s", "c2")
    assert c.get_cursor("s") == "c2"
    leftovers = list(c._cursors_path().parent.glob(".tmp-*"))
    assert leftovers == []


def test_get_cursor_missing_and_corrupt(tmp_path):
    c = _cache(tmp_path)
    assert c.get_cursor("nope") is None
    c._cursors_path().parent.mkdir(parents=True, exist_ok=True)
    c._cursors_path().write_text("{not json", encoding="utf-8")
    assert c.get_cursor("s") is None  # tolerated, no raise


# ─── rotation: bounded history, cursor preserved ─────────────────────────

def test_rotation_trims_oldest_keeps_cursor(tmp_path):
    c = _cache(tmp_path, max_bytes=2000, max_lines=20)
    c.set_cursor("s", "CUR")
    c.append_entries("s", [{"i": i, "pad": "x" * 100} for i in range(200)])
    loaded = c.load_entries("s")
    assert len(loaded) <= 20                      # line cap enforced
    assert loaded[-1]["i"] == 199                 # newest survives
    assert loaded[0]["i"] > 0                     # oldest dropped
    assert c._jsonl_path("s").stat().st_size <= 2000
    assert c.get_cursor("s") == "CUR"             # cursor untouched by rotation


# ─── cached_pages: replay then resume ────────────────────────────────────

class _FakeClient:
    """tail_logs driven by a {cursor -> page} table, with a call counter."""

    def __init__(self, pages):
        self._pages = pages
        self.calls = 0

    def tail_logs(self, job_id, source_id, *, cursor=None, sub_job_id=None):
        self.calls += 1
        return self._pages[cursor]


def _drive(cache, sid, client, *, follow, **kw):
    def fetch(cur):
        return client.tail_logs("j", sid, cursor=cur)
    return list(cached_pages(
        cache, sid, fetch, entries_key="entries", follow=follow,
        poll_interval=0.0, sleep=lambda *_: None, **kw,
    ))


def test_cached_pages_replays_then_resumes(tmp_path):
    c = _cache(tmp_path)
    c.append_entries("s", [{"n": 1}, {"n": 2}, {"n": 3}])
    c.set_cursor("s", "C0")
    client = _FakeClient({
        "C0": {"entries": [{"n": 4}, {"n": 5}], "next_cursor": "C1", "eof": False},
        "C1": {"entries": [], "next_cursor": "C1", "eof": True},
    })
    out = _drive(c, "s", client, follow=False)
    # Entries are yielded a whole batch at a time (cache replay, then per page).
    assert out == [
        ("cache", [{"n": 1}, {"n": 2}, {"n": 3}]),
        ("live", [{"n": 4}, {"n": 5}]),
    ]
    # New entries appended + cursor advanced and persisted.
    assert c.load_entries("s") == [{"n": i} for i in range(1, 6)]
    assert _cache(tmp_path).get_cursor("s") == "C1"


def test_cached_pages_terminal_stops_after_drain(tmp_path):
    c = _cache(tmp_path)
    client = _FakeClient({None: {"entries": [], "next_cursor": None, "eof": True}})
    out = _drive(c, "s", client, follow=False)
    assert out == []
    assert client.calls == 1  # one drain fetch, then stop (eof + not follow)


def test_cached_pages_follow_keeps_polling_past_eof(tmp_path):
    c = _cache(tmp_path)
    client = _FakeClient({None: {"entries": [], "next_cursor": None, "eof": True}})
    # Cancel after a few polls so a following tail doesn't loop forever.
    state = {"n": 0}

    def cancelled():
        state["n"] += 1
        return state["n"] > 5

    out = _drive(c, "s", client, follow=True, is_cancelled=cancelled)
    assert out == []
    assert client.calls >= 2  # kept polling despite eof (follow=True)


def test_cached_pages_cancel_during_replay(tmp_path):
    c = _cache(tmp_path)
    c.append_entries("s", [{"n": i} for i in range(5)])
    client = _FakeClient({})  # must never be called
    out = _drive(c, "s", client, follow=False, is_cancelled=lambda: True)
    assert out == []
    assert client.calls == 0


def test_cached_pages_live_false_is_cache_only(tmp_path):
    """Terminal job: replay cache and never call fetch (the op API would 500)."""
    c = _cache(tmp_path)
    c.append_entries("s", [{"n": 1}, {"n": 2}])
    client = _FakeClient({})  # must never be called
    out = list(cached_pages(
        c, "s", lambda cur: client.tail_logs("j", "s", cursor=cur),
        entries_key="entries", follow=False, live=False,
        poll_interval=0.0, sleep=lambda *_: None,
    ))
    assert out == [("cache", [{"n": 1}, {"n": 2}])]
    assert client.calls == 0


def test_cached_log_pages_tails_recent_on_first_open(tmp_path):
    """First open (no cached cursor) tails the last N via max_lines instead of
    streaming the whole pod log; the resume fetch sends no max_lines."""
    c = _cache(tmp_path)
    calls = []

    class Client:
        def tail_logs(self, job_id, *, cursor=None, max_lines=None,
                      sub_job_id=None, sub_job_type=None):
            calls.append({"cursor": cursor, "max_lines": max_lines})
            if cursor is None:
                return {"entries": [{"n": 1}], "next_cursor": "C1", "eof": False}
            return {"entries": [], "next_cursor": "C1", "eof": True}

    out = list(cached_log_pages(
        c, Client(), "j", "sub-1",
        follow=False, poll_interval=0.0, tail_lines=500,
    ))
    assert out == [("live", [{"n": 1}])]
    assert calls[0] == {"cursor": None, "max_lines": 500}   # first: tail recent
    assert calls[1] == {"cursor": "C1", "max_lines": None}  # resume: delta only


def test_cached_pages_throttles_active_polling(tmp_path):
    """Even a busy source (entries every page) must wait poll_interval between
    requests — reliability over freshness."""
    c = _cache(tmp_path)
    client = _FakeClient({
        None: {"entries": [{"n": 1}], "next_cursor": "C1", "eof": False},
        "C1": {"entries": [{"n": 2}], "next_cursor": "C2", "eof": False},
        "C2": {"entries": [], "next_cursor": "C2", "eof": False},  # caught up
    })
    sleeps = []

    def cancelled():
        return len(sleeps) >= 3  # stop after a few iterations

    list(cached_pages(
        c, "s", lambda cur: client.tail_logs("j", "s", cursor=cur),
        entries_key="entries", follow=True, poll_interval=2.0,
        sleep=lambda d: sleeps.append(d), is_cancelled=cancelled,
    ))
    # First two sleeps follow entries pages (active branch) — proves the floor
    # applies when busy, not just when idle.
    assert sleeps[0] == 2.0 and sleeps[1] == 2.0
