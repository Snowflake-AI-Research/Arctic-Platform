"""Unit tests for the TUI's pure formatting helpers (no ``textual`` needed)."""

from dss_client.tui.format import (
    entry_at_level,
    entry_matches,
    format_created_at,
    format_event,
    format_job_config,
    format_job_row,
    format_job_summary,
    format_log_entry,
    job_matches,
    job_status_color,
    sort_jobs,
    source_label,
    wrap_log_line,
)


def test_format_structured_log_entry():
    line = format_log_entry(
        {"ts": "2026-06-12T00:00:01Z", "level": "INFO", "logger": "dss.zones", "msg": "hello"}
    )
    assert "2026-06-12T00:00:01Z" in line
    assert "INFO" in line
    assert "dss.zones:" in line
    assert line.endswith("hello")


def test_format_raw_log_entry():
    assert format_log_entry({"_raw": "plain text"}) == "plain text"


def test_format_truncated_entry_marks_suffix():
    out = format_log_entry({"_raw": "big line", "_truncated": True})
    assert out.endswith("[truncated]")


def test_format_entry_missing_msg_falls_back_to_json():
    out = format_log_entry({"foo": "bar"})
    assert "foo" in out and "bar" in out  # nothing silently dropped


def test_format_non_dict_entry():
    assert format_log_entry("already a string") == "already a string"


def test_format_event_full():
    out = format_event(
        {
            "ts": "2026-06-12T00:01:00Z",
            "kind": "zone_transition",
            "zone": "training-1",
            "from": "creating",
            "to": "ready",
            "sub_job_id": "s:training:0",
            "detail": "zone_manager=zm-1",
        }
    )
    assert "zone_transition" in out
    assert "zone=training-1" in out
    assert "creating→ready" in out
    assert "sub_job=s:training:0" in out
    assert "zone_manager=zm-1" in out


def test_format_event_minimal():
    assert format_event({"kind": "route_added"}) == "route_added"


def test_source_label_names_the_component():
    # Sources are sub-jobs: "<type> #<index>", type from job_type or the id.
    assert source_label({"sub_job_id": "u:training:0", "job_type": "training"}) == "training #0"
    assert source_label({"sub_job_id": "u:sampling:1", "job_type": "sampling"}) == "sampling #1"
    assert source_label({"sub_job_id": "u:training:2"}) == "training #2"  # type from id
    assert source_label("u:training:0") == "training #0"  # bare string id


def test_source_label_falls_back_to_server_label_then_id():
    assert source_label({"source_id": "weird", "label": "Custom"}) == "Custom"
    assert source_label({"source_id": "weird"}) == "weird"


def test_format_job_row():
    row = format_job_row({"job_id": "abc-123", "status": "RUNNING",
                          "created_at": "2026-06-13T18:49:54Z",
                          "sub_jobs": [{"job_type": "training"}, {"job_type": "sampling"}]})
    assert "RUNNING" in row and "abc-123" in row and "training" in row and "sampling" in row
    assert "2026-06-1" in row  # created date shown (tz may shift the day by ±1)
    assert "│" in row  # column separators present
    # JOB_STATE_ prefix stripped; type shown; missing created → placeholder column
    no_ts = format_job_row({"id": "j", "status": "JOB_STATE_FAILED", "job_type": "training"})
    assert no_ts.startswith("FAILED") and "training" in no_ts and "│" in no_ts and "—" in no_ts
    assert format_job_row({"job_id": "j"}).startswith("?")  # missing status


def test_job_status_color():
    # Subtle: only live jobs get a hue (green); everything else stays default.
    assert job_status_color("RUNNING") == "green"
    assert job_status_color("JOB_STATE_PENDING") == "green"
    assert job_status_color("FAILED") == ""
    assert job_status_color("CANCELLED") == ""
    assert job_status_color("TERMINATED") == ""
    assert job_status_color(None) == ""
    assert job_status_color("weird") == ""


def test_format_created_at():
    import re
    out = format_created_at("2026-06-13T18:49:54Z")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", out)  # local YYYY-MM-DD HH:MM
    assert format_created_at(None) == ""
    assert format_created_at("") == ""
    assert format_created_at("not-a-date") == ""


def test_sort_jobs_newest_first():
    jobs = [
        {"job_id": "old", "created_at": "2026-06-10T00:00:00Z"},
        {"job_id": "new", "created_at": "2026-06-13T00:00:00Z"},
        {"job_id": "mid", "created_at": "2026-06-11T00:00:00Z"},
        {"job_id": "none"},  # missing created_at -> sorts last
    ]
    order = [j["job_id"] for j in sort_jobs(jobs)]
    assert order == ["new", "mid", "old", "none"]
    assert sort_jobs("not a list") == []


def test_sort_jobs_stable_on_equal_timestamps():
    jobs = [
        {"job_id": "first", "created_at": "2026-06-12T00:00:00Z"},
        {"job_id": "second", "created_at": "2026-06-12T00:00:00Z"},
    ]
    # equal created_at keeps original order (stable)
    assert [j["job_id"] for j in sort_jobs(jobs)] == ["first", "second"]


def test_format_job_summary():
    job = {"status": "RUNNING", "sub_jobs": [{
        "job_type": "TRAINING", "model_name": "Qwen/Qwen3-0.6B",
        "training_config": {"n_gpus": 1.0, "max_seq_len": 128.0, "train_batch_size": 1.0,
                            "optimizer": {"name": "adamw", "lr": 1e-5}},
    }]}
    s = format_job_summary(job)
    assert s.startswith("RUNNING")
    assert "training" in s and "Qwen/Qwen3-0.6B" in s
    assert "gpus=1" in s and "seq=128" in s and "bs=1" in s
    assert "opt=adamw/lr=1e-05" in s
    assert "gbs=" not in s and "clip=" not in s  # only what was passed
    # failure reason is appended
    failed = {"status": "FAILED", "reason": "oom killed", "sub_jobs": []}
    assert format_job_summary(failed).endswith("reason: oom killed")
    assert format_job_summary("not a dict") == ""


def test_format_job_config_is_config_only():
    # The reserved config line shows static params only — no status, no reason.
    job = {"status": "FAILED", "reason": "oom killed", "sub_jobs": [{
        "job_type": "TRAINING", "model_name": "Qwen/Qwen3-0.6B",
        "training_config": {"n_gpus": 1.0, "max_seq_len": 128.0, "train_batch_size": 1.0,
                            "optimizer": {"name": "adamw", "lr": 1e-5}},
    }]}
    s = format_job_config(job)
    assert "training" in s and "Qwen/Qwen3-0.6B" in s
    assert "gpus=1" in s and "seq=128" in s and "opt=adamw/lr=1e-05" in s
    assert "FAILED" not in s and "reason" not in s  # status/reason live on the subtitle
    assert format_job_config("not a dict") == ""


def test_job_matches():
    job = {"job_id": "abc-123", "status": "RUNNING",
           "sub_jobs": [{"job_type": "training", "model_name": "Qwen3"}]}
    assert job_matches(job, "") is True
    assert job_matches(job, "abc") and job_matches(job, "run")
    assert job_matches(job, "qwen") and job_matches(job, "training")
    assert not job_matches(job, "sampling")
    assert job_matches("not a dict", "q") is False


def test_entry_matches():
    assert entry_matches({"_raw": "loss=0.5 step=3"}, "loss")
    assert entry_matches({"msg": "hello", "level": "INFO"}, "hello")
    assert not entry_matches({"_raw": "x"}, "loss")
    assert entry_matches({"_raw": "x"}, "") is True


def test_entry_at_level():
    assert entry_at_level({"level": "ERROR"}, "WARNING")
    assert not entry_at_level({"level": "INFO"}, "WARNING")
    assert entry_at_level({"_raw": "x"}, "ERROR")     # unleveled lines always pass
    assert entry_at_level({"level": "INFO"}, None)    # no floor → all pass
    assert entry_at_level({"level": "INFO"}, "BOGUS")  # bad floor → all pass


def test_wrap_log_line_short_passes_through():
    assert wrap_log_line("hello world", 80) == ["hello world"]


def test_wrap_log_line_zero_or_negative_width_unwrapped():
    # Not laid out yet (width 0/neg): never wrap, never drop the line.
    assert wrap_log_line("a long line of text", 0) == ["a long line of text"]
    assert wrap_log_line("x" * 50, -1) == ["x" * 50]


def test_wrap_log_line_hard_breaks_long_token_without_loss():
    token = "abcdefghij" * 5  # 50 chars, no spaces (e.g. a URL or hash)
    pieces = wrap_log_line(token, 10)
    assert all(len(p) <= 10 for p in pieces)
    assert "".join(pieces) == token  # hard break loses nothing
    assert len(pieces) == 5


def test_wrap_log_line_word_wraps_within_width():
    pieces = wrap_log_line("the quick brown fox jumps over", 10)
    assert all(len(p) <= 10 for p in pieces)
    assert len(pieces) >= 3
    # every non-space character is preserved (only break points differ)
    assert "".join(pieces).replace(" ", "") == "thequickbrownfoxjumpsover"


def test_wrap_log_line_splits_embedded_newlines():
    assert wrap_log_line("a\nb\nc", 80) == ["a", "b", "c"]


def test_wrap_log_line_empty_is_one_blank_line():
    assert wrap_log_line("", 80) == [""]
    assert wrap_log_line("\n", 80) == ["", ""]  # blank lines preserved

