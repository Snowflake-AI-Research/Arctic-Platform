"""Unit tests for materializing a published skeleton from the node's mirror.

``semi_persistence/materialize.py`` imports only the standard library, so it is loaded
from its file (by ``test_image_cache_key``, under the name the engine imports)
and driven against a temporary mirror. Compressed skeletons have their own
tests in ``test_semip_compress.py``; these cover the raw copy every skeleton
goes through.

Run from the package directory::

    cd arctic_platform/inference/semi_persistence
    python -m pytest tests/test_materialize.py -v
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_image_cache_key import (  # noqa: E402
    _fake_published, _fresh_model_dir, _weight_hash_of, _weight_root, se, mat)

_BIG = 300 * 1024 + 17


@pytest.fixture(autouse=True)
def _knobs(monkeypatch):
    for name in (mat.CHUNK_ENV, mat.THREADS_ENV, mat.VERIFY_ENV, mat.DIRECT_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(mat.CHUNK_ENV, "64Ki")
    monkeypatch.setenv(mat.THREADS_ENV, "8")


def _write_manifest(verified_dir: str) -> None:
    entries = []
    for dirpath, _, names in os.walk(verified_dir):
        for name in names:
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, verified_dir)
            if rel in (se._VERIFIED_MARKER, mat._MANIFEST):
                continue
            with open(path, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
            entries.append({"path": rel, "size_bytes": os.path.getsize(path),
                            "sha256": digest})
    with open(os.path.join(verified_dir, mat._MANIFEST), "w") as handle:
        json.dump({"files": entries}, handle)


def _published(tmp, model_dir):
    """A verified skeleton with a multi-range file, a mode to keep, and a manifest."""
    source = _fake_published(str(tmp), model_dir=model_dir)
    with open(os.path.join(source, "image", "pages-1.img"), "wb") as handle:
        handle.write(os.urandom(_BIG))
    kernel = os.path.join(source, "compilation", "triton", "kernel.so")
    with open(kernel, "wb") as handle:
        handle.write(b"\x7fELF" + os.urandom(1000))
    os.chmod(kernel, 0o755)
    open(os.path.join(source, "image", "empty.img"), "wb").close()
    _write_manifest(source)
    return source


def _tree(root):
    out = {}
    for dirpath, _, names in os.walk(root):
        for name in names:
            path = os.path.join(dirpath, name)
            with open(path, "rb") as handle:
                out[os.path.relpath(path, root)] = (
                    handle.read(), stat.S_IMODE(os.stat(path).st_mode))
    return out


def _gather(source, incoming, verified=None):
    return mat.gather(source, verified or source, incoming, se._COPY_SUBDIRS)


def _same_subdirs(source, incoming):
    for sub in se._COPY_SUBDIRS:
        assert _tree(os.path.join(incoming, sub)) == _tree(
            os.path.join(source, sub))


def _overwrite(path, at, data):
    with open(path, "r+b") as handle:
        handle.seek(at)
        handle.write(data)


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------


def test_plan_takes_one_replicas_subdirs_from_the_key_manifest(tmp_path):
    """A multi-replica skeleton has one manifest at the key, over every replica."""
    key = tmp_path / "mirror" / "M" / "skeleton" / "k"
    for rel in ("replica0/image/a", "replica1/image/a b",
                "replica1/compilation/c", "replica1/other/d"):
        (key / rel).parent.mkdir(parents=True, exist_ok=True)
        (key / rel).write_bytes(b"x")
    _write_manifest(str(key))
    files = mat.plan(str(key / "replica1"), str(key), "/inc", se._COPY_SUBDIRS)
    assert sorted(f.dest for f in files) == ["/inc/compilation/c",
                                             "/inc/image/a b"]
    assert all(f.src.startswith(str(key / "replica1")) for f in files)
    assert all(f.frames is None and f.size == 1 for f in files)


def test_plan_without_a_manifest_lists_the_files_as_they_are(tmp_path):
    source = _published(tmp_path, str(tmp_path / "cache" / "k"))
    os.remove(os.path.join(source, mat._MANIFEST))
    files = mat.plan(source, source, "/inc", se._COPY_SUBDIRS)
    assert sorted(os.path.relpath(f.dest, "/inc") for f in files) == sorted(
        os.path.relpath(os.path.join(d, n), source)
        for sub in se._COPY_SUBDIRS
        for d, _, names in os.walk(os.path.join(source, sub)) for n in names)
    assert not any(f.sha256 for f in files)


def test_plan_refuses_compressed_pages_without_a_manifest(tmp_path):
    source = _published(tmp_path, str(tmp_path / "cache" / "k"))
    os.remove(os.path.join(source, mat._MANIFEST))
    open(os.path.join(source, "image", mat.INDEX_NAME), "w").close()
    with pytest.raises(RuntimeError, match="no _neutrino_manifest.json"):
        mat.plan(source, source, "/inc", se._COPY_SUBDIRS)


def test_ranges_are_cut_by_chunk_largest_first():
    big = mat._File("s", "d", 10, "")
    small = mat._File("s2", "d2", 3, "")
    out = mat._ranges([small, big], 4)
    assert [(r.file.size, r.start, r.end) for r in out] == [
        (10, 0, 4), (10, 4, 8), (3, 0, 3), (10, 8, 10)]


def test_chunk_accepts_binary_suffixes(monkeypatch):
    monkeypatch.setenv(mat.CHUNK_ENV, "64Ki")
    assert mat._env_int(mat.CHUNK_ENV, 1, mat._parse_bytes) == 65536
    monkeypatch.setenv(mat.CHUNK_ENV, "lots")
    assert mat._env_int(mat.CHUNK_ENV, 7, mat._parse_bytes) == 7


def test_engine_and_gather_name_the_same_manifest():
    assert se._DAEMON_MANIFEST == mat._MANIFEST


def test_direct_writes_only_aligned_heads(tmp_path, monkeypatch):
    """O_DIRECT gets whole blocks at block offsets; the file's tail does not."""
    dest = tmp_path / "f"
    dest.write_bytes(bytes(3 * mat._ALIGN + 100))
    out = mat._Writer(str(dest), direct=False)
    out.dfd = os.open(dest, os.O_WRONLY)
    calls = []
    real = os.pwrite

    def pwrite(fd, view, offset):
        calls.append(("direct" if fd == out.dfd else "plain", len(view), offset))
        return real(fd, view, offset)

    monkeypatch.setattr(mat.os, "pwrite", pwrite)
    data = os.urandom(2 * mat._ALIGN + 100)
    out.write(memoryview(data), mat._ALIGN)
    out.write(memoryview(data[:10]), 7)
    out.close()
    assert calls == [("direct", 2 * mat._ALIGN, mat._ALIGN),
                     ("plain", 100, 3 * mat._ALIGN),
                     ("plain", 10, 7)]
    body = dest.read_bytes()
    assert body[mat._ALIGN:] == data and body[7:17] == data[:10]


# --------------------------------------------------------------------------
# gather
# --------------------------------------------------------------------------


def test_gather_matches_the_mirror(tmp_path):
    source = _published(tmp_path, str(tmp_path / "cache" / "k"))
    incoming = str(tmp_path / "incoming")
    stats = _gather(source, incoming)
    _same_subdirs(source, incoming)
    assert stats.copied_bytes == stats.total_bytes > _BIG
    assert stats.compressed_files == 0 and stats.stored_bytes == 0
    kernel = os.path.join("triton", "kernel.so")
    assert os.stat(os.path.join(incoming, "compilation", kernel)).st_mtime == \
        os.stat(os.path.join(source, "compilation", kernel)).st_mtime


def test_every_chunk_is_its_own_range(tmp_path, monkeypatch):
    source = _published(tmp_path, str(tmp_path / "cache" / "k"))
    monkeypatch.setenv(mat.CHUNK_ENV, "1000")
    incoming = str(tmp_path / "incoming")
    stats = _gather(source, incoming)
    _same_subdirs(source, incoming)
    # 1000 rounds up to one 4 KiB block per range.
    assert stats.ranges == sum(
        -(-os.path.getsize(os.path.join(d, n)) // mat._ALIGN)
        for sub in se._COPY_SUBDIRS
        for d, _, names in os.walk(os.path.join(source, sub)) for n in names)


def test_a_file_cut_short_on_the_mirror_fails(tmp_path):
    source = _published(tmp_path, str(tmp_path / "cache" / "k"))
    os.truncate(os.path.join(source, "image", "pages-1.img"), _BIG // 2)
    with pytest.raises(RuntimeError, match="ended early on the mirror"):
        _gather(source, str(tmp_path / "incoming"))


def test_only_sha256_sees_wrong_bytes_of_the_right_size(tmp_path, monkeypatch):
    """``size`` trusts the range accounting; only ``sha256`` sees the bytes."""
    source = _published(tmp_path, str(tmp_path / "cache" / "k"))
    _overwrite(os.path.join(source, "image", "pages-1.img"), 100, b"\xff" * 8)
    _gather(source, str(tmp_path / "size"))
    monkeypatch.setenv(mat.VERIFY_ENV, "sha256")
    with pytest.raises(RuntimeError, match="fail their manifest sha256"):
        _gather(source, str(tmp_path / "sha256"))


def test_sha256_needs_a_manifest(tmp_path, monkeypatch):
    source = _published(tmp_path, str(tmp_path / "cache" / "k"))
    os.remove(os.path.join(source, mat._MANIFEST))
    monkeypatch.setenv(mat.VERIFY_ENV, "sha256")
    with pytest.raises(RuntimeError, match="no manifest sha256"):
        _gather(source, str(tmp_path / "incoming"))


# --------------------------------------------------------------------------
# _materialize_from_source
# --------------------------------------------------------------------------


def _materialized_line(caplog, source, model_dir):
    with caplog.at_level(logging.INFO, logger=se.logger.name):
        assert se._materialize_from_source(
            source, model_dir, _weight_root(source), _weight_hash_of(source))
    _same_subdirs(source, model_dir)
    assert not os.path.exists(os.path.join(model_dir, se._INCOMING_NAME))
    return next(r.getMessage() for r in caplog.records
                if "semi_p: materialized" in r.getMessage())


def test_materialize_copies_from_the_mirror(tmp_path, caplog):
    model_dir = _fresh_model_dir(str(tmp_path))
    source = _published(tmp_path, model_dir)
    line = _materialized_line(caplog, source, model_dir)
    assert " range(s) over " in line and "pages:" not in line


def test_the_materialized_line_says_how_long_the_pre_checks_took(tmp_path,
                                                                 caplog):
    model_dir = _fresh_model_dir(str(tmp_path))
    source = _published(tmp_path, model_dir)
    line = _materialized_line(caplog, source, model_dir)
    assert re.search(r"MB/s\) after \d+\.\d\ds of pre-checks;", line)


def test_materialize_runs_under_a_short_switch_interval(tmp_path, monkeypatch):
    """Pre-checks and gather both run under the short interval, and the
    process's own comes back afterwards."""
    model_dir = _fresh_model_dir(str(tmp_path))
    source = _published(tmp_path, model_dir)
    seen = {}
    real_gather, real_walk = mat.gather, se._newest_mtime

    def gather(*a, **k):
        seen["gather"] = sys.getswitchinterval()
        return real_gather(*a, **k)

    def walk(*a, **k):
        seen["walk"] = sys.getswitchinterval()
        return real_walk(*a, **k)

    monkeypatch.setattr(mat, "gather", gather)
    monkeypatch.setattr(se, "_newest_mtime", walk)
    before = sys.getswitchinterval()
    assert se._materialize_from_source(
        source, model_dir, _weight_root(source), _weight_hash_of(source))
    assert seen == pytest.approx({"gather": se._MATERIALIZE_SWITCH_INTERVAL_S,
                                  "walk": se._MATERIALIZE_SWITCH_INTERVAL_S})
    assert sys.getswitchinterval() == before


def test_the_switch_interval_comes_back_after_a_raise_and_when_nested():
    before = sys.getswitchinterval()
    with pytest.raises(RuntimeError):
        with se._gil_priority():
            with se._gil_priority():
                assert sys.getswitchinterval() == pytest.approx(
                    se._MATERIALIZE_SWITCH_INTERVAL_S)
            assert sys.getswitchinterval() == pytest.approx(
                se._MATERIALIZE_SWITCH_INTERVAL_S)
            raise RuntimeError("copy failed")
    assert sys.getswitchinterval() == before


def test_a_shorter_switch_interval_is_left_alone():
    before = sys.getswitchinterval()
    sys.setswitchinterval(se._MATERIALIZE_SWITCH_INTERVAL_S / 2)
    shorter = sys.getswitchinterval()
    try:
        with se._gil_priority():
            assert sys.getswitchinterval() == shorter
        assert sys.getswitchinterval() == shorter
    finally:
        sys.setswitchinterval(before)


def test_materialize_without_a_manifest_still_copies(tmp_path, caplog):
    model_dir = _fresh_model_dir(str(tmp_path))
    source = _published(tmp_path, model_dir)
    os.remove(os.path.join(source, mat._MANIFEST))
    assert " range(s) over " in _materialized_line(caplog, source, model_dir)


def test_the_retired_gather_knob_changes_nothing(tmp_path, monkeypatch,
                                                 caplog):
    model_dir = _fresh_model_dir(str(tmp_path))
    source = _published(tmp_path, model_dir)
    monkeypatch.setenv("SEMIP_GATHER", "off")
    assert " range(s) over " in _materialized_line(caplog, source, model_dir)


def test_a_failed_gather_is_a_cold_start_unless_strict(tmp_path, monkeypatch):
    model_dir = _fresh_model_dir(str(tmp_path))
    source = _published(tmp_path, model_dir)
    _overwrite(os.path.join(source, "image", "pages-1.img"), 0, b"\0" * 8)
    monkeypatch.setenv(mat.VERIFY_ENV, "sha256")
    args = (source, model_dir, _weight_root(source), _weight_hash_of(source))
    assert se._materialize_from_source(*args) is False
    assert not os.path.exists(os.path.join(model_dir, "image", "meta.json"))
    assert not os.path.exists(os.path.join(model_dir, se._INCOMING_NAME))
    with pytest.raises(RuntimeError, match="materializing"):
        se._materialize_from_source(*args, strict=True)
