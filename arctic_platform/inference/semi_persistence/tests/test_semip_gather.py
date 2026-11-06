"""Unit tests for gathering a published skeleton from peer nodes.

``server/semip_gather.py`` imports only the standard library, so it is loaded
from its file and driven against real HTTP servers on loopback that serve a
temporary mirror the way the model-cache daemon's ``/image-cache/`` route does
-- or refuse, cut a body short, or lie about it, which is what the fallback to
the local mirror exists for.

Run from the package directory::

    cd arctic_platform/inference/semi_persistence
    python -m pytest tests/test_semip_gather.py -v
"""
from __future__ import annotations

import hashlib
import http.server
import importlib.util
import json
import logging
import os
import re
import stat
import sys
import threading
import urllib.parse

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_image_cache_key import (  # noqa: E402
    _PKG, _fake_published, _fresh_model_dir, _weight_hash_of, _weight_root, se)

_GATHER_NAME = "arctic_platform.inference.server.semip_gather"


def _load_semip_gather():
    spec = importlib.util.spec_from_file_location(
        _GATHER_NAME,
        os.path.join(os.path.dirname(_PKG), "server", "semip_gather.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sg = _load_semip_gather()
# What semip_engine imports by name when SEMIP_GATHER is set.
sys.modules[_GATHER_NAME] = sg

_BIG = 300 * 1024 + 17


class _Peer:
    """A loopback stand-in for one node's daemon, serving ``mirror`` at /image-cache/."""

    def __init__(self, mirror: str, behaviour: str = "ok"):
        self.mirror = mirror
        self.behaviour = behaviour
        self.requests = 0
        peer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                peer.requests += 1
                if peer.behaviour == "refuse":
                    self.send_error(409)
                    return
                rel = urllib.parse.unquote(self.path[len("/image-cache/"):])
                with open(os.path.join(peer.mirror, rel), "rb") as handle:
                    data = handle.read()
                match = re.fullmatch(r"bytes=(\d+)-(\d+)",
                                     self.headers.get("Range", ""))
                start, last = int(match.group(1)), int(match.group(2))
                body = data[start:last + 1]
                if peer.behaviour == "lie":
                    body = bytes(len(body))
                self.send_response(206)
                self.send_header("Content-Range",
                                 f"bytes {start}-{last}/{len(data)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if peer.behaviour == "short":
                    body = body[:len(body) // 2]
                self.wfile.write(body)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def peers():
    started = []

    def start(mirror, *behaviours):
        for b in behaviours:
            started.append(_Peer(mirror, b))
        return started[-len(behaviours):]

    yield start
    for p in started:
        p.close()


def _use(monkeypatch, peers_, *, chunk="64Ki", verify=None):
    monkeypatch.setenv(sg.MODE_ENV, "peers")
    monkeypatch.setenv(sg.SERVICE_ENV,
                       ",".join(f"127.0.0.1:{p.port}" for p in peers_)
                       or "127.0.0.1:1")
    monkeypatch.setenv(sg.CHUNK_ENV, chunk)
    monkeypatch.setenv(sg.THREADS_ENV, "8")
    if verify:
        monkeypatch.setenv(sg.VERIFY_ENV, verify)


def _write_manifest(verified_dir: str) -> None:
    entries = []
    for dirpath, _, names in os.walk(verified_dir):
        for name in names:
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, verified_dir)
            if rel in (se._VERIFIED_MARKER, sg._MANIFEST):
                continue
            with open(path, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
            entries.append({"path": rel, "size_bytes": os.path.getsize(path),
                            "sha256": digest})
    with open(os.path.join(verified_dir, sg._MANIFEST), "w") as handle:
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


def _gather(source, incoming, mirror, verified=None):
    return sg.gather(source, verified or source, incoming, se._COPY_SUBDIRS,
                     mirror)


def _same_subdirs(source, incoming):
    for sub in se._COPY_SUBDIRS:
        assert _tree(os.path.join(incoming, sub)) == _tree(
            os.path.join(source, sub))


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------


def test_plan_takes_one_replicas_subdirs_from_the_key_manifest(tmp_path):
    """A multi-replica skeleton has one manifest at the key, over every replica."""
    mirror = tmp_path / "mirror"
    key = mirror / "M" / "skeleton" / "k"
    for rel in ("replica0/image/a", "replica1/image/a b",
                "replica1/compilation/c", "replica1/other/d"):
        (key / rel).parent.mkdir(parents=True, exist_ok=True)
        (key / rel).write_bytes(b"x")
    _write_manifest(str(key))
    files = sg.plan(str(key / "replica1"), str(key), "/inc",
                    se._COPY_SUBDIRS, str(mirror))
    assert sorted((f.dest, f.url) for f in files) == [
        ("/inc/compilation/c", "/image-cache/M/skeleton/k/replica1/compilation/c"),
        ("/inc/image/a b", "/image-cache/M/skeleton/k/replica1/image/a%20b"),
    ]
    assert all(f.src.startswith(str(key / "replica1")) for f in files)


def test_plan_refuses_a_skeleton_outside_the_image_source(tmp_path):
    key = tmp_path / "elsewhere"
    key.mkdir()
    (key / sg._MANIFEST).write_text(json.dumps({"files": []}))
    with pytest.raises(RuntimeError, match="not under the image source"):
        sg.plan(str(key), str(key), "/inc", se._COPY_SUBDIRS,
                str(tmp_path / "mirror"))


def test_ranges_are_dealt_round_robin_largest_first():
    big = sg._File("s", "d", "/u", 10, "")
    small = sg._File("s2", "d2", "/u2", 3, "")
    out = sg._ranges([small, big], 4, [("a", 1), ("b", 2)])
    assert [(r.file.size, r.start, r.end, r.peer[0]) for r in out] == [
        (10, 0, 4, "a"), (10, 4, 8, "b"), (10, 8, 10, "a"), (3, 0, 3, "b")]


def test_chunk_accepts_binary_suffixes(monkeypatch):
    monkeypatch.setenv(sg.CHUNK_ENV, "64Ki")
    assert sg._env_int(sg.CHUNK_ENV, 1, sg._parse_bytes) == 65536
    monkeypatch.setenv(sg.CHUNK_ENV, "lots")
    assert sg._env_int(sg.CHUNK_ENV, 7, sg._parse_bytes) == 7


def test_mode_is_off_unless_peers_or_local(monkeypatch):
    for raw, want in (("", "off"), ("off", "off"), ("peers", "peers"),
                      ("PEERS", "peers"), ("local", "local"), ("s3", "off")):
        monkeypatch.setenv(sg.MODE_ENV, raw)
        assert sg.mode() == want
        assert sg.enabled() is (want != "off")


def test_direct_writes_only_aligned_heads(tmp_path, monkeypatch):
    """O_DIRECT gets whole blocks at block offsets; the file's tail does not."""
    dest = tmp_path / "f"
    dest.write_bytes(bytes(3 * sg._ALIGN + 100))
    out = sg._Writer(str(dest), direct=False)
    out.dfd = os.open(dest, os.O_WRONLY)
    calls = []
    real = os.pwrite

    def pwrite(fd, view, offset):
        calls.append(("direct" if fd == out.dfd else "plain", len(view), offset))
        return real(fd, view, offset)

    monkeypatch.setattr(sg.os, "pwrite", pwrite)
    data = os.urandom(2 * sg._ALIGN + 100)
    out.write(memoryview(data), sg._ALIGN)
    out.write(memoryview(data[:10]), 7)
    out.close()
    assert calls == [("direct", 2 * sg._ALIGN, sg._ALIGN),
                     ("plain", 100, 3 * sg._ALIGN),
                     ("plain", 10, 7)]
    body = dest.read_bytes()
    assert body[sg._ALIGN:] == data and body[7:17] == data[:10]


# --------------------------------------------------------------------------
# gather
# --------------------------------------------------------------------------


def test_gather_from_every_peer_matches_the_mirror(tmp_path, monkeypatch, peers):
    mirror = str(tmp_path / "mirror")
    source = _published(tmp_path, str(tmp_path / "cache" / "k"))
    started = peers(mirror, "ok", "ok", "ok")
    _use(monkeypatch, started)
    incoming = str(tmp_path / "incoming")
    stats = _gather(source, incoming, mirror)
    _same_subdirs(source, incoming)
    assert stats.local_bytes == 0 and stats.local_ranges == 0
    assert stats.peer_bytes == stats.total_bytes > _BIG
    assert (stats.peers, stats.failed_peers) == (3, 0)
    assert all(p.requests > 0 for p in started)
    kernel = os.path.join("triton", "kernel.so")
    assert os.stat(os.path.join(incoming, "compilation", kernel)).st_mtime == \
        os.stat(os.path.join(source, "compilation", kernel)).st_mtime


@pytest.mark.parametrize("behaviour", ["refuse", "short"])
def test_a_failing_peers_ranges_come_from_the_local_mirror(
        tmp_path, monkeypatch, peers, behaviour):
    mirror = str(tmp_path / "mirror")
    source = _published(tmp_path, str(tmp_path / "cache" / "k"))
    _use(monkeypatch, peers(mirror, "ok", behaviour))
    incoming = str(tmp_path / "incoming")
    stats = _gather(source, incoming, mirror)
    _same_subdirs(source, incoming)
    assert stats.failed_peers == 1
    assert stats.local_bytes > 0 and stats.peer_bytes > 0
    assert stats.local_bytes + stats.peer_bytes == stats.total_bytes


def test_no_reachable_peer_is_a_local_copy(tmp_path, monkeypatch):
    mirror = str(tmp_path / "mirror")
    source = _published(tmp_path, str(tmp_path / "cache" / "k"))
    _use(monkeypatch, [])
    monkeypatch.setenv(sg.SERVICE_ENV, "no-such-host.invalid:8081")
    incoming = str(tmp_path / "incoming")
    stats = _gather(source, incoming, mirror)
    _same_subdirs(source, incoming)
    assert stats.peers == 0 and stats.peer_bytes == 0
    assert stats.local_bytes == stats.total_bytes


def test_local_mode_never_asks_a_peer(tmp_path, monkeypatch, peers):
    mirror = str(tmp_path / "mirror")
    source = _published(tmp_path, str(tmp_path / "cache" / "k"))
    started = peers(mirror, "ok")
    _use(monkeypatch, started, chunk="1000")
    monkeypatch.setenv(sg.MODE_ENV, "local")
    incoming = str(tmp_path / "incoming")
    stats = _gather(source, incoming, mirror)
    _same_subdirs(source, incoming)
    assert started[0].requests == 0
    assert stats.peers == 0 and stats.local_bytes == stats.total_bytes
    # 1000 rounds up to one 4 KiB block per range.
    assert stats.local_ranges == sum(
        -(-os.path.getsize(os.path.join(d, n)) // sg._ALIGN)
        for sub in se._COPY_SUBDIRS
        for d, _, names in os.walk(os.path.join(source, sub)) for n in names)


def test_past_the_deadline_ranges_come_from_the_local_mirror(
        tmp_path, monkeypatch, peers):
    mirror = str(tmp_path / "mirror")
    source = _published(tmp_path, str(tmp_path / "cache" / "k"))
    started = peers(mirror, "ok")
    _use(monkeypatch, started)
    monkeypatch.setattr(sg, "_DEADLINE_S", 0.0)
    incoming = str(tmp_path / "incoming")
    stats = _gather(source, incoming, mirror)
    _same_subdirs(source, incoming)
    assert stats.peer_bytes == 0 and started[0].requests == 0


def test_sha256_catches_a_peer_serving_wrong_bytes(tmp_path, monkeypatch, peers):
    """``size`` trusts the range accounting; only ``sha256`` sees the bytes."""
    mirror = str(tmp_path / "mirror")
    source = _published(tmp_path, str(tmp_path / "cache" / "k"))
    _use(monkeypatch, peers(mirror, "lie"), verify="sha256")
    with pytest.raises(RuntimeError, match="fail their manifest sha256"):
        _gather(source, str(tmp_path / "incoming"), mirror)


# --------------------------------------------------------------------------
# _materialize_from_source
# --------------------------------------------------------------------------


def test_materialize_gathers_when_asked(tmp_path, monkeypatch, peers, caplog):
    model_dir = _fresh_model_dir(str(tmp_path))
    source = _published(tmp_path, model_dir)
    mirror = str(tmp_path / "mirror")
    monkeypatch.setenv(se._IMAGE_SOURCE_ENV, mirror)
    _use(monkeypatch, peers(mirror, "ok", "ok"))
    with caplog.at_level(logging.INFO, logger=se.logger.name):
        assert se._materialize_from_source(
            source, model_dir, _weight_root(source), _weight_hash_of(source))
    _same_subdirs(source, model_dir)
    assert not os.path.exists(os.path.join(model_dir, se._INCOMING_NAME))
    line = next(r.getMessage() for r in caplog.records
                if "semi_p: materialized" in r.getMessage())
    assert "from 2/2 peer(s)" in line and "0.00 GiB (0 range(s))" in line


def test_materialize_without_the_switch_copies_as_before(tmp_path, monkeypatch,
                                                         caplog):
    model_dir = _fresh_model_dir(str(tmp_path))
    source = _published(tmp_path, model_dir)
    monkeypatch.delenv(sg.MODE_ENV, raising=False)
    with caplog.at_level(logging.INFO, logger=se.logger.name):
        assert se._materialize_from_source(
            source, model_dir, _weight_root(source), _weight_hash_of(source))
    _same_subdirs(source, model_dir)
    line = next(r.getMessage() for r in caplog.records
                if "semi_p: materialized" in r.getMessage())
    assert "gathered" not in line


def test_a_failed_gather_is_a_cold_start_unless_strict(tmp_path, monkeypatch,
                                                       peers):
    model_dir = _fresh_model_dir(str(tmp_path))
    source = _published(tmp_path, model_dir)
    mirror = str(tmp_path / "mirror")
    monkeypatch.setenv(se._IMAGE_SOURCE_ENV, mirror)
    _use(monkeypatch, peers(mirror, "lie"), verify="sha256")
    args = (source, model_dir, _weight_root(source), _weight_hash_of(source))
    assert se._materialize_from_source(*args) is False
    assert not os.path.exists(os.path.join(model_dir, "image", "meta.json"))
    with pytest.raises(RuntimeError, match="materializing"):
        se._materialize_from_source(*args, strict=True)
