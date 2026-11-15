"""Compressed skeleton pages: publish, node mirror, decompressing gather.

``semip_publish.py`` compresses a dump's ``image/pages-<N>.img`` files into
``.zst`` frames at publish; ``semip_gather.py`` decodes them back when a node
materializes the skeleton. Both are stdlib-only (libzstd through ctypes), so
the round trip runs here for real: a bucket stand-in performs every ``aws s3
cp`` the publisher issues into a local directory, that directory is the node's
mirror once it carries the daemon's marker, and ``_materialize_from_source``
restores from it. Each test then compares bytes, not just sizes.

Needs libzstd (``libzstd.so.1``, or whatever ``ctypes.util.find_library``
finds). Run from the package directory::

    python -m pytest tests/test_semip_compress.py -v
"""
from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_image_cache_key import BASE_CONFIG, KEY, SLUG, se  # noqa: E402
from test_publish_layout import _Bucket, sp  # noqa: E402
from test_semip_gather import sg  # noqa: E402

if sp._load_zstd()[0] is None:
    pytest.skip("libzstd is not loadable here", allow_module_level=True)

F = sp.FRAME_BYTES
_BUCKET = "bkt"
_PREFIX = "image-cache"


# --------------------------------------------------------------------------
# A dump, a bucket that keeps what is uploaded, and a node's mirror of it
# --------------------------------------------------------------------------


def _pattern_frames(seed: int, n: int) -> bytes:
    """Compressible, non-zero bytes."""
    unit = hashlib.sha256(str(seed).encode()).digest()
    return (unit * (n // len(unit) + 1))[:n]


def _pages(*parts: bytes | int) -> bytes:
    """Concatenate parts; an int is that many zero bytes."""
    return b"".join(bytes(p) if isinstance(p, int) else p for p in parts)


# Every shape the frame table has to describe: all zeros, zero frames mixed
# with data, a last frame that is short and not a multiple of 4 KiB, a file
# smaller than one frame, random (incompressible) bytes, and an empty file.
PAGES = {
    "pages-1.img": _pages(2 * F),
    "pages-2.img": _pages(_pattern_frames(2, F), 3 * F, os.urandom(F), F,
                          _pattern_frames(3, F + 12345)),
    "pages-3.img": _pages(_pattern_frames(4, 5000)),
    "pages-4.img": _pages(os.urandom(2 * F + 4096)),
    "pages-5.img": b"",
}


def _meta(model_dir: str) -> dict:
    return {"model_dir": model_dir, "uid": os.getuid(), "gpus": [0],
            "gpu_uuids": ["GPU-00000000-0000-0000-0000-000000000000"],
            "vllm_config": dict(BASE_CONFIG), "image_ref": "dss@sha256:abc",
            "driver_version": "580.159.03"}


def _write(path: str, data: bytes | str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data.encode() if isinstance(data, str) else data)
    return path


def _image(image_dir: str, model_dir: str, pages: dict[str, bytes]) -> None:
    _write(os.path.join(image_dir, "meta.json"), json.dumps(_meta(model_dir)))
    _write(os.path.join(image_dir, "files.img"), "criu-ish")
    # Raw because it is not directly under image/, whatever its name.
    _write(os.path.join(image_dir, "sub", "pages-9.img"), _pattern_frames(9, 99))
    for name, data in pages.items():
        _write(os.path.join(image_dir, name), data)


def _dump(tmp, pages=None, *, replicas: int = 0) -> tuple[str, str]:
    """``(dumped dir to publish, model_dir a restore materializes into)``.

    The dump sits at another path than the restore target, as on a node that
    did not dump it; only the key's name has to agree.
    """
    pages = PAGES if pages is None else pages
    dumped = os.path.join(str(tmp), "dump", KEY)
    target = os.path.join(str(tmp), "cache", KEY)
    for k in range(replicas) if replicas else [None]:
        sub = dumped if k is None else os.path.join(dumped, f"replica{k}")
        model_dir = target if k is None else os.path.join(target, f"replica{k}")
        _image(os.path.join(sub, "image"), model_dir, pages)
        _write(os.path.join(sub, "compilation", "triton", "kernel.json"), "{}")
        _write(os.path.join(sub, sp.WEIGHT_DIR, "weights_meta.json"), "{}")
        _write(os.path.join(sub, sp.WEIGHT_DIR, "shard_0000.bin"), "shard")
    os.makedirs(target, exist_ok=True)
    return dumped, target


class _FileBucket(_Bucket):
    """``_Bucket``, whose ``aws s3 cp`` also copies into ``root``, honouring
    ``--exclude`` the way the aws CLI does: patterns joined to the source dir
    and matched with fnmatch against each file's full path."""

    def __init__(self, root: str):
        super().__init__()
        self.root = root

    def local(self, url: str) -> str:
        return os.path.join(self.root, url[len(f"s3://{_BUCKET}/"):])

    def aws(self, *args: str) -> None:
        super().aws(*args)
        if args[:2] == ("s3", "rm"):
            shutil.rmtree(self.local(args[-1]), ignore_errors=True)
            gone = args[-1].rstrip("/")
            self.sentinels = {d for d in self.sentinels
                              if d != gone and not d.startswith(gone + "/")}
            self.manifests = {d: m for d, m in self.manifests.items()
                              if d in self.sentinels}
            return
        if args[:2] != ("s3", "cp"):
            return
        src, dest = args[-2], args[-1]
        excludes = [args[i + 1] for i, a in enumerate(args) if a == "--exclude"]
        if "--recursive" not in args:
            os.makedirs(os.path.dirname(self.local(dest)), exist_ok=True)
            shutil.copyfile(src, self.local(dest))
            return
        for dirpath, _, names in os.walk(src):
            for name in names:
                path = os.path.join(dirpath, name)
                if any(fnmatch.fnmatch(path, os.path.join(src, p))
                       for p in excludes):
                    continue
                out = os.path.join(self.local(dest), os.path.relpath(path, src))
                os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.copyfile(path, out)


@contextlib.contextmanager
def _bucket(tmp):
    bucket = _FileBucket(os.path.join(str(tmp), "bucket"))
    seams = ("_aws", "_aws_out", "_published", "_skeleton_referrers",
             "image_digest", "_published_replica_count",
             "_published_node_count", "_stage_put", "_stage_list",
             "_stage_clear", "_RENDEZVOUS_POLL_S")
    saved = {name: getattr(sp, name) for name in seams}
    sp._aws = bucket.aws
    sp._aws_out = bucket.aws_out
    sp._published = bucket.published
    sp._skeleton_referrers = lambda *a, **k: []
    sp.image_digest = lambda: ("abc", "dss@sha256:abc")
    sp._published_replica_count = bucket.replica_count
    sp._published_node_count = bucket.node_count
    sp._stage_put = bucket.stage_put
    sp._stage_list = bucket.stage_list
    sp._stage_clear = bucket.stage_clear
    sp._RENDEZVOUS_POLL_S = 0.01
    try:
        yield bucket
    finally:
        for name, value in saved.items():
            setattr(sp, name, value)


def _publish(dumped: str, **kw) -> tuple[str, str]:
    """``(skeleton key, stdout)``."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        key = sp.publish(sp.Path(dumped), _BUCKET, _PREFIX, **kw)
    return key, out.getvalue()


def _mirror(bucket: _FileBucket, key: str) -> str:
    """The published skeleton as a node holds it once verified: the daemon's
    marker written last, in the skeleton and in its weight directory."""
    model_root = os.path.join(bucket.root, _PREFIX, SLUG)
    source = os.path.join(model_root, sp.SKELETON_DIR, key)
    weights = os.path.join(model_root, sp.WEIGHT_DIR, key.rsplit("_", 1)[1])
    stamp = json.dumps({"manifest_digest": "d" * 64,
                        "verified_at": time.time() + 1})
    for where in (weights, source):
        _write(os.path.join(where, se._VERIFIED_MARKER), stamp)
    return source


def _manifest(source: str) -> dict:
    with open(os.path.join(source, sg._MANIFEST)) as handle:
        return json.load(handle)


def _rewrite(source: str, path: str, edit) -> None:
    """Apply ``edit`` to a JSON file on the mirror, then re-stamp the marker."""
    full = os.path.join(source, path)
    with open(full) as handle:
        doc = json.load(handle)
    edit(doc)
    _write(full, json.dumps(doc))
    _write(os.path.join(source, se._VERIFIED_MARKER), json.dumps(
        {"manifest_digest": "d" * 64, "verified_at": time.time() + 1}))


def _materialize(source: str, target: str, **kw) -> bool:
    skeleton = kw.get("verified_dir") or source
    key = os.path.basename(skeleton)
    weight_root = os.path.join(os.path.dirname(os.path.dirname(skeleton)),
                               se._WEIGHT_DIR)
    return se._materialize_from_source(source, target, weight_root,
                                       key.rsplit("_", 1)[1], **kw)


def _tree(root: str) -> dict[str, bytes]:
    out = {}
    for dirpath, _, names in os.walk(root):
        for name in names:
            path = os.path.join(dirpath, name)
            with open(path, "rb") as handle:
                out[os.path.relpath(path, root)] = handle.read()
    return out


def _restored_like_the_dump(dumped: str, target: str) -> None:
    for sub in se._COPY_SUBDIRS:
        assert _tree(os.path.join(target, sub)) == _tree(
            os.path.join(dumped, sub)), sub


def _untouched(target: str) -> None:
    """A failure before the flip: nothing materialized, nothing left over."""
    for name in (*se._COPY_SUBDIRS, se._INCOMING_NAME):
        assert not os.path.exists(os.path.join(target, name)), name


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    for name in (sg.VERIFY_ENV, sg.THREADS_ENV, sg.DIRECT_ENV,
                 sp.COMPRESS_ENV):
        monkeypatch.delenv(name, raising=False)
    # Two frames per range, so files span several ranges and threads.
    monkeypatch.setenv(sg.CHUNK_ENV, "2Mi")


@pytest.fixture
def published(tmp_path):
    """A compressed publish of PAGES and its node mirror."""
    dumped, target = _dump(tmp_path)
    with _bucket(tmp_path) as bucket:
        key, log = _publish(dumped)
        yield dumped, target, _mirror(bucket, key), bucket, log


def _materialized_line(caplog) -> str:
    return next(r.getMessage() for r in caplog.records
                if "semi_p: materialized" in r.getMessage())


# --------------------------------------------------------------------------
# Publish
# --------------------------------------------------------------------------


def test_publish_compresses_exactly_the_non_empty_pages(published):
    dumped, _, source, _, _ = published
    manifest = _manifest(source)
    rows = {r["path"]: r for r in manifest["files"]}
    compressed = sorted(p for p, r in rows.items() if "raw_path" in r)
    assert compressed == [f"image/pages-{n}.img.zst" for n in (1, 2, 3, 4)]
    # Raw: the empty pages file, pages not directly under image/, and every
    # other file, including compilation/.
    for raw in ("image/pages-5.img", "image/sub/pages-9.img",
                "image/files.img", "image/meta.json",
                "compilation/triton/kernel.json"):
        assert raw in rows and "raw_path" not in rows[raw], raw
    assert not any(p in rows for p in (f"image/pages-{n}.img"
                                       for n in (1, 2, 3, 4)))
    for path in compressed:
        row = rows[path]
        raw = os.path.join(dumped, row["raw_path"])
        with open(raw, "rb") as handle:
            assert row["raw_sha256"] == hashlib.sha256(handle.read()).hexdigest()
        assert row["raw_size_bytes"] == os.path.getsize(raw)
        # What the node cache verifies is the stored file.
        stored = os.path.join(source, path)
        assert row["size_bytes"] == os.path.getsize(stored)
        with open(stored, "rb") as handle:
            assert row["sha256"] == hashlib.sha256(handle.read()).hexdigest()
    assert f"image/{sp.COMPRESSION_INDEX}" in rows
    assert manifest["total_bytes"] == sum(r["size_bytes"] for r in rows.values())


def test_the_daemon_reads_the_three_keys_it_always_has(published):
    """sync_node_cache parses path/size_bytes/sha256 and digests path+sha256."""
    rows = _manifest(published[2])["files"]
    for row in rows:
        assert isinstance(row["path"], str)
        assert isinstance(row["size_bytes"], int)
        assert len(row["sha256"]) == 64


def test_the_format_marker_and_index(published):
    dumped, _, source, _, _ = published
    manifest = _manifest(source)
    block = manifest["compression"]
    assert block["format"] == sp.COMPRESSION_FORMAT == sg.FORMAT
    assert block["codec"] == "zstd" and block["level"] == 1
    assert block["frame_bytes"] == F and block["index"] == sp.COMPRESSION_INDEX
    assert block["stored_total_bytes"] == manifest["total_bytes"]
    raw_dump = sum(len(d) for d in _tree(os.path.join(dumped, "image")).values()) \
        + sum(len(d) for d in _tree(os.path.join(dumped, "compilation")).values())
    assert block["raw_total_bytes"] == raw_dump
    assert block["raw_total_bytes"] > 3 * block["stored_total_bytes"]
    with open(os.path.join(source, "image", sp.COMPRESSION_INDEX)) as handle:
        index = json.load(handle)
    assert index["format"] == 1 and index["frame_bytes"] == F
    two = index["files"]["pages-2.img.zst"]
    assert two["raw_size_bytes"] == len(PAGES["pages-2.img"])
    assert len(two["frame_stored_bytes"]) == 8
    assert two["zero_runs"] == [[1, 3], [5, 1]]
    assert index["files"]["pages-1.img.zst"]["zero_runs"] == [[0, 2]]
    assert index["files"]["pages-3.img.zst"]["zero_runs"] == []


def test_raw_pages_never_leave_and_the_sentinel_goes_last(published):
    _, _, source, bucket, _ = published
    image_dest = f"s3://{_BUCKET}/{_PREFIX}/{SLUG}/{sp.SKELETON_DIR}/" \
        f"{os.path.basename(source)}/image"
    image_cps = [a for a in bucket.copies if a[-1] == image_dest]
    assert len(image_cps) == 2
    excluded = [image_cps[0][i + 1] for i, a in enumerate(image_cps[0])
                if a == "--exclude"]
    assert excluded == [f"pages-{n}.img" for n in (1, 2, 3, 4)]
    assert image_cps[1][-2].endswith(os.path.join(sp.PUBLISH_STAGING_DIR,
                                                  "image"))
    assert bucket.copies[-1][-1].endswith(
        f"{os.path.basename(source)}/{sp.MANIFEST_FILENAME}")
    on_mirror = set(os.listdir(os.path.join(source, "image")))
    assert {"pages-5.img", "pages-1.img.zst", sp.COMPRESSION_INDEX} <= on_mirror
    assert not {"pages-1.img", "pages-2.img"} & on_mirror


def test_staging_is_gone_after_a_publish(published):
    assert not os.path.exists(os.path.join(published[0],
                                           sp.PUBLISH_STAGING_DIR))


def test_the_summary_is_the_last_line(published):
    """Onboarding parses it (test_onboarding.py); here, what it says."""
    last = published[4].rstrip("\n").splitlines()[-1]
    assert last.startswith("publish summary: ")
    summary = json.loads(last[len("publish summary: "):])
    manifest = _manifest(published[2])
    assert summary["format"] == "zstd1"
    assert summary["stored_bytes"] == manifest["total_bytes"]
    assert summary["raw_bytes"] == manifest["compression"]["raw_total_bytes"]
    assert summary["compressed_files"] == 4 and summary["zero_frames"] == 6
    assert "hash_compress_s" in summary and "upload_s" in summary


def test_compression_does_not_move_the_weight_hash(tmp_path):
    dumped, _ = _dump(tmp_path)
    with _bucket(tmp_path):
        on, _ = _publish(dumped)
        sp.unpublish_skeleton(sp.Path(dumped), _BUCKET, _PREFIX, key=on)
        off, _ = _publish(dumped, compress=False)
    assert on == off


@pytest.mark.parametrize("how", ["env", "argument"])
def test_off_publishes_todays_manifest(tmp_path, monkeypatch, how):
    dumped, target = _dump(tmp_path)
    if how == "env":
        monkeypatch.setenv(sp.COMPRESS_ENV, "0")
    with _bucket(tmp_path) as bucket:
        key, log = _publish(dumped, **({"compress": False}
                                       if how == "argument" else {}))
        source = _mirror(bucket, key)
    manifest = _manifest(source)
    assert "compression" not in manifest
    expected = sp._scoped_manifest(sp.build_manifest(
        sp.Path(dumped), list(sp.SKELETON_DIRS))["files"], sp.SKELETON_DIRS)
    assert manifest["files"] == expected["files"]
    assert not any(a for a in bucket.copies if "--exclude" in a)
    assert '"format": "raw"' in log.splitlines()[-1]
    assert _materialize(source, target)
    _restored_like_the_dump(dumped, target)


def test_a_dry_run_hashes_raw_and_says_so(tmp_path):
    dumped, _ = _dump(tmp_path)
    with _bucket(tmp_path) as bucket:
        _, log = _publish(dumped, dry_run=True)
    assert "a dry run does not compress" in log
    assert not bucket.copies


def test_without_libzstd_the_publish_is_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "_load_zstd", lambda: (None, "no libzstd here"))
    dumped, _ = _dump(tmp_path)
    with _bucket(tmp_path) as bucket:
        key, log = _publish(dumped)
        assert "compression" not in _manifest(_mirror(bucket, key))
    assert "libzstd cannot be loaded (no libzstd here)" in log


def test_too_little_disk_for_staging_publishes_raw(tmp_path, monkeypatch):
    real = shutil.disk_usage
    monkeypatch.setattr(sp.shutil, "disk_usage",
                        lambda p: real(p)._replace(free=1000))
    dumped, _ = _dump(tmp_path)
    with _bucket(tmp_path) as bucket:
        key, log = _publish(dumped)
        assert "compression" not in _manifest(_mirror(bucket, key))
    assert "less than the" in log and "staging could need" in log


def test_the_decode_check_catches_a_bad_encode(tmp_path, monkeypatch):
    """A frame that is valid zstd but of other bytes fails before upload."""
    real = sp._FrameCodec.compress

    def wrong(self, buf, start, n):
        other = bytearray(buf[start:start + n])
        other[n // 2] ^= 0xFF
        return real(self, other, 0, n)

    monkeypatch.setattr(sp._FrameCodec, "compress", wrong)
    dumped, _ = _dump(tmp_path, {"pages-1.img": _pattern_frames(1, F)})
    with _bucket(tmp_path) as bucket:
        with pytest.raises(RuntimeError, match="does not decode back"):
            _publish(dumped)
        assert not bucket.copies
    assert not os.path.exists(os.path.join(dumped, sp.PUBLISH_STAGING_DIR))


@pytest.mark.parametrize("first", [True, False])
def test_republishing_a_key_in_the_other_format_refuses(tmp_path, first):
    dumped, _ = _dump(tmp_path)
    with _bucket(tmp_path) as bucket:
        _publish(dumped, compress=first)
        before = len(bucket.copies)
        with pytest.raises(SystemExit, match="already published with"):
            _publish(dumped, compress=not first)
        assert len(bucket.copies) == before
        # The same format republishes.
        _publish(dumped, compress=first)


def test_a_replica_dump_compresses_each_replica(tmp_path):
    dumped, target = _dump(tmp_path, replicas=2)
    with _bucket(tmp_path) as bucket:
        key, _ = _publish(dumped)
        source = _mirror(bucket, key)
    rows = _manifest(source)["files"]
    for k in range(2):
        assert f"replica{k}/image/{sp.COMPRESSION_INDEX}" in {r["path"] for r in rows}
        assert f"replica{k}/image/pages-2.img.zst" in {r["path"] for r in rows}
    assert _materialize(os.path.join(source, "replica1"),
                        os.path.join(target, "replica1"),
                        verified_dir=source, strict=True)
    _restored_like_the_dump(os.path.join(dumped, "replica1"),
                            os.path.join(target, "replica1"))


def test_a_pod_spanning_dump_compresses_on_every_pod(tmp_path):
    import threading

    from test_publish_layout import _node_pod

    roots = [_node_pod(str(tmp_path), f"pod-{k}", k) for k in range(2)]
    for k, root in enumerate(roots):
        _write(os.path.join(root, f"node{k}", "image", "pages-1.img"),
               _pages(_pattern_frames(k, F), F))
    errors = []

    def run(root):
        try:
            _publish(root)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    with _bucket(tmp_path) as bucket:
        threads = [threading.Thread(target=run, args=(r,)) for r in roots]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert not errors, errors
        skel = next(v for k, v in bucket.manifests.items()
                    if f"/{sp.SKELETON_DIR}/" in k)
    paths = {r["path"] for r in skel["files"] if "raw_path" in r}
    assert paths == {"node0/image/pages-1.img.zst", "node1/image/pages-1.img.zst"}
    assert skel["compression"]["format"] == 1
    staged = [a for a in bucket.copies if sp.PUBLISH_STAGING_DIR in a[-2]]
    assert len(staged) == 2
    for root in roots:
        assert not os.path.exists(os.path.join(root, sp.PUBLISH_STAGING_DIR))


# --------------------------------------------------------------------------
# Materialize
# --------------------------------------------------------------------------


def test_a_compressed_skeleton_restores_byte_for_byte(published, caplog):
    dumped, target, source, _, _ = published
    with caplog.at_level(logging.INFO, logger=se.logger.name):
        assert _materialize(source, target)
    _restored_like_the_dump(dumped, target)
    image = set(os.listdir(os.path.join(target, "image")))
    assert not any(n.endswith(sg.COMPRESSED_SUFFIX) for n in image)
    assert sp.COMPRESSION_INDEX not in image
    line = _materialized_line(caplog)
    assert "; pages: " in line and "GiB stored -> " in line
    assert "in 4 file(s)" in line and "zero frames left unwritten" in line


def test_zero_frames_are_never_written(published, monkeypatch):
    dumped, target, source, _, _ = published
    writes = []
    real = sg._Writer.write

    def spy(self, view, offset):
        writes.append((os.fstat(self.fd).st_ino, offset, len(view)))
        return real(self, view, offset)

    monkeypatch.setattr(sg._Writer, "write", spy)
    assert _materialize(source, target)
    _restored_like_the_dump(dumped, target)
    inode = {name: os.stat(os.path.join(target, "image", name)).st_ino
             for name in ("pages-1.img", "pages-2.img")}
    assert not [w for w in writes if w[0] == inode["pages-1.img"]]
    two = sorted((o, n) for i, o, n in writes if i == inode["pages-2.img"])
    covered = set()
    for offset, n in two:
        assert offset % F == 0
        covered |= set(range(offset // F, -(-(offset + n) // F)))
    assert covered == {0, 4, 6, 7}


def test_every_file_is_reserved_at_its_full_raw_size(published, monkeypatch):
    """Zero frames read back as zeros because they are fallocated, not holes."""
    dumped, target, source, _, _ = published
    reserved = {}

    def fallocate(fd, offset, length):
        reserved[os.fstat(fd).st_ino] = (offset, length)
        os.ftruncate(fd, offset + length)

    monkeypatch.setattr(sg.os, "posix_fallocate", fallocate, raising=False)
    assert _materialize(source, target)
    _restored_like_the_dump(dumped, target)
    for n in (1, 2, 3, 4):
        path = os.path.join(target, "image", f"pages-{n}.img")
        assert reserved[os.stat(path).st_ino] == (0, len(PAGES[f"pages-{n}.img"]))


def test_gather_stats_account_for_every_raw_byte(published):
    dumped, target, source, _, _ = published
    incoming = os.path.join(target, "inc")
    stats = sg.gather(source, source, incoming, se._COPY_SUBDIRS)
    zero = 6 * F
    nonzero = sum(len(PAGES[f"pages-{n}.img"]) for n in (1, 2, 3, 4)) - zero
    assert stats.compressed_files == 4
    assert (stats.zero_bytes, stats.written_bytes) == (zero, nonzero)
    assert stats.written_bytes + stats.zero_bytes + stats.copied_bytes \
        == stats.total_bytes
    assert stats.stored_bytes == sum(
        r["size_bytes"] for r in _manifest(source)["files"] if "raw_path" in r)
    assert stats.decode_cpu_s >= 0.0


def test_sha256_mode_checks_the_raw_hash(published, monkeypatch):
    dumped, target, source, _, _ = published
    monkeypatch.setenv(sg.VERIFY_ENV, "sha256")
    assert _materialize(source, target)
    _restored_like_the_dump(dumped, target)


def _flip_a_byte(source: str) -> None:
    path = os.path.join(source, "image", "pages-2.img.zst")
    with open(path, "r+b") as handle:
        handle.seek(os.path.getsize(path) // 2)
        byte = handle.read(1)
        handle.seek(-1, os.SEEK_CUR)
        handle.write(bytes([byte[0] ^ 0x5A]))


def _truncate(source: str) -> None:
    path = os.path.join(source, "image", "pages-2.img.zst")
    os.truncate(path, os.path.getsize(path) - 10)


def _wrong_raw_hash(source: str) -> None:
    def edit(doc):
        for row in doc["files"]:
            if row["path"] == "image/pages-2.img.zst":
                row["raw_sha256"] = "0" * 64
    _rewrite(source, sg._MANIFEST, edit)


@pytest.mark.parametrize("damage,verify", [
    (_flip_a_byte, None),
    (_truncate, None),
    (_wrong_raw_hash, "sha256"),
])
def test_damage_fails_before_the_flip(published, monkeypatch, damage, verify):
    _, target, source, _, _ = published
    damage(source)
    if verify:
        monkeypatch.setenv(sg.VERIFY_ENV, verify)
    assert _materialize(source, target) is False
    _untouched(target)
    with pytest.raises(RuntimeError, match="materializing"):
        _materialize(source, target, strict=True)
    _untouched(target)


def test_a_corrupt_frame_names_the_file_frame_and_offset(published):
    _, target, source, _, _ = published
    _flip_a_byte(source)
    with pytest.raises(RuntimeError) as err:
        sg.gather(source, source, os.path.join(target, "inc"),
                  se._COPY_SUBDIRS)
    assert "pages-2.img.zst: frame " in str(err.value)
    assert "raw offset" in str(err.value)


def test_the_default_check_passes_a_wrong_raw_hash(published):
    """Frame checksums guard the bytes; the raw sha256 is the opt-in check."""
    dumped, target, source, _, _ = published
    _wrong_raw_hash(source)
    assert _materialize(source, target)
    _restored_like_the_dump(dumped, target)


def test_without_libzstd_a_compressed_skeleton_is_a_miss(published,
                                                         monkeypatch):
    _, target, source, _, _ = published

    def missing():
        raise RuntimeError("this skeleton is compressed and libzstd cannot be "
                           "loaded (test)")

    monkeypatch.setattr(sg, "_load_zstd", missing)
    assert _materialize(source, target) is False
    _untouched(target)


def _drop_index_row(doc):
    doc["files"] = [r for r in doc["files"]
                    if not r["path"].endswith(sp.COMPRESSION_INDEX)]


def _drop_block(doc):
    del doc["compression"]


def _block_only(doc):
    for row in doc["files"]:
        for key in ("raw_path", "raw_size_bytes", "raw_sha256"):
            row.pop(key, None)


def _list_raw_too(doc):
    doc["files"].append({"path": "image/pages-2.img", "size_bytes": 1,
                         "sha256": "0" * 64})


def _odd_frames(doc):
    doc["compression"]["frame_bytes"] = 1000


def _wrong_raw_size(doc):
    for row in doc["files"]:
        if row["path"] == "image/pages-2.img.zst":
            row["raw_size_bytes"] += 1


def _short_frame_table(doc):
    doc["files"]["pages-2.img.zst"]["frame_stored_bytes"].pop()


def _frame_sum(doc):
    doc["files"]["pages-2.img.zst"]["frame_stored_bytes"][0] += 1


def _zero_run_past_the_end(doc):
    doc["files"]["pages-2.img.zst"]["zero_runs"].append([7, 2])


def _missing_entry(doc):
    del doc["files"]["pages-3.img.zst"]


@pytest.mark.parametrize("path,edit,message", [
    (sg._MANIFEST, _drop_index_row, "but not its index"),
    (sg._MANIFEST, _drop_block, "no 'compression' block"),
    (sg._MANIFEST, _block_only, "no compressed rows"),
    (sg._MANIFEST, _list_raw_too, "both raw and compressed"),
    (sg._MANIFEST, _odd_frames, "unusable 'compression' block"),
    (sg._MANIFEST, _wrong_raw_size, "disagrees with its manifest row"),
    (f"image/{sp.COMPRESSION_INDEX}", _short_frame_table,
     "disagrees with its manifest row"),
    (f"image/{sp.COMPRESSION_INDEX}", _frame_sum,
     "disagrees with its manifest row"),
    (f"image/{sp.COMPRESSION_INDEX}", _zero_run_past_the_end, "bad zero run"),
    (f"image/{sp.COMPRESSION_INDEX}", _missing_entry, "does not describe"),
])
def test_plan_refuses_an_inconsistent_skeleton(published, path, edit, message):
    _, target, source, _, _ = published
    _rewrite(source, path, edit)
    with pytest.raises(RuntimeError, match=message):
        sg.plan(source, source, os.path.join(target, "inc"),
                se._COPY_SUBDIRS)
    assert _materialize(source, target) is False
    _untouched(target)


def test_a_newer_format_is_a_miss_before_any_copy(published, monkeypatch,
                                                  caplog):
    _, target, source, _, _ = published
    _rewrite(source, sg._MANIFEST,
             lambda doc: doc["compression"].update(format=2))
    called = []
    monkeypatch.setattr(sg, "gather", lambda *a, **k: called.append(a))
    with caplog.at_level(logging.INFO, logger=se.logger.name):
        assert _materialize(source, target, strict=True) is False
    assert not called
    _untouched(target)
    assert any("skeleton format 2 is newer than this engine" in r.getMessage()
               for r in caplog.records)


def test_a_compressed_skeleton_without_a_manifest_is_a_miss(published,
                                                            caplog):
    """Without the manifest there is no frame table: never copy the .zst."""
    _, target, source, _, _ = published
    os.remove(os.path.join(source, sg._MANIFEST))
    with caplog.at_level(logging.INFO, logger=se.logger.name):
        assert _materialize(source, target) is False
    _untouched(target)
    assert "holds compressed pages but" in caplog.text


def test_a_raw_skeleton_still_restores(tmp_path, caplog):
    dumped, target = _dump(tmp_path)
    with _bucket(tmp_path) as bucket:
        key, _ = _publish(dumped, compress=False)
        source = _mirror(bucket, key)
    with caplog.at_level(logging.INFO, logger=se.logger.name):
        assert _materialize(source, target)
    _restored_like_the_dump(dumped, target)
    assert "; pages: " not in _materialized_line(caplog)


@pytest.mark.skipif(not shutil.which("zstd"), reason="no zstd CLI")
def test_the_stock_cli_decodes_a_stored_file(published):
    dumped, _, source, _, _ = published
    for n in (1, 2, 3, 4):
        out = subprocess.run(
            ["zstd", "-dc", os.path.join(source, "image", f"pages-{n}.img.zst")],
            capture_output=True, check=True).stdout
        assert out == PAGES[f"pages-{n}.img"], n


def test_the_ranges_skip_zero_frames_and_go_largest_first():
    zero = bytes([1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    offsets = list(range(0, 11))
    f = sg._File("s", "d", 10 * F - 7, "", sg._Frames(F, offsets, zero))
    raw = sg._File("r", "rd", 3, "")
    out = sg._ranges([raw, f], 4 * F)
    assert [(r.start // F, -(-r.end // F)) for r in out[:-1]] == [
        (4, 8), (1, 3), (8, 9)]
    assert out[-1].file is raw and out[-1].end == 3
