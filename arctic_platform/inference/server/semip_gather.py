"""Gather a published skeleton from the model-cache daemons on other nodes.

Every node's neutrino-model-cache DaemonSet mirrors the same published tree and
serves byte ranges of its verified ``image-cache/`` files over HTTP
(``image_cache_peer.py`` in cortex's ``neutrino/scripts``). A materialize that
reads its skeleton from the local mirror is bound by one node's copy path --
measured at ~3.3 GB/s through ``shutil.copytree`` -- while pulling ranges of the
same files from every node at once is bound by the pod network and the
destination disk instead.

Nothing is distributed: every node still holds every file. So any range a peer
does not deliver (refused, timed out, short, gone, or simply never assigned
because there were no peers) is read from the local mirror and written at the
same offset. There is no S3 path and no retry against another peer, and a
gather with no reachable peer is a parallel local copy -- which is also what
``SEMIP_GATHER=local`` asks for outright. On p5en that local copy alone runs at
~11 GB/s against copytree's ~3.9, the same ceiling three peers reached: the
destination's buffered writes are the bound, not the network.

Standard library only, like ``semip_engine``: the tests load both straight from
their files, without the packages that pull in ray and torch.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import mmap
import os
import queue
import re
import shutil
import socket
import threading
import time
import urllib.parse
from typing import NamedTuple

logger = logging.getLogger("arctic_platform.inference.server.semip")

# "peers" gathers from every node's daemon; "local" is the same ranged, parallel
# copy with no peers, from this node's mirror alone. Unset, "" or "off" keeps
# shutil.copytree.
MODE_ENV = "SEMIP_GATHER"
MODES = ("peers", "local")
# host:port of the daemons, or several comma-separated. A hostname resolves to
# every address it carries, which for the headless Service is every ready node.
SERVICE_ENV = "SEMIP_GATHER_SERVICE"
CHUNK_ENV = "SEMIP_GATHER_CHUNK"
THREADS_ENV = "SEMIP_GATHER_THREADS"
# "size" trusts the range accounting; "sha256" also hashes every gathered file
# against the manifest.
VERIFY_ENV = "SEMIP_GATHER_VERIFY"
# "1" writes through O_DIRECT wherever offset and length are block-aligned,
# which skips the page cache -- and so also leaves the gathered files out of
# it for the restore that reads them next.
DIRECT_ENV = "SEMIP_GATHER_DIRECT"

DEFAULT_SERVICE = "neutrino-model-cache.neutrino.svc.cluster.local:8081"
DEFAULT_CHUNK = 128 * 2**20
DEFAULT_THREADS = 64

# The daemon's per-directory manifest. Paths in it are relative to the
# directory that holds it, which for a multi-replica skeleton is the key
# directory over every replica<K>/.
_MANIFEST = "_neutrino_manifest.json"

# The daemon serves <cache-dir>/image-cache/<rel> at this prefix, and the pod
# mounts <cache-dir>/image-cache as the image source -- so a file's URL is its
# path relative to the image source root.
_URL_PREFIX = "/image-cache/"

# Past this, every range not yet started is read from the local mirror instead.
_DEADLINE_S = 120.0
# Short, because a peer that does not answer is a peer whose ranges go local,
# and a NetworkPolicy that drops the SYN would otherwise cost this per flow.
_CONNECT_TIMEOUT_S = 2.0
_READ_TIMEOUT_S = 10.0
_BUF_BYTES = 8 * 2**20
# O_DIRECT wants offset, length and buffer address all aligned to the logical
# block size; 4 KiB covers every block device here. Ranges start on multiples
# of the chunk, which is rounded up to this.
_ALIGN = 4096

_SIZE_SUFFIXES = {"": 1, "k": 10**3, "m": 10**6, "g": 10**9,
                  "ki": 2**10, "mi": 2**20, "gi": 2**30}


class GatherStats(NamedTuple):
    files: int
    total_bytes: int
    peer_bytes: int
    local_bytes: int
    peers: int
    failed_peers: int
    flows: int
    local_ranges: int
    direct: bool
    elapsed_s: float


class _File(NamedTuple):
    src: str
    dest: str
    url: str
    size: int
    sha256: str


class _Range(NamedTuple):
    file: _File
    start: int
    end: int
    peer: tuple[str, int] | None


class _PeerFailed(Exception):
    pass


def mode() -> str:
    """One of MODES, or "off"."""
    raw = os.environ.get(MODE_ENV, "").strip().lower()
    if raw in ("", "off"):
        return "off"
    if raw in MODES:
        return raw
    logger.warning("semi_p: %s=%r is not one of off, %s; copying with "
                   "copytree", MODE_ENV, raw, ", ".join(MODES))
    return "off"


def enabled() -> bool:
    return mode() != "off"


def _parse_bytes(raw: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*([kKmMgG]i?)?\s*", raw)
    if not match:
        raise ValueError(f"{raw!r} is not a byte count")
    return int(match.group(1)) * _SIZE_SUFFIXES[(match.group(2) or "").lower()]


def _env_int(name: str, default: int, parse=int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = parse(raw)
    except ValueError:
        value = 0
    if value <= 0:
        logger.warning("semi_p: %s=%r is not a positive number; using %d",
                       name, raw, default)
        return default
    return value


def _verify_mode() -> str:
    raw = os.environ.get(VERIFY_ENV, "").strip().lower() or "size"
    if raw not in ("size", "sha256"):
        logger.warning("semi_p: %s=%r is not one of size, sha256; using size",
                       VERIFY_ENV, raw)
        return "size"
    return raw


def resolve_peers(service: str) -> list[tuple[str, int]]:
    """Every address the service names. Empty, with a warning, if none resolve."""
    peers: list[tuple[str, int]] = []
    for item in service.split(","):
        item = item.strip()
        if not item:
            continue
        host, _, port = item.rpartition(":")
        try:
            infos = socket.getaddrinfo(host.strip("[]"), int(port),
                                       type=socket.SOCK_STREAM)
        except (OSError, ValueError) as exc:
            logger.warning("semi_p: cannot resolve gather peers %r (%s)", item, exc)
            continue
        for info in infos:
            peer = (info[4][0], int(info[4][1]))
            if peer not in peers:
                peers.append(peer)
    return peers


def plan(source_dir: str, verified_dir: str, incoming: str,
         subdirs: tuple[str, ...], source_root: str) -> list[_File]:
    """The files of ``subdirs`` under ``source_dir``, from the manifest beside them."""
    manifest_path = os.path.join(verified_dir, _MANIFEST)
    with open(manifest_path) as handle:
        doc = json.load(handle)
    entries = doc.get("files") if isinstance(doc, dict) else None
    if not isinstance(entries, list):
        raise RuntimeError(f"{manifest_path} has no 'files' list")

    rel = os.path.relpath(source_dir, verified_dir)
    prefix = "" if rel == "." else rel.replace(os.sep, "/") + "/"
    url_base = os.path.relpath(verified_dir, source_root).replace(os.sep, "/")
    if url_base == ".." or url_base.startswith("../"):
        raise RuntimeError(f"{verified_dir} is not under the image source "
                           f"{source_root}, so peers cannot address it")

    files = []
    for entry in entries:
        path = str(entry.get("path") or "")
        for sub in subdirs:
            head = f"{prefix}{sub}/"
            if path.startswith(head):
                files.append(_File(
                    src=os.path.join(verified_dir, path),
                    dest=os.path.join(incoming, sub, path[len(head):]),
                    url=_URL_PREFIX + urllib.parse.quote(f"{url_base}/{path}"),
                    size=int(entry.get("size_bytes") or 0),
                    sha256=str(entry.get("sha256") or "").lower()))
                break
    return files


def _ranges(files: list[_File], chunk: int,
            peers: list[tuple[str, int]]) -> list[_Range]:
    """Largest file first, cut into ``chunk``-byte ranges dealt round-robin."""
    pieces = []
    for f in sorted(files, key=lambda f: f.size, reverse=True):
        for start in range(0, f.size, chunk):
            pieces.append((f, start, min(start + chunk, f.size)))
    return [_Range(f, start, end, peers[i % len(peers)] if peers else None)
            for i, (f, start, end) in enumerate(pieces)]


def _pwrite_all(fd: int, view: memoryview, offset: int) -> None:
    written = 0
    while written < len(view):
        written += os.pwrite(fd, view[written:], offset + written)


class _Writer:
    """``pwrite`` into one file, through O_DIRECT wherever alignment allows.

    Only a file's tail is ever unaligned, since every range starts on a chunk
    boundary and every write but a range's last fills the whole buffer; that
    tail goes through an ordinary descriptor onto pages no direct write touches.
    """

    def __init__(self, path: str, direct: bool):
        self.fd = os.open(path, os.O_WRONLY)
        self.dfd = None
        if direct:
            try:
                self.dfd = os.open(path, os.O_WRONLY | os.O_DIRECT)
            except OSError:
                pass

    def write(self, view: memoryview, offset: int) -> None:
        head = len(view) - len(view) % _ALIGN
        if self.dfd is not None and head and offset % _ALIGN == 0:
            written = os.pwrite(self.dfd, view[:head], offset)
            if written < len(view):
                _pwrite_all(self.fd, view[written:], offset + written)
        else:
            _pwrite_all(self.fd, view, offset)

    def close(self) -> None:
        os.close(self.fd)
        if self.dfd is not None:
            os.close(self.dfd)


def _transfer(rng: _Range, buf: memoryview, read_into, direct: bool) -> None:
    """Fill ``buf`` from ``read_into(view, offset) -> n`` and write it out, to ``rng.end``."""
    out = _Writer(rng.file.dest, direct)
    try:
        offset = rng.start
        while offset < rng.end:
            want = min(len(buf), rng.end - offset)
            filled = 0
            while filled < want:
                got = read_into(buf[filled:want], offset + filled)
                if not got:
                    raise EOFError(f"{offset + filled - rng.start} of "
                                   f"{rng.end - rng.start} bytes")
                filled += got
            out.write(buf[:want], offset)
            offset += want
    finally:
        out.close()


def _fetch(rng: _Range, buf: memoryview, direct: bool = False) -> None:
    """GET ``rng`` from its peer and write it in place. _PeerFailed on any fault."""
    host, port = rng.peer
    conn = http.client.HTTPConnection(host, port, timeout=_CONNECT_TIMEOUT_S)
    try:
        conn.connect()
        conn.sock.settimeout(_READ_TIMEOUT_S)
        conn.request("GET", rng.file.url,
                     headers={"Range": f"bytes={rng.start}-{rng.end - 1}"})
        resp = conn.getresponse()
        expected = f"bytes {rng.start}-{rng.end - 1}/{rng.file.size}"
        if resp.status != 206 or resp.getheader("Content-Range") != expected:
            raise _PeerFailed(f"HTTP {resp.status} "
                              f"{resp.getheader('Content-Range')!r} for {expected}")
        _transfer(rng, buf, lambda view, _: resp.readinto(view), direct)
    except EOFError as exc:
        raise _PeerFailed(f"short body at {exc}") from exc
    except (OSError, http.client.HTTPException) as exc:
        raise _PeerFailed(f"{type(exc).__name__}: {exc}") from exc
    finally:
        conn.close()


def _copy_local(rng: _Range, buf: memoryview, direct: bool = False) -> None:
    src = os.open(rng.file.src, os.O_RDONLY)
    try:
        _transfer(rng, buf, lambda view, offset: os.preadv(src, [view], offset),
                  direct)
    except EOFError as exc:
        raise RuntimeError(f"{rng.file.src} ended early on the mirror, at "
                           f"{exc}") from exc
    finally:
        os.close(src)


def _preallocate(files: list[_File], incoming: str,
                 subdirs: tuple[str, ...]) -> None:
    for sub in subdirs:
        os.makedirs(os.path.join(incoming, sub), exist_ok=True)
    for f in files:
        os.makedirs(os.path.dirname(f.dest), exist_ok=True)
        fd = os.open(f.dest, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            if f.size and hasattr(os, "posix_fallocate"):
                os.posix_fallocate(fd, 0, f.size)
            else:
                os.ftruncate(fd, f.size)
        finally:
            os.close(fd)


def _copy_stats(files: list[_File], source_dir: str, incoming: str,
                subdirs: tuple[str, ...]) -> None:
    """Modes and times from the mirror, as ``copytree``'s ``copy2`` would leave them."""
    for f in files:
        shutil.copystat(f.src, f.dest)
    for sub in subdirs:
        src_root = os.path.join(source_dir, sub)
        dest_root = os.path.join(incoming, sub)
        # Deepest first: writing a directory's children bumps its mtime.
        for dirpath, _, _ in sorted(os.walk(dest_root), key=lambda w: -len(w[0])):
            src = os.path.join(src_root, os.path.relpath(dirpath, dest_root))
            if os.path.isdir(src):
                shutil.copystat(src, dirpath)


def _sha256_mismatches(files: list[_File], threads: int) -> list[str]:
    work: queue.Queue[_File] = queue.Queue()
    for f in files:
        work.put(f)
    bad: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                f = work.get_nowait()
            except queue.Empty:
                return
            digest = hashlib.sha256()
            with open(f.dest, "rb") as handle:
                for block in iter(lambda: handle.read(_BUF_BYTES), b""):
                    digest.update(block)
            if digest.hexdigest() != f.sha256:
                with lock:
                    bad.append(f.dest)

    pool = [threading.Thread(target=worker, daemon=True)
            for _ in range(min(threads, max(1, len(files))))]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    return bad


def gather(source_dir: str, verified_dir: str, incoming: str,
           subdirs: tuple[str, ...], source_root: str) -> GatherStats:
    """Assemble ``subdirs`` of ``source_dir`` under ``incoming`` from every peer.

    Raises if a range cannot be had from the local mirror either, or if a
    gathered file fails verification; the caller treats that like a failed copy.
    """
    t0 = time.perf_counter()
    chunk = _env_int(CHUNK_ENV, DEFAULT_CHUNK, _parse_bytes)
    chunk = -(-chunk // _ALIGN) * _ALIGN
    threads = _env_int(THREADS_ENV, DEFAULT_THREADS)
    verify = _verify_mode()
    direct = (os.environ.get(DIRECT_ENV, "").strip() == "1"
              and hasattr(os, "O_DIRECT"))

    files = plan(source_dir, verified_dir, incoming, subdirs, source_root)
    peers: list[tuple[str, int]] = []
    if mode() == "peers":
        peers = resolve_peers(os.environ.get(SERVICE_ENV, "").strip()
                              or DEFAULT_SERVICE)
        if not peers:
            logger.warning("semi_p: no gather peers; reading every range of "
                           "%s from the local mirror", source_dir)
    _preallocate(files, incoming, subdirs)

    work: queue.Queue[_Range] = queue.Queue()
    for rng in _ranges(files, chunk, peers):
        work.put(rng)
    deadline = t0 + _DEADLINE_S
    lock = threading.Lock()
    failed: dict[tuple[str, int], str] = {}
    counts = {"peer": 0, "local": 0, "local_ranges": 0}
    errors: list[BaseException] = []

    def worker() -> None:
        # Page-aligned, as O_DIRECT requires of the buffer address.
        buf = memoryview(mmap.mmap(-1, _BUF_BYTES))
        while not errors:
            try:
                rng = work.get_nowait()
            except queue.Empty:
                return
            try:
                if (rng.peer is not None and rng.peer not in failed
                        and time.perf_counter() < deadline):
                    try:
                        _fetch(rng, buf, direct)
                        with lock:
                            counts["peer"] += rng.end - rng.start
                        continue
                    except _PeerFailed as exc:
                        with lock:
                            if rng.peer not in failed:
                                failed[rng.peer] = str(exc)
                                logger.warning(
                                    "semi_p: gather peer %s:%d failed on %s "
                                    "(%s); its ranges come from the local "
                                    "mirror", rng.peer[0], rng.peer[1],
                                    rng.file.url, exc)
                _copy_local(rng, buf, direct)
                with lock:
                    counts["local"] += rng.end - rng.start
                    counts["local_ranges"] += 1
            except BaseException as exc:
                with lock:
                    errors.append(exc)
                return

    flows = min(threads, max(1, work.qsize()))
    pool = [threading.Thread(target=worker, name=f"semip-gather-{i}",
                             daemon=True) for i in range(flows)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    if errors:
        raise RuntimeError(f"gathering {source_dir} failed: {errors[0]}") \
            from errors[0]

    total = sum(f.size for f in files)
    if counts["peer"] + counts["local"] != total:
        raise RuntimeError(f"gathered {counts['peer'] + counts['local']} of "
                           f"{total} bytes of {source_dir}")
    for f in files:
        if os.stat(f.dest).st_size != f.size:
            raise RuntimeError(f"{f.dest} is {os.stat(f.dest).st_size} bytes, "
                               f"the manifest says {f.size}")
    if verify == "sha256":
        bad = _sha256_mismatches(files, threads)
        if bad:
            raise RuntimeError(f"{len(bad)} gathered file(s) fail their "
                               f"manifest sha256, e.g. {bad[0]}")
    _copy_stats(files, source_dir, incoming, subdirs)
    return GatherStats(
        files=len(files), total_bytes=total, peer_bytes=counts["peer"],
        local_bytes=counts["local"], peers=len(peers),
        failed_peers=len(failed), flows=flows,
        local_ranges=counts["local_ranges"], direct=direct,
        elapsed_s=time.perf_counter() - t0)
