"""Materialize a published skeleton from this node's mirror.

The neutrino-model-cache DaemonSet mirrors the published tree onto every node,
so a materialize is a local copy: the files of ``compilation/`` and ``image/``
are cut into ranges and copied by a pool of threads, with O_DIRECT writes
wherever offset and length are block-aligned. On p5en that moves GLM-5.3's
79 GB in ~4.2 s, against ~20 s through ``shutil.copytree``.

A compressed skeleton (``semip_publish.py``, "Compressed pages") goes through
the same workers: each ``pages-<N>.img.zst`` becomes the raw ``pages-<N>.img``
CRIU reads, decoded frame by frame straight into the write buffer. The file is
preallocated at its full raw size and only non-zero frames are written; the
all-zero frames stay preallocated, unwritten ranges, which read as zeros.
Every frame's checksum and decoded length are checked on the way, and
``sha256`` mode also hashes the result against the raw sha256. Every other
file, and every file of a raw skeleton, is copied as raw ranges.

Standard library only, like ``semip_engine``: the tests load both straight from
their files, without the packages that pull in ray and torch. libzstd is
reached through ctypes, and only for a compressed skeleton.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import logging
import mmap
import os
import posixpath
import queue
import re
import shutil
import threading
import time
from typing import NamedTuple

logger = logging.getLogger("arctic_platform.inference.server.semip")

CHUNK_ENV = "SEMIP_GATHER_CHUNK"
THREADS_ENV = "SEMIP_GATHER_THREADS"
# "size" trusts the range accounting (and, for compressed pages, every frame's
# checksum and length); "sha256" also hashes every gathered file against the
# manifest's raw sha256.
VERIFY_ENV = "SEMIP_GATHER_VERIFY"
# On by default ("0" turns it off): write through O_DIRECT wherever offset and
# length are block-aligned. Buffered writes cap at ~11 GB/s on ~45 cores of
# kernel time on p5en; O_DIRECT reaches the disk's ~19 GB/s on ~5. Leaving the
# gathered files out of the page cache cost the restore's read of image/
# nothing measurable (10-11 s either way).
DIRECT_ENV = "SEMIP_GATHER_DIRECT"

DEFAULT_CHUNK = 128 * 2**20
DEFAULT_THREADS = 32

# The daemon's per-directory manifest. Paths in it are relative to the
# directory that holds it, which for a multi-replica skeleton is the key
# directory over every replica<K>/.
_MANIFEST = "_neutrino_manifest.json"

_BUF_BYTES = 8 * 2**20
# O_DIRECT wants offset, length and buffer address all aligned to the logical
# block size; 4 KiB covers every block device here. Ranges start on multiples
# of the chunk, which is rounded up to this.
_ALIGN = 4096

_SIZE_SUFFIXES = {"": 1, "k": 10**3, "m": 10**6, "g": 10**9,
                  "ki": 2**10, "mi": 2**20, "gi": 2**30}

# The compressed format this engine reads, and its spellings: pinned to
# semip_publish's by tests/test_layout_names.py.
FORMAT = 1
CODEC = "zstd"
COMPRESSED_SUFFIX = ".zst"
INDEX_NAME = "compression.json"
_LIBZSTD_SONAME = "libzstd.so.1"
# A zstd frame starts with this magic and a descriptor byte whose bit 2 says a
# checksum follows the frame and whose low two bits size a dictionary id.
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_ZSTD_CHECKSUM_BIT = 0x04
_ZSTD_DICT_BITS = 0x03


class GatherStats(NamedTuple):
    files: int
    total_bytes: int        # raw bytes now under incoming
    copied_bytes: int       # of those, copied as raw ranges
    ranges: int
    flows: int
    direct: bool
    elapsed_s: float
    # Of the compressed files: bytes on the mirror, raw bytes decoded and
    # written, raw bytes of all-zero frames left unwritten, and decode CPU.
    stored_bytes: int = 0
    written_bytes: int = 0
    zero_bytes: int = 0
    compressed_files: int = 0
    decode_cpu_s: float = 0.0


class _Frames(NamedTuple):
    """A compressed file's frame table, from its image dir's index."""
    frame_bytes: int
    offsets: list[int]      # stored offset of every frame, then the stored size
    zero: bytes             # 1 for every all-zero frame


class _File(NamedTuple):
    src: str
    dest: str
    size: int               # bytes at dest: raw, for a compressed file
    sha256: str             # of the bytes at dest
    frames: _Frames | None = None


class _Range(NamedTuple):
    file: _File
    start: int
    end: int


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


def compression_format(verified_dir: str) -> int:
    """The skeleton's compression format, from its manifest: 0 for raw.

    Also 0 when there is no readable manifest, which ``plan`` then handles.
    """
    try:
        with open(os.path.join(verified_dir, _MANIFEST)) as handle:
            block = json.load(handle).get("compression")
    except (OSError, ValueError, AttributeError):
        return 0
    if block is None:
        return 0
    fmt = block.get("format") if isinstance(block, dict) else None
    return fmt if isinstance(fmt, int) and fmt > 0 else FORMAT


def _compression(doc: dict, entries: list, manifest_path: str
                 ) -> tuple[dict | None, set[str]]:
    """``(compression block, index paths)``, once the rows agree with the block.

    Raises on anything ``plan`` must not guess about.
    """
    block = doc.get("compression")
    compressed = [e for e in entries if "raw_path" in e]
    if block is None:
        if compressed:
            raise RuntimeError(f"{manifest_path} has compressed rows but no "
                               f"'compression' block")
        return None, set()
    if not isinstance(block, dict):
        raise RuntimeError(f"{manifest_path}: 'compression' is not an object")
    if block.get("format") != FORMAT:
        raise RuntimeError(f"skeleton format {block.get('format')!r} is newer "
                           f"than this engine, which reads format {FORMAT}")
    frame = block.get("frame_bytes")
    index = block.get("index")
    if (block.get("codec") != CODEC or not isinstance(frame, int)
            or frame <= 0 or frame % _ALIGN or not isinstance(index, str)
            or not index or "/" in index):
        raise RuntimeError(f"{manifest_path}: unusable 'compression' block "
                           f"{block!r}")
    if not compressed:
        raise RuntimeError(f"{manifest_path} has a 'compression' block but no "
                           f"compressed rows")
    listed = {str(e.get("path") or "") for e in entries}
    indexes = set()
    for e in compressed:
        path, raw = str(e.get("path") or ""), str(e.get("raw_path") or "")
        if path != raw + COMPRESSED_SUFFIX:
            raise RuntimeError(f"{manifest_path}: {path!r} is not "
                               f"{raw!r} + {COMPRESSED_SUFFIX}")
        if raw in listed:
            raise RuntimeError(f"{manifest_path} lists {raw!r} both raw and "
                               f"compressed")
        where = posixpath.join(posixpath.dirname(path), index)
        if where not in listed:
            raise RuntimeError(f"{manifest_path} lists {path!r} but not its "
                               f"index {where!r}")
        indexes.add(where)
    return block, indexes


def _frames(entry: dict, block: dict, verified_dir: str,
            index_docs: dict[str, dict]) -> _Frames:
    """``entry``'s frame table from its index, checked against its row."""
    path = entry["path"]
    where = posixpath.join(posixpath.dirname(path), block["index"])
    if where not in index_docs:
        with open(os.path.join(verified_dir, where)) as handle:
            index_docs[where] = json.load(handle)
    doc = index_docs[where]
    frame = block["frame_bytes"]
    item = ((doc.get("files") or {}).get(posixpath.basename(path))
            if isinstance(doc, dict) else None)
    if (not isinstance(item, dict) or doc.get("format") != FORMAT
            or doc.get("frame_bytes") != frame):
        raise RuntimeError(f"{where} does not describe {path} in format "
                           f"{FORMAT} with {frame}-byte frames")
    raw = entry.get("raw_size_bytes")
    sizes = item.get("frame_stored_bytes")
    if (not isinstance(raw, int) or raw <= 0
            or item.get("raw_size_bytes") != raw
            or not isinstance(sizes, list)
            or len(sizes) != -(-raw // frame)
            or not all(isinstance(n, int) and n > 0 for n in sizes)
            or sum(sizes) != entry.get("size_bytes")):
        raise RuntimeError(f"{where}'s frame table for {path} disagrees with "
                           f"its manifest row")
    zero = bytearray(len(sizes))
    for run in item.get("zero_runs") or ():
        if (not isinstance(run, list) or len(run) != 2
                or not all(isinstance(x, int) for x in run)
                or run[0] < 0 or run[1] <= 0 or sum(run) > len(sizes)):
            raise RuntimeError(f"{where}: bad zero run {run!r} for {path}")
        zero[run[0]:sum(run)] = b"\x01" * run[1]
    offsets = [0]
    for n in sizes:
        offsets.append(offsets[-1] + n)
    return _Frames(frame, offsets, bytes(zero))


def _walk_plan(source_dir: str, incoming: str,
               subdirs: tuple[str, ...]) -> list[_File]:
    """Every file of ``subdirs``, raw, for a source with no daemon manifest."""
    files = []
    for sub in subdirs:
        root = os.path.join(source_dir, sub)
        for dirpath, _, names in os.walk(root):
            if INDEX_NAME in names:
                raise RuntimeError(
                    f"{dirpath} holds compressed pages but {source_dir} has "
                    f"no {_MANIFEST} to decode them by")
            for name in sorted(names):
                src = os.path.join(dirpath, name)
                files.append(_File(
                    src=src,
                    dest=os.path.join(incoming, sub,
                                      os.path.relpath(src, root)),
                    size=os.stat(src).st_size, sha256=""))
    return files


def plan(source_dir: str, verified_dir: str, incoming: str,
         subdirs: tuple[str, ...]) -> list[_File]:
    """The files of ``subdirs`` under ``source_dir``, from the manifest beside them.

    A compressed row becomes its raw file at ``dest``, read from the ``.zst``
    at ``src``; the index it was described by is consumed, not copied. The
    daemon writes the manifest before the verified marker, so a source with
    no manifest is one placed by hand: its files are listed as they are.
    """
    manifest_path = os.path.join(verified_dir, _MANIFEST)
    if not os.path.exists(manifest_path):
        return _walk_plan(source_dir, incoming, subdirs)
    with open(manifest_path) as handle:
        doc = json.load(handle)
    entries = doc.get("files") if isinstance(doc, dict) else None
    if not isinstance(entries, list):
        raise RuntimeError(f"{manifest_path} has no 'files' list")
    if not all(isinstance(e, dict) for e in entries):
        raise RuntimeError(f"{manifest_path} has a row that is not an object")
    block, indexes = _compression(doc, entries, manifest_path)
    index_docs: dict[str, dict] = {}

    rel = os.path.relpath(source_dir, verified_dir)
    prefix = "" if rel == "." else rel.replace(os.sep, "/") + "/"
    files = []
    for entry in entries:
        path = str(entry.get("path") or "")
        if path in indexes:
            continue
        for sub in subdirs:
            head = f"{prefix}{sub}/"
            if not path.startswith(head):
                continue
            if "raw_path" in entry:
                frames = _frames(entry, block, verified_dir, index_docs)
                src = os.path.join(verified_dir, path)
                stored = os.stat(src).st_size
                if stored != frames.offsets[-1]:
                    raise RuntimeError(
                        f"{src} is {stored} bytes on the mirror, the manifest "
                        f"says {frames.offsets[-1]}")
                raw = entry["raw_path"]
                files.append(_File(
                    src=src,
                    dest=os.path.join(incoming, sub, raw[len(head):]),
                    size=entry["raw_size_bytes"],
                    sha256=str(entry.get("raw_sha256") or "").lower(),
                    frames=frames))
                break
            files.append(_File(
                src=os.path.join(verified_dir, path),
                dest=os.path.join(incoming, sub, path[len(head):]),
                size=int(entry.get("size_bytes") or 0),
                sha256=str(entry.get("sha256") or "").lower()))
            break
    return files


def _ranges(files: list[_File], chunk: int) -> list[_Range]:
    """Every range to copy or decode, largest first.

    A raw file is cut into ``chunk``-byte ranges. A compressed one becomes runs
    of non-zero frames of up to ``chunk`` raw bytes; its zero frames become no
    range at all.
    """
    pieces = []
    for f in files:
        if f.frames is None:
            for start in range(0, f.size, chunk):
                pieces.append(_Range(f, start, min(start + chunk, f.size)))
            continue
        frame, zero = f.frames.frame_bytes, f.frames.zero
        step = max(1, chunk // frame)
        for first in range(0, len(zero), step):
            last = min(first + step, len(zero))
            i = first
            while i < last:
                if zero[i]:
                    i += 1
                    continue
                j = i
                while j < last and not zero[j]:
                    j += 1
                pieces.append(_Range(f, i * frame, min(j * frame, f.size)))
                i = j
    pieces.sort(key=lambda r: r.end - r.start, reverse=True)
    return pieces


def _zero_bytes(f: _File) -> int:
    """Raw bytes of ``f``'s all-zero frames."""
    if f.frames is None:
        return 0
    frame = f.frames.frame_bytes
    return sum(min(frame, f.size - i * frame)
               for i, z in enumerate(f.frames.zero) if z)


def _pwrite_all(fd: int, view: memoryview, offset: int) -> None:
    written = 0
    while written < len(view):
        written += os.pwrite(fd, view[written:], offset + written)


class _Writer:
    """``pwrite`` into one file, through O_DIRECT wherever alignment allows.

    Only a file's tail is ever unaligned, since every range starts on a chunk
    or frame boundary and every write but a range's last fills the whole
    buffer; that tail goes through an ordinary descriptor onto pages no direct
    write touches.
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


def _copy_local(rng: _Range, buf: memoryview, direct: bool = False) -> None:
    """Copy ``rng`` of a raw file from the mirror and write it in place."""
    src = os.open(rng.file.src, os.O_RDONLY)
    out = _Writer(rng.file.dest, direct)
    try:
        offset = rng.start
        while offset < rng.end:
            want = min(len(buf), rng.end - offset)
            filled = 0
            while filled < want:
                got = os.preadv(src, [buf[filled:want]], offset + filled)
                if not got:
                    raise RuntimeError(
                        f"{rng.file.src} ended early on the mirror, at "
                        f"{offset + filled - rng.start} of "
                        f"{rng.end - rng.start} bytes")
                filled += got
            out.write(buf[:want], offset)
            offset += want
    finally:
        os.close(src)
        out.close()


class _Zstd:
    """libzstd's decoder through ctypes, which drops the GIL around each call."""

    def __init__(self, lib):
        size_t, ptr = ctypes.c_size_t, ctypes.c_void_p
        lib.ZSTD_compressBound.argtypes = [size_t]
        lib.ZSTD_compressBound.restype = size_t
        lib.ZSTD_isError.argtypes = [size_t]
        lib.ZSTD_isError.restype = ctypes.c_uint
        lib.ZSTD_getErrorName.argtypes = [size_t]
        lib.ZSTD_getErrorName.restype = ctypes.c_char_p
        lib.ZSTD_createDCtx.restype = ptr
        lib.ZSTD_freeDCtx.argtypes = [ptr]
        lib.ZSTD_decompressDCtx.argtypes = [ptr, ptr, size_t, ptr, size_t]
        lib.ZSTD_decompressDCtx.restype = size_t
        self.lib = lib

    def bound(self, n: int) -> int:
        return int(self.lib.ZSTD_compressBound(n))

    def decode(self, dctx: int, dst: int, cap: int, src: int, n: int) -> int:
        """Decode the one frame at ``src``; raises unless it checks out."""
        got = self.lib.ZSTD_decompressDCtx(dctx, dst, cap, src, n)
        if self.lib.ZSTD_isError(got):
            raise ValueError(self.lib.ZSTD_getErrorName(got).decode())
        return got


_ZSTD: list[_Zstd] = []
_ZSTD_LOCK = threading.Lock()


def _load_zstd() -> _Zstd:
    """libzstd, loaded once. The backend image ships ``libzstd.so.1``."""
    with _ZSTD_LOCK:
        if not _ZSTD:
            errors = []
            for name in dict.fromkeys((_LIBZSTD_SONAME,
                                       ctypes.util.find_library("zstd"))):
                if not name:
                    continue
                try:
                    _ZSTD.append(_Zstd(ctypes.CDLL(name)))
                    break
                except (OSError, AttributeError) as exc:
                    errors.append(f"{name}: {exc}")
            else:
                raise RuntimeError(
                    "this skeleton is compressed and libzstd cannot be loaded "
                    f"({'; '.join(errors) or 'not found'})")
        return _ZSTD[0]


def _addr(buf) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


class _Decoder:
    """One worker's decode context and stored-bytes buffer, made on first use."""

    def __init__(self, zstd: _Zstd):
        self.zstd = zstd
        self.dctx = zstd.lib.ZSTD_createDCtx()
        if not self.dctx:
            raise MemoryError("libzstd could not allocate a decode context")
        self.frame = 0
        self.inbuf = self.outbuf = None
        self.cpu_s = 0.0

    def buffers(self, frame: int, buf: memoryview
                ) -> tuple[memoryview, memoryview]:
        """``(stored, raw)`` buffers for frames of ``frame`` bytes: the
        worker's own page-aligned ``buf`` holds the raw side when it fits."""
        if frame != self.frame:
            raw = buf if frame <= len(buf) else memoryview(mmap.mmap(-1, frame))
            batch = len(raw) // frame
            self.inbuf = memoryview(mmap.mmap(-1, batch * self.zstd.bound(frame)))
            self.outbuf = raw[:batch * frame]
            self.frame = frame
        return self.inbuf, self.outbuf

    def close(self) -> None:
        self.zstd.lib.ZSTD_freeDCtx(self.dctx)


def _decode_local(rng: _Range, buf: memoryview, dec: _Decoder,
                  direct: bool) -> int:
    """Decode ``rng``'s frames from the mirror and write them in place.

    Every frame is read whole, must carry a checksum, and must decode, with
    that checksum verified, to exactly its share of the raw file. Returns the
    stored bytes read.
    """
    f, fr = rng.file, rng.file.frames
    frame = fr.frame_bytes
    inbuf, outbuf = dec.buffers(frame, buf)
    in_addr, out_addr = _addr(inbuf), _addr(outbuf)
    batch = len(outbuf) // frame
    first, end = rng.start // frame, -(-rng.end // frame)
    src = os.open(f.src, os.O_RDONLY)
    out = _Writer(f.dest, direct)
    read = 0
    try:
        k = first
        while k < end:
            m = min(k + batch, end)
            base, span = fr.offsets[k], fr.offsets[m] - fr.offsets[k]
            if span > len(inbuf):
                raise RuntimeError(f"{f.src}: frames {k}-{m - 1} take {span} "
                                   f"bytes, more than {frame}-byte frames can")
            filled = 0
            while filled < span:
                got = os.preadv(src, [inbuf[filled:span]], base + filled)
                if not got:
                    raise RuntimeError(
                        f"{f.src} ended early on the mirror, at stored byte "
                        f"{base + filled} (frame {k}+)")
                filled += got
            cpu = time.thread_time()
            for j in range(k, m):
                at = fr.offsets[j] - base
                head = inbuf[at:at + 5]
                want = min(frame, f.size - j * frame)
                try:
                    if (len(head) < 5 or head[:4] != _ZSTD_MAGIC
                            or not head[4] & _ZSTD_CHECKSUM_BIT
                            or head[4] & _ZSTD_DICT_BITS):
                        raise ValueError("not a checksummed, dictionary-free "
                                         "zstd frame")
                    got = dec.zstd.decode(
                        dec.dctx, out_addr + (j - k) * frame, want,
                        in_addr + at, fr.offsets[j + 1] - fr.offsets[j])
                    if got != want:
                        raise ValueError(f"decoded {got} bytes, expected {want}")
                except ValueError as exc:
                    raise RuntimeError(
                        f"{f.src}: frame {j} (raw offset {j * frame}) fails "
                        f"to decode: {exc}") from None
            dec.cpu_s += time.thread_time() - cpu
            out.write(outbuf[:min(m * frame, f.size) - k * frame], k * frame)
            read += span
            k = m
    finally:
        os.close(src)
        out.close()
    return read


def _preallocate(files: list[_File], incoming: str,
                 subdirs: tuple[str, ...]) -> None:
    """Every file at its full size up front, so a short disk fails here."""
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
           subdirs: tuple[str, ...]) -> GatherStats:
    """Assemble ``subdirs`` of ``source_dir`` under ``incoming`` from the mirror.

    Raises if a file cannot be read or decoded, or fails verification; the
    caller treats that like a failed copy.
    """
    t0 = time.perf_counter()
    chunk = _env_int(CHUNK_ENV, DEFAULT_CHUNK, _parse_bytes)
    chunk = -(-chunk // _ALIGN) * _ALIGN
    threads = _env_int(THREADS_ENV, DEFAULT_THREADS)
    verify = _verify_mode()
    direct = (os.environ.get(DIRECT_ENV, "").strip() != "0"
              and hasattr(os, "O_DIRECT"))

    files = plan(source_dir, verified_dir, incoming, subdirs)
    compressed = [f for f in files if f.frames is not None]
    zstd = _load_zstd() if compressed else None
    if verify == "sha256" and not all(f.sha256 for f in files):
        raise RuntimeError(f"{VERIFY_ENV}=sha256, but {verified_dir} has no "
                           f"manifest sha256 for every file")
    _preallocate(files, incoming, subdirs)

    work: queue.Queue[_Range] = queue.Queue()
    for rng in _ranges(files, chunk):
        work.put(rng)
    lock = threading.Lock()
    counts = {"copied": 0, "ranges": 0, "stored": 0, "written": 0}
    errors: list[BaseException] = []
    decoders: list[_Decoder] = []

    def worker() -> None:
        # Page-aligned, as O_DIRECT requires of the buffer address.
        buf = memoryview(mmap.mmap(-1, _BUF_BYTES))
        dec = None
        while not errors:
            try:
                rng = work.get_nowait()
            except queue.Empty:
                return
            try:
                if rng.file.frames is None:
                    _copy_local(rng, buf, direct)
                    with lock:
                        counts["copied"] += rng.end - rng.start
                        counts["ranges"] += 1
                    continue
                if dec is None:
                    dec = _Decoder(zstd)
                    with lock:
                        decoders.append(dec)
                read = _decode_local(rng, buf, dec, direct)
                with lock:
                    counts["stored"] += read
                    counts["written"] += rng.end - rng.start
                    counts["ranges"] += 1
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
    decode_cpu_s = sum(d.cpu_s for d in decoders)
    for d in decoders:
        d.close()
    if errors:
        raise RuntimeError(f"gathering {source_dir} failed: {errors[0]}") \
            from errors[0]

    total = sum(f.size for f in files)
    zero = sum(_zero_bytes(f) for f in compressed)
    if counts["copied"] + counts["written"] + zero != total:
        raise RuntimeError(f"gathered {counts['copied'] + counts['written']} "
                           f"bytes and skipped {zero} zero bytes of {total} "
                           f"bytes of {source_dir}")
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
        files=len(files), total_bytes=total, copied_bytes=counts["copied"],
        ranges=counts["ranges"], flows=flows, direct=direct,
        elapsed_s=time.perf_counter() - t0,
        stored_bytes=sum(f.frames.offsets[-1] for f in compressed),
        written_bytes=counts["written"], zero_bytes=zero,
        compressed_files=len(compressed), decode_cpu_s=decode_cpu_s)
