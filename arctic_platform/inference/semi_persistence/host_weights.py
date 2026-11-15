"""Large pinned host parameters, kept out of the CRIU image.

DeepSeek-V4.1-Flash keeps its Engram tables on the host: four pinned CPU
parameters per rank (``layers.{1,14}.engram.embed_tokens.{weight,
weight_scale_inv}``, 23.6 GiB) that its kernels read through their UVA address.
Like every parameter they are already in the weight set, so the 33 GiB per rank
CRIU stores for them (torch's pinned allocator rounds each up to a power of
two) is a second copy.

``rehome`` runs at the end of ``SemipGPUWorker.load_model``, before any graph is
captured.  It moves each qualifying parameter into a private anonymous mapping
owned here and registers it with ``cudaHostRegister``, so the device pointer
that FULL graphs bake in is the mapping's own host address.  ``release`` runs at
the dump, after ``save_weights``: a parameter unchanged since ``rehome`` is
unregistered, its pages are dropped and the mapping is made PROT_NONE, so CRIU
keeps an empty mapping at the same address.  At the restore ``vllm_child`` reads
the parameter's shard ranges back into the mapping, and ``finish_refill``
registers it again and checks its checksum.  The weight set does not change.

``SEMIP_THIN_PINNED_PARAMS=0`` leaves every parameter where vLLM put it.
"""
from __future__ import annotations

import collections
import ctypes
import inspect
import mmap
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

FLAG_ENV = "SEMIP_THIN_PINNED_PARAMS"
MIN_BYTES = 256 * 2**20
_COPY_THREADS = 8
# Words per checksum chunk, as in vllm_child._semip_digest.
_DIGEST_CHUNK = 1 << 22
_PROT_NONE = 0
_MAP_NORESERVE = getattr(mmap, "MAP_NORESERVE",
                         0x4000 if sys.platform.startswith("linux") else 0)
# Paths in /proc/self/maps that are not a file some other process can map by
# name: torch's pinned blocks come from the driver.
_ANONYMOUS_PATHS = ("/memfd:", "/dev/zero", "/dev/nvidia")

_CUDART = None
_LIBC = None


def enabled():
    return os.environ.get(FLAG_ENV, "1") != "0"


def _cudart():
    global _CUDART
    if _CUDART is None:
        lib = ctypes.CDLL("libcudart.so")
        lib.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        lib.cudaHostRegister.restype = ctypes.c_int
        lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        lib.cudaHostUnregister.restype = ctypes.c_int
        lib.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                                 ctypes.c_void_p, ctypes.c_uint]
        lib.cudaHostGetDevicePointer.restype = ctypes.c_int
        _CUDART = lib
    return _CUDART


def _register(ptr, nbytes):
    """``cudaHostRegister``'s error code, 0 on success."""
    return _cudart().cudaHostRegister(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes),
                                      ctypes.c_uint(0))


def _unregister(ptr):
    rc = _cudart().cudaHostUnregister(ctypes.c_void_p(ptr))
    if rc != 0:
        raise RuntimeError(f"cudaHostUnregister(0x{ptr:x}) failed with cudaError={rc}")


def _device_pointer(ptr):
    out = ctypes.c_void_p()
    rc = _cudart().cudaHostGetDevicePointer(ctypes.byref(out), ctypes.c_void_p(ptr),
                                            ctypes.c_uint(0))
    return None if rc != 0 else (out.value or 0)


def _protect(ptr, nbytes, prot):
    global _LIBC
    if _LIBC is None:
        _LIBC = ctypes.CDLL(None, use_errno=True)
        _LIBC.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        _LIBC.mprotect.restype = ctypes.c_int
    if _LIBC.mprotect(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), prot) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"mprotect(0x{ptr:x}, {nbytes}, {prot}): {os.strerror(err)}")


def _drop_pages(mapping):
    mapping.madvise(mmap.MADV_DONTNEED)


def _pinned(tensor):
    return tensor.is_pinned()


def _uva_view(tensor):
    """The CUDA tensor aliasing a registered host tensor, as Engram reads it.

    Only for a tensor ``is_pinned()`` says is registered: for anything else
    vLLM silently returns a view of a private pinned copy."""
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    return get_accelerator_view_from_cpu_tensor(tensor)


def _mappings():
    """``[(start, end, path)]`` from /proc/self/maps, or None where it cannot be read."""
    try:
        with open("/proc/self/maps") as f:
            lines = f.readlines()
    except OSError:
        return None
    out = []
    for line in lines:
        fields = line.split(None, 5)
        lo, _, hi = fields[0].partition("-")
        out.append((int(lo, 16), int(hi, 16),
                    fields[5].strip() if len(fields) > 5 else ""))
    return out


def _host_reserved_bytes():
    try:
        return int(torch.cuda.host_memory_stats()["reserved_bytes.current"])
    except Exception:  # noqa: BLE001 - reporting only; older torch has no stats
        return None


def _host_empty_cache():
    """Hand torch's cached pinned blocks back to the driver."""
    fn = getattr(torch._C, "_host_emptyCache", None)
    if fn is not None:
        fn()


def digest(view):
    """Content checksum of a tensor, computed where it lives: the sum of its
    words and their position-weighted sum, both mod 2**64.  The same checksum
    as ``vllm_child._semip_digest``, which this module cannot import."""
    flat = view.reshape(-1)
    if flat.numel() % 4 == 0:
        flat = flat.view(torch.int32)
    n = flat.numel()
    total = torch.zeros((), dtype=torch.int64, device=flat.device)
    weighted = torch.zeros((), dtype=torch.int64, device=flat.device)
    pos = torch.arange(1, min(n, _DIGEST_CHUNK) + 1, dtype=torch.int64,
                       device=flat.device)
    for start in range(0, n, _DIGEST_CHUNK):
        chunk = flat[start:start + _DIGEST_CHUNK].to(torch.int64)
        s = chunk.sum()
        total += s
        weighted += s * start + chunk.mul_(pos[:chunk.numel()]).sum()
    return int(total.item()), int(weighted.item())


def _models(worker):
    mr = worker.model_runner
    models = [("main", mr.model)]
    drafter = getattr(mr, "drafter", None)
    dm = getattr(drafter, "model", None) if drafter is not None else None
    if dm is not None:
        models.append(("drafter", dm))
    return models


def _span(tensor):
    """``(start, end)`` of a tensor's storage, or None for one with no memory."""
    try:
        storage = tensor.untyped_storage()
        lo, n = storage.data_ptr(), storage.nbytes()
    except (RuntimeError, NotImplementedError):  # fake / meta tensors
        return None
    return (lo, lo + n) if n and lo else None


def _disqualified(param, min_bytes, maps):
    """Why a large pinned CPU parameter cannot move, ``""`` when it can, and
    None for a parameter that is not one (on the GPU, or small)."""
    data = param.data
    if data.device.type != "cpu":
        return None
    span = _span(data)
    if span is None or span[1] - span[0] < min_bytes:
        return None
    if not _pinned(data):
        return "not pinned"
    if (data.storage_offset() != 0 or not data.is_contiguous()
            or data.nbytes != span[1] - span[0]):
        return "a view of a larger storage"
    if maps is None:
        return "/proc/self/maps unreadable"
    for lo, hi, path in maps:
        if lo <= span[0] < hi:
            if path.startswith("/") and not path.startswith(_ANONYMOUS_PATHS):
                return f"file-backed ({path})"
            return ""
    return "address not in /proc/self/maps"


def _aliases(models, candidates, max_depth=8):
    """``{name: path}`` of the candidates some other tensor overlaps.

    Every parameter and buffer, and every tensor a module reaches through its
    attributes -- containers, dataclasses and plain objects too, breadth first
    -- is compared by storage address, so a UVA view (whose device address is
    the host address), a packed parent or a second CPU tensor over the same
    memory all count.  Only the parameter object itself does not: ``rehome``
    swaps its data, which no other reference would follow."""
    spans = [(span, name, param) for name, param in candidates
             for span in [_span(param.data)]]
    found = {}

    def check(path, tensor):
        own = _span(tensor)
        if own is None:
            return
        for (lo, hi), name, param in spans:
            if tensor is not param and name not in found and own[0] < hi and lo < own[1]:
                found[name] = path

    queue = collections.deque()
    for ns, model in models:
        for n, t in [*model.named_parameters(), *model.named_buffers()]:
            check(f"{ns}:{n}", t)
        for mod_name, mod in model.named_modules():
            for attr, value in vars(mod).items():
                if attr not in ("_parameters", "_buffers", "_modules"):
                    path = f"{mod_name}.{attr}" if mod_name else attr
                    queue.append((f"{ns}:{path}", value, 1))
    visited = set()
    while queue:
        path, value, depth = queue.popleft()
        if isinstance(value, torch.Tensor):
            check(path, value)
            continue
        if (depth >= max_depth or id(value) in visited
                or isinstance(value, (torch.nn.Module, type, str, bytes))
                or inspect.ismodule(value) or inspect.isroutine(value)):
            continue
        if isinstance(value, dict):
            items = [(str(k), v) for k, v in value.items()]
        elif isinstance(value, (list, tuple, set, frozenset)):
            items = list(enumerate(value))
        elif hasattr(value, "__dict__"):
            items = list(vars(value).items())
        else:
            continue
        visited.add(id(value))
        queue.extend((f"{path}.{key}", item, depth + 1) for key, item in items)
    return found


def _copy(dst, src, threads=_COPY_THREADS):
    """``dst.copy_(src)`` split across threads; ``copy_`` drops the GIL and the
    worker runs one OpenMP thread."""
    n = dst.numel()
    step = max(1, -(-n // threads))
    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(lambda a: dst[a:a + step].copy_(src[a:a + step]),
                      range(0, n, step)))


class _Skip(Exception):
    pass


class HostEntry:
    """One rehomed parameter: its mapping and where it stands.

    ``registered`` -- resident and registered, what serving needs.
    ``dropped`` -- released at the dump: unregistered, no pages, PROT_NONE.
    Neither -- unregistered with its pages (unpinned, or kept in the image).
    ``expected`` is the checksum the contents must have when registered."""

    def __init__(self, name, param, mapping, flat):
        self.name = name
        self.param = param
        self.mapping = mapping
        self.flat = flat
        self.nbytes = flat.numel()
        self.ptr = flat.data_ptr()
        self.registered = True
        self.dropped = False
        self.expected = None

    def register(self):
        rc = _register(self.ptr, self.nbytes)
        if rc != 0:
            raise RuntimeError(f"{self.name}: cudaHostRegister of {self.nbytes} B "
                               f"failed with cudaError={rc} (memlock limit?)")
        self.registered = True
        dptr = _device_pointer(self.ptr)
        if dptr != self.ptr or not _pinned(self.param.data):
            raise RuntimeError(
                f"{self.name}: registered at 0x{self.ptr:x} but the device pointer "
                f"is {dptr if dptr is None else hex(dptr)} (pinned="
                f"{_pinned(self.param.data)}); captured graphs read the host address")

    def unregister(self):
        _unregister(self.ptr)
        self.registered = False

    def checksum(self):
        return digest(_uva_view(self.param.data))

    def verify_unchanged(self):
        """Raise unless the parameter still holds what it was rehomed with.

        The weight-sync receiver calls this after every sync of base weights
        (``param._semip_verify_unchanged``): a rehomed parameter is refilled
        from the dump-time weight set after every restore, so a sync that
        updates it in place is not supported."""
        if self.param.data.data_ptr() != self.ptr:
            raise RuntimeError(
                f"weight sync replaced the storage of {self.name}, which semi_p "
                f"keeps in a registered host mapping that captured graphs read; "
                f"run with {FLAG_ENV}=0 to sync it")
        if not self.registered:
            raise RuntimeError(f"{self.name} is not registered (released or unpinned)")
        if self.checksum() != self.expected:
            raise RuntimeError(
                f"weight sync wrote {self.name}, a parameter semi_p keeps out of "
                f"the CRIU image and refills from the dump-time weight set; "
                f"in-place updates of it are not supported (run with {FLAG_ENV}=0)")

    def view(self):
        return memoryview(self.mapping)

    def open(self):
        """Make a dropped mapping writable again, for the refill."""
        _protect(self.ptr, self.nbytes, mmap.PROT_READ | mmap.PROT_WRITE)


def _rehome_one(name, param):
    old = param.data
    n = old.nbytes
    mapping = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | _MAP_NORESERVE)
    flat = torch.from_numpy(np.frombuffer(mapping, dtype=np.uint8))
    try:
        _copy(flat, old.reshape(-1).view(torch.uint8))
        rc = _register(flat.data_ptr(), n)
        if rc != 0:
            raise _Skip(f"cudaHostRegister failed with cudaError={rc}")
        dptr = _device_pointer(flat.data_ptr())
        if dptr != flat.data_ptr() or not _pinned(flat):
            _unregister(flat.data_ptr())
            raise _Skip(f"device pointer {dptr if dptr is None else hex(dptr)} != "
                        f"host 0x{flat.data_ptr():x} (pinned={_pinned(flat)})")
    except BaseException:
        flat = None
        try:
            mapping.close()
        except BufferError:
            pass
        raise
    param.data = flat.view(old.dtype).view(old.shape)
    entry = HostEntry(name, param, mapping, flat)
    entry.expected = entry.checksum()
    param._semip_verify_unchanged = entry.verify_unchanged
    return entry


class HostWeights(dict):
    """``{namespaced name: HostEntry}``, kept on the worker as ``_semip_host``."""

    def __init__(self):
        super().__init__()
        self.skipped = []
        self.torch_reserved = (None, None)
        self.seconds = 0.0

    def down(self):
        return any(not e.registered for e in self.values())

    def nbytes(self):
        return sum(e.nbytes for e in self.values())

    def census(self):
        return {"n": len(self), "bytes": self.nbytes(), "names": sorted(self),
                "skipped": [list(s) for s in self.skipped], "down": self.down()}

    def unpin(self):
        for e in self.values():
            if e.registered:
                e.unregister()
        return {"n": len(self), "bytes": self.nbytes(), "down": self.down()}

    def repin(self):
        for e in self.values():
            if not e.registered and not e.dropped:
                e.register()
        return {"n": len(self), "bytes": self.nbytes(), "down": self.down()}

    def release(self):
        """At the dump: drop every parameter unchanged since ``rehome``.

        One that changed keeps its pages, so it rides the image as it did
        before this module existed, and is reported as kept.  Either way it
        ends unregistered, since nothing registered may reach cuda_checkpoint."""
        t0 = time.perf_counter()
        released, kept = [], []
        for e in self.values():
            if not e.registered:
                continue
            now = e.checksum()
            e.unregister()
            if now == e.expected:
                _drop_pages(e.mapping)
                _protect(e.ptr, e.nbytes, _PROT_NONE)
                e.dropped = True
                released.append([e.name, e.nbytes])
            else:
                e.expected = now
                kept.append([e.name, e.nbytes])
        return {"n": len(self), "bytes": self.nbytes(), "released": released,
                "kept": kept, "seconds": time.perf_counter() - t0,
                "down": self.down()}

    def finish_refill(self, refilled, read_s=0.0):
        """Register every unregistered entry and check it holds what it should.

        ``refilled`` names the dropped entries whose bytes the caller has just
        read back in.  Raises on the first entry that cannot be registered at
        its own address or whose checksum is wrong, naming it."""
        for name in refilled:
            self[name].dropped = False
        stale = [e.name for e in self.values() if e.dropped]
        if stale:
            raise RuntimeError(f"host-resident parameter(s) {stale} were released "
                               f"at the dump and not refilled")
        t0 = time.perf_counter()
        todo = [e for e in self.values() if not e.registered]
        for e in todo:
            e.register()
        t1 = time.perf_counter()
        for e in todo:
            got = e.checksum()
            if got != e.expected:
                raise RuntimeError(
                    f"{e.name}: checksum after refill {got} != {e.expected} "
                    f"recorded at the dump")
        return {"n": len(self), "refilled": len(refilled),
                "refilled_bytes": sum(self[n].nbytes for n in refilled),
                "registered": len(todo), "read_s": read_s,
                "register_s": t1 - t0, "verify_s": time.perf_counter() - t1,
                "down": self.down()}


def rehome(worker, host, min_bytes=MIN_BYTES):
    """Move each qualifying parameter of ``worker``'s models into ``host``.

    Qualifying: on the CPU, pinned, at least ``min_bytes``, the whole of its
    storage, not backed by a named file, and overlapped by no other tensor
    (``_aliases``).  A parameter that fails a check, or whose mapping cannot be
    registered at its own address, stays where it is and is listed in
    ``host.skipped`` with the reason.  One at a time, so the transient cost is
    one parameter's size."""
    t0 = time.perf_counter()
    models = _models(worker)
    maps = _mappings()
    candidates = []
    for ns, model in models:
        for n, p in model.named_parameters():
            why = _disqualified(p, min_bytes, maps)
            if why:
                host.skipped.append((f"{ns}:p:{n}", why))
            elif why is not None:
                candidates.append((f"{ns}:p:{n}", p))
    aliased = _aliases(models, candidates)
    before = _host_reserved_bytes()
    for name, param in candidates:
        if name in aliased:
            host.skipped.append((name, f"aliased by {aliased[name]}"))
            continue
        try:
            host[name] = _rehome_one(name, param)
        except _Skip as exc:
            host.skipped.append((name, str(exc)))
            continue
        _host_empty_cache()
    if host:
        torch.cuda.empty_cache()
    host.torch_reserved = (before, _host_reserved_bytes())
    host.seconds = time.perf_counter() - t0
    return host


def _gib(n):
    return "?" if n is None else f"{n / 2**30:.2f}"


def describe(host):
    """The one line ``rehome``'s outcome gets in the log."""
    before, after = host.torch_reserved
    return (f"rehomed {len(host)} param(s) {_gib(host.nbytes())} GiB in "
            f"{host.seconds:.1f}s (torch pinned reserved {_gib(before)} -> "
            f"{_gib(after)} GiB) skipped {len(host.skipped)} {host.skipped}"
            + (" uva_identity=ok" if host else ""))
