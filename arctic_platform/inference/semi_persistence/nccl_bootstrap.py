"""Reset NCCL's process-global bootstrap interface cache.

NCCL picks its bootstrap interface once per process (``bootstrapNetInit`` in
``bootstrap.cc``) and never again: ``bootstrapNetInitDone`` latches to 1 and
``bootstrapNetIfAddr`` keeps the address it found. Aborting every communicator
does not clear either. A process restored by CRIU onto a pod with a different
IP therefore keeps advertising the dump-time address to its peers, which is
harmless on loopback and fatal across nodes.

Clearing the latch alone is not enough: ``bootstrapNetInit`` is only ever
called from ``ncclInit``'s ``std::call_once``, so nothing would run discovery
again and the next communicator would bootstrap from the zeroed address
(``ncclSocketInit: ... family 0``). ``reset`` therefore calls
``bootstrapNetInit`` itself, under the caller's current
``NCCL_SOCKET_IFNAME``.

None of these symbols is exported, so they are found through the ``.symtab`` of
the ``libnccl.so.2`` that is actually mapped into this process and reached
through its load base. Builds without a ``.symtab`` cannot be reset; ``reset``
raises rather than leaving a stale address in place.
"""
import ctypes
import mmap
import os
import socket
import struct

_INIT_DONE = "_ZL20bootstrapNetInitDone"
_IF_ADDR = "_ZL18bootstrapNetIfAddr"
_IF_NAME = "_ZL18bootstrapNetIfName"
_SOCKET_NET_IFS = "_ZL10ncclNetIfs"
_NET_INIT = "_Z16bootstrapNetInitv"
_SYMBOLS = (_INIT_DONE, _IF_ADDR, _IF_NAME, _SOCKET_NET_IFS, _NET_INIT)

# Libraries whose presence after a full NCCL teardown means a net plugin was
# not released, and device nodes that must not stay open or mapped.
_PLUGIN_MARKERS = ("libnccl-net", "libnccl-tuner", "libnccl-gin", "libfabric",
                   "libefa", "libibverbs", "/dev/infiniband/")

_SHT_SYMTAB = 2
_PT_LOAD = 1


def _maps():
    """``(start, offset, path)`` for every file-backed mapping of this process."""
    rows = []
    with open("/proc/self/maps") as handle:
        for line in handle:
            parts = line.split(None, 5)
            if len(parts) < 6:
                continue
            start = int(parts[0].split("-")[0], 16)
            rows.append((start, int(parts[2], 16), parts[5].strip()))
    return rows


def mapped_nccl_libraries():
    """Real paths of every ``libnccl.so*`` mapped into this process."""
    return sorted({os.path.realpath(path) for _, _, path in _maps()
                   if os.path.basename(path).startswith("libnccl.so")})


def plugin_mappings():
    """Mapped files that a released NCCL net plugin should have taken with it."""
    return sorted({path for _, _, path in _maps()
                   if any(marker in path for marker in _PLUGIN_MARKERS)})


def _symbols(path, names):
    """``{name: (st_value, st_size)}`` from the ELF64 ``.symtab`` of *path*."""
    wanted = {name.encode(): name for name in names}
    with open(path, "rb") as handle, \
            mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as elf:
        if elf[:4] != b"\x7fELF" or elf[4] != 2:
            raise RuntimeError(f"{path}: not an ELF64 object")
        e_shoff, = struct.unpack_from("<Q", elf, 0x28)
        e_shentsize, e_shnum = struct.unpack_from("<HH", elf, 0x3A)
        sections = [struct.unpack_from("<IIQQQQIIQQ", elf,
                                       e_shoff + i * e_shentsize)
                    for i in range(e_shnum)]
        symtab = next((s for s in sections if s[1] == _SHT_SYMTAB), None)
        if symtab is None:
            raise RuntimeError(f"{path}: no .symtab (stripped build)")
        strtab = sections[symtab[6]]
        str_off, str_size = strtab[4], strtab[5]
        name_at = {}
        for raw, name in wanted.items():
            idx = elf.find(b"\0" + raw + b"\0", str_off, str_off + str_size)
            if idx >= 0:
                name_at[idx + 1 - str_off] = name
        found = {}
        sym_off, sym_size, sym_ent = symtab[4], symtab[5], symtab[9]
        for off in range(sym_off, sym_off + sym_size, sym_ent):
            st_name, = struct.unpack_from("<I", elf, off)
            name = name_at.get(st_name)
            if name is not None:
                st_value, st_size = struct.unpack_from("<QQ", elf, off + 8)
                found[name] = (st_value, st_size)
                if len(found) == len(name_at):
                    break
        return found


def _first_load_vaddr(path):
    with open(path, "rb") as handle:
        header = handle.read(64)
        e_phoff, = struct.unpack_from("<Q", header, 0x20)
        e_phentsize, e_phnum = struct.unpack_from("<HH", header, 0x36)
        handle.seek(e_phoff)
        table = handle.read(e_phentsize * e_phnum)
    for i in range(e_phnum):
        p_type, _, p_offset, p_vaddr = struct.unpack_from(
            "<IIQQ", table, i * e_phentsize)
        if p_type == _PT_LOAD and p_offset == 0:
            return p_vaddr
    raise RuntimeError(f"{path}: no PT_LOAD at file offset 0")


def _load_base(path):
    real = os.path.realpath(path)
    starts = [start for start, offset, mapped in _maps()
              if offset == 0 and os.path.realpath(mapped) == real]
    if not starts:
        raise RuntimeError(f"{path} is not mapped into this process")
    return min(starts) - _first_load_vaddr(real)


def _resolve(path):
    symbols = _symbols(path, _SYMBOLS)
    missing = [name for name in (_INIT_DONE, _IF_ADDR, _NET_INIT)
               if name not in symbols]
    if missing:
        raise RuntimeError(f"{path}: symbols {missing} not found")
    base = _load_base(path)
    return {name: base + value for name, (value, _) in symbols.items()}


def _read_address(addr):
    raw = ctypes.string_at(addr, 28)
    family, = struct.unpack_from("<H", raw, 0)
    if family == socket.AF_INET:
        return socket.inet_ntop(socket.AF_INET, raw[4:8])
    if family == socket.AF_INET6:
        return socket.inet_ntop(socket.AF_INET6, raw[8:24])
    return None


def bootstrap_state():
    """Per mapped ``libnccl``: whether the cache is latched and what it holds."""
    states = []
    for path in mapped_nccl_libraries():
        addrs = _resolve(path)
        name = None
        if _IF_NAME in addrs:
            name = ctypes.string_at(addrs[_IF_NAME]).decode(errors="replace")
        states.append({
            "library": path,
            "init_done": ctypes.c_int.from_address(addrs[_INIT_DONE]).value,
            "ifname": name or None,
            "address": _read_address(addrs[_IF_ADDR]),
        })
    return states


def reset():
    """Re-run bootstrap interface discovery under the current environment.

    Must not race a communicator init in this process; call it only with every
    communicator torn down. Raises if no ``libnccl`` is mapped, a mapped one
    cannot be resolved, or discovery fails.
    """
    libraries = mapped_nccl_libraries()
    if not libraries:
        raise RuntimeError("no libnccl is mapped into this process")
    for path in libraries:
        addrs = _resolve(path)
        ctypes.c_int.from_address(addrs[_INIT_DONE]).value = 0
        ctypes.memset(addrs[_IF_ADDR], 0, 28)
        if _SOCKET_NET_IFS in addrs:
            ctypes.c_int.from_address(addrs[_SOCKET_NET_IFS]).value = -1
        result = ctypes.CFUNCTYPE(ctypes.c_int)(addrs[_NET_INIT])()
        if result != 0:
            raise RuntimeError(
                f"{path}: bootstrapNetInit returned ncclResult {result}")
    return bootstrap_state()
