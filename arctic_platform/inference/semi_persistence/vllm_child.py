"""vLLM child process loop.

Spawned by the worker process.  Owns CUDA and vLLM.
Reads (cmd, kwargs) from a pipe, puts results on result_queue.

Init loads real weights (load_format=auto) so that vLLM runs
process_weights_after_loading and produces its internal kernel format
(Marlin-packed for GPTQ, cutlass layout for FP8, plain tensors for
BF16).

The generate path drives LLMEngine directly via add_request() + step()
instead of using the blocking LLM.generate().  This allows the child
to accept new generate requests (and other commands) while the engine
is actively decoding, enabling concurrent request handling without
asyncio or extra threads.

Attach allocates CPU memory sized to model.named_parameters().  Stage
snapshots the post-processed GPU parameters into that buffer.
plan_restore_weights walks the param index once and caches a chunk plan
(chunk_lo, chunk_hi, members) bounded by max_buffer_bytes.
restore_weights then loops over the cached plan: per chunk, copy a
slice of host memory into a single reused GPU staging buffer and
scatter into model parameters by name.  If no plan is cached,
restore_weights falls back to a single-chunk path.

All of that state lives on each vLLM worker (``worker._semip_*``) and
runs there via ``collective_rpc``, not in this process.  At TP>1 the
callable is cloudpickled into every worker subprocess, so a buffer held
here would be copied by value per worker and its writes discarded; each
rank also owns a different shard of the parameters.  The same code path
serves TP=1, where the single worker is in-process.
"""
import ctypes, json, os, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor


def _drop_caps_for_portable_image():
    """Zero this process's Linux capabilities so its CRIU image restores on
    nodes whose capability bounding set can't grant them.

    Must run in the spawned vLLM child BEFORE ``import torch`` -- torch/vLLM
    spawn many background threads (cuda, jemalloc, hf-xet, gloo, ...) and
    CRIU records credentials *per thread*.  A thread inherits the creating
    thread's capabilities, so dropping here (before any are created) yields
    an image where every task records an empty cap set; restore_creds()
    (capset) then trivially succeeds instead of failing with EPERM on a
    node lacking CAP_SYS_ADMIN etc.  (Dropping later, e.g. in
    prepare_criu_dump, would only affect the main thread and leave the
    others with full caps.)

    Gated by the internal _SEMIP_CHILD_DROP_CAPS signal (set transiently by
    the worker only across this child's spawn -- NOT a user-facing flag; the
    user-facing switch is SEMIP_UNPRIVILEGED) so ONLY the child drops -- the
    parent worker keeps its caps to seize this child with ``criu``.  Safe for
    inference: host pinning uses RLIMIT_MEMLOCK (unlimited on GPU nodes), not
    CAP_IPC_LOCK, and all sockets bind to unprivileged ports.

    NO_NEW_PRIVS is set for the restore, not for this process.  CRIU's
    restore_creds() trims the bounding set by calling PR_CAPBSET_DROP for
    every capability *absent from the recorded cap_bnd*, without first
    checking whether it is already absent from its own -- and the kernel's
    cap_prctl_drop() tests CAP_SETPCAP before it tests anything else, so an
    already-dropped capability still returns EPERM.  On a pod granting only
    CAP_CHECKPOINT_RESTORE + CAP_SYS_PTRACE that is fatal for all ~200 tasks
    (restorer.c:317 "Unable to drop capability 0", then BUG at
    restorer.c:820).

    `setcap cap_setpcap` on criu cannot fix it, and makes things worse rather
    than merely not better.  CAP_SETPCAP is outside that bounding set, and
    bprm_caps_from_vfs_caps() computes pP' = (pB & fP) | (pI & fI) and then
    returns EPERM if any file-permitted bit did not survive -- whenever the
    file's effective bit is set, which the +e we require for a root worker
    does.  So a criu carrying a capability the pod cannot grant fails to
    execve at all.  CRIU's own escape hatch is the image recording
    no_new_privs with an empty permitted set, in which case the EPERM
    degrades to a warning -- so set it here, where it lands in every task's
    recorded creds.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_CAPBSET_DROP = 24
        PR_CAP_AMBIENT = 47
        PR_CAP_AMBIENT_CLEAR_ALL = 4
        PR_SET_DUMPABLE = 4
        PR_SET_NO_NEW_PRIVS = 38
        # Drop the bounding set first -- it needs CAP_SETPCAP, which the
        # capset() below removes.  EINVAL past the last valid cap is ignored.
        # These calls are expected to fail with EPERM on a pod that never had
        # CAP_SETPCAP; the image then records the pod's own bounding set, and
        # the NO_NEW_PRIVS below is what makes that restorable.
        for cap in range(64):
            libc.prctl(PR_CAPBSET_DROP, cap, 0, 0, 0)
        libc.prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0)
        # Needs no privilege, is irreversible, and is inherited by every
        # thread and child forked after it -- which is why it belongs here,
        # before torch spawns any: CRIU dumps creds per thread.  It only
        # forbids gaining privilege through execve (setuid/setgid binaries and
        # file capabilities), and nothing the child execs wants that.  It must
        # not leak to the worker, which execs a file-capability `criu`.
        nnp = libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)

        class _CapHeader(ctypes.Structure):
            _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

        class _CapData(ctypes.Structure):
            _fields_ = [("effective", ctypes.c_uint32),
                        ("permitted", ctypes.c_uint32),
                        ("inheritable", ctypes.c_uint32)]

        _LINUX_CAPABILITY_VERSION_3 = 0x20080522
        hdr = _CapHeader(_LINUX_CAPABILITY_VERSION_3, 0)
        data = (_CapData * 2)()  # zero-initialized -> clears eff/prm/inh
        rc = libc.capset(ctypes.byref(hdr), ctypes.byref(data))
        # Keep the process dumpable so `criu` seizes it cleanly after
        # the credential change (capset can reset dumpable to suid_dumpable).
        libc.prctl(PR_SET_DUMPABLE, 1, 0, 0, 0)
        print(f"[semip] dropped capabilities for portable image "
              f"(capset rc={rc}, no_new_privs rc={nnp})", flush=True)
    except Exception as e:  # never block startup on a cap-drop failure
        print(f"[semip] cap-drop failed (continuing): {e}", flush=True)


# Internal signal (leading underscore): the worker sets _SEMIP_CHILD_DROP_CAPS
# only in this spawned child's environment when running unprivileged.  It is
# deliberately NOT the user-facing SEMIP_UNPRIVILEGED flag -- the worker itself
# imports this module and must keep its caps to run `criu` against this child.
if os.environ.get("_SEMIP_CHILD_DROP_CAPS") == "1":
    _drop_caps_for_portable_image()

import torch

import semip_logging

# Ensure this package dir is importable in the child + its TP worker
# subprocesses (worker_cls="_semip_worker.SemipGPUWorker" and ca_graph_rebind
# are resolved by bare module name).
_semip_here = os.path.dirname(os.path.abspath(__file__))
if _semip_here not in sys.path:
    sys.path.insert(0, _semip_here)

try:
    import ca_graph_rebind  # dense-TP graph reuse support
    _CA_REBIND_AVAILABLE = True
except Exception as _ca_e:  # best-effort: reuse degrades to full recapture
    _CA_REBIND_AVAILABLE = False
    ca_graph_rebind = None
    print(f"[semip-ca-rebind] unavailable: {_ca_e}", flush=True)


def _truncate_for_display(value, limit=200):
    """Truncate strings (or strings inside a list/tuple) to ``limit`` chars,
    appending ``...(<n> chars)`` when the original exceeds ``limit``.
    """
    if isinstance(value, str):
        if len(value) > limit:
            return f"{value[:limit]}...({len(value)} chars)"
        return value
    if isinstance(value, (list, tuple)):
        out = [_truncate_for_display(v, limit) for v in value]
        return out if isinstance(value, list) else tuple(out)
    return value


# Shard size / parallelism for save_weights + load_weights disk I/O.
_WEIGHTS_SHARD_BYTES = 2 * 2**30   # 2 GiB per shard
_WEIGHTS_IO_WORKERS = 8            # thread pool size for shard I/O

# Fraction of *measured* free VRAM the restore staging buffer may claim.
# The budget that reaches this file is predicted on the host from
# ``total_gpu_bytes * gpu_memory_utilization - weights``, which treats
# everything in the allotment that is not weights as available.  It is not:
# CUDA graph private pools, one CUDA context per rank *on every GPU* (NCCL
# P2P/CUMEM peer mapping maps them all), and activations sit in there too.
# On GLM-5.3 at TP=8 that came to ~9 GiB of headroom that did not exist.
_STAGING_FREE_FRACTION = 0.9

_cudart = ctypes.CDLL("libcudart.so")
_cudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
_cudart.cudaHostUnregister.restype = ctypes.c_int
_cudart.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
_cudart.cudaHostRegister.restype = ctypes.c_int
_cudart.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
_cudart.cudaGetDevice.restype = ctypes.c_int
_cudart.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
_cudart.cudaGetDeviceCount.restype = ctypes.c_int
_cudart.cudaDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                          ctypes.c_int]
_cudart.cudaDeviceGetPCIBusId.restype = ctypes.c_int
_cudart.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                  ctypes.c_size_t, ctypes.c_uint]
_cudart.cudaHostAlloc.restype = ctypes.c_int
_cudart.cudaFreeHost.argtypes = [ctypes.c_void_p]
_cudart.cudaFreeHost.restype = ctypes.c_int
_cudart.cudaGetErrorString.argtypes = [ctypes.c_int]
_cudart.cudaGetErrorString.restype = ctypes.c_char_p


def _cuda_err(ret):
    """``cudaError`` as ``<int> (<name>)``, which is what a reader needs."""
    try:
        text = _cudart.cudaGetErrorString(ctypes.c_int(ret))
        return f"{ret} ({(text or b'?').decode(errors='replace')})"
    except Exception:  # noqa: BLE001
        return str(ret)


_NVML_SO = "libnvidia-ml.so.1"
_NVML_P2P_CAPS_INDEX_READ = 0
# One directory per GPU on the *node*, named by its real PCI address. Measured
# 2026-09-22: a pod holding only /dev/nvidia1 still sees all eight entries here,
# so this is the node's inventory and not the pod's allocation -- useful for
# knowing the node width, and wrong to use as "an address we own".
_NVIDIA_PROC_GPUS = "/proc/driver/nvidia/gpus"


class _NvmlPciInfo(ctypes.Structure):
    """``nvmlPciInfo_t``. Only ``busId`` is read; the rest fixes the layout."""

    _fields_ = [
        ("busIdLegacy", ctypes.c_char * 16),
        ("domain", ctypes.c_uint),
        ("bus", ctypes.c_uint),
        ("device", ctypes.c_uint),
        ("pciDeviceId", ctypes.c_uint),
        ("pciSubSystemId", ctypes.c_uint),
        ("busId", ctypes.c_char * 32),
    ]


def _load_nvml():
    """``libnvidia-ml.so.1`` via ``dlopen``, the same way NCCL reaches it.

    NCCL does not link NVML; ``nvmlwrap.cc`` dlopens it and caches the function
    pointers. Loading it the same way keeps this probe answering the question
    NCCL asks rather than a nearby one.
    """
    lib = ctypes.CDLL(_NVML_SO)
    lib.nvmlErrorString.argtypes = [ctypes.c_int]
    lib.nvmlErrorString.restype = ctypes.c_char_p
    lib.nvmlInit_v2.argtypes = []
    lib.nvmlInit_v2.restype = ctypes.c_int
    lib.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
    lib.nvmlDeviceGetCount_v2.restype = ctypes.c_int
    lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint,
                                                  ctypes.POINTER(ctypes.c_void_p)]
    lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
    lib.nvmlDeviceGetHandleByPciBusId_v2.argtypes = [ctypes.c_char_p,
                                                     ctypes.POINTER(ctypes.c_void_p)]
    lib.nvmlDeviceGetHandleByPciBusId_v2.restype = ctypes.c_int
    lib.nvmlDeviceGetIndex.argtypes = [ctypes.c_void_p,
                                       ctypes.POINTER(ctypes.c_uint)]
    lib.nvmlDeviceGetIndex.restype = ctypes.c_int
    lib.nvmlDeviceGetPciInfo_v3.argtypes = [ctypes.c_void_p,
                                            ctypes.POINTER(_NvmlPciInfo)]
    lib.nvmlDeviceGetPciInfo_v3.restype = ctypes.c_int
    lib.nvmlDeviceGetP2PStatus.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_int,
                                           ctypes.POINTER(ctypes.c_uint)]
    lib.nvmlDeviceGetP2PStatus.restype = ctypes.c_int
    return lib


def _nvml_err(lib, ret):
    try:
        text = lib.nvmlErrorString(ctypes.c_int(ret))
        return f"{ret} ({(text or b'?').decode(errors='replace')})"
    except Exception:  # noqa: BLE001
        return str(ret)


def _local_pci_bus_ids():
    """The PCI addresses of the GPUs this pod actually holds, uppercased."""
    try:
        return sorted(name.upper() for name in os.listdir(_NVIDIA_PROC_GPUS))
    except OSError:
        return []


def _bus_id_variants(bus_id):
    """``bus_id`` in every spelling NVML might want, most literal first.

    Both domain widths and both cases. Ordered so the caller's own string is
    tried first and the answer stays attributable to it when it works.
    """
    out = []
    for text in (bus_id, bus_id.upper(), bus_id.lower()):
        parts = text.split(":")
        forms = [text]
        if len(parts) == 3:
            try:
                domain = int(parts[0], 16)
            except ValueError:
                domain = None
            if domain is not None:
                rest = ":".join(parts[1:])
                forms += ["%04X:%s" % (domain, rest), "%08X:%s" % (domain, rest),
                          "%04x:%s" % (domain, rest), "%08x:%s" % (domain, rest)]
        for form in forms:
            if form not in out:
                out.append(form)
    return out


def _nvml_probe(stale_bus_id):
    """Which of the seven lines in NCCL's ``commAlloc`` actually fails.

    A slot-mismatched TP>1 restore dies inside ``commAlloc``: the last NCCL line
    is ``Using network Socket`` and ``Init START`` -- logged immediately after
    ``commAlloc`` returns -- never appears. Between them sit
    ``ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev)`` and
    ``ncclNvmlDeviceGetIndex``, and the former returns ``ncclSystemError`` when
    NVML cannot resolve the address, which is the error observed.

    But the matching ``NVMLCHECK`` WARN (``nvmlDeviceGetHandleByPciBusId()
    failed: Not Found``) is *absent* from the logs while other NCCL WARNs are
    present, and there is a second path that would explain that silence:
    ``ncclNvmlDeviceGetHandleByPciBusId`` first calls
    ``ncclNvmlEnsureInitialized``, which returns a cached ``initResult`` with no
    new warning if NVML initialisation failed earlier -- and that init walks a
    P2P-status matrix over every visible device.

    So two candidates, indistinguishable from the logs. This separates them by
    making both calls directly:

    ``init`` / ``p2p_matrix``
        Reproduces ``ncclNvmlEnsureInitialized``. A failure here means the
        cached-init path, and the bus-ID lookup is never the real story.
    ``lookup_stale``
        The suspect call, on the address CUDA still reports from the dump node.
        ``NOT_FOUND`` here with a healthy init confirms the bus-ID reading.
    ``lookup_owned``
        Control, on an address NVML itself enumerated. It must succeed; if it
        does not, NVML is broken generally rather than confused by a stale id.

    Answered on hardware 2026-09-22, on a slot-mismatched TP=2 restore:
    ``init`` Success, ``p2p_matrix`` OK, ``lookup_owned`` Success,
    ``lookup_stale`` ``6 (Not Found)``. So it is the bus-ID lookup, and the
    cached-init path is not involved.

    Never raises: this runs on a restore that is already in trouble.
    """
    out = {}
    try:
        lib = _load_nvml()
    except BaseException as exc:  # noqa: BLE001
        return {"dlopen": f"{type(exc).__name__}: {exc}"}
    try:
        rc = lib.nvmlInit_v2()
        out["init"] = _nvml_err(lib, rc)
        if rc != 0:
            return out

        count = ctypes.c_uint(0)
        rc = lib.nvmlDeviceGetCount_v2(ctypes.byref(count))
        out["device_count"] = count.value if rc == 0 else _nvml_err(lib, rc)

        handles = []
        for i in range(count.value if rc == 0 else 0):
            h = ctypes.c_void_p()
            hrc = lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(i),
                                                    ctypes.byref(h))
            if hrc != 0:
                out.setdefault("handle_by_index_failures", []).append(
                    f"{i}: {_nvml_err(lib, hrc)}")
            else:
                handles.append(h)

        # The walk ncclNvmlEnsureInitialized performs. One bad pair is enough to
        # fail the init and poison every later call with a cached result.
        p2p_failures = []
        for i, a in enumerate(handles):
            for j, b in enumerate(handles):
                if i == j:
                    continue
                status = ctypes.c_uint(0)
                prc = lib.nvmlDeviceGetP2PStatus(
                    a, b, ctypes.c_int(_NVML_P2P_CAPS_INDEX_READ),
                    ctypes.byref(status))
                if prc != 0:
                    p2p_failures.append(f"{i}->{j}: {_nvml_err(lib, prc)}")
        out["p2p_matrix"] = ("OK (%d pairs)" % (len(handles) * (len(handles) - 1))
                             if not p2p_failures else p2p_failures)

        def lookup(bus_id):
            """Resolve ``bus_id``, trying every spelling before believing a miss.

            CUDA renders the domain in 8 hex digits (``00000000:8B:00.0``) and
            /proc in 4 (``0000:8b:00.0``). Without trying both, a formatting
            difference would read as ``NOT_FOUND`` and be mistaken for the very
            thing this probe exists to detect.
            """
            if not bus_id:
                return "no bus id to try"
            first_err = None
            for form in _bus_id_variants(bus_id):
                h = ctypes.c_void_p()
                lrc = lib.nvmlDeviceGetHandleByPciBusId_v2(
                    form.encode(), ctypes.byref(h))
                if lrc != 0:
                    if first_err is None:
                        first_err = _nvml_err(lib, lrc)
                    continue
                # NCCL's very next call, so a handle that resolves but cannot be
                # indexed still shows up as the failure it would cause.
                idx = ctypes.c_uint(0)
                irc = lib.nvmlDeviceGetIndex(h, ctypes.byref(idx))
                got = (f"OK -> index {idx.value}" if irc == 0
                       else f"handle OK but GetIndex {_nvml_err(lib, irc)}")
                return got if form == bus_id else f"{got} (as {form})"
            return f"{first_err} (tried {len(_bus_id_variants(bus_id))} spellings)"

        # The control has to be an address NVML *owns*, which is not the same as
        # an address on this node. Measured 2026-09-22: /proc/driver/nvidia/gpus
        # lists all eight node GPUs even in a one-GPU pod, so taking its first
        # entry tested a device the pod was never allocated and returned
        # NOT_FOUND for a reason that had nothing to do with the bug. Enumerate
        # through NVML instead, so a passing control really does mean "NVML can
        # resolve what it holds".
        owned = []
        for i, h in enumerate(handles):
            info = _NvmlPciInfo()
            prc = lib.nvmlDeviceGetPciInfo_v3(h, ctypes.byref(info))
            if prc == 0:
                owned.append(info.busId.decode(errors="replace"))
            else:
                out.setdefault("pci_info_failures", []).append(
                    f"{i}: {_nvml_err(lib, prc)}")
        out["owned_bus_ids"] = owned
        out["node_bus_ids"] = _local_pci_bus_ids()
        out["stale_bus_id"] = stale_bus_id
        out["lookup_stale"] = lookup(stale_bus_id)
        out["lookup_owned"] = lookup(owned[0] if owned else None)
    except BaseException as exc:  # noqa: BLE001
        out["probe_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _unpin_buffer(buf):
    ret = _cudart.cudaHostUnregister(ctypes.c_void_p(buf.data_ptr()))
    if ret != 0:
        raise RuntimeError(f"cudaHostUnregister failed with cudaError={ret}")


def _repin_buffer(buf):
    ret = _cudart.cudaHostRegister(
        ctypes.c_void_p(buf.data_ptr()),
        ctypes.c_size_t(buf.numel() * buf.element_size()),
        ctypes.c_uint(0),
    )
    if ret != 0:
        raise RuntimeError(f"cudaHostRegister failed with cudaError={ret}")


def _install_faulthandler():
    """Make ``kill -USR1 <pid>`` dump every thread's Python stack.

    The 302 s post-restore warmup hangs left a rank burning a core in
    userspace with nothing in the log, and the image ships neither py-spy nor
    gdb. ``faulthandler`` is stdlib and needs no privilege at all, unlike ptrace
    -- which matters because the pod runs with ``CapEff: 0000000000000000``.

    Registered from ``_cuda_restore_probe`` because that already runs in every
    rank on the restore path, so a hung rank is always dumpable by the time
    anything can hang. The handler is left installed for the process's life.

    Never raises: this is diagnostics.
    """
    try:
        import faulthandler
        import signal
        if getattr(_install_faulthandler, "_done", False):
            return "already"
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
        _install_faulthandler._done = True
        return "registered on SIGUSR1"
    except BaseException as exc:  # noqa: BLE001
        return f"unavailable: {type(exc).__name__}: {exc}"


def _cuda_restore_probe(worker):
    """What CUDA believes about this rank's device, and whether pinning works.

    Answers the question a failed ``reinit_nccl`` cannot. NCCL's init calls
    ``ncclCudaHostCalloc`` and, when that fails, returns ``ncclSystemError`` --
    surfaced as ``NCCL error: unhandled system error`` with the underlying
    ``cudaError`` discarded. Measured 2026-09-20 on a slot-mismatched restore,
    the last thing NCCL logs is ``init.cc:1994 Cuda Host Alloc`` and then it
    dies, so the pinned host allocation is the operation that fails and its
    error code is the one fact nobody has. This performs the same call directly
    and names the result.

    The device identity is recorded beside it because a restored rank carries
    its dump-time view: NCCL prints the *cached* ``busId`` (a5000, i.e. GPU
    slots 6,7) while the pod it landed in actually holds different PCI
    addresses. Printing both CUDA's answer and the physical one is how a stale
    mapping becomes visible rather than inferred.

    The ``nvml`` key carries ``_nvml_probe``, which narrows the failure from
    "somewhere in ``commAlloc``" to a specific call. See its docstring.

    Never raises. This is diagnostics on the path of a restore that is already
    in trouble; it must not be what breaks it.
    """
    out = {"rank": getattr(worker, "rank", "?")}
    out["faulthandler"] = _install_faulthandler()
    try:
        dev = ctypes.c_int(-1)
        out["get_device"] = _cuda_err(_cudart.cudaGetDevice(ctypes.byref(dev)))
        out["device"] = dev.value
        count = ctypes.c_int(-1)
        _cudart.cudaGetDeviceCount(ctypes.byref(count))
        out["device_count"] = count.value

        buf = ctypes.create_string_buffer(64)
        rc = _cudart.cudaDeviceGetPCIBusId(buf, 64, dev.value)
        out["pci_bus_id"] = (buf.value.decode(errors="replace") if rc == 0
                             else _cuda_err(rc))

        # The operation NCCL dies on, done in isolation. 4 KiB: the question is
        # whether pinning is possible at all, not how much of it.
        ptr = ctypes.c_void_p()
        rc = _cudart.cudaHostAlloc(ctypes.byref(ptr), ctypes.c_size_t(4096),
                                   ctypes.c_uint(0))
        out["cudaHostAlloc"] = _cuda_err(rc)
        if rc == 0:
            out["cudaFreeHost"] = _cuda_err(_cudart.cudaFreeHost(ptr))

        # Which line of commAlloc kills a slot-mismatched restore. Logged as its
        # own key so the answer is greppable rather than buried in prose.
        out["nvml"] = _nvml_probe(out.get("pci_bus_id"))
    except BaseException as exc:  # noqa: BLE001
        out["probe_error"] = f"{type(exc).__name__}: {exc}"
    return out


# ---------------------------------------------------------------------------
# Worker-local semi-persistence primitives (TP-safe).
#
# These run inside each vLLM worker via ``llm.collective_rpc``.  At TP>1 the
# callable is cloudpickled and executed in every worker process, so the
# staging buffer must live on the worker object (``worker._semip_*``), not in
# the vllm_child process -- a captured closure over a vllm_child-local buffer
# would be copied by value per worker and its writes discarded.  The same code
# path works at TP=1 (a single worker, in-process).  ``worker`` is whatever
# ``collective_rpc`` passes (the WorkerWrapperBase driver at TP=1, the real
# Worker at TP>1); attribute reads forward to the underlying Worker either way,
# and attribute writes persist across calls because the object is stable.
#
# Param names are namespaced ``main:p:`` / ``drafter:p:`` to match the layout
# built at attach so stage/restore dispatch each entry to the right tensor.
# ---------------------------------------------------------------------------
def _semip_layout(worker):
    layout = []
    mr = worker.model_runner
    for name, p in mr.model.named_parameters():
        d = p.data
        layout.append((f"main:p:{name}", d.nbytes, d.dtype, tuple(d.shape)))
    drafter = getattr(mr, "drafter", None)
    dm = getattr(drafter, "model", None) if drafter is not None else None
    if dm is not None:
        for name, p in dm.named_parameters():
            d = p.data
            layout.append((f"drafter:p:{name}", d.nbytes, d.dtype, tuple(d.shape)))
    return layout


def _semip_param_tensors(worker):
    mr = worker.model_runner
    t = {f"main:p:{n}": p.data for n, p in mr.model.named_parameters()}
    drafter = getattr(mr, "drafter", None)
    dm = getattr(drafter, "model", None) if drafter is not None else None
    if dm is not None:
        for n, p in dm.named_parameters():
            t[f"drafter:p:{n}"] = p.data
    return t


def _semip_attach(worker):
    layout = _semip_layout(worker)
    total_size = sum(nb for _, nb, _, _ in layout)
    index = {}
    offset = 0
    for name, nbytes, dtype, shape in layout:
        index[name] = (offset, nbytes, dtype, shape)
        offset += nbytes
    worker._semip_buf = torch.empty(total_size, dtype=torch.uint8)
    worker._semip_index = index
    worker._semip_chunk_plan = None
    worker._semip_chunk_size = None
    worker._semip_pinned = False
    return (total_size, len(layout))


def _semip_stage(worker):
    buf = worker._semip_buf
    index = worker._semip_index
    sources = _semip_param_tensors(worker)
    for name, (offset, nbytes, dtype, shape) in index.items():
        src = sources[name].contiguous().reshape(-1).view(torch.uint8)
        buf[offset:offset + nbytes].copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    return buf.numel()


def _semip_unpin(worker):
    # Idempotent: the attach buffer starts unpinned, so a cudaHostUnregister on
    # an unregistered (or already-unpinned) buffer would hard-error.
    buf = getattr(worker, "_semip_buf", None)
    if buf is None or not getattr(worker, "_semip_pinned", False):
        return 0
    _unpin_buffer(buf)
    worker._semip_pinned = False
    return buf.numel()


def _semip_repin(worker):
    # Idempotent: a double cudaHostRegister would hard-error.
    buf = getattr(worker, "_semip_buf", None)
    if buf is None:
        return 0
    if getattr(worker, "_semip_pinned", False):
        return buf.numel()
    _repin_buffer(buf)
    worker._semip_pinned = True
    return buf.numel()


def _semip_plan_load_weights(worker, max_buffer_bytes=None):
    buf = worker._semip_buf
    index = worker._semip_index
    total_bytes = buf.numel()
    cs = total_bytes if max_buffer_bytes is None else min(int(max_buffer_bytes), total_bytes)
    # ``max_buffer_bytes`` is a prediction made on the host, which cannot see
    # this device and sends one number for every rank.  Ask the driver the same
    # question instead: this runs under collective_rpc with the rank's device
    # current, and the buffer it is sizing is a plain cudaMalloc that the driver
    # -- not torch's bookkeeping -- has to satisfy.  min() only, so a measured
    # reading can shrink a plan but never grow one, and the reading is taken
    # where it is valid: weights are already resident (wake_up_weights) and the
    # KV cache is not yet mapped (wake_up_kv_cache runs after restore_weights).
    free_bytes, _ = torch.cuda.mem_get_info(worker.device)
    free_cap = int(_STAGING_FREE_FRACTION * free_bytes)
    clamped = free_cap < cs
    cs = min(cs, free_cap)
    plan = []
    cur = []
    cur_lo = 0
    for name, (off, nbytes, dtype, shape) in index.items():
        if nbytes > cs:
            raise RuntimeError(f"param {name} ({nbytes}B) exceeds chunk_size ({cs}B)")
        if cur and (off + nbytes - cur_lo) > cs:
            cur_hi = cur[-1][1] + cur[-1][2]
            plan.append((cur_lo, cur_hi, cur))
            cur = []
            cur_lo = off
        cur.append((name, off, nbytes, dtype, shape))
    if cur:
        cur_hi = cur[-1][1] + cur[-1][2]
        plan.append((cur_lo, cur_hi, cur))
    worker._semip_chunk_plan = plan
    worker._semip_chunk_size = cs
    return {"bytes": total_bytes, "n_chunks": len(plan), "chunk_size": cs,
            "free_bytes": free_bytes, "clamped": clamped}


def _semip_restore_weights(worker):
    buf = worker._semip_buf
    index = worker._semip_index
    targets = _semip_param_tensors(worker)
    total_bytes = buf.numel()
    plan = worker._semip_chunk_plan
    cs = worker._semip_chunk_size
    if plan is None:
        plan = [(0, total_bytes,
                 [(n, o, nb, dt, sh) for n, (o, nb, dt, sh) in index.items()])]
        cs = total_bytes
    torch.cuda.synchronize()
    gpu_buf = torch.empty(cs, dtype=torch.uint8, device=worker.device)
    loaded = 0
    for chunk_lo, chunk_hi, members in plan:
        n = chunk_hi - chunk_lo
        gpu_buf[:n].copy_(buf[chunk_lo:chunk_hi], non_blocking=True)
        torch.cuda.synchronize()
        for name, off, nbytes, dtype, shape in members:
            start = off - chunk_lo
            src = gpu_buf[start:start + nbytes].view(dtype).reshape(shape)
            targets[name].copy_(src)
            loaded += 1
        torch.cuda.synchronize()
    gpu_buf.storage().resize_(0)
    del gpu_buf
    # Return the staging buffer to the CUDA *driver*, not just to torch's
    # caching allocator.  resize_(0)/free only hands the ~cs-byte block back
    # to torch's pool (cudaMalloc arena); it stays reserved from the driver's
    # point of view.  The next step, wake_up(["kv_cache"]), maps the KV cache
    # via vLLM's cumem allocator (cuMemCreate/cuMemMap), which allocates from
    # driver-free memory -- so a torch-cached staging block (up to the full
    # 50+ GiB weight size at chunk_size == total) starves it and the KV map
    # OOMs.  empty_cache() releases torch's cached blocks so cumem can map.
    torch.cuda.empty_cache()
    return {"bytes": total_bytes, "loaded": loaded,
            "n_chunks": len(plan), "chunk_size": cs}


def _semip_detach(worker):
    buf = getattr(worker, "_semip_buf", None)
    total = buf.numel() if buf is not None else 0
    if buf is not None and getattr(worker, "_semip_pinned", False):
        _unpin_buffer(buf)
    worker._semip_buf = None
    worker._semip_index = None
    worker._semip_chunk_plan = None
    worker._semip_chunk_size = None
    worker._semip_pinned = False
    return total


def _semip_save_weights(worker, weights_dir, shard_bytes=None, io_workers=None):
    buf = worker._semip_buf
    index = worker._semip_index
    shard_bytes = int(shard_bytes or _WEIGHTS_SHARD_BYTES)
    workers = int(io_workers or _WEIGHTS_IO_WORKERS)
    # TP1 keeps the flat weights/ layout; TP>1 fans out per-rank shards into
    # weights/rank{R}/, since each rank holds a different slice of the params.
    rank_dir = (weights_dir if _semip_tp_size(worker) <= 1
                else os.path.join(weights_dir, f"rank{worker.rank}"))
    if os.path.exists(rank_dir):
        try:
            shutil.rmtree(rank_dir)
        except PermissionError as e:
            raise RuntimeError(
                f"cannot clear stale weights at {rank_dir}: {e}. They were "
                f"written by a run under a different uid; remove them by hand "
                f"as that user and re-run save_weights()") from e
    os.makedirs(rank_dir, exist_ok=True)

    total = buf.numel()
    mv = memoryview(buf.numpy())
    ranges = []
    lo = 0
    i = 0
    while lo < total:
        hi = min(lo + shard_bytes, total)
        ranges.append((i, lo, hi))
        lo = hi
        i += 1

    def _write_shard(i, lo, hi):
        fd = os.open(os.path.join(rank_dir, f"shard_{i:04d}.bin"),
                     os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            pos = lo
            while pos < hi:
                pos += os.write(fd, mv[pos:hi])
            os.fsync(fd)
        finally:
            os.close(fd)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(ranges)))) as ex:
        for fu in [ex.submit(_write_shard, *r) for r in ranges]:
            fu.result()

    manifest = {
        "total_bytes": total,
        "n_params": len(index),
        "shard_bytes": shard_bytes,
        "shards": [{"name": f"shard_{i:04d}.bin", "offset": lo, "nbytes": hi - lo}
                   for (i, lo, hi) in ranges],
        "layout": [[name, off, nbytes, str(dtype), list(shape)]
                   for name, (off, nbytes, dtype, shape) in index.items()],
    }
    with open(os.path.join(rank_dir, "weights_meta.json"), "w") as f:
        json.dump(manifest, f)
    return total


def _semip_load_weights(worker, weights_dir, io_workers=None):
    buf = worker._semip_buf
    index = worker._semip_index
    workers = int(io_workers or _WEIGHTS_IO_WORKERS)
    rank_dir = (weights_dir if _semip_tp_size(worker) <= 1
                else os.path.join(weights_dir, f"rank{worker.rank}"))
    meta_path = os.path.join(rank_dir, "weights_meta.json")
    with open(meta_path) as f:
        manifest = json.load(f)

    total = buf.numel()
    if int(manifest["total_bytes"]) != total:
        raise RuntimeError(
            f"weights size mismatch: manifest {manifest['total_bytes']}B but "
            f"attached buffer {total}B (config/model changed?)")
    cur_layout = [[name, off, nbytes]
                  for name, (off, nbytes, dtype, shape) in index.items()]
    man_layout = [[row[0], row[1], row[2]] for row in manifest.get("layout", [])]
    if man_layout and man_layout != cur_layout:
        raise RuntimeError("weights layout mismatch between manifest and "
                           "attached model (param order/sizes differ)")

    mv = memoryview(buf.numpy())
    shards = manifest["shards"]

    def _read_shard(s):
        lo = int(s["offset"])
        n = int(s["nbytes"])
        dst = mv[lo:lo + n]
        got = 0
        with open(os.path.join(rank_dir, s["name"]), "rb", buffering=0) as f:
            while got < n:
                r = f.readinto(dst[got:])
                if r == 0:
                    break
                got += r
        if got != n:
            raise RuntimeError(f"{s['name']}: read {got} != {n}")

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(shards)))) as ex:
        for fu in [ex.submit(_read_shard, s) for s in shards]:
            fu.result()
    return total


# ---------------------------------------------------------------------------
# TP>1 NCCL teardown / reinit around CRIU (worker-local, via collective_rpc).
#
# CRIU cannot restore live NCCL communicators or CustomAllreduce IPC handles,
# so they are torn down before checkpoint and rebuilt after restore.  The
# teardown is always graph-preserving (unilateral ncclCommAbort, not the
# collective ncclCommDestroy that a live captured graph would deadlock); for
# MoE/EP the abort must also be concurrent (see _nccl_abort_comms_concurrent).
#
# Graph reuse is the only mode.  There was a `full` alternative -- drop the
# preserved graphs and rebuild them with capture_model() -- selectable at dump
# time through SEMIP_GRAPH_MODE.  Retired 2026-09-24 after wave 10 (17/17) and
# GLM-5.3 at TP=8: reuse-against-a-warm-image is the only method we trust, so
# it is the only one anyone can select.  Warming the image is likewise no longer
# a knob; it is part of `init`.  See skills/tp_DESIGN.md section 5.
# ---------------------------------------------------------------------------


_COMPILE_CACHE_ENVS = ("TRITON_CACHE_DIR", "VLLM_CACHE_ROOT",
                       "TORCHINDUCTOR_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE")


def _compile_cache_fingerprint(budget_s=2.0):
    """File count and byte total under each JIT cache dir.

    Taken on both sides of the checkpoint. The caches are pinned into
    ``<model_dir>/compilation`` so CRIU can find the dlopen'd .so's at restore,
    which also puts them on shared storage, outside the image, writable by every
    instance of this model at once. With a reading from each side, a restore that
    JIT-compiles a kernel can be attributed: absent from the cache means it was
    never built, present means the restored process missed the cache key. Those
    have different fixes and are indistinguishable today.

    Time-boxed rather than exhaustive -- this walks a network filesystem, and a
    truncated count compared against another truncated count is still a signal.
    """
    out = {}
    deadline = time.monotonic() + budget_s
    for env in _COMPILE_CACHE_ENVS:
        path = os.environ.get(env)
        if not path:
            continue
        n = total = 0
        truncated = False
        try:
            for root, _dirs, files in os.walk(path):
                for fn in files:
                    try:
                        total += os.stat(os.path.join(root, fn)).st_size
                        n += 1
                    except OSError:
                        pass
                if time.monotonic() > deadline:
                    truncated = True
                    break
        except OSError:
            continue
        out[env] = {"files": n, "bytes": total, "truncated": truncated}
    return out


_MISSING = object()
_LIBNCCL = None


def _semip_config_value(config, key, default=None):
    """Read ``key`` from a vllm config object or dict, falling back to its
    nested ``parallel_config``."""
    if config is None:
        return default
    if isinstance(config, dict):
        if key in config:
            return config[key]
        pc = config.get("parallel_config")
    else:
        if hasattr(config, key):
            return getattr(config, key)
        pc = getattr(config, "parallel_config", None)
    if pc is not None:
        if isinstance(pc, dict):
            if key in pc:
                return pc[key]
        elif hasattr(pc, key):
            return getattr(pc, key)
    return default


def _semip_tp_size(worker):
    return int(_semip_config_value(
        getattr(worker, "vllm_config", None), "tensor_parallel_size", 1) or 1)


def _is_arctic_parallel_worker(worker):
    # Ulysses / shift SP is out of scope for this port (dense TP + EP only).
    return False


def _clear_fd_backed_nccl_env():
    """Drop restored NCCL/OFI env that points at process-local fds (closed
    before CRIU save), so the next NCCL init regenerates them."""
    native_keys = (
        "NCCL_TOPO_FILE", "NCCL_TUNER_PLUGIN", "NCCL_NETDEVS_POLICY",
        "NCCL_NET_FORCE_FLUSH", "NCCL_NVLS_CHUNKSIZE",
        "NCCL_NVLSTREE_MAX_CHUNKSIZE", "NCCL_P2P_NET_CHUNKSIZE",
        "FI_EFA_FORK_SAFE",
    )
    for key in native_keys:
        os.environ.pop(key, None)
        os.unsetenv(key)
    for key, value in list(os.environ.items()):
        if key == "NCCL_TOPO_FILE" or (
                key.startswith("NCCL_") and value.startswith("/proc/self/fd/")):
            os.environ.pop(key, None)
            os.unsetenv(key)


def _nccl_abort_comm(comm):
    """ncclCommAbort(comm) via ctypes -- UNILATERAL and non-blocking, unlike the
    collective ncclCommDestroy which deadlocks when a live captured graph or an
    overlapping MoE/EP topology pins the comm."""
    global _LIBNCCL
    if _LIBNCCL is None:
        _LIBNCCL = ctypes.CDLL("libnccl.so.2")
        _LIBNCCL.ncclCommAbort.restype = ctypes.c_int
        _LIBNCCL.ncclCommAbort.argtypes = [ctypes.c_void_p]
    cp = comm if isinstance(comm, ctypes.c_void_p) else ctypes.c_void_p(int(comm))
    return _LIBNCCL.ncclCommAbort(cp)


def _nccl_abort_comms_concurrent(targets, timeout=60.0, after_fire=None):
    """Abort a set of pynccl comms CONCURRENTLY (one thread each).  Sequential
    abort deadlocks on DeepSeek-style overlapping world/tp/dp/ep comms: the
    shared per-rank proxy only drains once every comm's abort flag is set.  The
    ``after_fire`` hook sets the torch ProcessGroupNCCL abort flags while the
    pynccl aborts are in flight, so all flags are set together."""
    import threading
    import time as _t
    results = {}

    def _worker(nm, ptr):
        try:
            results[nm] = ("ok", _nccl_abort_comm(ptr))
        except BaseException as e:  # noqa: BLE001
            results[nm] = ("err", f"{type(e).__name__}: {e}")

    threads = []
    for nm, ptr in targets:
        th = threading.Thread(target=_worker, args=(nm, ptr), daemon=True)
        th.start()
        threads.append(th)
    if after_fire is not None:
        try:
            after_fire()
        except BaseException:  # noqa: BLE001
            pass
    deadline = _t.monotonic() + timeout
    for th in threads:
        th.join(max(0.0, deadline - _t.monotonic()))
    return results


def _abort_torch_process_groups():
    """Abort (non-blocking) every torch NCCL process group so the subsequent
    clean destroy does not deadlock on a comm a live graph still pins."""
    import torch
    from torch.distributed import distributed_c10d as c
    try:
        world = getattr(c, "_world", None)
        pg_map = dict(getattr(world, "pg_map", {}) or {})
        for pg in list(pg_map.keys()):
            try:
                be = pg._get_backend(torch.device("cuda"))
            except Exception:  # noqa: BLE001
                be = None
            fn = getattr(be, "abort", None) or getattr(pg, "abort", None)
            if fn is not None:
                fn()
    except Exception:  # noqa: BLE001
        pass


def _mark_rst_on_close(fd_int):
    """Set SO_LINGER(1,0) on an inet TCP socket so its eventual close sends RST
    (skips TIME_WAIT) -> avoids restore-time EADDRINUSE on the rebind. AF_UNIX
    and non-stream sockets are left untouched. Operates on a *dup* so we never
    close the worker's live fd here -- the option lives on the shared socket, so
    the original fd RSTs when it is finally closed by teardown/CRIU-kill."""
    import socket, struct
    try:
        dup = os.dup(fd_int)
    except OSError:
        return None
    try:
        s = socket.socket(fileno=dup)   # Linux auto-detects family/type/proto
    except OSError:
        os.close(dup)
        return None
    try:
        if (s.family in (socket.AF_INET, socket.AF_INET6)
                and s.type == socket.SOCK_STREAM):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                         struct.pack("ii", 1, 0))
            try:
                return (fd_int, s.family.name, s.getsockname())
            except OSError:
                return (fd_int, s.family.name, None)
    finally:
        s.close()   # closes the dup only; fd_int stays open for CRIU
    return None


def _mark_inet_sockets_rst(where):
    """Scan this worker's fds and SO_LINGER(1,0) every inet TCP socket, so the
    next close (distributed teardown / destructive CRIU dump) sends RST and the
    tuple skips TIME_WAIT -> restore rebinds the rendezvous port immediately with
    no cooldown. AF_UNIX / non-stream sockets are left alone. Returns marked
    (fd, family, addr) tuples."""
    pid = os.getpid()
    marked = []
    try:
        fd_names = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        return []
    for fd_name in fd_names:
        try:
            fd_int = int(fd_name)
            if fd_int <= 2:
                continue
            link = os.readlink(f"/proc/{pid}/fd/{fd_name}")
        except (OSError, ValueError):
            continue
        if not link.startswith("socket:"):
            continue
        r = _mark_rst_on_close(fd_int)
        if r is not None:
            marked.append(r)
    if marked:
        print(f"[rst-on-dump] {where}: SO_LINGER(1,0) on {len(marked)} "
              f"inet TCP socket(s): {marked}", flush=True)
    return marked


# Set once this process has aborted its NCCL comms in ``_destroy_nccl``.  The
# abort is unilateral, so once it returns the torch rendezvous is dead whether or
# not the clean destroy that follows it succeeded -- and that, not torch's
# bookkeeping, is what the dump-side listener close needs to know.  On this path
# the bookkeeping is exactly what fails: the ``destroy_process_group()`` that
# clears ``GroupMember.WORLD`` (and so flips ``is_initialized()``) runs only
# inside vLLM's ``destroy_distributed_environment()``, which
# ``destroy_model_parallel()`` never reaches when it raises on an aborted comm.
#
# Per-process, and deliberately so: ``_destroy_nccl`` is a ``collective_rpc``
# target, so this becomes true in the worker processes and stays False in the
# driver -- which keeps relying on its own ``destroyed_pg``.  It also stays False
# at TP1, where ``_destroy_nccl`` returns before the abort.  See Complication 15.
_PG_ABORTED = False


def _communicator_checkpoint_targets(ps):
    """Every distinct device communicator, in an order all ranks agree on.

    vLLM 0.30's communicator checkpoint hooks end in a cross-rank barrier
    (``flashinfer/comm/allreduce.py``: "Do not return until every rank has
    released all workspace handles"), so a rank that visits a different set --
    or the same set in a different order -- wedges the job rather than failing
    it. Group names are identical on every rank, so sort by them instead of
    trusting ``_groups`` insertion order.

    Deduped by communicator identity because several group names share one
    communicator. The hooks are idempotent ("repeated successful calls are
    no-ops"), so this is a tidiness measure, not a correctness one.
    """
    seen = set()
    out = []
    groups = getattr(ps, "_groups", {})
    for name in sorted(groups):
        ref = groups[name]
        group = ref() if callable(ref) else ref
        if group is None:
            continue
        comm = getattr(group, "device_communicator", None)
        if comm is None or id(comm) in seen:
            continue
        seen.add(id(comm))
        out.append((name, comm))
    return out


# Where ``checkpoint_prepare``'s inventory waits for the restore half to read it.
# A worker attribute rather than a module global because that is what
# ``_reinit_nccl`` already holds, and it rides the CRIU image exactly the way
# ``_semip_rank_data_keep`` does -- both are ordinary in-process memory.  The
# references are strong on purpose: the prepared workspaces are orphaned the
# moment ``_reinit_nccl`` replaces the groups that own them, and a weakref would
# let them be collected somewhere in the middle of the dump.
_CKPT_PREPARED_ATTR = "_semip_ckpt_prepared"

# Group-name prefix -> the ``parallel_state`` getter for that role.  The restore
# half resolves roles through these rather than through ``ps._groups[name]``
# because the names carry a disambiguating suffix that a rebuild bumps (``ep:0``
# becomes ``ep:1``), while the getters always name the live group.
_CKPT_ROLE_GETTERS = {
    "tp": "get_tp_group",
    "ep": "get_ep_group",
    "dp": "get_dp_group",
    "pp": "get_pp_group",
    "world": "get_world_group",
}


def _fi_ar_workspaces_for(comm):
    """The FlashInfer all-reduce workspaces this communicator's hooks will touch.

    The same list ``checkpoint_prepare`` and ``checkpoint_restore`` iterate --
    vLLM reaches it through ``_fi_ar_workspaces_for_group(self.cpu_group)`` --
    which is the whole point of computing it here.  Counting the list on both
    sides is what turns "the loop body never ran" from an invisible no-op into a
    number that disagrees.

    A missing module means vLLM 0.26 or older, where none of this exists and the
    answer is honestly zero.  Anything the lookup itself raises is a real fault
    and propagates: it refuses a workspace whose creating group was dropped
    ("process group was not retained"), which is precisely the condition the
    caller needs to hear about.
    """
    try:
        from vllm.distributed.device_communicators import (
            flashinfer_all_reduce as fiar)
    except Exception:  # noqa: BLE001 - pre-0.30 has no fabric workspaces
        return []
    group = getattr(comm, "cpu_group", None)
    if group is None:
        return []
    return list(fiar._fi_ar_workspaces_for_group(group))


def _checkpoint_target_inventory(name, comm):
    """What one target's checkpoint hook is about to act on.

    Must be taken *before* ``checkpoint_prepare`` runs: afterwards the workspaces
    are detached and the all2all buffers are gone, and neither leaves a record of
    how many there were.

    Two resources, reached two different ways, and they need opposite handling on
    the way back.  The FlashInfer workspaces are process-level globals that vLLM
    matches to their creating group by object identity, so re-keying them is
    enough to put them back in the hook's path.  The MoE all2all buffers hang off
    *this* communicator's ``all2all_manager``, an object the restore side never
    sees again -- so the manager itself has to be carried across, and restored by
    hand.
    """
    rec = {"name": name, "role": name.split(":")[0],
           "workspaces": [], "all2all": None, "err": None}
    try:
        rec["workspaces"] = _fi_ar_workspaces_for(comm)
    except Exception as exc:  # noqa: BLE001
        rec["err"] = f"workspace inventory: {type(exc).__name__}: {exc}"
    mgr = getattr(comm, "all2all_manager", None)
    # `initialized` is what gates vLLM's own all2all hooks; an uninitialized
    # manager's `checkpoint_prepare` is a no-op, so counting it would invent an
    # obligation the restore side could never discharge.
    if mgr is not None and getattr(mgr, "initialized", False):
        rec["all2all"] = mgr
    return rec


def _run_communicator_checkpoint_hook(ps, which, worker=None):
    """Drive ``checkpoint_prepare`` / ``checkpoint_restore`` across every group.

    vLLM 0.30 backs the FlashInfer all-reduce workspace -- and, under EP, the
    MoE all2all buffers -- with MNNVL fabric handles. ``cuCheckpointProcess*``
    cannot carry those: NVIDIA's list of unsupported memory is "IPC, UVM, RDMA,
    or fabric handle" (cuda-checkpoint#14), and because the driver "does not
    attempt to keep the process in a good state" on error it shows up two ways.
    Measured on GLM-5.3 TP=8, same payload throughout: vLLM 0.26 checkpointed in
    26.8 s and restored in 15.9 s; 0.30 either stalled in ``cuda_checkpoint``
    indefinitely or checkpointed in 43.9 s and then failed
    ``cuCheckpointProcessRestore`` with CUresult=801 (CUDA_ERROR_NOT_SUPPORTED).

    The hooks detach the *physical* backing and keep the virtual address stable,
    so nothing baked into a captured graph moves and ``ca_graph_rebind`` has
    nothing extra to do -- the opposite of custom all-reduce, whose meta_ptrs
    really do move.

    A failure is logged rather than raised. Every rank runs identical code over
    an identical group set, so a raise here is uniform across ranks and none of
    them reached the barrier; letting one rank abandon the loop early would turn
    a clean failure into a wedge.

    ``worker`` is what makes the two halves comparable. On the prepare side each
    target's inventory is stashed on it; on the restore side the same counts are
    recomputed so the caller can check that what was detached came back. Passing
    None keeps the old fire-and-forget behaviour, which is all the TP1 and 0.26
    paths need.

    ``failed_material`` is the subset of failures on targets that actually held
    something. A stale group left over from a rebuild can raise here while owning
    no workspace at all, and failing the restore over that would be a regression;
    a raise on a target that *was* carrying a workspace is a different matter,
    and the counts cannot see it because the inventory is taken before the call.
    """
    done, failed, failed_material, targets = [], [], [], []
    n_workspaces = 0
    for name, comm in _communicator_checkpoint_targets(ps):
        fn = getattr(comm, which, None)
        if fn is None:
            continue
        rec = _checkpoint_target_inventory(name, comm)
        material = bool(rec["workspaces"]) or rec["all2all"] is not None
        n_workspaces += len(rec["workspaces"])
        targets.append(rec)
        if rec["err"]:
            failed.append(f"{name}: {rec['err']}")
            failed_material.append(f"{name}: {rec['err']}")
        try:
            fn()
            done.append(name)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
            if material:
                failed_material.append(f"{name}: {type(exc).__name__}: {exc}")
            # The message alone is not enough when the raise comes from inside
            # FlashInfer: `checkCudaErrors` renders every driver failure the same
            # way, so "CUDA error code=101" could be any of half a dozen calls
            # and only the frame says which. Printed rather than returned -- the
            # return value crosses a collective_rpc and is quoted into a job
            # error, where a multi-line traceback per rank is unreadable.
            import traceback as _tb
            print(f"[ckpt-hook] {which} traceback for {name}:\n"
                  f"{_tb.format_exc()}", flush=True)
    n_all2all = sum(1 for rec in targets if rec["all2all"] is not None)
    if worker is not None and which == "checkpoint_prepare":
        setattr(worker, _CKPT_PREPARED_ATTR,
                {"targets": targets, "n_workspaces": n_workspaces,
                 "n_all2all": n_all2all})
    if done or failed:
        # Worker process, not the child: `log` is a child-process local, and the
        # worker-side idiom here is stdout, which vLLM prefixes with the rank.
        # The counts are on the same line as the names because the names alone
        # are what made this bug survive three runs: two matching target sets say
        # nothing about whether either one moved any memory.
        print(f"[ckpt-hook] {which}: ok={','.join(done) or 'none'} "
              f"failed={'; '.join(failed) or 'none'} "
              f"workspaces={n_workspaces} all2all={n_all2all}", flush=True)
    return {"ok": done, "failed": failed, "failed_material": failed_material,
            "n_workspaces": n_workspaces, "n_all2all": n_all2all}


def _prepared_device_indices(prepared):
    """The CUDA devices the prepared workspaces were built on.

    ``SymmDeviceMemory`` keeps the ``device_idx`` it was constructed with and
    feeds it straight back into ``allocation_prop.location.id`` and
    ``cuMulticastBindMem`` when the handles are re-mapped, so this -- not
    whatever the restoring thread happens to have current -- is the device the
    restore has to run on.

    Reach it down the same attribute path FlashInfer's own hooks use:
    ``MNNVLAllReduceFusionWorkspace.checkpoint_restore`` reads
    ``self.handle.mcast_device_memory`` and refuses anything that is not a
    ``SymmDeviceMemory``.  ``mem_handles`` is a field of ``MnnvlMemory``'s
    allocation record, not of the fusion workspace, and asking only for it is
    why job 6e386668 printed ``probe={}`` on all eight ranks: the list came
    back empty, ``_probe_restore_devices`` queried no device at all, and the
    context binding fell through to ``worker.local_rank``.  Both shapes are
    accepted here because both exist in FlashInfer; neither is assumed.
    """
    out = []
    for rec in prepared.get("targets", ()):
        for ws in rec.get("workspaces", ()):
            memory = getattr(getattr(ws, "handle", None),
                             "mcast_device_memory", None)
            for holder in (memory, *(getattr(ws, "mem_handles", None) or ())):
                idx = getattr(holder, "device_idx", None)
                if isinstance(idx, int) and idx not in out:
                    out.append(idx)
    return out


def _bind_cuda_context_for_restore(worker, prepared):
    """Make the right primary context current before FlashInfer re-maps handles.

    ``SymmDeviceMemory.__init__`` establishes it and only then maps::

        cu_device   = cuDeviceGet(device_idx)
        primary_ctx = cuDevicePrimaryCtxRetain(cu_device)
        cuCtxSetCurrent(primary_ctx)
        cudaSetDevice(device_idx)
        ...
        self._create_and_map_handles(self.comm_backend)

    The restore path reaches ``_create_and_map_handles`` directly and it repeats
    none of that -- it reads ``self.device_idx`` into the allocation properties
    and the multicast bind and trusts the caller's context.  Its only guard,
    ``_verify_cuda_context``, logs a warning and carries on.  Nothing on the
    semi-p restore path sets a device either, so the thread runs with whatever
    CRIU and ``cuda_restore`` left current, and the driver answers
    CUDA_ERROR_INVALID_DEVICE (101).

    The before/after device is reported rather than just fixed, because the two
    already agreeing would mean the 101 comes from a stale ``device_idx``
    instead -- a different fault with a different fix, and one this cannot
    distinguish without saying what it saw.
    """
    out = {"want": None, "want_src": None, "before": None, "after": None,
           "problems": []}
    wanted = _prepared_device_indices(prepared)
    if len(wanted) > 1:
        out["problems"].append(
            f"prepared workspaces span devices {wanted}; expected one per rank")
    if wanted:
        out["want_src"] = "workspace"
    else:
        # Say so. On job 6e386668 this fallback fired on all eight ranks and
        # `want` was read as the workspace's device when it was only the
        # worker's -- so `before == want` proved less than it appeared to.
        out["want_src"] = "local_rank_fallback"
        idx = getattr(worker, "local_rank", None)
        wanted = [idx] if isinstance(idx, int) else []
    if not wanted:
        out["problems"].append("no device index to bind: no workspace handle "
                               "carried one and the worker has no local_rank")
        return out
    device_idx = wanted[0]
    out["want"] = device_idx
    try:
        from cuda.bindings import driver as _cu
    except Exception:  # noqa: BLE001 - cuda-python < 13 spells it differently
        try:
            from cuda import cuda as _cu
        except Exception as exc:  # noqa: BLE001
            out["problems"].append(
                f"no cuda driver bindings: {type(exc).__name__}: {exc}")
            return out

    def _ck(res):
        err = res[0]
        if err != _cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(str(err))
        return res[1] if len(res) > 1 else None

    try:
        out["before"] = int(_ck(_cu.cuCtxGetDevice()))
    except Exception as exc:  # noqa: BLE001 - no current context is the finding
        out["before"] = f"none ({type(exc).__name__}: {exc})"
    try:
        import torch
        torch.cuda.set_device(device_idx)
        cu_device = _ck(_cu.cuDeviceGet(device_idx))
        primary = _ck(_cu.cuDevicePrimaryCtxRetain(cu_device))
        _ck(_cu.cuCtxSetCurrent(primary))
        out["after"] = int(_ck(_cu.cuCtxGetDevice()))
    except Exception as exc:  # noqa: BLE001
        out["problems"].append(
            f"binding device {device_idx}: {type(exc).__name__}: {exc}")
    return out


def _probe_restore_devices(prepared):
    """What the driver thinks of the devices the prepared handles name.

    ``_create_and_map_handles``' very first act is ``make_handle_exchanger()``,
    which calls ``is_mnnvl_fabric_supported(device_idx)``, which is::

        checkCudaErrors(cuda.cuDeviceGetAttribute(
            CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device_idx))

    That runs *before* FlashInfer's own ``_verify_cuda_context``, and it is a
    device-level query rather than a context-level one -- so a
    CUDA_ERROR_INVALID_DEVICE out of it says the ordinal is wrong, not that the
    context is unset.  The ordinal is whatever the workspace was constructed
    with on the dump side, and nothing guarantees it still resolves here: the
    restore may land on a different node with a different visible set, which is
    the whole point of not requiring ``SEMIP_REQUIRE_DEVICE_MATCH``.

    So ask the same questions first, in our own code, where the answer can be
    logged rather than raised from six frames inside a vendor package.
    """
    import os as _os
    out = {"visible": _os.environ.get("CUDA_VISIBLE_DEVICES"), "count": None,
           "devices": {}}
    try:
        import torch
        out["count"] = torch.cuda.device_count()
    except Exception as exc:  # noqa: BLE001
        out["count"] = f"unavailable ({type(exc).__name__})"
    try:
        from cuda.bindings import driver as _cu
    except Exception:  # noqa: BLE001
        try:
            from cuda import cuda as _cu
        except Exception:  # noqa: BLE001
            return out
    for idx in _prepared_device_indices(prepared):
        info = {}
        # Report the attribute *value*, not merely that the driver answered.
        # The first version recorded "ok" whenever the call returned
        # CUDA_SUCCESS and threw away ``res[1]``, so job 07fd79b4 printed
        # ``multicast: 'ok'`` on all eight ranks while saying nothing at all
        # about whether multicast is supported -- a successful query returning
        # 0 and a successful query returning 1 read identically. That is the
        # question this whole probe exists to answer, and it is the same
        # silent-success shape as the bugs it was written to catch.
        # ``cuDeviceGet`` returns a handle rather than a flag, so it stays a
        # status; the two attribute queries report what the driver said.
        for label, call, is_attr in (
                ("get", lambda i=idx: _cu.cuDeviceGet(i), False),
                ("fabric", lambda i=idx: _cu.cuDeviceGetAttribute(
                    _cu.CUdevice_attribute
                    .CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, i),
                 True),
                ("multicast", lambda i=idx: _cu.cuDeviceGetAttribute(
                    _cu.CUdevice_attribute
                    .CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, i), True)):
            try:
                res = call()
                err = res[0]
                if err != _cu.CUresult.CUDA_SUCCESS:
                    info[label] = str(err)
                elif not is_attr:
                    info[label] = "ok"
                else:
                    value = res[1] if len(res) > 1 else None
                    info[label] = int(value) if isinstance(
                        value, int) else f"ok(value={value!r})"
            except Exception as exc:  # noqa: BLE001
                info[label] = f"{type(exc).__name__}: {exc}"
        out["devices"][idx] = info
    return out


def _current_role_communicator(ps, role):
    """The device communicator ``parallel_state`` currently gives for a role."""
    getter = _CKPT_ROLE_GETTERS.get(role)
    if getter is None:
        return None
    try:
        grp = getattr(ps, getter)()
    except Exception:  # noqa: BLE001 - absent roles are not an error here
        return None
    if grp is None:
        return None
    return getattr(grp, "device_communicator", None)


def _restore_prepared_comm_state(ps, worker):
    """Point what ``checkpoint_prepare`` detached at the groups that exist now.

    ``_reinit_nccl`` tears the distributed environment down and rebuilds it, so
    every ``ProcessGroup`` and every device communicator on this side of the
    restore is a new object.  vLLM finds a FlashInfer workspace by comparing
    ``workspace_group is group`` against the group that *created* it, so without
    this the lookup misses on every target, the loop body never runs, and the
    hook reports ``failed=none`` over an empty list -- the workspace keeps its
    reserved VA with no physical backing and the first graph replay faults.

    The two prepared resources need opposite treatment:

    * the workspaces are re-keyed here and then left to the normal
      ``checkpoint_restore`` hook, which hands FlashInfer a live
      ``TorchDistBackend(group=...)``.  That parameter exists precisely so a
      workspace can be restored against a group other than its creator's --
      ``checkpoint_prepare`` barriers over the backend it stored at creation,
      ``checkpoint_restore`` over whatever it is given.  Replaying the *old*
      communicators instead would satisfy the identity check and then hang, since
      the re-map's allgather needs a group whose sockets still exist.
    * the all2all managers are restored here, by hand.  The hook drives the new
      communicator, whose ``all2all_manager`` is a different object that was
      never prepared; the prepared one is reachable only through the stash.

    Re-keying assigns straight into ``_fi_ar_workspace_groups`` rather than
    re-registering: vLLM's registration path raises "already associated with a
    different process group" by design, and that guard is worth leaving intact.
    The entry is reassigned and never removed -- ``_fi_ar_workspaces_for_group``
    raises on a workspace whose group entry has gone missing.

    Targets are walked in the order the prepare side recorded them, which is
    sorted by group name and therefore identical on every rank.  The all2all
    restore below is collective, so that ordering is load-bearing here for the
    same reason it is in the hook.
    """
    out = {"n_workspaces": 0, "n_all2all": 0, "problems": []}
    prepared = getattr(worker, _CKPT_PREPARED_ATTR, None)
    if not prepared:
        return out
    # Before anything is re-mapped, and only when there is something to re-map:
    # FlashInfer's re-map reads the device out of the handle and trusts the
    # caller's context, and nothing on this path has set one.
    ctx = None
    if prepared["n_workspaces"] or prepared["n_all2all"]:
        probe = _probe_restore_devices(prepared)
        print(f"[ckpt-hook] devices: visible={probe['visible']} "
              f"count={probe['count']} probe={probe['devices']}", flush=True)
        ctx = _bind_cuda_context_for_restore(worker, prepared)
        out["problems"].extend(ctx["problems"])
        print(f"[ckpt-hook] cuda-ctx: want={ctx['want']} "
              f"src={ctx['want_src']} "
              f"before={ctx['before']} after={ctx['after']}", flush=True)
    try:
        from vllm.distributed.device_communicators import (
            flashinfer_all_reduce as fiar)
    except Exception as exc:  # noqa: BLE001
        if prepared["n_workspaces"]:
            out["problems"].append(
                f"flashinfer_all_reduce import: {type(exc).__name__}: {exc}")
        fiar = None
    for rec in prepared["targets"]:
        if not rec["workspaces"] and rec["all2all"] is None:
            continue
        comm = _current_role_communicator(ps, rec["role"])
        group = getattr(comm, "cpu_group", None) if comm is not None else None
        if group is None:
            # Nothing to re-key onto. Never skip quietly: a detached workspace
            # with no live group is exactly the state that faults later.
            out["problems"].append(
                f"{rec['name']}: no live '{rec['role']}' group after reinit")
            continue
        if fiar is not None:
            for ws in rec["workspaces"]:
                try:
                    fiar._fi_ar_workspace_groups[id(ws)] = group
                    out["n_workspaces"] += 1
                except Exception as exc:  # noqa: BLE001
                    out["problems"].append(
                        f"{rec['name']}: re-key: {type(exc).__name__}: {exc}")
        mgr = rec["all2all"]
        if mgr is not None:
            try:
                mgr.cpu_group = group
                mgr.checkpoint_restore()
                out["n_all2all"] += 1
            except Exception as exc:  # noqa: BLE001
                out["problems"].append(
                    f"{rec['name']}: all2all restore: "
                    f"{type(exc).__name__}: {exc}")
    print(f"[ckpt-hook] rekey: workspaces={out['n_workspaces']} "
          f"all2all={out['n_all2all']} "
          f"problems={'; '.join(out['problems']) or 'none'}", flush=True)
    return out


def _assert_checkpoint_state_restored(worker, rekey, restored):
    """Fail here rather than let a detached workspace fault in the first forward.

    Every step involved is a loop over a set that can be empty, and an empty loop
    returns cleanly: ``checkpoint_restore_fi_ar_workspaces`` over no workspaces
    reports exactly what a full restore reports.  That is how a workspace stayed
    detached across three cluster runs while the hook printed ``failed=none``, so
    the count is the check -- what ``checkpoint_prepare`` detached has to come
    back, and any other number is an error even though nothing raised.

    Zero prepared stays legal.  A bf16 model never builds a workspace and the
    0.26 path has no hooks at all; the invariant is equality, not presence.

    Called after the hook's loop has finished, never from inside it. Every rank
    has to clear the barriers in there before any rank is allowed to unwind --
    failing earlier would leave the others waiting on a rank that has left.
    """
    prepared = getattr(worker, _CKPT_PREPARED_ATTR, None)
    if not prepared:
        return
    problems = list(rekey.get("problems", ()))
    problems += [f"restore hook: {f}"
                 for f in restored.get("failed_material", ())]
    want_ws, got_ws = prepared["n_workspaces"], restored.get("n_workspaces", 0)
    if got_ws != want_ws:
        problems.append(
            f"checkpoint_prepare detached {want_ws} FlashInfer workspace(s), "
            f"checkpoint_restore saw {got_ws}")
    want_a2a, got_a2a = prepared["n_all2all"], rekey.get("n_all2all", 0)
    if got_a2a != want_a2a:
        problems.append(
            f"checkpoint_prepare prepared {want_a2a} all2all manager(s), "
            f"{got_a2a} were restored")
    if problems:
        raise RuntimeError("communicator checkpoint state was not fully "
                           "restored: " + "; ".join(problems))


def _destroy_nccl(worker):
    """Tear down NCCL process groups before checkpoint.  No-op at TP1.

    Always graph-preserving: abort (unilateral ncclCommAbort) rather than
    collective-destroy, because a live captured graph pins the comm and the
    MoE/EP topology would deadlock a collective destroy.  The graphs are always
    preserved now, so the collective path is gone with the `full` mode that was
    its only caller.
    """
    import torch.distributed as dist

    if _semip_tp_size(worker) <= 1:
        return None

    def _close_custom_allreduce_ipc_handles(ca):
        if ca is None or getattr(ca, "disabled", True):
            return
        rank = ca.rank
        for pointers in (getattr(ca, "meta_ptrs", []),
                         getattr(ca, "buffer_ptrs", [])):
            for i, ptr in enumerate(pointers):
                if i == rank or not ptr:
                    continue
                ret = _cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))
                if ret != 0:
                    raise RuntimeError(
                        f"cudaIpcCloseMemHandle failed with cudaError={ret}")

    if not dist.is_initialized():
        return {}
    from vllm.distributed import parallel_state as ps
    from vllm.distributed.parallel_state import (
        destroy_model_parallel, destroy_distributed_environment)

    # RST-on-close: mark the rendezvous inet TCP sockets now, while they are
    # still open, so the teardown below closes them with RST (skips TIME_WAIT).
    # Otherwise the destructive dump leaves the tuple in TIME_WAIT and the
    # restore rebind fails with EADDRINUSE (sk-inet.c: Address already in use).
    # Runs per-worker (this is a collective_rpc target), so it hits each rank's
    # rendezvous socket.
    _mark_inet_sockets_rst("destroy_nccl")

    # Before the abort, while cpu_group still works: the hooks barrier across
    # ranks, and an aborted comm cannot carry one.  The worker carries the
    # inventory to the restore side, which cannot rediscover it: by then the
    # groups that owned these workspaces have been replaced.
    _run_communicator_checkpoint_hook(ps, "checkpoint_prepare", worker)

    seen_pynccl_ids = set()
    seen_ca_ids = set()
    abort_targets = []
    abort_pynccls = []
    # Every group, not just tp/world: MoE adds ep (and may add dp/dcp/pcp/eplb);
    # leaving any undestroyed leaks NCCL/NVLS handles across the checkpoint.
    for name, ref in list(getattr(ps, "_groups", {}).items()):
        group = ref() if callable(ref) else ref
        if group is None:
            continue
        comm = getattr(group, "device_communicator", None)
        if comm is None:
            continue
        pynccl = getattr(comm, "pynccl_comm", None)
        if pynccl is not None and getattr(pynccl, "comm", None) is not None:
            if id(pynccl) not in seen_pynccl_ids:
                seen_pynccl_ids.add(id(pynccl))
                cp = pynccl.comm
                cp = (int(cp.value) if isinstance(cp, ctypes.c_void_p)
                      else int(cp))
                abort_targets.append((name, cp))
                abort_pynccls.append(pynccl)
        ca = getattr(comm, "ca_comm", None)
        if ca is not None and id(ca) not in seen_ca_ids:
            seen_ca_ids.add(id(ca))
            _close_custom_allreduce_ipc_handles(ca)
            ca.close()
            comm.ca_comm = None

    if abort_targets:
        _nccl_abort_comms_concurrent(abort_targets)
        for pynccl in abort_pynccls:
            pynccl.comm = None

    # The ``after_fire`` hook on the concurrent abort existed to set these flags
    # while the pynccl aborts were in flight, which only mattered on the clean
    # collective-destroy path.  Preserving the graph means we always abort, so
    # this runs afterwards and unconditionally.
    _abort_torch_process_groups()

    # Recorded here rather than at the top of the function: every early return
    # above (TP1, never-initialized) leaves the comms alone, so only code that
    # has actually aborted them may claim it.  And recorded before the clean
    # destroy rather than after, because that destroy is expected to raise and
    # the listener close downstream depends on this surviving the raise.
    global _PG_ABORTED
    _PG_ABORTED = True

    try:
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:  # noqa: BLE001
        # Aborted comms make the clean destroy raise; parallel_state is reset
        # enough for reinit to rebuild.  We always abort, so this is expected
        # rather than a failure to report.
        pass
    _clear_fd_backed_nccl_env()
    return {}


def _force_dist_uninitialized_for_restore():
    """Force torch.distributed + parallel_state to a clean uninitialized state so
    the next init rebuilds a FRESH process group / TCP store / NCCL comms.  After
    an abort-based teardown the CRIU image can carry is_initialized()==True with
    dead PGs, which makes post-restore init skip init_process_group and deadlock."""
    from vllm.distributed import parallel_state as ps
    try:
        import torch.distributed as dist  # noqa: F401
        from torch.distributed import distributed_c10d as c10d
        try:
            c10d._update_default_pg(None)
        except BaseException:  # noqa: BLE001
            c10d.GroupMember.WORLD = None
        w = getattr(c10d, "_world", None)
        if w is not None:
            for _attr in ("comms", "pg_map", "pg_names", "pg_group_ranks",
                          "pg_backend_config", "pg_to_tag", "tags_to_pg",
                          "pg_coalesce_state"):
                try:
                    getattr(w, _attr).clear()
                except BaseException:  # noqa: BLE001
                    pass
            try:
                w.group_count = 0
            except BaseException:  # noqa: BLE001
                pass
        try:
            c10d._unregister_all_process_groups()
        except BaseException:  # noqa: BLE001
            pass
    except BaseException:  # noqa: BLE001
        pass
    # `initialize_model_parallel` asserts each of its groups is None, so one
    # left behind turns the next init into an AssertionError instead of a
    # rebuild -- and the assert names the group, which is the only reason the
    # 0.30 upgrade was diagnosable at all.  The set is version-dependent: 0.30
    # added `_ETP` and `_ENGRAM_DP`, and a hand-kept list silently misses the
    # next one, so discovery is the primary mechanism and the names below are
    # only the floor.  Discovery cannot stand alone because a group left as
    # None-but-present, or a non-coordinator like `_NODE_COUNT`, is invisible
    # to it.
    _named = ("_WORLD", "_INNER_DP_WORLD", "_NODE_COUNT", "_TP", "_PP", "_DP",
              "_DCP", "_PCP", "_EP", "_EPLB", "_ETP", "_ENGRAM_DP", "_SP",
              "_SP_TP")
    _found = []
    try:
        from vllm.distributed.parallel_state import GroupCoordinator
        _found = [n for n in dir(ps)
                  if n.startswith("_")
                  and isinstance(getattr(ps, n, None), GroupCoordinator)]
    except BaseException:  # noqa: BLE001
        pass
    for _g in dict.fromkeys(_named + tuple(_found)):
        try:
            if hasattr(ps, _g):
                setattr(ps, _g, None)
        except BaseException:  # noqa: BLE001
            pass


_RPC_TIMEOUT_DEFAULT_S = 300.0


def _descendant_pids(root=None):
    """Every descendant of ``root`` (default: this process), breadth first."""
    root = os.getpid() if root is None else root
    found, queue = [], [root]
    while queue:
        pid = queue.pop(0)
        try:
            tasks = os.listdir(f"/proc/{pid}/task")
        except OSError:
            continue
        for tid in tasks:
            try:
                with open(f"/proc/{pid}/task/{tid}/children") as f:
                    kids = f.read().split()
            except OSError:
                continue
            for kid in kids:
                kid = int(kid)
                if kid not in found:
                    found.append(kid)
                    queue.append(kid)
    return found


def _hang_report():
    """Where every descendant is blocked -- one line each, for a timeout.

    ``wchan`` is the kernel function a task is parked in, and it is enough to
    tell the three candidates apart without a debugger in the image:
    ``futex_wait_queue`` is an internal lock, ``ep_poll``/``do_poll`` is
    waiting on a peer, and anything in ``ioctl`` is the CUDA driver.
    """
    lines = []
    for pid in _descendant_pids():
        try:
            with open(f"/proc/{pid}/comm") as f:
                comm = f.read().strip()
            with open(f"/proc/{pid}/stat") as f:
                state = f.read().rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            continue
        try:
            with open(f"/proc/{pid}/wchan") as f:
                wchan = f.read().strip() or "-"
        except OSError:
            wchan = "?"
        lines.append(f"{pid}({comm}) state={state} wchan={wchan}")
    return "; ".join(lines) or "no descendants readable"


def _collective_rpc_with_timeout(llm, fn, args, timeout_s=None, log=None):
    """``llm.collective_rpc``, bounded, and self-diagnosing when it expires.

    An unbounded ``collective_rpc`` is how a single deadlocked rank turns into
    a job that sits idle for its full timeout with nothing in the log after
    ``>>> reinit_nccl``. The RPC cannot be cancelled once a worker is wedged,
    so this does not try: it reports where every process is parked and raises,
    which is strictly more than the caller had before.

    ``timeout_s`` arrives as a command kwarg rather than from the environment
    on purpose. This process is restored, so its ``environ`` is the dump's and
    no ``extra_env`` set on the restoring job would ever be visible here.
    """
    timeout_s = _RPC_TIMEOUT_DEFAULT_S if timeout_s is None else float(timeout_s)
    box = {}

    def _run():
        try:
            box["result"] = llm.collective_rpc(fn, args=args)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    thread = threading.Thread(target=_run, daemon=True,
                              name=f"rpc-{getattr(fn, '__name__', fn)}")
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        report = _hang_report()
        if log is not None:
            log.error("  %s did not return within %.0fs; %s",
                      getattr(fn, "__name__", fn), timeout_s, report)
        raise TimeoutError(
            f"{getattr(fn, '__name__', fn)} did not return within "
            f"{timeout_s:.0f}s. Where each process is blocked: {report}")
    if "error" in box:
        raise box["error"]
    return box["result"]


def _reinit_nccl(worker, port):
    """Re-initialize NCCL after restore on a fresh TCP port, then rebind the
    canonical tp:0/world:0 (+ ep:0/dp:0 for MoE) group slots the captured graphs
    look up."""
    import traceback
    import weakref
    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state as ps
    from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
    try:
        _clear_fd_backed_nccl_env()
        _force_dist_uninitialized_for_restore()
        # NVLS / SymmMem exchange fds that do not survive CRIU -> keep them off.
        # FlashInfer allreduce is off for a different reason -- the captured
        # graphs must hold `cross_device_reduce` nodes for the rebind to have
        # anything to rewrite -- but it has to hold on both sides, or the
        # rebuilt communicator disagrees with the graphs it is rebound into.
        os.environ["NCCL_NVLS_ENABLE"] = "0"
        os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
        os.environ["VLLM_ALLREDUCE_USE_FLASHINFER"] = "0"
        _rd_keep = getattr(worker, "_semip_rank_data_keep", None)
        with set_current_vllm_config(worker.vllm_config):
            # Reuse the preserved cold-start rank_data tensor (same VA) so the
            # kept-graph CA _dp pointers stay valid.
            _rd_patched = bool(
                _CA_REBIND_AVAILABLE and ca_graph_rebind is not None
                and _rd_keep is not None
                and ca_graph_rebind.install_rank_data_reuse_patch(_rd_keep))
            try:
                init_worker_distributed_environment(
                    worker.vllm_config,
                    worker.rank,
                    distributed_init_method=f"tcp://127.0.0.1:{port}",
                    local_rank=worker.local_rank,
                    backend="nccl",
                )
            finally:
                if _rd_patched:
                    ca_graph_rebind.restore_rank_data_reuse_patch()
        new_tp = ps.get_tp_group()
        new_world = ps.get_world_group()
        if hasattr(ps, "_groups"):
            ps._groups["tp:0"] = weakref.ref(new_tp)
            ps._groups["world:0"] = weakref.ref(new_world)
            # Gated on the group existing, not on `enable_expert_parallel`.
            # vLLM builds ep/dp groups for any MoE model -- the flag only decides
            # whether experts are sharded across them -- so testing the flag left
            # GLM-5.3, which is MoE with the flag off, holding a dead weakref at
            # `ep:0` while the rebuilt group registered itself as `ep:1`. That is
            # what made the two `[ckpt-hook]` lines name different group sets.
            for _grp_name, _getter in (("ep:0", "get_ep_group"),
                                       ("dp:0", "get_dp_group")):
                try:
                    _grp = getattr(ps, _getter)()
                except Exception:  # noqa: BLE001
                    _grp = None
                if _grp is not None:
                    ps._groups[_grp_name] = weakref.ref(_grp)
        # Re-attach the physical backing that checkpoint_prepare detached on the
        # dump side. Last, because the hook barriers over the rebuilt cpu_group
        # and reads the canonical slots written just above.
        _rekey = _restore_prepared_comm_state(ps, worker)
        _restored = _run_communicator_checkpoint_hook(ps, "checkpoint_restore",
                                                      worker)
        # After the hook's loop, never inside it: every rank has to clear the
        # barriers in there before any rank is allowed to fail.
        _assert_checkpoint_state_restored(worker, _rekey, _restored)
        return {"ok": True, "rank": getattr(worker, "rank", "?")}
    except BaseException as e:  # noqa: BLE001
        return {"ok": False, "rank": getattr(worker, "rank", "?"),
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}


# Threads torch leaves behind around its rendezvous.  Two lists, not one, and the
# difference matters:
#
# ``_WAIT_STORE_THREADS`` are the ones a teardown is expected to reap, so waiting
# on them terminates quickly.  ``pt_gloo_runloop`` is deliberately absent -- gloo's
# listening socket lives for the whole process (see the ``GLOO_SOCKET_IFNAME`` note
# in ``init``), so waiting on it would turn a fast poll into a guaranteed timeout
# and a warning on every dump.
#
# ``_CENSUS_STORE_THREADS`` adds it anyway, for reporting only.  The live-process
# probe behind Complication 15 found ``pt_gloo_runloop`` holding one of the
# restored listeners, so whether it is still alive at dump time is worth knowing
# -- it decides whether thread death could ever gate the close.
_WAIT_STORE_THREADS = ("pt_tcpstore", "pt_nccl_watchdg", "pt_nccl_heartbt")
_CENSUS_STORE_THREADS = _WAIT_STORE_THREADS + ("pt_gloo_runloop",)


def _live_store_threads(pid=None, names=_CENSUS_STORE_THREADS):
    """``tid(comm)`` for every rendezvous/watchdog thread still alive in *pid*."""
    pid = os.getpid() if pid is None else pid
    alive = []
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return alive
    for tid_name in tids:
        try:
            with open(f"/proc/{pid}/task/{tid_name}/comm") as f:
                comm = f.read().strip()
        except OSError:
            continue
        if any(comm.startswith(n) for n in names):
            alive.append(f"{tid_name}({comm})")
    return alive


def _wait_store_threads_exit(pid=None, attempts=50, interval=0.05):
    """Wait for the reapable rendezvous threads to exit.

    Returns ``(ok, still_alive, polls)``.  Advisory: callers log the outcome and
    carry on either way, because a thread outliving the teardown costs a noisy
    dump, not a wrong one.  Waits on ``_WAIT_STORE_THREADS`` only -- see the note
    there for why gloo is excluded.
    """
    alive = _live_store_threads(pid, _WAIT_STORE_THREADS)
    for n in range(attempts):
        if not alive:
            return True, [], n
        time.sleep(interval)
        alive = _live_store_threads(pid, _WAIT_STORE_THREADS)
    return False, alive, attempts


def _ephemeral_port_range():
    """``(low, high)`` from ``ip_local_port_range``, or ``(None, None)``.

    Duplicated from ``worker.py`` rather than imported: these two modules do not
    import each other, and the child must not grow a dependency on the worker.
    """
    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range") as f:
            lo, hi = f.read().split()[:2]
        return int(lo), int(hi)
    except (OSError, ValueError):
        return None, None


def _loopback_listeners():
    """This process's listening loopback TCP sockets, split by port kind.

    Returns ``(ephemeral, fixed)``, each a list of ``(fd, "addr:port")``.  Pure:
    nothing is closed and no fd is consumed, so a caller that only wants to
    describe what the image is about to record cannot accidentally alter it.
    That separation is deliberate -- see ``_close_loopback_listeners``, which is
    the only thing in this module allowed to close one of these.

    The split is the whole question of Complication 15:

    * **ephemeral** (inside ``ip_local_port_range``) is torch's rendezvous at
      TP=1, and the restore discards it -- seven ports were recorded in the
      measured image and none of the seven was live afterwards, because
      ``reinit_nccl`` rebuilds the rendezvous from scratch.
    * **fixed** is a service advertising itself.  NCCL's RAS listener sits on
      ``localhost:28028`` (``NCCL_RAS_ADDR``), behind a thread no torch teardown
      stops.

    An unreadable port range reports nothing as ephemeral, which keeps the safe
    direction: describe less, and so close less.
    """
    import socket as _sock
    pid = os.getpid()
    lo, hi = _ephemeral_port_range()
    ephemeral = []
    fixed = []
    for fd_name in sorted(os.listdir(f"/proc/{pid}/fd"), key=int):
        try:
            fd_int = int(fd_name)
            if fd_int <= 2:
                continue
            link = os.readlink(f"/proc/{pid}/fd/{fd_name}")
            if not link.startswith("socket:"):
                continue
            # socket(fileno=) takes ownership of the fd, so detach() before
            # deciding anything -- letting the wrapper be collected would
            # close sockets that have to survive the dump.
            s = _sock.socket(fileno=fd_int)
            try:
                fam = s.family
                listening = s.getsockopt(_sock.SOL_SOCKET,
                                         _sock.SO_ACCEPTCONN)
                addr = s.getsockname()
            finally:
                s.detach()
            if not listening:
                continue
            if fam not in (_sock.AF_INET, _sock.AF_INET6):
                continue
            if addr[0] not in ("127.0.0.1", "::1"):
                continue
            row = (fd_int, f"{addr[0]}:{addr[1]}")
            if lo is not None and lo <= addr[1] <= hi:
                ephemeral.append(row)
            else:
                fixed.append(row)
        except (OSError, ValueError):
            pass
    return ephemeral, fixed


def _close_loopback_listeners():
    """Close this process's orphaned torch rendezvous listeners.  **TP=1 only.**

    ``destroy_process_group()`` retires torch's TCPStore and Gloo threads but
    not their listening sockets: the threads exit, the fds stay open, and the FD
    keep-list preserves every ``socket:`` fd into the image.  CRIU then has to
    ``bind()`` every recorded port on every restore, and those ports are
    ephemeral -- the kernel assigned them on the dump host -- so a live occupant
    in the restoring pod fails the whole restore with EADDRINUSE.
    ``SO_REUSEADDR`` does not help against a live listener, and retrying never
    clears it.  Nothing reuses the ports either: ``reinit_nccl`` builds a fresh
    rendezvous, so the rebind buys nothing.

    **Callers must confirm the process group is down first**, and must be on the
    TP=1 path.  Both conditions, because only their conjunction makes these
    sockets nobody's:

    * At TP=1 this is measured to be exactly torch's set -- seven listeners, all
      on the driver's own fds, with ``pt_tcpstore`` and fourteen
      ``pt_gloo_runloop`` threads beside them -- and NCCL's RAS subsystem is not
      running at all, so there is nothing else here to catch.
    * **At TP>1 there is nothing of torch's left to close.**  Its teardown
      retires its own listeners before the dump (measured: 28 across two ranks
      during init, 2 by dump time), and the ephemeral loopback listener that
      survives belongs to RAS.  Closing it left RAS calling ``accept()`` on a
      dead fd after restore, spinning ``EBADF`` with no backoff at ~143 MB/s
      while the job reported ``RUNNING`` and served correct answers.  So the
      TP>1 path censuses via ``_loopback_listeners`` and closes nothing.

    Fixed ports are skipped even here.  It costs nothing at TP=1, where there are
    none, and it is the second line of defence if RAS ever does appear on this
    path: a fixed loopback port is how a long-lived service advertises itself,
    never something the kernel handed out by accident.

    Returns ``(closed, skipped)`` as ``addr:port`` strings.
    """
    ephemeral, fixed = _loopback_listeners()
    closed = []
    for fd_int, tag in ephemeral:
        try:
            os.close(fd_int)
            closed.append(tag)
        except OSError:
            pass
    return closed, [tag for _, tag in fixed]


_PSM_PREFIXES = ("/dev/shm/psm_", "/dev/shm/psm2_")
_SEM_PREFIXES = ("/dev/shm/sem.",)


def _own_shm_paths(pids, prefixes, proc="/proc", shm_dir="/dev/shm"):
    """The ``/dev/shm`` files *pids* map or hold open, among *prefixes*.

    The dump unlinks these so CRIU captures their mappings as memory rather than
    as files to reopen. It used to glob ``/dev/shm`` instead, which reached every
    other replica in the pod: a segment unlinked under a live sibling breaks the
    next process that opens it by name -- a rank attaching to its broadcast
    queue, or a worker spawned with a queue whose semaphores it reopens.
    Restricting the unlink to what this tree actually maps leaves the siblings
    alone. Already-unlinked files are skipped.

    **A mapping's path is not always the file's name.** glibc's ``sem_open``
    creates the semaphore under a temporary ``sem.XXXXXX``, links it to the real
    name (``sem.loky-<pid>-...``) and unlinks the temporary, so ``maps`` shows
    ``sem.XXXXXX (deleted)`` while the file still has a live name. Left linked,
    CRIU cannot ghost it and link-remaps it instead: it hard-links
    ``/dev/shm/link_remap.<id>`` in the dumping pod, which the image then needs
    at restore -- nowhere else has it, the first restore consumes it, and
    replicas whose trees number their files alike collide on the same ``<id>``
    and fail the dump. So every mapped ``(device, inode)`` under *shm_dir* is
    matched back to the names that still point at it, and those are returned
    too.
    """
    found = set()
    inodes = set()
    for pid in pids:
        try:
            with open(f"{proc}/{pid}/maps") as handle:
                for line in handle:
                    parts = line.split(None, 5)
                    if len(parts) < 6:
                        continue
                    path = parts[5].strip()
                    if not path.startswith(prefixes):
                        continue
                    if not path.endswith("(deleted)"):
                        found.add(path)
                    try:
                        major, minor = (int(x, 16) for x in parts[3].split(":"))
                        inodes.add((os.makedev(major, minor), int(parts[4])))
                    except ValueError:
                        pass
        except OSError:
            pass
        try:
            fds = os.listdir(f"{proc}/{pid}/fd")
        except OSError:
            continue
        for fd_name in fds:
            try:
                link = os.readlink(f"{proc}/{pid}/fd/{fd_name}")
            except OSError:
                continue
            if link.startswith(prefixes) and not link.endswith("(deleted)"):
                found.add(link)
    if inodes:
        names = tuple(p[len("/dev/shm/"):] for p in prefixes
                      if p.startswith("/dev/shm/"))
        try:
            entries = os.listdir(shm_dir)
        except OSError:
            entries = []
        for name in entries:
            if not name.startswith(names):
                continue
            path = os.path.join(shm_dir, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            if (st.st_dev, st.st_ino) in inodes:
                found.add(path)
    return sorted(found)


def _tree_pids(root, proc="/proc"):
    """*root* and every descendant, via ``/proc/<pid>/task/<tid>/children``."""
    seen = []
    stack = [root]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.append(pid)
        try:
            tids = os.listdir(f"{proc}/{pid}/task")
        except OSError:
            continue
        for tid in tids:
            try:
                with open(f"{proc}/{pid}/task/{tid}/children") as handle:
                    stack.extend(int(c) for c in handle.read().split())
            except (OSError, ValueError):
                pass
    return seen


def _unlink_paths(paths):
    removed = []
    for path in paths:
        try:
            os.remove(path)
            removed.append(path)
        except OSError:
            pass
    return removed


def _prepare_worker_dump(worker):
    """Clean up per-worker FDs / io_uring / IB verbs mappings for CRIU dump.
    Invoked from the parent's prepare_criu_dump handler via collective_rpc so
    each TP worker subprocess sheds state CRIU cannot serialize."""
    pid = os.getpid()
    closed_fds = []
    unmapped = []
    # fd 1/2 stay on the pod-log pipe inherited from the child: CRIU dumps it as
    # the same external pipe as the child's, and the restore hands it back.
    # Complication 15, worker side.  At TP>1 each rank is its own process with its
    # own torch rendezvous listeners, so the driver-side close in
    # prepare_criu_dump cannot reach them -- and these are the only ones that
    # matter: CRIU seized nine processes for the TP8 image and every recorded
    # listener sat in one of the eight workers, none in the driver.
    # `destroy_nccl` runs before `criu_dump` (auto-inserted by
    # Instance.cuda_checkpoint when n_gpus > 1), so the group is down by now.
    #
    # **Nothing is closed here.**  This runs only at TP>1 -- prepare_criu_dump
    # gates the collective_rpc on len(gpus) > 1 -- and at TP>1 torch has already
    # retired its own rendezvous listeners by the time the dump prep runs.
    # Measured on a 35B TP=2: 28 ephemeral loopback listeners across the two
    # ranks during init, 26 of them gone without anyone closing anything, and the
    # one per rank that survived belonged to NCCL's RAS thread, which no torch
    # teardown stops.  Closing that one is what produced a restore that spun
    # accept() on EBADF at ~143 MB/s while the job reported RUNNING.
    #
    # So the ports ride into the image, which is both what RAS needs and the only
    # thing the restore survives.  It costs no collision exposure that was
    # avoidable: the survivor is RAS's, and RAS needs it recorded either way.
    # The TP=1 driver path in prepare_criu_dump still closes, because there the
    # listeners really are torch's and RAS is not running -- see
    # `_close_loopback_listeners`.
    #
    # The census travels back in the reply as well as the log, because what the
    # image records is the thing a restore failure will ask about.
    listener_diag = {"pg_aborted": _PG_ABORTED,
                     "store_threads": _live_store_threads()}
    try:
        import torch.distributed as _dist
        listener_diag["dist_initialized"] = bool(_dist.is_initialized())
    except Exception as _e:  # noqa: BLE001
        listener_diag["dist_initialized"] = f"unreadable: {_e!r}"
    try:
        _eph, _fixed = _loopback_listeners()
        # Named for what they are -- what this image will carry -- rather than
        # `would_close`, which implied a close was being withheld pending some
        # condition.  It is not: at TP>1 there is no condition under which these
        # should be closed.
        listener_diag["recorded_ephemeral"] = [tag for _, tag in _eph]
        listener_diag["recorded_fixed"] = [tag for _, tag in _fixed]
    except Exception as _e:  # noqa: BLE001
        listener_diag["error"] = repr(_e)

    # Before the fd sweep below closes them, so the fds still name the files.
    own_psm = _own_shm_paths([pid], _PSM_PREFIXES)

    close_anon = ("infinibandevent", "io_uring")
    close_shm = ("/dev/shm/psm_",)
    keep_prefixes = ("/dev/nvidia", "/dev/shm", "anon_inode:", "socket:", "pipe:")
    for fd_name in sorted(os.listdir(f"/proc/{pid}/fd"), key=int):
        try:
            fd_int = int(fd_name)
            if fd_int <= 2:
                continue
            link = os.readlink(f"/proc/{pid}/fd/{fd_name}")
            if link.startswith("anon_inode:"):
                if any(bad in link for bad in close_anon):
                    os.close(fd_int)
                    closed_fds.append(fd_int)
                continue
            if any(link.startswith(p) for p in close_shm):
                os.close(fd_int)
                closed_fds.append(fd_int)
                continue
            if any(link.startswith(p) for p in keep_prefixes):
                continue
            os.close(fd_int)
            closed_fds.append(fd_int)
        except (OSError, ValueError):
            pass
    libc = ctypes.CDLL("libc.so.6")
    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            if (("io_uring" in line) or ("/dev/infiniband/" in line)
                    or ("uverbs" in line)):
                start_s, end_s = line.split()[0].split("-")
                start = int(start_s, 16)
                length = int(end_s, 16) - start
                libc.munmap(ctypes.c_void_p(start), ctypes.c_size_t(length))
                unmapped.append(f"0x{start:x}")

    # PSM (IB/EFA) shm: do NOT munmap -- the mapping may still be referenced by
    # a live provider thread even after NCCL teardown, so munmap SIGSEGVs the
    # worker.  Instead unlink the /dev/shm/psm_* file while keeping the mapping;
    # the inode stays alive so the worker is unaffected, and CRIU then captures
    # the mapping as anonymous memory (no file to reopen on restore).
    removed_psm = _unlink_paths(own_psm)
    return {"closed_fds": closed_fds, "unmapped": unmapped,
            "removed_psm": removed_psm,
            "listener_diag": listener_diag}


# ---------------------------------------------------------------------------
# CUDA graph handling around CRIU (worker-local, via collective_rpc).
#
# Graphs are preserved in the CRIU image and their stale CustomAllreduce
# addresses are rewritten after reinit by ca_graph_rebind.  Nothing is ever
# recaptured: the `full` path that dropped the graphs and rebuilt them with
# capture_model() was retired 2026-09-24, along with the cleargraph primitive
# that only it could reach.
# ---------------------------------------------------------------------------
def _semip_prepare_graph_reuse_snapshot(worker):
    """Cold-start hook: record the CA meta/buffer/rank_data snapshot the
    post-reinit rebind rewrites against.  No-op at TP1 or without ca_graph_rebind.
    The cold capture was already forced onto the CA copy path + keep_graph by
    SemipGPUWorker.compile_or_warm_up_model, so this only records state."""
    if _semip_tp_size(worker) <= 1:
        return {"enabled": False, "reason": "tp1"}
    if not _CA_REBIND_AVAILABLE or ca_graph_rebind is None:
        return {"available": False}
    try:
        return ca_graph_rebind.store_snapshot(worker, None, None)
    except Exception as e:  # noqa: BLE001
        return {"available": True, "ok": False, "error": f"{type(e).__name__}: {e}"}


def _semip_rebind_graphs(worker):
    """Repair preserved CUDA graphs against the post-restore runtime by rewriting
    the moved CustomAllreduce addresses (ca_graph_rebind).  No capture_model().

    This is address staleness, not missing execs.  A warm image carries every
    cudaGraphExec_t through CRIU intact (measured on GLM-5.3 at TP=8: 4080
    exec_ok and 0 uninstantiated at the dump, and the same on the restored
    side).  What does not survive is the CA meta/buffer *contents* baked into
    the graph nodes, because destroy_nccl -> reinit_nccl reallocates them at
    new addresses -- 144126 kernel argument slots and 16014 memcpy pointers per
    rank on that run.  No amount of dump-time warming can pre-empt that; the
    addresses only go stale after the restore.
    """
    if _semip_tp_size(worker) <= 1:
        torch.cuda.synchronize()
        return {"ok": True, "recaptured": False,
                "skipped": "tp1_no_rebind",
                "rank": getattr(worker, "rank", "?")}
    if not _CA_REBIND_AVAILABLE or ca_graph_rebind is None:
        return {"ok": False, "error": "ca_graph_rebind unavailable",
                "rank": getattr(worker, "rank", "?")}
    ca_rebind = ca_graph_rebind.rebind_after_reinit(worker)
    torch.cuda.synchronize()
    return {"ok": bool(ca_rebind.get("ok")), "recaptured": False,
            "ca_rebind": ca_rebind,
            "rank": getattr(worker, "rank", "?")}


def _semip_graph_census(worker):
    """Per-rank count of captured graphs that currently hold a cudaGraphExec_t.

    Cheap enough to call between warmup rungs: a warm image holds the count flat
    across the whole ladder, a cold one steps by one wrapper-set per rung, and
    that difference is the direct read on whether a rung replayed or built."""
    if not _CA_REBIND_AVAILABLE or ca_graph_rebind is None:
        return {"ok": False, "error": "ca_graph_rebind unavailable",
                "rank": getattr(worker, "rank", "?")}
    out = ca_graph_rebind.graph_exec_census(worker)
    out["rank"] = getattr(worker, "rank", "?")
    return out


def _log_graph_census(llm, log, when):
    """Run the census on every rank, log one compact line each, return the dicts.

    The return value is what lets a caller enforce ``uninstantiated == 0``
    rather than leave it to whoever reads the log.  An empty list means the
    census itself failed, which is deliberately not the same as a clean reading:
    a census must never be the reason a dump or a restore fails, so callers that
    treat it as an invariant have to decide what an absent answer means.
    """
    try:
        census = [c for c in llm.collective_rpc(_semip_graph_census)
                  if isinstance(c, dict)]
    except Exception:
        log.warning("  census[%s] failed", when, exc_info=True)
        return []
    for _c in census:
        log.info("  census[%s] rank=%s exec_ok=%s uninstantiated=%s "
                 "shapes=%s per_shape=%s", when, _c.get("rank", "?"),
                 _c.get("n_exec_ok"), _c.get("n_uninstantiated"),
                 _c.get("n_shapes_instantiated"),
                 _c.get("execs_per_shape_hist"))
    return census


def vllm_child_loop(pipe_conn, instance_id, gpus, model_dir=None):
    """Runs in a spawned child process: owns CUDA and vLLM.

    ``gpus`` is the physical GPU list for this instance (a single-element
    list at TP=1).  The main loop has two modes:
    - **Idle**: blocks on pipe_conn.recv() (zero CPU).
    - **Active** (engine has unfinished requests): alternates between
      engine.step() and non-blocking pipe_conn.poll() so new generate
      requests can be submitted mid-decode.
    """
    if isinstance(gpus, int):
        gpus = [gpus]
    gpus = list(gpus)
    rank = gpus[0]
    if len(gpus) > 1:
        # TP>1: keep ALL GPUs visible so tensor parallelism can span the
        # group; each vLLM worker is placed on its physical GPU by
        # SemipGPUWorker.init_device via SEMIP_GPU_MAP.  Any inherited
        # CUDA_VISIBLE_DEVICES mask must be cleared first, else the physical
        # indices in SEMIP_GPU_MAP disagree with the visible-device namespace
        # (and it would confuse the cuda-checkpoint physical-GPU addressing).
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["SEMIP_GPU_MAP"] = ",".join(str(g) for g in gpus)
    else:
        # TP1: unchanged single-GPU behavior -- pin to the one GPU so it is
        # visible as device 0.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["USE_LIBUV"] = "0"

    # Bind vLLM's internal rendezvous (the torch.distributed TCPStore) to
    # loopback instead of the node's routable IP.  get_ip() otherwise picks
    # the routable interface address, which CRIU bakes into the image as the
    # listening socket's bound address; restoring on a different node then
    # fails at bind() with EADDRNOTAVAIL.  127.0.0.1 exists identically on
    # every node, so loopback makes images node-portable.
    os.environ["VLLM_HOST_IP"] = "127.0.0.1"
    # VLLM_HOST_IP only steers vLLM's own rendezvous.  The collective
    # libraries pick their transport interface independently and default to
    # the routable NIC: gloo keeps a persistent listening socket for the life
    # of the process and NCCL opens a bootstrap listener, both of which get
    # baked into the image bound to the capture node's IP.  Pin them to
    # loopback so every internal socket binds to 127.0.0.1.  An instance is
    # single-node even at TP>1 -- the TP group's ranks are local, so their
    # NCCL bootstrap reaches across loopback fine and the data path rides
    # NVLink/P2P rather than these sockets.
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"

    # JIT/compile caches (Triton, vLLM torch.compile, torch inductor,
    # FlashInfer) all produce .so's that get dlopen()'d into the process.
    # CRIU records those mappings by absolute path and requires the files
    # to exist at restore; their defaults live in node-local dirs
    # ($HOME/.triton, ~/.cache/vllm, /tmp/torchinductor_*), so on another
    # node they are absent and restore fails with "Can't open file ...".
    #
    # When the Instance supplies a model_dir, the cache lives at
    # <model_dir>/compilation -- embedded next to the CRIU image so the
    # compile cache is isolated per model and travels with the image as one
    # unit.  This makes restore-on-another-node require model_dir to exist
    # at the same absolute path on that node (copy the whole model_dir over
    # first).  Absent a model_dir, the caches keep their defaults, which is
    # correct for same-node restore.
    #
    # Must be set before vLLM (and FlashInfer, whose env module resolves
    # these at import) is imported.
    if model_dir:
        _compile_root = os.path.join(model_dir, "compilation")
        os.environ["TRITON_CACHE_DIR"] = os.path.join(_compile_root, "triton")
        os.environ["VLLM_CACHE_ROOT"] = os.path.join(_compile_root, "vllm")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
            _compile_root, "inductor")
        os.environ["FLASHINFER_WORKSPACE_BASE"] = os.path.join(
            _compile_root, "flashinfer")
        for _cache_dir in (os.environ["TRITON_CACHE_DIR"],
                           os.environ["VLLM_CACHE_ROOT"],
                           os.environ["TORCHINDUCTOR_CACHE_DIR"],
                           os.environ["FLASHINFER_WORKSPACE_BASE"]):
            os.makedirs(_cache_dir, exist_ok=True)

    # vLLM's own lines carry the replica tag too, since several replicas share
    # one pod log. Before vLLM is imported: its logger formats this in once.
    os.environ.setdefault(
        "VLLM_LOGGING_PREFIX",
        f"[r{os.environ.get('SEMIP_REPLICA_ID') or '0'}] ")

    semip_logging.init_process(role="child")
    log = semip_logging.child(instance_id, rank)
    # Route this process's stdout/stderr to the pod log from the very first
    # byte. The TP ranks inherit both fds, and CRIU dumps them as an external
    # pipe that the restore hands back with --inherit-fd -- so a restored tree
    # keeps writing to the new pod's log with nothing to rebind.
    semip_logging.redirect_stdio_to_pod_log()

    # Detach from the controlling terminal before anything is captured.
    # fd 1/2 are already the pod log (above); fd 0 is still
    # the interactive shell's pts, inherited down the spawn chain.  A pts
    # on fd 0 makes CRIU dump the process as a --shell-job tied to an
    # external terminal/session, which cannot be reattached when the tree
    # is restored inside a private PID namespace (criu tty.c: TIOCSPGRP
    # fails because the namespaced pgid can't own the host terminal).
    # Point fd 0 at /dev/null and start a fresh session so the captured
    # tree owns its own session and holds no controlling terminal.
    try:
        _devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(_devnull, 0)
        os.close(_devnull)
    except OSError:
        pass
    try:
        os.setsid()
    except OSError:
        # Already a session/group leader (rare for a spawned child); the
        # fd-0 redirect above is the part that matters for CRIU.
        pass

    # TP1 pins its single GPU via CUDA_VISIBLE_DEVICES, so it is device 0
    # here.  At TP>1 all GPUs stay visible, so address the group's first
    # physical GPU directly.
    torch.cuda.set_device(0 if len(gpus) == 1 else gpus[0])

    llm = None
    engine = None
    # The staging buffer, param index and chunk plan live on each vLLM worker
    # (``worker._semip_*``), not here -- see the worker-local primitives above.
    # A buffer held in this process could not be written by the workers at
    # TP>1, where collective_rpc cloudpickles the callable into subprocesses.

    _active_reqs = {}     # req_id -> {"t0", "engine_ids", "finished"}
    _engine_to_req = {}   # engine_request_id -> req_id
    _next_engine_id = 0
    _deferred_cmds = []   # non-generate commands received during drain

    # Pause/resume state.  `_paused` gates the engine.step() call in
    # the main loop, and is also the single switch that routes
    # generate-while-paused submits: `_submit_generate` parks them
    # in `_saved_requests` (skipping the engine entirely) iff
    # `_paused` is True.  `_saved_requests` is populated by both
    # `pause` (which snapshots active requests and aborts them in
    # the engine) and post-pause `_submit_generate` (which
    # synthesises a never-stepped record), then drained on `resume`
    # via `engine.add_request` for each entry.  All of this lives
    # in plain Python state so CRIU dumps and restores it for free
    # across cuda_checkpoint/cuda_restore cycles, and keeping the
    # engine untouched while paused makes the path robust to any
    # pipe interleaving of pause / sleep / cuda_checkpoint /
    # generate that the orchestrator's "Walking down past `up`
    # while paused" rule permits.
    #
    # `_dormant` is a separate, *defensive* flag that brackets the
    # span where the vLLM engine is unsafe to mutate because
    # `llm.sleep(level=2)` discarded its KV cache and possibly
    # `cuda-checkpoint` froze the entire CUDA context.  Set True at
    # the bottom of the `sleep` handler, False at the bottom of the
    # `wake_up_kv_cache` handler.  `_submit_generate` checks it
    # BEFORE `_paused` and, when True, sends back a `generate_done`
    # ack carrying a `RuntimeError("generate against dormant
    # engine")` instead of touching the engine -- so any race that
    # slips past the orchestrator's Phase-2 eviction sentinel
    # (`Orchestrator._evict_for_phase2`) surfaces as a loud, fail-
    # fast future exception in `_on_generate_done` instead of a
    # silent hang inside `engine.step()` on a torn-down executor.
    # Defense in depth: with the sentinel intact this branch is
    # unreachable in normal operation; the historical record of the
    # `_engine_dormant` / `_paused` unification (commit `ad74086`)
    # is in `orchestrator_DESIGN.md` "Eviction-mid-generate
    # dormant-engine wedge".
    _paused = False
    _dormant = False
    _saved_requests = []

    def _alloc_engine_id():
        nonlocal _next_engine_id
        eid = f"req-{_next_engine_id}"
        _next_engine_id += 1
        return eid

    def _submit_generate(req_id, prompts, sampling_params_dict):
        if _dormant and not _paused:
            # Defense-in-depth fail-fast: the orchestrator should
            # never enqueue a generate cmd onto an engine that has
            # been put to sleep without a corresponding pause (the
            # Phase-2 eviction sentinel in
            # ``Orchestrator._evict_for_phase2`` gates this).  If
            # that gate ever has a hole, abort with a loud error
            # ack instead of silently hanging inside
            # ``engine.step()`` on a torn-down executor.  Routes
            # through the demuxer's standard error path
            # (``error is not None`` on the result tuple), which
            # latches the error, decrements ``_pending_count``
            # cleanly, and surfaces to ``Orchestrator.
            # _on_generate_done`` as the ``error`` arg so the
            # in-flight ``done_event.set()`` happens with
            # ``q_rec["state"]="error"``.
            err = RuntimeError(
                f"generate req_id={req_id} arrived against dormant "
                f"engine (sleep without prior pause); orchestrator "
                f"sentinel breach -- see orchestrator_DESIGN.md "
                f"'Eviction-mid-generate dormant-engine wedge'")
            log.error(
                "  _dormant fail-fast: rejecting req_id=%s "
                "(prompts=%s)  -- %s",
                req_id, _truncate_for_display(list(prompts)), err)
            pipe_conn.send((
                "generate_done", 0.0, err, {"req_id": req_id}))
            return
        if _paused:
            # Single rule: while paused, the vLLM engine sees no
            # scheduler mutations from this child.  Park the request
            # in `_saved_requests` and let the next `resume` reload
            # it; `pause` already did the same for whatever was
            # in-flight at pause-time, so on resume the deferred
            # entries and the pause-snapshotted entries flow back
            # into the engine through one code path.
            #
            # This keeps the engine untouched for the entire dormant
            # span -- `llm.sleep` discards cumem-allocated KV blocks
            # and `cuda-checkpoint` (during `cuda_checkpoint`)
            # freezes the CUDA context, so any `engine.add_request`
            # / `engine.abort_request` call inside that window would
            # either enqueue into a scheduler that can never `step`
            # or block on a torn-down executor.  It is also
            # order-independent w.r.t. pipe interleavings of
            # generate/sleep/checkpoint/etc. while paused.
            #
            # `prompt_token_ids: []` (not None) matches the shape
            # `_snapshot_active_into_saved` produces for an empty
            # per-eid state via its `list(... or [])` clause, so
            # the resume branch's `len(prompt_tids)` works and its
            # `if prompt_tids:` test falls through to the
            # `elif i < len(prompts_orig)` re-prefill branch.
            _saved_requests.append({
                "req_id": req_id,
                "t0": time.perf_counter(),
                "first_token_ts": None,
                "prompts": list(prompts),
                "sampling_params": dict(sampling_params_dict),
                "eids": [{"prompt_token_ids": [],
                          "output_token_ids": [],
                          "output_text": ""}
                         for _ in prompts],
            })
            log.info("  submitted req_id=%s  prompts=%s  "
                     "(deferred to _saved_requests; paused)",
                     req_id, _truncate_for_display(list(prompts)))
            return

        from vllm import SamplingParams
        sp = SamplingParams(**sampling_params_dict)
        engine_ids = []
        for prompt in prompts:
            eid = _alloc_engine_id()
            engine.add_request(eid, prompt, sp)
            _engine_to_req[eid] = req_id
            engine_ids.append(eid)
        # `per_eid` tracks the latest cumulative engine output per
        # sub-request so that `pause` can snapshot the current state
        # without poking engine internals.  Updated in
        # `_process_step_outputs` on every step.
        per_eid = {eid: {"prompt_token_ids": None,
                         "output_token_ids": [],
                         "output_text": ""} for eid in engine_ids}
        _active_reqs[req_id] = {
            "t0": time.perf_counter(),
            "engine_ids": engine_ids,
            "finished": {},
            "prompts": list(prompts),
            "first_token_ts": None,
            "sampling_params": dict(sampling_params_dict),
            "per_eid": per_eid,
        }
        log.info("  submitted req_id=%s  prompts=%s",
                 req_id, _truncate_for_display(list(prompts)))

    def _process_step_outputs(step_outputs):
        for output in step_outputs:
            eid = output.request_id
            req_id = _engine_to_req.get(eid)
            if req_id is None:
                continue
            entry = _active_reqs.get(req_id)
            if entry is None:
                continue

            # First-token detection: stamp on the first step that
            # produced any decoded tokens for any sub-request of this
            # req_id.  output_kind defaults to CUMULATIVE so token_ids
            # is the running total -- non-empty iff at least one token
            # has been generated.
            if entry["first_token_ts"] is None and any(
                    o.token_ids for o in output.outputs):
                entry["first_token_ts"] = time.perf_counter()

            # Per-eid cumulative snapshot used by `pause`.  This must
            # happen on every step (not just the finishing one)
            # because pause can be invoked mid-decode.  We track only
            # the n=1 case (outputs[0]).
            per_eid_state = entry.get("per_eid", {}).get(eid)
            if per_eid_state is not None:
                if (per_eid_state["prompt_token_ids"] is None
                        and output.prompt_token_ids):
                    per_eid_state["prompt_token_ids"] = list(
                        output.prompt_token_ids)
                if output.outputs:
                    per_eid_state["output_token_ids"] = list(
                        output.outputs[0].token_ids)
                    per_eid_state["output_text"] = output.outputs[0].text

            if not output.finished:
                continue
            _engine_to_req.pop(eid, None)
            entry["finished"][eid] = output

            if len(entry["finished"]) == len(entry["engine_ids"]):
                ordered = [entry["finished"][e] for e in entry["engine_ids"]]

                # If this entry was resumed via `resume`, fold
                # pre-pause output back into the reported view
                # so the caller sees seamless continuation.
                pre_completion = entry.get("pre_pause_completion")
                pre_text = entry.get("pre_pause_text")
                orig_prompt_tokens = entry.get("original_prompt_tokens")

                if pre_completion is not None:
                    eid_index = {e: i for i, e in enumerate(entry["engine_ids"])}
                    outputs = [
                        [pre_text[eid_index[r.request_id]] + o.text
                         for o in r.outputs]
                        for r in ordered]
                    completion_tokens = sum(
                        len(o.token_ids)
                        + pre_completion[eid_index[r.request_id]]
                        for r in ordered for o in r.outputs)
                    prompt_tokens = sum(orig_prompt_tokens)
                else:
                    outputs = [[o.text for o in r.outputs] for r in ordered]
                    prompt_tokens = sum(
                        len(r.prompt_token_ids) for r in ordered)
                    completion_tokens = sum(
                        len(o.token_ids) for r in ordered for o in r.outputs)
                cached_tokens = sum(
                    (r.num_cached_tokens or 0) for r in ordered)
                finish_reasons = sorted({
                    o.finish_reason for r in ordered for o in r.outputs
                    if o.finish_reason is not None
                })
                info = {
                    "req_id": req_id,
                    "outputs": outputs,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "num_cached_tokens": cached_tokens,
                    "finish_reasons": finish_reasons,
                }

                t_done = time.perf_counter()
                elapsed = t_done - entry["t0"]
                first_token_ts = entry["first_token_ts"]
                ttft = (first_token_ts - entry["t0"]
                        if first_token_ts is not None else None)
                decode_time = (t_done - first_token_ts
                               if first_token_ts is not None else None)
                tpot_ms = (decode_time * 1000.0 / (completion_tokens - 1)
                           if (decode_time is not None
                               and completion_tokens > 1) else None)
                gen_tput = (completion_tokens / elapsed
                            if elapsed > 0 else 0.0)

                info["ttft_s"] = ttft
                info["decode_s"] = decode_time
                info["tpot_ms"] = tpot_ms
                info["gen_tput_tok_s"] = gen_tput

                prompts = entry.get("prompts")
                del _active_reqs[req_id]
                log.info(
                    "<<< generate req_id=%s OK (%.3fs)  "
                    "prompt_tokens=%s  completion_tokens=%s  "
                    "cached_tokens=%s  finish=%s  "
                    "prompt=%s  output=%s",
                    req_id, elapsed,
                    prompt_tokens, completion_tokens,
                    cached_tokens, finish_reasons,
                    _truncate_for_display(prompts),
                    _truncate_for_display(outputs),
                )
                log.info(
                    "    perf  ttft=%s  decode=%s  tpot=%s  "
                    "gen_tput=%.1f tok/s",
                    f"{ttft * 1000:.1f}ms" if ttft is not None else "n/a",
                    (f"{decode_time:.3f}s"
                     if decode_time is not None else "n/a"),
                    (f"{tpot_ms:.2f}ms"
                     if tpot_ms is not None else "n/a"),
                    gen_tput,
                )
                pipe_conn.send(("generate_done", elapsed, None, info))

    def _drain_engine():
        while engine is not None and engine.has_unfinished_requests():
            _process_step_outputs(engine.step())

    # The five rungs the post-restore pass drives to verify the rebind.  Still
    # wave 9's ladder: it is no longer a warmup, but keeping the shapes fixed
    # keeps its timings comparable against every earlier wave.
    _REBIND_VERIFY_SIZES = (1, 2, 4, 8, 16)

    def _capture_sizes():
        """Every batch size vLLM captured a graph for, ascending.

        The warm pass drives this rather than a hand-picked ladder. Instantiation
        is already shape-agnostic -- ``instantiate_captured_graphs`` walks every
        entry of every wrapper -- but JIT is not: a kernel is only compiled by
        running the shape that dispatches to it, and the dispatch demonstrably
        moves with batch size (the CuTeDSL delta-rule kernel first appears
        between decode 8 and 16, while the preserved graphs bake the Triton one).
        Warming five of fifty-one shapes would just relocate the first-run
        compile from the restore warmup into live traffic.

        Falls back to the old ladder if the config cannot be read: a warm pass
        over the wrong sizes is still better than none, and this must not be
        able to fail a cold start.
        """
        try:
            cc = llm.llm_engine.vllm_config.compilation_config
            sizes = sorted({int(s) for s in
                            (cc.cudagraph_capture_sizes or []) if int(s) > 0})
            if sizes:
                return tuple(sizes)
        except Exception:  # noqa: BLE001
            log.warning("  could not read cudagraph_capture_sizes; warming the "
                        "verify rungs only", exc_info=True)
        return _REBIND_VERIFY_SIZES

    def _ladder_plan(sizes):
        """The original warmup's ``(nreq, toklen)`` pairs for a set of sizes.

        Kept exactly as wave 9 ran it -- a 64-token budget per rung, capped at
        20 tokens per request -- because the restore path's timings are only
        comparable against the earlier waves if its rungs are identical.
        """
        return [(n, max(1, min(64 // n, 20))) for n in sizes]

    def _warm_plan(sizes):
        """``(nreq, toklen)`` pairs that reach every captured shape on both axes.

        The size sweep covers all the decode shapes, and -- because the token
        budget collapses toklen to 1 once nreq passes 64 -- most of the mixed
        prefill ones too. What it cannot reach is the small end: at nreq below
        40 the budget yields 20-, 40- and 64-token prefills, so a 1-, 2-, 4-,
        8-, 16- or 32-token mixed batch never occurs. Those are exactly the
        shapes a single short request lands on, so leaving them cold would move
        the first-run JIT out of the warmup and into live traffic, which is the
        failure this whole change exists to remove.

        A single-request pass closes them. Prompts are held a little under
        max_model_len so the two sampled tokens still fit.
        """
        plan = _ladder_plan(sizes)

        def _pad(n):
            for s in sizes:
                if s >= n:
                    return s
            return None

        covered = {_pad(n * t) for n, t in plan}
        try:
            budget = int(llm.llm_engine.vllm_config.model_config.max_model_len) - 4
        except Exception:  # noqa: BLE001
            budget = 0
        missed = [s for s in sizes if s not in covered]
        reachable = [s for s in missed if 0 < s <= budget]
        plan += [(1, s) for s in reachable]
        if missed:
            log.info("  mixed shapes the size sweep misses: %s; warming %s "
                     "single-request (budget=%d tokens)",
                     missed, reachable, budget)
        return plan

    def _drive_warmup_ladder(phase, plan):
        """Drive a set of batch shapes through the engine.

        Runs on both sides of the checkpoint, doing a different job on each.
        Before the dump it is what makes the image warm: each rung forces the
        lazy first-replay instantiate of its shapes and the first-run JIT of the
        kernels only that shape reaches, so the restore finds both done. After a
        restore it builds nothing -- the warm image already carries the execs --
        and serves as the first collective traffic after the rebind, which is
        what would catch a rebind that read back clean but does not replay.

        Instrumented per rung. This loop, not the rebind, is where the 302 s
        hangs happened: a rank wedges, the peer starves in shm_broadcast, and the
        stall surfaces as "RPC call to sample_tokens timed out" because
        engine.step() samples. A rung that never completes leaves its "start"
        line as the last one rather than nothing at all. Those hangs were a cold
        image paying for its instantiate here, four ranks deep; warming the dump
        is what removed them.

        Callers census either side of this, never between rungs: a collective_rpc
        in the gaps would drop a fresh synchronisation point immediately before
        nreq=16, the rung the hang lives on, and perturbing the one measurement
        we are trying to take is not worth the detail.
        """
        from vllm import SamplingParams
        for nreq, toklen in plan:
            log.info("    %s nreq=%d toklen=%d start", phase, nreq, toklen)
            _t_warm = time.monotonic()
            for _ in range(nreq):
                engine.add_request(
                    _alloc_engine_id(),
                    {"prompt_token_ids": [0] * toklen},
                    SamplingParams(max_tokens=2, ignore_eos=True))
            _steps = 0
            while engine.has_unfinished_requests():
                engine.step()
                _steps += 1
            log.info("    %s nreq=%d OK (%.3fs, %d steps)",
                     phase, nreq, time.monotonic() - _t_warm, _steps)

    def _snapshot_active_into_saved() -> tuple[int, int]:
        """Snapshot every active sub-request into ``_saved_requests`` and
        abort it in the engine.  Mirrors the pause-time snapshot path,
        factored out so callers other than ``pause`` (e.g. a ``sleep``
        arriving on a paused engine that picked up a generate after
        pause) can preserve those requests for the next ``resume``
        instead of silently draining them to completion.

        Returns ``(saved_count, aborted_eid_count)``.  Safe to call when
        ``_active_reqs`` is empty (returns ``(0, 0)``).  Caller is
        responsible for any state flag updates (``_paused``) and for
        emitting the appropriate log line; this helper only touches the
        ledgers.
        """
        saved = []
        for req_id, entry in list(_active_reqs.items()):
            sp_dict = entry.get("sampling_params") or {}
            n_branch = sp_dict.get("n", 1)
            if n_branch != 1:
                raise RuntimeError(
                    f"snapshot with n={n_branch} not supported "
                    "(n=1 only)")
            eids_data = []
            for eid in entry["engine_ids"]:
                per_eid_state = entry["per_eid"].get(eid, {})
                eids_data.append({
                    "prompt_token_ids": list(
                        per_eid_state.get("prompt_token_ids") or []),
                    "output_token_ids": list(
                        per_eid_state.get("output_token_ids") or []),
                    "output_text":
                        per_eid_state.get("output_text", ""),
                })
            saved.append({
                "req_id": req_id,
                "t0": entry["t0"],
                "first_token_ts": entry["first_token_ts"],
                "prompts": list(entry.get("prompts") or []),
                "sampling_params": dict(sp_dict),
                "eids": eids_data,
            })

        all_eids = [eid
                    for entry in _active_reqs.values()
                    for eid in entry["engine_ids"]]
        if all_eids:
            try:
                engine.abort_request(all_eids)
            except Exception as _e:
                log.warning("  snapshot: abort_request failed: %s", _e)

        _saved_requests.extend(saved)
        _active_reqs.clear()
        _engine_to_req.clear()
        return len(saved), len(all_eids)

    def _handle_command(cmd, kwargs):
        nonlocal llm, engine
        nonlocal _paused, _dormant

        error = None
        info = {}

        try:
            if cmd == "init":
                vllm_config = dict(kwargs["vllm_config"])
                vllm_config["enable_sleep_mode"] = True

                # TP>1 only: steer the collective path onto shapes that survive
                # a checkpoint.  Fused allreduce+RMS and NVLS/symmetric-memory
                # allreduce keep state CRIU cannot serialize, and the MoE
                # shared-experts side stream complicates graph capture.
                _tp = int(vllm_config.get("tensor_parallel_size", 1) or 1)
                if _tp >= 2:
                    _cc = vllm_config.get("compilation_config")
                    _cc = dict(_cc) if isinstance(_cc, dict) else {}
                    _pc = dict(_cc.get("pass_config") or {})
                    _pc["fuse_allreduce_rms"] = False
                    _cc["pass_config"] = _pc
                    vllm_config["compilation_config"] = _cc
                    os.environ["NCCL_NVLS_ENABLE"] = "0"
                    os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
                    os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"
                    # FlashInfer allreduce has to be off for the same reason,
                    # and it is the rebind's load-bearing assumption rather
                    # than a preference: `ca_graph_rebind` finds the pointers
                    # it must rewrite by matching `cross_device_reduce` kernel
                    # nodes, and FlashInfer's allreduce is not one, so its
                    # workspace addresses ride into the image unrebindable.
                    # This was true by default until vLLM 0.30 flipped
                    # VLLM_ALLREDUCE_USE_FLASHINFER to True, at which point
                    # every TP>1 restore patched 0 nodes of 2142 graphs and
                    # died on an illegal memory access. Pin it rather than
                    # inherit it.
                    os.environ["VLLM_ALLREDUCE_USE_FLASHINFER"] = "0"
                    # ...and that flag is not enough, which cost this ticket
                    # five bugs and twelve waves. It gates `cuda_communicator`
                    # building a FlashInferAllReduce; it does NOT gate
                    # `get_fi_ar_workspace`. Job 14f13e5a caught the real
                    # caller: fp8 models route through the DeepSeek-V3.2 layer
                    # path, which calls the getter directly --
                    #   deepseek_v32/nvidia/model.py:158 fused_allreduce_rms_norm
                    #     common/ops/fused_allreduce_rms_norm.py:41 _can_use_flashinfer
                    #       fused_allreduce_gemma_rms_norm.py:91   get_fi_ar_workspace
                    # -- during `determine_available_memory`'s
                    # `capture_model(profile_only=True)`. So the backends log
                    # honestly reads ['CUSTOM', 'PYNCCL'] while an MNNVL
                    # multicast workspace is built anyway, and that is the one
                    # class of memory `cuCheckpointProcess*` cannot carry:
                    # the restore's `cuMulticastAddDevice` returns
                    # CUDA_ERROR_INVALID_DEVICE and `reinit_nccl` dies.
                    #
                    # Refusing the allocation is the fix. `get_fi_ar_workspace`
                    # returning None is a supported outcome -- it is what every
                    # caller already gets on a GPU without NVSwitch multicast,
                    # and `_can_use_flashinfer` falls back to the unfused path.
                    # With it, GLM-5.3 TP=8 completes the round trip for the
                    # first time on 0.30 (job 14f13e5a, zero failures).
                    #
                    # `setdefault`, not assignment: unlike the pins above this
                    # one keeps an escape hatch, because a future driver or
                    # FlashInfer that can rebuild multicast would want the
                    # workspace back. Put SEMIP_SUPPRESS_FI_AR_WORKSPACE=0 in
                    # the payload's extra_env to get the old behaviour.
                    os.environ.setdefault("SEMIP_SUPPRESS_FI_AR_WORKSPACE", "1")

                # Per-model env vars: vllm_config["_env"] is a reserved
                # mapping applied to os.environ before vLLM is imported,
                # so flags vLLM reads at import time take effect.  The
                # trio set at the top of vllm_child_loop is reserved
                # (CUDA isolation + in-process EngineCore + libuv off);
                # silently drop any attempt to override it from _env.
                _RESERVED_ENV = {
                    "CUDA_VISIBLE_DEVICES",
                    "VLLM_ENABLE_V1_MULTIPROCESSING",
                    "USE_LIBUV",
                    # Loopback pinning: repointing these at a routable NIC
                    # bakes the capture node's IP into the image and breaks
                    # restore elsewhere with EADDRNOTAVAIL.
                    "VLLM_HOST_IP",
                    "NCCL_SOCKET_IFNAME",
                    "GLOO_SOCKET_IFNAME",
                    "SEMIP_GPU_MAP",
                    # Compile-cache roots: CRIU bakes the resulting .so
                    # paths into the image, so a user override here would
                    # make the image unrestorable.
                    "TRITON_CACHE_DIR",
                    "VLLM_CACHE_ROOT",
                    "TORCHINDUCTOR_CACHE_DIR",
                    "FLASHINFER_WORKSPACE_BASE",
                }
                for k, v in (vllm_config.pop("_env", None) or {}).items():
                    if k in _RESERVED_ENV:
                        log.warning(
                            "ignoring reserved env key in _env: %s", k)
                        continue
                    os.environ[k] = str(v)

                # Force vLLM plugins (e.g. arctic_inference) to load
                # *before* `LLM(**vllm_config)` so plugin-installed
                # EngineArgs fields like `ulysses_sequence_parallel_size`
                # are present when EngineArgs is instantiated from
                # vllm_config.  vLLM normally loads plugins itself
                # during LLMEngine construction, but that fires too
                # late for plugins that extend the EngineArgs dataclass.
                # NB: `vllm.plugins` is a submodule, not auto-attached
                # to the `vllm` package on `import vllm` -- use the
                # `from vllm.plugins import ...` form.
                from vllm.plugins import load_general_plugins
                load_general_plugins()
                from vllm import LLM
                llm = LLM(**vllm_config)
                engine = llm.llm_engine
                info["pid"] = os.getpid()

                # Opt this worker out of arctic_inference's level-2
                # sleep/wake fast paths.  We restore main and drafter
                # params from a host-side pinned buffer ourselves
                # (stage / restore_weights), so:
                #   - skip the disk reload of the main model on wake_up
                #   - skip the per-sleep CPU snapshot of drafter
                #     ``named_parameters()`` (drafter ``named_buffers()``
                #     are still snapshotted; sub-MB)
                # Default arctic behavior is preserved for other users
                # because the flags are read via ``getattr(..., False)``.
                #
                # Note: ``GPUModelRunnerPatch.reload_weights`` is not
                # gated -- semi-persistence never calls
                # ``model_runner.reload_weights`` (the patched
                # ``Worker.wake_up`` reaches the unpatched original via
                # ``GPUModelRunnerPatch._orig_reload_weights``), so the
                # drafter-load augmentation in that path is never hit
                # from this child.
                def _enable_semi_persistence_flags(self):
                    # ``self`` here is the ``WorkerWrapperBase`` driver
                    # (see ``UniProcExecutor.collective_rpc`` ->
                    # ``run_method(self.driver_worker, ...)``).  Plain
                    # ``self.X = ...`` writes onto the wrapper; the
                    # wrapper only forwards ``__getattr__`` to
                    # ``self.worker``, so arctic's patched
                    # ``Worker.wake_up`` (where ``self`` is the real
                    # ``Worker``) would never see these flags and would
                    # fall back to the disk reload.  Write through to
                    # ``self.worker`` so the gating actually fires.
                    self.worker._skip_main_reload_on_wake = True
                    self.worker._skip_drafter_param_snapshot = True

                llm.collective_rpc(_enable_semi_persistence_flags)

                # Record the cold-start CustomAllreduce snapshot that the
                # post-reinit graph rebind rewrites against (TP>=2, reuse).
                if _tp >= 2:
                    # The return value used to be dropped. It carries this
                    # rank's meta_ptrs and the cold-start graph inventory --
                    # the only dump-side view of what goes into the image, and
                    # the counterpart to every post-restore dict we have been
                    # reading. Logged per rank rather than summarised: the four
                    # ranks are what the rebind later has to agree about.
                    for _i, _snap in enumerate(llm.collective_rpc(
                            _semip_prepare_graph_reuse_snapshot)):
                        log.info("  graph reuse snapshot rank=%d: %s",
                                 _i, _snap)
                    if engine is None:
                        engine = llm.llm_engine
                    # Warm the image while the process is still healthy, so the
                    # restore inherits built execs and compiled kernels instead
                    # of building them four ranks deep in the warmup ladder.
                    # The per-rank instantiate already ran inside
                    # compile_or_warm_up_model; this is the other half, the part
                    # that needs the engine and so cannot live in the worker.
                    # Unconditional since 2026-09-24: this is part of `init`,
                    # not a knob.  A cold image is not a supported artefact.
                    _sizes = _capture_sizes()
                    _plan = _warm_plan(_sizes)
                    log.info("  warming %d captured shape(s) in %d rung(s)",
                             len(_sizes), len(_plan))
                    _t_warm_all = time.monotonic()
                    try:
                        _drive_warmup_ladder("dump", _plan)
                        log.info("  warm pass done in %.1fs",
                                 time.monotonic() - _t_warm_all)
                    except Exception:
                        # A rung that fails costs warmth, not the image. The
                        # census below reports how far it got, and a
                        # partially warm image still beats no image at all
                        # -- this runs during init, so raising here would
                        # mean the dump never happens.
                        log.warning(
                            "  warm pass failed after %.1fs; dumping a "
                            "partially warm image",
                            time.monotonic() - _t_warm_all, exc_info=True)
                        _drain_engine()
                    # The invariant, checked rather than merely printed. Every
                    # captured graph must hold an exec before the checkpoint;
                    # a cold image restores into the 302 s warmup hang, and
                    # downstream a key minted from one is indistinguishable
                    # from a warm one. Loud rather than fatal because the dump
                    # has not happened yet and a partially warm image is still
                    # worth more than no image -- but nothing should be able to
                    # publish one quietly.
                    _cold = [c for c in _log_graph_census(llm, log, "dump:warm")
                             if (c.get("n_uninstantiated") or 0) > 0]
                    if _cold:
                        log.error(
                            "  COLD IMAGE: %d of %d rank(s) still hold "
                            "uninstantiated graphs after the warm pass %s -- "
                            "do not publish a key from this dump",
                            len(_cold), _tp,
                            [(c.get("rank", "?"), c.get("n_uninstantiated"))
                             for c in _cold])
                    log.info("  compile cache before dump: %s",
                             _compile_cache_fingerprint())

            elif cmd == "attach":
                if llm is None:
                    raise RuntimeError("attach requires init first")
                # Each worker allocates a plain (unpinned) CPU buffer sized to
                # its own shard and records the param layout on itself; pinning
                # is the explicit repin step.  Per-worker byte counts are
                # reported so the instance can size the restore chunk budget by
                # the largest worker shard (not the TP-aggregate sum).
                results = llm.collective_rpc(_semip_attach)
                worker_bytes = [r[0] for r in results]
                total_size = sum(worker_bytes)
                info["pinned_cpu_bytes"] = total_size
                info["pinned_bytes_per_worker"] = worker_bytes
                info["max_pinned_bytes_per_worker"] = max(worker_bytes, default=0)
                log.info("  attached %.2f GiB across %d worker(s)",
                         total_size / 2**30, len(results))

            elif cmd == "attach_pinned":
                raise RuntimeError(
                    "attach_pinned is not implemented for the worker-local "
                    "staging path; use attach -> repin instead")

            elif cmd == "detach":
                if llm is not None:
                    results = llm.collective_rpc(_semip_detach)
                    total = sum(results)
                    log.info("  freed %.2f GiB pinned memory", total / 2**30)

            elif cmd == "unpin":
                if llm is None:
                    raise RuntimeError("unpin requires attach first")
                results = llm.collective_rpc(_semip_unpin)
                log.info("  unpinned %.2f GiB", sum(results) / 2**30)

            elif cmd == "repin":
                if llm is None:
                    raise RuntimeError("repin requires attach first")
                results = llm.collective_rpc(_semip_repin)
                log.info("  repinned %.2f GiB", sum(results) / 2**30)

            elif cmd == "sleep":
                # Invariant: while `_paused` is True, `_active_reqs`
                # is empty -- `pause` snapshot-and-aborts whatever
                # was in flight at pause time and post-pause submits
                # go straight to `_saved_requests` via
                # `_submit_generate`, never touching the engine.
                # `_drain_engine` therefore has no scheduled work to
                # step through here.  If we are not paused, the
                # drain just runs the engine to completion as in a
                # normal cold sleep.
                _drain_engine()
                llm.sleep(level=2)
                torch.cuda.synchronize(0)
                torch.cuda.empty_cache()
                # Set _dormant AFTER llm.sleep so the fail-fast in
                # _submit_generate only fires once the engine is
                # actually torn down.  The flag is a defensive net
                # against the eviction-mid-generate wedge described
                # in orchestrator_DESIGN.md; the orchestrator's
                # Phase-2 sentinel is the primary gate.
                _dormant = True

            elif cmd == "stage":
                if llm is None:
                    raise RuntimeError("stage requires attach first")
                results = llm.collective_rpc(_semip_stage)
                total_bytes = sum(results)
                info["bytes"] = total_bytes
                log.info("  staged %.2f GiB across %d worker(s)",
                         total_bytes / 2**30, len(results))

            elif cmd == "wake_up_weights":
                llm.wake_up(tags=["weights"])

            elif cmd == "plan_restore_weights":
                if llm is None:
                    raise RuntimeError("plan_restore_weights requires init first")
                mb = kwargs.get("max_buffer_bytes")
                results = llm.collective_rpc(_semip_plan_load_weights, args=(mb,))
                worker_bytes = [r["bytes"] for r in results]
                chunk_bytes = [r["chunk_size"] for r in results]
                n_chunks = [r["n_chunks"] for r in results]
                info["bytes"] = sum(worker_bytes)
                info["pinned_bytes_per_worker"] = worker_bytes
                info["max_pinned_bytes_per_worker"] = max(worker_bytes, default=0)
                info["chunk_bytes_per_worker"] = chunk_bytes
                info["max_chunk_bytes_per_worker"] = max(chunk_bytes, default=0)
                info["n_chunks_per_worker"] = n_chunks
                info["n_chunks"] = max(n_chunks, default=0)
                info["chunk_size"] = max(chunk_bytes, default=0)
                log.info("  planned <= %d chunk(s) per worker (total %.2f GiB)",
                         info["n_chunks"], info["bytes"] / 2**30)
                # Predicted against actual, per rank.  Without this the only
                # record of the config budget overshooting free VRAM was the
                # OOM traceback itself, 22 minutes into a restore.
                free_bytes = [r["free_bytes"] for r in results]
                clamped_ranks = [i for i, r in enumerate(results) if r["clamped"]]
                info["free_bytes_per_worker"] = free_bytes
                info["clamped_workers"] = clamped_ranks
                log.info("  staging: host asked %.2f GiB, tightest device free "
                         "%.2f GiB, chunk %.2f GiB%s",
                         (mb if mb else info["bytes"]) / 2**30,
                         min(free_bytes, default=0) / 2**30,
                         min(chunk_bytes, default=0) / 2**30,
                         (f" -- clamped on rank(s) {clamped_ranks}"
                          if clamped_ranks else ""))

            elif cmd == "restore_weights":
                if llm is None:
                    raise RuntimeError("restore_weights requires init first")
                results = llm.collective_rpc(_semip_restore_weights)
                total_bytes = sum(r["bytes"] for r in results)
                total_loaded = sum(r["loaded"] for r in results)
                info["bytes"] = total_bytes
                info["n_chunks"] = max(r["n_chunks"] for r in results)
                log.info("  loaded %d params in <= %d chunk(s) (total %.2f GiB)",
                         total_loaded, info["n_chunks"], total_bytes / 2**30)

            elif cmd == "wake_up_kv_cache":
                llm.wake_up(tags=["kv_cache"])
                # Clear _dormant AFTER wake_up so the engine is
                # actually back up before _submit_generate stops
                # short-circuiting.  Pairs with the set in the
                # `sleep` handler.
                _dormant = False

            elif cmd == "pause":
                if engine is None:
                    raise RuntimeError("pause requires init first")

                was_paused = _paused
                _paused = True

                # Snapshot every active sub-request and abort it in
                # the engine so the upcoming `unpin` / `sleep` /
                # `cuda_checkpoint` runs against an empty scheduler.
                # Pending `generate_done` messages are deferred until
                # `resume` re-adds the requests via prefill.
                saved_count, aborted_count = _snapshot_active_into_saved()

                info["paused"] = True
                info["was_paused"] = was_paused
                info["saved"] = saved_count
                log.info("  pause: saved %d req_id(s) "
                         "(%d sub-requests aborted, was_paused=%s)",
                         saved_count, aborted_count, was_paused)

            elif cmd == "resume":
                if engine is None:
                    raise RuntimeError("resume requires init first")

                was_paused = _paused

                from vllm import SamplingParams
                from vllm.inputs import TokensPrompt

                restored = 0
                synthesized = 0
                for record in _saved_requests:
                    req_id = record["req_id"]
                    sp_dict = dict(record["sampling_params"] or {})
                    n_branch = sp_dict.get("n", 1)
                    if n_branch != 1:
                        raise RuntimeError(
                            f"resume with n={n_branch} not supported "
                            "(n=1 only)")
                    original_max = sp_dict.get("max_tokens")
                    eids_data = record["eids"]
                    prompts_orig = record["prompts"]

                    new_engine_ids = []
                    pre_pause_completion = []
                    pre_pause_text = []
                    original_prompt_tokens = []
                    all_finished_outputs = []

                    for i, eid_data in enumerate(eids_data):
                        prompt_tids = eid_data["prompt_token_ids"]
                        output_tids = eid_data["output_token_ids"]
                        output_text = eid_data["output_text"]
                        original_prompt_tokens.append(len(prompt_tids))
                        all_finished_outputs.append(output_text)

                        remaining = (original_max - len(output_tids)
                                     if original_max is not None else None)
                        if (remaining is not None and remaining <= 0):
                            # Already at max_tokens pre-pause; skip
                            # re-submission and synthesize the result.
                            continue

                        if prompt_tids:
                            full_token_ids = prompt_tids + list(output_tids)
                            prompt_obj = TokensPrompt(
                                prompt_token_ids=full_token_ids)
                        elif i < len(prompts_orig):
                            prompt_obj = prompts_orig[i]
                        else:
                            log.warning(
                                "  resume: req_id=%s eid#%d has no "
                                "prompt_token_ids and no original "
                                "prompt; skipping", req_id, i)
                            continue

                        sp_kwargs = dict(sp_dict)
                        if remaining is not None:
                            sp_kwargs["max_tokens"] = remaining
                        sp = SamplingParams(**sp_kwargs)

                        new_eid = _alloc_engine_id()
                        engine.add_request(new_eid, prompt_obj, sp)
                        _engine_to_req[new_eid] = req_id
                        new_engine_ids.append(new_eid)
                        pre_pause_completion.append(len(output_tids))
                        pre_pause_text.append(output_text)

                    if not new_engine_ids:
                        # Every branch was already finished pre-pause;
                        # emit a synthetic generate_done so the
                        # original waiter unblocks.
                        completion_tokens = sum(
                            len(d["output_token_ids"]) for d in eids_data)
                        prompt_tokens = sum(original_prompt_tokens)
                        synth_info = {
                            "req_id": req_id,
                            "outputs": [[t] for t in all_finished_outputs],
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "num_cached_tokens": 0,
                            "finish_reasons": ["length"],
                            "ttft_s": None,
                            "decode_s": None,
                            "tpot_ms": None,
                            "gen_tput_tok_s": 0.0,
                        }
                        pipe_conn.send(("generate_done", 0.0, None, synth_info))
                        synthesized += 1
                        log.info("  resume: req_id=%s synthesized "
                                 "(all branches at max_tokens pre-pause)",
                                 req_id)
                        continue

                    new_per_eid = {
                        new_eid: {"prompt_token_ids": None,
                                   "output_token_ids": [],
                                   "output_text": ""}
                        for new_eid in new_engine_ids
                    }
                    _active_reqs[req_id] = {
                        "t0": record["t0"],
                        "engine_ids": new_engine_ids,
                        "finished": {},
                        "prompts": list(prompts_orig),
                        "first_token_ts": record["first_token_ts"],
                        "sampling_params": dict(sp_dict),
                        "per_eid": new_per_eid,
                        "pre_pause_completion": pre_pause_completion,
                        "pre_pause_text": pre_pause_text,
                        "original_prompt_tokens": original_prompt_tokens,
                    }
                    restored += 1

                _saved_requests.clear()
                _paused = False

                info["paused"] = False
                info["was_paused"] = was_paused
                info["restored"] = restored
                info["synthesized"] = synthesized
                log.info("  resume: restored=%d synthesized=%d "
                         "(was_paused=%s)",
                         restored, synthesized, was_paused)

            elif cmd == "get_pipe_fd":
                info["pipe_fd"] = pipe_conn.fileno()

            elif cmd == "prepare_criu_dump":
                _drain_engine()

                # TP>1: each worker is a separate subprocess with its own FDs /
                # io_uring rings / IB verbs mappings that CRIU cannot serialize.
                # Clean them up before the tree is dumped.  Skipped at TP1, where
                # the "worker" is this same (driver) process -- running it here
                # would close this process's own FDs (incl. the worker pipe).
                if llm is not None and len(gpus) > 1:
                    try:
                        wr = llm.collective_rpc(_prepare_worker_dump)
                        info["worker_closed_fds"] = [r["closed_fds"] for r in wr]
                        info["worker_unmapped"] = [r["unmapped"] for r in wr]
                        # No `worker_closed_listeners`: the TP>1 path closes
                        # nothing, and an always-empty field reads like a close
                        # that found no work rather than one that is not
                        # attempted.  The census below is what this image
                        # carries, and it is the thing a restore failure asks
                        # about.
                        info["worker_listener_diag"] = [
                            r.get("listener_diag", {}) for r in wr]
                    except Exception as _e:
                        log.warning(
                            "  prepare_criu_dump: worker dump prep error: %s", _e)

                closed_fds = []
                unmapped = []
                destroyed_pg = False
                pid = os.getpid()

                try:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        dist.destroy_process_group()
                        destroyed_pg = True
                except Exception as _e:
                    log.warning("  prepare_criu_dump: dist teardown error: %s", _e)

                if destroyed_pg:
                    _thr_ok, _thr_alive, _polls = _wait_store_threads_exit(pid)
                    if _thr_ok:
                        log.info("  prepare_criu_dump: store threads "
                                 "exited after %d polls", _polls)
                    else:
                        log.warning("  prepare_criu_dump: store threads "
                                    "still alive: %s", _thr_alive)

                # Complication 15, driver side -- **the TP=1 path**, and the only
                # one that still closes anything.  Only reachable once the group
                # is down, hence the gate: a failed teardown leaves sockets that
                # may still be live, and those must be left alone.  Here the
                # driver *is* the process holding the listeners and its own
                # destroy_process_group above is clean, because there was no
                # abort to make it raise.
                #
                # Measured at TP=1 on 35B: seven ephemeral loopback listeners,
                # all on this process's fds, `pt_tcpstore` and fourteen
                # `pt_gloo_runloop` threads beside them, and **no RAS listener
                # anywhere** -- NCCL does not start its RAS subsystem for a
                # single rank.  So this set is torch's, entirely, and closing it
                # is what keeps criu from rebinding seven ports the restore
                # throws away.  The restore that followed was clean: 0 NCCL
                # warnings and an 8 KB restore-side log.
                #
                # At TP>1 the ranks are separate processes handled in
                # _prepare_worker_dump, which closes nothing, and the driver
                # holds no listeners at all.
                #
                # `_PG_ABORTED` is accepted alongside destroyed_pg, but it is
                # False here by construction: _destroy_nccl is a collective_rpc
                # target, so it sets the flag in the workers and never in this
                # process.  It is listed because the condition should say what
                # makes the close safe, not what happens to be true today.
                closed_listeners = []
                listener_diag = {"destroyed_pg": destroyed_pg,
                                 "pg_aborted": _PG_ABORTED,
                                 "store_threads": _live_store_threads()}
                try:
                    import torch.distributed as _dist
                    listener_diag["dist_initialized"] = bool(
                        _dist.is_initialized())
                except Exception as _e:  # noqa: BLE001
                    listener_diag["dist_initialized"] = f"unreadable: {_e!r}"
                try:
                    if destroyed_pg or _PG_ABORTED:
                        closed_listeners, _skipped = _close_loopback_listeners()
                    else:
                        # Refused, so report the cost of refusing: these are the
                        # ports the image is about to demand back on every
                        # restore.  An empty `closed_listeners` otherwise cannot
                        # be told from a refused one, and the two mean opposite
                        # things.
                        _eph, _fix = _loopback_listeners()
                        listener_diag["would_close"] = [t for _, t in _eph]
                        _skipped = [t for _, t in _fix]
                    # A fixed loopback port left in the image on purpose.  There
                    # are none at TP=1 today; this is the line that would say so
                    # if RAS ever did appear on this path.
                    if _skipped:
                        listener_diag["skipped_fixed_port"] = _skipped
                except Exception as _e:  # noqa: BLE001
                    listener_diag["error"] = repr(_e)
                info["closed_listeners"] = closed_listeners
                info["listener_diag"] = listener_diag
                if closed_listeners:
                    log.info("  prepare_criu_dump: closed %d orphaned "
                             "loopback listener(s), so the image records no "
                             "port to re-bind: %s",
                             len(closed_listeners),
                             ", ".join(closed_listeners))
                else:
                    log.info("  prepare_criu_dump: closed no loopback "
                             "listeners; gate was %s", listener_diag)

                pipe_fd = kwargs.get("pipe_fd", -1)
                # stdout/stderr are the pod-log pipe (redirect_stdio_to_pod_log
                # at startup), which CRIU dumps as an external pipe and the
                # restore hands back with --inherit-fd.

                keep_prefixes = ("/dev/nvidia", "/dev/shm", "anon_inode:",
                                 "socket:", "pipe:")
                for fd_name in sorted(os.listdir(f"/proc/{pid}/fd"),
                                      key=int):
                    try:
                        fd_int = int(fd_name)
                        if fd_int == pipe_fd or fd_int <= 2:
                            continue
                        link = os.readlink(f"/proc/{pid}/fd/{fd_name}")
                        if any(link.startswith(p) for p in keep_prefixes):
                            continue
                        os.close(fd_int)
                        closed_fds.append(fd_int)
                    except (OSError, ValueError):
                        pass

                libc = ctypes.CDLL("libc.so.6")
                with open(f"/proc/{pid}/maps") as f:
                    for line in f:
                        if "io_uring" in line:
                            addr_range = line.split()[0]
                            start_s, end_s = addr_range.split("-")
                            start = int(start_s, 16)
                            length = int(end_s, 16) - start
                            libc.munmap(ctypes.c_void_p(start),
                                        ctypes.c_size_t(length))
                            unmapped.append(f"0x{start:x}")

                # Only this tree's semaphores: under spawn a SemLock keeps its
                # name for its whole life, and another replica's live Instance
                # reopens its queues' semaphores by name in every worker it
                # spawns -- a pod-wide glob here broke those.
                info["removed_sem"] = _unlink_paths(
                    _own_shm_paths(_tree_pids(pid), _SEM_PREFIXES))

                # Reap our own dead children before the tree is frozen.  This
                # process setsid()s at startup, so its pid names both its
                # process group and its session, and every child it ever
                # forked holds a reference on that pid -- zombies included,
                # because a corpse keeps those links.  Any stray left here
                # therefore keeps the leader's id allocated after the dump
                # kills the tree, and the restore cannot place the leader back
                # at it: clone3(set_tid) fails EEXIST on an id that /proc shows
                # as free.  Complements the worker's post-dump sweep, which
                # only reaches strays still alive at dump time; these are
                # already dead, so they are not in the image either way and
                # reaping them only drops references.  WNOHANG throughout, so
                # a live TP worker -- part of the tree being dumped -- is never
                # waited on.
                reaped_strays = []
                while True:
                    try:
                        _stray, _ = os.waitpid(-1, os.WNOHANG)
                    except OSError:
                        break          # ECHILD: nothing left to collect
                    if _stray == 0:
                        break          # children remain, but all still alive
                    reaped_strays.append(_stray)

                remaining_threads = []
                for tid_name in os.listdir(f"/proc/{pid}/task"):
                    try:
                        comm = open(
                            f"/proc/{pid}/task/{tid_name}/comm"
                        ).read().strip()
                        if comm != "python":
                            remaining_threads.append(f"{tid_name}({comm})")
                    except (OSError, ValueError):
                        pass
                info["closed_fds"] = closed_fds
                info["unmapped"] = unmapped
                info["destroyed_pg"] = destroyed_pg
                info["remaining_threads"] = remaining_threads
                info["reaped_strays"] = reaped_strays
                log.info("  prepare_criu_dump: fds=%s, unmapped=%s, "
                         "destroyed_pg=%s, remaining_threads=%s, "
                         "reaped_strays=%s",
                         closed_fds, unmapped, destroyed_pg,
                         remaining_threads, reaped_strays)

            elif cmd == "destroy_nccl":
                if llm is None:
                    raise RuntimeError("destroy_nccl requires init first")
                results = llm.collective_rpc(_destroy_nccl)
                if all(r is None for r in results):
                    log.info("  destroy_nccl: TP1 no-op")
                else:
                    log.info("  NCCL destroyed across %d workers", len(results))

            elif cmd == "reinit_nccl":
                if llm is None:
                    raise RuntimeError("reinit_nccl requires init first")
                from vllm.utils.network_utils import get_open_port
                # The ranks' stdout is the pod log they kept through the dump,
                # but NCCL_DEBUG_FILE overrides it with a FILE* opened at cold
                # start, which the dump closed -- the combination is silent
                # rather than noisy, and cost the 2026-09-19 investigation all
                # six of its instrumented restores. The dump's environ is what
                # the restored ranks carry, so say so here.
                _ndf = os.environ.get("NCCL_DEBUG_FILE")
                if _ndf:
                    log.warning(
                        "  NCCL_DEBUG_FILE=%s was set when this image was "
                        "dumped, so NCCL writes to a FILE* that did not survive "
                        "the restore and its output will NOT appear in the pod "
                        "log. Re-dump without it to see NCCL from the ranks.",
                        _ndf)
                # What CUDA thinks it has, before NCCL asks it for anything.
                # A mismatched restore fails inside ncclCudaHostCalloc and NCCL
                # throws the cudaError away, so take it here while the answer is
                # still attributable to the restore rather than to NCCL.
                try:
                    for _p in llm.collective_rpc(_cuda_restore_probe):
                        log.info("  cuda probe: %s", _p)
                except Exception:
                    log.warning("  cuda probe failed", exc_info=True)
                port = get_open_port()
                results = _collective_rpc_with_timeout(
                    llm, _reinit_nccl, (port,),
                    timeout_s=kwargs.get("timeout_s"), log=log)
                failures = [r for r in results
                            if isinstance(r, dict) and not r.get("ok", True)]
                if failures:
                    # Re-probe on the way out: "unhandled system error" is the
                    # same string whatever went wrong, so the post-failure CUDA
                    # state is what distinguishes a device that vanished from
                    # one that merely cannot pin host memory.
                    try:
                        for _p in llm.collective_rpc(_cuda_restore_probe):
                            log.info("  cuda probe after failure: %s", _p)
                    except Exception:
                        log.warning("  post-failure cuda probe failed",
                                    exc_info=True)
                    raise RuntimeError(f"NCCL reinit failed on workers: {failures}")
                log.info("  NCCL re-initialized on port %d across %d workers",
                         port, len(results))

            elif cmd == "rebind_graphs":
                if llm is None:
                    raise RuntimeError("rebind_graphs requires init/load first")
                results = llm.collective_rpc(_semip_rebind_graphs)
                failures = [r for r in results
                            if not (isinstance(r, dict) and r.get("ok", False))]
                if failures:
                    raise RuntimeError(
                        f"semip rebind_graphs failed: {failures}")
                log.info("  rebind_graphs across %d worker(s)", len(results))
                # The rebind's own verdict, which used to be computed and
                # dropped. Measured 2026-09-22: a TP=2 restore logged the line
                # above at 1.7s and then hung for 302s, so the rebind "succeeds"
                # and something after it wedges -- but with the result discarded
                # there was no way to see what it thought it had done.
                for _r in results:
                    if isinstance(_r, dict):
                        log.info("    rebind rank=%s ok=%s ca_rebind=%s",
                                 _r.get("rank", "?"), _r.get("ok"),
                                 _r.get("ca_rebind"))
                # Verify the rebind before live traffic sees it.  This used to
                # be a warmup, on the premise that "restore drops most
                # cudaGraphExec handles" and they had to be re-instantiated in
                # lockstep across ranks.  That was true of a cold image, where
                # the execs never existed.  A warm image falsifies it: on
                # GLM-5.3 at TP=8 the census read exec_ok=4080 uninstantiated=0
                # identically before and after these rungs, 336 ms apart, on all
                # eight ranks.  Nothing is built here any more.
                #
                # It stays because it is the first collective traffic after the
                # rebind, so it is what catches a bad rebind -- rewritten
                # addresses that read back clean but do not replay -- while the
                # census either side still says whether the image arrived warm.
                if engine is None:
                    engine = llm.llm_engine
                info["compile_cache"] = _compile_cache_fingerprint()
                log.info("  compile cache after restore: %s",
                         info["compile_cache"])
                _log_graph_census(llm, log, "restore:pre-verify")
                _drive_warmup_ladder("verify",
                                     _ladder_plan(_REBIND_VERIFY_SIZES))
                _log_graph_census(llm, log, "restore:post-verify")

            elif cmd == "save_weights":
                if llm is None:
                    raise RuntimeError("save_weights requires attach+stage first")
                results = llm.collective_rpc(
                    _semip_save_weights,
                    args=(kwargs["weights_dir"], kwargs.get("shard_bytes"),
                          kwargs.get("io_workers")))
                total = sum(results)
                info["weights_dir"] = kwargs["weights_dir"]
                info["bytes"] = total
                log.info("  saved weights: %.2f GiB across %d worker(s)",
                         total / 2**30, len(results))

            elif cmd == "load_weights":
                if llm is None:
                    raise RuntimeError("load_weights requires attach first")
                results = llm.collective_rpc(
                    _semip_load_weights,
                    args=(kwargs["weights_dir"], kwargs.get("io_workers")))
                total = sum(results)
                info["bytes"] = total
                log.info("  loaded weights: %.2f GiB across %d worker(s)",
                         total / 2**30, len(results))

            else:
                error = f"unknown command: {cmd}"

        except Exception as e:
            import traceback
            traceback.print_exc()
            error = f"{type(e).__name__}: {e}"

        return error, info

    # -- Main loop --------------------------------------------------------------

    # The current in-flight command, captured outside the per-iteration
    # scope so the fatal-error reporter below can blame the right cmd.
    cmd = None

    try:
        while True:
            if engine is None and llm is not None:
                engine = llm.llm_engine

            has_active = (engine is not None
                          and engine.has_unfinished_requests()
                          and not _paused)

            if has_active:
                _process_step_outputs(engine.step())
                if not pipe_conn.poll(0):
                    continue

            if _deferred_cmds:
                cmd, kwargs = _deferred_cmds.pop(0)
            else:
                try:
                    cmd, kwargs = pipe_conn.recv()
                except EOFError:
                    break

            if cmd == "exit":
                _drain_engine()
                log.info("exit")
                pipe_conn.send("exit_ack")
                break

            log.info(">>> %s", cmd)

            if cmd == "generate":
                req_id = kwargs.get("req_id")
                if req_id is None:
                    req_id = f"auto-{_next_engine_id}"
                try:
                    _submit_generate(req_id, kwargs["prompts"],
                                     kwargs["sampling_params"])
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    pipe_conn.send(("generate_done", 0.0,
                                    f"{type(e).__name__}: {e}",
                                    {"req_id": req_id}))
                # Drain any additional generate commands already on the pipe
                # so they get added to the engine before the first step().
                while pipe_conn.poll(0):
                    try:
                        cmd2, kwargs2 = pipe_conn.recv()
                    except EOFError:
                        break
                    if cmd2 == "generate":
                        rid2 = kwargs2.get("req_id",
                                           f"auto-{_next_engine_id}")
                        try:
                            _submit_generate(rid2, kwargs2["prompts"],
                                             kwargs2["sampling_params"])
                        except Exception as e2:
                            import traceback
                            traceback.print_exc()
                            pipe_conn.send(("generate_done", 0.0,
                                            f"{type(e2).__name__}: {e2}",
                                            {"req_id": rid2}))
                    else:
                        log.info(">>> %s (deferred)", cmd2)
                        _deferred_cmds.append((cmd2, kwargs2))
                continue

            t0 = time.perf_counter()
            error, info = _handle_command(cmd, kwargs)
            elapsed = time.perf_counter() - t0
            status = "OK" if error is None else "FAILED"
            log.info("<<< %s %s (%.3fs)", cmd, status, elapsed)
            pipe_conn.send((cmd, elapsed, error, info))

    except BaseException as _fatal:
        # Last-resort reporter: any unhandled exception in the main loop
        # (including KeyboardInterrupt, SystemExit) gets a final error
        # frame on the pipe so the worker can attribute the failure to a
        # specific cmd instead of just seeing "child pipe broken".  Both
        # the traceback and the offending cmd are logged to the per-rank
        # log file via log so the post-mortem survives a CRIU restore.
        import traceback as _tb
        _trace = _tb.format_exc()
        log.error("FATAL in main loop (cmd=%s): %s: %s",
                  cmd, type(_fatal).__name__, _fatal)
        log.error("%s", _trace)
        try:
            pipe_conn.send((
                cmd if cmd is not None else "__fatal__",
                0.0,
                f"FATAL {type(_fatal).__name__}: {_fatal}",
                {"traceback": _trace, "cmd": cmd},
            ))
        except Exception:
            pass
        raise
