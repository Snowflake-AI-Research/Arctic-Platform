"""Worker process for a single Instance.

Each worker is spawned from the main process.  On "init", it spawns the
vLLM child via mp.get_context("spawn").  Checkpoint/restore run in the
worker via CUDA driver ctypes.  CRIU save/load enables dumping the
child process tree to disk and restoring it with new PIDs.

Command protocol:  (cmd, kwargs)
Result protocol:   (cmd, elapsed, error, info)
"""
import glob, json, os, shutil, signal, stat, subprocess, sys, tempfile, time, ctypes, threading, queue, struct

import pynvml
import torch.multiprocessing as mp

import semip_logging


def _unprivileged():
    """Single switch for running dump and restore on a non-privileged pod
    that grants only CAP_CHECKPOINT_RESTORE + CAP_SYS_PTRACE (no
    CAP_SYS_ADMIN).  SEMIP_UNPRIVILEGED=1 changes exactly three things:

    1. ``--unprivileged`` is added to criu's argv, on dump and restore
       alike, so it skips the netns kerndat probe that needs CAP_SYS_ADMIN.
    2. The spawned child sheds its capabilities before importing torch
       (``_SEMIP_CHILD_DROP_CAPS``), so every task in the image records an
       empty cap set.  This one is NOT a runtime difference -- it changes
       what is written to disk, which is why an image's capability level is
       fixed at dump time and the two configurations are not
       interchangeable after the fact.
    3. ``criu_restore`` takes _worker_criu_load_lowcap instead of
       _worker_criu_load, dropping the private PID namespace -- and with it
       the ``sudo``, the reaper, and the guarantee that recorded task ids
       are free.  Hence the collision preflight on that path.

    See the "Reduced-capability restore" section below for the capability
    rationale, and Complication 11 in skills/CRIU_PLUMBING.md.
    """
    return os.environ.get("SEMIP_UNPRIVILEGED") == "1"


# ---------------------------------------------------------------------------
# Dead-child diagnostics
# ---------------------------------------------------------------------------


def _diagnose_dead_child(child_pid):
    """Return a one-line forensic summary of a (probably) dead child.

    Distinguishes SIGKILL (signal 9 -- usually kernel OOM-killer),
    SIGSEGV (11 -- typically a CUDA / native crash), SIGABRT (6 --
    Python ``assert`` or C ``abort``) and clean exits, and reports the
    process state ("Z" zombie, "R" running, etc.) plus VmRSS at death.
    Best-effort: any failure to read ``/proc`` or reap the zombie is
    swallowed and reported as ``unknown``.
    """
    parts = []

    # Process state from /proc/<pid>/status -- captures whether the
    # child is a zombie awaiting reap, frozen, etc.  May not exist if
    # the kernel already cleaned it up by the time we read.
    state = "unknown"
    rss_kib = None
    try:
        with open(f"/proc/{child_pid}/status", "r") as _sf:
            for _line in _sf:
                if _line.startswith("State:"):
                    # Format: "State:\tZ (zombie)"
                    state = _line.split(":", 1)[1].strip()
                elif _line.startswith("VmRSS:"):
                    rss_kib = int(_line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    parts.append(f"state={state}")
    if rss_kib is not None:
        parts.append(f"rss={rss_kib / 1024:.1f}MiB")

    # Reap the zombie (if any) and decode the exit reason.  WNOHANG
    # so we never block when the child is somehow still running (e.g.
    # frozen via cuda-checkpoint -- in which case we won't reap and
    # just record that fact).
    exit_desc = "no_reap"
    try:
        _wpid, _wstatus = os.waitpid(child_pid, os.WNOHANG)
        if _wpid == 0:
            exit_desc = "running_or_frozen"
        elif os.WIFSIGNALED(_wstatus):
            _sig = os.WTERMSIG(_wstatus)
            try:
                _name = signal.Signals(_sig).name
            except ValueError:
                _name = f"sig{_sig}"
            _hint = ""
            if _sig == signal.SIGKILL:
                _hint = " (likely kernel OOM-killer)"
            elif _sig == signal.SIGSEGV:
                _hint = " (native/CUDA crash)"
            elif _sig == signal.SIGABRT:
                _hint = " (assert/abort)"
            exit_desc = f"killed_by={_name}({_sig}){_hint}"
        elif os.WIFEXITED(_wstatus):
            exit_desc = f"exited_code={os.WEXITSTATUS(_wstatus)}"
        else:
            exit_desc = f"raw_status=0x{_wstatus:x}"
    except ChildProcessError:
        exit_desc = "already_reaped"
    except OSError as _e:
        exit_desc = f"waitpid_err={type(_e).__name__}"
    parts.append(exit_desc)

    return ", ".join(parts)

# ---------------------------------------------------------------------------
# CUDA driver API bindings (checkpoint / restore)
# ---------------------------------------------------------------------------

_CU_CHECKPOINT_ALREADY_DONE = 401

_cu_bindings = None
_cu_bindings_pid = None
_cu_lock = threading.Lock()


def _check_cu(name, ret, *, ignore=None):
    if ret != 0 and ret != ignore:
        raise RuntimeError(f"{name} failed with CUresult={ret}")


def _get_cu():
    """Return CUDA driver bindings, lazily initializing per-process."""
    global _cu_bindings, _cu_bindings_pid
    with _cu_lock:
        if _cu_bindings_pid != os.getpid():
            lib = ctypes.CDLL("libcuda.so")

            lib.cuInit.argtypes = [ctypes.c_uint]
            lib.cuInit.restype = ctypes.c_int

            lib.cuCheckpointProcessLock.argtypes = [ctypes.c_int, ctypes.c_void_p]
            lib.cuCheckpointProcessLock.restype = ctypes.c_int

            lib.cuCheckpointProcessCheckpoint.argtypes = [ctypes.c_int, ctypes.c_void_p]
            lib.cuCheckpointProcessCheckpoint.restype = ctypes.c_int

            lib.cuCheckpointProcessUnlock.argtypes = [ctypes.c_int, ctypes.c_void_p]
            lib.cuCheckpointProcessUnlock.restype = ctypes.c_int

            lib.cuCheckpointProcessRestore.argtypes = [ctypes.c_int, ctypes.c_void_p]
            lib.cuCheckpointProcessRestore.restype = ctypes.c_int

            _check_cu("cuInit", lib.cuInit(0))
            _cu_bindings = lib
            _cu_bindings_pid = os.getpid()
    return _cu_bindings


def _get_descendant_pids(pid):
    """Return PIDs of all descendant processes, leaves first (bottom-up)."""
    import psutil
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return []
    children = proc.children(recursive=True)
    filtered = []
    for c in children:
        try:
            if "resource_tracker" in " ".join(c.cmdline()):
                continue
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
            pass
        filtered.append(c.pid)
    filtered.reverse()
    return filtered


_TCP_STATES = {"01": "ESTABLISHED", "06": "TIME_WAIT", "08": "CLOSE_WAIT",
               "0A": "LISTEN"}


def _inet_census(pids):
    """Every TCP socket in *pids* whose local address is not loopback.

    Loopback exists identically in every pod; anything else names the dump
    pod's address, which a restore elsewhere cannot re-bind.
    """
    import socket as _sock
    import struct as _struct

    def _addr(hexaddr):
        ip_hex, port_hex = hexaddr.split(":")
        raw = bytes.fromhex(ip_hex)
        if len(raw) == 4:
            ip = _sock.inet_ntop(_sock.AF_INET, _struct.pack("<I", int(ip_hex, 16)))
        else:
            words = _struct.unpack("<4I", raw)
            ip = _sock.inet_ntop(_sock.AF_INET6, _struct.pack(">4I", *words))
        return ip, int(port_hex, 16)

    by_inode = {}
    for table in ("tcp", "tcp6"):
        try:
            with open(f"/proc/net/{table}") as f:
                next(f)
                for line in f:
                    parts = line.split()
                    by_inode[parts[9]] = (_addr(parts[1]), _addr(parts[2]),
                                          _TCP_STATES.get(parts[3], parts[3]))
        except OSError:
            continue
    rows = []
    for pid in pids:
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                link = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if not link.startswith("socket:["):
                continue
            entry = by_inode.get(link[8:-1])
            if entry is None:
                continue
            (lip, lport), (rip, rport), state = entry
            if lip.startswith("127.") or lip in ("::1", "::ffff:127.0.0.1"):
                continue
            rows.append({"pid": pid, "fd": int(fd), "state": state,
                         "local": f"{lip}:{lport}", "remote": f"{rip}:{rport}"})
    return rows


def _kill_process_tree(pid):
    """SIGKILL a process and all its descendants (leaves first)."""
    import signal as _sig
    for desc_pid in _get_descendant_pids(pid):
        try:
            os.kill(desc_pid, _sig.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.kill(pid, _sig.SIGKILL)
    except ProcessLookupError:
        pass


_is_root = os.geteuid() == 0

# The dump invokes criu directly rather than through sudo (see
# _worker_criu_save), so resolve the binary here instead of relying on sudo's
# secure_path to cover /usr/sbin.
_CRIU_BIN = shutil.which("criu") or "/usr/sbin/criu"


def _worker_checkpoint(child_pid, unlock=True):
    """Checkpoint the vLLM child and all its GPU-holding descendants.

    If unlock=False, leaves processes in CUDA 'checkpointed' state for a
    subsequent CRIU dump, which is what lets that dump find no GPU state to
    handle: _worker_criu_save points --libdir at an empty directory, so no
    CRIU plugin (the CUDA one included) is loaded at dump time at all, and
    prepare_criu_dump has already closed every /dev/nvidia* fd.  The restore
    is the asymmetric half -- it passes no --libdir and so does load
    /usr/lib/criu/cuda_plugin.so.

    Always uses the CUDA driver API, including from an unprivileged caller:
    cuCheckpointProcess* needs ptrace permission over the target, which a
    parent has over its own same-uid child, not root.
    """
    descendant_pids = _get_descendant_pids(child_pid)
    all_pids = descendant_pids + [child_pid]
    cu = _get_cu()
    for pid in all_pids:
        _check_cu(f"Lock({pid})", cu.cuCheckpointProcessLock(pid, None))
        _check_cu(f"Checkpoint({pid})", cu.cuCheckpointProcessCheckpoint(pid, None))
        if unlock:
            _check_cu(f"Unlock({pid})", cu.cuCheckpointProcessUnlock(pid, None),
                      ignore=_CU_CHECKPOINT_ALREADY_DONE)
    return all_pids


def _gpu_uuids():
    """Return list of GPU UUID strings (e.g. 'GPU-<uuid>') via NVML."""
    pynvml.nvmlInit()
    uuids = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        u = pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(i))
        if isinstance(u, bytes):
            u = u.decode()
        uuids.append(u)
    return uuids


_DEVICE_GLOBS = ("/dev/nvidia[0-9]*", "/dev/infiniband/uverbs[0-9]*")


def _visible_device_nodes():
    """Sorted basenames of the GPU and EFA device nodes present in this pod.

    Deliberately the device *nodes* rather than PCI addresses or UUIDs: the
    constraint is whether the path a captured context refers to can be opened
    again, and on an unprivileged, scheduled pod only the pod's own allocation
    appears under /dev at all. The other NICs are still visible in /sys, which
    is why /sys is no use for this.
    """
    names = set()
    for pattern in _DEVICE_GLOBS:
        for path in glob.glob(pattern):
            names.add(os.path.basename(path))
    return sorted(names)


def _gpu_migration_permutation(old_gpus, new_gpus, n):
    """Full ``n``-GPU bijection that pins ``old_gpus[k] -> new_gpus[k]``.

    ``cuCheckpointProcessRestore`` needs a bijection over every visible GPU.
    Only ``old_gpus`` hold this process's allocations; the remaining GPUs are
    paired in order to complete the permutation (works even if the sets
    overlap, e.g. ``[4,5,6,7] -> [0,1,4,5]``).
    """
    perm = dict(zip(old_gpus, new_gpus))
    spare = iter(g for g in range(n) if g not in set(new_gpus))
    return [perm[i] if i in perm else next(spare) for i in range(n)]


def _placement_changed(old_gpus, new_gpus, old_uuids, new_uuids):
    """Whether this restore needs a device map, i.e. the placement moved.

    Two ways it can move, and both need the map:

    * **Different indices**, on this node or any other.  The original case.
    * **The same indices on a different node.**  The checkpoint bakes the
      capture node's GPU UUIDs, which do not exist here, and only the map
      rewrites them -- without it ``cuCheckpointProcessRestore`` fails with
      ``CUDA_ERROR_INVALID_VALUE``.

    The second case used to be unreachable, which is why the index comparison
    alone was enough: an image lived and died in the pod that dumped it, so
    "same indices" implied "same node".  A distributed image cache breaks that
    implication -- every node mounts the cache at one path, so an image
    restores under its own ``model_dir`` anywhere, and Ray is as likely to
    assign the dumped index as any other.  On an 8-GPU node that was a
    one-in-eight hard failure.

    ``old_uuids`` missing means an image dumped before they were recorded; such
    an image can only be restored on its capture node, and there is nothing
    here that can detect that, so it reports no change and leaves the old
    behaviour exactly as it was.
    """
    if not old_gpus or not new_gpus:
        return False
    if list(old_gpus) != list(new_gpus):
        return True
    return bool(old_uuids) and list(old_uuids) != list(new_uuids)


def _build_restore_args(old_gpus, new_gpus, old_uuids=None, new_uuids=None):
    """Build a CUcheckpointRestoreArgs ctypes buffer for GPU migration.

    Layout (64-bit): gpuPairs* (8), gpuPairsCount (4), reserved (52 zeroed).
    Each CUcheckpointGpuPair is oldUuid(16) + newUuid(16).  ``old_uuids`` are
    the capture node's GPU UUIDs (from meta.json); the "new" side is local, and
    is read here unless the caller already has it.
    """
    if not old_uuids:
        raise ValueError(
            "image is missing 'gpu_uuids' in meta.json; recapture with the "
            "updated worker so the capture node's GPU UUIDs are recorded")

    def _to_bytes(us):
        return [bytes.fromhex(u.replace("GPU-", "").replace("-", ""))
                for u in us]

    new_uuids = list(new_uuids) if new_uuids else _gpu_uuids()
    n = len(new_uuids)
    if len(old_uuids) != n:
        # Named here rather than as the IndexError the pairing loop would
        # raise: the map is a bijection over every visible GPU, so a capture
        # node with a different GPU count cannot be mapped onto this one.
        raise ValueError(
            f"image recorded {len(old_uuids)} GPU UUID(s) but this node has "
            f"{n}; the cuda-checkpoint device map is a bijection over every "
            f"visible GPU, so the counts must match")
    perm = _gpu_migration_permutation(old_gpus, new_gpus, n)
    old_bytes = _to_bytes(old_uuids)
    new_bytes = _to_bytes(new_uuids)

    pairs_data = bytearray()
    for i in range(n):
        pairs_data += old_bytes[i] + new_bytes[perm[i]]

    pairs_buf = (ctypes.c_char * len(pairs_data))(*pairs_data)
    pairs_ptr = ctypes.cast(pairs_buf, ctypes.c_void_p)
    args_data = bytearray(64)
    struct.pack_into("<Q", args_data, 0, pairs_ptr.value)
    struct.pack_into("<I", args_data, 8, n)
    args_buf = (ctypes.c_char * 64)(*args_data)
    return args_buf, pairs_buf


_SETTLE_TIMEOUT_ENV = "SEMIP_SETTLE_TIMEOUT_S"
_SETTLE_TIMEOUT_DEFAULT_S = 30.0
_SETTLE_POLL_S = 0.05
_REINIT_TIMEOUT_ENV = "SEMIP_REINIT_NCCL_TIMEOUT_S"
_REINIT_TIMEOUT_DEFAULT_S = 300.0


def _env_seconds(name, default):
    """A positive float from the environment, or ``default``.

    Unusable values fall back rather than raising: these are diagnostics
    budgets, and a typo in one should not be what takes a restore down.
    """
    try:
        value = float(os.environ.get(name) or "")
    except ValueError:
        return default
    return value if value > 0 else default


def _reinit_timeout_s():
    """How long ``reinit_nccl`` may run in the child before it is called hung."""
    return _env_seconds(_REINIT_TIMEOUT_ENV, _REINIT_TIMEOUT_DEFAULT_S)


def _settle_timeout_s():
    """How long to wait for a CRIU-restored tree to quiesce, in seconds.

    Tunable because the right value is not known. What is known is that a
    healthy tree settles well inside a second, and that a tree which is going
    to hang blows any budget we have tried. This process is the Ray actor, not
    a restored one, so it has a live ``environ`` and ``extra_env`` reaches it
    normally -- which means retuning this costs a job, not an image rebuild.

    The effective number is named in the failure message this feeds, so a typo
    shows up there rather than taking the restore down with it.
    """
    return _env_seconds(_SETTLE_TIMEOUT_ENV, _SETTLE_TIMEOUT_DEFAULT_S)


def _proc_state(pid):
    """The one-line ``State:`` field of ``/proc/<pid>/status``, or None if gone."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("State:"):
                    return line.split("\t", 1)[1].strip()
    except OSError:
        return None
    return None


def _proc_wchan(pid):
    """``/proc/<pid>/wchan``: the kernel symbol the task sleeps in, else None.

    ``0`` means it is not blocked on one.
    """
    try:
        with open(f"/proc/{pid}/wchan") as f:
            return f.read().strip()
    except OSError:
        return None


def _proc_syscall(pid):
    """First field of ``/proc/<pid>/syscall``: a syscall number, or ``running``.

    ``running`` means the task is executing userspace instructions rather than
    sitting in a syscall.
    """
    try:
        with open(f"/proc/{pid}/syscall") as f:
            return f.read().split(maxsplit=1)[0]
    except (OSError, IndexError):
        return None


def _is_userspace_spin(pid):
    """Whether ``pid`` is burning CPU in userspace rather than working.

    ``wchan == 0`` says the task is not blocked on a kernel symbol and
    ``syscall == running`` says it is not inside a syscall, so it is executing
    userspace instructions and nothing else.

    Why this is safe to treat as settled, which is the whole point of the
    quiescence wait: the wait exists so that ``cuCheckpointProcessRestore`` is
    not called on a task in the middle of a kernel or driver operation. At this
    point in the sequence the ranks have been CRIU-restored and have *no* CUDA
    context yet -- restoring it is the next phase -- so a rank spinning here
    cannot be doing GPU work. It can only be spinning on the CPU.

    And it is: vLLM's ``SpinCondition.wait`` in ``shm_broadcast.py`` compares
    ``time.monotonic()`` against ``self.last_read + busy_loop_s`` and calls
    ``sched_yield()`` while that holds. A restore cannot preserve
    ``CLOCK_MONOTONIC`` here -- that needs a time namespace, which needs
    ``CAP_SYS_ADMIN``, and the pod runs with ``CapEff: 0000000000000000`` under
    ``SEMIP_UNPRIVILEGED=1`` -- so ``last_read`` still holds a reading from the
    *dump* host's clock. When the restore host booted later than the dump host
    by more than the dump-to-restore gap, the comparison stays true and the
    yield loop never ends. Measured 2026-09-22 across 29 restores: that
    inequality predicted quiescence failure every time, 29 for 29.

    Conservative on error. If either file cannot be read the answer is False,
    which keeps the old behaviour of failing the restore.
    """
    return _proc_wchan(pid) == "0" and _proc_syscall(pid) == "running"


def _wait_for_quiescence(pids, timeout_s=None):
    """Wait for every pid to reach 'S' (sleeping) or 'T' (stopped).

    Returns ``(unsettled_pids, states)``. A pid that disappears counts as
    settled -- there is nothing left to restore for it.

    Polled against a *single* deadline for the whole set rather than per pid.
    The per-pid form this replaced cost ``timeout x len(pids)`` in the bad
    case, so a TP=8 tree spent 45 s discovering a failure that a TP=2 tree
    reported in 10 s -- and the elapsed time of this phase, rather than any
    error, was the only way to tell the failure had happened at all.
    """
    if timeout_s is None:
        timeout_s = _settle_timeout_s()
    deadline = time.monotonic() + timeout_s
    pending = list(pids)
    states = {}
    while True:
        still = []
        for pid in pending:
            state = _proc_state(pid)
            if state is None:
                continue        # exited: nothing to restore
            states[pid] = state
            if state.startswith("S") or state.startswith("T"):
                continue
            still.append(pid)
        pending = still
        if not pending or time.monotonic() >= deadline:
            return pending, states
        time.sleep(_SETTLE_POLL_S)


def _worker_restore(pids, old_gpus=None, new_gpus=None, old_uuids=None):
    """Restore CUDA context on pids.

    State machine: checkpointed → restore → locked → unlock → running

    Always uses the CUDA driver API, including from an unprivileged caller,
    for the same reason as _worker_checkpoint.

    ``old_gpus`` / ``new_gpus`` are the capture and restore GPU lists.  When
    the placement moved -- different indices, *or* the same indices on another
    node -- a device-map bijection remaps ``old_gpus[k] -> new_gpus[k]`` (TP1
    and TP>1).  ``old_uuids`` (from meta.json) are the "old" side, so the
    capture and restore nodes need not be the same machine.  See
    ``_placement_changed``.

    Two-pass: restore ALL pids, THEN unlock ALL pids.  With a TP process tree
    an unlocked worker could otherwise touch a still-locked sibling (NCCL/IPC)
    and race; restoring the whole tree before unlocking any avoids that.
    """
    # NVML is only read when the indices match and the image recorded UUIDs to
    # compare against -- the one case where the index test cannot answer the
    # question.  Every other path costs exactly what it did before.
    new_uuids = None
    if old_gpus and new_gpus and old_uuids and list(old_gpus) == list(new_gpus):
        new_uuids = _gpu_uuids()
    migrate = _placement_changed(old_gpus, new_gpus, old_uuids, new_uuids)
    ordered = list(reversed(pids))
    cu = _get_cu()
    args_ptr = None
    _kept_alive = None
    if migrate:
        args_buf, pairs_buf = _build_restore_args(
            old_gpus, new_gpus, old_uuids, new_uuids=new_uuids)
        args_ptr = ctypes.cast(args_buf, ctypes.c_void_p)
        _kept_alive = (args_buf, pairs_buf)
    for pid in ordered:
        _check_cu(f"Restore({pid})",
                  cu.cuCheckpointProcessRestore(pid, args_ptr),
                  ignore=_CU_CHECKPOINT_ALREADY_DONE)
    for pid in ordered:
        _check_cu(f"Unlock({pid})",
                  cu.cuCheckpointProcessUnlock(pid, None),
                  ignore=_CU_CHECKPOINT_ALREADY_DONE)


# ---------------------------------------------------------------------------
# CRIU save / load (process image to/from disk)
# ---------------------------------------------------------------------------

def _resolve_fd_resource(pid, fd):
    """Read /proc/<pid>/fd/<fd> to determine the CRIU resource identifier."""
    link = os.readlink(f"/proc/{pid}/fd/{fd}")
    return link


def _stdout_inherit(meta):
    """``(fd, argv)`` that hand this pod's log back to a restored tree.

    The dumped tree's fd 1/2 are a pipe whose reader was the dumping pod's
    container runtime, which CRIU cannot recreate. Restored without
    ``--inherit-fd`` it comes back as a fresh pipe with no reader, and the first
    write fails with ``EPIPE``; so the restore refuses rather than produce a
    tree that dies the first time it logs. ``fd`` is a write end on this pod's
    log for the caller to pass into the criu helper and close afterwards;
    ``argv`` names it with the ``@LOGFD@`` placeholder the helper fills in.

    A tree dumped outside a pod records ``/dev/null``, which CRIU reopens by
    path and needs neither.
    """
    resource = meta.get("stdout_resource")
    if not resource:
        raise RuntimeError(
            "image records no stdout_resource, so it predates the pod-log "
            "stdout and its fd 1/2 name a /tmp/inst<N>.log that nothing "
            "creates any more; re-dump it")
    if not resource.startswith("pipe:"):
        return None, []
    fd = semip_logging.open_pod_log()
    if not stat.S_ISFIFO(os.fstat(fd).st_mode):
        os.close(fd)
        raise RuntimeError(
            f"image's stdout is the pipe {resource}, but "
            f"{semip_logging.pod_log_target()} is not a writable pipe here; "
            f"restoring would leave the tree writing into a pipe with no "
            f"reader")
    return fd, ["--inherit-fd", f"fd[@LOGFD@]:{resource}"]


# Helper-side fd plumbing shared by both restore paths. The helper receives the
# command pipe and (optionally) the pod log over SCM_RIGHTS, puts the pipe at the
# number criu's argv names, moves the log above it, and fills @LOGFD@ in. F_DUPFD
# runs first so neither dup2 can clobber the other fd.
def _helper_fd_prologue(sock_path, new_pipe_fd):
    return (
        "import os, socket, array, fcntl\n"
        "s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        f"s.connect({sock_path!r})\n"
        "msg, ancdata, _, _ = s.recvmsg(1, socket.CMSG_SPACE(8))\n"
        "fds = array.array('i')\n"
        "for cl, ct, cd in ancdata:\n"
        "    if cl == socket.SOL_SOCKET and ct == socket.SCM_RIGHTS:\n"
        "        fds.frombytes(cd[:len(cd) - len(cd) % fds.itemsize])\n"
        "s.close()\n"
        "LFD = None\n"
        "if len(fds) > 1:\n"
        f"    LFD = fcntl.fcntl(fds[1], fcntl.F_DUPFD, {new_pipe_fd + 1})\n"
        "    os.close(fds[1])\n"
        f"os.dup2(fds[0], {new_pipe_fd})\n"
        f"if fds[0] != {new_pipe_fd}: os.close(fds[0])\n"
    )


def _helper_argv_expr(criu_argv):
    """Python source for criu's argv with ``@LOGFD@`` filled from ``LFD``."""
    return f"[a.replace('@LOGFD@', str(LFD)) for a in {criu_argv!r}]"


def _worker_criu_save(child_pid, image_dir, pipe_fd, pipe_resource, gpus,
                      meta_extra=None):
    """Dump the vLLM child process tree to disk via CRIU (destructive).

    ``gpus`` is the physical GPU list the tree currently occupies; it is
    recorded in meta.json (with the capture node's GPU UUIDs) so a later
    restore can build the migration device map.

    The child process is killed after a successful dump.  The on-disk
    image is later restored via criu_restore().
    """
    # Clear any stale dump in the target directory so leftover files from a
    # prior criu_dump() (which CRIU may not overwrite) cannot corrupt the new
    # image.  Files owned by another uid (an earlier root-run capture of the
    # same model) are not ours to remove: dump and restore must share one uid,
    # which Instance.criu_restore already enforces, so name the collision
    # rather than trying to elevate past it.
    if os.path.exists(image_dir):
        try:
            shutil.rmtree(image_dir)
        except PermissionError as e:
            raise RuntimeError(
                f"cannot clear stale image at {image_dir}: {e}. It holds files "
                f"from a run under a different uid; dump and restore must share "
                f"one uid, so remove it by hand as that user and re-dump") from e
    os.makedirs(image_dir, exist_ok=True)

    # At TP>1 the child spawns worker subprocesses whose Unix-domain IPC
    # sockets (multiproc-executor rpc / shm-broadcast) must be declared
    # --external to CRIU, so scan every descendant PID (deduped by inode), not
    # just the child.  nvidia fds are only recorded off the child.
    external_unix = []
    nvidia_fds = {}
    seen_inodes = set()
    # What the tree's stdout is: the pod-log pipe (see
    # semip_logging.redirect_stdio_to_pod_log), whose reader is outside the
    # tree, so CRIU records it as an external pipe and the restore must hand a
    # live one back with --inherit-fd under this name.
    stdout_resource = _resolve_fd_resource(child_pid, 1)
    stray_stdio = []
    for scan_pid in _get_descendant_pids(child_pid) + [child_pid]:
        proc_fd_dir = f"/proc/{scan_pid}/fd"
        try:
            fd_names = os.listdir(proc_fd_dir)
        except OSError:
            continue
        for fd_name in fd_names:
            try:
                link = os.readlink(f"{proc_fd_dir}/{fd_name}")
                if fd_name in ("1", "2"):
                    if link != stdout_resource:
                        stray_stdio.append(f"{scan_pid}:fd{fd_name}={link}")
                    continue
                if link.startswith("socket:["):
                    ino = link.split("[")[1].rstrip("]")
                    if ino not in seen_inodes:
                        seen_inodes.add(ino)
                        external_unix.append(f"unix[{ino}]")
                elif scan_pid == child_pid and "/dev/nvidia" in link:
                    nvidia_fds[int(fd_name)] = link
            except OSError:
                pass
    if stray_stdio:
        # Only stdout_resource is handed back at restore; another external
        # target on fd 1/2 would come back as a pipe with no reader, and the
        # first write to it fails with EPIPE.
        print(f"[semip] WARNING: stdio outside {stdout_resource} in the "
              f"dumped tree: {', '.join(stray_stdio)}", flush=True)

    # criu runs at this worker's own uid, never through sudo: it reads every
    # rlimit of its target with prlimit(), and check_prlimit_permission() grants
    # a cross-uid read only to CAP_SYS_RESOURCE -- outside a cap-prod pod's
    # bounding set, so not even sudo can acquire it.  A root criu against an
    # unprivileged child therefore dies at cr-dump.c:389 ("Can't get rlimit 0:
    # Operation not permitted") before writing any image.  The target is this
    # worker's own child, so matching its uid just means not elevating: a root
    # worker is already root, and an unprivileged one takes criu's capabilities
    # from the binary instead --
    #   sudo setcap cap_sys_ptrace,cap_checkpoint_restore,cap_setpcap,cap_setgid+eip \
    #       /usr/sbin/criu
    # The dump needs only the first two; cap_setpcap and cap_setgid are for the
    # restore's restore_creds() (see _worker_criu_load_lowcap).  Keep the +e:
    # without it a real-uid-0 execve lands with an empty effective set, which
    # would break a root worker.
    #
    # criu's --libdir is its plugin directory (default /usr/lib/criu, which
    # holds cuda_plugin.so).  Point the dump at an EMPTY one so no plugin
    # loads: cuda_checkpoint() has already driven cuCheckpointProcess* through
    # the driver API, and prepare_criu_dump closed every /dev/nvidia* fd, so
    # there is nothing for the CUDA plugin to do here.  The restore, which does
    # want it, passes no --libdir and gets the real one.  The whole contract is
    # "exists, holds no .so", so make a private one per dump rather than a fixed
    # path under /usr/lib that only root can create.  mkdtemp is fresh (so
    # guaranteed empty -- a stray .so in a long-lived directory would be loaded
    # by a binary carrying file capabilities), mode 0700 from creation, and
    # race-free for the concurrent dumps a cold start issues.
    empty_libdir = tempfile.mkdtemp(prefix="semip-criu-noplugins-")
    cmd = [
        _CRIU_BIN, "dump",
        "-t", str(child_pid),
        "-D", image_dir,
        "-o", "dump.log",
        # No --shell-job: the child detaches from the controlling terminal
        # at startup (fd 0 -> /dev/null + setsid), so no tty enters the
        # image.  A tty would be unreattachable inside the private PID
        # namespace the restore path uses.
        "--tcp-close",
        "--ext-unix-sk",
        "--link-remap",
        "--libdir", empty_libdir,
        "-v4",
    ]
    # SEMIP_UNPRIVILEGED=1: skip CRIU's network-namespace kerndat probe (which
    # needs CAP_SYS_ADMIN), so the dump runs on a non-privileged pod with only
    # CAP_CHECKPOINT_RESTORE + CAP_SYS_PTRACE.  Pairs with the lowcap restore
    # path selected by the same flag; see the "Reduced-capability restore"
    # section below.
    if _unprivileged():
        cmd.append("--unprivileged")
    for ext in external_unix:
        cmd.extend(["--external", ext])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        shutil.rmtree(empty_libdir, ignore_errors=True)
    if result.returncode != 0:
        # Both streams: criu's pre-log refusals (check_caps among them) go to
        # stdout via pr_msg, while stderr carries only its run id.
        detail = "\n".join(
            s.strip() for s in (result.stderr, result.stdout)
            if s and s.strip()) or "(no output)"
        log_path = os.path.join(image_dir, "dump.log")
        if os.path.exists(log_path):
            with open(log_path) as f:
                detail += "\n--- dump.log ---\n" + f.read()[-2000:]
        raise RuntimeError(
            f"criu dump failed (rc={result.returncode}): {detail}")

    gpus = [gpus] if isinstance(gpus, int) else list(gpus)
    meta = {
        "child_pid": child_pid,
        "pipe_fd": pipe_fd,
        "pipe_resource": pipe_resource,
        "stdout_resource": stdout_resource,
        "nvidia_fds": {str(k): v for k, v in nvidia_fds.items()},
        "rank": gpus[0] if gpus else 0,
        "gpus": gpus,
        # Physical GPU UUIDs of the *capture* node.  Required for
        # cross-node CUDA restore: cuCheckpointProcessRestore's device
        # map must use the UUID baked into the checkpoint (capture node)
        # as the "old" side; the restore node reads its own UUIDs for
        # the "new" side.  Without this, cross-node restore fails with
        # CUDA_ERROR_INVALID_VALUE (CUresult=1).
        "gpu_uuids": _gpu_uuids(),
        # The device nodes that existed in this pod when the image was taken.
        # A TP>1 image only restores into a pod whose /dev holds every device
        # its captured state references, and on the scheduled path a pod is
        # given only its allocated slice of /dev -- so an image dumped on
        # GPUs 2,3 is unrestorable in a pod holding 0,1. Recording the set is
        # what lets the restore say so instead of failing inside NCCL with
        # "unhandled system error".  See _check_device_visibility.
        "device_nodes": _visible_device_nodes(),
        # Identity of the dumped tree.  The restored child keeps it, so the
        # image is only restorable by the same user; Instance.criu_restore
        # rejects the mix up front.  See the uid check there.
        "uid": os.getuid(),
        "gid": os.getgid(),
    }
    if meta_extra:
        meta.update(meta_extra)
    with open(os.path.join(image_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def _find_pid_by_pipe(pipe_inode):
    """Scan /proc to find the PID that holds a socket with the given inode.

    Skips the calling process's own pid: ``multiprocessing.Pipe()`` is built
    on ``socketpair()``, so both endpoints share the same socket inode.  The
    worker keeps the parent end and would otherwise be returned (it has the
    smaller pid, and ``/proc`` enumeration is pid-sorted) instead of the
    actual peer we are trying to locate.
    """
    target = f"socket:[{pipe_inode}]"
    self_pid = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        if int(entry) == self_pid:
            continue
        fd_dir = f"/proc/{entry}/fd"
        try:
            for fd_name in os.listdir(fd_dir):
                try:
                    link = os.readlink(f"{fd_dir}/{fd_name}")
                    if link == target:
                        return int(entry)
                except OSError:
                    pass
        except (OSError, PermissionError):
            pass
    return None


# ---------------------------------------------------------------------------
# Recorded task-id collisions
# ---------------------------------------------------------------------------
#
# CRIU recreates every task at its recorded id with clone3(set_tid), and a PID
# is just the id of a thread group's leader: ids for leaders and for threads
# come out of ONE per-namespace number space.  So an unrelated process's
# *thread* blocks a restore exactly as a same-PID process does, and the set
# that must be free is the whole recorded task list -- for a TP2 image that is
# ~900 ids across three leaders, not three PIDs.
#
# The kernel reports the two cases through different CRIU code paths, and
# neither names the occupant:
#   leader  -> "Can't fork for <pid>: File exists"
#   thread  -> pie: "Unable to create a thread: -17"   (-17 == EEXIST)
# so resolve the occupants here instead, both before a restore and after one
# fails.  Only the reduced-capability path needs this; the namespace path
# restores into a private PID namespace where every recorded id is free.

_PID_COLLISION_MARKER = "recorded task-id collision"

# Substrings that identify a collision, whichever layer reported it.
_PID_COLLISION_SIGNATURES = (
    "File exists",
    "Unable to create a thread: -17",
    _PID_COLLISION_MARKER,
)


def _is_pid_collision(exc_or_text):
    """True when a restore failure is a recorded task-id collision."""
    text = str(exc_or_text)
    return any(sig in text for sig in _PID_COLLISION_SIGNATURES)


_PORT_COLLISION_MARKER = "semi_p listening-port collision"
_PORT_COLLISION_SIGNATURES = (
    "Can't bind inet socket",
    "Address already in use",
)


def _is_port_collision(exc_or_text):
    """True when a restore failure is a recorded listening-port collision.

    Deliberately NOT added to the retry signatures.  A task id can be freed by
    a zombie being reaped, which is what makes retrying PID collisions worth
    it; a listening port is held by a live socket that outlives the restore.
    Seven consecutive attempts have been observed failing on the same port.
    """
    text = str(exc_or_text)
    return any(sig in text for sig in _PORT_COLLISION_SIGNATURES)


def _criu_log_excerpt(log_path, tail_chars=1200):
    """The lines that explain a criu failure, not just the last N bytes.

    CRIU logs the cause and then carries on unwinding: a failed restore ends
    with hundreds of ghost-remap unlinks whose long absolute paths are exactly
    what a blind tail shows.  A real case had "Can't fork for 2104: File
    exists" 19 lines from the end of 13859 and still outside a 2000-char tail.
    So lift the Error/Warn lines out first, then append a short tail for
    context.
    """
    try:
        with open(log_path, errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return ""
    errors = [ln.rstrip() for ln in lines if "Error" in ln or "Warn " in ln]
    out = ""
    if errors:
        out += "--- restore.log errors ---\n" + "\n".join(errors[-12:]) + "\n"
    return out + "--- restore.log tail ---\n" + "".join(lines)[-tail_chars:]


def _image_recorded_tids(image_dir):
    """Every task id the image will demand, from ``pstree.img``.

    Returns ``(leaders, tids)``, or ``(None, None)`` when the image cannot be
    decoded.  ``crit`` ships with CRIU, but a missing or newer-format
    ``pstree.img`` must never be the reason a restore fails, so every caller
    treats ``None`` as "unknown, carry on".
    """
    pstree = os.path.join(image_dir, "pstree.img")
    if not os.path.exists(pstree):
        return None, None
    try:
        res = subprocess.run(["crit", "decode", "-i", pstree],
                             capture_output=True, timeout=60)
        if res.returncode != 0:
            return None, None
        doc = json.loads(res.stdout.decode("utf-8", "replace"))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None, None
    leaders, tids = [], set()
    for ent in doc.get("entries", []):
        pid = ent.get("pid")
        if pid is None:
            continue
        leaders.append(int(pid))
        tids.add(int(pid))
        tids.update(int(t) for t in (ent.get("threads") or []))
    if not tids:
        return None, None
    return leaders, tids


def _live_task_ids():
    """Map every task id live in this PID namespace to its leader PID.

    Threads are only listed under ``/proc/<pid>/task``, never in ``/proc``
    itself, which is why a plain ``/proc`` scan (or ``ps``) misses most of
    what actually blocks a restore.
    """
    live = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            for tid in os.listdir(f"/proc/{entry}/task"):
                live[int(tid)] = int(entry)
        except OSError:
            continue  # exited while we walked it
    return live


def _own_ancestry():
    """This process and its ancestors, for labelling a self-collision."""
    chain, pid = set(), os.getpid()
    while pid and pid not in chain:
        chain.add(pid)
        try:
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
    return chain


def _pid_group_references(wanted):
    """Which live processes reference each of ``wanted`` as a pgid or sid.

    Resolving occupancy from ``/proc/<pid>/task`` alone is not enough, and the
    gap is not academic -- it is the dev-cluster restore failure.  A pid is a
    refcounted ``struct pid`` in the namespace's IDR, and references come from
    the task itself *and* from every process using it as a process-group or
    session id, since a group is named by its leader's pid.  An unreaped
    zombie keeps those links.  So an id can be unused by any task and still be
    unallocatable: ``clone3(set_tid=N)``, which is how CRIU places a restored
    task, fails ``EEXIST`` while ``/proc`` shows nothing at ``N``, because
    ``/proc`` lists tasks.

    Returns ``{id: [(pid, comm, state, kind), ...]}``.  A process referencing
    its *own* id is skipped: that is a task collision, which ``_live_task_ids``
    already reports with the occupant named.
    """
    refs = {}
    if not wanted:
        return refs
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        # comm (field 2) is parenthesised and may itself contain parens and
        # spaces, so split after the last ')': index 0 is then field 3.
        # Fields 5 and 6 -- pgrp and session -- are what /proc/<pid>/task
        # cannot show.
        try:
            with open(f"/proc/{entry}/stat") as f:
                head, _, tail = f.read().rpartition(")")
            fields = tail.split()
            comm = head.partition("(")[2] or "?"
            state, pgrp, sid = fields[0], int(fields[2]), int(fields[3])
        except (OSError, IndexError, ValueError):
            continue  # exited while we walked it
        pid = int(entry)
        for rid in {pgrp, sid}:
            if rid not in wanted or rid == pid:
                continue
            kind = "+".join(k for k, v in (("pgid", pgrp), ("sid", sid))
                            if v == rid)
            refs.setdefault(rid, []).append((pid, comm, state, kind))
    return refs


def _pid_collision_report(image_dir):
    """Describe recorded task ids that are unavailable right now, or ``None``.

    Covers both ways an id can be held: by a live task, and -- with no task at
    all -- by a process group or session reference (``_pid_group_references``).
    Reporting only the first is what let this preflight answer "no collisions"
    through a restore that then failed on the very first id it tried.

    Task occupancy is grouped by occupying process, because a single
    200-thread process accounts for hundreds of ids and the operator can only
    act on the process.  Reference occupancy is grouped by held id, because
    the action there is to reap the holders.
    """
    _leaders, tids = _image_recorded_tids(image_dir)
    if not tids:
        return None
    live = _live_task_ids()
    taken_by = {}
    for tid in sorted(tids.intersection(live)):
        taken_by.setdefault(live[tid], []).append(tid)
    # Only ids with no task of their own: an id whose leader is live is
    # already named above, and saying it twice helps nobody.
    refs = _pid_group_references(tids.difference(live))
    if not taken_by and not refs:
        return None
    mine = _own_ancestry()
    parts = []
    for owner, taken in sorted(taken_by.items()):
        try:
            with open(f"/proc/{owner}/comm") as f:
                comm = f.read().strip()
        except OSError:
            comm = "?"
        if owner in mine:
            comm += ", an ancestor of this restore"
        span = (str(taken[0]) if len(taken) == 1
                else f"{len(taken)} ids in {taken[0]}-{taken[-1]}")
        parts.append(f"pid {owner} ({comm}): {span}")
    ref_parts = []
    for rid, holders in sorted(refs.items()):
        who = []
        for hpid, hcomm, hstate, hkind in holders[:4]:
            label = f"pid {hpid} ({hcomm}"
            if hstate == "Z":
                label += ", zombie"
            if hpid in mine:
                label += ", an ancestor of this restore"
            who.append(f"{label}) as its {hkind}")
        if len(holders) > 4:
            who.append(f"and {len(holders) - 4} more")
        ref_parts.append(f"{rid} by " + ", ".join(who))
    msg = (f"{_PID_COLLISION_MARKER}: the image needs {len(tids)} task id(s) "
           f"in {min(tids)}-{max(tids)}")
    if parts:
        msg += "; occupied by " + "; ".join(parts)
    if ref_parts:
        msg += ("; held with no task of its own -- " + "; ".join(ref_parts)
                + " -- so the id stays allocated until every member of that "
                  "group or session is reaped, and CRIU gets EEXIST on an id "
                  "/proc shows as free")
    return msg + (f". Free them, or advance this PID namespace's counter past "
                  f"{max(tids)} before launching -- see scripts/pidcheck.py")


def _ephemeral_port_range():
    """``(low, high)`` from ``ip_local_port_range``, or ``(None, None)``."""
    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range") as f:
            lo, hi = f.read().split()[:2]
        return int(lo), int(hi)
    except (OSError, ValueError):
        return None, None


def _tcp_listen_holder(port):
    """``pid N (comm)`` for whoever holds ``port`` in TCP_LISTEN, or ``None``.

    Two hops, because neither half is enough on its own: ``/proc/net/tcp``
    knows the port and the socket inode but no pid, and ``/proc/<pid>/fd``
    knows the inode but not the port.
    """
    inodes = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path) as f:
                lines = f.read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":   # 0A = TCP_LISTEN
                continue
            try:
                if int(fields[1].split(":")[1], 16) != port:
                    continue
            except (IndexError, ValueError):
                continue
            inodes.add(fields[9])
    if not inodes:
        return None
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                link = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if link.startswith("socket:[") and link[8:-1] in inodes:
                try:
                    with open(f"/proc/{pid}/comm") as f:
                        comm = f.read().strip()
                except OSError:
                    comm = "?"
                return f"pid {pid} ({comm})"
    # Listening in this netns but owned from a pid namespace we cannot see.
    return f"an unreachable process (socket inode {sorted(inodes)[0]})"


def _port_collision_report(excerpt):
    """Describe the listening port CRIU could not re-bind, or ``None``.

    The inet analogue of ``_pid_collision_report``.  CRIU reports only
    "Address already in use" plus an in-image socket id, and neither
    identifies the occupant, so without this the failure arrives as an opaque
    log excerpt and the port has to be recovered by hand.
    """
    import re
    port = None
    # The last port CRIU announced restoring is the one that failed: the
    # "Restore: ... state TCP_LISTEN" line immediately precedes the error.
    for m in re.finditer(r"port (\d+)\s+state TCP_LISTEN", excerpt):
        port = int(m.group(1))
    if port is None:
        m = re.search(r"Can't bind inet socket \(id (\d+)\)", excerpt)
        return (f"{_PORT_COLLISION_MARKER}: CRIU could not bind recorded "
                f"socket id {m.group(1)}, but the port is not in the excerpt"
                if m else None)
    msg = (f"{_PORT_COLLISION_MARKER}: the image records a listening socket "
           f"on port {port}, which CRIU has to re-bind to restore")
    holder = _tcp_listen_holder(port)
    if holder:
        msg += f"; it is already held by {holder}"
    lo, hi = _ephemeral_port_range()
    if lo is not None and lo <= port <= hi:
        msg += (f". {port} is inside this host's ephemeral range {lo}-{hi}, "
                f"so the kernel can hand it to any process in this pod")
    # Unconditional: this is the actionable half, and gating it on a sysctl
    # read would drop it exactly where /proc is unavailable.
    msg += (". Retrying will not help -- the holder outlives the restore -- so "
            "this image cannot restore in this pod until the port is free. An "
            "image built after the prepare_criu_dump listener close should "
            "record no such port at all; if it does, it predates that fix")
    return msg


def _proc_starttime(pid):
    """Field 22 of ``/proc/<pid>/stat``: the task's start time in clock ticks.

    Paired with a pid this is an identity that survives pid recycling -- a
    recycled pid always reports a strictly later start time.  ``comm``
    (field 2) can contain spaces and parens, so parse after the last ')':
    index 0 is then field 3, which puts starttime at index 19.
    """
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def _tree_identity(root_pid):
    """Snapshot a restored tree while every task is still known-live.

    Teardown then never has to infer membership from ids that may since have
    been reused.  That matters on this path specifically: the lowcap restore
    places tasks in the shared number space, so its ids do get recycled.

    ``sid`` is recorded but unused by the strict teardown; it is the input
    the optional guarded session sweep would need.
    """
    ident = {"kind": "tree", "pid": root_pid,
             "start": _proc_starttime(root_pid), "members": {}}
    try:
        ident["sid"] = os.getsid(root_pid)
    except OSError:
        ident["sid"] = None
    for p in _get_descendant_pids(root_pid) + [root_pid]:
        st = _proc_starttime(p)
        if st is not None:
            ident["members"][p] = st
    return ident


# CRIU restores every task at its *recorded* PID (via clone3(set_tid)).
# Two images captured on the same node with their child trees alive at
# the same time carry adjacent/interleaved PIDs, and a destructive dump
# leaves the killed child as a zombie under its still-live worker, which
# keeps that PID occupied.  Either way the restore fails with
# "Can't fork for <pid>: File exists" (EEXIST) on whatever TID is already
# taken.
#
# The fix is to give each restore its own PID namespace so the recorded
# PIDs are free.  CRIU 4.2 can't ``--join-ns pid:`` an existing PID
# namespace ("join-ns pid namespace not supported"), so instead we run
# criu itself *inside* a fresh PID namespace: a privileged helper does
# ``unshare(CLONE_NEWPID)`` + ``fork()`` so its child is PID 1, that PID 1
# unshares a mount namespace and mounts a private /proc (so criu sees the
# namespace's PID view), forks criu ``restore -d``, records criu's exit
# code, then reaps forever so the namespace (and the detached restored
# tree, reparented onto it) stays alive.  Everything the rest of the
# worker touches keys off the restored root's *host* PID (found via the
# inherited pipe), which is unaffected by the nested namespace.  The
# holder tuple is ``(pid1_host_pid, popen)``; SIGKILLing PID 1 destroys
# the namespace and the whole restored tree in one shot.


def _kill_restored_tree(ident, log=None):
    """SIGKILL exactly the tasks of a CRIU-restored tree, and nothing else.

    Takes the ``_tree_identity`` snapshot taken at restore, not a bare pid.
    Used by the reduced-capability restore path, which has no PID namespace
    to collapse.  The restored tasks carry the image's uid, which equals this
    worker's -- dump and restore must share one uid, which
    ``Instance.criu_restore`` enforces -- so ``os.kill`` reaches every one of
    them without elevating.  A failure is logged rather than swallowed: an
    unkilled task holds GPU memory and squats on the task ids the next restore
    needs.

    This must never signal a negative pid.  The previous implementation
    ended with ``kill -9 -<root_pid>`` to catch tasks that had reparented
    away from the root.  procps-ng ``kill(1)`` parses a multi-digit negative
    pid as an option cluster and derives its target from the first digit
    alone (``pid = '0' - optopt``), so ``kill -9 -1181`` ran ``kill(-1,
    SIGKILL)``: as root that is every process it may signal.  It killed this
    worker mid-sweep and PID 1's only child with it, ending the container.
    Membership is enumerated explicitly instead, and every victim is matched
    against the restore-time snapshot so a recycled pid is never hit.
    """
    if not ident:
        return
    root_pid = ident.get("pid")
    if not root_pid or root_pid <= 1:
        if log is not None:
            log.error("  refusing to kill restored tree: bogus root pid=%r",
                      root_pid)
        return

    protected = _own_ancestry()
    members = dict(ident.get("members") or {})
    # Anything forked since the snapshot that is still parented under the
    # root, but only while the root is provably still ours: a pid picked up
    # by the live walk carries no snapshot start time to check against, so
    # were the root recycled this would enumerate an impostor's children.
    root_start = ident.get("start")
    live_root = _proc_starttime(root_pid)
    if live_root is not None and root_start in (None, live_root):
        for p in _get_descendant_pids(root_pid) + [root_pid]:
            members.setdefault(p, _proc_starttime(p))

    victims = []
    for pid, start in members.items():
        if pid <= 1 or pid in protected:
            continue
        live = _proc_starttime(pid)
        if live is None:
            continue                      # already gone
        if start is not None and live != start:
            continue                      # pid was recycled: not our task
        victims.append(pid)

    # Log BEFORE killing.  The old code logged after, so when the sweep took
    # out this worker the record of what it targeted was lost with it.
    if log is not None:
        log.info("  killing restored tree root=%s victims=%s",
                 root_pid, sorted(victims))
    for pid in sorted(victims, reverse=True):     # leaves first
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass                    # exited between the snapshot check and here
        except PermissionError:
            if log is not None:
                log.error("  cannot kill restored task pid=%s: not permitted. "
                          "It holds GPU memory and its recorded task ids will "
                          "block the next restore; kill it by hand.", pid)


def _kill_pidns_holder(holder, log=None):
    """Tear down a restore's process tree.

    Two holder shapes are accepted:

    * ``(pid1_host_pid, popen)`` -- the namespace-based path.  SIGKILLing
      PID 1 of the namespace makes the kernel kill every task in it (the
      restored tree), so this doubles as the restored-tree cleanup.
    * a ``_tree_identity`` dict -- the reduced-capability (host-PID-
      namespace) path.  There is no namespace to collapse, so the recorded
      tasks are SIGKILLed directly.
    """
    if not holder:
        return
    if isinstance(holder, dict):
        if holder.get("kind") == "tree":
            _kill_restored_tree(holder, log=log)
        return
    host_pid, proc = holder
    try:
        subprocess.run(["sudo", "kill", "-9", str(host_pid)],
                       capture_output=True)
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except OSError:
            pass
    if log is not None:
        log.info("  PID-namespace holder torn down (reaper host_pid=%s)",
                 host_pid)


def _worker_criu_load(image_dir, new_pipe_fd):
    """Restore a vLLM child process tree from a CRIU image on disk.

    Restores into a dedicated PID namespace (criu runs *inside* a fresh
    namespace held open by a PID 1 reaper) so the image's recorded PIDs
    can't collide with a concurrently-restored sibling's tree or with a
    zombie left behind by the dump.  Returns ``(host_pid, meta, holder)``
    where ``host_pid`` is the restored root's *host-visible* PID
    (discovered by scanning /proc for the inherited pipe -- the
    ``--pidfile`` reports the in-namespace PID, which is meaningless on
    the host) and ``holder`` is ``(pid1_host_pid, popen)`` to hand to
    ``_kill_pidns_holder`` at teardown.
    """
    import fcntl

    meta_path = os.path.join(image_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    pipe_resource = meta["pipe_resource"]
    pidfile = os.path.join(image_dir, "restored.pid")
    for _stale in [pidfile, os.path.join(image_dir, "restore.log")]:
        if os.path.exists(_stale):
            os.remove(_stale)

    flags = fcntl.fcntl(new_pipe_fd, fcntl.F_GETFD)
    fcntl.fcntl(new_pipe_fd, fcntl.F_SETFD, flags & ~fcntl.FD_CLOEXEC)

    pipe_inode = os.readlink(f"/proc/self/fd/{new_pipe_fd}")
    if pipe_inode.startswith("socket:["):
        pipe_inode = pipe_inode.split("[")[1].rstrip("]")

    # Opened here, in the host PID namespace: inside the helper's private one,
    # /proc/1 is the reaper rather than the container's PID 1.
    log_fd, log_argv = _stdout_inherit(meta)
    send_fds = [new_pipe_fd] + ([log_fd] if log_fd is not None else [])

    # sudo closes all FDs >= 3, so we pass the pipe fd (and the pod log) to
    # the child via a Unix domain socket (SCM_RIGHTS) bound to a temp path
    # that sudo can connect to.
    import socket as _socket, tempfile, array, threading

    sock_path = os.path.join(tempfile.gettempdir(),
                             f"criu_fd_{os.getpid()}_{new_pipe_fd}.sock")
    if os.path.exists(sock_path):
        os.remove(sock_path)

    srv = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    srv.bind(sock_path)
    os.chmod(sock_path, 0o777)
    srv.listen(1)

    def _send_fd():
        conn, _ = srv.accept()
        conn.sendmsg(
            [b"\x00"],
            [(_socket.SOL_SOCKET, _socket.SCM_RIGHTS,
              array.array("i", send_fds))]
        )
        conn.close()
        srv.close()

    sender = threading.Thread(target=_send_fd, daemon=True)
    sender.start()

    criu_argv = [
        "criu", "restore",
        "-D", image_dir,
        "-o", "restore.log",
        # No --shell-job: images captured by the updated dump path own
        # their session and hold no controlling terminal, so there is no
        # external tty to reattach -- which is what would otherwise break
        # restore inside the private PID namespace.
        "--tcp-close",
        "--inherit-fd", f"fd[{new_pipe_fd}]:{pipe_resource}",
        *log_argv,
        "--pidfile", pidfile,
        "--link-remap",
        "-d",
        "-v4",
    ]

    # Runtime side-channels (on shared storage, visible from inside the
    # helper's private mount namespace and from the host worker alike):
    #   reaper.pid -- PID 1's host PID, published by the host-ns parent
    #   restore.rc -- criu's exit code, published by PID 1 once criu exits
    reaper_pidfile = os.path.join(image_dir, "reaper.pid")
    rc_path = os.path.join(image_dir, "restore.rc")
    # A prior run that died abnormally (SIGKILL, crash) may have left its
    # reaper -- and thus its namespace + restored tree -- alive.  If a
    # stale reaper.pid points at a live process, kill it before reusing
    # this image so we don't accumulate orphaned namespaces.
    if os.path.exists(reaper_pidfile):
        try:
            _stale_reaper = int(open(reaper_pidfile).read().strip())
            if _stale_reaper > 1 and os.path.exists(f"/proc/{_stale_reaper}"):
                subprocess.run(["sudo", "kill", "-9", str(_stale_reaper)],
                               capture_output=True)
        except (OSError, ValueError):
            pass
    for _stale in (reaper_pidfile, rc_path):
        if os.path.exists(_stale):
            os.remove(_stale)

    # Helper (run as root via sudo).  sudo strips fds >= 3, so it first
    # re-receives the inherited pipe fd over the SCM_RIGHTS socket, then
    # unshares a PID namespace and forks: the host-ns parent publishes
    # PID 1's host PID and blocks; PID 1 unshares a mount namespace with a
    # private /proc, forks criu ``restore -d`` into the new namespace,
    # records criu's exit code, and reaps forever so the namespace (and
    # the detached restored tree) survives.
    helper_script = (
        "import ctypes, signal, sys, time\n"
        "libc = ctypes.CDLL('libc.so.6', use_errno=True)\n"
        "CLONE_NEWPID = 0x20000000\n"
        "CLONE_NEWNS = 0x00020000\n"
        "MS_REC = 0x4000\n"
        "MS_PRIVATE = 0x40000\n"
        + _helper_fd_prologue(sock_path, new_pipe_fd) +
        "if libc.unshare(CLONE_NEWPID) != 0:\n"
        "    e = ctypes.get_errno()\n"
        "    sys.stderr.write('unshare(CLONE_NEWPID): %s\\n' % os.strerror(e))\n"
        "    os._exit(3)\n"
        "pid1 = os.fork()\n"
        "if pid1 > 0:\n"
        # host-ns parent: fd belongs to the restored tree now, drop our
        # copy so the pipe scan uniquely finds the restored root; publish
        # PID 1's host pid; block so this process (and the sudo handle)
        # lives as long as the namespace.
        f"    os.close({new_pipe_fd})\n"
        "    if LFD is not None: os.close(LFD)\n"
        f"    open({reaper_pidfile!r}, 'w').write(str(pid1) + '\\n')\n"
        "    try:\n"
        "        os.waitpid(pid1, 0)\n"
        "    except OSError:\n"
        "        pass\n"
        "    os._exit(0)\n"
        "os.setsid()\n"
        "if libc.unshare(CLONE_NEWNS) == 0:\n"
        "    libc.mount(b'none', b'/', None, MS_REC | MS_PRIVATE, None)\n"
        "    libc.mount(b'proc', b'/proc', b'proc', 0, None)\n"
        "criu_pid = os.fork()\n"
        "if criu_pid == 0:\n"
        f"    os.execvp({criu_argv[0]!r}, {_helper_argv_expr(criu_argv)})\n"
        "    os._exit(127)\n"
        f"os.close({new_pipe_fd})\n"
        "if LFD is not None: os.close(LFD)\n"
        "_, status = os.waitpid(criu_pid, 0)\n"
        "rc = os.waitstatus_to_exitcode(status)\n"
        "try:\n"
        f"    open({rc_path!r}, 'w').write(str(rc) + '\\n')\n"
        "except OSError:\n"
        "    pass\n"
        "signal.signal(signal.SIGTERM, lambda *a: os._exit(0))\n"
        "while True:\n"
        "    try:\n"
        "        os.waitpid(-1, 0)\n"
        "    except ChildProcessError:\n"
        "        time.sleep(0.2)\n"
        "    except OSError:\n"
        "        time.sleep(0.2)\n"
    )
    cmd = ["python3", "-c", helper_script]
    if not _is_root:
        cmd.insert(0, "sudo")

    holder = None
    # From here on, any failure must tear the namespace down (which also
    # kills anything CRIU partially restored into it) so we don't leak a
    # namespace / orphan tree per attempt.
    try:
        # stdout -> /dev/null: criu logs to restore.log (via -o) and we
        # signal completion through files, so nothing must be drained from
        # stdout.  Keep stderr for helper diagnostics.
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True)

        # Wait for PID 1's host pid and criu's exit code to appear.  criu
        # restore of a large model tree can take a while; poll generously.
        deadline = time.time() + 300
        reaper_pid = None
        rc = None
        while time.time() < deadline:
            if reaper_pid is None and os.path.exists(reaper_pidfile):
                try:
                    reaper_pid = int(open(reaper_pidfile).read().strip())
                    holder = (reaper_pid, proc)
                except (OSError, ValueError):
                    reaper_pid = None
            if os.path.exists(rc_path):
                try:
                    rc = int(open(rc_path).read().strip())
                    break
                except (OSError, ValueError):
                    rc = None
            if proc.poll() is not None and reaper_pid is None:
                # Helper died before even publishing PID 1 (e.g. unshare
                # failed); surface its stderr.
                break
            time.sleep(0.1)

        sender.join(timeout=2)
        try:
            os.remove(sock_path)
        except OSError:
            pass
        subprocess.run(["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", image_dir],
                       capture_output=True)

        if rc is None:
            _err = ""
            try:
                if proc.poll() is not None:
                    _err = (proc.stderr.read() or "")[-500:]
            except Exception:
                pass
            raise RuntimeError(
                f"criu restore did not complete (reaper_pid={reaper_pid!r}, "
                f"helper stderr={_err!r})")
        if rc != 0:
            detail = f"rc={rc}"
            log_path = os.path.join(image_dir, "restore.log")
            if os.path.exists(log_path):
                excerpt = _criu_log_excerpt(log_path)
                detail += "\n" + excerpt
                if _is_port_collision(excerpt):
                    clash = _port_collision_report(excerpt)
                    if clash:
                        detail += f"\n--- {clash}"
            raise RuntimeError(f"criu restore failed ({detail})")

        # The restored root lives in the private namespace, so the
        # ``--pidfile`` value is its in-namespace PID -- meaningless on the
        # host.  Find its host PID by the inherited pipe instead.
        new_pid = _find_pid_by_pipe(pipe_inode)
        if new_pid is None or new_pid == os.getpid():
            raise RuntimeError(
                f"failed to discover CRIU-restored root host pid "
                f"(pipe scan returned {new_pid!r})")
    except BaseException:
        _kill_pidns_holder(holder)
        raise
    finally:
        if log_fd is not None:
            os.close(log_fd)

    return new_pid, meta, holder


# ---------------------------------------------------------------------------
# Reduced-capability restore (no private PID namespace)
# ---------------------------------------------------------------------------
#
# Selected by the single SEMIP_UNPRIVILEGED=1 switch (see _unprivileged()),
# which also makes _worker_criu_save pass --unprivileged so BOTH dump and
# restore run on a non-privileged pod.
#
# The namespace-based path above needs CAP_SYS_ADMIN for two things:
#   1. CRIU's kerndat init, which builds a throwaway *network* namespace to
#      probe kernel features (fails with EPERM where netns creation is
#      blocked -- see criu check / dump.log "Could not initialize kernel
#      features detection").  This affects the dump too, which is why
#      _worker_criu_save also passes --unprivileged in unprivileged mode.
#   2. This worker's own unshare(CLONE_NEWPID|CLONE_NEWNS) helper, used only
#      to avoid PID collisions between *concurrent* restores on one node.
#
# Neither is fundamental to restoring a single instance:
#   * (1) is bypassed by CRIU's ``--unprivileged`` mode, which SKIPS the
#     network-namespace kerndat probe entirely.
#   * (2) is only needed for concurrency.  CRIU places each task at its
#     recorded PID via clone3(set_tid), which in the *host* PID namespace is
#     authorized by CAP_CHECKPOINT_RESTORE -- so it does NOT need to write
#     the read-only ns_last_pid sysctl.  Dropping the namespace trades
#     concurrent-restore safety for not needing CAP_SYS_ADMIN: every recorded
#     task id (leaders *and* threads -- one number space) must be free on the
#     host, so at most one live restore per node, and any unrelated process
#     whose threads happen to span the image's id range blocks it too.  The
#     preflight in _worker_criu_load_lowcap names whatever is in the way.
#
# Net effect: this path targets a floor of roughly
#   CAP_CHECKPOINT_RESTORE (+ CAP_SYS_PTRACE)
# with no unshare() calls.  The CUDA restore (cuCheckpointProcessRestore) is
# orthogonal -- it is gated by /dev/nvidia* device access, not capabilities.
#
# Enable with SEMIP_UNPRIVILEGED=1.  Validate a node first with:
#   sudo criu check --unprivileged
# The image must record shed capabilities for restore_creds() to succeed on
# the low-cap node -- in this mode the child drops its caps at dump
# automatically (implied by SEMIP_UNPRIVILEGED; no separate flag).


def _worker_criu_load_lowcap(image_dir, new_pipe_fd):
    """Restore a vLLM child tree WITHOUT a private PID namespace.

    Drop-in alternative to ``_worker_criu_load`` with the same
    ``(host_pid, meta, holder)`` return contract, but:

    * runs ``criu restore -d`` directly in the host PID namespace (no
      unshare), so it needs no CAP_SYS_ADMIN for namespace creation;
    * passes ``--unprivileged`` so CRIU skips the network-namespace kerndat
      probe that otherwise aborts on nodes where netns creation is blocked;
    * reads the restored root's host PID straight from ``--pidfile`` (with a
      pipe-scan fallback), since without a PID namespace the recorded PID is
      the host PID;
    * returns ``holder = _tree_identity(host_pid)`` -- the tree's task ids
      with their start times, snapshotted while every task is known-live --
      for ``_kill_pidns_holder`` -> ``_kill_restored_tree`` at teardown.

    Constraint: every recorded task id -- thread group leaders *and* their
    threads -- must be free on the host, which in practice means at most one
    live restore per node.  ``_pid_collision_report`` resolves the occupants
    up front and again if CRIU fails, since neither of CRIU's two collision
    messages ("File exists" for a leader, "Unable to create a thread: -17"
    for a thread) says which id or which process is in the way.
    """
    import fcntl, socket as _socket, tempfile, array, threading

    meta_path = os.path.join(image_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    pipe_resource = meta["pipe_resource"]

    # Fail before spawning criu: a collision is fatal to the restore anyway,
    # and reporting it here names the occupying process instead of leaving an
    # EEXIST buried in restore.log.  Carries the marker the caller's retry
    # loop keys on, so a transient holder (a zombie awaiting reap) still gets
    # the same handful of retries it used to.
    clash = _pid_collision_report(image_dir)
    if clash:
        raise RuntimeError(f"criu restore aborted ({clash})")

    pidfile = os.path.join(image_dir, "restored.pid")
    for _stale in (pidfile, os.path.join(image_dir, "restore.log")):
        if os.path.exists(_stale):
            os.remove(_stale)

    flags = fcntl.fcntl(new_pipe_fd, fcntl.F_GETFD)
    fcntl.fcntl(new_pipe_fd, fcntl.F_SETFD, flags & ~fcntl.FD_CLOEXEC)

    pipe_inode = os.readlink(f"/proc/self/fd/{new_pipe_fd}")
    if pipe_inode.startswith("socket:["):
        pipe_inode = pipe_inode.split("[")[1].rstrip("]")

    log_fd, log_argv = _stdout_inherit(meta)
    send_fds = [new_pipe_fd] + ([log_fd] if log_fd is not None else [])

    # subprocess closes fds >= 3, so hand the pipe fd (and the pod log) to the
    # helper over a Unix socket via SCM_RIGHTS, same as the namespace path.
    sock_path = os.path.join(tempfile.gettempdir(),
                             f"criu_fd_{os.getpid()}_{new_pipe_fd}.sock")
    if os.path.exists(sock_path):
        os.remove(sock_path)
    srv = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    srv.bind(sock_path)
    os.chmod(sock_path, 0o777)
    srv.listen(1)

    def _send_fd():
        try:
            conn, _ = srv.accept()
            conn.sendmsg(
                [b"\x00"],
                [(_socket.SOL_SOCKET, _socket.SCM_RIGHTS,
                  array.array("i", send_fds))])
            conn.close()
        finally:
            srv.close()

    sender = threading.Thread(target=_send_fd, daemon=True)
    sender.start()

    criu_argv = [
        _CRIU_BIN, "restore",
        "-D", image_dir,
        "-o", "restore.log",
        "--tcp-close",
        "--inherit-fd", f"fd[{new_pipe_fd}]:{pipe_resource}",
        *log_argv,
        "--pidfile", pidfile,
        "--link-remap",
        "-d",
        "-v4",
    ]
    # --unprivileged makes CRIU operate within CAP_CHECKPOINT_RESTORE limits
    # (skipping operations that would need CAP_SYS_ADMIN, e.g. the netns
    # kerndat probe).  This path is only reached when SEMIP_UNPRIVILEGED=1, so
    # it is always required here -- without it the restore fails on the very
    # low-capability nodes this path exists for.
    criu_argv.append("--unprivileged")

    # Helper: re-receive the pipe fd, dup2 it to the number criu expects, then
    # exec `criu restore -d` DIRECTLY -- no unshare, no new namespaces.  With
    # -d, criu detaches the restored tree (reparented to the nearest subreaper)
    # and exits with the restore rc, so the helper's exit code IS criu's rc; no
    # long-lived reaper is required.  It stays a separate process even without
    # sudo because subprocess closes fds >= 3, so the fd still arrives over the
    # SCM_RIGHTS socket rather than by inheritance.
    helper_script = (
        _helper_fd_prologue(sock_path, new_pipe_fd)
        + f"os.execvp({criu_argv[0]!r}, {_helper_argv_expr(criu_argv)})\n"
    )
    # Restore at the worker's own uid, never through sudo -- the mirror of the
    # dump (see _worker_criu_save), and required for two independent reasons.
    #
    # Credentials: the image's uid must equal ours (Instance.criu_restore
    # asserts it), so restoring at that uid means restore_creds() has no
    # setresuid to do -- which it could not perform unprivileged.
    #
    # It does NOT follow that PR_CAPBSET_DROP has nothing to do, as this
    # comment claimed until 2026-09-14.  criu drops every capability absent
    # from the image's recorded cap_bnd without checking whether it is already
    # absent here, and the kernel tests CAP_SETPCAP first, so each call returns
    # EPERM even when there is nothing to drop.  What makes that survivable is
    # a dump-side property: the child records no_new_privs, which is criu's own
    # licence to demote the EPERM to a warning.  See Complication 11.
    #
    # Pipe sizes: CRIU recreates each pipe at its recorded capacity with
    # F_SETPIPE_SZ, which needs CAP_SYS_RESOURCE once the *creating uid* is
    # over fs.pipe-user-pages-soft.  That capability is outside a cap-prod
    # pod's bounding set, so sudo cannot supply it, and uid 0 is the account
    # most likely to be over the limit (every root-run tree on the node shares
    # one budget).  Measured 2026-09-10: a root pipe came back at 8 KiB and
    # F_SETPIPE_SZ failed outright, while the same call as uid 1000 succeeded:
    #   Error (criu/pipes.c:165): Can't restore pipe size: Operation not permitted
    #   Error (criu/files.c:1221): Unable to open fd=3 id=0x1b3
    cmd = [sys.executable or "python3", "-c", helper_script]

    holder = None
    try:
        # stdout/stderr -> /dev/null: criu logs everything to restore.log
        # (via -o).  These must NOT be pipes we drain -- criu's -d children
        # can hold them open after criu exits, so a pipe would never EOF and
        # draining it (subprocess.run) would deadlock.
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)

        sender.join(timeout=2)
        try:
            os.remove(sock_path)
        except OSError:
            pass

        if result.returncode != 0:
            detail = f"rc={result.returncode}"
            log_path = os.path.join(image_dir, "restore.log")
            if os.path.exists(log_path):
                excerpt = _criu_log_excerpt(log_path)
                detail += "\n" + excerpt
                # An id can be claimed between the preflight and the restore
                # (criu's own helpers spawn into the same number space), so
                # re-resolve rather than assume the preflight covered it.
                if _is_pid_collision(excerpt):
                    clash = _pid_collision_report(image_dir)
                    if clash:
                        detail += f"\n--- {clash}"
                if _is_port_collision(excerpt):
                    clash = _port_collision_report(excerpt)
                    if clash:
                        detail += f"\n--- {clash}"
            raise RuntimeError(f"criu restore failed ({detail})")

        # No PID namespace: the recorded PID *is* the host PID, so --pidfile
        # is authoritative.  Fall back to the pipe scan if it is missing or
        # stale.
        new_pid = None
        if os.path.exists(pidfile):
            try:
                new_pid = int(open(pidfile).read().strip())
            except (OSError, ValueError):
                new_pid = None
        if not new_pid or not os.path.exists(f"/proc/{new_pid}"):
            new_pid = _find_pid_by_pipe(pipe_inode)
        if new_pid is None or new_pid == os.getpid():
            raise RuntimeError(
                "failed to discover CRIU-restored root host pid "
                "(pidfile + pipe scan both failed)")
        holder = _tree_identity(new_pid)
    except BaseException:
        _kill_pidns_holder(holder)
        raise
    finally:
        if log_fd is not None:
            os.close(log_fd)

    return new_pid, meta, holder


# ---------------------------------------------------------------------------
# Child thread -- communicates with the vLLM child process via pipe
# ---------------------------------------------------------------------------

def _child_thread(instance_id, gpus, child_pid, pipe,
                  child_queue, result_queue, completed_counter,
                  initial_state="alive", capture_gpu_uuids=None,
                  child_proc=None):
    """Thread that owns the single child.  Pulls commands from child_queue,
    executes them serially, puts results on result_queue.

    ``gpus`` is the physical GPU list this tree currently occupies (the
    "old" side of a later migration); single-element at TP=1.
    ``capture_gpu_uuids`` are the capture node's GPU UUIDs from meta.json,
    needed to build the restore device map across nodes.

    ``child_proc`` is the ``Process`` handle when this worker spawned the
    child, so ``criu_dump`` can reap the corpse it leaves behind and free its
    task ids (see ``_reap_dumped_child``).  ``None`` on the restore path: criu
    detached that child and it reparented away, so it is not ours to reap.

    Generate commands are fire-and-forget on the pipe.  The child sends
    back ("generate_done", ...) when done (new async child) or
    ("generate", ...) immediately (old CRIU-restored child with sync
    llm.generate).  The worker accepts both.
    """
    if isinstance(gpus, int):
        gpus = [gpus]
    gpus = list(gpus)
    rank = gpus[0]
    log = semip_logging.worker(instance_id, rank)

    def _emit_result(cmd, elapsed, error, info):
        result_queue.put((cmd, elapsed, error, info))
        with completed_counter.get_lock():
            completed_counter.value += 1

    _pending_generates = 0
    # Mirrors the child's `_paused` flag.  Set on a successful `pause`
    # ack, cleared on a successful `resume` ack.  While true, the child
    # has frozen its step loop so no `generate_done` will arrive: we
    # turn `_drain_pipe_generates` into a no-op so that subsequent
    # synchronous commands (`unpin`, `sleep`, `cuda_checkpoint`, ...)
    # do not deadlock waiting on completions that will not come until
    # `resume()` re-engages the engine via prefill.
    _worker_paused = False

    def _handle_pipe_result(result):
        """Handle a single result tuple from the pipe.  Returns True if
        it was a generate completion, False otherwise (caller should
        handle it)."""
        nonlocal _pending_generates
        if isinstance(result, tuple) and len(result) == 4:
            if result[0] in ("generate_done", "generate"):
                _pending_generates -= 1
                _emit_result("generate", result[1], result[2], result[3])
                return True
        return False

    def _drain_pipe_generates():
        nonlocal _pending_generates
        if _worker_paused:
            return
        while _pending_generates > 0:
            try:
                result = pipe.recv()
            except (BrokenPipeError, ConnectionResetError, EOFError):
                while _pending_generates > 0:
                    _emit_result("generate", 0.0,
                                 "child process died during generate", {})
                    _pending_generates -= 1
                break
            _handle_pipe_result(result)

    def _recv_sync():
        """Receive a synchronous command response, transparently consuming
        any generate completions that arrive first."""
        while True:
            result = pipe.recv()
            if not _handle_pipe_result(result):
                return result

    def _get_next_command():
        """Get next command, polling pipe for generate results while waiting."""
        nonlocal _pending_generates
        if _pending_generates == 0:
            return child_queue.get()
        while True:
            while _pending_generates > 0 and pipe.poll(0):
                try:
                    result = pipe.recv()
                except (BrokenPipeError, ConnectionResetError, EOFError):
                    while _pending_generates > 0:
                        _emit_result("generate", 0.0,
                                     "child process died during generate", {})
                        _pending_generates -= 1
                    break
                _handle_pipe_result(result)
            try:
                return child_queue.get(timeout=0.01)
            except queue.Empty:
                if _pending_generates == 0:
                    return child_queue.get()

    state = initial_state
    checkpointed_pids = (
        _get_descendant_pids(child_pid) + [child_pid]
        if initial_state == "checkpointed" else None
    )
    # When cuda_restore fails the child's CUDA driver state is left in
    # an inconsistent (locked / partially-restored) state.  Any
    # subsequent pipe-bound op (repin, generate, sleep, ...) would
    # deadlock the child in the driver.  Latch a "broken" flag so we
    # auto-fail those instead of forwarding them, until a later
    # cuda_restore succeeds (or the worker is torn down).
    cuda_broken = False
    cuda_broken_reason = None

    while True:
        cmd, kwargs = _get_next_command()
        log.info(">>> %s", cmd)

        if cmd == "exit":
            if state == "alive":
                _drain_pipe_generates()
                pipe.send(("exit", {}))
                while True:
                    result = pipe.recv()
                    if not _handle_pipe_result(result):
                        break
            pipe.close()
            _kill_process_tree(child_pid)
            log.info("exited")
            break

        if cmd == "cuda_checkpoint":
            _drain_pipe_generates()
            t0 = time.perf_counter()
            error = None
            info = {}
            try:
                checkpointed_pids = _worker_checkpoint(child_pid, unlock=False)
                log.info("  checkpointed pids: %s", checkpointed_pids)
                state = "checkpointed"
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            elapsed = time.perf_counter() - t0
            log.info("<<< cuda_checkpoint %s (%.3fs)",
                     'OK' if error is None else 'FAILED', elapsed)
            _emit_result(cmd, elapsed, error, info)
            continue

        if cmd == "cuda_restore":
            t0 = time.perf_counter()
            error = None
            # TP>1 sends the whole placement list; a scalar gpu= is still
            # accepted for TP=1 callers.
            if kwargs.get("gpus") is not None:
                target_gpus = list(kwargs["gpus"])
            else:
                target_gpus = [kwargs["gpu"]]
            target_gpu = target_gpus[0]
            info = {}
            try:
                if state == "alive":
                    checkpointed_pids = _get_descendant_pids(child_pid) + [child_pid]
                    log.info("  process is alive, checkpointing before restore...")
                    cu = _get_cu()
                    for _p in checkpointed_pids:
                        _check_cu(f"Lock({_p})", cu.cuCheckpointProcessLock(_p, None))
                        _check_cu(f"Checkpoint({_p})", cu.cuCheckpointProcessCheckpoint(_p, None))
                    state = "checkpointed"
                if checkpointed_pids is None:
                    raise RuntimeError("restore called but no checkpointed PIDs stored")
                _t_settle = time.monotonic()
                _unsettled, _states = _wait_for_quiescence(checkpointed_pids)
                _settle_s = time.monotonic() - _t_settle
                # A pid that missed the window is not necessarily a problem. The
                # wait exists to keep cuCheckpointProcessRestore away from a task
                # inside a kernel or driver operation; a task spinning on the CPU
                # in userspace is not that, and here it cannot be, because the
                # ranks have no CUDA context until the next phase. Root-caused
                # 2026-09-22: the spin is vLLM's SpinCondition busy-loop running
                # against a CLOCK_MONOTONIC that the restore could not preserve,
                # so it can last indefinitely and no budget will outlast it. See
                # _is_userspace_spin.
                _spinning = [_p for _p in _unsettled if _is_userspace_spin(_p)]
                _blocked = [_p for _p in _unsettled if _p not in _spinning]
                if _blocked:
                    # Never skip and carry on. Skipping leaves the ranks with
                    # no GPU device fds at all, and the very next phase
                    # (reinit_nccl) then deadlocks in futex with no diagnostic
                    # -- a job wedged for its full 45-minute idle timeout. This
                    # used to log and fall through to `<<< cuda_restore OK`,
                    # which is why it went unnoticed across twelve debugging
                    # sessions. Report the states too: a pid that misses the
                    # window is in *some* state, and that is the only clue to
                    # why it never quiesced.
                    raise RuntimeError(
                        "CUDA restore skipped: %d of %d pid(s) never quiesced "
                        "within %.1fs of the CRIU restore and are blocked in "
                        "the kernel, so the ranks would come up with no GPUs "
                        "and reinit_nccl would hang. States: %s"
                        % (len(_blocked), len(checkpointed_pids),
                           _settle_timeout_s(),
                           ", ".join(f"{_p}={_states.get(_p, 'gone')}"
                                     for _p in _blocked)))
                if _spinning:
                    # Logged rather than passed over in silence: this is the
                    # only signal that a restore hit the clock bug, and without
                    # it a working fix and an absent bug look identical.
                    log.info("  %d pid(s) accepted as userspace spins after "
                             "%.1fs (wchan=0, syscall=running, no CUDA context "
                             "yet, so not GPU work): %s",
                             len(_spinning), _settle_timeout_s(),
                             ", ".join(str(_p) for _p in _spinning))
                # Logged separately from the phase total because the two answer
                # different questions and only their sum was ever visible.
                # Measured 2026-09-20 at TP=2: the wait is the bulk of it, and
                # the ranks that miss a short budget report State R (running) --
                # busy, not blocked -- so this is a duration to calibrate
                # against, not a wedge. Sitting at 4.0-5.0s against the 5s
                # per-pid budget the old code used is the whole explanation for
                # why the hangs looked intermittent.
                log.info("  %d pid(s) quiesced in %.3fs (budget %.1fs)",
                         len(checkpointed_pids), _settle_s, _settle_timeout_s())
                _worker_restore(checkpointed_pids,
                                old_gpus=gpus,
                                new_gpus=target_gpus,
                                old_uuids=capture_gpu_uuids)
                log.info("  restored pids: %s", checkpointed_pids)
                info["gpu"] = target_gpu
                info["gpus"] = list(target_gpus)
                gpus = list(target_gpus)
                rank = target_gpu
                log.set_gpu(target_gpu)
                checkpointed_pids = None
                state = "alive"
                cuda_broken = False
                cuda_broken_reason = None
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                cuda_broken = True
                cuda_broken_reason = f"cuda_restore failed: {error}"
            elapsed = time.perf_counter() - t0
            log.info("<<< cuda_restore %s (%.3fs)",
                     'OK' if error is None else 'FAILED', elapsed)
            _emit_result(cmd, elapsed, error, info)
            continue

        if cmd == "criu_dump":
            _drain_pipe_generates()
            t0 = time.perf_counter()
            error = None
            info = {}
            try:
                image_dir = kwargs["filename"]
                pipe.send(("get_pipe_fd", {}))
                fd_result = _recv_sync()
                _fd_cmd, _fd_elapsed, _fd_error, fd_info = fd_result
                if _fd_error is not None:
                    raise RuntimeError(f"get_pipe_fd failed: {_fd_error}")
                child_pipe_fd = fd_info["pipe_fd"]
                log.info("  child pipe fd: %s", child_pipe_fd)

                pipe.send(("prepare_criu_dump", {"pipe_fd": child_pipe_fd}))
                prep_result = _recv_sync()
                _pr_cmd, _pr_elapsed, _pr_error, pr_info = prep_result
                if _pr_error is not None:
                    log.warning("  prepare_criu_dump failed: %s", _pr_error)
                else:
                    log.info("  prepare_criu_dump: fds=%s, unmapped=%s, "
                             "reaped_strays=%s, closed_listeners=%s",
                             pr_info.get('closed_fds', []),
                             pr_info.get('unmapped', []),
                             pr_info.get('reaped_strays', []),
                             pr_info.get('closed_listeners', []))
                    log.info("  prepare_criu_dump: driver listener gate=%s",
                             pr_info.get('listener_diag', {}))
                    # At TP>1 every listener in the image belongs to a rank, and
                    # none of them is closed -- torch retires its own before the
                    # dump, and what survives is NCCL's RAS socket, which has to
                    # ride in.  So this line is a census of what the image
                    # carries, not a report of work done.
                    if 'worker_listener_diag' in pr_info:
                        log.info("  prepare_criu_dump: per-rank listeners "
                                 "recorded into the image (none closed at "
                                 "TP>1): %s",
                                 pr_info.get('worker_listener_diag', []))
                        # Called out separately because a rank with no ephemeral
                        # row is the shape that used to mean "RAS's socket was
                        # closed", and that regression is only visible a restore
                        # later.  Non-empty here is the healthy reading.
                        _eph = [d.get('recorded_ephemeral')
                                for d in pr_info.get('worker_listener_diag', [])
                                if isinstance(d, dict)]
                        if not all(_eph):
                            log.warning(
                                "  prepare_criu_dump: a rank recorded no "
                                "ephemeral loopback listener (%s). At TP>1 that "
                                "socket is NCCL RAS's and should be present; "
                                "its absence is what preceded the post-restore "
                                "EBADF spin. See CRIU_PLUMBING.md "
                                "Complication 15.", _eph)

                _park = pr_info.get("mq_park") if _pr_error is None else None
                if _park is not None and not _park.get("ok"):
                    raise RuntimeError(
                        "message-queue park failed, so the tree still holds "
                        f"live queue sockets: {_park.get('error')}")

                pipe_resource = _resolve_fd_resource(child_pid, child_pipe_fd)

                _census = _inet_census(
                    [child_pid] + _get_descendant_pids(child_pid))
                info["inet_census"] = _census
                log.info("  inet sockets off loopback going into the image: "
                         "%d %s", len(_census), _census)
                _vcfg = (kwargs.get("meta_extra") or {}).get("vllm_config") or {}
                if int(_vcfg.get("nnodes", 1) or 1) > 1:
                    # A connection needs TCP repair to restore and names the
                    # dump pod's address, and so does a listener bound to a
                    # specific one; only a wildcard listener restores anywhere.
                    _bad = [r for r in _census
                            if r["state"] != "LISTEN"
                            or r["local"].rsplit(":", 1)[0] not in ("::",
                                                                    "0.0.0.0")]
                    if _bad:
                        raise RuntimeError(
                            f"{len(_bad)} inet socket(s) would go into a "
                            f"multi-node image: {_bad}")

                meta = _worker_criu_save(
                    child_pid, image_dir, child_pipe_fd, pipe_resource, gpus,
                    meta_extra=kwargs.get("meta_extra"),
                )
                info["image_dir"] = image_dir
                info["meta"] = meta
                state = "saved"
                log.info("  CRIU saved to %s (child killed by dump)", image_dir)
            except Exception as e:
                import traceback; traceback.print_exc()
                error = f"{type(e).__name__}: {e}"
            if error is None:
                # Deliberately outside the try above: the image is already on
                # disk by here, so a reap problem must degrade to "the ids may
                # still be taken" and never turn a good dump into a failed one.
                try:
                    _reap_dumped_child(child_proc, child_pid, log)
                    # Strictly after the child's own reap: the sweep waits on
                    # -1, so running it first would consume the status
                    # multiprocessing needs and leave child_proc.exitcode None
                    # forever.  The grandchildren it collects are what hold
                    # the leader's pgid and sid.
                    _reap_orphaned_descendants(log)
                except Exception as reap_exc:
                    log.warning("  reaping dumped tree failed: %s: %s",
                                type(reap_exc).__name__, reap_exc)
            elapsed = time.perf_counter() - t0
            log.info("<<< criu_dump %s (%.3fs)",
                     'OK' if error is None else 'FAILED', elapsed)
            _emit_result(cmd, elapsed, error, info)
            try:
                pipe.close()
            except OSError:
                pass
            break

        if cmd == "teardown":
            _drain_pipe_generates()
            t0 = time.perf_counter()
            error = None
            info = {}
            try:
                pipe.close()
                _kill_process_tree(child_pid)
                state = "removed"
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            elapsed = time.perf_counter() - t0
            log.info("<<< teardown %s (%.3fs)",
                     'OK' if error is None else 'FAILED', elapsed)
            _emit_result(cmd, elapsed, error, info)
            break

        if cmd == "generate":
            if cuda_broken:
                log.warning("<<< generate FAILED (cuda_broken: %s)",
                            cuda_broken_reason)
                _emit_result(cmd, 0.0,
                             f"skipped: {cuda_broken_reason}", {})
                continue
            try:
                pipe.send((cmd, kwargs))
                _pending_generates += 1
            except (BrokenPipeError, ConnectionResetError, EOFError):
                _alive = os.path.exists(f"/proc/{child_pid}")
                _diag = _diagnose_dead_child(child_pid)
                log.error("<<< generate FAILED (child pipe broken, "
                          "pid=%s alive=%s, %s)",
                          child_pid, _alive, _diag)
                _emit_result(cmd, 0.0, "child process died", {})
            continue

        if cuda_broken:
            log.warning("<<< %s FAILED (cuda_broken: %s)",
                        cmd, cuda_broken_reason)
            _emit_result(cmd, 0.0,
                         f"skipped: {cuda_broken_reason}", {})
            continue

        try:
            # `pause` is the command that must skip the drain even
            # while `_worker_paused` is False: its whole purpose is
            # to stop the very generates we'd otherwise be waiting
            # on.  `resume` arrives while already paused (no-op
            # drain), so its skip is redundant but harmless.
            if cmd not in ("pause", "resume"):
                _drain_pipe_generates()
            if cmd == "reinit_nccl" and "timeout_s" not in kwargs:
                # The child is a restored process, so its environ is the
                # dump's and nothing set on *this* job would be visible to it.
                # Resolve the budget here, where the environment is live, and
                # hand it over as a kwarg.
                kwargs = dict(kwargs, timeout_s=_reinit_timeout_s())
            pipe.send((cmd, kwargs))
            result = _recv_sync()
        except (BrokenPipeError, ConnectionResetError, EOFError) as _pipe_err:
            _alive = os.path.exists(f"/proc/{child_pid}")
            _diag = _diagnose_dead_child(child_pid)
            log.error("<<< %s FAILED (child pipe broken, "
                      "pid=%s alive=%s, %s)",
                      cmd, child_pid, _alive, _diag)
            _emit_result(cmd, 0.0, "child process died", {})
            _orphaned = 0
            while not child_queue.empty():
                try:
                    _ocmd, _okw = child_queue.get_nowait()
                    _emit_result(_ocmd, 0.0, "child process died", {})
                    _orphaned += 1
                except queue.Empty:
                    break
            if _orphaned:
                log.info("  flushed %d orphaned commands", _orphaned)
            break
        # Track child pause state so `_drain_pipe_generates` can no-op
        # for the duration of the pause/save/restore/resume cycle.
        _result_cmd, _, _result_error, _ = result
        if _result_error is None:
            if _result_cmd == "pause":
                _worker_paused = True
            elif _result_cmd == "resume":
                _worker_paused = False
        _emit_result(*result)

    log.info("child thread done")


# ---------------------------------------------------------------------------
# Reaping the dumped tree
# ---------------------------------------------------------------------------
#
# A destructive dump leaves the whole tree dead but *unreaped*, and an unreaped
# task holds pids two different ways.  Both have to be collected, because
# restore_and_wrap's cache-miss path restores the image it has just dumped a
# few seconds later and needs every recorded task id back, the leader's
# included.  Measured on the dev cluster: the restore began 4.8s after the dump and still
# could not have the leader's id, then exhausted its retry budget ~9.4s in.
#
# 1. Its own task id, while it is a zombie.  _reap_dumped_child covers this:
#    the dump worker is the child's parent, so it can waitpid() it directly.
#
# 2. The pid that names its process group and session -- which is the dumped
#    leader's, since the child setsid()s at startup.  A pid is a refcounted
#    struct pid in the namespace's IDR, and every member of a group or session
#    holds a reference on the id naming it, zombies included.  So the leader's
#    id can be unused by any task and still be unallocatable: clone3(set_tid),
#    which is how CRIU places a restored task, fails EEXIST on it while /proc
#    shows nothing there, because /proc lists tasks.  Those members are the
#    worker's *grand*children, so waitpid() cannot see them until
#    _set_child_subreaper makes us their reaper; _reap_orphaned_descendants
#    then collects them.
#
# Whether any of this is visible depends on what PID 1 is in the pod.  Under a
# shell (`bash -c '... sleep infinity'`, the research pods) PID 1 sits in
# waitpid(-1) and reaps anything reparented to it within milliseconds, so
# nobody noticed.  A DSS zone pod runs `dss-zone-worker` as PID 1 -- an
# ordinary application that never wait()s -- and there the corpses, and the
# references they hold, stay for the life of the pod.

# Bounded so a wedged child cannot strand the dump.  On expiry the ids are
# still taken and the restore's own collision preflight reports it, which is a
# better failure than hanging here.
_DUMP_REAP_TIMEOUT_S = 60.0

# The sweep's own bound, far shorter: by the time it runs the tree has already
# been SIGKILLed, so anything still alive is wedged rather than slow.
_ORPHAN_SWEEP_TIMEOUT_S = 10.0

# prctl(2).  Absent from the stdlib; Linux 3.4+, and needs no privilege.
_PR_SET_CHILD_SUBREAPER = 36


def _reap_dumped_child(child_proc, child_pid, log):
    """Reap the child ``criu dump`` just killed, releasing its task ids.

    ``waitpid`` on a thread group leader returns only once every thread in the
    group has exited, so this frees the whole recorded id set rather than just
    the leader's -- which is what the restore needs.
    """
    if child_proc is None:
        # A restored child was detached by criu and reparented away, so it is
        # not ours to reap.  Only the spawn path (the one a cache-miss dump
        # takes) has a Process handle here.
        return
    try:
        child_proc.join(timeout=_DUMP_REAP_TIMEOUT_S)
    except Exception as e:
        log.warning("  reaping dumped child %s failed: %s", child_pid, e)
        return
    if child_proc.exitcode is None:
        log.warning(
            "  dumped child %s not reapable after %.0fs; its task ids stay "
            "taken and may collide with the restore that follows",
            child_pid, _DUMP_REAP_TIMEOUT_S)
    else:
        log.info("  reaped dumped child %s (exit %s)", child_pid,
                 child_proc.exitcode)


def _set_child_subreaper(log):
    """Become the reaper for this worker's orphaned descendants.

    Without it an orphan reparents to PID 1, which in a DSS zone pod is an
    application that never ``wait()``s, so the corpse -- and the process-group
    and session references it holds on the dumped leader's pid -- outlive the
    pod's every restore.  As a subreaper we adopt those orphans instead, which
    is the only thing that brings them within ``waitpid``'s reach: they are the
    worker's grandchildren, so before this ``waitpid`` answers ``ECHILD``.

    Set once at worker start, before any child is spawned.  The flag is
    consulted when a descendant is orphaned, not when it is forked, so this
    covers the whole subtree for the life of the worker.

    Best-effort: ``prctl`` needs no privilege, but if it fails the cost is a
    sweep that finds nothing, which must never stop a worker from starting.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(),
                          os.strerror(ctypes.get_errno()))
    except (OSError, AttributeError) as e:
        log.warning("not a child subreaper (%s); descendants orphaned by a "
                    "dump will reparent to PID 1, and if it does not reap "
                    "them their pgid/sid references will hold the leader's "
                    "id against the restore", e)
        return False
    return True


def _reap_orphaned_descendants(log, timeout_s=_ORPHAN_SWEEP_TIMEOUT_S):
    """Collect every orphaned descendant, dropping the pgid/sid references
    that keep the dumped leader's id allocated.

    Pairs with ``_set_child_subreaper``: these tasks are the worker's
    grandchildren, adopted only because that flag is set.  Each reap drops a
    reference on the id naming their process group and session -- the dumped
    leader's -- and the last one frees it.

    Runs until ``waitpid`` reports no children left, since that is the only
    answer that means the tree is fully collected.  Bounded, because by the
    time this runs the tree has been SIGKILLed: a task still alive is wedged,
    and must not strand a dump whose image is already on disk.

    Returns the pids reaped.  Never raises.
    """
    reaped = []
    t0 = time.monotonic()
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break                      # no children left: fully collected
        except OSError as e:
            log.warning("  orphan sweep failed: %s", e)
            break
        if pid == 0:
            # Children exist but none has exited yet -- the tree is still on
            # its way down.  Poll rather than block: WNOHANG is the whole
            # point, so a wedged task costs the deadline and nothing more.
            if time.monotonic() - t0 >= timeout_s:
                log.warning(
                    "  orphan sweep gave up after %.0fs with a live "
                    "descendant left; it may still reference recorded ids",
                    timeout_s)
                break
            time.sleep(0.02)
            continue
        reaped.append(pid)
    if reaped:
        shown = ", ".join(str(p) for p in reaped[:12])
        if len(reaped) > 12:
            shown += ", ..."
        log.info("  reaped %d orphaned descendant(s): %s", len(reaped), shown)
    return reaped


# ---------------------------------------------------------------------------
# Worker main loop
# ---------------------------------------------------------------------------

def worker_loop(instance_id, gpus, cmd_queue, result_queue, completed_counter,
                model_dir=None):
    """Main loop for a per-Instance worker process.

    ``gpus`` is the physical GPU list for this instance (single-element at
    TP=1; a bare int is still accepted).  ``model_dir`` is threaded to the
    vLLM child so it can point its compile cache at
    ``<model_dir>/compilation``.

    Node identity for a multi-node TP group arrives as the ``init`` command's
    ``multinode`` kwarg and is handed to the child at spawn; see the ``init``
    branch below.
    """
    if isinstance(gpus, int):
        gpus = [gpus]
    gpus = list(gpus)
    rank = gpus[0]
    # Capture-node GPU UUIDs, hydrated from meta.json on the restore path.
    capture_gpu_uuids = None
    semip_logging.init_process(role="worker")
    log = semip_logging.worker(instance_id, rank)
    # Route everything this process emits (worker.N records, prints,
    # tracebacks) to the pod log, which the child it spawns inherits too.
    semip_logging.redirect_stdio_to_pod_log()

    child_pid = None
    child_proc = None
    child_queue = None
    child_thread_obj = None
    # Set when a restore stands up a private PID namespace for the child
    # tree; SIGKILLing its PID 1 at teardown destroys the namespace and
    # the whole restored tree in one shot.
    ns_holder = None

    # Fallback cleanup: teardown/exit clear ns_holder after killing it, so
    # this only fires when the worker dies abnormally (Ctrl-C, unhandled
    # exception).  The closure reads the *current* ns_holder at exit time.
    # SIGKILL can't run this; that case is covered by the stale-reaper
    # sweep in _worker_criu_load on the next restore of the same image.
    import atexit
    atexit.register(lambda: _kill_pidns_holder(ns_holder, log))

    # Before any child exists, so every descendant this worker ever orphans is
    # ours to reap rather than PID 1's to ignore.  See "Reaping the dumped
    # tree" above for why that decides whether a dump frees its ids.
    _set_child_subreaper(log)

    log.info("started")

    while True:
        cmd, kwargs = cmd_queue.get()
        log.info(">>> %s", cmd)

        if cmd == "init":
            from vllm_child import vllm_child_loop

            vllm_config = kwargs["vllm_config"]

            pipe_parent, pipe_child = mp.Pipe()

            spawn_ctx = mp.get_context("spawn")
            child_proc = spawn_ctx.Process(
                target=vllm_child_loop,
                args=(pipe_child, instance_id, list(gpus), model_dir,
                      kwargs.get("multinode")),
            )
            # ``multinode`` is a spawn argument rather than an ``init`` kwarg
            # alone because the child pins its NCCL/gloo interface, VLLM_HOST_IP
            # and the pinned aws-ofi-nccl values before it imports vLLM -- which
            # happens at module scope in the child, long before the ``init``
            # command is read off the pipe.
            # In unprivileged mode (SEMIP_UNPRIVILEGED=1), ask ONLY the spawned
            # child to drop its Linux capabilities (at its module import,
            # before torch) so the CRIU image records an empty cap set and
            # restores on low-capability nodes (restore_creds/capset succeeds).
            # Cap-drop has no separate flag: it is implied by the same switch
            # that adds --unprivileged to dump and picks the lowcap restore.
            # _SEMIP_CHILD_DROP_CAPS is an internal signal (leading underscore),
            # NOT a user flag: set it only across start() so it lands in the
            # child's environment.  The worker must not drop its own caps --
            # it imports vllm_child too, and it has to stay able to run criu
            # and signal the tree.
            _drop_caps = _unprivileged()
            _prev_child_flag = os.environ.get("_SEMIP_CHILD_DROP_CAPS")
            if _drop_caps:
                os.environ["_SEMIP_CHILD_DROP_CAPS"] = "1"
            try:
                child_proc.start()
            finally:
                if _prev_child_flag is None:
                    os.environ.pop("_SEMIP_CHILD_DROP_CAPS", None)
                else:
                    os.environ["_SEMIP_CHILD_DROP_CAPS"] = _prev_child_flag
            pipe_child.close()
            child_pid = child_proc.pid

            child_queue = queue.Queue()
            child_thread_obj = threading.Thread(
                target=_child_thread,
                args=(instance_id, list(gpus), child_pid, pipe_parent,
                      child_queue, result_queue, completed_counter),
                kwargs={"child_proc": child_proc},
                daemon=True,
            )
            child_thread_obj.start()

            child_queue.put((cmd, kwargs))
            continue

        if cmd == "criu_restore":
            t0 = time.perf_counter()
            error = None
            info = {}
            try:
                image_dir = kwargs["filename"]

                # Reduced-capability path (no private PID namespace, no
                # unshare) when SEMIP_UNPRIVILEGED=1 -- see
                # _worker_criu_load_lowcap.  Defaults to the namespace-based
                # path.
                _load_fn = (_worker_criu_load_lowcap
                            if _unprivileged()
                            else _worker_criu_load)

                max_retries = 5
                for _attempt in range(max_retries):
                    pipe_parent, pipe_child = mp.Pipe()
                    new_pipe_fd = pipe_child.fileno()
                    try:
                        new_pid, meta, _holder = _load_fn(
                            image_dir, new_pipe_fd)
                        _kill_pidns_holder(ns_holder, log)
                        ns_holder = _holder
                        pipe_child.close()
                        break
                    except RuntimeError as exc:
                        # Kill any orphaned process tree from the failed restore
                        _pipe_ino = os.readlink(f"/proc/self/fd/{new_pipe_fd}")
                        if _pipe_ino.startswith("socket:["):
                            _pipe_ino = _pipe_ino.split("[")[1].rstrip("]")
                        _orphan = _find_pid_by_pipe(_pipe_ino)
                        if _orphan:
                            log.warning("  killing orphan pid=%s from failed restore",
                                        _orphan)
                            _kill_process_tree(_orphan)
                        pipe_child.close()
                        pipe_parent.close()
                        # Matches a leader collision ("File exists"), a thread
                        # collision ("Unable to create a thread: -17") and the
                        # preflight's own marker alike.  Retries only rescue a
                        # short-lived holder (a zombie being reaped); a live
                        # process squatting on the range survives all five, and
                        # the final exception names it.
                        if (_is_pid_collision(exc)
                                and _attempt < max_retries - 1):
                            log.warning("  task-id collision, retrying (%d/%d)",
                                        _attempt + 1, max_retries)
                            time.sleep(0.5)
                            continue
                        raise

                child_pid = new_pid
                child_proc = None
                # Placement the image was captured on: the "old" side of any
                # subsequent migration.  Legacy images carry only "rank".
                old_gpus = meta.get("gpus") or [meta.get("rank", gpus[0])]
                old_gpus = list(old_gpus)
                old_rank = old_gpus[0]
                gpus = old_gpus
                # Capture-node GPU UUIDs; required to build the cuda-restore
                # device map when the restore node is not the capture node.
                capture_gpu_uuids = meta.get("gpu_uuids")

                child_queue = queue.Queue()
                child_thread_obj = threading.Thread(
                    target=_child_thread,
                    args=(instance_id, list(old_gpus), child_pid, pipe_parent,
                          child_queue, result_queue, completed_counter,
                          "checkpointed", capture_gpu_uuids),
                    daemon=True,
                )
                child_thread_obj.start()

                info["pid"] = new_pid
                info["rank"] = old_rank
                info["gpus"] = list(old_gpus)
                info["image_dir"] = image_dir
                log.info("  CRIU restored pid=%s (checkpointed, awaiting restore)",
                         new_pid)
            except Exception as e:
                import traceback; traceback.print_exc()
                error = f"{type(e).__name__}: {e}"
            elapsed = time.perf_counter() - t0
            log.info("<<< criu_restore %s (%.3fs)",
                     'OK' if error is None else 'FAILED', elapsed)
            result_queue.put(("criu_restore", elapsed, error, info))
            with completed_counter.get_lock():
                completed_counter.value += 1
            continue

        if cmd == "teardown":
            if child_queue is not None:
                child_queue.put(("teardown", {}))
            if child_thread_obj is not None:
                child_thread_obj.join(timeout=30)
            if child_proc is not None:
                child_proc.join(timeout=5)
                if child_proc.is_alive():
                    log.warning("child_proc still alive after join, force-killing")
                    _kill_process_tree(child_proc.pid)
                    child_proc.join(timeout=5)
            _kill_pidns_holder(ns_holder, log)
            ns_holder = None
            # Being a subreaper, we have adopted the corpses of the tree this
            # teardown just killed -- including a restored one, which criu
            # detached from us but whose descendants still reparent here.
            # Collect them while we still can: once this worker exits they go
            # to PID 1, and if it does not reap they hold their ids, and the
            # ids naming their group and session, for the life of the pod.
            # Shorter bound than the dump's: teardown latency is visible to
            # the orchestrator, and the tree has already been SIGKILLed.
            _reap_orphaned_descendants(log, timeout_s=5.0)
            break

        if cmd == "exit":
            if child_queue is not None:
                child_queue.put(("exit", {}))
            if child_thread_obj is not None:
                child_thread_obj.join(timeout=30)
            if child_proc is not None:
                child_proc.join(timeout=5)
                if child_proc.is_alive():
                    log.warning("child_proc still alive after join, force-killing")
                    _kill_process_tree(child_proc.pid)
                    child_proc.join(timeout=5)
            _kill_pidns_holder(ns_holder, log)
            ns_holder = None
            _reap_orphaned_descendants(log, timeout_s=5.0)
            result_queue.put(("exit", 0.0, None, {}))
            break

        if child_queue is not None:
            child_queue.put((cmd, kwargs))
        else:
            log.error("ERROR: no child for cmd=%s", cmd)
            result_queue.put((cmd, 0.0, f"no child initialized", {}))
            with completed_counter.get_lock():
                completed_counter.value += 1

    log.info("exiting")
