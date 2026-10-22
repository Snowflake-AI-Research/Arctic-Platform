"""Park vLLM's multiproc-executor message queues before a dump; rebuild them after.

The executor drives its ranks through one broadcast ``MessageQueue`` (written
by the engine process) and one single-reader response queue per rank (written
by the rank). Each is a set of ZMQ sockets plus, between same-node peers, a
shared-memory ring. Across nodes those sockets are TCP endpoints bound to the
pod IP, and a restore onto another pod has a different IP: CRIU would re-bind
them to an address the new pod does not own. So the whole plane is closed on
both sides before the dump, and a fresh one is built and swapped in after the
restore -- local ends included, so the swap uses only public constructors.

A parked rank's busy loop blocks in ``_ParkedReader.dequeue``, which polls
``<unpark_dir>/rank<R>.in`` for the new broadcast handle and answers with its
new response handle in ``rank<R>.out``. Files are the channel because nothing
else is left: the sockets are gone, and a restored rank knows only what it knew
at dump time. That is also why the directory is fixed per image key, and why a
node may run only one restore of a given key at a time.
"""
import os
import pickle
import sys
import threading
import time

_POLL_S = 0.05
# A closing response writer may still hold the park ack in its send queue.
_FLUSH_LINGER_MS = 10_000
_ACK_TIMEOUT_S = 60.0


def unpark_dir_for(key):
    """The rendezvous directory for images stored under *key*."""
    return os.path.join("/dev/shm", f"semip-unpark-{key}")


def _publish(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as handle:
        pickle.dump(obj, handle)
    os.replace(tmp, path)


def _nap(seconds):
    """Sleep without an absolute deadline.

    A parked rank is dumped mid-poll and may be restored on a node whose
    CLOCK_MONOTONIC is days behind the dump node's. CRIU returns EINTR from the
    interrupted sleep, and time.sleep retries against its original absolute
    deadline -- which on that node is days away. A raw relative nanosleep that
    ignores EINTR just returns, and the caller polls again.
    """
    import ctypes
    ts = (ctypes.c_long * 2)(int(seconds), int((seconds % 1) * 1e9))
    _libc().nanosleep(ts, None)


_LIBC = []


def _libc():
    if not _LIBC:
        import ctypes
        _LIBC.append(ctypes.CDLL("libc.so.6", use_errno=True))
    return _LIBC[0]


def _take(path, timeout_s=None, cancelled=None):
    """Wait for *path*, return its unpickled content, and remove it."""
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    while True:
        try:
            with open(path, "rb") as handle:
                obj = pickle.load(handle)
            os.unlink(path)
            return obj
        except FileNotFoundError:
            pass
        if cancelled is not None and cancelled():
            raise SystemExit(f"shut down while waiting for {path}")
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(f"{path} did not appear within {timeout_s:.0f}s")
        _nap(_POLL_S)


def close_queue(mq, linger_ms=0):
    """Close every ZMQ socket and context a ``MessageQueue`` holds."""
    if mq is None:
        return
    spin = getattr(mq, "_spin_condition", None)
    sockets = [getattr(mq, "local_socket", None),
               getattr(mq, "remote_socket", None)]
    if spin is not None:
        sockets += [getattr(spin, name, None) for name in
                    ("local_notify_socket", "write_cancel_socket",
                     "read_cancel_socket")]
    contexts = []
    for sock in sockets:
        if sock is None or sock.closed:
            continue
        if all(sock.context is not ctx for ctx in contexts):
            contexts.append(sock.context)
        sock.close(linger=linger_ms)
    mq.local_socket = None
    mq.remote_socket = None
    mq._spin_condition = None
    buffer = getattr(mq, "buffer", None)
    if buffer is not None and getattr(buffer, "is_creator", False):
        # Before a dump, _prepare_worker_dump has already unlinked every
        # /dev/shm/psm_* segment, so the creator's unlink would fail.
        name = buffer.shared_memory.name.lstrip("/")
        if not os.path.exists(os.path.join("/dev/shm", name)):
            buffer.is_creator = False
    # Drops the ring; its creator unlinks the segment on collection.
    mq.buffer = None
    for ctx in contexts:
        ctx.term()


def _queue_params(mq):
    buffer = getattr(mq, "buffer", None)
    if buffer is None:
        return {}
    return {"max_chunk_bytes": buffer.max_chunk_bytes,
            "max_chunks": buffer.max_chunks}


# -- rank side ----------------------------------------------------------------

def _worker_proc():
    """The ``WorkerProc`` whose RPC dispatch is running the caller."""
    from vllm.v1.executor.multiproc_executor import WorkerProc
    frame = sys._getframe(1)
    while frame is not None:
        candidate = frame.f_locals.get("self")
        if isinstance(candidate, WorkerProc):
            return candidate
        frame = frame.f_back
    raise RuntimeError("not called from a vLLM WorkerProc RPC")


class _AckLatch:
    """Forwards to the response writer and records that the park ack left.

    The ack is enqueued after ``park_rank`` returns -- by the busy loop, or by
    the async-output thread under async scheduling -- so the writer may only be
    closed once this has fired.
    """

    def __init__(self, writer):
        self.writer = writer
        self.sent = threading.Event()

    def enqueue(self, *args, **kwargs):
        self.writer.enqueue(*args, **kwargs)
        self.sent.set()

    def __getattr__(self, name):
        return getattr(self.writer, name)


def _retarget_death_monitor(queues):
    """Point vLLM's parent-death monitor at the queues it should wake."""
    for thread in threading.enumerate():
        if thread.name != "DeathPipeMonitor":
            continue
        args = getattr(thread, "_args", None)
        if args and isinstance(args[0], list):
            args[0][:] = queues


class _ParkedReader:
    """Stands in for a rank's broadcast reader from park to unpark."""

    def __init__(self, proc, unpark_dir, latch):
        self._proc = proc
        self._old_reader = proc.rpc_broadcast_mq
        self._latch = latch
        self._dir = unpark_dir
        self._rank = proc.rank
        self._shutting_down = False

    def shutdown(self):
        self._shutting_down = True

    def _path(self, suffix):
        return os.path.join(self._dir, f"rank{self._rank}.{suffix}")

    def dequeue(self, *args, **kwargs):
        from vllm.distributed.device_communicators.shm_broadcast import (
            MessageQueue)
        proc = self._proc
        if not self._latch.sent.wait(_ACK_TIMEOUT_S):
            raise RuntimeError(
                f"rank {self._rank}: the park ack was never enqueued")
        params = _queue_params(self._latch.writer)
        close_queue(self._old_reader)
        close_queue(self._latch.writer, linger_ms=_FLUSH_LINGER_MS)
        self._old_reader = None
        proc.worker_response_mq = None
        _retarget_death_monitor([self])
        # On a follower node nothing else has created it: the executor's park
        # ran on the leader's pod.
        os.makedirs(self._dir, exist_ok=True)
        _publish(self._path("parked"), {"rank": self._rank,
                                        "pid": os.getpid()})

        order = _take(self._path("in"), cancelled=lambda: self._shutting_down)
        reader = MessageQueue.create_from_handle(order["broadcast"], self._rank)
        writer = MessageQueue(1, 0 if order["remote"] else 1,
                              connect_ip=order.get("connect_ip"), **params)
        _publish(self._path("out"), writer.export_handle())
        reader.wait_until_ready()
        writer.wait_until_ready()
        proc.rpc_broadcast_mq = reader
        proc.worker_response_mq = writer
        _retarget_death_monitor([reader, writer])
        return reader.dequeue(*args, **kwargs)


def park_rank(worker, unpark_dir):
    """``collective_rpc`` target: park this rank after it acks."""
    proc = _worker_proc()
    if isinstance(proc.rpc_broadcast_mq, _ParkedReader):
        raise RuntimeError(f"rank {proc.rank} is already parked")
    latch = _AckLatch(proc.worker_response_mq)
    proc.worker_response_mq = latch
    proc.rpc_broadcast_mq = _ParkedReader(proc, unpark_dir, latch)
    return {"rank": proc.rank, "pid": os.getpid()}


# -- executor side ------------------------------------------------------------

class _ParkedWriter:
    """Fails any RPC attempted while the plane is parked, instead of hanging."""

    def enqueue(self, *args, **kwargs):
        raise RuntimeError("the executor's message queues are parked; "
                           "unpark_mq must run before any collective_rpc")

    def shutdown(self):
        pass


def executor_of(llm):
    """The in-process ``MultiprocExecutor`` behind an ``LLM``."""
    core = getattr(llm.llm_engine.engine_core, "engine_core", None)
    executor = getattr(core, "model_executor", None)
    if executor is None or not hasattr(executor, "response_mqs"):
        raise RuntimeError(
            "no in-process MultiprocExecutor (park needs "
            "VLLM_ENABLE_V1_MULTIPROCESSING=0 and the mp backend)")
    return executor


def _wait_files(unpark_dir, ranks, suffix, timeout_s):
    return {rank: _take(os.path.join(unpark_dir, f"rank{rank}.{suffix}"),
                        timeout_s=timeout_s)
            for rank in ranks}


def close_executor_side(executor):
    """Close the executor's writer and response readers once every rank acked."""
    executor._semip_mq_params = _queue_params(executor.rpc_broadcast_mq)
    close_queue(executor.rpc_broadcast_mq)
    for mq in executor.response_mqs:
        close_queue(mq)
    executor.rpc_broadcast_mq = _ParkedWriter()
    executor.response_mqs = []


def park(llm, unpark_dir, rpc, ranks=None, timeout_s=120.0):
    """Park every rank, close the executor's ends, wait for *ranks* to report.

    *rpc* is the caller's bounded ``collective_rpc``. *ranks* are the ones on
    this node; all of them when ``None``.
    """
    executor = executor_of(llm)
    os.makedirs(unpark_dir, exist_ok=True)
    for name in os.listdir(unpark_dir):
        os.unlink(os.path.join(unpark_dir, name))
    acks = rpc(park_rank, (unpark_dir,))
    close_executor_side(executor)
    if ranks is None:
        ranks = range(executor.world_size)
    parked = _wait_files(unpark_dir, ranks, "parked", timeout_s)
    return {"acks": acks, "parked": sorted(parked)}


def begin_unpark(llm, remote_ranks=(), connect_ip=None):
    """Bind a new broadcast writer; return it and the handle every rank needs.

    *remote_ranks* read over TCP from *connect_ip* (this node's address) and
    write their responses over TCP; the rest use shared memory.
    """
    from vllm.distributed.device_communicators.shm_broadcast import (
        MessageQueue)
    executor = executor_of(llm)
    if not isinstance(executor.rpc_broadcast_mq, _ParkedWriter):
        raise RuntimeError("unpark called on a plane that is not parked")
    world = executor.world_size
    remote = sorted(set(remote_ranks))
    local = [rank for rank in range(world) if rank not in remote]
    writer = MessageQueue(world, len(local), local_reader_ranks=local,
                          connect_ip=connect_ip if remote else None,
                          **getattr(executor, "_semip_mq_params", {}))
    return writer, writer.export_handle()


def write_orders(unpark_dir, handle, ranks, remote_ranks=(), connect_ip=None):
    """Hand *ranks* (on this node) the new broadcast handle."""
    os.makedirs(unpark_dir, exist_ok=True)
    remote = set(remote_ranks)
    for rank in ranks:
        _publish(os.path.join(unpark_dir, f"rank{rank}.in"),
                 {"broadcast": handle, "remote": rank in remote,
                  "connect_ip": connect_ip if rank in remote else None})


def collect_handles(unpark_dir, ranks, timeout_s=120.0):
    """The new response handles of *ranks* (on this node)."""
    return _wait_files(unpark_dir, ranks, "out", timeout_s)


def finish_unpark(llm, writer, response_handles, timeout_s=120.0):
    """Connect to every rank's response writer, handshake, and swap in."""
    from vllm.distributed.device_communicators.shm_broadcast import (
        MessageQueue)
    executor = executor_of(llm)
    world = executor.world_size
    missing = [rank for rank in range(world) if rank not in response_handles]
    if missing:
        raise RuntimeError(f"no response handle from ranks {missing}")
    readers = [MessageQueue.create_from_handle(response_handles[rank], 0)
               for rank in range(world)]
    stage = {"at": "broadcast"}

    def _handshake():
        writer.wait_until_ready()
        for rank, reader in enumerate(readers):
            stage["at"] = f"response rank {rank}"
            reader.wait_until_ready()
        stage["at"] = "done"

    thread = threading.Thread(target=_handshake, daemon=True,
                              name="semip-unpark-handshake")
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise TimeoutError(f"unpark handshake stuck at {stage['at']} after "
                           f"{timeout_s:.0f}s")
    executor.rpc_broadcast_mq = writer
    executor.response_mqs = readers
    for proc in getattr(executor, "workers", []):
        proc.worker_response_mq = readers[proc.rank]
        peers = getattr(proc, "peer_worker_response_mqs", None)
        if isinstance(peers, list):
            for rank, peer in enumerate(peers):
                if peer is not None:
                    peers[rank] = readers[rank]
    return {"world": world, "remote": [rank for rank, reader in
                                       enumerate(readers)
                                       if reader._is_remote_reader]}


def unpark(llm, unpark_dir, remote_ranks=(), connect_ip=None, timeout_s=120.0):
    """Single-node unpark: every rank's files are in *unpark_dir*."""
    writer, handle = begin_unpark(llm, remote_ranks, connect_ip)
    world = executor_of(llm).world_size
    write_orders(unpark_dir, handle, range(world), remote_ranks, connect_ip)
    handles = collect_handles(unpark_dir, range(world), timeout_s)
    return finish_unpark(llm, writer, handles, timeout_s)
