"""Route TP all-gathers through CustomAllreduce, so captured graphs hold no NCCL.

semi_p keeps its CUDA graphs across the CRIU dump only when they hold no NCCL
kernel. ``ncclCommAbort`` waits for every graph that captured one of the
comm's collectives, so a dump whose graphs hold one drops them before
``destroy_nccl`` and the restore recaptures (``_semip_nccl_graph_census``).
vLLM 0.32's ``CudaCommunicator.all_gather`` never uses CustomAllreduce: every
TP all-gather a graph captures is a PyNccl ``ncclDevKernel_AllGather_RING_LL``.
DeepSeek-V4.1-Flash holds 816 per rank (Engram's ``embed`` and ``wkv`` gathers,
two per Engram layer), and so recaptures 4692 graphs, ~85 s, on every restore.

``CustomAllreduce.custom_all_gather`` is the graph-capturable alternative
already in vLLM: kernel ``cross_device_all_gather``, with the 6-argument ABI
and the IPC staging buffer (``buffer_ptrs[rank]``, RankData slot 0) of the
copy-path all-reduce that semi_p forces, so ``ca_graph_rebind`` rebinds it the
same way. ``install`` makes ``CudaCommunicator.all_gather`` take it whenever
it can, and falls back to the original otherwise.

Installed from ``SemipGPUWorker.init_device`` and never withdrawn: the
restore's ``reinit_nccl`` builds a fresh CustomAllreduce whose limit has to be
widened too, and a recapture after a dropped dump has to capture the same
kernels the cold start did. The matched baseline runs the same worker class,
so its kernels match. ``SEMIP_CUSTOM_ALL_GATHER=0`` turns the routing off.
"""
from __future__ import annotations

import os

ENV = "SEMIP_CUSTOM_ALL_GATHER"

# The C++ dispatch in `CustomAllreduce::allgather` covers these and throws for
# any other world size (csrc/custom_all_gather_reduce_scatter.cu).
_WORLD_SIZES = (2, 4, 6, 8)

_state: dict = {"active": False, "orig_all_gather": None, "orig_init": None,
                "err": None}
_counts: dict = {"custom": 0, "fallback": {}, "capture_fallback": {},
                 "max_custom_bytes": 0, "widened": 0}


def enabled(environ=os.environ) -> bool:
    return environ.get(ENV, "1") != "0"


def widen_limit(ca) -> int:
    """Let ``custom_all_gather`` take whatever the staging buffer holds.

    The constructor sizes ``buffer_ptrs`` to max(max_size,
    max_all_gather_size, max_reduce_scatter_size) -- 16 MiB at the defaults --
    but admits only 2 MiB of all-gather input. Only the local shard is staged
    (the C++ bound is ``input_size <= reg_buffer_sz_bytes``), so raising the
    limit to the buffer size costs no memory. Returns the limit, 0 for a
    disabled CA."""
    if ca is None or getattr(ca, "disabled", True):
        return 0
    staged = max(int(ca.max_size), int(ca.max_all_gather_size),
                 int(ca.max_reduce_scatter_size))
    if int(ca.max_all_gather_size) < staged:
        ca.max_all_gather_size = staged
        _counts["widened"] += 1
    return int(ca.max_all_gather_size)


def decline_reason(comm, input_) -> str | None:
    """Why this all-gather cannot take the CustomAllreduce path, or None.

    Every rank of the group runs the same call with the same shape, so every
    rank reaches the same answer; a split decision would pair a CA barrier
    with an NCCL collective and hang."""
    if str(getattr(comm, "unique_name", "") or "").split(":")[0] != "tp":
        return "not_tp"
    ca = getattr(comm, "ca_comm", None)
    if ca is None or getattr(ca, "disabled", True):
        return "no_ca"
    # The multicast branch of custom_all_gather (MNNVL Lamport) bakes a
    # multicast mapping, which cuCheckpointProcess* cannot carry.
    if getattr(ca, "mnnvl_multicast_ptr", 0):
        return "multicast"
    if getattr(comm, "world_size", 0) not in _WORLD_SIZES:
        return "world_size"
    if not input_.is_contiguous():
        return "non_contiguous"
    if input_.numel() * input_.element_size() > int(ca.max_all_gather_size):
        return "size"
    return None


def gathered(out, input_size, dim, world_size):
    """Rank-major ``[world * n0, ...]`` output -> concatenated along ``dim``,
    exactly as the PyNccl branch of ``CudaCommunicator.all_gather`` does."""
    input_size = tuple(input_size)
    out = out.reshape((world_size,) + input_size)
    out = out.movedim(0, dim)
    return out.reshape(input_size[:dim] + (world_size * input_size[dim],)
                       + input_size[dim + 1:])


def _capturing() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001 - a counter must never fail a gather
        return False


def _bump(table, key):
    table[key] = table.get(key, 0) + 1


def route(comm, input_, dim):
    """The CustomAllreduce all-gather of ``input_`` along ``dim``, or None to
    send the caller to the original path."""
    why = decline_reason(comm, input_)
    if why is None:
        out = comm.ca_comm.custom_all_gather(input_)
        if out is not None:
            if dim < 0:
                dim += input_.dim()
            _counts["custom"] += 1
            nbytes = input_.numel() * input_.element_size()
            if nbytes > _counts["max_custom_bytes"]:
                _counts["max_custom_bytes"] = nbytes
            return gathered(out, input_.size(), dim, comm.world_size)
        why = "ca_refused"
    _bump(_counts["fallback"], why)
    # A fallback inside a capture is an NCCL kernel in that graph, which the
    # dump-time census will then count.
    if _capturing():
        _bump(_counts["capture_fallback"], why)
    return None


def _op_compiled(torch) -> bool:
    try:
        return hasattr(torch.ops._C_custom_ar, "custom_all_gather")
    except Exception:  # noqa: BLE001
        return False


def _widen_live():
    """Widen the CA ``init_device`` already built: vLLM constructs the TP
    communicator inside ``Worker.init_device``, before this patch exists."""
    try:
        from vllm.distributed import parallel_state as ps
    except Exception:  # noqa: BLE001
        return
    tp = getattr(ps, "_TP", None)
    comm = getattr(tp, "device_communicator", None) if tp is not None else None
    widen_limit(getattr(comm, "ca_comm", None))


def install(environ=os.environ) -> dict:
    """Patch ``CudaCommunicator.all_gather`` and ``CustomAllreduce.__init__``.
    Idempotent; returns ``state()``."""
    if _state["active"]:
        return state()
    if not enabled(environ):
        _state["err"] = f"off ({ENV}=0)"
        return state()
    try:
        import torch
        from vllm.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator)
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce)
    except Exception as exc:  # noqa: BLE001
        _state["err"] = f"import: {type(exc).__name__}: {exc}"
        return state()
    if not hasattr(CustomAllreduce, "custom_all_gather"):
        _state["err"] = "CustomAllreduce has no custom_all_gather"
        return state()
    if not _op_compiled(torch):
        _state["err"] = "_C_custom_ar.custom_all_gather is not in this build"
        return state()
    orig_ag = CudaCommunicator.all_gather
    orig_init = CustomAllreduce.__init__

    def _all_gather(self, input_, dim=-1):
        out = route(self, input_, dim)
        if out is not None:
            return out
        return orig_ag(self, input_, dim)

    def _init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        widen_limit(self)

    CudaCommunicator.all_gather = _all_gather
    CustomAllreduce.__init__ = _init
    _state.update(active=True, orig_all_gather=orig_ag, orig_init=orig_init,
                  err=None)
    _widen_live()
    return state()


def state() -> dict:
    """Live state and traffic. ``capture_fallback`` non-empty means some
    captured graph holds a non-CA all-gather."""
    return {"active": _state["active"], "err": _state["err"],
            "custom": _counts["custom"],
            "fallback": dict(_counts["fallback"]),
            "capture_fallback": dict(_counts["capture_fallback"]),
            "max_custom_bytes": _counts["max_custom_bytes"],
            "widened": _counts["widened"]}


def restore() -> None:
    if not _state["active"]:
        return
    try:
        from vllm.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator)
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce)
        CudaCommunicator.all_gather = _state["orig_all_gather"]
        CustomAllreduce.__init__ = _state["orig_init"]
    except Exception:  # noqa: BLE001
        pass
    _state.update(active=False, orig_all_gather=None, orig_init=None)
