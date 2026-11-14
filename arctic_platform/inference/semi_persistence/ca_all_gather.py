"""Route TP all-gathers through CustomAllreduce, so captured graphs hold no NCCL.

semi_p keeps its CUDA graphs across the CRIU dump only when they hold no NCCL
kernel. ``ncclCommAbort`` waits for every graph that captured one of the
comm's collectives, so a dump whose graphs hold one drops them before
``destroy_nccl`` and the restore recaptures (``_semip_nccl_graph_census``).
vLLM 0.32's ``CudaCommunicator.all_gather`` never uses CustomAllreduce: every
TP all-gather a graph captures is a PyNccl ``ncclDevKernel_AllGather_RING_LL``.
DeepSeek-V4.1-Flash holds 816 per rank (Engram's ``embed`` and ``wkv`` gathers,
two per Engram layer), and so recaptures 4692 graphs, ~85 s, on every restore.

CustomAllreduce already has a graph-capturable all-gather, with two kernels.
The classic ``cross_device_all_gather`` has the 6-argument ABI of the
copy-path all-reduce that semi_p forces and reads only the IPC staging buffer
(``buffer_ptrs[rank]``, RankData slot 0) and the signal pads, which
``ca_graph_rebind`` already rewrites. The multicast
``mnnvl_lamport_all_gather`` bakes the multicast address and the epoch buffer,
which the rebind cannot patch, and ``CustomAllreduce.custom_all_gather`` takes
it whenever the CA has a multicast buffer -- which ``_init_mnnvl_buffer`` gets
even on a single NVSwitch node (H200). So the router launches the classic op
itself, under vLLM's gate for that branch, and never the multicast one.
``install`` makes ``CudaCommunicator.all_gather`` take it whenever it can, and
falls back to the original otherwise.

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
                "ops": None, "err": None}
_counts: dict = {"custom": 0, "fallback": {}, "capture_fallback": {},
                 "max_custom_bytes": 0, "widened": 0, "multicast_present": 0}


def enabled(environ=os.environ) -> bool:
    return environ.get(ENV, "1") != "0"


def widen_limit(ca) -> int:
    """Let the classic all-gather take whatever the staging buffer holds.

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


def _adopt(ca) -> None:
    """Widen a CA the router will use and count it if it has a multicast
    buffer."""
    widen_limit(ca)
    if (ca is not None and not getattr(ca, "disabled", True)
            and getattr(ca, "mnnvl_multicast_ptr", 0)):
        _counts["multicast_present"] += 1


def decline_reason(comm, input_) -> str | None:
    """Why this all-gather cannot take the classic CustomAllreduce kernel, or
    None.

    The gates after ``world_size`` are the non-multicast branch of vLLM's
    ``should_custom_all_gather``. Every rank of the group runs the same call
    with the same shape, so every rank reaches the same answer; a split
    decision would pair a CA barrier with an NCCL collective and hang."""
    import torch
    if str(getattr(comm, "unique_name", "") or "").split(":")[0] != "tp":
        return "not_tp"
    ca = getattr(comm, "ca_comm", None)
    if ca is None or getattr(ca, "disabled", True):
        return "no_ca"
    if getattr(comm, "world_size", 0) not in _WORLD_SIZES:
        return "world_size"
    if not getattr(ca, "fully_connected", False):
        return "not_fully_connected"
    if input_.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return "dtype"
    if not input_.is_contiguous():
        return "non_contiguous"
    nbytes = input_.numel() * input_.element_size()
    if not 0 < nbytes <= int(ca.max_all_gather_size):
        return "size"
    if nbytes % 16:
        return "unaligned"
    return None


def _classic_all_gather(ca, input_):
    """The non-multicast branch of ``CustomAllreduce.custom_all_gather``:
    rank-major ``[world * n0, ...]``."""
    import torch
    out = torch.empty((input_.shape[0] * ca.world_size,)
                      + tuple(input_.shape[1:]),
                      dtype=input_.dtype, device=input_.device)
    _state["ops"].custom_all_gather(ca._ptr, input_, out,
                                    ca.buffer_ptrs[ca.rank],
                                    int(ca.max_all_gather_size))
    return out


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
        out = _classic_all_gather(comm.ca_comm, input_)
        if dim < 0:
            dim += input_.dim()
        _counts["custom"] += 1
        nbytes = input_.numel() * input_.element_size()
        if nbytes > _counts["max_custom_bytes"]:
            _counts["max_custom_bytes"] = nbytes
        return gathered(out, input_.size(), dim, comm.world_size)
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
    _adopt(getattr(comm, "ca_comm", None))


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
        from vllm import _custom_ops as ops
        from vllm.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator)
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce)
    except Exception as exc:  # noqa: BLE001
        _state["err"] = f"import: {type(exc).__name__}: {exc}"
        return state()
    if not hasattr(ops, "custom_all_gather"):
        _state["err"] = "vllm._custom_ops has no custom_all_gather"
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
        _adopt(self)

    _state["ops"] = ops
    CudaCommunicator.all_gather = _all_gather
    CustomAllreduce.__init__ = _init
    _state.update(active=True, orig_all_gather=orig_ag, orig_init=orig_init,
                  err=None)
    _widen_live()
    return state()


def state() -> dict:
    """Live state and traffic. ``capture_fallback`` non-empty means some
    captured graph holds a non-CA all-gather. ``multicast_present`` counts the
    live CAs that came with a multicast buffer; the router never uses it."""
    return {"active": _state["active"], "err": _state["err"],
            "custom": _counts["custom"],
            "fallback": dict(_counts["fallback"]),
            "capture_fallback": dict(_counts["capture_fallback"]),
            "max_custom_bytes": _counts["max_custom_bytes"],
            "widened": _counts["widened"],
            "multicast_present": _counts["multicast_present"]}


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
    _state.update(active=False, orig_all_gather=None, orig_init=None,
                  ops=None)
