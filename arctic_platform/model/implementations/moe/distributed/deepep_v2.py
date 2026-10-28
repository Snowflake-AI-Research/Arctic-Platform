"""Expert-parallel dispatch/combine over DeepEP V2's buffer (``ep_comm_backend="deepep_v2"``).

Built for the AWS EFA ports of DeepEP V2 (akkart-aws ``NCCLOFI-2044-deepep-v2.5-migration``, ``EPBuffer``;
amazon-contributing ``main``, ``ElasticBuffer``), which carry the internode legs over NCCL GIN through aws-ofi-nccl,
so EP can span nodes on EFA. GIN needs GDRCopy (``/dev/gdrdrv`` and ``libgdrapi``) even on its CPU-proxy backend.

The dispatch uses V2's expanded layout: the received rows come grouped by local expert, one row per (token, expert)
pair, each with its routing weight, so there is no permutation here and the combine only sums. The buffer is sized
once for the largest per-rank token count of the first call (agreed over the EP group, as every rank must use the
same value); ``ARL_DEEPEP_V2_MAX_TOKENS`` sizes it explicitly. ``ARL_DEEPEP_V2_NUM_SMS`` overrides the configured SM
count, with 0 for the library's own choice.
"""

import filecmp
import glob
import os
from dataclasses import dataclass

import torch
from torch.distributed import ProcessGroup

from ..routing_weights import apply_routing_weights
from .combine_replay import CombineReplay


def _import_deep_ep():
    """``deep_ep`` asserts at import that one NCCL runtime is mapped, but it counts every mapped ``libnccl*`` file,
    so aws-ofi-nccl's plugins (``libnccl-net-ofi.so``, ``libnccl-tuner-ofi.so``), which NCCL has loaded by the time a
    trainer imports this module, fail it. Skip that check and run it here on the runtime libraries alone."""
    suppressed = os.environ.get("EP_SUPPRESS_NCCL_CHECK")
    os.environ["EP_SUPPRESS_NCCL_CHECK"] = "1"
    try:
        import deep_ep
        from deep_ep.utils.find_pkgs import find_nccl_root
    finally:
        if suppressed is None:
            del os.environ["EP_SUPPRESS_NCCL_CHECK"]
        else:
            os.environ["EP_SUPPRESS_NCCL_CHECK"] = suppressed
    if int(suppressed or 0):
        return deep_ep
    with open("/proc/self/maps") as maps:
        loaded = {line.split()[-1] for line in maps if os.path.basename(line.split()[-1]).startswith("libnccl.so")}
    if len(loaded) > 1:
        raise RuntimeError(f"more than one NCCL runtime is mapped: {sorted(loaded)}")
    linked = sorted(glob.glob(f"{find_nccl_root()}/lib/libnccl.so*"))
    if loaded and not (linked and filecmp.cmp(next(iter(loaded)), linked[0], shallow=False)):
        raise RuntimeError(f"the mapped NCCL runtime {sorted(loaded)} is not the one deep_ep was linked with {linked}")
    return deep_ep


deep_ep = _import_deep_ep()
_BufferType = getattr(deep_ep, "EPBuffer", None) or deep_ep.ElasticBuffer
_buffers: dict[tuple, object] = {}
_num_sms = 0


@dataclass
class _Handle:
    """The dispatch handle, passed from the dispatch to its combine (and both backwards) outside autograd."""

    value: object = None


def configure_num_sms(num_sms: int) -> None:
    global _num_sms
    _num_sms = int(os.environ.get("ARL_DEEPEP_V2_NUM_SMS", num_sms))


def _get_buffer(group: ProcessGroup, num_tokens: int, hidden: int, num_topk: int):
    key = (id(group), hidden, num_topk)
    entry = _buffers.get(key)
    if entry is None:
        # The buffer reuses torch's NCCL communicator for the group, which torch creates at its first collective.
        max_tokens = torch.tensor([int(os.environ.get("ARL_DEEPEP_V2_MAX_TOKENS", num_tokens))], device="cuda")
        torch.distributed.all_reduce(max_tokens, op=torch.distributed.ReduceOp.MAX, group=group)
        max_tokens = int(max_tokens.item())
        buffer = _BufferType(group, num_max_tokens_per_rank=max_tokens, hidden=hidden, num_topk=num_topk)
        entry = _buffers[key] = (buffer, max_tokens)
    buffer, max_tokens = entry
    if num_tokens > max_tokens:
        raise ValueError(
            f"DeepEP V2 buffer holds {max_tokens} tokens per rank, this call has {num_tokens}; set "
            "ARL_DEEPEP_V2_MAX_TOKENS to the largest per-rank token count of any MoE call"
        )
    return buffer


class _Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, topk_idx, topk_weights, buffer, num_experts, handle: _Handle):
        recv_x, _, recv_weights, value, _ = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=num_experts,
            expert_alignment=1,
            num_sms=_num_sms,
            do_expand=True,
        )
        handle.value = value
        ctx.buffer, ctx.handle, ctx.input_dtype, ctx.weight_dtype = buffer, value, x.dtype, topk_weights.dtype
        return recv_x, recv_weights.reshape(-1)

    @staticmethod
    def backward(ctx, grad_recv_x, grad_recv_weights):
        if grad_recv_x is None:
            return None, None, None, None, None, None
        weights = None if grad_recv_weights is None else grad_recv_weights.float().contiguous()
        grad_x, grad_weights, _ = ctx.buffer.combine(grad_recv_x.contiguous(), ctx.handle, topk_weights=weights)
        grad_weights = None if grad_weights is None else grad_weights.to(ctx.weight_dtype)
        return grad_x.to(ctx.input_dtype), None, grad_weights, None, None, None


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, buffer, handle: _Handle):
        combined, _, _ = buffer.combine(x.contiguous(), handle.value)
        ctx.buffer, ctx.handle = buffer, handle.value
        return combined

    @staticmethod
    def backward(ctx, grad_combined):
        handle = ctx.handle
        # A cached dispatch infers its SM count from ``num_experts`` before the handle, so pass the forward's.
        grad_x = ctx.buffer.dispatch(grad_combined.contiguous(), handle=handle, num_sms=handle.num_sms, do_expand=True)[0]
        return grad_x, None, None


class _ReplayedCombine(torch.autograd.Function):
    """A recompute's combine, replayed from the original forward's output (``CombineReplay``). The recompute's
    graph is discarded once the saved tensors are rebuilt, so this node never runs backward."""

    @staticmethod
    def forward(ctx, x, combined):
        return combined.clone()

    @staticmethod
    def backward(ctx, grad_combined):
        raise RuntimeError("a replayed DeepEP V2 combine is only part of a recompute and has no backward")


@dataclass
class _DispatchState:
    buffer: object
    handle: _Handle
    permuted_scores: torch.Tensor | None = None


@dataclass
class _PendingDispatchState:
    hidden_states: torch.Tensor
    scores: torch.Tensor
    num_tokens_per_expert: torch.Tensor
    buffer: object
    handle: _Handle
    score_before_experts: bool


@torch.compiler.disable()
def dispatch_tokens_async(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_experts: int,
    group: ProcessGroup,
    *,
    score_before_experts: bool = True,
) -> _PendingDispatchState:
    top_scores = top_scores.contiguous()
    topk_idx = selected_experts_indices.masked_fill(top_scores == 0, -1).to(deep_ep.topk_idx_t).contiguous()
    buffer = _get_buffer(group, hidden_states.shape[0], hidden_states.shape[1], topk_idx.shape[1])
    handle = _Handle()
    recv_x, recv_scores = _Dispatch.apply(
        hidden_states.contiguous(), topk_idx, top_scores.float(), buffer, num_experts, handle
    )
    counts = handle.value.num_recv_tokens_per_expert_list
    num_rows = sum(counts)
    if recv_x.shape[0] != num_rows:
        raise RuntimeError(f"DeepEP V2 expanded dispatch returned {recv_x.shape[0]} rows for {num_rows} expert slots")
    return _PendingDispatchState(
        hidden_states=recv_x,
        scores=recv_scores,
        num_tokens_per_expert=torch.tensor(counts, dtype=torch.int32),
        buffer=buffer,
        handle=handle,
        score_before_experts=score_before_experts,
    )


def finalize_dispatch_tokens(
    pending_state: _PendingDispatchState,
) -> tuple[torch.Tensor, torch.Tensor, _DispatchState]:
    hidden_states = pending_state.hidden_states
    scores = pending_state.scores
    if pending_state.score_before_experts:
        hidden_states = apply_routing_weights(hidden_states, scores)
        scores = None
    state = _DispatchState(buffer=pending_state.buffer, handle=pending_state.handle, permuted_scores=scores)
    return hidden_states, pending_state.num_tokens_per_expert.to(hidden_states.device), state


@torch.compiler.disable()
def combine_tokens(
    hidden_states: torch.Tensor, state: _DispatchState, replay: CombineReplay | None = None
) -> torch.Tensor:
    if state.permuted_scores is not None:
        hidden_states = apply_routing_weights(hidden_states, state.permuted_scores)
    if replay is None:
        return _Combine.apply(hidden_states, state.buffer, state.handle)
    return replay.run(
        lambda: _Combine.apply(hidden_states, state.buffer, state.handle),
        lambda combined: _ReplayedCombine.apply(hidden_states, combined),
    )


def sync_combine() -> None:
    """Dispatch and combine finish on the current stream before they return; nothing is pending."""


__all__ = [
    "combine_tokens",
    "configure_num_sms",
    "dispatch_tokens_async",
    "finalize_dispatch_tokens",
    "sync_combine",
]
