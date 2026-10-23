from dataclasses import dataclass

import deep_ep
import torch
from deep_ep import EPBuffer, EPHandle, EventOverlap
from torch.distributed import ProcessGroup, ReduceOp

_buffer: EPBuffer | None = None
_buffer_group: ProcessGroup | None = None
_buffer_hidden = 0
_buffer_topk = 0
_num_sms = 0
_handle_cache: dict[int, EPHandle] = {}
_pending_dispatch_events: dict[int, EventOverlap] = {}
_handle_counter = 0
_pending_combine_event: EventOverlap | None = None
_deepep_cuda_ops_registered = False
_deepep_cuda_lib: torch.library.Library | None = None


def configure_num_sms(num_sms: int) -> None:
    """Remember the SM count; EPBuffer takes it on each operation."""
    global _num_sms
    _num_sms = num_sms


def _get_next_handle_id() -> torch.Tensor:
    global _handle_counter
    _handle_counter += 1
    return torch.tensor([_handle_counter], dtype=torch.int64, device="cpu")


def _ensure_buffer(
    group: ProcessGroup,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    device: torch.device,
) -> EPBuffer:
    global _buffer, _buffer_group, _buffer_hidden, _buffer_topk

    local_tokens = torch.tensor([num_tokens], dtype=torch.int64, device=device)
    torch.distributed.all_reduce(local_tokens, op=ReduceOp.MAX, group=group)
    needed = int(local_tokens.item())
    if (
        _buffer is not None
        and _buffer_group == group
        and _buffer_hidden == hidden
        and _buffer_topk == num_topk
        and _buffer.num_max_tokens_per_rank >= needed
    ):
        return _buffer

    _buffer = EPBuffer(
        group,
        num_max_tokens_per_rank=needed,
        hidden=hidden,
        num_topk=num_topk,
        prefer_overlap_with_compute=True,
    )
    _buffer_group = group
    _buffer_hidden = hidden
    _buffer_topk = num_topk
    return _buffer


def _expand_counts(handle: EPHandle, device: torch.device) -> torch.Tensor:
    """Return packed expert row counts on the grouped-GEMM device."""
    if handle.num_recv_tokens_per_expert_list:
        return torch.tensor(
            handle.num_recv_tokens_per_expert_list,
            dtype=torch.int32,
            device=device,
        )

    psum = handle.psum_num_recv_tokens_per_expert
    alignment = int(handle.expert_alignment)
    previous = torch.cat((psum.new_zeros(1), psum[:-1]))
    if alignment != 1:
        previous = (previous + alignment - 1) // alignment * alignment
    return (psum - previous).to(device=device, dtype=torch.int32)


def _apply_scores(hidden: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    return (hidden.float() * scores.unsqueeze(-1)).to(hidden.dtype)


@dataclass
class _DispatchState:
    handle_id: torch.Tensor
    dispatched_scores: torch.Tensor
    apply_score_after_experts: bool


@dataclass
class _PendingDispatchState:
    hidden_states: torch.Tensor
    dispatched_scores: torch.Tensor
    handle_id: torch.Tensor
    score_before_experts: bool


def _launch_dispatch(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, EPHandle, EventOverlap]:
    assert _buffer is not None
    recv_x, _, recv_scores, handle, after_event = _buffer.dispatch(
        hidden_states,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=_buffer.num_max_tokens_per_rank,
        expert_alignment=1,
        num_sms=_num_sms,
        previous_event=_buffer.capture(),
        async_with_compute_stream=True,
        allocate_on_comm_stream=True,
        do_cpu_sync=True,
        do_expand=True,
    )
    if recv_scores is None:
        recv_scores = hidden_states.new_empty(recv_x.shape[0], dtype=torch.float32)
    return recv_x, recv_scores, handle, after_event


def _dispatch_op_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recv_x, recv_scores, handle, after_event = _launch_dispatch(
        x,
        topk_idx,
        topk_weights,
        num_experts=num_experts,
    )
    handle_id = _get_next_handle_id()
    handle_key = handle_id.item()
    _handle_cache[handle_key] = handle
    _pending_dispatch_events[handle_key] = after_event
    return recv_x, recv_scores, handle_id


def _dispatch_setup_context(ctx, inputs, output) -> None:
    x, *_ = inputs
    *_, handle_id = output
    ctx.input_dtype = x.dtype
    ctx.saved_handle = _handle_cache.get(handle_id.item())


def _dispatch_backward(ctx, grad_recv_x, grad_recv_scores, grad_handle_id):
    if grad_recv_x is None:
        return None, None, None, None

    assert _buffer is not None
    handle = ctx.saved_handle
    assert handle is not None
    grad_x, grad_scores, after_event = _buffer.combine(
        grad_recv_x,
        handle=handle,
        topk_weights=grad_recv_scores.float() if grad_recv_scores is not None else None,
        num_sms=_num_sms,
        previous_event=_buffer.capture(),
        async_with_compute_stream=True,
        allocate_on_comm_stream=True,
    )
    after_event.current_stream_wait()
    grad_topk_weights = grad_scores.to(ctx.input_dtype) if grad_scores is not None else None
    return grad_x.to(ctx.input_dtype), None, grad_topk_weights, None


def register_deepep_cuda_ops() -> None:
    global _deepep_cuda_lib, _deepep_cuda_ops_registered
    if _deepep_cuda_ops_registered:
        return

    _deepep_cuda_lib = torch.library.Library("ap_deepep_v2", "DEF")
    _deepep_cuda_lib.define(
        "dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, int num_experts) "
        "-> (Tensor, Tensor, Tensor)"
    )
    torch.library.impl(_deepep_cuda_lib, "dispatch", "CUDA")(_dispatch_op_impl)
    torch.library.register_autograd(
        "ap_deepep_v2::dispatch",
        _dispatch_backward,
        setup_context=_dispatch_setup_context,
    )
    _deepep_cuda_ops_registered = True


class _DeepEPCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, handle_id: torch.Tensor) -> torch.Tensor:
        global _pending_combine_event

        assert _buffer is not None
        handle = _handle_cache.pop(handle_id.item(), None)
        assert handle is not None, f"Handle not found for handle_id={handle_id.item()}"
        combined, _, after_event = _buffer.combine(
            x,
            handle=handle,
            num_sms=_num_sms,
            previous_event=_buffer.capture(),
            async_with_compute_stream=True,
            allocate_on_comm_stream=True,
        )
        _pending_combine_event = after_event
        ctx.handle = handle
        return combined

    @staticmethod
    def backward(ctx, grad_combined: torch.Tensor) -> tuple[torch.Tensor, None]:
        assert _buffer is not None
        grad_x, _, _, _, after_event = _buffer.dispatch(
            grad_combined,
            handle=ctx.handle,
            num_sms=_num_sms,
            previous_event=_buffer.capture(),
            async_with_compute_stream=True,
            allocate_on_comm_stream=True,
            do_expand=True,
            do_zero_padding=True,
        )
        after_event.current_stream_wait()
        return grad_x, None


@torch.compiler.disable()
def sync_combine() -> None:
    global _pending_combine_event

    if _pending_combine_event is not None:
        _pending_combine_event.current_stream_wait()
        _pending_combine_event = None


@torch.compiler.disable()
def _sync_dispatch(handle_id: torch.Tensor) -> None:
    pending_event = _pending_dispatch_events.pop(handle_id.item(), None)
    if pending_event is not None:
        pending_event.current_stream_wait()


def _prepare_routing(
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_experts_indices = selected_experts_indices.contiguous()
    top_scores = top_scores.contiguous()
    selected_experts_indices = selected_experts_indices.masked_fill(top_scores == 0, -1)
    if selected_experts_indices.dtype != deep_ep.topk_idx_t:
        selected_experts_indices = selected_experts_indices.to(deep_ep.topk_idx_t)
    return selected_experts_indices, top_scores.float()


def dispatch_tokens_async(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_experts: int,
    group: ProcessGroup,
    *,
    score_before_experts: bool = True,
) -> _PendingDispatchState:
    selected_experts_indices, top_scores = _prepare_routing(selected_experts_indices, top_scores)
    _ensure_buffer(
        group,
        hidden_states.shape[0],
        hidden_states.shape[1],
        top_scores.shape[1],
        hidden_states.device,
    )
    hidden_states, dispatched_scores, handle_id = torch.ops.ap_deepep_v2.dispatch(
        hidden_states,
        selected_experts_indices,
        top_scores,
        num_experts,
    )
    return _PendingDispatchState(
        hidden_states=hidden_states,
        dispatched_scores=dispatched_scores,
        handle_id=handle_id,
        score_before_experts=score_before_experts,
    )


def finalize_dispatch_tokens(
    pending_state: _PendingDispatchState,
) -> tuple[torch.Tensor, torch.Tensor, _DispatchState]:
    _sync_dispatch(pending_state.handle_id)
    handle = _handle_cache[pending_state.handle_id.item()]
    current_stream = torch.cuda.current_stream(pending_state.hidden_states.device)
    pending_state.hidden_states.record_stream(current_stream)
    pending_state.dispatched_scores.record_stream(current_stream)
    counts = _expand_counts(handle, pending_state.hidden_states.device)
    hidden_states = pending_state.hidden_states
    if pending_state.score_before_experts:
        hidden_states = _apply_scores(hidden_states, pending_state.dispatched_scores)
    state = _DispatchState(
        handle_id=pending_state.handle_id,
        dispatched_scores=pending_state.dispatched_scores,
        apply_score_after_experts=not pending_state.score_before_experts,
    )
    return hidden_states, counts, state


def combine_tokens(hidden_states: torch.Tensor, state: _DispatchState) -> torch.Tensor:
    if state.apply_score_after_experts:
        hidden_states = _apply_scores(hidden_states, state.dispatched_scores)
    return _DeepEPCombine.apply(hidden_states, state.handle_id)


register_deepep_cuda_ops()

__all__ = [
    "combine_tokens",
    "configure_num_sms",
    "dispatch_tokens_async",
    "finalize_dispatch_tokens",
    "sync_combine",
]
