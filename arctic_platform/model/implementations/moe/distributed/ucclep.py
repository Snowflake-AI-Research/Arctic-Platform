# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""UCCL-EP dispatch/combine via ``uccl.ep`` (not the DeepEP drop-in wrapper)."""

from dataclasses import dataclass

import torch
from torch.distributed import ProcessGroup

from arctic_platform.model.implementations.gpu import debug

from ..token_combine import sum_rows_by_token
from .uccl_ep import Buffer
from .uccl_ep import EventOverlap

_buffer: Buffer | None = None
_handle_cache: dict[int, tuple | None] = {}
_pending_dispatch_events: dict[int, EventOverlap] = {}
_handle_counter = 0
_pending_combine_events: list[EventOverlap] = []
_ucclep_cuda_ops_registered = False
_ucclep_cuda_lib: torch.library.Library | None = None


def _get_next_handle_id() -> torch.Tensor:
    global _handle_counter
    _handle_counter += 1
    return torch.tensor([_handle_counter], dtype=torch.int64, device="cpu")


def _new_event_overlap() -> EventOverlap:
    # DeepEP's no-argument EventHandle records at::cuda's current stream.
    # UCCL's no-argument constructor records CUDA stream 0, which lets the
    # communication stream race tensors produced on PyTorch's current stream.
    return Buffer.capture()


def register_ucclep_cuda_ops() -> None:
    global _ucclep_cuda_lib, _ucclep_cuda_ops_registered
    if _ucclep_cuda_ops_registered:
        return

    # Keep the Library alive so PyTorch does not deregister the custom ops.
    _ucclep_cuda_lib = torch.library.Library("ap_ucclep", "DEF")
    _ucclep_cuda_lib.define(
        "dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, "
        "Tensor num_tokens_per_rank, Tensor? num_tokens_per_rdma_rank, "
        "Tensor is_token_in_rank, Tensor num_tokens_per_expert) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
    )

    torch.library.impl(_ucclep_cuda_lib, "dispatch", "CUDA")(_dispatch_op_impl)

    torch.library.register_autograd("ap_ucclep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context)

    _ucclep_cuda_ops_registered = True


def _dispatch_op_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens_per_rank: torch.Tensor,
    num_tokens_per_rdma_rank: torch.Tensor | None,
    is_token_in_rank: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert _buffer is not None, "UCCL-EP buffer must be initialized before dispatch."

    previous_event = _new_event_overlap()
    recv_x, recv_indices, recv_scores, recv_num_tokens_per_expert_list, handle, after_event = _buffer.dispatch(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights.to(torch.float32),
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    handle_id = _get_next_handle_id()
    _handle_cache[handle_id.item()] = handle
    _pending_dispatch_events[handle_id.item()] = after_event
    recv_num_tokens_per_expert = torch.tensor(recv_num_tokens_per_expert_list, dtype=torch.int32, device="cpu")
    return recv_x, recv_indices, recv_scores, recv_num_tokens_per_expert, handle_id


def _dispatch_setup_context(ctx, inputs, output) -> None:
    x, *_ = inputs
    *_, handle_id = output
    ctx.input_dtype = x.dtype
    ctx.saved_handle = _handle_cache.get(handle_id.item())


def _dispatch_backward(
    ctx,
    grad_recv_x,
    grad_recv_indices,
    grad_recv_scores,
    grad_recv_num_tokens_per_expert,
    grad_handle_id,
):
    if grad_recv_x is None:
        return None, None, None, None, None, None, None

    handle = ctx.saved_handle
    assert handle is not None

    previous_event = _new_event_overlap()
    grad_x, grad_scores, after_event = _buffer.combine(
        x=grad_recv_x,
        handle=handle,
        topk_weights=grad_recv_scores.float() if grad_recv_scores is not None else None,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    after_event.current_stream_wait()

    grad_x = grad_x.to(ctx.input_dtype)
    grad_topk_weights = grad_scores.to(ctx.input_dtype) if grad_scores is not None else None
    return grad_x, None, grad_topk_weights, None, None, None, None


class _UcclEPCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, handle_id: torch.Tensor) -> torch.Tensor:
        assert _buffer is not None, "UCCL-EP buffer must be initialized before combine."
        handle = _handle_cache.pop(handle_id.item(), None)
        assert handle is not None, f"Handle not found for handle_id={handle_id.item()}"

        previous_event = _new_event_overlap()
        combined, _, after_event = _buffer.combine(
            x=x,
            handle=handle,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        # Keep every outstanding event alive until sync_combine(). Besides
        # deferring the stream wait like DeepEP, retaining all EventOverlap
        # objects preserves their event handles and extra tensor references
        # when token chunking queues more than one combine.
        _pending_combine_events.append(after_event)
        ctx.handle = handle
        return combined

    @staticmethod
    def backward(ctx, grad_combined: torch.Tensor) -> tuple[torch.Tensor, None]:
        assert _buffer is not None
        handle = ctx.handle
        assert handle is not None, "Handle not found in UCCL-EP combine backward."

        previous_event = _new_event_overlap()
        grad_x, _, _, _, _, after_event = _buffer.dispatch(
            x=grad_combined,
            topk_idx=None,
            topk_weights=None,
            num_tokens_per_rank=None,
            num_tokens_per_rdma_rank=None,
            is_token_in_rank=None,
            num_tokens_per_expert=None,
            handle=handle,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        after_event.current_stream_wait()
        return grad_x, None


@torch.compiler.disable()
def sync_combine() -> None:
    for pending_event in _pending_combine_events:
        pending_event.current_stream_wait()
    _pending_combine_events.clear()


@torch.compiler.disable()
def _sync_dispatch(handle_id: torch.Tensor | int) -> None:
    handle_key = handle_id if isinstance(handle_id, int) else handle_id.item()
    pending_event = _pending_dispatch_events.pop(handle_key, None)
    if pending_event is not None:
        pending_event.current_stream_wait()


def configure_num_sms(num_sms: int) -> None:
    """Set the number of SMs for UCCL-EP intranode dispatch/combine kernels.

    Must be called before the first dispatch/combine. Also determines
    internode RDMA channel count (num_channels = num_sms / 2).
    """
    Buffer.set_num_sms(num_sms)


def get_hidden_bytes(x: torch.Tensor) -> int:
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: ProcessGroup, hidden_bytes: int) -> Buffer:
    global _buffer

    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)

    return _buffer


def _permute_tokens(
    hidden_states: torch.Tensor,
    dispatched_indices: torch.Tensor,
    dispatched_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = dispatched_indices != -1
    valid_expert_ids = dispatched_indices[mask]
    valid_scores = dispatched_scores[mask]

    sort_order = torch.argsort(valid_expert_ids, stable=True)
    permuted_indices = torch.arange(len(hidden_states), device=hidden_states.device).repeat_interleave(
        mask.sum(dim=1)
    )[sort_order]
    permuted_hidden_states = hidden_states.index_select(0, permuted_indices)
    permuted_scores = valid_scores[sort_order]
    return permuted_hidden_states, permuted_scores, permuted_indices


def _unpermute_tokens(
    permuted_hidden_states: torch.Tensor,
    permuted_indices: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    return sum_rows_by_token(permuted_hidden_states, permuted_indices, num_tokens)


@dataclass
class _DispatchState:
    handle_id: torch.Tensor
    permuted_indices: torch.Tensor
    num_recv_tokens: int
    permuted_scores: torch.Tensor | None = None


@dataclass
class _PendingDispatchState:
    hidden_states: torch.Tensor
    dispatched_indices: torch.Tensor
    dispatched_scores: torch.Tensor
    num_tokens_per_expert: torch.Tensor
    handle_id: torch.Tensor
    score_before_experts: bool


def dispatch_tokens_async(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_experts: int,
    group: ProcessGroup,
    *,
    score_before_experts: bool = True,
) -> _PendingDispatchState:
    selected_experts_indices = selected_experts_indices.contiguous()
    top_scores = top_scores.contiguous()
    selected_experts_indices = selected_experts_indices.masked_fill(top_scores == 0, -1)
    if top_scores.dtype != torch.float32:
        top_scores = top_scores.float()

    buffer = get_buffer(group, get_hidden_bytes(hidden_states))
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert_dispatch, is_token_in_rank, _ = (
        buffer.get_dispatch_layout(topk_idx=selected_experts_indices, num_experts=num_experts)
    )

    hidden_states, dispatched_indices, dispatched_expert_scores, num_tokens_per_expert, handle_id = (
        torch.ops.ap_ucclep.dispatch(
            hidden_states,
            selected_experts_indices,
            top_scores,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            num_tokens_per_expert_dispatch,
        )
    )

    return _PendingDispatchState(
        hidden_states=hidden_states,
        dispatched_indices=dispatched_indices,
        dispatched_scores=dispatched_expert_scores,
        num_tokens_per_expert=num_tokens_per_expert,
        handle_id=handle_id,
        score_before_experts=score_before_experts,
    )


def finalize_dispatch_tokens(
    pending_state: _PendingDispatchState,
) -> tuple[torch.Tensor, torch.Tensor, _DispatchState]:
    _sync_dispatch(pending_state.handle_id)

    hidden_states = pending_state.hidden_states
    num_recv_tokens = hidden_states.shape[0]
    debug.maybe_record_recv_tokens(num_recv_tokens)
    hidden_states, permuted_scores, permuted_indices = _permute_tokens(
        hidden_states,
        pending_state.dispatched_indices,
        pending_state.dispatched_scores,
    )
    num_tokens_per_expert = pending_state.num_tokens_per_expert.to(hidden_states.device)

    if pending_state.score_before_experts and permuted_scores is not None:
        hidden_states = (hidden_states.to(torch.float32) * permuted_scores.to(torch.float32).reshape(-1, 1)).to(
            hidden_states.dtype
        )
        permuted_scores_for_state = None
    else:
        permuted_scores_for_state = permuted_scores

    state = _DispatchState(
        handle_id=pending_state.handle_id,
        permuted_indices=permuted_indices,
        num_recv_tokens=num_recv_tokens,
        permuted_scores=permuted_scores_for_state,
    )
    return hidden_states, num_tokens_per_expert, state


def combine_tokens(hidden_states: torch.Tensor, state: _DispatchState) -> torch.Tensor:
    if state.permuted_scores is not None:
        hidden_states = (hidden_states.to(torch.float32) * state.permuted_scores.to(torch.float32).reshape(-1, 1)).to(
            hidden_states.dtype
        )
    hidden_states = _unpermute_tokens(hidden_states, state.permuted_indices, state.num_recv_tokens)
    return _UcclEPCombine.apply(hidden_states, state.handle_id)


register_ucclep_cuda_ops()

__all__ = [
    "combine_tokens",
    "configure_num_sms",
    "dispatch_tokens_async",
    "finalize_dispatch_tokens",
    "sync_combine",
]
