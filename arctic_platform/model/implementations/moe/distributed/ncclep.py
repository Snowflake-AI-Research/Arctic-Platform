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

"""Expert dispatch/combine using differentiable NCCL all-to-all."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from ..routing_weights import apply_routing_weights
from ..token_combine import gather_token_rows, sum_rows_by_token


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, value: torch.Tensor, send_splits: list[int], recv_splits: list[int], group: dist.ProcessGroup
    ):
        ctx.send_splits, ctx.recv_splits, ctx.group = send_splits, recv_splits, group
        output = value.new_empty((sum(recv_splits), *value.shape[1:]))
        dist.all_to_all_single(output, value.contiguous(), recv_splits, send_splits, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor):
        output = gradient.new_empty((sum(ctx.send_splits), *gradient.shape[1:]))
        dist.all_to_all_single(output, gradient.contiguous(), ctx.send_splits, ctx.recv_splits, group=ctx.group)
        return output, None, None, None


@dataclass
class _PendingDispatchState:
    hidden_states: torch.Tensor
    indices: torch.Tensor
    scores: torch.Tensor
    counts: torch.Tensor
    sent_token_ids: torch.Tensor
    send_splits: list[int]
    recv_splits: list[int]
    num_input_tokens: int
    group: dist.ProcessGroup
    score_before_experts: bool


@dataclass
class _DispatchState:
    pending: _PendingDispatchState
    permuted_indices: torch.Tensor
    permuted_scores: torch.Tensor | None
    num_recv_tokens: int


def dispatch_tokens_async(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_experts: int,
    group: dist.ProcessGroup,
    *,
    score_before_experts: bool = True,
) -> _PendingDispatchState:
    world = dist.get_world_size(group)
    if num_experts % world:
        raise ValueError("Expert count must be divisible by EP size")
    local_experts = num_experts // world
    valid = (selected_experts_indices >= 0) & (top_scores != 0)
    destinations = selected_experts_indices.clamp_min(0) // local_experts
    destinations = destinations.masked_fill(~valid, 0)
    membership = torch.zeros(hidden_states.shape[0], world, dtype=torch.int32, device=hidden_states.device)
    membership.scatter_add_(1, destinations, valid.int())
    ranks, tokens = torch.where(membership.T > 0)
    counts = torch.bincount(ranks, minlength=world)
    received_counts = torch.empty_like(counts)
    dist.all_to_all_single(received_counts, counts, group=group)
    send_splits, recv_splits = counts.tolist(), received_counts.tolist()
    belongs = valid[tokens] & (destinations[tokens] == ranks[:, None])
    indices = (selected_experts_indices[tokens] % local_experts).masked_fill(~belongs, -1)
    scores = top_scores[tokens].masked_fill(~belongs, 0)
    received_hidden = _AllToAll.apply(gather_token_rows(hidden_states, tokens), send_splits, recv_splits, group)
    received_indices = indices.new_empty((sum(recv_splits), indices.shape[1]))
    dist.all_to_all_single(received_indices, indices.contiguous(), recv_splits, send_splits, group=group)
    received_scores = _AllToAll.apply(scores, send_splits, recv_splits, group)
    expert_counts = torch.bincount(received_indices[received_indices >= 0], minlength=local_experts).int()
    return _PendingDispatchState(
        received_hidden,
        received_indices,
        received_scores,
        expert_counts,
        tokens,
        send_splits,
        recv_splits,
        hidden_states.shape[0],
        group,
        score_before_experts,
    )


def finalize_dispatch_tokens(pending: _PendingDispatchState) -> tuple[torch.Tensor, torch.Tensor, _DispatchState]:
    mask = pending.indices >= 0
    order = torch.argsort(pending.indices[mask], stable=True)
    tokens = torch.arange(len(pending.hidden_states), device=pending.hidden_states.device)
    permuted_indices = tokens.repeat_interleave(mask.sum(dim=1))[order]
    hidden = gather_token_rows(pending.hidden_states, permuted_indices)
    scores = pending.scores[mask][order]
    if pending.score_before_experts:
        hidden = apply_routing_weights(hidden, scores)
        scores = None
    state = _DispatchState(pending, permuted_indices, scores, len(pending.hidden_states))
    return hidden, pending.counts, state


def combine_tokens(hidden_states: torch.Tensor, state: _DispatchState, replay=None) -> torch.Tensor:
    if replay is not None:
        raise NotImplementedError("NCCL dispatch does not support combine replay")
    if state.permuted_scores is not None:
        hidden_states = apply_routing_weights(hidden_states, state.permuted_scores)
    received = sum_rows_by_token(hidden_states, state.permuted_indices, state.num_recv_tokens)
    pending = state.pending
    if torch.is_grad_enabled():
        # Empty expert outputs still participate in both backward collectives.
        dependency = (
            pending.hidden_states.flatten()[:1].sum() * 0 + pending.scores.flatten()[:1].sum().to(received.dtype) * 0
        )
        received = received + dependency
    returned = _AllToAll.apply(received, pending.recv_splits, pending.send_splits, pending.group)
    return sum_rows_by_token(returned, pending.sent_token_ids, pending.num_input_tokens)


def sync_combine() -> None:
    pass


def configure_num_sms(num_sms: int) -> None:
    pass
