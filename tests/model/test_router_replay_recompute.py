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

"""CPU unit test for ``install_self_router_replay``: MoE routing stays deterministic across an
activation-checkpoint recompute. A tiny stub router is made deliberately non-deterministic across calls (a
per-call ramp models the last-bit topk drift); without replay a real ``torch.utils.checkpoint`` recompute
picks different experts than the forward, with replay ``topk`` runs once and the recompute reuses the captured
indices. No GPU, gateway, or 35B model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from arctic_platform.model.implementations.gpu.router_replay_recompute import _in_recompute
from arctic_platform.model.implementations.gpu.router_replay_recompute import _router_supports_routed_experts
from arctic_platform.model.implementations.gpu.router_replay_recompute import install_self_router_replay


class NonDeterministicRouter(nn.Module):
    """Stub matching ``TokenChoiceTopKRouter.forward(x, expert_bias=None, routed_experts=None) ->
    (weights, indices, aux)``. Its raw ``topk`` picks a different expert set on each call; with
    ``routed_experts`` it gathers instead. Each call records ``(in_recompute, routed_given, indices)``."""

    def __init__(self, hidden: int, num_experts: int, top_k: int):
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.top_k = top_k
        self.raw_topk_calls = 0
        self.calls: list[tuple[bool, bool, torch.Tensor]] = []

    def forward(self, x, expert_bias=None, routed_experts=None):
        scores = self.gate(x)  # [tokens, num_experts]
        if routed_experts is None:
            # A per-call ramp reorders topk every call, so forward and recompute diverge unless replayed.
            ramp = torch.arange(scores.shape[-1], dtype=scores.dtype) * float(self.raw_topk_calls + 1)
            self.raw_topk_calls += 1
            indices = torch.topk(scores + ramp, self.top_k, dim=-1).indices
        else:
            indices = routed_experts
        weights = torch.gather(torch.softmax(scores, dim=-1), -1, indices)
        self.calls.append((_in_recompute(), routed_experts is not None, indices.detach().clone()))
        return weights, indices, None


class MoEBlock(nn.Module):
    def __init__(self, hidden: int, num_experts: int, top_k: int):
        super().__init__()
        self.router = NonDeterministicRouter(hidden, num_experts, top_k)
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        weights, _indices, _ = self.router(x)
        return self.proj(x) * weights.sum(dim=-1, keepdim=True)


def _run_checkpointed(block: MoEBlock, tokens: int, hidden: int) -> None:
    torch.manual_seed(7)
    x = torch.randn(tokens, hidden, requires_grad=True)
    y = ckpt.checkpoint(block, x, use_reentrant=False)
    y.sum().backward()


def test_supports_routed_experts_detection():
    good = NonDeterministicRouter(8, 4, 2)
    assert _router_supports_routed_experts(good) is True

    class NoReplayRouter(nn.Module):
        def forward(self, x, expert_bias=None):  # no routed_experts parameter
            return x, None, None

    assert _router_supports_routed_experts(NoReplayRouter()) is False


def test_without_replay_recompute_diverges():
    """Control: the stub genuinely reproduces the non-deterministic-topk bug under a real checkpoint."""
    block = MoEBlock(hidden=16, num_experts=8, top_k=3)
    _run_checkpointed(block, tokens=12, hidden=16)

    # No replay installed -> topk runs on BOTH the forward and the recompute.
    assert block.router.raw_topk_calls == 2
    forward_idx, recompute_idx = block.router.calls[0][2], block.router.calls[1][2]
    assert not torch.equal(forward_idx, recompute_idx), "stub must diverge, else the replay test is vacuous"


def test_with_replay_recompute_is_deterministic():
    block = MoEBlock(hidden=16, num_experts=8, top_k=3)
    num_wrapped = install_self_router_replay(block)
    assert num_wrapped == 1

    _run_checkpointed(block, tokens=12, hidden=16)

    # Replay drives the graph through the gather path on both passes; topk is computed exactly once (on the
    # original forward, off the autograd graph) and never re-run on recompute.
    assert block.router.raw_topk_calls == 1

    gather_calls = [(in_recompute, idx) for (in_recompute, routed_given, idx) in block.router.calls if routed_given]
    forward_gathers = [idx for (in_recompute, idx) in gather_calls if not in_recompute]
    recompute_gathers = [idx for (in_recompute, idx) in gather_calls if in_recompute]

    # The recompute detector must have fired (proves the checkpoint recompute went through the wrapper), and
    # the recompute must gather the exact indices captured on the forward.
    assert forward_gathers, "expected a graph-building forward gather"
    assert recompute_gathers, "expected a recompute gather (detector did not fire)"
    torch.testing.assert_close(recompute_gathers[0], forward_gathers[0], rtol=0, atol=0)
