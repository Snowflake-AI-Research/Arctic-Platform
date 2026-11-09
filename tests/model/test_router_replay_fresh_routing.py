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

"""Router replay with fresh tokens on CPU: ``TokenChoiceTopKRouter`` and the recompute wrapper.

A caller that tolerates missing captures (such as a trainer replaying sampler routing) marks tokens without
captured routing ``ROUTER_REPLAY_FRESH`` in ``routed_experts``. Once a router opts in with
``enable_fresh_replay_routing()`` it picks such tokens' experts with its own gate and gathers every other token's
replayed experts; a router that has not opted in keeps the gather-only replay path. Under whole-block activation
checkpointing the self-replay wrapper hands the recompute the experts the forward picked.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from arctic_platform.model.implementations.gpu.router_replay_recompute import _in_recompute
from arctic_platform.model.implementations.gpu.router_replay_recompute import install_self_router_replay
from arctic_platform.model.implementations.moe.layers.moe import ROUTER_REPLAY_FRESH
from arctic_platform.model.implementations.moe.layers.moe import TokenChoiceTopKRouter
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal


@pytest.fixture(autouse=True)
def _cpu_integer_histc(monkeypatch):
    # The router counts tokens per expert with ``torch.histc`` on int64 ids, which CUDA supports and CPU does
    # not; the counts are the same either way.
    histc = torch.histc
    monkeypatch.setattr(torch, "histc", lambda values, *args, **kwargs: histc(values.float(), *args, **kwargs))


def _router(score_func="softmax", route_norm=False, fresh_routing=True):
    set_seed(0)
    router = TokenChoiceTopKRouter(
        dim=8, num_experts=6, top_k=2, score_func=score_func, route_norm=route_norm, route_scale=1.0
    )
    if fresh_routing:
        router.enable_fresh_replay_routing()
    return router


def _mixed_routing(fresh_rows=(1, 3)):
    routed = torch.tensor([[5, 4], [0, 1], [3, 2], [1, 0], [2, 5]], dtype=torch.int64)
    routed[list(fresh_rows)] = ROUTER_REPLAY_FRESH
    return routed


@pytest.mark.parametrize("use_expert_bias", [False, True])
@pytest.mark.parametrize("score_func,route_norm", [("softmax", False), ("sigmoid", True)])
def test_fresh_tokens_match_replay_off_and_replayed_tokens_gather(use_expert_bias, score_func, route_norm):
    router = _router(score_func, route_norm)
    x = torch.randn(5, 8)
    expert_bias = torch.linspace(-0.5, 0.5, 6) if use_expert_bias else None
    routed = _mixed_routing()
    fresh = (routed == ROUTER_REPLAY_FRESH).all(dim=1)

    off_scores, off_experts, _ = router(x, expert_bias)
    scores, experts, num_tokens_per_expert = router(x, expert_bias, routed_experts=routed)

    torch_assert_equal(experts[fresh], off_experts[fresh])
    torch_assert_equal(scores[fresh], off_scores[fresh])
    torch_assert_equal(experts[~fresh], routed[~fresh])
    assert int(num_tokens_per_expert.sum()) == 5 * 2


@pytest.mark.parametrize("all_fresh", [True, False])
def test_all_fresh_equals_replay_off_and_none_fresh_equals_strict_replay(all_fresh):
    router = _router()
    x = torch.randn(5, 8)
    routed = _mixed_routing(fresh_rows=range(5) if all_fresh else ())

    scores, experts, _ = router(x, routed_experts=routed)
    expected = router(x) if all_fresh else (router.gate(x).softmax(dim=1).gather(1, routed), routed)

    torch_assert_equal(experts, expected[1])
    torch_assert_equal(scores, expected[0])


def test_strict_router_keeps_gather_only_replay_and_rejects_fresh_sentinel():
    router = _router(fresh_routing=False)
    x = torch.randn(5, 8)
    routed = _mixed_routing(fresh_rows=())

    scores, experts, _ = router(x, routed_experts=routed)

    assert experts is routed  # the gather path hands the replayed tensor straight through
    torch_assert_equal(scores, router.gate(x).softmax(dim=1).gather(1, routed))
    with pytest.raises(RuntimeError, match="out of bounds"):
        router(x, routed_experts=_mixed_routing())


class _RouterBlock(nn.Module):
    def __init__(self, router):
        super().__init__()
        self.router = router
        self.proj = nn.Linear(8, 8, bias=False)

    def forward(self, x, routed_experts):
        scores, _experts, _ = self.router(x, routed_experts=routed_experts)
        return self.proj(x) * scores.sum(dim=-1, keepdim=True)


def test_checkpoint_recompute_with_fresh_tokens_matches_uncheckpointed_gradients():
    routed = _mixed_routing()
    set_seed(3)
    x = torch.randn(5, 8)

    reference = _RouterBlock(_router())
    reference(x, routed).sum().backward()

    block = _RouterBlock(_router())
    assert install_self_router_replay(block) == 1
    ckpt.checkpoint(block, x, routed, use_reentrant=False).sum().backward()

    assert len(block.router._self_replay_queue) == 0  # every captured forward was drained by its recompute
    torch_assert_close(block.router.gate.weight.grad, reference.router.gate.weight.grad, rtol=0, atol=0)
    torch_assert_close(block.proj.weight.grad, reference.proj.weight.grad, rtol=0, atol=0)


class _DriftingRouter(nn.Module):
    """``TokenChoiceTopKRouter``-shaped stub whose raw ``topk`` picks a different expert set on every call.

    With ``routed_experts`` it gathers, except for ``ROUTER_REPLAY_FRESH`` tokens, which take the drifting topk
    off the autograd graph. Each call records ``(in_recompute, routed_given, indices)``.
    """

    def __init__(self, hidden=16, num_experts=8, top_k=3):
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.top_k = top_k
        self.raw_topk_calls = 0
        self.calls = []

    def _raw_topk(self, scores):
        ramp = torch.arange(scores.shape[-1], dtype=scores.dtype) * float((-1) ** self.raw_topk_calls)
        self.raw_topk_calls += 1
        return torch.topk(scores + ramp, self.top_k, dim=-1).indices

    def forward(self, x, expert_bias=None, routed_experts=None):
        scores = self.gate(x)
        if routed_experts is None:
            indices = self._raw_topk(scores)
        elif (routed_experts == ROUTER_REPLAY_FRESH).any():
            fresh = routed_experts == ROUTER_REPLAY_FRESH
            indices = torch.where(fresh, self._raw_topk(scores.detach()), routed_experts)
        else:
            indices = routed_experts
        weights = torch.gather(torch.softmax(scores, dim=-1), -1, indices)
        self.calls.append((_in_recompute(), routed_experts is not None, indices.detach().clone()))
        return weights, indices, None


class _DriftingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = _DriftingRouter()
        self.proj = nn.Linear(16, 16, bias=False)

    def forward(self, x, routed_experts=None):
        weights, _indices, _ = self.router(x, routed_experts=routed_experts)
        return self.proj(x) * weights.sum(dim=-1, keepdim=True)


def _routing(fresh_rows, tokens=12, top_k=3):
    routed = torch.arange(tokens * top_k).reshape(tokens, top_k) % 8
    routed[list(fresh_rows)] = ROUTER_REPLAY_FRESH
    return routed


def _backward_through_checkpoint(block, routed):
    set_seed(7)
    x = torch.randn(12, 16, requires_grad=True)
    return ckpt.checkpoint(block, x, routed, use_reentrant=False)


def test_without_wrapper_fresh_tokens_diverge_on_recompute():
    """Control: fresh tokens topk on each pass, so without the wrapper they pick different experts."""
    block = _DriftingBlock()
    block.router.fresh_replay_routing = True
    _backward_through_checkpoint(block, _routing((2, 5, 9))).sum().backward()

    assert block.router.raw_topk_calls == 2
    assert not torch.equal(block.router.calls[0][2], block.router.calls[1][2])


def test_wrapper_replays_forward_experts_for_fresh_tokens():
    block = _DriftingBlock()
    assert install_self_router_replay(block) == 1
    block.router.fresh_replay_routing = True  # opted in after install, as a caller does
    routed = _routing((2, 5, 9))

    _backward_through_checkpoint(block, routed).sum().backward()

    assert block.router.raw_topk_calls == 1
    (forward_in_recompute, _, forward_idx), (recompute_in_recompute, _, recompute_idx) = block.router.calls
    assert (forward_in_recompute, recompute_in_recompute) == (False, True)
    torch_assert_equal(recompute_idx, forward_idx)
    replayed = routed[:, 0] != ROUTER_REPLAY_FRESH
    torch_assert_equal(forward_idx[replayed], routed[replayed])
    assert len(block.router._self_replay_queue) == 0


def test_wrapper_raises_on_capture_underflow_for_fresh_tokens():
    block = _DriftingBlock()
    assert install_self_router_replay(block) == 1
    block.router.fresh_replay_routing = True
    y = _backward_through_checkpoint(block, _routing((2,)))
    block.router._self_replay_queue.clear()  # the capture the recompute would replay is gone

    with pytest.raises(RuntimeError, match="no captured experts"):
        y.sum().backward()


def test_wrapping_defaults_routers_to_strict_production_replay():
    """A router that never opts in (like this stub) is strict: production replay defers to it, unqueued."""
    block = _DriftingBlock()
    assert not hasattr(block.router, "fresh_replay_routing")
    assert install_self_router_replay(block) == 1
    assert block.router.fresh_replay_routing is False
    routed = _routing(())

    _backward_through_checkpoint(block, routed).sum().backward()

    assert block.router.raw_topk_calls == 0
    assert all(torch.equal(idx, routed) for (_, _, idx) in block.router.calls)
    assert len(block.router._self_replay_queue) == 0


def test_aborted_forward_does_not_leak_experts_into_the_next_step():
    """A captured forward whose backward never ran must not hand its experts to the next step's recompute."""
    block = _DriftingBlock()
    assert install_self_router_replay(block) == 1
    block.router.fresh_replay_routing = True
    _backward_through_checkpoint(block, _routing((2,)))  # forward only: the step aborts before backward
    assert len(block.router._self_replay_queue) == 1

    routed = _routing((5, 9))
    _backward_through_checkpoint(block, routed).sum().backward()

    (_, _, forward_idx), (_, _, recompute_idx) = block.router.calls[-2:]
    torch_assert_equal(recompute_idx, forward_idx)
    replayed = routed[:, 0] != ROUTER_REPLAY_FRESH
    torch_assert_equal(recompute_idx[replayed], routed[replayed])
    assert len(block.router._self_replay_queue) == 0


def test_install_sets_the_wrapped_marker_trainers_check():
    block = _DriftingBlock()
    assert not getattr(block.router, "_self_replay_wrapped", False)

    assert install_self_router_replay(block) == 1

    assert block.router._self_replay_wrapped is True
    assert install_self_router_replay(block) == 0  # already wrapped
