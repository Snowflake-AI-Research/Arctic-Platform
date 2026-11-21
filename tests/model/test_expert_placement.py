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
"""Static expert placement (moe/expert_placement.py) only relabels which EP rank holds which expert: a fake EP
dispatch over 4 CPU (gloo) ranks through GLM-5.3's MoE, with and without a placement."""

import json
import os
import types
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from arctic_platform.model.implementations.moe import expert_placement as placement

E, EP, TOPK, DIM, FFN = 8, 4, 2, 16, 8
L = E // EP
# perm[slot] = logical expert; every rank's slots draw experts from other ranks.
PERMS = {0: [5, 2, 7, 0, 6, 1, 3, 4], 1: [1, 6, 0, 7, 3, 2, 4, 5]}


def _spec(layers=PERMS):
    return json.dumps({"ep_size": EP, "layers": {str(k): v for k, v in layers.items()}})


# --- multi-rank (gloo) ---------------------------------------------------------------------------------------------


def _run(worker, tmp_path, *args):
    results = mp.Manager().dict()
    mp.spawn(_entry, args=(worker, str(tmp_path / "store"), args, results), nprocs=EP, join=True)
    return dict(results)


def _entry(rank, worker, store, args, results):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dist.init_process_group("gloo", store=dist.FileStore(store, EP), rank=rank, world_size=EP)
    try:
        results[rank] = worker(rank, *args)
    finally:
        dist.destroy_process_group()


class _Block(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp


class _Stack(nn.Module):
    def __init__(self, moes):
        super().__init__()
        self.layers = nn.ModuleList(_Block(moe) for moe in moes)


def _init(moe):
    """O(1) weights, so that two layers keep O(1) activations and any misrouted row shows."""
    moe.init_weights(0.02, torch.device("cpu"))
    with torch.no_grad():
        for parameter in moe.parameters():
            parameter.normal_(0.0, 0.3)
        moe.expert_bias.normal_(0.0, 0.1)  # the router's bias is in logical ids
    return moe


# A dispatch/combine with the DeepEP interface over the gloo group: each (token, choice) row goes to the rank owning
# its slot, which runs its local experts on the rows sorted by local expert; combine sums the rows back per token.
@dataclass
class _Pending:
    rows: torch.Tensor
    scores: torch.Tensor
    local_ids: torch.Tensor
    local_experts: int
    tokens: torch.Tensor
    send: list
    recv: list
    num_tokens: int
    group: object
    score_before_experts: bool


@dataclass
class _State:
    order: torch.Tensor
    pending: _Pending
    permuted_scores: torch.Tensor | None = None


def _a2a(source, send, recv, group):
    from torch.distributed.nn.functional import all_to_all_single

    out = source.new_empty((sum(recv), *source.shape[1:]))
    return all_to_all_single(out, source.contiguous(), recv, send, group=group)


def _fake_dispatch(x, selected, scores, num_experts, group, *, score_before_experts=True):
    ranks = dist.get_world_size(group)
    per = num_experts // ranks
    flat = selected.reshape(-1)
    tokens = torch.arange(x.shape[0]).repeat_interleave(selected.shape[1])
    order = torch.argsort(flat // per, stable=True)
    send = torch.bincount(flat // per, minlength=ranks)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    send, recv = send.tolist(), recv.tolist()
    local_ids = torch.empty(sum(recv), dtype=torch.long)
    dist.all_to_all_single(local_ids, (flat % per)[order].contiguous(), recv, send, group=group)
    return _Pending(
        rows=_a2a(x[tokens[order]], send, recv, group),
        scores=_a2a(scores.reshape(-1)[order].float(), send, recv, group),
        local_ids=local_ids,
        local_experts=per,
        tokens=tokens[order],
        send=send,
        recv=recv,
        num_tokens=x.shape[0],
        group=group,
        score_before_experts=score_before_experts,
    )


def _fake_finalize(pending):
    from arctic_platform.model.implementations.moe.routing_weights import apply_routing_weights

    order = torch.argsort(pending.local_ids, stable=True)
    hidden, scores = pending.rows[order], pending.scores[order]
    counts = torch.bincount(pending.local_ids, minlength=pending.local_experts)
    if pending.score_before_experts:
        hidden, scores = apply_routing_weights(hidden, scores), None
    return hidden, counts, _State(order, pending, permuted_scores=scores)


def _fake_combine(out, state, **_):
    from arctic_platform.model.implementations.moe.routing_weights import apply_routing_weights

    if state.permuted_scores is not None:
        out = apply_routing_weights(out, state.permuted_scores)
    p = state.pending
    back = _a2a(out[placement.inverse(state.order)], p.recv, p.send, p.group)
    return out.new_zeros(p.num_tokens, out.shape[1]).index_add(0, p.tokens, back)


_FAKE_EP = types.SimpleNamespace(
    dispatch_tokens_async=_fake_dispatch,
    finalize_dispatch_tokens=_fake_finalize,
    combine_tokens=_fake_combine,
    sync_combine=lambda: None,
)


def _shard_experts(model, group):
    """Keep this rank's contiguous expert rows, as the DCP load leaves them."""
    rank = dist.get_rank(group)
    for block in model.layers:
        experts = block.mlp.experts
        experts.use_grouped_mm = False  # grouped mm is CUDA-only
        experts._ep_group = group
        for name in ("w1", "w2", "w3"):
            full = getattr(experts, name)
            setattr(experts, name, nn.Parameter(full.data[rank * L : (rank + 1) * L].clone()))


def _all_gather(t):
    out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t.contiguous())
    return torch.cat(out)


def _generic_moes(**overrides):
    from arctic_platform.model.implementations.moe.layers.moe import MoE
    from arctic_platform.model.implementations.moe.layers.moe import MoEArgs

    torch.manual_seed(1)
    fields = dict(num_experts=E, top_k=TOPK, score_before_experts=False, route_norm=True, use_grouped_mm=False)
    args = MoEArgs(**(fields | overrides))
    return [_init(MoE(args, dim=DIM, hidden_dim=FFN)) for _ in PERMS]


# GLM-5.2/5.3's MoE (glm_moe_dsa/modeling_glm_moe_dsa.py): sigmoid scores, normalized and scaled, a shared expert.
_FAMILIES = {
    "glm": lambda: _generic_moes(score_func="sigmoid", route_scale=2.5, num_shared_experts=1),
}


def _dispatch_worker(rank, family):
    import arctic_platform.model.implementations.moe.layers.moe as moe_layers
    from arctic_platform.model.implementations.deepseek_v41.moe import _integer_histc_on_cpu

    moe_layers.get_ep_comm_module = lambda backend: _FAKE_EP
    group = dist.group.WORLD
    runs = []
    for placed in (False, True):
        model = _Stack(_FAMILIES[family]())
        _shard_experts(model, group)
        if placed:
            placement.configure_from_options(_spec())
            assert placement.apply_expert_placement(model) == len(PERMS)
        g = torch.Generator().manual_seed(10 + rank)
        x = torch.randn(1, 24, DIM, generator=g, requires_grad=True)
        h = x
        with _integer_histc_on_cpu():  # the generic router's histc has no CPU integer kernel
            for block in model.layers:
                h = h + block.mlp(h)
        (h * torch.randn(h.shape, generator=g)).sum().backward()
        counts = [block.mlp.tokens_per_expert.clone() for block in model.layers]
        weights = [
            [_all_gather(getattr(block.mlp.experts, name).grad) for name in ("w1", "w2", "w3")]
            for block in model.layers
        ]
        gates = [block.mlp.router.gate.weight.grad.clone() for block in model.layers]
        runs.append((h.detach(), x.grad.clone(), counts, weights, gates))
    (h0, gx0, c0, w0, g0), (h1, gx1, c1, w1, g1) = runs
    torch.testing.assert_close(h1, h0, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gx1, gx0, rtol=1e-5, atol=1e-6)
    for layer, perm in enumerate(PERMS.values()):
        assert torch.equal(c1[layer], c0[layer])  # tokens_per_expert stays in logical order
        torch.testing.assert_close(g1[layer], g0[layer], rtol=1e-5, atol=1e-6)
        for got, want in zip(w1[layer], w0[layer]):
            # Slot s's gradient is its logical expert's gradient: optimizer state follows the rows.
            torch.testing.assert_close(got, want[torch.tensor(perm)], rtol=1e-5, atol=1e-6)
    return True


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_fake_dispatch_with_placement_matches_contiguous(tmp_path, family):
    assert all(_run(_dispatch_worker, tmp_path, family).values())
