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
"""CPU: Nemotron-H consumes router replay. ``NemotronHForCausalLM.forward(routed_experts=...)`` reaches every MoE
layer's ``NemotronHRouter``, which then runs exactly the replayed experts, weighted by its own unbiased scores as vLLM
weights the experts it picked; without routes the routing is what it was. Covered on the paths a trainer takes: the
EP dispatch (``LatentMoE._run_deepep_routed_experts``, here a local stand-in that runs the same experts), activation
checkpointing with the router-replay recompute wrapper, and the sequence-parallel backbone wrapper."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from arctic_platform.model.implementations.gpu.router_replay_recompute import _router_supports_routed_experts
from arctic_platform.model.implementations.gpu.router_replay_recompute import install_self_router_replay
from arctic_platform.model.implementations.moe import sequence_parallel
from arctic_platform.model.implementations.moe.layers.moe import LatentMoE
from arctic_platform.model.implementations.moe.layers.moe import NemotronHRouter
from arctic_platform.model.implementations.nemotron_h.configuration_nemotron_h import NemotronHConfig
from arctic_platform.model.implementations.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM

LAYERS = ["moe", "attention", "moe"]
TOP_K = 2
EXPERTS = 8


def _router(n_group: int = 1) -> NemotronHRouter:
    torch.manual_seed(0)
    router = NemotronHRouter(
        dim=16, num_experts=EXPERTS, top_k=TOP_K, n_group=n_group, topk_group=1, norm_topk_prob=True
    )
    router.init_weights(0.3)
    router.e_score_correction_bias.copy_(torch.linspace(0.0, 0.5, EXPERTS))
    return router


def _reference_selection(router: NemotronHRouter, x: torch.Tensor) -> torch.Tensor:
    """The gate's top-k as it was before replay existed (scores + bias, within the top groups)."""
    scores = F.linear(x.float(), router.gate.float()).sigmoid()
    choice = scores + router.e_score_correction_bias
    if router.n_group > 1:
        groups = choice.view(-1, router.n_group, EXPERTS // router.n_group)
        group_idx = groups.topk(2, dim=-1)[0].sum(dim=-1).topk(router.topk_group, dim=-1, sorted=False)[1]
        mask = torch.zeros(choice.shape[0], router.n_group).scatter_(1, group_idx, 1)
        mask = mask.unsqueeze(-1).expand(-1, router.n_group, EXPERTS // router.n_group).reshape(-1, EXPERTS)
        choice = choice.masked_fill(~mask.bool(), 0.0)
    return torch.topk(choice, k=TOP_K, dim=-1, sorted=False)[1]


@pytest.mark.parametrize("n_group", [1, 2])
def test_router_without_routes_is_unchanged(n_group):
    router = _router(n_group)
    x = torch.randn(32, 16)
    weights, selected, counts = router(x)
    assert torch.equal(selected, _reference_selection(router, x))
    scores = F.linear(x, router.gate).sigmoid().gather(1, selected)
    torch.testing.assert_close(weights, scores / (scores.sum(-1, keepdim=True) + 1e-20), rtol=0, atol=0)
    assert torch.equal(counts, torch.bincount(selected.reshape(-1), minlength=EXPERTS).float())


def test_router_runs_the_replayed_experts_with_its_own_unbiased_weights():
    router = _router()
    x = torch.randn(32, 16)
    own = router(x)[1]
    # Experts the bias would never pick (the lowest-biased ones), so a replay that leaked the gate's choice would show.
    replayed = torch.stack([torch.arange(32) % 3, 3 + torch.arange(32) % 2], dim=-1)
    assert not torch.equal(torch.sort(replayed, -1).values, torch.sort(own, -1).values)
    weights, selected, counts = router(x, None, routed_experts=replayed)
    assert torch.equal(selected, replayed)
    scores = torch.sigmoid(x @ router.gate.T).gather(1, replayed)
    torch.testing.assert_close(weights, scores / scores.sum(-1, keepdim=True), rtol=1e-6, atol=0)
    assert torch.equal(counts, torch.bincount(replayed.reshape(-1), minlength=EXPERTS).float())
    assert _router_supports_routed_experts(router)


def _model() -> NemotronHForCausalLM:
    config = NemotronHConfig(
        hidden_size=16,
        layers_block_type=LAYERS,
        vocab_size=32,
        n_routed_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=8,
        moe_shared_expert_intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        routed_scaling_factor=2.5,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(1)
    model = NemotronHForCausalLM(config).float()
    for name, tensor in model.named_parameters():
        torch.nn.init.normal_(tensor, std=0.3 if "router" in name else 0.1)
    for module in model.modules():
        if isinstance(module, NemotronHRouter):
            module.e_score_correction_bias.normal_(0.0, 0.05)
    return model


@pytest.fixture
def local_experts(monkeypatch):
    """Run LatentMoE's EP dispatch path locally: the same relu^2 experts, weights and routed scaling, on one process.
    Records, per MoE module, the expert ids and weights the router handed to the dispatch."""
    seen: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    def run(self, x, selected_experts_indices, top_scores, num_tokens_per_expert=None):
        seen.setdefault(id(self), []).append((selected_experts_indices.detach().clone(), top_scores.detach().clone()))
        w1, w2 = self.experts.w1, self.experts.w2
        hidden = F.relu(torch.einsum("th,tkih->tki", x, w1[selected_experts_indices])).square()
        expert_out = torch.einsum("tki,tkhi->tkh", hidden, w2[selected_experts_indices])
        routed = (expert_out * top_scores.unsqueeze(-1)).sum(1) * self.routed_scaling_factor
        return self.shared_expert(x) + routed

    monkeypatch.setattr(LatentMoE, "_run_deepep_routed_experts", run)
    return seen


def _moe_modules(model):
    return [layer.mlp for layer, kind in zip(model.model.layers, LAYERS) if kind == "moe"]


def _inputs(model, seq: int = 12):
    torch.manual_seed(2)
    ids = torch.randint(0, model.config.vocab_size, (1, seq))
    return ids, torch.arange(seq).unsqueeze(0)


def _own_routes(model, seen, ids, positions):
    seen.clear()
    with torch.no_grad():
        model.model(input_ids=ids, position_ids=positions)
    routes = torch.zeros(1, ids.shape[1], len(LAYERS), TOP_K, dtype=torch.long)
    for layer_idx, kind in enumerate(LAYERS):
        if kind == "moe":
            routes[0, :, layer_idx] = seen[id(model.model.layers[layer_idx].mlp)][-1][0]
    return routes


def test_model_replays_its_own_routes_bit_exactly(local_experts):
    model = _model()
    ids, positions = _inputs(model)
    routes = _own_routes(model, local_experts, ids, positions)
    with torch.no_grad():
        base = model.model(input_ids=ids, position_ids=positions).last_hidden_state
        replayed = model.model(input_ids=ids, position_ids=positions, routed_experts=routes).last_hidden_state
    assert torch.equal(base, replayed)


def test_every_moe_layer_runs_the_replayed_experts(local_experts):
    model = _model()
    ids, positions = _inputs(model)
    routes = _own_routes(model, local_experts, ids, positions)
    base = model.model(input_ids=ids, position_ids=positions).last_hidden_state.detach()
    foreign = routes.clone()
    foreign[0, :, 0] = torch.stack([torch.arange(12) % 4, 4 + torch.arange(12) % 4], -1)
    foreign[0, :, 2] = torch.stack([7 - torch.arange(12) % 3, torch.arange(12) % 2], -1)
    local_experts.clear()

    class Head(torch.nn.Module):  # the injected PrimeLm head's call signature
        def forward(self, hidden, labels=None, temperature=None):
            return {"hidden": hidden}

    model.lm_head = Head()
    router_inputs = {}
    hooks = [
        moe.router.register_forward_pre_hook(lambda module, args, key=id(moe): router_inputs.__setitem__(key, args[0]))
        for moe in _moe_modules(model)
    ]
    out = model(input_ids=ids, position_ids=positions, routed_experts=foreign)["hidden"]
    for hook in hooks:
        hook.remove()
    for layer_idx, moe in zip((0, 2), _moe_modules(model)):
        ((selected, weights),) = local_experts[id(moe)]
        assert torch.equal(selected, foreign[0, :, layer_idx])
        scores = torch.sigmoid(router_inputs[id(moe)] @ moe.router.gate.T).gather(1, selected)
        torch.testing.assert_close(weights, scores / scores.sum(-1, keepdim=True), rtol=1e-6, atol=0)
    assert not torch.allclose(out, base)


def test_routed_experts_must_index_every_layer(local_experts):
    model = _model()
    ids, positions = _inputs(model)
    with pytest.raises(ValueError, match="routed_experts holds 2 layers, the model 3"):
        model.model(
            input_ids=ids, position_ids=positions, routed_experts=torch.zeros(1, 12, 2, TOP_K, dtype=torch.long)
        )


def _grads(model, ids, positions, routes, checkpointed: bool):
    model.zero_grad(set_to_none=True)
    layers = model.model.layers
    if checkpointed:
        for index, layer in enumerate(list(layers)):
            install_self_router_replay(layer)
            layers[index] = checkpoint_wrapper(layer)
    hidden = model.model(input_ids=ids, position_ids=positions, routed_experts=routes).last_hidden_state
    (hidden * torch.linspace(-1, 1, hidden.numel()).view_as(hidden)).sum().backward()
    return hidden.detach(), {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("replay", [True, False])
def test_activation_checkpointing_with_the_replay_wrapper(local_experts, replay):
    """Under full activation checkpointing with the router-replay recompute wrapper (ac_config.router_replay_recompute),
    the forward and its recompute route alike, replayed or not, and match an uncheckpointed run."""
    ids, positions = _inputs(_model())
    routes = _own_routes(_model(), local_experts, ids, positions) if replay else None
    if replay:
        routes[0, :, 0, 0] = (routes[0, :, 0, 1] + 1) % EXPERTS  # not the gate's own choice
    plain_hidden, plain = _grads(_model(), ids, positions, routes, checkpointed=False)
    model = _model()
    hidden, grads = _grads(model, ids, positions, routes, checkpointed=True)
    assert sum(getattr(moe.router, "_self_replay_wrapped", False) for moe in _moe_modules(model)) == 2
    torch.testing.assert_close(hidden, plain_hidden, rtol=0, atol=0)
    by_name = {name.replace("._checkpoint_wrapped_module", ""): grad for name, grad in grads.items()}
    assert by_name.keys() == plain.keys()
    for name, grad in plain.items():
        torch.testing.assert_close(by_name[name], grad, rtol=1e-6, atol=1e-7, msg=name)


def test_sequence_parallel_backbone_wrapper_forwards_the_routes(tmp_path, local_experts):
    """The generic SP wrapper (moe/sequence_parallel.py) passes routed_experts on to a backbone whose forward accepts
    it, so each rank's MoE layers replay their shard's routes. Uses the session's single-rank group (tests/conftest.py)
    when there is one."""
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'rendezvous'}")
    try:
        model = _model()
        ids, positions = _inputs(model)
        routes = _own_routes(model, local_experts, ids, positions)
        routes[0, :, 2] = torch.stack([torch.arange(12) % 8, (torch.arange(12) + 5) % 8], -1)
        backbone = model.model
        wrapped = sequence_parallel._make_backbone_sp_forward(backbone.forward, dist.group.WORLD, [], [], [])
        local_experts.clear()
        with torch.no_grad():
            wrapped(backbone, input_ids=ids, position_ids=positions, routed_experts=routes)
        ((selected, _),) = local_experts[id(backbone.layers[2].mlp)]
        assert torch.equal(selected, routes[0, :, 2])
    finally:
        if owns_group:
            dist.destroy_process_group()
