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
"""Static expert placement (moe/expert_placement.py) only relabels which EP rank holds which expert.

The multi-rank tests run 4 CPU (gloo) ranks over a file store: a DCP load followed by the placement, a fake EP
dispatch through the MoE layers with and without a placement, and the HF / vLLM exports of a placed model.
"""

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


def test_inverse_round_trips():
    perm = torch.randperm(384, generator=torch.Generator().manual_seed(0))
    inv = placement.inverse(perm)
    assert torch.equal(inv[perm], torch.arange(384)) and torch.equal(perm[inv], torch.arange(384))
    assert torch.equal(placement.inverse(inv), perm)


def test_read_placement_accepts_a_path_or_the_json(tmp_path):
    path = tmp_path / "placement.json"
    path.write_text(_spec())
    assert placement.read_placement(str(path)) == placement.read_placement(_spec()) == json.loads(_spec())
    with pytest.raises(ValueError):
        placement.read_placement(json.dumps({"ep_size": EP}))


def _placed_shards(full: torch.Tensor, perm: list[int], dim: int = 0, k: int = 1) -> list[torch.Tensor]:
    """What each EP rank holds of ``full`` (experts in logical order along ``dim``, ``k`` entries each) under
    ``perm``."""
    pieces = [full.narrow(dim, e * k, k) for e in perm]
    return [torch.cat(pieces[r * L : (r + 1) * L], dim=dim) for r in range(EP)]


def _tagged(perm: list[int]) -> torch.Tensor:
    tensor = torch.empty(0)
    setattr(tensor, placement._SLOT_ATTR, placement.inverse(torch.tensor(perm)))
    return tensor


def test_cat_expert_shards_restores_logical_order():
    perm = PERMS[0]
    full = torch.randn(E, 3, 5)
    assert torch.equal(placement.cat_expert_shards(_tagged(perm), _placed_shards(full, perm)), full)
    # PEFT expert lora_A: [experts * r, in], expert-major.
    lora_a = torch.randn(E * 4, 5)
    assert torch.equal(placement.cat_expert_shards(_tagged(perm), _placed_shards(lora_a, perm, k=4)), lora_a)
    # PEFT expert lora_B: [out, r * experts] with the expert index fastest (weight_export._gather_ep_expert_lora).
    lora_b = torch.randn(6, 4 * E)
    shards = [s.reshape(6, -1) for s in _placed_shards(lora_b.view(6, 4, E), perm, dim=2)]
    gathered = placement.cat_expert_shards(_tagged(perm), [s.view(6, 4, -1) for s in shards], dim=2).flatten(1)
    assert torch.equal(gathered, lora_b)
    # Untagged tensors concatenate as before.
    assert torch.equal(placement.cat_expert_shards(torch.empty(0), list(full.chunk(EP))), full)


def test_expert_export_rows_place_gathered_rows_in_logical_order():
    perm = PERMS[1]
    full = torch.randn(E * 2, 3)
    gathered = torch.cat(_placed_shards(full, perm, k=2))
    dest = placement.expert_export_rows(_tagged(perm), gathered.shape[0])
    assert torch.equal(torch.empty_like(gathered).index_copy_(0, dest, gathered), full)
    assert placement.expert_export_rows(torch.empty(0), 4) is None


def _tiny_v41_config(**overrides):
    from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41Config

    config = dict(
        vocab_size=32,
        hidden_size=DIM,
        moe_intermediate_size=FFN,
        num_hidden_layers=2,
        num_nextn_predict_layers=0,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        max_position_embeddings=16,
        sliding_window=4,
        o_groups=2,
        o_lora_rank=4,
        num_experts_per_tok=TOPK,
        n_routed_experts=E,
        n_shared_experts=1,
        hc_mult=2,
        vision_n_layers=0,
        image_token_id=1,
        dspark_noise_token_id=1,
        pad_token_id=0,
        swiglu_limit=10.0,
    )
    config.update(overrides)
    return DeepseekV41Config(**config)


def test_slot_counts_match_the_slot_ids():
    from arctic_platform.model.implementations.deepseek_v41.moe import DeepseekV41MoE

    moe = DeepseekV41MoE(_tiny_v41_config(), layer_idx=0)
    selected = torch.randint(0, E, (64, TOPK))
    counts = torch.bincount(selected.flatten(), minlength=E).float()
    assert moe._to_expert_slots(selected, counts) == (selected, counts)
    moe.slot_expert = torch.tensor(PERMS[0])
    moe.expert_slot = placement.inverse(moe.slot_expert)
    slots, slot_counts = moe._to_expert_slots(selected, counts)
    assert torch.equal(moe.slot_expert[slots], selected)  # every choice lands on its own expert's slot
    assert torch.equal(slot_counts, torch.bincount(slots.flatten(), minlength=E).float())


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


def _v41_moes():
    from arctic_platform.model.implementations.deepseek_v41.moe import DeepseekV41MoE

    torch.manual_seed(0)
    return [_init(DeepseekV41MoE(_tiny_v41_config(), layer_idx=i)) for i in PERMS]


def _dcp_worker(rank, ckpt):
    from safetensors.torch import load_file
    from torch.distributed.checkpoint import HuggingFaceStorageReader
    from torch.distributed.checkpoint import load as dcp_load
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor.parallel import parallelize_module

    from arctic_platform.model.implementations.moe.deepspeed_integration import convert_dtensors_to_local
    from arctic_platform.model.implementations.moe.distributed.expert_parallel import DeepEPExpertParallel

    model = _Stack(_v41_moes())
    mesh = DeviceMesh.from_group(dist.group.WORLD, "cpu", mesh_dim_names=("ep",))
    for block in model.layers:
        parallelize_module(block.mlp.experts, device_mesh=mesh, parallelize_plan=DeepEPExpertParallel())
    state = {name: t for name, t in model.state_dict().items() if ".experts." in name}
    dcp_load(state, storage_reader=HuggingFaceStorageReader(path=ckpt), process_group=dist.group.WORLD)
    convert_dtensors_to_local(model)
    os.environ[placement.PLACEMENT_ENV] = _spec()
    assert placement.apply_expert_placement(model) == len(PERMS)
    with pytest.raises(RuntimeError, match="already placed"):  # a second placement must not permute the rows again
        placement.apply_expert_placement(model)
    full = load_file(os.path.join(ckpt, "model.safetensors"))
    for layer, perm in PERMS.items():
        experts = model.layers[layer].mlp.experts
        for name in ("w1", "w2", "w3"):
            want = full[f"layers.{layer}.mlp.experts.{name}"][perm[rank * L : (rank + 1) * L]]
            assert torch.equal(getattr(experts, name).data, want), (rank, layer, name)
    return True


def test_dcp_load_then_placement_puts_each_slots_expert_on_its_rank(tmp_path):
    from safetensors.torch import save_file

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    state = {name: t.contiguous() for name, t in _Stack(_v41_moes()).state_dict().items() if ".experts." in name}
    for name in state:  # distinct rows per expert
        state[name] = torch.randn_like(state[name])
    save_file(state, str(ckpt / "model.safetensors"))
    assert all(_run(_dcp_worker, tmp_path, str(ckpt)).values())


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


def _generic_moes():
    from arctic_platform.model.implementations.moe.layers.moe import MoE
    from arctic_platform.model.implementations.moe.layers.moe import MoEArgs

    torch.manual_seed(1)
    args = MoEArgs(num_experts=E, top_k=TOPK, score_before_experts=False, route_norm=True, use_grouped_mm=False)
    return [_init(MoE(args, dim=DIM, hidden_dim=FFN)) for _ in PERMS]


def _dispatch_worker(rank, family):
    import arctic_platform.model.implementations.moe.layers.moe as moe_layers
    from arctic_platform.model.implementations.deepseek_v41.moe import _integer_histc_on_cpu

    moe_layers.get_ep_comm_module = lambda backend: _FAKE_EP
    group = dist.group.WORLD
    runs = []
    for placed in (False, True):
        model = _Stack(_v41_moes() if family == "v41" else _generic_moes())
        _shard_experts(model, group)
        if placed:
            os.environ[placement.PLACEMENT_ENV] = _spec()
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


@pytest.mark.parametrize("family", ["v41", "generic"])
def test_fake_dispatch_with_placement_matches_contiguous(tmp_path, family):
    assert all(_run(_dispatch_worker, tmp_path, family).values())


def _export_worker(rank, stream):
    import deepspeed.utils.groups as ds_groups

    from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41ForCausalLM
    from arctic_platform.model.implementations.moe import deepspeed_integration
    from arctic_platform.model.implementations.moe.vllm_weights import build_iter_full_vllm_weights

    ds_groups._get_expert_parallel_group = lambda name: dist.group.WORLD
    if stream:
        deepspeed_integration._EXPORT_GATHER_BYTES = 1  # the row-sliced host path
    exports = []
    for placed in (False, True):
        torch.manual_seed(0)
        model = DeepseekV41ForCausalLM(_tiny_v41_config())
        # V4.1 has no per-layer HF conversion of its own yet; V4's is this.
        model.convert_layer_to_hf = lambda state, index, model=model: model.convert_to_hf(state)
        if placed:
            stack = model.model
            _shard_experts(stack, dist.group.WORLD)
            for block in stack.layers:
                for parameter in block.mlp.experts.parameters():
                    parameter.group_name, parameter.allreduce = "ep_size_4", False
            os.environ[placement.PLACEMENT_ENV] = _spec()
            assert placement.apply_expert_placement(model) == len(PERMS)
        hf = dict(deepspeed_integration.build_iter_full_hf_weights(model)())
        vllm = dict(build_iter_full_vllm_weights(model)())
        exports.append((hf, vllm))
    if rank == 0:
        for want, got in zip(*exports):
            assert want.keys() == got.keys() and any(".experts." in k for k in want)
            for name in want:
                assert torch.equal(got[name], want[name]), name
    return True


@pytest.mark.parametrize("stream", [False, True])
def test_hf_and_vllm_exports_of_a_placed_model_match_the_original(tmp_path, stream):
    assert all(_run(_export_worker, tmp_path, stream).values())
