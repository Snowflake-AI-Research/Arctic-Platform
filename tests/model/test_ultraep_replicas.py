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
"""UltraEP spares for trainable BF16 experts (moe/ultraep_replicas.py) hold their owners' weights and return their
gradients to them; ``ultraep_plan`` sends block-FP8 experts to moe/ultraep_frozen.py and refuses what it cannot copy.

Under pytest (CPU) a single-rank stand-in for ``ultra_ep.Manager`` checks the fetch's layout and gradients, the
recompute's routing replay and the path selection. Multi-GPU with the real UltraEP, run directly:
    torchrun --nproc-per-node 8 tests/model/test_ultraep_replicas.py
(``MAX_NUM_NVL_PEERS=4`` splits the 8 GPUs into two NVLink domains, as an EP group spanning two nodes would be.)
"""

import os
import sys
import types

import pytest
import torch
import torch.distributed as dist
from torch import nn

from arctic_platform.model.implementations.moe import ultraep_replicas

L, S, FFN, DIM, TOKENS, TOPK = 4, 2, 64, 128, 4096, 4


class _FakeManager:
    """One EP rank's ``ultra_ep.Manager`` on the CPU: the plan copies local expert ``j % L`` into spare ``j``, and
    ``reroute_sparse`` alternates a copied expert's tokens between its native slot and its copy with a counter that
    persists across calls, as UltraEP's does."""

    def __init__(self, *, num_layers, num_local_master_experts, num_local_redundant_experts, **kwargs):
        self.L, self.S = num_local_master_experts, num_local_redundant_experts
        fc1, fc2, dtype = kwargs["expert_fc1_numel"], kwargs["expert_fc2_numel"], kwargs["weight_data_dtype"]
        self.local_replica_weight_buffer = torch.zeros(self.S, fc1 + fc2, dtype=dtype)
        self.local_replica_fc1_weight_buffer = self.local_replica_weight_buffer[:, :fc1]
        self.local_replica_fc2_weight_buffer = self.local_replica_weight_buffer[:, fc1:]
        self.local_replica_fc1_grad_buffer = torch.zeros(self.S, fc1)
        self.local_replica_fc2_grad_buffer = torch.zeros(self.S, fc2)
        self.physical_to_logical_map = torch.full((num_layers, self.L + self.S), -1, dtype=torch.int64)
        self.logical_to_physical_map = torch.full((num_layers, self.L, 2), -1, dtype=torch.int64)
        self.logical_replica_counts = torch.ones(num_layers, self.L, dtype=torch.int64)
        self.pools: dict[int, tuple] = {}
        self.counter = 0

    def construct_local_master_ptr_pool(self, lid, fc1, fc2, grad1, grad2):
        self.pools[lid] = (fc1, fc2, grad1, grad2)

    def update_placement_sparse(self, lid, topk):
        spare = [j % self.L for j in range(self.S)]
        self.physical_to_logical_map[lid] = torch.tensor(list(range(self.L)) + spare)

    def _spares(self, lid, physical_to_logical_map=None):
        p2l = self.physical_to_logical_map[lid] if physical_to_logical_map is None else physical_to_logical_map
        return [(j, int(p2l[self.L + j])) for j in range(self.S) if int(p2l[self.L + j]) >= 0]

    def weight_sync(self, lid, physical_to_logical_map=None, **_):
        fc1, fc2, _, _ = self.pools[lid]
        for j, e in self._spares(lid, physical_to_logical_map):
            self.local_replica_fc1_weight_buffer[j].copy_(fc1[e].flatten())
            self.local_replica_fc2_weight_buffer[j].copy_(fc2[e].flatten())

    def grad_reduce(self, lid, physical_to_logical_map=None, **_):
        _, _, grad1, grad2 = self.pools[lid]
        for j, e in self._spares(lid, physical_to_logical_map):
            grad1[e].add_(self.local_replica_fc1_grad_buffer[j].view_as(grad1[e]))
            grad2[e].add_(self.local_replica_fc2_grad_buffer[j].view_as(grad2[e]))

    def reroute_sparse(self, lid, physical):
        copy_of = {e: self.L + j for j, e in self._spares(lid)}
        flat = physical.view(-1)
        for i in range(flat.numel()):
            e = int(flat[i])
            if e in copy_of:
                if self.counter % 2:
                    flat[i] = copy_of[e]
                self.counter += 1


@pytest.fixture
def fake_ultraep(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultra_ep", types.SimpleNamespace(Manager=_FakeManager))
    monkeypatch.setattr(ultraep_replicas.dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(ultraep_replicas, "_STATES", {})
    ultraep_replicas.configure_from_options(S)
    yield
    ultraep_replicas.configure_from_options(None)


def _bf16_experts(experts=4, dim=16, hidden=8, seed=0):
    from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts

    torch.manual_seed(seed)
    module = GroupedExperts(dim, hidden, experts, use_grouped_mm=False)
    for name in ("w1", "w2", "w3"):
        nn.init.normal_(getattr(module, name), std=0.3)
    return module


def _expert(x, w1, w2, w3):
    return (torch.nn.functional.silu(x @ w1.T) * (x @ w3.T)) @ w2.T


def test_spares_compute_with_their_owners_weights_and_return_their_gradients(fake_ultraep):
    """One call as the MoE runs it after dispatch: rows packed by physical slot, the experts extended by the call's
    spares; every slot computes its logical expert, and each copy's gradient lands on its owner's parameters."""
    from arctic_platform.model.implementations.moe.layers.moe import _ReplicaCall

    experts, layer, group = _bf16_experts(), nn.Module(), object()
    torch.manual_seed(1)
    selected = torch.cat([torch.zeros(64, 1, dtype=torch.long), torch.randint(1, 4, (64, 1))], dim=1)
    plan = ultraep_replicas.ultraep_plan(layer, experts, group, selected, building=True)
    physical = plan.remap(selected)
    assert plan.num_slots == 4 + S
    p2l = plan.state.gate_up.physical_to_logical_map[plan.lid]
    assert torch.equal(p2l[physical], selected), "remap keeps every token on a copy of its expert"
    assert (physical >= 4).any(), "the hot expert's tokens reach its copies"

    order = torch.argsort(physical.flatten(), stable=True)
    x = torch.randn(selected.shape[0], 16)
    rows = x[order // selected.shape[1]].requires_grad_(True)
    counts = torch.bincount(physical.flatten(), minlength=plan.num_slots)
    experts._arl_replica_call = _ReplicaCall(plan, group)
    out = experts(rows, counts)
    experts._arl_replica_call = None

    ref = {n: getattr(experts, n).detach().clone().requires_grad_(True) for n in ("w1", "w2", "w3")}
    slots = physical.flatten()[order]
    want = torch.stack([_expert(r, *(ref[n][p2l[s]] for n in ("w1", "w2", "w3"))) for r, s in zip(rows, slots)])
    torch.testing.assert_close(out, want)
    upstream = torch.randn_like(out)
    out.backward(upstream)
    want.backward(upstream)
    for n in ("w1", "w2", "w3"):
        torch.testing.assert_close(getattr(experts, n).grad, ref[n].grad, msg=n)


def test_the_recompute_replays_its_forwards_routes(fake_ultraep):
    """UltraEP's reroute counters move on between calls, so a recompute takes its forward's physical routes back,
    oldest first, instead of rerouting; a no-grad forward keeps none."""
    experts, layer, group = _bf16_experts(), nn.Module(), object()
    selected = torch.zeros(9, 1, dtype=torch.long)  # an odd count: the alternation does not restart in step
    forwards = []
    for _ in range(2):
        plan = ultraep_replicas.ultraep_plan(layer, experts, group, selected, building=True)
        forwards.append(plan.remap(selected).clone())
    assert not torch.equal(forwards[0], forwards[1]), "the counter carried over: the same tokens route differently"
    with torch.no_grad():
        ultraep_replicas.ultraep_plan(layer, experts, group, selected, building=True).remap(selected)
    # A graph-building forward drops what an earlier one kept (it has been recomputed by then), so only the last
    # forward's routes are replayed.
    replayed = ultraep_replicas.ultraep_plan(layer, experts, group, selected, building=False).remap(selected)
    assert torch.equal(replayed, forwards[1])
    with pytest.raises(RuntimeError, match="no routing from its forward"):
        ultraep_replicas.ultraep_plan(layer, experts, group, selected, building=False).remap(selected)


def test_block_fp8_experts_take_the_frozen_path(fake_ultraep, monkeypatch):
    from arctic_platform.model.implementations.moe import ultraep_frozen
    from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts

    calls = []
    monkeypatch.setattr(ultraep_frozen, "frozen_ultraep_plan", lambda *args: calls.append(args) or "frozen plan")
    experts = GroupedExperts(256, 256, 4, use_grouped_mm=True, fp8_block_size=128)
    ab = {"w1": "factors"}
    experts._ap_lora_ab = ab
    selected = torch.zeros(3, 2, dtype=torch.long)
    assert ultraep_replicas.ultraep_plan("layer", experts, "group", selected, building=True) == "frozen plan"
    assert calls == [("layer", experts, "group", selected, ab, True)]
    assert ultraep_replicas._STATES == {}


def test_bf16_experts_ultraep_cannot_copy_are_refused(fake_ultraep):
    from torch.nn.utils import parametrize

    class _Delta(nn.Module):
        def forward(self, weight):
            return weight + 1

    selected = torch.zeros(3, 2, dtype=torch.long)
    with_lora = _bf16_experts()
    with_lora._ap_lora_ab = {"w1": ("a", "b", 2, 1.0)}
    merged = _bf16_experts()
    parametrize.register_parametrization(merged, "w2", _Delta())
    zero3 = _bf16_experts()
    zero3.w1.ds_status = "NOT_AVAILABLE"
    for experts, match in ((with_lora, "expert LoRA"), (merged, "expert LoRA"), (zero3, "ZeRO-3")):
        with pytest.raises(NotImplementedError, match=match):
            ultraep_replicas.ultraep_plan(nn.Module(), experts, object(), selected, building=True)


def _moe(fp8: bool, packed4: str | None = None):
    from arctic_platform.model.implementations.moe.layers.moe import MoE
    from arctic_platform.model.implementations.moe.layers.moe import MoEArgs

    args = MoEArgs(
        num_experts=4, num_shared_experts=0, top_k=2, fp8_block_size=128 if fp8 else None, routed_packed4=packed4
    )
    return MoE(args, dim=256, hidden_dim=256)


def test_the_moe_refuses_spares_it_cannot_serve(fake_ultraep):
    from arctic_platform.model.implementations.moe import expert_replicas
    from arctic_platform.model.implementations.moe.layers.checkpointing import checkpoint_method

    selected = torch.zeros(3, 2, dtype=torch.long)
    selective = _moe(fp8=False)
    checkpoint_method(selective, "_run_local_routed_experts")  # ac_config.targets ["routed_experts"]
    with pytest.raises(NotImplementedError, match="selective activation checkpointing"):
        selective._expert_replica_plan(selective.experts, object(), selected, None)
    assert selective._expert_replica_plan(selective.experts, None, selected, None) is None  # no EP group

    ultraep_replicas.configure_from_options(0)
    expert_replicas.configure_spares(2)
    try:
        for frozen in (_moe(fp8=True), _moe(fp8=True, packed4="int4")):
            with pytest.raises(NotImplementedError, match="packed 4-bit experts take UltraEP"):
                frozen._expert_replica_plan(frozen.experts, object(), selected, None)
    finally:
        expert_replicas.configure_spares(None)


class _Experts(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        g = torch.Generator(device="cuda").manual_seed(100 + rank)
        bf16 = dict(device="cuda", dtype=torch.bfloat16)
        self.w1 = torch.nn.Parameter(torch.randn(L, FFN, DIM, generator=g, **bf16))
        self.w2 = torch.nn.Parameter(torch.randn(L, DIM, FFN, generator=g, **bf16))
        self.w3 = torch.nn.Parameter(torch.randn(L, FFN, DIM, generator=g, **bf16))


def _gather(t):
    out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t.contiguous())
    return torch.stack(out)


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    os.environ[ultraep_replicas.ULTRAEP_SPARES_ENV] = str(S)
    group = dist.new_group(list(range(world)))
    experts = _Experts(rank)
    module = torch.nn.Module()
    E = L * world
    g = torch.Generator(device="cuda").manual_seed(7 + rank)
    # Skewed routing: experts 0 and 1 are hot on every rank.
    hot = torch.randint(0, 2, (TOKENS, 1), device="cuda", generator=g)
    rest = torch.randint(2, E, (TOKENS, TOPK - 1), device="cuda", generator=g)
    selected = torch.cat([hot, rest], dim=1)

    plan = ultraep_replicas.ultraep_plan(module, experts, group, selected, building=True)
    physical = plan.remap(selected)
    p2l = plan.state.gate_up.physical_to_logical_map[plan.lid].clone()
    assert torch.equal(p2l[physical].long(), selected), "remap must keep every token on a copy of its expert"
    loads = torch.bincount(physical.flatten(), minlength=plan.num_slots).view(world, L + S).sum(1)
    logical = torch.bincount(selected.flatten(), minlength=E).view(world, L).sum(1)
    dist.all_reduce(loads)
    dist.all_reduce(logical)
    before, after = (logical.max() / logical.float().mean()).item(), (loads.max() / loads.float().mean()).item()

    full = plan.fetch((experts.w1, experts.w2, experts.w3), group)
    all_w = [_gather(w.detach()) for w in (experts.w1, experts.w2, experts.w3)]  # [world, L, ...]
    for j in range(L, L + S):
        lid = p2l[rank * (L + S) + j].item()
        if lid >= 0:
            owner, local = divmod(lid, L)
            for got, ref in zip(full, all_w):
                assert torch.equal(got[j], ref[owner, local]), f"spare {j} != expert {lid}"

    gen = torch.Generator(device="cuda").manual_seed(1000 + rank)
    upstream = [torch.randn(t.shape, device="cuda", dtype=torch.bfloat16, generator=gen) for t in full]
    torch.autograd.backward(full, upstream)

    all_up = [_gather(u.float()) for u in upstream]  # [world, L + S, ...]
    p2l_all = p2l.view(world, L + S)
    for k, w in enumerate((experts.w1, experts.w2, experts.w3)):
        expected = upstream[k][:L].float().clone()
        for r in range(world):
            for j in range(L, L + S):
                lid = p2l_all[r, j].item()
                if lid >= 0 and lid // L == rank:
                    expected[lid % L] += all_up[k][r, j]
        torch.testing.assert_close(w.grad.float(), expected.to(torch.bfloat16).float(), rtol=1e-2, atol=1e-2)
    if rank == 0:
        domain = plan.state.gate_up.nvl_domain_size
        print(
            f"ok: EP{world}, NVLink domain {domain}: spares hold their owners' weights, gradients return; "
            f"rank load max/mean {before:.3f} -> {after:.3f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
