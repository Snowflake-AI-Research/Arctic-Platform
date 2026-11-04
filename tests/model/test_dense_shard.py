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
"""Trainable replicated weights sharded over the GPUs of a node (moe/dense_shard.py) train as the replicated ones do.

CPU: the option's environment fallback, the PEFT refusal, the weight-decay marker.

GPU (8): two "nodes" of 4 ranks (shard group 4; ZeRO then reduces each shard over the 2 ranks at its position), a model
of embedding, three blocks (one recomputed with early stop, one without, one not recomputed) and an untied LM head,
whose blocks hold norms, biases, a 3-element parameter, a projection run under no_grad (no gradient, as GLM's DSA
indexer) and an expert-parallel weight (EP = the world, as GLM-5.3 at EP64). DeepSpeed ZeRO-2 in BF16 with FP32
masters, weight decay, gradient clipping at 1.0 and two micro-batches per step, built by ``initialize_training_engine``
as the training job does; without and with a sequence-parallel group of the node (SP 4, the job's SP8 x DP8 layout),
and with DSS's streamed optimizer-state swap when ``dss`` is importable. The sharded run's losses, gradient norms and
weights after three steps match the replicated run's, and each GPU holds about a quarter of the dense weights.

    python -m pytest -q tests/model/test_dense_shard.py
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

from arctic_platform.common.deepspeed_engine import NO_WEIGHT_DECAY_ATTR
from arctic_platform.common.deepspeed_engine import exempt_from_weight_decay
from arctic_platform.model.implementations.moe import dense_shard
from arctic_platform.testing_utils import set_seed

WORLD, NODE = 8, 4
VOCAB, DIM, TOKENS = 512, 256, 256
GAS, STEPS = 2, 3


def test_the_option_wins_and_the_environment_is_the_fallback(monkeypatch):
    monkeypatch.delenv(dense_shard.SHARD_DENSE_ENV, raising=False)
    monkeypatch.delenv(dense_shard.SHARD_DENSE_GROUP_SIZE_ENV, raising=False)
    assert not dense_shard.resolve_dense_sharding(None)
    monkeypatch.setenv(dense_shard.SHARD_DENSE_ENV, "1")
    assert dense_shard.resolve_dense_sharding(None)
    assert not dense_shard.resolve_dense_sharding(False)
    monkeypatch.setenv(dense_shard.SHARD_DENSE_GROUP_SIZE_ENV, "4")
    try:
        assert dense_shard.configured_group_size() == 4
        dense_shard.configure_group_size(2)
        assert dense_shard.configured_group_size() == 2
    finally:
        dense_shard.configure_group_size(None)


def test_lora_is_refused():
    from arctic_platform.model.config import ModelSpec
    from arctic_platform.model.config import Patches
    from arctic_platform.model.loader import LoaderContext
    from arctic_platform.model.loaders.glm_moe_dsa import load_glm_moe_dsa

    spec = ModelSpec(
        model_path_or_name="local",
        loader="glm_moe_dsa",
        attn_implementation="sdpa",
        patches=Patches(peft={"peft_type": "Lora"}),
        loader_options={"shard_dense": True},
    )
    with pytest.raises(ValueError, match="does not support PEFT"):
        load_glm_moe_dsa(LoaderContext(spec=spec, parallel_groups={"ep_group": object()}))


def test_every_rank_lists_the_moe_groups_in_one_order():
    """DeepSpeed gathers the expert group names in a set; the experts' and the shards' groups come out sorted."""
    from arctic_platform.common.deepspeed_engine import split_expert_param_groups

    def param(group_name=None):
        p = nn.Parameter(torch.zeros(2))
        if group_name is not None:
            p.allreduce, p.group_name = False, group_name
        return p

    groups = split_expert_param_groups(
        [
            {"params": [param(), param("ep_size_8"), param("ep_size_64")], "weight_decay": 0.1},
            {"params": [param(), param("ep_size_8")], "weight_decay": 0.0},
        ]
    )
    assert [(g["name"], g.get("moe", False), g["weight_decay"]) for g in groups] == [
        ("group_0", False, 0.1),
        ("group_1", False, 0.0),
        ("ep_size_64", True, 0.1),
        ("ep_size_8", True, 0.1),
        ("ep_size_8", True, 0.0),
    ]


def test_a_marked_parameter_is_exempt_from_weight_decay():
    param = nn.Parameter(torch.zeros(2))
    assert not exempt_from_weight_decay("layers.0.dense_shard.shard_0", param)
    assert exempt_from_weight_decay("layers.0.input_layernorm.weight", param)
    setattr(param, NO_WEIGHT_DECAY_ATTR, True)
    assert exempt_from_weight_decay("layers.0.dense_shard.shard_1", param)


class _Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * self.weight


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = _Norm(dim)
        self.q = nn.Linear(dim, dim, bias=True)
        self.up = nn.Linear(dim, 2 * dim, bias=False)
        self.down = nn.Linear(2 * dim, dim, bias=False)
        self.indexer = nn.Linear(dim, 8, bias=False)  # trainable, under no_grad: never gets a gradient
        self.mix = nn.Parameter(torch.tensor([1.0, 0.5, 0.0]))  # 3 elements: the members after it are re-aligned
        self.expert = nn.Parameter(torch.zeros(dim, dim))
        self.expert.allreduce = False
        self.expert.group_name = f"ep_size_{WORLD}"

    def forward(self, x):
        h = self.norm(x)
        with torch.no_grad():
            gate = self.indexer(h).sigmoid().mean(-1, keepdim=True)
        h = self.q(h) * (0.5 + gate)
        h = h + self.down(F.silu(self.up(h)))
        h = h * self.mix[0] + (h @ self.expert) * self.mix[1] + self.mix[2]
        return x + h


class _Model(nn.Module):
    def __init__(self):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.layers = nn.ModuleList(
            [
                checkpoint_wrapper(_Block(DIM)),  # torch's default early stop: the post-hook does not run
                checkpoint_wrapper(_Block(DIM), early_stop=False),  # as the MoE builders wrap their blocks
                _Block(DIM),  # not recomputed: keeps its gathered weights for the backward
            ]
        )
        self.norm = _Norm(DIM)
        self.lm_head = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, ids, labels):
        h = self.embed(ids)
        for layer in self.layers:
            h = layer(h)
        logits = self.lm_head(self.norm(h)).float()
        # Scaled so the gradient norm is above the clipping threshold.
        return 8 * F.cross_entropy(logits.view(-1, VOCAB), labels.view(-1))


def _inner(module):
    return getattr(module, "_checkpoint_wrapped_module", module)


def _build(rank: int, shard: bool, dtype: torch.dtype):
    set_seed(0)
    model = _Model()
    nn.init.normal_(model.embed.weight, std=0.02)
    generator = torch.Generator().manual_seed(100 + rank)  # each EP rank holds its own experts
    for layer in model.layers:
        _inner(layer).expert.data.normal_(0, DIM**-0.5, generator=generator)
    model = model.cuda().to(dtype)
    for layer in model.layers:  # the cast replaced the parameters' data, not the objects or their tags
        assert _inner(layer).expert.allreduce is False
    sharded = []
    if shard:
        modules = [_inner(layer) for layer in model.layers] + [model.embed, model.lm_head]
        sharded = dense_shard.install_dense_shards(model, modules, group_size=NODE)
    return model, sharded


def _force_moe_detection():
    """The test model's expert weights have no MoE module around them; the GLM model's MoE layers tell DeepSpeed."""
    import deepspeed.runtime.engine as ds_engine

    from arctic_platform.model.implementations.moe.deepspeed_integration import patch_deepspeed_moe_detection

    patch_deepspeed_moe_detection()
    if getattr(ds_engine.DeepSpeedEngine, "_test_dense_shard_patched", False):
        return
    original = ds_engine.DeepSpeedEngine._configure_distributed_model

    def patched(self, model):
        original(self, model)
        if any(getattr(p, "allreduce", True) is False for p in self.module.parameters()):
            self.has_moe_layers = True

    ds_engine.DeepSpeedEngine._configure_distributed_model = patched
    ds_engine.DeepSpeedEngine._test_dense_shard_patched = True


def _dense_bytes(model) -> int:
    return sum(
        p.numel() * p.element_size()
        for p in model.parameters()
        if getattr(p, "allreduce", True) is not False or dense_shard.is_dense_shard(p)
    )


def _full_weights(model) -> dict[str, torch.Tensor]:
    """Every trainable weight whole, on the CPU in FP32: the sharded ones gathered (collective)."""
    out = {
        name: p.detach().float().cpu().clone()
        for name, p in model.named_parameters()
        if not dense_shard.is_dense_shard(p)
    }
    for module in model.modules():
        sharded = getattr(module, "_ap_dense_shards", None)
        if sharded is not None:
            out.update({name: t.float().cpu() for name, t in sharded.full_parameters().items()})
    return out


def _train(rank: int, shard: bool, dtype: torch.dtype, sp: int, mpu, swap_config: dict | None):
    import contextlib

    from arctic_platform.common.deepspeed_engine import OptimizerSettings
    from arctic_platform.common.deepspeed_engine import TrainingEngineSpec
    from arctic_platform.common.deepspeed_engine import build_deepspeed_config
    from arctic_platform.common.deepspeed_engine import initialize_training_engine

    model, sharded = _build(rank, shard, dtype)
    # Several reduce-scatters per bucket: the LM head's shard is reduced in 8 slices.
    dense_shard._REDUCE_CHUNK_ELEMENTS = 1 << 14
    initial = _full_weights(model)
    ds_config = build_deepspeed_config(
        TrainingEngineSpec(
            dtype="bfloat16" if dtype == torch.bfloat16 else "float32",
            world_size=WORLD,
            sequence_parallel=sp,
            gradient_clipping=1.0,
            fp32_masters=True,
            ds_overrides={
                "gradient_accumulation_steps": GAS,
                "train_batch_size": GAS * WORLD // sp,
                # Small buckets: many reductions per backward, and the LM head is reduced on its own.
                "zero_optimization": {"stage": 2, "overlap_comm": True, "reduce_bucket_size": 1 << 16},
            },
        )
    )
    _force_moe_detection()
    swap_mod: Any = None
    split: Any = contextlib.nullcontext()
    if swap_config is not None:
        import importlib

        swap_mod = importlib.import_module("dss.ray_dss.jobs.gpu.optimizer_state_swap")
        split = swap_mod.param_group_split(swap_config)
    with split:
        engine, _, _ = initialize_training_engine(
            model,
            ds_config,
            optimizer=OptimizerSettings(lr=2e-3, weight_decay=0.1),
            ep_size=WORLD,
            mpu=mpu,
            model_path="test",
            rank=rank,
        )
    swap = swap_mod.maybe_install(engine.optimizer, swap_config) if swap_mod is not None else None
    groups = [
        (group.get("name"), bool(group.get("moe")), group["weight_decay"], sum(p.numel() for p in group["params"]))
        for group in engine.optimizer.optimizer.param_groups
    ]
    dense = _dense_bytes(model)
    generator = torch.Generator(device="cuda").manual_seed(1000 + rank)
    losses, norms, first = [], [], None
    for step in range(STEPS):
        for _ in range(GAS):
            ids = torch.randint(0, VOCAB, (1, TOKENS), device="cuda", generator=generator)
            loss = engine(ids, ids.roll(-1, dims=1))
            engine.backward(loss)
            engine.step()
            losses.append(loss.item())
        norms.append(float(engine.get_global_grad_norm()))
        if step == 0:
            first = _full_weights(model)
    final = _full_weights(model)
    with pytest.raises(RuntimeError, match="sharded over the node") if shard else contextlib.nullcontext():
        model.state_dict()
    offloaded = None
    if swap is not None:
        tensors = [t for i in range(len(engine.optimizer.single_partition_of_fp32_groups)) for t in swap.tensors(i)]
        offloaded = sum(t.untyped_storage().nbytes() == 0 for t in tensors) / len(tensors)
    engine.destroy()
    del engine, model, sharded
    torch.cuda.empty_cache()
    return {
        "losses": losses,
        "norms": norms,
        "initial": initial,
        "first": first,
        "final": final,
        "dense": dense,
        "groups": groups,
        "offloaded": offloaded,
    }


def _worker(rank: int, port: int, dtype: torch.dtype, sp: int, swap: bool, results) -> None:
    import deepspeed
    import torch.distributed as dist

    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD), LOCAL_RANK=str(rank)
    )
    torch.cuda.set_device(rank)
    deepspeed.init_distributed(dist_backend="nccl")
    from arctic_platform.model.parallel.groups import setup_parallel_groups

    mpu = setup_parallel_groups(sp, WORLD, world_size=WORLD, gpus_per_node=NODE).mpu
    swap_config = None
    if swap:
        # 1 MiB of FP32 master per slice: several slices per group, each fetched and evicted around its step.
        swap_config = {
            "optimizer_state_swap_fraction": 1.0,
            "optimizer_state_swap_stream": True,
            "optimizer_state_swap_group_gib": 1 / 1024,
        }
    replicated = _train(rank, False, dtype, sp, mpu, swap_config)
    sharded = _train(rank, True, dtype, sp, mpu, swap_config)
    import deepspeed.utils.groups as ds_groups

    # With SP of the node, the shards gather over the node's sequence-parallel group (no new communicator).
    reused = ds_groups._EXPERT_PARALLEL_GROUP[f"ep_size_{NODE}"] is (
        mpu.get_sequence_parallel_group() if mpu else None
    )
    assert reused == (sp == NODE), (sp, reused)

    assert sharded["final"].keys() == replicated["final"].keys()
    updates = {}
    for name, want in replicated["final"].items():
        assert torch.equal(sharded["initial"][name], replicated["initial"][name]), name
        update = want - replicated["initial"][name]
        diff = sharded["final"][name] - want
        # An element whose update moved by at least a quarter of one Adam step (lr): a flipped sign of a gradient
        # that nearly cancels.
        updates[name] = (
            float(diff.norm() / update.norm().clamp_min(1e-30)),
            float((diff.abs() > 5e-4).float().mean()),
            float(diff.abs().max()),
            want.numel(),
        )
    # After the first step: Adam's first update is lr * g / (|g| + eps), whatever the gradient's scale, so a BF16
    # weight differs only where the rounding of a node's partial sum flipped the sign of a gradient that nearly
    # cancels; an FP32 one also by the last bits of the update. Per parameter: (fraction differing, max abs diff).
    first_differ = {
        name: (
            float((sharded["first"][name] != want).float().mean()),
            float((sharded["first"][name] - want).abs().max()),
        )
        for name, want in replicated["first"].items()
    }
    results[rank] = {
        "first_differ": first_differ,
        "losses": (replicated["losses"], sharded["losses"]),
        "norms": (replicated["norms"], sharded["norms"]),
        "updates": updates,
        "dense_bytes": (replicated["dense"], sharded["dense"]),
        "groups": (replicated["groups"], sharded["groups"]),
        "offloaded": (replicated["offloaded"], sharded["offloaded"]),
    }
    dist.barrier()
    dist.destroy_process_group()


def _run(dtype: torch.dtype, sp: int, swap: bool) -> dict:
    from arctic_platform.testing_utils import get_unique_port_number
    from arctic_platform.testing_utils import reserve_free_port

    port = reserve_free_port(get_unique_port_number() + 3, span=5)
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(_worker, args=(port, dtype, sp, swap, results), nprocs=WORLD, join=True)
        return dict(results)


def _rel(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) / abs(y) for x, y in zip(a, b))


def _report(label: str, results: dict) -> None:
    r = results[0]
    print(f"\n{label}: losses replicated {r['losses'][0]}\n{label}: losses sharded    {r['losses'][1]}")
    print(f"{label}: grad norms replicated {r['norms'][0]} sharded {r['norms'][1]}")
    for rank in sorted(results):
        worst = max(results[rank]["updates"].items(), key=lambda item: item[1][0])
        print(
            f"{label}: rank {rank} loss rel {_rel(*results[rank]['losses'][::-1]):.2e} norm rel "
            f"{_rel(*results[rank]['norms'][::-1]):.2e} worst update {worst[0]} (rel, moved fraction, max abs, "
            f"numel) {worst[1]}"
        )
    worst_first = max(r["first_differ"].items(), key=lambda item: item[1][0])
    print(
        f"{label}: after step 1, the parameter with most differing elements: {worst_first[0]} (fraction, max abs) "
        f"{worst_first[1]}; largest difference {max(d for _, d in r['first_differ'].values()):.2e}"
    )
    print(
        f"{label}: dense bytes per GPU replicated / sharded {r['dense_bytes']} = "
        f"{r['dense_bytes'][1] / r['dense_bytes'][0]:.3f}; groups {r['groups'][1]}; offloaded {r['offloaded']}"
    )


_REQUIRES_GPUS = pytest.mark.skipif(torch.cuda.device_count() < WORLD, reason=f"needs {WORLD} GPUs")


@_REQUIRES_GPUS
@pytest.mark.parametrize("sp", [1, NODE], ids=["dp8", "sp4xdp2"])
def test_sharded_fp32_weights_train_as_replicated_ones(sp):
    """FP32 throughout: the two runs differ only in the order of FP32 sums, so they agree to FP32 noise."""
    results = _run(torch.float32, sp, False)
    _report(f"fp32 sp={sp}", results)
    for rank, r in results.items():
        (want_losses, got_losses), (want_norms, got_norms) = r["losses"], r["norms"]
        assert got_losses[0] == want_losses[0], rank  # the gathered weights are the replicated ones, bit for bit
        assert min(want_norms) > 1.0, want_norms  # the clipping is active
        assert _rel(got_norms, want_norms) < 1e-5 and _rel(got_losses, want_losses) < 1e-5, (rank, r)
        # The order of the FP32 sums changes a nearly cancelling gradient, and Adam's normalization turns that into
        # a small fraction of lr (2e-3; seen: 1.4e-3 of it); a wrong reduction would move elements by about lr.
        assert max(d for _, d in r["first_differ"].values()) < 2e-5, r["first_differ"]
        for name, (update_rel, moved, max_abs, _) in r["updates"].items():
            assert update_rel < 1e-3 and moved == 0.0, (rank, name, r["updates"][name])
        replicated, sharded = r["dense_bytes"]
        assert sharded < 0.26 * replicated, r["dense_bytes"]  # a quarter, and the final norm left whole


@_REQUIRES_GPUS
@pytest.mark.parametrize("sp, swap", [(1, False), (NODE, False), (NODE, True)], ids=["dp8", "sp4xdp2", "sp4xdp2-swap"])
def test_sharded_bf16_weights_train_as_replicated_ones(sp, swap):
    """BF16 with FP32 masters, as the job trains. The gradient of a shard is the BF16 rounding of its node's FP32
    partial sum before ZeRO sums the nodes (the replicated run rounds once, after the whole sum): the gradient norm
    agrees to ~1e-5 and the losses to the BF16 noise of the forward; the updates differ only in elements whose
    gradient nearly cancels (Adam's first steps move each element by about lr times its gradient's sign)."""
    if swap:
        pytest.importorskip("dss.ray_dss.jobs.gpu.optimizer_state_swap")
    results = _run(torch.bfloat16, sp, swap)
    _report(f"bf16 sp={sp} swap={swap}", results)
    for rank, r in results.items():
        (want_losses, got_losses), (want_norms, got_norms) = r["losses"], r["norms"]
        assert got_losses[0] == want_losses[0], rank
        assert min(want_norms) > 1.0, want_norms
        assert _rel(got_norms, want_norms) < 1e-4, (rank, r["norms"])
        assert _rel(got_losses, want_losses) < 1e-3, (rank, r["losses"])
        for name, (fraction, _) in r["first_differ"].items():
            # Seen: at most 1.5e-3 of a weight's elements (a gradient sum whose sign the rounding flipped, or an
            # update that moved its BF16 rounding); a missing or misplaced partial sum flips about half of them.
            # One flipped element of a bias is 1/256 of it.
            if r["updates"][name][3] >= 4096:
                assert fraction < 1e-2, (rank, name, r["first_differ"][name])
        for name, (update_rel, moved, max_abs, numel) in r["updates"].items():
            if numel >= 4096:  # a few flipped elements of a 3-element or bias parameter are a large fraction
                assert moved < 1e-2, (rank, name, r["updates"][name])
        replicated, sharded = r["dense_bytes"]
        assert sharded < 0.26 * replicated, r["dense_bytes"]
        if swap:
            # Some groups split into slices (3 and 4 groups without the swap), all their state off the GPU.
            assert len(r["groups"][0]) > 3 and len(r["groups"][1]) > 4 and r["offloaded"] == (1.0, 1.0), r
