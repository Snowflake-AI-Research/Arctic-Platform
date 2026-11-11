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
"""Frozen replicated layer weights sharded across ranks and gathered per call, under activation checkpointing."""

import io
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(64, 64, bias=False).requires_grad_(False)
        self.scale = nn.Parameter(torch.rand(64, dtype=torch.float32), requires_grad=False)
        self.lora = nn.Linear(64, 64, bias=False)
        self.expert = nn.Parameter(torch.randn(3, 5), requires_grad=False)
        self.expert.allreduce = False

    def forward(self, x):
        return (self.frozen(x) * self.scale.to(x.dtype) + self.lora(x)).relu()


def _run(rank, world, port, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    from arctic_platform.model.implementations.moe.frozen_shard import install_frozen_layer_shards

    torch.manual_seed(0)
    reference = nn.ModuleList([_Layer(), _Layer()]).cuda().bfloat16()
    for layer in reference:
        layer.scale.data = layer.scale.data.float()
    sharded = nn.ModuleList([checkpoint_wrapper(_Layer(), early_stop=False) for _ in range(2)]).cuda().bfloat16()
    for ref, layer in zip(reference, sharded):
        params = dict(layer._checkpoint_wrapped_module.named_parameters())
        for name, param in ref.named_parameters():
            params[name].data = param.data.clone()
    install_frozen_layer_shards(sharded, root=sharded)

    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    want = reference[1](reference[0](x))
    want.float().sum().backward()
    for step in range(2):
        for layer in sharded:
            layer.zero_grad(set_to_none=True)
        got = sharded[1](sharded[0](x))
        got.float().sum().backward()
        assert torch.equal(got, want), step
        for ref, layer in zip(reference, sharded):
            assert torch.equal(layer._checkpoint_wrapped_module.lora.weight.grad, ref.lora.weight.grad), step
    inner = sharded[0]._checkpoint_wrapped_module
    results[rank] = (
        inner.frozen.weight.numel(),
        inner.scale.numel(),
        inner.expert.numel(),
        inner._ap_frozen_shard.shard.numel(),
    )
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_sharded_frozen_weights_match_replicated_ones():
    world = 2
    results = mp.Manager().dict()
    mp.spawn(_run, args=(world, 29511, results), nprocs=world, join=True)
    for rank in range(world):
        frozen, scale, expert, shard = results[rank]
        assert frozen == 0 and scale == 0, "frozen replicated weights keep only their shard"
        assert expert == 15, "expert-parallel parameters stay whole"
        assert shard < 64 * 64 * 2 + 64 * 4


# The weight export and the state dict of a model whose frozen weights are sharded (moe/frozen_held.py).

DIM, ROWS, LAYERS = 64, 3, 3


class _Block(nn.Module):
    """A decoder layer: frozen BF16 attention and FP32 norm weights, a trainable projection and frozen EP experts."""

    def __init__(self, experts: bool):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(DIM, DIM, bias=False).requires_grad_(False)
        self.input_layernorm = nn.Module()
        self.input_layernorm.weight = nn.Parameter(torch.rand(DIM), requires_grad=False)
        self.lora = nn.Linear(DIM, DIM, bias=False)
        self.mlp = nn.Module()
        if experts:
            self.mlp.experts = nn.Module()
            self.mlp.experts.w2 = nn.Parameter(torch.randn(ROWS, DIM), requires_grad=False)

    def forward(self, x):
        h = self.self_attn.q_proj(x) * self.input_layernorm.weight.to(x.dtype) + self.lora(x)
        experts = getattr(self.mlp, "experts", None)
        if experts is not None:
            h = h + (x @ experts.w2.t()).sum(-1, keepdim=True)
        return h.relu()


class _LM(nn.Module):
    def __init__(self, experts: bool):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [checkpoint_wrapper(_Block(experts), early_stop=False) for _ in range(LAYERS)]
        )
        self.model.norm = nn.Module()
        self.model.norm.weight = nn.Parameter(torch.rand(DIM), requires_grad=False)

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x * self.model.norm.weight.to(x.dtype)

    def convert_layer_to_hf(self, state, layer_index):
        """A conversion that reads the values: the query projection with its norm folded in, as a second tensor."""
        if layer_index >= 0:
            prefix = f"model.layers.{layer_index}."
            q, norm = state[prefix + "self_attn.q_proj.weight"], state[prefix + "input_layernorm.weight"]
            state[prefix + "self_attn.q_norm_folded"] = q.float() * norm


def _lm(seed: int, rank: int, device: str, experts: bool = True, contract: bool = True) -> _LM:
    """The model in BF16 (norms FP32) with the MoE families' export contract; each EP rank has its own experts."""
    from arctic_platform.model.implementations.moe.deepspeed_integration import build_iter_full_hf_weights
    from arctic_platform.model.implementations.moe.vllm_weights import build_iter_full_vllm_weights
    from arctic_platform.model.weight_export import WeightExportContract
    from arctic_platform.model.weight_export import register_weight_export

    torch.manual_seed(seed)
    model = _LM(experts)
    if experts:
        generator = torch.Generator().manual_seed(1000 + 10 * seed + rank)
        for layer in model.model.layers:
            w2 = layer._checkpoint_wrapped_module.mlp.experts.w2
            w2.data = torch.randn(w2.shape, generator=generator)
    model = model.to(device).bfloat16()
    for name, param in model.named_parameters():
        if name.endswith("layernorm.weight") or name.endswith("norm.weight"):
            param.data = param.data.float()
        if ".experts." in name:
            param.allreduce = False
            param.group_name = "ep_size_2"
    if contract:
        register_weight_export(
            model, WeightExportContract(hf=build_iter_full_hf_weights, vllm=build_iter_full_vllm_weights)
        )
    return model


def _export(model, weight_format: str, kwargs: dict) -> list:
    """The export on every rank, rank 0's tensors cloned to the host (with their names), in order."""
    from arctic_platform.model import iter_model_weights

    return [
        (name, tensor.detach().cpu().clone())
        for name, tensor in iter_model_weights(
            model, weight_format, is_master=dist.get_rank() == 0, is_zero3=False, **kwargs
        )
    ]


def _same(got: list, want: list) -> None:
    assert [name for name, _ in got] == [name for name, _ in want]
    for (name, g), (_, w) in zip(got, want):
        assert g.dtype == w.dtype and g.shape == w.shape and torch.equal(g, w), name


_FORMATS = (("hf", {"include_skipped": True}), ("hf", {}), ("vllm", {}))


def _train(model, x) -> None:
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        model(x).float().square().sum().backward()


def _export_worker(rank: int, world: int, rendezvous: str, device: str, contract: bool, results) -> None:
    import deepspeed.utils.groups as ds_groups

    from arctic_platform.model.implementations.moe.frozen_held import whole_frozen_weights
    from arctic_platform.model.implementations.moe.frozen_shard import install_frozen_layer_shards

    if device == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl" if device == "cuda" else "gloo", init_method=rendezvous, rank=rank, world_size=world
    )
    ds_groups._EXPERT_PARALLEL_GROUP["ep_size_2"] = dist.group.WORLD
    experts = contract  # the generic export (no family contract) refuses expert-parallel parameters
    formats = _FORMATS if contract else (("hf", {"include_skipped": True}), ("hf", {}))
    reference, sharded, other = (_lm(seed, rank, device, experts, contract) for seed in (0, 0, 1))
    want = {f"{f}{sorted(k)}": _export(reference, f, k) for f, k in formats}
    install_frozen_layer_shards(sharded.model.layers, root=sharded)
    for f, k in formats:  # before the first call nothing is sharded
        _same(_export(sharded, f, k), want[f"{f}{sorted(k)}"])

    x = torch.randn(4, DIM, device=device, dtype=torch.bfloat16)
    _train(reference, x)
    _train(sharded, x)
    inner = sharded.model.layers[0]._checkpoint_wrapped_module
    assert inner.self_attn.q_proj.weight.numel() == 0 and inner.input_layernorm.weight.numel() == 0
    for f, k in formats:
        got = _export(sharded, f, k)
        _same(got, want[f"{f}{sorted(k)}"])
        if rank == 0 and f == "hf":
            assert dict(got)["model.layers.1.self_attn.q_proj.weight"].numel() == DIM * DIM
    assert inner.self_attn.q_proj.weight.numel() == 0, "the export leaves the layers sharded"

    # The state dict: refused outside whole_frozen_weights, the unsharded model's inside (on the host), and a load
    # there becomes the shards (the next calls and the export run on the loaded weights).
    with pytest.raises(RuntimeError, match="holds frozen weights"):
        sharded.state_dict()
    with whole_frozen_weights(sharded):
        state = sharded.state_dict()
        want_state = reference.state_dict()
        assert list(state) == list(want_state)
        assert all(torch.equal(state[key].cpu(), want_state[key].cpu()) for key in state)
        assert inner.self_attn.q_proj.weight.device.type == "cpu"
        # What DeepSpeed's checkpoint does with it: the BF16 and FP32 weights of a layer must not share a storage.
        saved = io.BytesIO()
        torch.save({"module": state, "frozen": [p.detach().cpu() for p in sharded.parameters()]}, saved)
        saved.seek(0)
        loaded = torch.load(saved, weights_only=True)["module"]
        assert all(torch.equal(loaded[key].cpu(), want_state[key].cpu()) for key in state)
        sharded.load_state_dict(other.state_dict())
    assert inner.self_attn.q_proj.weight.numel() == 0
    want_other = {f"{f}{sorted(k)}": _export(other, f, k) for f, k in formats}
    for f, k in formats:
        _same(_export(sharded, f, k), want_other[f"{f}{sorted(k)}"])
    assert torch.equal(sharded(x), other(x))
    ds_groups._EXPERT_PARALLEL_GROUP.pop("ep_size_2", None)
    results[rank] = True
    dist.destroy_process_group()


def _spawn_export(device: str, contract: bool, tmp_path) -> None:
    world = 2
    results = mp.Manager().dict()
    mp.spawn(
        _export_worker,
        args=(world, f"file://{tmp_path}/rendezvous", device, contract, results),
        nprocs=world,
        join=True,
    )
    assert dict(results) == {0: True, 1: True}


def test_a_sharded_model_exports_the_unsharded_models_weights_cpu(tmp_path):
    """Gloo on CPU: the MoE families' HF and vLLM builders (expert-parallel experts gathered beside the sharded
    weights) before and after the first call; the state dict and a load."""
    _spawn_export("cpu", True, tmp_path)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_a_sharded_model_exports_the_unsharded_models_weights(tmp_path):
    _spawn_export("cuda", True, tmp_path)


def _composite_worker(rank: int, world: int, rendezvous: str, results) -> None:
    """Both holdings on one model (DeepSeek-V4.1's long LoRA): frozen dense weights sharded, expert codes offloaded."""
    import deepspeed.utils.groups as ds_groups

    from arctic_platform.model.implementations.moe.frozen_held import whole_frozen_weights
    from arctic_platform.model.implementations.moe.frozen_offload import install_frozen_expert_offload
    from arctic_platform.model.implementations.moe.frozen_shard import install_frozen_layer_shards
    from tests.model import test_frozen_offload as offload_tests

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=world)
    ds_groups._EXPERT_PARALLEL_GROUP["ep_size_2"] = dist.group.WORLD
    models = reference, held, other = [offload_tests._LM(seed, rank) for seed in (0, 0, 1)]
    for seed, model in zip((0, 0, 1), models):
        for i, layer in enumerate(model.model.layers):
            # Frozen and replicated: the same on every rank (each EP rank's experts are its own).
            weight = layer._checkpoint_wrapped_module.proj.weight
            generator = torch.Generator().manual_seed(10 * seed + i)
            weight.data = (torch.randn(weight.shape, generator=generator) * 0.02).to(weight)
            weight.requires_grad_(False)
    want = {f: offload_tests._export(reference, f, k) for f, k in offload_tests._EXPORT_FORMATS}
    install_frozen_layer_shards(held.model.layers, root=held)
    install_frozen_expert_offload(held.model.layers, root=held)
    x = torch.randn(sum(offload_tests.COUNTS), offload_tests.DIM, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        offload_tests._step(reference, x)
        offload_tests._step(held, x)
    inner = held.model.layers[1]._checkpoint_wrapped_module
    assert inner.proj.weight.numel() == 0 and inner.experts.w1.numel() == 0
    for f, k in offload_tests._EXPORT_FORMATS:
        offload_tests._same(offload_tests._export(held, f, k), want[f])
    with whole_frozen_weights(held):
        saved = io.BytesIO()  # as DeepSpeed's checkpoint saves the module state and the frozen parameters
        torch.save({"module": held.state_dict(), "frozen": [p.detach().cpu() for p in held.parameters()]}, saved)
        saved.seek(0)
        state = torch.load(saved, weights_only=True)["module"]
        assert all(
            torch.equal(state[key].cpu().view(torch.uint8), value.cpu().view(torch.uint8))
            for key, value in reference.state_dict().items()
        )
        held.load_state_dict(other.state_dict())
    assert inner.proj.weight.numel() == 0 and inner.experts.w1.numel() == 0
    for f, k in offload_tests._EXPORT_FORMATS:
        offload_tests._same(offload_tests._export(held, f, k), offload_tests._export(other, f, k))
    assert torch.equal(held(x), other(x))
    ds_groups._EXPERT_PARALLEL_GROUP.pop("ep_size_2", None)
    results[rank] = True
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_a_model_with_sharded_weights_and_offloaded_experts_exports_and_saves_the_whole_model(tmp_path):
    world = 2
    results = mp.Manager().dict()
    mp.spawn(_composite_worker, args=(world, f"file://{tmp_path}/rendezvous", results), nprocs=world, join=True)
    assert dict(results) == {0: True, 1: True}


def test_a_sharded_model_without_an_export_contract_exports_the_unsharded_models_weights_cpu(tmp_path):
    """The generic export of a model without a family contract materializes the held parameters too."""
    _spawn_export("cpu", False, tmp_path)
