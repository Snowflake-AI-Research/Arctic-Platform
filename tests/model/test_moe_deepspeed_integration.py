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

import os
import sys
from datetime import timedelta
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from arctic_platform.model.implementations.moe import deepspeed_integration as common
from arctic_platform.testing_utils import require_torch_gpu


def _mock_ep_all_gather(monkeypatch, shards):
    flat_shards = [shard.detach().contiguous().view(-1) for shard in shards]
    state = {"offset": 0, "calls": []}

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: len(flat_shards))

    def all_gather(outputs, local, *, group=None):
        del group
        offset = state["offset"]
        width = local.numel()
        assert len(outputs) == len(flat_shards)
        assert torch.equal(local, flat_shards[0][offset : offset + width])
        for output, shard in zip(outputs, flat_shards, strict=True):
            assert output.shape == local.shape
            assert output.dtype == local.dtype
            output.copy_(shard[offset : offset + width])
        state["calls"].append(sum(output.numel() * output.element_size() for output in outputs))
        state["offset"] += width

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    return state


def test_ep_weight_gather_to_cpu_preserves_order_layout_dtype_and_bound(monkeypatch):
    shards = [(torch.arange(15, dtype=torch.int64) + rank * 100).reshape(3, 5).transpose(0, 1) for rank in range(3)]
    assert not shards[0].is_contiguous()
    expected = torch.cat(shards, dim=0)
    max_gather_bytes = 64
    state = _mock_ep_all_gather(monkeypatch, shards)
    monkeypatch.setattr(
        torch,
        "cat",
        lambda *_args, **_kwargs: pytest.fail("bounded gather must copy directly into its CPU destination"),
    )

    result = common._gather_ep_weight_to_cpu(
        shards[0],
        object(),
        is_master=True,
        max_gather_bytes=max_gather_bytes,
    )

    assert result is not None
    assert result.device.type == "cpu"
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    assert torch.equal(result, expected)
    assert state["offset"] == shards[0].numel()
    assert len(state["calls"]) > 1
    assert state["calls"][-1] < state["calls"][0]
    assert max(state["calls"]) <= max_gather_bytes


def test_ep_weight_gather_to_cpu_preserves_bfloat16_and_nonmaster_schedule(monkeypatch):
    shards = [
        (torch.arange(10, dtype=torch.float32) + rank * 20).reshape(2, 5).to(torch.bfloat16) for rank in range(2)
    ]
    expected = torch.cat(shards, dim=0)

    master = _mock_ep_all_gather(monkeypatch, shards)
    result = common._gather_ep_weight_to_cpu(shards[0], object(), is_master=True, max_gather_bytes=24)
    master_calls = list(master["calls"])

    nonmaster = _mock_ep_all_gather(monkeypatch, shards)
    discarded = common._gather_ep_weight_to_cpu(shards[0], object(), is_master=False, max_gather_bytes=24)

    assert result is not None
    assert result.dtype == torch.bfloat16
    assert torch.equal(result, expected)
    assert discarded is None
    assert nonmaster["offset"] == shards[0].numel()
    assert nonmaster["calls"] == master_calls


@require_torch_gpu
@pytest.mark.parametrize("is_master", [True, False])
def test_ep_weight_gather_to_cpu_nccl_uses_bounded_packed_buffer(monkeypatch, is_master):
    shards = [
        (torch.arange(15, dtype=torch.float32, device="cuda") + rank * 100)
        .reshape(3, 5)
        .transpose(0, 1)
        .to(torch.bfloat16)
        for rank in range(3)
    ]
    expected = torch.cat([shard.cpu() for shard in shards], dim=0)
    flat_shards = [shard.contiguous().view(-1) for shard in shards]
    group = object()
    state = {"offset": 0, "bytes": []}
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 3)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group=None: "nccl")
    monkeypatch.setattr(
        torch.distributed,
        "all_gather",
        lambda *_args, **_kwargs: pytest.fail("NCCL export must not allocate a second packed buffer"),
    )

    def all_gather_into_tensor(outputs, inputs, *, group):
        assert group is not None and outputs.ndim == inputs.ndim == 1
        assert outputs.is_contiguous() and inputs.is_contiguous()
        width = inputs.numel()
        assert outputs.numel() == 3 * width
        assert torch.equal(inputs, flat_shards[0][state["offset"] : state["offset"] + width])
        for rank, shard in enumerate(flat_shards):
            outputs[rank * width : (rank + 1) * width].copy_(shard[state["offset"] : state["offset"] + width])
        state["offset"] += width
        state["bytes"].append(outputs.numel() * outputs.element_size())

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", all_gather_into_tensor)

    result = common._gather_ep_weight_to_cpu(shards[0], group, is_master=is_master, max_gather_bytes=24)

    assert state["offset"] == shards[0].numel()
    assert state["bytes"] == [24, 24, 24, 18]
    if is_master:
        assert result is not None and result.device.type == "cpu"
        assert result.dtype == torch.bfloat16 and torch.equal(result, expected)
    else:
        assert result is None


def test_ep_weight_gather_to_cpu_handles_empty_and_single_rank(monkeypatch):
    empty = torch.empty((0, 3), dtype=torch.bfloat16)
    empty_master_state = _mock_ep_all_gather(monkeypatch, [empty, empty.clone()])

    gathered_empty = common._gather_ep_weight_to_cpu(empty, object(), is_master=True)

    assert gathered_empty is not None
    assert gathered_empty.shape == (0, 3)
    assert gathered_empty.dtype == empty.dtype
    assert torch.equal(gathered_empty, empty)
    assert empty_master_state["calls"] == []

    empty_nonmaster_state = _mock_ep_all_gather(monkeypatch, [empty, empty.clone()])

    discarded_empty = common._gather_ep_weight_to_cpu(empty, object(), is_master=False)

    assert discarded_empty is None
    assert empty_nonmaster_state["calls"] == empty_master_state["calls"]

    local = torch.arange(6, dtype=torch.int64).reshape(2, 3)
    single_master_state = _mock_ep_all_gather(monkeypatch, [local])

    gathered_local = common._gather_ep_weight_to_cpu(local, object(), is_master=True, max_gather_bytes=48)

    assert gathered_local is not None
    assert torch.equal(gathered_local, local)
    assert max(single_master_state["calls"], default=0) <= 48

    single_nonmaster_state = _mock_ep_all_gather(monkeypatch, [local])

    discarded_local = common._gather_ep_weight_to_cpu(local, object(), is_master=False, max_gather_bytes=48)

    assert discarded_local is None
    assert single_nonmaster_state["calls"] == single_master_state["calls"]


def test_ep_weight_gather_to_cpu_rejects_budget_smaller_than_one_group_element(monkeypatch):
    local = torch.arange(4, dtype=torch.int64).reshape(1, 4)
    state = _mock_ep_all_gather(monkeypatch, [local, local + 10])

    with pytest.raises(ValueError, match="max_gather_bytes"):
        common._gather_ep_weight_to_cpu(local, object(), is_master=True, max_gather_bytes=15)

    assert state["calls"] == []


def test_hf_export_delegates_tagged_expert_weights_to_cpu_gather(monkeypatch):
    converted = {}

    class ExportableModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Parameter(torch.arange(2, dtype=torch.float32))
            self.expert = nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16))
            self.expert.group_name = "ep_size_2"
            self.expert.allreduce = False

        @staticmethod
        def convert_layer_to_hf(layer_state, layer_index):
            converted["layer_index"] = layer_index
            converted["expert_device"] = layer_state["expert"].device.type
            return layer_state

    group = object()
    groups_module = ModuleType("deepspeed.utils.groups")
    groups_module._get_expert_parallel_group = lambda name: group
    utils_module = ModuleType("deepspeed.utils")
    utils_module.groups = groups_module
    deepspeed_module = ModuleType("deepspeed")
    deepspeed_module.utils = utils_module
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed_module)
    monkeypatch.setitem(sys.modules, "deepspeed.utils", utils_module)
    monkeypatch.setitem(sys.modules, "deepspeed.utils.groups", groups_module)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    calls = []

    def gather(tensor, ep_group, *, is_master, max_gather_bytes=64 * 1024 * 1024):
        calls.append((tensor, ep_group, is_master, max_gather_bytes))
        return tensor.detach().cpu().contiguous()

    monkeypatch.setattr(common, "_gather_ep_weight_to_cpu", gather)
    model = ExportableModel()

    exported = dict(common.build_iter_full_hf_weights(model)())

    assert list(exported) == ["dense", "expert"]
    assert converted == {"layer_index": -1, "expert_device": "cpu"}
    assert len(calls) == 1
    tensor, ep_group, is_master, max_gather_bytes = calls[0]
    assert tensor is model.expert
    assert ep_group is group
    assert is_master is True
    assert max_gather_bytes == 64 * 1024 * 1024


def _gloo_ep_weight_gather_worker(rank, world_size, init_method):
    import torch.distributed as dist

    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        local = (torch.arange(5, dtype=torch.int64) + rank * 100).reshape(1, 5)
        result = common._gather_ep_weight_to_cpu(
            local,
            dist.group.WORLD,
            is_master=rank == 0,
            max_gather_bytes=32,
        )
        if rank == 0:
            expected = torch.cat(
                [torch.arange(5, dtype=torch.int64), torch.arange(5, dtype=torch.int64) + 100]
            ).reshape(2, 5)
            assert result is not None
            assert result.device.type == "cpu"
            assert torch.equal(result, expected)
        else:
            assert result is None
    finally:
        dist.destroy_process_group()


def test_ep_weight_gather_to_cpu_two_rank_gloo(tmp_path):
    mp.spawn(
        _gloo_ep_weight_gather_worker,
        args=(2, (tmp_path / "ep-gather").as_uri()),
        nprocs=2,
        join=True,
    )


def test_shared_setup_runs_family_hooks_in_order(monkeypatch):
    events = []
    model = nn.Module()

    def record(name, result=None):
        def call(*_args, **_kwargs):
            events.append(name)
            return result

        return call

    monkeypatch.setattr(common, "apply_ep_with_mesh", record("ep"))
    monkeypatch.setattr(common, "convert_dtensors_to_local", record("convert", 0))
    monkeypatch.setattr(common, "enable_tiled_mlp", record("tiled"))
    adapter = common.MoEDeepSpeedAdapter(
        dtype_map={"bfloat16": torch.bfloat16},
        get_model=record("model", model),
        configure_moe_ep_backend=record("moe_backend"),
        configure_family_backend=record("family_backend"),
        inject_lm_head=record("lm_head"),
        apply_sequence_parallelism=record("sp"),
        apply_ac=record("ac"),
        load_dcp_from_hf=record("weights"),
        reset_runtime_moe_buffers=record("reset"),
        shared_expert_type=nn.Linear,
        shared_expert_forward=lambda module, value: module(value),
        build_model_config=lambda *_args: None,
    )
    config = SimpleNamespace(
        optimization_dtype="bfloat16",
        fused_lm_head_token_chunk_size="disabled",
        fp32_lm_head=False,
        ac=object(),
    )

    common.setup_model_local_no_train(
        adapter,
        config,
        SimpleNamespace(ep_enabled=True),
        object(),
        sp_size=2,
        sp_group=object(),
    )

    assert events == [
        "model",
        "moe_backend",
        "family_backend",
        "lm_head",
        "ep",
        "sp",
        "ac",
        "weights",
        "reset",
        "convert",
        "tiled",
    ]


def test_shared_expert_tagging(monkeypatch):
    class DummyMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.Linear(2, 2)

    monkeypatch.setattr(common, "MoE", DummyMoE)
    monkeypatch.setattr(common, "LatentMoE", type("DummyLatentMoE", (nn.Module,), {}))
    model = nn.Sequential(DummyMoE())

    count = common.tag_expert_params_for_deepspeed(model, "ep_size_2")

    assert count == 2
    assert all(
        parameter.group_name == "ep_size_2" and parameter.allreduce is False
        for parameter in model[0].experts.parameters()
    )


def test_family_defaults_remain_distinct():
    from arctic_platform.model.implementations.glm52 import deepspeed_integration as glm
    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen
    from arctic_platform.model.loaders.glm_moe_dsa import GlmMoeDsaOptions
    from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions

    qwen_config = qwen._build_model_config("qwen", 2, 4, "bfloat16", "flash_attention_3", Qwen3_5MoeOptions())
    glm_config = glm._build_model_config("glm", 2, 4, "bfloat16", "flash_attention_2", GlmMoeDsaOptions())

    assert (qwen_config.attn, qwen_config.moe_use_grouped_mm) == (
        "flash_attention_3",
        True,
    )
    assert (
        glm_config.attn,
        glm_config.moe_use_grouped_mm,
        glm_config.sparse_mla_backend,
    ) == ("flash_attention_2", False, "ref")


def test_qwen_adapter_keeps_sp_and_vllm_export(monkeypatch):
    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen

    calls = []
    monkeypatch.setattr(
        qwen,
        "apply_sequence_parallelism",
        lambda model, size, group: calls.append((model, size, group)),
    )
    model, group = nn.Module(), object()
    qwen._apply_sequence_parallelism(model, 2, group)

    assert calls == [(model, 2, group)]
    assert qwen._adapter().vllm_weight_export is qwen._build_iter_full_vllm_weights


def test_selective_checkpointing_falls_back_for_glm():
    from arctic_platform.model.implementations.moe.layers.checkpointing import (
        supports_selective_activation_checkpointing,
    )

    layer_type = type(
        "DecoderLayer",
        (nn.Module,),
        {"__module__": "arctic_platform.model.implementations.glm52.models.decoder"},
    )

    assert supports_selective_activation_checkpointing(layer_type()) is False


@pytest.mark.parametrize("use_family_override", [False, True])
def test_shared_loader_registers_family_hf_export_or_unchanged_default(monkeypatch, use_family_override):
    from arctic_platform.model.weight_export import weight_export_contract

    model = nn.Module()

    def default_builder(built_model):
        return iter(())

    def family_builder(built_model):
        return iter(())

    def sampler_builder(built_model):
        return iter(())

    adapter = SimpleNamespace(
        hf_weight_export=family_builder if use_family_override else None,
        vllm_weight_export=sampler_builder,
    )
    monkeypatch.setattr(common, "setup_model_local_no_train", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(common, "tag_expert_params_for_deepspeed", lambda *_args: 0)
    monkeypatch.setattr(common, "build_iter_full_hf_weights", default_builder)

    loaded = common.load_moe_model_for_deepspeed(adapter, object(), object(), object(), "ep_size_2")

    assert loaded is model
    contract = weight_export_contract(model)
    assert contract is not None
    assert contract.hf is (family_builder if use_family_override else default_builder)
    assert contract.vllm is sampler_builder


def test_qwen_mtp_hf_export_override_does_not_leak_to_generic_family():
    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen

    assert qwen._adapter().hf_weight_export is qwen._build_iter_full_hf_weights_with_mtp
    assert qwen._generic_adapter().hf_weight_export is qwen._build_iter_full_hf_weights
