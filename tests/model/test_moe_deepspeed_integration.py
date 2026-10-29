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
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from arctic_platform.model.implementations.moe import deepspeed_integration as common
from arctic_platform.testing_utils import require_torch_gpu


@pytest.fixture
def ep_checkpoint_engine(monkeypatch):
    class FixtureMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.expert = nn.Parameter(torch.tensor([1.0, 2.0]))
            self.frozen = nn.Parameter(torch.tensor([3.0, 4.0]), requires_grad=False)
            for parameter in self.parameters():
                parameter.allreduce = False
                parameter.group_name = "ep_size_2"

    monkeypatch.setattr(common, "MoE", FixtureMoE)
    groups = ModuleType("deepspeed.utils.groups")
    groups._get_expert_parallel_world_size = lambda name: 2
    groups._get_expert_parallel_rank = lambda name: 1
    groups._get_expert_data_parallel_rank = lambda name: 0
    constants = ModuleType("deepspeed.checkpoint.constants")
    constants.FROZEN_PARAM_FRAGMENTS = "frozen_param_fragments"
    utils = ModuleType("deepspeed.utils")
    utils.groups = groups
    deepspeed = ModuleType("deepspeed")
    deepspeed.utils = utils
    checkpoint = ModuleType("deepspeed.checkpoint")
    checkpoint.constants = constants
    for name, module in (
        ("deepspeed", deepspeed),
        ("deepspeed.utils", utils),
        ("deepspeed.utils.groups", groups),
        ("deepspeed.checkpoint", checkpoint),
        ("deepspeed.checkpoint.constants", constants),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    class CheckpointStorage:
        writer = True

        def __init__(self):
            self.saved_paths = []

        def is_data_parallel_writer(self, rank):
            assert rank == 0
            return self.writer

        def save(self, state, path):
            self.saved_paths.append(path)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, path)

        def load(self, path, map_location):
            return torch.load(path, map_location=map_location, weights_only=False)

    class Engine:
        partition_weights = False

        def __init__(self):
            self.module = FixtureMoE()
            self.checkpoint_engine = CheckpointStorage()
            self.calls = []
            self.saved_checkpoint = None

        def zero_optimization_partition_weights(self):
            return self.partition_weights

        def _save_checkpoint(self, save_dir, tag, client_state=None, exclude_frozen_parameters=False):
            self.calls.append(("save", save_dir, tag, client_state, exclude_frozen_parameters))
            self.saved_checkpoint = dict(
                module={
                    name: torch.zeros_like(parameter)
                    for name, parameter in self.module.state_dict(keep_vars=True).items()
                    if parameter.requires_grad or not exclude_frozen_parameters
                },
            )
            if not exclude_frozen_parameters:
                self.saved_checkpoint["frozen_param_fragments"] = {
                    name: torch.zeros_like(parameter)
                    for name, parameter in self.module.named_parameters()
                    if not parameter.requires_grad
                }
            return "saved"

        _save_moe_checkpoint = _save_checkpoint

        def _load_checkpoint(
            self,
            load_dir,
            tag,
            load_module_strict=True,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
            load_module_only=False,
            custom_load_fn=None,
        ):
            self.calls.append(
                ("load", load_module_strict, load_optimizer_states, load_lr_scheduler_states, load_module_only)
            )
            self.load_module_state_dict(self.saved_checkpoint, load_module_strict, custom_load_fn)
            return "loaded", dict(tag=tag)

        def load_module_state_dict(self, checkpoint, strict=True, custom_load_fn=None):
            self.calls.append(("module", strict, custom_load_fn))
            if custom_load_fn is None:
                self.module.load_state_dict(checkpoint["module"], strict=strict)
            else:
                custom_load_fn(src=checkpoint["module"], dst=self.module)
            for name, value in checkpoint.get("frozen_param_fragments", {}).items():
                self.module.get_parameter(name).data.copy_(value)

    common._patch_deepspeed_ep_checkpoints(Engine)
    return Engine()


@pytest.mark.parametrize("save_method", ["_save_moe_checkpoint", "_save_checkpoint"])
def test_ep_checkpoint_restores_rank_local_trainable_and_frozen_weights(ep_checkpoint_engine, tmp_path, save_method):
    engine = ep_checkpoint_engine
    expected = {name: parameter.detach().clone() for name, parameter in engine.module.named_parameters()}
    assert getattr(engine, save_method)(tmp_path, "tiny", dict(case="local")) == "saved"
    with torch.no_grad():
        for parameter in engine.module.parameters():
            parameter.add_(10)

    assert engine._load_checkpoint(tmp_path, "tiny") == ("loaded", dict(tag="tiny"))

    for name, parameter in engine.module.named_parameters():
        torch.testing.assert_close(parameter, expected[name], atol=0, rtol=0)
    assert not hasattr(engine, "_ap_ep_checkpoint_state")
    assert torch.count_nonzero(engine.saved_checkpoint["module"]["expert"]) == 0
    assert torch.count_nonzero(engine.saved_checkpoint["frozen_param_fragments"]["frozen"]) == 0


def test_ep_checkpoint_excluded_frozen_weights_remain_untouched(ep_checkpoint_engine, tmp_path):
    engine = ep_checkpoint_engine
    engine._save_moe_checkpoint(tmp_path, "tiny", exclude_frozen_parameters=True)
    with torch.no_grad():
        engine.module.expert.add_(10)
        engine.module.frozen.fill_(23)

    engine._load_checkpoint(tmp_path, "tiny", load_module_strict=False)

    torch.testing.assert_close(engine.module.expert, torch.tensor([1.0, 2.0]), atol=0, rtol=0)
    torch.testing.assert_close(engine.module.frozen, torch.full((2,), 23.0), atol=0, rtol=0)


def test_ep_checkpoint_preserves_custom_loader_and_load_flags(ep_checkpoint_engine, tmp_path):
    engine = ep_checkpoint_engine
    engine._save_moe_checkpoint(tmp_path, "tiny")
    calls = []

    def custom_loader(*, src, dst):
        calls.append(src)
        dst.load_state_dict(src)

    engine._load_checkpoint(
        tmp_path,
        "tiny",
        load_module_strict=False,
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
        load_module_only=True,
        custom_load_fn=custom_loader,
    )

    assert len(calls) == 1
    torch.testing.assert_close(calls[0]["expert"], torch.tensor([1.0, 2.0]), atol=0, rtol=0)
    assert ("load", False, False, False, True) in engine.calls
    assert ("module", False, custom_loader) in engine.calls


@pytest.mark.parametrize("save_method", ["_save_moe_checkpoint", "_save_checkpoint"])
@pytest.mark.parametrize("placement", ["root", "block", "nested"])
@pytest.mark.parametrize("exclude_frozen", [False, True])
@pytest.mark.parametrize("custom_load", [False, True])
def test_ep_checkpoint_restores_activation_wrapped_weights(
    ep_checkpoint_engine, tmp_path, save_method, placement, exclude_frozen, custom_load
):
    engine = ep_checkpoint_engine
    if placement == "root":
        engine.module = checkpoint_wrapper(engine.module)
    elif placement == "block":
        engine.module = nn.Sequential(checkpoint_wrapper(engine.module))
    else:
        engine.module = checkpoint_wrapper(nn.Sequential(checkpoint_wrapper(engine.module)))
    expected = {name: parameter.detach().clone() for name, parameter in engine.module.named_parameters()}
    getattr(engine, save_method)(tmp_path, "wrapped", exclude_frozen_parameters=exclude_frozen)
    shard = engine.checkpoint_engine.load(engine.checkpoint_engine.saved_paths[0], torch.device("cpu"))
    assert set(shard["module"]) == {
        name for name, parameter in engine.module.named_parameters() if parameter.requires_grad or not exclude_frozen
    }
    if placement == "block":
        assert all("_checkpoint_wrapped_module" in name for name in shard["module"])
    assert all("_checkpoint_wrapped_module" not in name for name in engine.saved_checkpoint["module"])
    with torch.no_grad():
        for parameter in engine.module.parameters():
            parameter.fill_(23)

    loaded = []

    def custom_loader(*, src, dst):
        loaded.append(src)
        dst.load_state_dict(src, strict=not exclude_frozen)

    engine._load_checkpoint(
        tmp_path,
        "wrapped",
        load_module_strict=not exclude_frozen,
        custom_load_fn=custom_loader if custom_load else None,
    )

    for name, parameter in engine.module.named_parameters():
        target = torch.full_like(parameter, 23) if exclude_frozen and not parameter.requires_grad else expected[name]
        torch.testing.assert_close(parameter, target, atol=0, rtol=0)
    if custom_load:
        assert len(loaded) == 1
        assert all("_checkpoint_wrapped_module" not in name for name in loaded[0])
    assert not hasattr(engine, "_ap_ep_checkpoint_state")


def test_ep_checkpoint_uses_distinct_paths_for_same_size_groups(ep_checkpoint_engine, tmp_path):
    engine = ep_checkpoint_engine
    engine.module.frozen.group_name = "other/ep_size_2"
    engine._save_moe_checkpoint(tmp_path, "tiny")

    paths = engine.checkpoint_engine.saved_paths
    assert len(paths) == len(set(paths)) == 2
    assert all(Path(path).parent == tmp_path / "tiny" for path in paths)
    engine._load_checkpoint(tmp_path, "tiny")
    torch.testing.assert_close(engine.module.frozen, torch.tensor([3.0, 4.0]), atol=0, rtol=0)


@pytest.mark.parametrize("failure", ["missing", "version", "identity", "coverage", "shape", "non_tensor", "policy"])
def test_ep_checkpoint_rejects_incompatible_shard_before_loading(ep_checkpoint_engine, tmp_path, failure):
    engine = ep_checkpoint_engine
    engine._save_moe_checkpoint(tmp_path, "tiny")
    path = Path(engine.checkpoint_engine.saved_paths[0])
    if failure == "missing":
        path.rename(path.with_suffix(".missing"))
        error = FileNotFoundError
    else:
        state = engine.checkpoint_engine.load(path, torch.device("cpu"))
        if failure == "version":
            state["version"] = True
        elif failure == "identity":
            state["ep_rank"] = 0
        elif failure == "coverage":
            del state["module"]["expert"]
        elif failure == "shape":
            state["module"]["expert"] = torch.zeros(3)
        elif failure == "non_tensor":
            state["module"]["expert"] = [1, 2]
        else:
            state["exclude_frozen_parameters"] = None
        torch.save(state, path)
        error = ValueError

    before = len(engine.calls)
    with pytest.raises(error, match="AP expert-parallel checkpoint"):
        engine._load_checkpoint(tmp_path, "tiny")
    assert len(engine.calls) == before
    assert not hasattr(engine, "_ap_ep_checkpoint_state")


def test_ep_checkpoint_failure_clears_local_state(ep_checkpoint_engine, tmp_path):
    engine = ep_checkpoint_engine
    engine._save_moe_checkpoint(tmp_path, "tiny")

    def fail(*, src, dst):
        raise RuntimeError("custom load failed")

    with pytest.raises(RuntimeError, match="custom load failed"):
        engine._load_checkpoint(tmp_path, "tiny", custom_load_fn=fail)
    assert not hasattr(engine, "_ap_ep_checkpoint_state")


@pytest.mark.parametrize("operation", ["save", "load"])
def test_ep_checkpoint_rejects_partitioned_weights_before_original_call(ep_checkpoint_engine, tmp_path, operation):
    engine = ep_checkpoint_engine
    engine.partition_weights = True
    with pytest.raises(ValueError, match="ZeRO stage 3"):
        if operation == "save":
            engine._save_moe_checkpoint(tmp_path, "tiny")
        else:
            engine._load_checkpoint(tmp_path, "tiny")
    assert engine.calls == []


def test_ep_checkpoint_nonwriter_does_not_save_shards(ep_checkpoint_engine, tmp_path):
    engine = ep_checkpoint_engine
    engine.checkpoint_engine.writer = False
    engine._save_moe_checkpoint(tmp_path, "tiny")
    assert engine.checkpoint_engine.saved_paths == []
    assert len(engine.calls) == 1


def test_ep_checkpoint_dense_model_keeps_original_paths(ep_checkpoint_engine, tmp_path):
    engine = ep_checkpoint_engine
    engine.module = nn.Linear(2, 2)
    engine._save_moe_checkpoint(tmp_path, "tiny", exclude_frozen_parameters=True)
    engine._load_checkpoint(tmp_path, "tiny")
    assert engine.checkpoint_engine.saved_paths == []
    assert engine.calls[-1] == ("module", True, None)


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
