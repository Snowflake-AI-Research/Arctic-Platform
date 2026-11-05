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
import time
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
from arctic_platform.testing_utils import require_deepspeed
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import require_torch_multi_gpu
from arctic_platform.testing_utils import torch_assert_equal


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

    assert (qwen_config.attn, qwen_config.moe_use_grouped_mm, qwen_config.moe_use_sonicmoe) == (
        "flash_attention_3",
        True,
        False,
    )
    assert (
        glm_config.attn,
        glm_config.moe_use_grouped_mm,
        glm_config.moe_use_sonicmoe,
        glm_config.sparse_mla_backend,
    ) == ("flash_attention_2", False, False, "ref")


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
    runtime = ModuleType("deepspeed.runtime")
    runtime_utils = ModuleType("deepspeed.runtime.utils")
    runtime_utils.is_optimized_parameter = lambda parameter: parameter.requires_grad and parameter.numel() > 0
    runtime.utils = runtime_utils
    for name, module in (
        ("deepspeed", deepspeed),
        ("deepspeed.utils", utils),
        ("deepspeed.utils.groups", groups),
        ("deepspeed.checkpoint", checkpoint),
        ("deepspeed.checkpoint.constants", constants),
        ("deepspeed.runtime", runtime),
        ("deepspeed.runtime.utils", runtime_utils),
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
            self.optimizer = SimpleNamespace(update_lp_params=lambda: None, partition_count=[1])
            self.calls = []
            self.saved_checkpoint = None
            self.native_expert_names = set()

        def zero_optimization_partition_weights(self):
            return self.partition_weights

        def zero_load_from_fp32_weights(self):
            return True

        def _load_zero_checkpoint(self, load_dir, tag, load_optimizer_states=True):
            return getattr(self, "zero_load_result", True)

        def _get_zero_ckpt_name(self, directory, tag):
            return str(Path(directory) / tag / "zero_pp_rank_1_mp_rank_00_optim_states.pt")

        @property
        def param_names(self):
            return {parameter: name for name, parameter in self.module.named_parameters()}

        def _get_zero_frozen_param_attributes(self, attr_func):
            return {
                self.param_names[parameter]: attr_func(parameter)
                for parameter in self.module.parameters()
                if not runtime_utils.is_optimized_parameter(parameter)
            }

        def _get_non_moe_state_dict(self, full_state_dict):
            for name in self.native_expert_names:
                full_state_dict.pop(name, None)
            return full_state_dict

        def _save_checkpoint(self, save_dir, tag, client_state=None, exclude_frozen_parameters=False):
            self.calls.append(("save", save_dir, tag, client_state, exclude_frozen_parameters))
            self.saved_checkpoint = dict(
                module=self.module_state_dict(exclude_frozen_parameters=exclude_frozen_parameters),
            )
            if not exclude_frozen_parameters:
                self.saved_checkpoint["frozen_param_fragments"] = self._get_zero_frozen_param_attributes(
                    lambda parameter: parameter.detach().cpu().clone()
                )
            return "saved"

        def module_state_dict(self, destination=None, prefix="", keep_vars=False, exclude_frozen_parameters=False):
            frozen_names = {
                common.canonical_parameter_name(name)
                for name, parameter in self.module.named_parameters()
                if not parameter.requires_grad
            }
            return {
                name: (
                    torch.zeros_like(parameter)
                    if getattr(parameter, "allreduce", True) is False
                    else parameter.detach().clone()
                )
                for name, parameter in self.module.state_dict(keep_vars=True).items()
                if name not in frozen_names or not exclude_frozen_parameters
            }

        def _save_moe_checkpoint(self, save_dir, tag, client_state=None, exclude_frozen_parameters=False):
            result = fixture_save_checkpoint(self, save_dir, tag, client_state, exclude_frozen_parameters)
            self.saved_checkpoint["module"] = self._get_non_moe_state_dict(self.saved_checkpoint["module"])
            return result

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

        def load_checkpoint(
            self,
            load_dir,
            tag,
            load_module_strict=True,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
            load_module_only=False,
            custom_load_fn=None,
        ):
            result = self._load_checkpoint(
                load_dir,
                tag,
                load_module_strict,
                load_optimizer_states,
                load_lr_scheduler_states,
                load_module_only,
                custom_load_fn,
            )
            # Native DeepSpeed loads ZeRO only after its private module-loading call returns.
            if load_optimizer_states and not load_module_only:
                self._load_zero_checkpoint(load_dir, tag, load_optimizer_states)
                if getattr(self, "fail_after_zero_load", False):
                    raise RuntimeError("native checkpoint completion failed")
            return result

        def load_module_state_dict(self, checkpoint, strict=True, custom_load_fn=None):
            self.calls.append(("module", strict, custom_load_fn))
            if custom_load_fn is None:
                self.module.load_state_dict(checkpoint["module"], strict=strict)
            else:
                custom_load_fn(src=checkpoint["module"], dst=self.module)
            if checkpoint.get("frozen_param_fragments") is not None:
                for parameter in self.module.parameters():
                    if not runtime_utils.is_optimized_parameter(parameter):
                        parameter.data.copy_(checkpoint["frozen_param_fragments"][self.param_names[parameter]])

    fixture_save_checkpoint = Engine._save_checkpoint
    original_save_shard = common._save_ep_checkpoint_shard
    engine = Engine()

    def save_shard(state, path):
        engine.checkpoint_engine.saved_paths.append(str(path))
        return original_save_shard(state, path)

    monkeypatch.setattr(common, "_save_ep_checkpoint_shard", save_shard)
    common._patch_deepspeed_ep_checkpoints(Engine)
    return engine


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
    assert not hasattr(engine, "_ap_ep_checkpoint_save_parameters")
    assert "expert" not in engine.saved_checkpoint["module"]
    assert "frozen" not in engine.saved_checkpoint["frozen_param_fragments"]


@pytest.mark.parametrize("save_method", ["_save_moe_checkpoint", "_save_checkpoint"])
def test_ep_checkpoint_does_not_serialize_unrelated_flat_storage(ep_checkpoint_engine, tmp_path, save_method):
    engine = ep_checkpoint_engine
    flat = torch.arange(1024 * 1024, dtype=torch.float32)
    engine.module.expert.data = flat[5:7]
    engine.module.frozen.data = flat[17:19]
    expected = {name: parameter.detach().clone() for name, parameter in engine.module.named_parameters()}

    getattr(engine, save_method)(tmp_path, "compact")

    path = Path(engine.checkpoint_engine.saved_paths[-1])
    assert path.stat().st_size < 65536
    with torch.no_grad():
        flat.fill_(23)
    engine._load_checkpoint(tmp_path, "compact")
    for name, parameter in engine.module.named_parameters():
        torch.testing.assert_close(parameter, expected[name], atol=0, rtol=0)


@pytest.mark.parametrize("placement", ["plain", "root", "block", "nested"])
@pytest.mark.parametrize("exclude_frozen", [False, True])
@pytest.mark.parametrize("save_method", ["_save_moe_checkpoint", "_save_checkpoint"])
def test_ep_checkpoint_filters_experts_before_dense_save_and_preserves_dense_state(
    ep_checkpoint_engine, tmp_path, placement, exclude_frozen, save_method
):
    engine = ep_checkpoint_engine
    engine.module.add_module("dense", nn.Linear(2, 2))
    engine.module.register_buffer("router_bias", torch.tensor([5.0, 6.0]))
    if placement == "root":
        engine.module = checkpoint_wrapper(engine.module)
    elif placement == "block":
        engine.module = nn.Sequential(checkpoint_wrapper(engine.module))
    elif placement == "nested":
        engine.module = checkpoint_wrapper(nn.Sequential(checkpoint_wrapper(engine.module)))
    expected = {name: value.detach().clone() for name, value in engine.module.state_dict().items()}
    parameters = {
        common.canonical_parameter_name(name): parameter for name, parameter in engine.module.named_parameters()
    }
    expert_names = {name for name, parameter in parameters.items() if getattr(parameter, "allreduce", True) is False}

    getattr(engine, save_method)(tmp_path, "filtered", exclude_frozen_parameters=exclude_frozen)

    dense_state = engine.saved_checkpoint["module"]
    assert set(dense_state) == set(expected) - expert_names
    for name, value in dense_state.items():
        torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
    shard = engine.checkpoint_engine.load(engine.checkpoint_engine.saved_paths[-1], torch.device("cpu"))
    shard_names = {common.canonical_parameter_name(name) for name in shard["module"]}
    assert shard_names == {name for name in expert_names if parameters[name].requires_grad or not exclude_frozen}
    with torch.no_grad():
        for value in engine.module.state_dict().values():
            value.fill_(23)

    engine._load_checkpoint(tmp_path, "filtered", load_module_strict=not exclude_frozen)

    for name, value in engine.module.state_dict().items():
        parameter = parameters.get(name)
        target = (
            torch.full_like(value, 23)
            if exclude_frozen and parameter is not None and not parameter.requires_grad
            else expected[name]
        )
        torch.testing.assert_close(value, target, atol=0, rtol=0)

    assert not hasattr(engine, "_ap_ep_checkpoint_save_parameters")
    assert set(engine.module_state_dict()) == set(expected)


@pytest.mark.parametrize("save_method", ["_save_moe_checkpoint", "_save_checkpoint"])
@pytest.mark.parametrize("failure_stage", ["dense", "sidefile"])
def test_ep_checkpoint_failed_save_preserves_non_save_exports(
    ep_checkpoint_engine, tmp_path, monkeypatch, save_method, failure_stage
):
    engine = ep_checkpoint_engine
    engine.module.add_module("dense", nn.Linear(2, 2))
    expected_names = set(engine.module_state_dict())
    assert expected_names == {"expert", "frozen", "dense.weight", "dense.bias"}

    def fail(*args, **kwargs):
        raise RuntimeError("checkpoint write failed")

    with monkeypatch.context() as patch:
        if failure_stage == "dense":
            patch.setattr(engine.module, "state_dict", fail)
        else:
            patch.setattr(common, "_save_ep_checkpoint_shard", fail)
        with pytest.raises(RuntimeError, match="checkpoint write failed"):
            getattr(engine, save_method)(tmp_path, "failed")

    assert not hasattr(engine, "_ap_ep_checkpoint_save_parameters")
    assert set(engine.module_state_dict()) == expected_names
    assert getattr(engine, save_method)(tmp_path, "retry") == "saved"
    assert set(engine.module_state_dict()) == expected_names


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_ep_checkpoint_streams_exact_tensor_bytes(tmp_path, monkeypatch, dtype, strided):
    flat = torch.arange(2048, dtype=torch.float32).to(dtype)
    expert = flat.reshape(32, 64)
    if strided:
        expert = expert.t()
    expected = expert.clone()
    calls = []
    original_write = os.pwritev

    def partial_write(descriptor, buffers, offset):
        count = sum(len(buffer) for buffer in buffers)
        calls.append(count)
        return original_write(descriptor, [buffers[0][: max(1, count // 2)]], offset)

    monkeypatch.setattr(os, "pwritev", partial_write)
    path = tmp_path / "experts.pt"
    common._save_ep_checkpoint_shard({"version": 1, "module": {"expert": expert, "empty": expert[:0]}}, path)
    loaded = torch.load(path, weights_only=True)

    assert loaded["version"] == 1
    assert calls and max(calls) <= 16 * 1024 * 1024
    torch.testing.assert_close(loaded["module"]["expert"], expected, atol=0, rtol=0)
    torch.testing.assert_close(loaded["module"]["empty"], expert[:0], atol=0, rtol=0)
    assert not list(tmp_path.glob(".ap-ep-checkpoint-*"))


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ep_checkpoint_layout_allocates_no_real_expert_storage(tmp_path, monkeypatch, deterministic, dtype):
    from torch._subclasses.fake_tensor import FakeTensor

    expert = torch.arange(2048, dtype=torch.float32).to(dtype).reshape(32, 64).t()
    original_save = torch.save

    def metadata_only_save(state, *args, **kwargs):
        assert all(isinstance(tensor, FakeTensor) for tensor in state["module"].values())
        return original_save(state, *args, **kwargs)

    monkeypatch.setattr(torch, "save", metadata_only_save)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_fill = torch.utils.deterministic.fill_uninitialized_memory
    path = tmp_path / "experts.pt"
    try:
        torch.use_deterministic_algorithms(deterministic)
        torch.utils.deterministic.fill_uninitialized_memory = True
        common._save_ep_checkpoint_shard({"module": {"expert": expert, "empty": expert[:0]}}, path)
    finally:
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = previous_fill

    loaded = torch.load(path, weights_only=True)
    torch.testing.assert_close(loaded["module"]["expert"], expert, atol=0, rtol=0)
    torch.testing.assert_close(loaded["module"]["empty"], expert[:0], atol=0, rtol=0)


def test_ep_checkpoint_streaming_failure_keeps_previous_shard(tmp_path, monkeypatch):
    path = tmp_path / "experts.pt"
    previous = {"module": {"expert": torch.tensor([1.0, 2.0])}}
    torch.save(previous, path)
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("checkpoint storage full")

    monkeypatch.setattr(os, "pwritev", fail)
    with pytest.raises(OSError, match="checkpoint storage full"):
        common._save_ep_checkpoint_shard({"module": {"expert": torch.tensor([3.0, 4.0])}}, path)

    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".ap-ep-checkpoint-*"))


def test_ep_checkpoint_streaming_rejects_stalled_write(tmp_path, monkeypatch):
    path = tmp_path / "experts.pt"
    monkeypatch.setattr(os, "pwritev", lambda *args: 0)

    with pytest.raises(OSError, match="write made no progress"):
        common._save_ep_checkpoint_shard({"module": {"expert": torch.tensor([1.0])}}, path)

    assert not path.exists()
    assert not list(tmp_path.glob(".ap-ep-checkpoint-*"))


@pytest.mark.parametrize("shape", [(3, 5), (2, 3, 5), (2, 1, 3, 5)])
def test_ep_checkpoint_tensor_chunks_bound_strided_copies(shape):
    tensor = torch.arange(30, dtype=torch.float32).reshape(2, 3, 5)
    if len(shape) == 2:
        tensor = tensor[0]
    else:
        tensor = tensor.reshape(shape)
    tensor = tensor.transpose(0, -1)
    expected = tensor.contiguous().view(-1).view(torch.uint8)

    chunks = list(common._ep_checkpoint_tensor_chunks(tensor, 16))

    assert all(chunk.numel() <= 16 for chunk in chunks)
    torch.testing.assert_close(torch.cat(chunks), expected, atol=0, rtol=0)


def _gpu_ep_checkpoint_writer_worker(rank, directory, large_flat_view):
    from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine

    torch.cuda.set_device(rank)
    if large_flat_view:
        elements = 18 * 2048 * 4096
        flat = torch.full((2 * elements,), rank + 1, dtype=torch.bfloat16, device="cuda")
        source = flat[1024 : 1024 + elements].reshape(18, 2048, 4096)
        source[0, 0, 0] = -(rank + 1)
    else:
        flat = torch.arange(9 * 1024 * 1024, dtype=torch.float32, device="cuda")
        source = flat[1024 : 1024 + 8 * 1024 * 1024].reshape(2048, 4096).t()
    path = Path(directory) / f"rank-{rank}.pt"
    writes = []
    original_write = os.pwritev

    def bounded_write(descriptor, buffers, offset):
        count = sum(len(buffer) for buffer in buffers)
        assert 0 < count <= 16 * 1024 * 1024
        writes.append(count)
        return original_write(descriptor, buffers, offset)

    os.pwritev = bounded_write
    try:
        common._save_ep_checkpoint_shard(dict(version=1, module=dict(expert=source)), path)
    finally:
        os.pwritev = original_write
    assert len(writes) >= 2
    assert path.stat().st_size < source.numel() * source.element_size() + 65536
    loaded = TorchCheckpointEngine().load(str(path), map_location=torch.device("cpu"))
    torch.testing.assert_close(loaded["module"]["expert"], source.cpu(), atol=0, rtol=0)
    print(f"EP_CHECKPOINT_GPU_WRITER_PASS rank={rank} chunks={len(writes)} bytes={sum(writes)}", flush=True)


@require_torch_gpu
@pytest.mark.parametrize("large_flat_view", [False, True], ids=["strided-small", "glm-first-expert-shape"])
def test_ep_checkpoint_gpu_writer_native_loader_all_visible_gpus(tmp_path, large_flat_view):
    mp.spawn(
        _gpu_ep_checkpoint_writer_worker,
        args=(str(tmp_path), large_flat_view),
        nprocs=torch.cuda.device_count(),
        join=True,
    )


def test_ep_checkpoint_filter_preserves_native_deepspeed_exclusions(ep_checkpoint_engine):
    engine = ep_checkpoint_engine
    engine.module.add_module("native_experts", nn.Linear(2, 2, bias=False))
    engine.module.register_buffer("router_bias", torch.tensor([5.0, 6.0]))
    engine.native_expert_names = {"native_experts.weight"}

    filtered = engine._get_non_moe_state_dict(engine.module.state_dict())

    assert set(filtered) == {"router_bias"}
    torch.testing.assert_close(filtered["router_bias"], torch.tensor([5.0, 6.0]), atol=0, rtol=0)


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
        common.canonical_parameter_name(name)
        for name, parameter in engine.module.named_parameters()
        if parameter.requires_grad or not exclude_frozen
    }
    assert all("_checkpoint_wrapped_module" not in name for name in shard["module"])
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


@pytest.mark.parametrize(
    "failure", ["missing", "version", "identity", "coverage", "shape", "dtype", "non_tensor", "policy"]
)
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
        elif failure == "dtype":
            state["module"]["expert"] = state["module"]["expert"].bfloat16()
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


def test_ep_checkpoint_frozen_callbacks_never_copy_experts(ep_checkpoint_engine, tmp_path):
    engine = ep_checkpoint_engine
    engine.module.add_module("dense", nn.Linear(2, 2))
    engine.module.dense.bias.requires_grad_(False)
    copied = []
    native = engine._get_zero_frozen_param_attributes

    def attributes(callback):
        def observed(parameter):
            copied.append(engine.param_names[parameter])
            return callback(parameter)

        return native(observed)

    engine._get_zero_frozen_param_attributes = attributes
    engine._save_checkpoint(tmp_path, "frozen")
    assert copied == ["dense.bias"]
    assert set(engine.saved_checkpoint["frozen_param_fragments"]) == {"dense.bias"}
    assert set(engine._get_zero_frozen_param_attributes(lambda parameter: parameter.shape)) == {"frozen", "dense.bias"}


def test_ep_checkpoint_empty_trainable_expert_restores_frozen_fragment(ep_checkpoint_engine, tmp_path):
    engine = ep_checkpoint_engine
    empty = nn.Parameter(torch.empty(0))
    empty.allreduce = False
    empty.group_name = engine.module.expert.group_name
    engine.module.register_parameter("empty_expert", empty)
    engine._save_checkpoint(tmp_path, "empty")
    assert "empty_expert" not in engine.saved_checkpoint["frozen_param_fragments"]
    assert engine._load_checkpoint(tmp_path, "empty")[0] == "loaded"
    assert engine.module.empty_expert.shape == torch.Size([0])
    assert engine.module.empty_expert.requires_grad


def test_ep_checkpoint_patch_is_idempotent(ep_checkpoint_engine):
    engine_type = type(ep_checkpoint_engine)
    save = engine_type._save_checkpoint
    load = engine_type.load_module_state_dict
    common._patch_deepspeed_ep_checkpoints(engine_type)
    assert engine_type._save_checkpoint is save
    assert engine_type.load_module_state_dict is load


@pytest.mark.parametrize("empty", [False, True])
def test_ep_checkpoint_syncs_data_before_rename_and_directory_after(tmp_path, monkeypatch, empty):
    events = []
    for name in ("fdatasync", "replace", "fsync"):
        native = getattr(common.os, name)

        def record(*args, native=native, name=name):
            events.append(name)
            return native(*args)

        monkeypatch.setattr(common.os, name, record)
    tensor = torch.empty(0) if empty else torch.ones(4)
    common._save_ep_checkpoint_shard({"module": {"expert": tensor}}, tmp_path / "expert.pt")
    assert events[-3:] == ["fdatasync", "replace", "fsync"]


def test_ep_checkpoint_syncs_only_owned_tag_directory(tmp_path, monkeypatch):
    synced = []
    native = common.os.fsync

    def record(descriptor):
        synced.append(Path(common.os.readlink(f"/proc/self/fd/{descriptor}")))
        native(descriptor)

    monkeypatch.setattr(common.os, "fsync", record)
    save_dir = tmp_path / "new-root" / "checkpoints"
    common._save_ep_checkpoint_shard({"module": {"expert": torch.ones(1)}}, save_dir / "tag" / "expert.pt")
    assert synced == [save_dir / "tag"]


@pytest.mark.parametrize("save_wrapped", [False, True])
def test_ep_checkpoint_resumes_across_activation_wrapping(ep_checkpoint_engine, tmp_path, save_wrapped):
    engine = ep_checkpoint_engine
    module = engine.module
    engine.module = nn.Sequential(checkpoint_wrapper(module) if save_wrapped else module)
    expected = {name: tensor.clone() for name, tensor in engine.module.state_dict().items()}
    engine._save_moe_checkpoint(tmp_path, "cross-wrapping")
    engine.module = nn.Sequential(module if save_wrapped else checkpoint_wrapper(module))
    with torch.no_grad():
        for parameter in engine.module.parameters():
            parameter.zero_()
    engine._load_checkpoint(tmp_path, "cross-wrapping")
    for name, tensor in engine.module.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("complete", [False, True])
def test_ep_checkpoint_legacy_dense_fallback_for_ep_one(ep_checkpoint_engine, tmp_path, monkeypatch, complete):
    engine = ep_checkpoint_engine
    groups = sys.modules["deepspeed.utils.groups"]
    monkeypatch.setattr(groups, "_get_expert_parallel_world_size", lambda name: 1)
    expected = {name: value.detach().clone() for name, value in engine.module.state_dict().items()}
    engine.saved_checkpoint = dict(module=expected if complete else {})
    with torch.no_grad():
        for parameter in engine.module.parameters():
            parameter.zero_()
    if complete:
        engine._load_checkpoint(tmp_path, "legacy")
        for name, parameter in engine.module.named_parameters():
            torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
    else:
        with pytest.raises(FileNotFoundError, match="neither a rank-local shard nor compatible dense expert"):
            engine._load_checkpoint(tmp_path, "legacy")
    assert not hasattr(engine, "_ap_ep_legacy_parameters")


@pytest.mark.parametrize("optimizer_states", [False, True])
@pytest.mark.parametrize("zero_files", [False, True])
def test_ep_checkpoint_legacy_ep_resume_requires_and_restores_zero_masters(
    ep_checkpoint_engine, tmp_path, monkeypatch, optimizer_states, zero_files
):
    engine = ep_checkpoint_engine
    engine.module.frozen.requires_grad_(True)
    expected = {name: tensor.clone() for name, tensor in engine.module.state_dict().items()}
    engine.saved_checkpoint = dict(module={name: torch.zeros_like(value) for name, value in expected.items()})
    tag = tmp_path / "legacy"
    tag.mkdir()
    if zero_files:
        torch.save(
            dict(
                optimizer_state_dict=dict(
                    partition_count=[1], base_optimizer_state=dict(param_groups=[dict(moe=True, name="ep_size_2")])
                )
            ),
            engine._get_zero_ckpt_name(tmp_path, "legacy"),
        )
    restored = []

    def update():
        engine.module.load_state_dict(expected)
        restored.append(True)

    engine.optimizer.update_lp_params = update
    if optimizer_states and zero_files:
        engine.load_checkpoint(tmp_path, "legacy", load_optimizer_states=True)
        assert restored == [True]
        for name, value in engine.module.state_dict().items():
            torch_assert_equal(value, expected[name])
    else:
        with pytest.raises(FileNotFoundError, match="rank-local shard|this rank's ZeRO optimizer file"):
            engine.load_checkpoint(tmp_path, "legacy", load_optimizer_states=optimizer_states)
    assert restored == ([True] if optimizer_states and zero_files else [])
    assert not hasattr(engine, "_ap_ep_restore_legacy_masters")


@pytest.mark.parametrize("failure", ["zero-load", "after-zero-load"])
def test_legacy_nested_restore_preserves_native_failure(ep_checkpoint_engine, tmp_path, failure):
    engine = ep_checkpoint_engine
    engine.module.frozen.requires_grad_(True)
    engine.saved_checkpoint = dict(module=engine.module.state_dict())
    path = Path(engine._get_zero_ckpt_name(tmp_path, "legacy"))
    path.parent.mkdir()
    torch.save(
        dict(
            optimizer_state_dict=dict(
                partition_count=[1], base_optimizer_state=dict(param_groups=[dict(moe=True, name="ep_size_2")])
            )
        ),
        path,
    )
    engine.zero_load_result = failure != "zero-load"
    engine.fail_after_zero_load = failure == "after-zero-load"
    message = "successfully loaded ZeRO master weights" if failure == "zero-load" else "native checkpoint completion"
    with pytest.raises(RuntimeError, match=message):
        engine.load_checkpoint(tmp_path, "legacy")
    for attribute in ("_ap_ep_restore_legacy_masters", "_ap_ep_checkpoint_state", "_ap_ep_legacy_parameters"):
        assert not hasattr(engine, attribute)


@pytest.mark.parametrize("failure", ["ep-size", "dp-size", "other-rank"])
def test_legacy_ep_resume_rejects_incompatible_layout_or_missing_rank(ep_checkpoint_engine, tmp_path, failure):
    engine = ep_checkpoint_engine
    engine.module.frozen.requires_grad_(True)
    path = Path(engine._get_zero_ckpt_name(tmp_path, "legacy"))
    path.parent.mkdir()
    if failure == "other-rank":
        path = path.with_name("zero_pp_rank_0_mp_rank_00_optim_states.pt")
    torch.save(
        dict(
            optimizer_state_dict=dict(
                partition_count=[2] if failure == "dp-size" else [1],
                base_optimizer_state=dict(
                    param_groups=[dict(moe=True, name="ep_size_4" if failure == "ep-size" else "ep_size_2")]
                ),
            )
        ),
        path,
    )
    with pytest.raises((ValueError, FileNotFoundError), match="layout|expert-parallel size|this rank"):
        engine._load_checkpoint(tmp_path, "legacy")
    assert engine.calls == []


@pytest.mark.parametrize(
    "expert,stage_three,ep_size", [(True, True, 2), (True, False, 2), (False, True, 2), (True, True, 1)]
)
def test_ap_expert_zero_three_is_rejected_before_native_model_setup(
    ep_checkpoint_engine, monkeypatch, expert, stage_three, ep_size
):
    engine = ep_checkpoint_engine
    engine_type = type(engine)
    engine.partition_weights = stage_three
    model = engine.module if expert else nn.Linear(2, 2)
    calls = []
    monkeypatch.setattr(
        common, "_ep_checkpoint_identity", lambda name: dict(group_name=name, ep_size=ep_size, ep_rank=0)
    )

    def native(self, model):
        calls.append(True)
        self.module = model
        self.has_moe_layers = False
        self.num_experts = []

    monkeypatch.setattr(engine_type, "_configure_distributed_model", native, raising=False)
    module = ModuleType("deepspeed.runtime.engine")
    module.DeepSpeedEngine = engine_type
    monkeypatch.setitem(sys.modules, "deepspeed.runtime.engine", module)
    monkeypatch.setattr(sys.modules["deepspeed.runtime"], "engine", module, raising=False)
    monkeypatch.setattr(sys.modules["deepspeed"], "runtime", sys.modules["deepspeed.runtime"], raising=False)
    common.patch_deepspeed_moe_detection()
    if expert and stage_three and ep_size > 1:
        with pytest.raises(ValueError, match="use ZeRO stage 1 or 2"):
            engine._configure_distributed_model(model)
        assert calls == []
    else:
        engine._configure_distributed_model(model)
        assert calls == [True]
        assert engine.has_moe_layers is (expert and not stage_three)


def test_ep_one_zero_three_uses_native_checkpoint_paths(ep_checkpoint_engine, monkeypatch, tmp_path):
    engine = ep_checkpoint_engine
    engine.partition_weights = True
    monkeypatch.setattr(common, "_ep_checkpoint_identity", lambda name: dict(group_name=name, ep_size=1, ep_rank=0))
    assert common._ep_checkpoint_parameters(engine.module, partition_weights=True) == {}
    engine._save_checkpoint(tmp_path, "native")
    assert "expert" in engine.saved_checkpoint["module"]
    assert engine.checkpoint_engine.saved_paths == []
    engine._load_checkpoint(tmp_path, "native", load_optimizer_states=False)
    assert engine.calls[-1][0] == "module"


@pytest.mark.parametrize("verify", ["0", "1", "invalid"])
def test_ep_checkpoint_verification_switch(monkeypatch, tmp_path, verify):
    monkeypatch.setenv("ARCTIC_EXPERT_CHECKPOINT_VERIFY", verify)
    native_load = torch.load
    readbacks = []

    def load(*args, **kwargs):
        if kwargs.get("mmap"):
            readbacks.append(True)
        return native_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    path = tmp_path / "expert.pt"
    if verify == "invalid":
        with pytest.raises(ValueError, match="ARCTIC_EXPERT_CHECKPOINT_VERIFY"):
            common._save_ep_checkpoint_shard(dict(module=dict(weight=torch.ones(2))), path)
        assert not path.exists()
    else:
        common._save_ep_checkpoint_shard(dict(module=dict(weight=torch.ones(2))), path)
        assert readbacks == [True]
        torch_assert_equal(native_load(path, weights_only=True)["module"]["weight"], torch.ones(2))


@pytest.mark.parametrize("code", [22, 38, 95])
def test_ep_checkpoint_tolerates_unsupported_directory_sync(tmp_path, monkeypatch, code):
    path = tmp_path / "existing-save" / "tag" / "expert.pt"
    path.parent.mkdir(parents=True)
    synced = []

    def sync(descriptor):
        synced.append(Path(common.os.readlink(f"/proc/self/fd/{descriptor}")))
        raise OSError(code, "unsupported directory sync")

    monkeypatch.setattr(common.os, "fsync", sync)
    common._save_ep_checkpoint_shard(dict(module=dict(expert=torch.ones(1))), path)
    assert synced == [path.parent]


@pytest.mark.parametrize("operation", ["fdatasync", "posix_fadvise"])
@pytest.mark.parametrize("code", [22, 38, 95])
def test_ep_checkpoint_tolerates_unsupported_file_sync(tmp_path, monkeypatch, operation, code):
    def unsupported(*args):
        raise OSError(code, "filesystem operation unsupported")

    monkeypatch.setattr(common.os, operation, unsupported)
    path = tmp_path / "expert.pt"
    common._save_ep_checkpoint_shard(dict(module=dict(expert=torch.ones(2))), path)
    torch_assert_equal(torch.load(path, weights_only=True)["module"]["expert"], torch.ones(2))


@pytest.mark.parametrize("verify", ["0", "1"])
def test_ep_checkpoint_rejects_shifted_in_range_storage_offset(tmp_path, monkeypatch, verify):
    monkeypatch.setenv("ARCTIC_EXPERT_CHECKPOINT_VERIFY", verify)
    native = common.os.pwritev

    def shifted(descriptor, buffers, offset):
        return native(descriptor, buffers, offset + 1)

    monkeypatch.setattr(common.os, "pwritev", shifted)
    with pytest.raises(ValueError, match="payload verification"):
        common._save_ep_checkpoint_shard(dict(module=dict(expert=torch.ones(2))), tmp_path / "expert.pt")
    assert not (tmp_path / "expert.pt").exists()


@pytest.mark.parametrize("verify", ["0", "1"])
def test_ep_checkpoint_checks_zip_boundary_for_zero_payload(tmp_path, monkeypatch, verify):
    monkeypatch.setenv("ARCTIC_EXPERT_CHECKPOINT_VERIFY", verify)
    native = common.torch.load

    def shifted(*args, **kwargs):
        state = native(*args, **kwargs)
        if not kwargs.get("mmap"):
            storage = state["module"]["expert"].untyped_storage()
            storage._checkpoint_offset += 1
        return state

    monkeypatch.setattr(common.torch, "load", shifted)
    with pytest.raises(ValueError, match="storage offset does not match ZIP record"):
        common._save_ep_checkpoint_shard(dict(module=dict(expert=torch.zeros(16))), tmp_path / "expert.pt")
    assert not (tmp_path / "expert.pt").exists()


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("old_name", [False, True])
def test_legacy_ep_optimizer_accepts_other_precision(ep_checkpoint_engine, tmp_path, monkeypatch, bf16, old_name):
    engine = ep_checkpoint_engine
    current = "bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt" if bf16 else "zero_pp_rank_1_mp_rank_00_optim_states.pt"
    monkeypatch.setattr(engine, "_get_zero_ckpt_name", lambda directory, tag: str(Path(directory) / tag / current))
    other = current.removeprefix("bf16_") if bf16 else "bf16_" + current
    if old_name:
        other = other.removesuffix("_optim_states.pt") + "optim_states.pt"
    path = tmp_path / "legacy" / other
    path.parent.mkdir()
    torch.save(
        dict(
            optimizer_state_dict=dict(
                partition_count=[1], base_optimizer_state=dict(param_groups=[dict(moe=True, name="ep_size_2")])
            )
        ),
        path,
    )
    common._validate_legacy_ep_zero_layout(engine, tmp_path, "legacy", dict(group_name="ep_size_2", ep_size=2))


@pytest.mark.parametrize("elastic", [False, True])
def test_legacy_ep_optimizer_old_filename_and_elastic_metadata(ep_checkpoint_engine, tmp_path, elastic):
    engine = ep_checkpoint_engine
    path = Path(engine._get_zero_ckpt_name(tmp_path, "legacy"))
    path.parent.mkdir()
    path = path.with_name(path.name.removesuffix("_optim_states.pt") + "optim_states.pt")
    base = dict(param_groups=[dict(moe=True, name="ep_size_2")])
    torch.save(
        dict(optimizer_state_dict=dict(partition_count=[1], base_optimizer_state=[base] if elastic else base)),
        path,
    )
    if elastic:
        with pytest.raises(ValueError, match="non-elastic"):
            common._validate_legacy_ep_zero_layout(engine, tmp_path, "legacy", dict(group_name="ep_size_2", ep_size=2))
    else:
        common._validate_legacy_ep_zero_layout(engine, tmp_path, "legacy", dict(group_name="ep_size_2", ep_size=2))


def test_legacy_ep_layout_validation_uses_mapped_optimizer_metadata(ep_checkpoint_engine, tmp_path, monkeypatch):
    engine = ep_checkpoint_engine
    path = Path(engine._get_zero_ckpt_name(tmp_path, "legacy"))
    path.parent.mkdir()
    torch.save(
        dict(
            optimizer_state_dict=dict(
                partition_count=[1],
                base_optimizer_state=dict(param_groups=[dict(moe=True, name="ep_size_2")]),
                fp32_flat_groups=[torch.ones(1024)],
            )
        ),
        path,
    )
    native_load = torch.load
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return native_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    monkeypatch.setattr(engine.checkpoint_engine, "load", lambda *args, **kwargs: pytest.fail("eager optimizer read"))
    common._validate_legacy_ep_zero_layout(engine, tmp_path, "legacy", dict(group_name="ep_size_2", ep_size=2))
    assert len(calls) == 1
    assert calls[0]["mmap"] is True


def test_ep_checkpoint_capability_failure_is_reported_before_patching(monkeypatch):
    class UnsupportedEngine:
        pass

    monkeypatch.setattr(torch.serialization, "skip_data", lambda: None)
    with pytest.raises(RuntimeError, match="startup capability probe failed"):
        common._patch_deepspeed_ep_checkpoints(UnsupportedEngine)
    assert not hasattr(UnsupportedEngine, "_ap_ep_ckpt_patched")


def test_ep_checkpoint_cleans_only_stale_workspace_for_current_shard(tmp_path):
    import os
    import time

    path = tmp_path / "expert.pt"
    stale = tmp_path / ".ap-ep-checkpoint-expert-old"
    stale.mkdir()
    recent = tmp_path / ".ap-ep-checkpoint-expert-recent"
    recent.mkdir()
    other = tmp_path / ".ap-ep-checkpoint-other-old"
    other.mkdir()
    timestamp = time.time() - 61
    for directory in (stale, other):
        os.utime(directory, (timestamp, timestamp))
    common._save_ep_checkpoint_shard(dict(module=dict(expert=torch.arange(8.0))), path)
    assert not stale.exists()
    assert recent.exists()
    assert other.exists()
    torch_assert_equal(torch.load(path, weights_only=True)["module"]["expert"], torch.arange(8.0))


def test_ep_checkpoint_sweeps_other_tags_using_owner_liveness(tmp_path):
    import os
    import time

    old_tag = tmp_path / "old-tag"
    old_tag.mkdir()
    stale = old_tag / ".ap-ep-checkpoint-expert-dead"
    live = old_tag / ".ap-ep-checkpoint-expert-live"
    other = old_tag / ".ap-ep-checkpoint-other-dead"
    timestamp = time.time() - 61
    for workspace in (stale, live, other):
        workspace.mkdir()
        (workspace / ".owner").touch()
        os.utime(workspace, (timestamp, timestamp))
        if workspace != live:
            os.utime(workspace / ".owner", (timestamp, timestamp))
    path = tmp_path / "new-tag" / "expert.pt"
    common._save_ep_checkpoint_shard(dict(module=dict(expert=torch.zeros(16))), path)
    assert not stale.exists()
    assert live.exists()
    assert other.exists()


@pytest.mark.parametrize("operation", ["stat", "remove"])
def test_ep_checkpoint_workspace_cleanup_failure_does_not_fail_save(tmp_path, monkeypatch, operation):
    messages = []
    monkeypatch.setattr(
        common, "get_logger", lambda: SimpleNamespace(info=lambda message, *a, **k: messages.append(message))
    )
    stale = tmp_path / ".ap-ep-checkpoint-expert-stale"
    stale.mkdir()
    if operation == "stat":
        native = Path.stat

        def stat(path, *args, **kwargs):
            if path == stale:
                raise PermissionError("stale workspace is inaccessible")
            return native(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", stat)
    else:
        stamp = time.time() - 61
        os.utime(stale, (stamp, stamp))
        native_remove = common.shutil.rmtree

        def remove(path, *args, **kwargs):
            if Path(path) == stale:
                raise PermissionError("stale workspace cannot be removed")
            return native_remove(path, *args, **kwargs)

        monkeypatch.setattr(common.shutil, "rmtree", remove)
    path = tmp_path / "expert.pt"
    common._save_ep_checkpoint_shard(dict(module=dict(expert=torch.arange(8.0))), path)
    torch_assert_equal(torch.load(path, weights_only=True)["module"]["expert"], torch.arange(8.0))
    assert any("Could not" in message for message in messages)


@pytest.mark.parametrize("interval,expected_syncs", [("0", 1), ("1", 3), ("33554432", 1)])
def test_ep_checkpoint_configurable_sync_interval(tmp_path, monkeypatch, interval, expected_syncs):
    monkeypatch.setenv("ARCTIC_EXPERT_CHECKPOINT_SYNC_BYTES", interval)
    calls = []
    native = common.os.fdatasync

    def sync(descriptor):
        calls.append(descriptor)
        native(descriptor)

    monkeypatch.setattr(common.os, "fdatasync", sync)
    common._save_ep_checkpoint_shard(dict(module=dict(one=torch.ones(2), two=torch.ones(3))), tmp_path / "expert.pt")
    assert len(calls) == expected_syncs


@pytest.mark.parametrize("interval", ["bad", "-1"])
def test_ep_checkpoint_rejects_invalid_sync_interval(tmp_path, monkeypatch, interval):
    monkeypatch.setenv("ARCTIC_EXPERT_CHECKPOINT_SYNC_BYTES", interval)
    with pytest.raises(ValueError, match="ARCTIC_EXPERT_CHECKPOINT_SYNC_BYTES"):
        common._save_ep_checkpoint_shard(dict(module=dict(expert=torch.ones(1))), tmp_path / "expert.pt")


@require_torch_gpu
@require_deepspeed
@pytest.mark.parametrize("moe_save", [False, True])
@pytest.mark.parametrize("exclude_frozen", [False, True])
def test_ep_checkpoint_native_deepspeed_save_load(monkeypatch, tmp_path, single_rank_group, moe_save, exclude_frozen):
    import deepspeed
    import deepspeed.utils.groups as groups

    model = nn.Module()
    model.dense = nn.Linear(2, 2)
    model.dense.bias.requires_grad_(False)
    model.experts = common.MoE.__new__(common.MoE)
    nn.Module.__init__(model.experts)
    model.experts.weight = nn.Parameter(torch.arange(4, dtype=torch.float32).reshape(2, 2))
    model.experts.frozen = nn.Parameter(torch.tensor([3.0, 4.0]), requires_grad=False)
    model.experts.empty_trainable = nn.Parameter(torch.empty(0))
    engine, *_ = deepspeed.initialize(
        model=model.cuda(),
        optimizer=torch.optim.AdamW(model.parameters()),
        config=dict(
            train_micro_batch_size_per_gpu=1,
            zero_allow_untested_optimizer=True,
            zero_optimization=dict(stage=2),
        ),
    )
    group_name = "ep_size_1"
    monkeypatch.setitem(groups._EXPERT_PARALLEL_GROUP, group_name, torch.distributed.group.WORLD)
    monkeypatch.setitem(groups._EXPERT_DATA_PARALLEL_GROUP, group_name, torch.distributed.group.WORLD)
    for parameter in model.experts.parameters():
        parameter.allreduce = False
        parameter.group_name = group_name
    engine.has_moe_layers = moe_save
    engine.num_experts = [1] if moe_save else []
    patched_methods = (
        "_get_non_moe_state_dict",
        "_get_zero_frozen_param_attributes",
        "module_state_dict",
        "_save_moe_checkpoint",
        "_save_checkpoint",
        "_load_checkpoint",
        "load_checkpoint",
        "_load_zero_checkpoint",
        "load_module_state_dict",
    )
    for name in patched_methods:
        monkeypatch.setattr(type(engine), name, getattr(type(engine), name))
    monkeypatch.setattr(type(engine), "_ap_ep_ckpt_patched", False, raising=False)
    common._patch_deepspeed_ep_checkpoints(type(engine))
    expected = {name: value.detach().clone() for name, value in model.named_parameters()}
    assert engine.save_checkpoint(str(tmp_path), tag="native", exclude_frozen_parameters=exclude_frozen)
    dense = torch.load(engine._get_ckpt_name(str(tmp_path), "native"), weights_only=False)
    assert not any(name.startswith("experts.") for name in dense["module"])
    fragments = dense.get("frozen_param_fragments")
    if fragments is not None:
        assert set(fragments) == {"dense.bias"}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100)
    path, _ = engine.load_checkpoint(
        str(tmp_path), tag="native", load_optimizer_states=False, load_module_strict=not exclude_frozen
    )
    assert path is not None
    for name, parameter in model.named_parameters():
        target = expected[name] + 100 if exclude_frozen and not parameter.requires_grad else expected[name]
        torch_assert_equal(parameter, target)


def _native_legacy_ep_resume_worker(rank, rendezvous, directory, legacy=True):
    import deepspeed
    import deepspeed.utils.groups as groups
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
    deepspeed.init_distributed(
        dist_backend="nccl",
        auto_mpi_discovery=False,
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        group_name = "ep_size_2"
        groups._EXPERT_PARALLEL_GROUP[group_name] = dist.group.WORLD
        for peer in range(2):
            group = dist.new_group([peer])
            if peer == rank:
                groups._EXPERT_DATA_PARALLEL_GROUP[group_name] = group
        model = nn.Module()
        model.dense = nn.Linear(2, 2, bias=False)
        model.experts = common.MoE.__new__(common.MoE)
        nn.Module.__init__(model.experts)
        model.experts.num_experts = 2
        model.experts.weight = nn.Parameter(torch.full((2, 2), 11.0 + rank))
        model.experts.weight.allreduce = False
        model.experts.weight.group_name = group_name
        model.cuda()
        common.patch_deepspeed_moe_detection()
        optimizer = torch.optim.AdamW(
            [
                dict(params=list(model.dense.parameters())),
                dict(params=[model.experts.weight], moe=True, name=group_name),
            ]
        )
        engine, *_ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=dict(
                train_micro_batch_size_per_gpu=1,
                zero_allow_untested_optimizer=True,
                zero_optimization=dict(stage=2),
            ),
        )
        expected = model.experts.weight.detach().clone()
        # Legacy saves use rank-zero dense experts; AP saves must preserve both ranks' distinct shards.
        if legacy:
            engine.has_moe_layers = False
        assert engine.save_checkpoint(directory, tag="legacy")
        sidefile = common._ep_checkpoint_path(directory, "legacy", common._ep_checkpoint_identity(group_name))
        shard = torch.load(sidefile, weights_only=True)
        torch_assert_equal(shard["module"]["experts.weight"], expected.cpu())
        assert shard["ep_rank"] == rank
        assert shard["ep_size"] == 2
        if legacy:
            sidefile.unlink()
        if rank == 0 and legacy:
            dense_path = engine._get_ckpt_name(directory, "legacy")
            dense = torch.load(dense_path, weights_only=False)
            dense["module"]["experts.weight"] = expected.cpu()
            torch.save(dense, dense_path)
        dist.barrier()
        with torch.no_grad():
            model.experts.weight.add_(100)
            for master in engine.optimizer.single_partition_of_fp32_groups:
                master.add_(200)
        steps = engine.global_steps
        loaded, _ = engine.load_checkpoint(directory, tag="legacy", load_optimizer_states=True)
        assert loaded is not None
        assert engine.global_steps == steps
        torch_assert_equal(model.experts.weight, expected)
        assert not hasattr(engine, "_ap_ep_restore_legacy_masters")
        print(f"LEGACY_EP_NATIVE_PRE_STEP_RESTORE_PASS rank={rank}", flush=True)
    finally:
        dist.destroy_process_group()


@require_torch_multi_gpu
@require_deepspeed
def test_legacy_ep_native_deepspeed_public_resume_restores_experts_before_step(tmp_path):
    mp.spawn(
        _native_legacy_ep_resume_worker,
        args=((tmp_path / "legacy-rendezvous").as_uri(), str(tmp_path / "checkpoints")),
        nprocs=2,
        join=True,
    )


@require_torch_multi_gpu
@require_deepspeed
def test_ap_ep_native_deepspeed_two_rank_shard_round_trip(tmp_path):
    mp.spawn(
        _native_legacy_ep_resume_worker,
        args=((tmp_path / "ap-rendezvous").as_uri(), str(tmp_path / "checkpoints"), False),
        nprocs=2,
        join=True,
    )


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
    flat_shards = [shard.detach().contiguous().view(torch.uint8).view(-1) for shard in shards]
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
    assert state["offset"] == shards[0].numel() * shards[0].element_size()
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
    assert nonmaster["offset"] == shards[0].numel() * shards[0].element_size()
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
    flat_shards = [shard.contiguous().view(torch.uint8).view(-1) for shard in shards]
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

    assert state["offset"] == shards[0].numel() * shards[0].element_size()
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


@pytest.mark.parametrize("rank,group_ranks", [(0, [0, 1]), (2, [2, 3])])
def test_hf_export_delegates_only_master_replica_to_cpu_gather(monkeypatch, rank, group_ranks):
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
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda ep_group: group_ranks)
    calls = []

    def gather(tensor, ep_group, *, is_master, max_gather_bytes=64 * 1024 * 1024):
        calls.append((tensor, ep_group, is_master, max_gather_bytes))
        return tensor.detach().cpu().contiguous()

    monkeypatch.setattr(common, "_gather_ep_weight_to_cpu", gather)
    model = ExportableModel()

    exported = dict(common.build_iter_full_hf_weights(model)())

    if rank != 0:
        assert exported == {}
        assert converted == {}
        assert calls == []
        return

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
