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

"""CPU tests for loading a checkpoint's PEFT adapter pack into the sampler."""

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from arctic_platform.inference.server import worker as worker_module
from arctic_platform.inference.server.weight_sync.receiver import WeightSyncExtension

TEXT_ONLY_NAME = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
NESTED_NAME = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
SERVED_NAME = "base_model.model.language_model.model.layers.0.self_attn.q_proj.lora_A.weight"


def _decoder() -> nn.Module:
    root = nn.Module()
    root.layers = nn.ModuleList([nn.Linear(2, 2)])
    return root


def _extension(model: nn.Module, adapter_manager=None) -> WeightSyncExtension:
    ext = object.__new__(WeightSyncExtension)
    ext.model_runner = SimpleNamespace(model=model, lora_manager=SimpleNamespace(_adapter_manager=adapter_manager))
    return ext


def test_served_lora_names_moves_trainer_roots_under_a_multimodal_wrapper():
    served = nn.Module()
    served.language_model = nn.Module()
    served.language_model.model = _decoder()

    names = _extension(served).served_lora_names([TEXT_ONLY_NAME, NESTED_NAME])

    assert names == [SERVED_NAME, SERVED_NAME]


def test_served_lora_names_keeps_a_dense_models_names():
    served = nn.Module()
    served.model = _decoder()

    assert _extension(served).served_lora_names([TEXT_ONLY_NAME]) == [TEXT_ONLY_NAME]


def test_unbound_lora_modules_lists_adapter_modules_with_no_served_layer():
    loras = {"model.layers.0.self_attn.qkv_proj": object(), "model.layers.0.self_attn.q_proj": object()}
    manager = SimpleNamespace(
        get_adapter=lambda lora_int_id: SimpleNamespace(loras=loras) if lora_int_id == 1 else None,
        modules={"model.layers.0.self_attn.qkv_proj": object()},
    )
    ext = _extension(nn.Module(), manager)

    assert ext.unbound_lora_modules(1) == ["model.layers.0.self_attn.q_proj"]
    with pytest.raises(RuntimeError, match="not registered"):
        ext.unbound_lora_modules(2)


class _Engine:
    def __init__(self, unbound_per_rank):
        self.unbound_per_rank = unbound_per_rank
        self.requests = []

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        if method == "served_lora_names":
            (names,) = args
            return [[name.replace("base_model.model.model.", "base_model.model.language_model.model.") for name in names]]
        assert method == "unbound_lora_modules" and args == (1,)
        return self.unbound_per_rank

    async def add_lora(self, request):
        self.requests.append(request)
        return True


def _pack(tmp_path):
    adapter_dir = tmp_path / "default"
    adapter_dir.mkdir()
    save_file({TEXT_ONLY_NAME: torch.ones(2, 2)}, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 2, "target_modules": ["q_proj"]}))
    return adapter_dir


def _worker(engine):
    cls = worker_module.InferenceWorker.__ray_metadata__.modified_class
    worker = object.__new__(cls)
    worker.llm = engine
    worker._served_checkpoint_lora_dir = None
    return worker


def test_checkpoint_lora_adapter_loads_under_served_names(tmp_path):
    adapter_dir = _pack(tmp_path)
    engine = _Engine(unbound_per_rank=[[], []])
    worker = _worker(engine)

    asyncio.run(worker._load_checkpoint_lora_adapter(str(adapter_dir)))

    [request] = engine.requests
    assert request.lora_path != str(adapter_dir)
    loaded = load_file(f"{request.lora_path}/adapter_model.safetensors")
    assert list(loaded) == [SERVED_NAME]
    assert torch.equal(loaded[SERVED_NAME], torch.ones(2, 2))
    with open(f"{request.lora_path}/adapter_config.json") as f:
        assert json.load(f)["target_modules"] == ["q_proj"]
    assert worker._active_lora_int_id == 1


def test_checkpoint_lora_adapter_tolerates_modules_another_rank_serves(tmp_path):
    engine = _Engine(unbound_per_rank=[["layers.3.self_attn.qkv_proj"], []])

    asyncio.run(_worker(engine)._load_checkpoint_lora_adapter(str(_pack(tmp_path))))

    assert len(engine.requests) == 1


def test_checkpoint_lora_adapter_fails_when_a_module_binds_nowhere(tmp_path):
    engine = _Engine(unbound_per_rank=[["model.layers.0.self_attn.q_proj"], ["model.layers.0.self_attn.q_proj"]])

    with pytest.raises(RuntimeError, match="1 module"):
        asyncio.run(_worker(engine)._load_checkpoint_lora_adapter(str(_pack(tmp_path))))
