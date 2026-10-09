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
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from arctic_platform.model import PEFT_ADAPTER_DIRNAME
from arctic_platform.model import WeightExportContract
from arctic_platform.model import gather_peft_adapter_state_dict
from arctic_platform.model import hf_export_parameter_name
from arctic_platform.model import iter_lora_weights
from arctic_platform.model import iter_model_weights
from arctic_platform.model import save_hf_pretrained
from arctic_platform.model import save_peft_adapters
from arctic_platform.model import supports_weight_format
from arctic_platform.model.weight_export import register_weight_export
from arctic_platform.model.weight_export import transfer_weight_export


class _Model(nn.Module):
    def __init__(self, parameters=(), peft_config=None):
        super().__init__()
        self._parameters_for_test = list(parameters)
        self.peft_config = peft_config

    def named_parameters(self, *args, **kwargs):
        del args, kwargs
        return iter(self._parameters_for_test)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ),
        (
            "model.layers.0.self_attn.q_proj._checkpoint_wrapped_module.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ),
        (
            "model.layers.0.mlp.experts.base_layer.base_layer.w1.weight",
            "model.layers.0.mlp.experts.w1",
        ),
        (
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
            None,
        ),
    ],
)
def test_hf_export_parameter_name(name, expected):
    assert hf_export_parameter_name(name) == expected


def test_registered_weight_export_survives_model_replacement():
    source = nn.Linear(2, 2)
    target = nn.Sequential(source)

    def hf_builder(model):
        return lambda: iter((("registered.weight", next(model.parameters())),))

    register_weight_export(source, WeightExportContract(hf=hf_builder))
    transfer_weight_export(source, target)

    assert supports_weight_format(target, "hf")
    assert not supports_weight_format(target, "vllm")
    names = [
        name
        for name, _ in iter_model_weights(
            target,
            "hf",
            is_master=True,
            is_zero3=False,
        )
    ]
    assert names == ["registered.weight"]
    assert not hasattr(target, "_iter_full_hf_weights")


def test_iter_lora_weights_normalizes_dense_adapter_names():
    tensor = torch.ones(2, 2)
    model = _Model(
        [
            (
                "base_model.model.q_proj.lora_A.default.weight",
                SimpleNamespace(requires_grad=True, data=tensor),
            )
        ],
        peft_config={"default": SimpleNamespace(r=8)},
    )

    assert list(iter_lora_weights(model, is_master=True, is_zero3=False)) == [
        ("base_model.model.q_proj.lora_A.weight", tensor)
    ]


def test_iter_lora_weights_reshapes_expert_adapters(monkeypatch):
    num_experts, rank, hidden, intermediate = 4, 8, 16, 32
    lora_a = torch.arange(num_experts * rank * hidden, dtype=torch.float32).reshape(num_experts * rank, hidden)
    lora_b = torch.arange(intermediate * num_experts * rank, dtype=torch.float32).reshape(
        intermediate, num_experts * rank
    )

    def fake_iter(_model):
        yield (
            "model.layers.0.mlp.experts",
            "w1",
            "A",
            SimpleNamespace(
                data=lora_a,
                group_name=None,
                allreduce=True,
            ),
        )
        yield (
            "model.layers.0.mlp.experts",
            "w1",
            "B",
            SimpleNamespace(
                data=lora_b,
                group_name=None,
                allreduce=True,
            ),
        )

    monkeypatch.setattr(
        "arctic_platform.model.weight_export._iter_param_wrapper_expert_loras",
        fake_iter,
    )
    model = _Model(peft_config={"default": SimpleNamespace(r=rank)})
    exported = [
        (name, tuple(tensor.shape))
        for name, tensor in iter_lora_weights(
            model,
            is_master=True,
            is_zero3=False,
        )
    ]
    assert exported == [
        ("model.layers.0.mlp.experts.w1.lora_A.weight", (num_experts, rank, hidden)),
        (
            "model.layers.0.mlp.experts.w1.lora_B.weight",
            (num_experts, intermediate, rank),
        ),
    ]


def test_gather_peft_adapter_state_dict_uses_expert_layout(monkeypatch):
    a_shards = [torch.ones(2, 3), torch.full((2, 3), 2)]
    b_shards = [torch.ones(4, 2), torch.full((4, 2), 2)]
    model = _Model(
        [
            (
                "model.layers.0.mlp.experts.lora_A.default.weight",
                SimpleNamespace(data=a_shards[0], group_name="ep_size_2", allreduce=False),
            ),
            (
                "model.layers.0.mlp.experts.lora_B.default.weight",
                SimpleNamespace(data=b_shards[0], group_name="ep_size_2", allreduce=False),
            ),
        ]
    )

    group = object()
    groups_module = ModuleType("deepspeed.utils.groups")
    groups_module._get_expert_parallel_group = lambda _name: group
    utils_module = ModuleType("deepspeed.utils")
    utils_module.__path__ = []
    utils_module.groups = groups_module
    deepspeed_module = ModuleType("deepspeed")
    deepspeed_module.__path__ = []
    deepspeed_module.utils = utils_module
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed_module)
    monkeypatch.setitem(sys.modules, "deepspeed.utils", utils_module)
    monkeypatch.setitem(sys.modules, "deepspeed.utils.groups", groups_module)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    shard_sets = iter((a_shards, b_shards))

    def all_gather(outputs, local, group=None):
        expected = next(shard_sets)
        for output, shard in zip(outputs, expected, strict=True):
            output.copy_(shard)

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    state = gather_peft_adapter_state_dict(model, rank=0, is_zero3=False)

    assert state["model.layers.0.mlp.experts.lora_A.default.weight"].shape == (4, 3)
    assert state["model.layers.0.mlp.experts.lora_B.default.weight"].shape == (4, 4)


class _Base:
    def __init__(self):
        self.saved_state = None

    def save_pretrained(self, path, state_dict=None, **_kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.saved_state = state_dict
        (Path(path) / "config.json").write_text("{}")


class _PeftModel(_Model):
    def __init__(self, base):
        super().__init__(
            [
                ("base_model.model.proj.base_layer.weight", torch.ones(2, 2)),
                ("base_model.model.proj.lora_A.default.weight", torch.ones(1, 2)),
            ],
            peft_config={"default": object()},
        )
        self.base = base

    def get_base_model(self):
        return self.base

    def save_pretrained(self, path, state_dict=None, **_kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_config.json").write_text("{}")
        if state_dict:
            (Path(path) / "adapter_model.safetensors").write_bytes(b"adapter")


def test_hf_and_adapter_checkpoint_contract(tmp_path):
    base = _Base()
    model = _PeftModel(base)

    save_hf_pretrained(model, str(tmp_path))
    save_peft_adapters(model, str(tmp_path))

    assert set(base.saved_state) == {"proj.weight"}
    adapter = tmp_path / PEFT_ADAPTER_DIRNAME
    assert (adapter / "adapter_config.json").is_file()
    assert (adapter / "adapter_model.safetensors").is_file()


def test_iter_lora_weights_rejects_non_lora_trainables():
    model = _Model(
        [
            (
                "base_model.model.q_proj.lora_A.default.weight",
                SimpleNamespace(requires_grad=True, data=torch.ones(1)),
            ),
            (
                "base_model.model.lm_head.modules_to_save.default.weight",
                SimpleNamespace(requires_grad=True, data=torch.ones(1)),
            ),
        ],
        peft_config={"default": SimpleNamespace(r=8)},
    )

    with pytest.raises(NotImplementedError, match="Unsupported trainable"):
        list(iter_lora_weights(model, is_master=True, is_zero3=False))


def _export_names(model, weight_format, monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    return [
        name
        for name, _ in iter_model_weights(
            model,
            weight_format,
            is_master=True,
            is_zero3=False,
        )
    ]


def test_peft_wrapped_qwen3_moe_keeps_hf_family_export(monkeypatch):
    from peft import LoraConfig
    from peft import get_peft_model

    from arctic_platform.model.implementations.moe.deepspeed_integration import build_iter_full_hf_weights
    from arctic_platform.model.implementations.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from arctic_platform.model.implementations.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    family = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            num_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=16,
            use_grouped_mm=False,
            attn_implementation="sdpa",
        )
    )
    register_weight_export(family, WeightExportContract(hf=build_iter_full_hf_weights))
    wrapped = get_peft_model(family, LoraConfig(r=4, lora_alpha=8, target_modules=["lm_head"]))
    transfer_weight_export(family, wrapped)

    assert wrapped.get_base_model() is family
    assert type(wrapped).__name__.startswith("Peft")
    with pytest.raises(RuntimeError, match="convert_layer_to_hf"):
        build_iter_full_hf_weights(wrapped)

    names = _export_names(wrapped, "hf", monkeypatch)
    assert "model.layers.0.mlp.gate.weight" in names
    assert any(name.startswith("model.layers.0.mlp.experts.") for name in names)
    assert not any("lora_" in name for name in names)


def test_peft_wrapped_qwen3_5_moe_keeps_vllm_family_export(monkeypatch):
    from peft import LoraConfig
    from peft import get_peft_model

    from arctic_platform.model.implementations.qwen35.deepspeed_integration import _build_iter_full_vllm_weights
    from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeConfig,
    )
    from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForCausalLM,
    )

    family = Qwen3_5MoeForCausalLM(
        Qwen3_5MoeConfig(
            vocab_size=32,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
            num_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            layer_types=["full_attention"],
            use_grouped_mm=False,
            attn_implementation="sdpa",
        )
    )
    register_weight_export(family, WeightExportContract(vllm=_build_iter_full_vllm_weights))
    wrapped = get_peft_model(family, LoraConfig(r=4, lora_alpha=8, target_modules=["lm_head"]))
    transfer_weight_export(family, wrapped)

    assert wrapped.get_base_model() is family
    with pytest.raises(AssertionError, match="PeftModel"):
        _build_iter_full_vllm_weights(wrapped)

    names = _export_names(wrapped, "vllm", monkeypatch)
    assert "model.layers.0.mlp.gate.weight" in names
    assert "model.layers.0.mlp.experts.w13_weight" in names
