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
"""The sampler's adapter pack carries a family's routed-expert LoRA one expert at a time when the family declares
``vllm_lora_expert_projections`` (weight_export.save_peft_adapters), here Qwen3-MoE's."""

import copy
import json

import pytest
import torch

pytest.importorskip("peft")

_LAYERS = 2
_EXPERTS = 4
_RANK = 2
_ALPHA = 4
# The catalog's Qwen3-30B-A3B standard LoRA targets.
_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_TARGET_PARAMETERS = ["experts.w1", "experts.w2", "experts.w3"]
_PROJECTIONS = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def _tiny_peft_qwen3_moe():
    from peft import LoraConfig
    from peft import get_peft_model
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    from arctic_platform.model.implementations.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from arctic_platform.model.implementations.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=_LAYERS,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
            num_experts=_EXPERTS,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            use_grouped_mm=False,
            attn_implementation="sdpa",
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():  # the expert stacks start uninitialized (torch.empty)
            parameter.normal_(std=0.02)
    lora = LoraConfig(
        r=_RANK,
        lora_alpha=_ALPHA,
        target_modules=_TARGET_MODULES,
        target_parameters=_TARGET_PARAMETERS,
        init_lora_weights=False,
    )
    model = get_peft_model(model, lora)
    layers = model.base_model.model.model.layers
    for index, layer in enumerate(layers):
        layers[index] = checkpoint_wrapper(layer)
    return model


def _served_lora_modules() -> set[str]:
    """The module names vLLM's Qwen3MoeForCausalLM reads a LoRA under."""
    served: set[str] = set()
    for index in range(_LAYERS):
        prefix = f"model.layers.{index}"
        served |= {f"{prefix}.self_attn.{name}" for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
        served |= {f"{prefix}.mlp.experts.{e}.{p}" for e in range(_EXPERTS) for p in _PROJECTIONS.values()}
    return served


def _module_of(name: str) -> str:
    assert name.startswith("base_model.model."), name
    module, _, leaf = name.removeprefix("base_model.model.").rpartition(".lora_")
    assert leaf in ("A.weight", "B.weight"), name
    return module


def test_every_saved_qwen3_moe_lora_tensor_names_a_served_vllm_module(tmp_path):
    from safetensors.torch import load_file

    from arctic_platform.model import save_peft_adapters
    from arctic_platform.model.weight_export import PEFT_ADAPTER_DIRNAME

    model = _tiny_peft_qwen3_moe()
    save_peft_adapters(model, str(tmp_path))

    adapter_dir = tmp_path / PEFT_ADAPTER_DIRNAME
    saved = load_file(str(adapter_dir / "adapter_model.safetensors"))
    assert {_module_of(name) for name in saved} == _served_lora_modules()
    with open(adapter_dir / "adapter_config.json") as f:
        config = json.load(f)
    # Per-expert modules now: the sampler would read target_parameters as PEFT's 3-D expert layout.
    assert "target_parameters" not in config
    assert config["target_modules"] == sorted({"q_proj", "k_proj", "v_proj", "o_proj", *_PROJECTIONS.values()})
