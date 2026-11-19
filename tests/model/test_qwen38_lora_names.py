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
"""Qwen3.8-Flash-Next (qwen4_exp) routed-expert LoRA tensors go into the sampler's adapter pack one expert at a time,
under vLLM's module names."""

import copy
import json

import pytest
import torch

from tests.model.test_qwen4_exp import _tiny_config

pytest.importorskip("peft")

_RANK = 2
_ALPHA = 4
_PROJECTIONS = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
# The catalog's Qwen3.8-Flash-Next long LoRA targets (models/qwen3.8-flash-next/long/h200-training-lora.json).
_PEFT_CONFIG = {
    "peft_type": "Lora",
    "r": _RANK,
    "lora_alpha": _ALPHA,
    "lora_dropout": 0.0,
    "bias": "none",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "target_parameters": ["experts.w1", "experts.w2", "experts.w3"],
    "init_lora_weights": False,
}


def _tiny_qwen38():
    from arctic_platform.model.implementations.qwen38.modeling_qwen4_exp import Qwen4ExpForConditionalGenerationPrimeRL
    from arctic_platform.model.patches.peft import apply_peft

    config = _tiny_config()
    torch.manual_seed(0)
    model = Qwen4ExpForConditionalGenerationPrimeRL(config)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point():
                parameter.normal_(std=0.02)
    return apply_peft(model, dict(_PEFT_CONFIG))


def test_qwen38_routed_expert_lora_is_saved_per_expert_under_vllm_names(tmp_path):
    """Each MoE layer's experts.w1 / w2 / w3 LoRA goes out as experts.{e}.gate_proj / down_proj / up_proj, whose
    ``B @ A * alpha / r`` is that expert's delta in PEFT's merge, in vLLM's [out, in] slots. vLLM drops the
    experts.base_layer.base_layer nest of PEFT's 3-D layout, one per MoE layer."""
    from safetensors.torch import load_file

    from arctic_platform.model import save_peft_adapters
    from arctic_platform.model.weight_export import PEFT_ADAPTER_DIRNAME

    model = _tiny_qwen38()
    text_config = model.base_model.model.config.text_config
    base = {
        name.replace(".base_layer", ""): parameter.detach().clone()
        for name, parameter in model.base_model.model.named_parameters()
        if ".mlp.experts." in name and ".lora_" not in name
    }
    save_peft_adapters(model, str(tmp_path))
    adapter_dir = tmp_path / PEFT_ADAPTER_DIRNAME
    saved = load_file(str(adapter_dir / "adapter_model.safetensors"))
    merged = dict(copy.deepcopy(model).merge_and_unload().named_parameters())

    assert [name for name in saved if ".base_layer." in name] == []
    layers = range(text_config.num_hidden_layers)
    experts = range(text_config.num_experts)
    assert {name.rpartition(".lora_")[0] for name in saved if ".mlp.experts." in name} == {
        f"base_model.model.model.language_model.layers.{index}.mlp.experts.{expert}.{projection}"
        for index in layers
        for expert in experts
        for projection in _PROJECTIONS.values()
    }
    for index in layers:
        for parameter, projection in _PROJECTIONS.items():
            routed = f"model.language_model.layers.{index}.mlp.experts.{parameter}"
            delta = merged[routed].detach() - base[routed]
            for expert in experts:
                module = f"base_model.model.model.language_model.layers.{index}.mlp.experts.{expert}.{projection}"
                lora_a = saved[f"{module}.lora_A.weight"]
                lora_b = saved[f"{module}.lora_B.weight"]
                torch.testing.assert_close(lora_b @ lora_a * (_ALPHA / _RANK), delta[expert], rtol=1e-5, atol=1e-6)
    with open(adapter_dir / "adapter_config.json") as f:
        config = json.load(f)
    # The sampler reads target_parameters as PEFT's 3-D expert layout (worker._load_checkpoint_lora_adapter).
    assert "target_parameters" not in config
    assert config["target_modules"] == sorted({*_PEFT_CONFIG["target_modules"], *_PROJECTIONS.values()})
