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
"""DeepSeek-V4.1 LoRA tensors carry vLLM's module names, in the sampler's adapter pack and in LoRA weight sync.

The sampler (inference/vllm/deepseek_v41_lora_model.py over vLLM's DeepSeek-V4.1) serves the MoE under ``ffn``: the
routed experts take LoRA one expert at a time (``layers.N.ffn.experts.{e}.w1|w2|w3``), and the shared expert's w1 / w2
/ w3 are ``layers.N.ffn.shared_experts.w1|w2|w3`` (vLLM's DeepSeek-V4 weights mapper packs w1 / w3 into gate_up_proj
and names w2 down_proj); the attention projections keep their published names. Prime-rl's ``mlp.experts`` /
``mlp.shared_expert`` bound no module, and LoRA weight sync dropped them silently."""

import copy
import json

import pytest
import torch

pytest.importorskip("peft")

_EXPERTS = 4
_LAYERS = 4
_RANK = 2
_ALPHA = 4
_PROJECTIONS = ("w1", "w2", "w3")
# The catalog's DeepSeek-V4.1-Flash long LoRA targets without the ones vLLM cannot serve: the compressor's wkv /
# wgate (vLLM multiplies fused_wkv_wgate.weight directly) and o_a_proj.weight (vLLM's FP8 o_proj reads wo_a.weight).
_PEFT_CONFIG = {
    "peft_type": "Lora",
    "r": _RANK,
    "lora_alpha": _ALPHA,
    "lora_dropout": 0.0,
    "bias": "none",
    "target_modules": ["q_a_proj", "q_b_proj", "kv_proj", "o_b_proj"],
    "target_parameters": [*(f"experts.{p}" for p in _PROJECTIONS), *(f"shared_expert.{p}" for p in _PROJECTIONS)],
    "init_lora_weights": False,
}


def _tiny_deepseek_v41(mxfp4_experts: bool):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41ForCausalLM
    from arctic_platform.model.patches.peft import apply_peft
    from tests.model.test_deepseek_v41_streaming import _config

    config = _config()
    config.hidden_size, config.moe_intermediate_size = 64, 32
    config.engram_layer_ids, config.engram_num_embeddings, config.vision_n_layers = [], [], 0
    if mxfp4_experts:
        config.quantization_config = {
            "quant_method": "fp8",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        }
    torch.manual_seed(0)
    model = DeepseekV41ForCausalLM(config)
    assert config.n_routed_experts == _EXPERTS and config.num_hidden_layers == _LAYERS
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dtype in (torch.float32, torch.bfloat16):
                parameter.normal_(std=0.02)
    model = apply_peft(model, dict(_PEFT_CONFIG))
    layers = model.base_model.model.model.layers
    for index, layer in enumerate(layers):
        layers[index] = checkpoint_wrapper(layer)
    return model


def _served_lora_modules() -> set[str]:
    """The module names DeepSeek-V4.1's sampler reads a LoRA under, before vLLM's weights mapper puts them under
    ``language_model.model`` and packs wq_a / wkv, the per-expert projections and the shared expert's w1 / w3."""
    served: set[str] = set()
    for index in range(_LAYERS):
        prefix = f"layers.{index}"
        served |= {f"{prefix}.attn.{name}" for name in ("wq_a", "wkv", "wq_b", "wo_b")}
        served |= {f"{prefix}.ffn.shared_experts.{p}" for p in _PROJECTIONS}
        served |= {f"{prefix}.ffn.experts.{e}.{p}" for e in range(_EXPERTS) for p in _PROJECTIONS}
    return served


def _module_of(name: str) -> str:
    assert name.startswith("base_model.model."), name
    module, _, leaf = name.removeprefix("base_model.model.").rpartition(".lora_")
    assert leaf in ("A.weight", "B.weight"), name
    return module


@pytest.mark.parametrize("mxfp4_experts", [True, False])
def test_every_saved_and_synced_deepseek_v41_lora_tensor_names_a_served_module(tmp_path, mxfp4_experts):
    from safetensors.torch import load_file

    from arctic_platform.model import save_peft_adapters
    from arctic_platform.model.weight_export import PEFT_ADAPTER_DIRNAME
    from arctic_platform.model.weight_export import iter_lora_weights

    model = _tiny_deepseek_v41(mxfp4_experts)
    save_peft_adapters(model, str(tmp_path))
    adapter_dir = tmp_path / PEFT_ADAPTER_DIRNAME
    saved = load_file(str(adapter_dir / "adapter_model.safetensors"))
    assert {_module_of(name) for name in saved} == _served_lora_modules()
    with open(adapter_dir / "adapter_config.json") as f:
        assert "target_parameters" not in json.load(f)

    # Weight sync: the stacked experts the receiver packs into ffn.experts, everything else as in the pack.
    stacked = {f"layers.{index}.ffn.experts.{p}" for index in range(_LAYERS) for p in _PROJECTIONS}
    sent = {_module_of(name) for name, _ in iter_lora_weights(model, is_master=True, is_zero3=False)}
    assert stacked <= sent
    assert sent - stacked == {name for name in _served_lora_modules() if ".ffn.experts." not in name}


def test_deepseek_v41_expert_lora_carries_each_experts_trained_delta(tmp_path):
    """``B @ A * alpha / r`` of every saved routed-expert and shared-expert module is that weight's delta in PEFT's
    merged model (BF16 experts: PEFT does not merge into the MXFP4 codes)."""
    from safetensors.torch import load_file

    from arctic_platform.model import save_peft_adapters
    from arctic_platform.model.weight_export import PEFT_ADAPTER_DIRNAME

    model = _tiny_deepseek_v41(mxfp4_experts=False)
    unwrapped = {
        name.replace("._checkpoint_wrapped_module", "").replace(".base_layer", ""): parameter.detach().clone()
        for name, parameter in model.base_model.model.named_parameters()
        if ".experts." in name or ".shared_expert." in name
    }
    save_peft_adapters(model, str(tmp_path))
    saved = load_file(str(tmp_path / PEFT_ADAPTER_DIRNAME / "adapter_model.safetensors"))
    merged = {
        name.replace("._checkpoint_wrapped_module", ""): parameter.detach()
        for name, parameter in copy.deepcopy(model).merge_and_unload().named_parameters()
    }

    def check(module: str, delta: torch.Tensor) -> None:
        lora_a = saved[f"base_model.model.{module}.lora_A.weight"]
        lora_b = saved[f"base_model.model.{module}.lora_B.weight"]
        torch.testing.assert_close(lora_b @ lora_a * (_ALPHA / _RANK), delta, rtol=1e-5, atol=1e-6)

    checked = 0
    for index in range(_LAYERS):
        for parameter in _PROJECTIONS:
            routed = f"model.layers.{index}.mlp.experts.{parameter}"
            delta = merged[routed] - unwrapped[routed]
            for expert in range(_EXPERTS):
                check(f"layers.{index}.ffn.experts.{expert}.{parameter}", delta[expert])
                checked += 1
            shared = f"model.layers.{index}.mlp.shared_expert.{parameter}"
            delta = merged[shared] - unwrapped[shared]
            check(f"layers.{index}.ffn.shared_experts.{parameter}", delta.reshape(delta.shape[-2:]))
            checked += 1
    assert checked == _LAYERS * len(_PROJECTIONS) * (_EXPERTS + 1)


def test_vllm_lora_name_maps_the_moe_under_ffn_and_keeps_the_attention_names():
    from arctic_platform.model.implementations.deepseek_v41.modeling_deepseek_v41 import DeepseekV41ForCausalLM
    from arctic_platform.model.implementations.deepseek_v41.modeling_deepseek_v41 import vllm_lora_name

    assert DeepseekV41ForCausalLM.vllm_lora_name is vllm_lora_name
    cases = {
        "base_model.model.model.layers.2.mlp.experts.w1.lora_A.weight": (
            "base_model.model.layers.2.ffn.experts.w1.lora_A.weight"
        ),
        "base_model.model.model.layers.2.mlp.experts.7.w2.lora_B.weight": (
            "base_model.model.layers.2.ffn.experts.7.w2.lora_B.weight"
        ),
        "model.layers.0.mlp.shared_expert.w3.lora_A.weight": "layers.0.ffn.shared_experts.w3.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.kv_proj.lora_B.weight": (
            "base_model.model.layers.0.attn.wkv.lora_B.weight"
        ),
    }
    for name, served in cases.items():
        assert vllm_lora_name(name) == served
