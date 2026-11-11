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
"""Nemotron-H LoRA tensors carry vLLM's module names, in the checkpoint's adapter pack and in LoRA weight sync.

vLLM's Nemotron-H names every block's mixer ``mixer`` (attention, Mamba-2, MoE) and the MoE's shared expert
``mixer.shared_experts``, and takes the non-gated experts' LoRA one expert at a time (``experts.{e}.up_proj`` /
``down_proj``). An adapter under prime-rl's names (``self_attn``, ``mlp``, ``mlp.shared_expert``) binds no module:
LoRA weight sync then leaves the sampler on the base model, and a restored sampler refuses the pack."""

import copy
import json

import pytest
import torch

pytest.importorskip("peft")

_LAYERS = ["mamba", "moe", "attention", "moe"]
_EXPERTS = 4
_RANK = 2
_ALPHA = 4
# The catalog's Nemotron-3-Nano LoRA targets (gate_proj matches nothing: the experts and shared expert are non-gated).
_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_TARGET_PARAMETERS = ["experts.w1", "experts.w2"]


def _tiny_peft_model(checkpoint_wrapped: bool):
    from peft import LoraConfig
    from peft import get_peft_model
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    from arctic_platform.model.implementations.nemotron_h.configuration_nemotron_h import NemotronHConfig
    from arctic_platform.model.implementations.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM

    torch.manual_seed(0)
    config = NemotronHConfig(
        vocab_size=64,
        hidden_size=64,
        layers_block_type=_LAYERS,
        ssm_state_size=16,
        mamba_num_heads=4,
        mamba_n_groups=2,
        mamba_head_dim=16,
        mamba_chunk_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=48,
        moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=40,
        n_routed_experts=_EXPERTS,
        num_experts_per_tok=2,
    )
    model = NemotronHForCausalLM(config)
    lora = LoraConfig(
        r=_RANK,
        lora_alpha=_ALPHA,
        target_modules=_TARGET_MODULES,
        target_parameters=_TARGET_PARAMETERS,
        init_lora_weights=False,
    )
    model = get_peft_model(model, lora)
    if checkpoint_wrapped:
        layers = model.base_model.model.model.layers
        for index, layer in enumerate(layers):
            layers[index] = checkpoint_wrapper(layer)
    return model


def _served_lora_modules() -> set[str]:
    """The module names vLLM's NemotronHForCausalLM reads a LoRA under (vllm/model_executor/models/nemotron_h.py:
    every block's ``mixer``; q/k/v packed into ``qkv_proj``; per-expert ``experts.{e}.up_proj|down_proj`` packed into
    ``experts`` through ``get_expert_mapping``)."""
    served: set[str] = set()
    for index, kind in enumerate(_LAYERS):
        mixer = f"model.layers.{index}.mixer"
        if kind == "mamba":
            served |= {f"{mixer}.in_proj", f"{mixer}.out_proj"}
        elif kind == "attention":
            served |= {f"{mixer}.{name}" for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
        else:
            served |= {f"{mixer}.shared_experts.up_proj", f"{mixer}.shared_experts.down_proj"}
            served |= {f"{mixer}.experts.{e}.{p}" for e in range(_EXPERTS) for p in ("up_proj", "down_proj")}
    return served


def _module_of(name: str) -> str:
    assert name.startswith("base_model.model."), name
    module, _, leaf = name.removeprefix("base_model.model.").rpartition(".lora_")
    assert leaf in ("A.weight", "B.weight"), name
    return module


@pytest.mark.parametrize("checkpoint_wrapped", [False, True])
def test_every_saved_lora_tensor_names_a_served_vllm_module(tmp_path, checkpoint_wrapped):
    from safetensors.torch import load_file

    from arctic_platform.model import save_peft_adapters
    from arctic_platform.model.weight_export import PEFT_ADAPTER_DIRNAME

    model = _tiny_peft_model(checkpoint_wrapped)
    save_peft_adapters(model, str(tmp_path))

    adapter_dir = tmp_path / PEFT_ADAPTER_DIRNAME
    saved = load_file(str(adapter_dir / "adapter_model.safetensors"))
    modules = {_module_of(name) for name in saved}
    # Every targeted module of every attention and MoE block, and nothing vLLM does not serve.
    assert modules == _served_lora_modules() - {
        f"model.layers.{index}.mixer.{p}"
        for index, kind in enumerate(_LAYERS)
        if kind == "mamba"
        for p in ("in_proj", "out_proj")
    }
    with open(adapter_dir / "adapter_config.json") as f:
        config = json.load(f)
    # Per-expert modules now: the sampler would read target_parameters as PEFT's 3-D expert layout.
    assert "target_parameters" not in config
    assert config["target_modules"] == sorted({"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"})


def test_per_expert_lora_carries_each_experts_trained_delta(tmp_path):
    """``B_e @ A_e * alpha / r`` of every saved expert module is that expert's delta in PEFT's merged weights."""
    from safetensors.torch import load_file

    from arctic_platform.model import save_peft_adapters
    from arctic_platform.model.weight_export import PEFT_ADAPTER_DIRNAME

    model = _tiny_peft_model(checkpoint_wrapped=False)
    base = {
        name: parameter.detach().clone()
        for name, parameter in model.base_model.model.named_parameters()
        if name.endswith(("experts.w1", "experts.w2")) or ".experts.base_layer" in name
    }
    save_peft_adapters(model, str(tmp_path))
    saved = load_file(str(tmp_path / PEFT_ADAPTER_DIRNAME / "adapter_model.safetensors"))
    merged = dict(copy.deepcopy(model).merge_and_unload().named_parameters())

    scaling = _ALPHA / _RANK
    checked = 0
    for index, kind in enumerate(_LAYERS):
        if kind != "moe":
            continue
        for parameter, projection in (("w1", "up_proj"), ("w2", "down_proj")):
            weight = f"model.layers.{index}.mlp.experts.{parameter}"
            (original,) = [tensor for name, tensor in base.items() if name.replace(".base_layer", "") == weight]
            delta = merged[weight].detach() - original
            for expert in range(_EXPERTS):
                prefix = f"base_model.model.model.layers.{index}.mixer.experts.{expert}.{projection}"
                lora_a, lora_b = saved[f"{prefix}.lora_A.weight"], saved[f"{prefix}.lora_B.weight"]
                assert lora_a.shape[0] == _RANK and lora_b.shape[1] == _RANK
                torch.testing.assert_close(lora_b @ lora_a * scaling, delta[expert], rtol=1e-5, atol=1e-6)
                checked += 1
    assert checked == 2 * 2 * _EXPERTS


def test_lora_weight_sync_sends_served_vllm_module_names():
    """LoRA weight sync's tensors (iter_lora_weights) under vLLM's names: the stacked experts.w1 / experts.w2 that the
    sampler packs into ``mixer.experts`` (weight_sync/receiver.py), and the served modules otherwise."""
    from arctic_platform.model.weight_export import iter_lora_weights

    model = _tiny_peft_model(checkpoint_wrapped=True)
    served = _served_lora_modules()
    stacked = {
        f"model.layers.{index}.mixer.experts.{p}"
        for index, kind in enumerate(_LAYERS)
        if kind == "moe"
        for p in ("w1", "w2")
    }
    sent = {_module_of(name) for name, _ in iter_lora_weights(model, is_master=True, is_zero3=False)}
    assert stacked <= sent
    assert sent - stacked <= served
    assert {name for name in sent - stacked if ".shared_experts." in name or ".mixer.q_proj" in name}


def test_vllm_lora_name_maps_each_prime_mixer():
    from arctic_platform.model.implementations.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM
    from arctic_platform.model.implementations.nemotron_h.modeling_nemotron_h import vllm_lora_name

    assert NemotronHForCausalLM.vllm_lora_name is vllm_lora_name
    cases = {
        "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight": (
            "base_model.model.model.layers.3.mixer.q_proj.lora_A.weight"
        ),
        "model.layers.12.mlp.shared_expert.down_proj.lora_B.weight": (
            "model.layers.12.mixer.shared_experts.down_proj.lora_B.weight"
        ),
        "model.layers.1.mlp.experts.w1.lora_A.weight": "model.layers.1.mixer.experts.w1.lora_A.weight",
        "model.layers.1.mlp.fc1_latent_proj.lora_A.weight": "model.layers.1.mixer.fc1_latent_proj.lora_A.weight",
        "model.layers.0.mamba.in_proj.lora_A.weight": "model.layers.0.mixer.in_proj.lora_A.weight",
        "lm_head.lora_A.weight": "lm_head.lora_A.weight",
    }
    for name, served in cases.items():
        assert vllm_lora_name(name) == served
        assert vllm_lora_name(served) == served
