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
``vllm_lora_expert_projections`` (weight_export.save_peft_adapters), here Qwen3-MoE's.

PEFT nests one ParamWrapper per targeted expert parameter (target_parameters experts.w1, w2, w3 saves
``experts.base_layer.base_layer`` / ``experts.base_layer`` / ``experts``), and vLLM's PEFT 3-D loader reads only
``experts.base_layer`` (as gate_up) and ``experts`` (as down): the w1 nest matched no served module, and the other
two would have been applied as the wrong projections. vLLM's Qwen3-MoE maps per-expert LoRA as
``experts.{e}.gate_proj|down_proj|up_proj`` (RoutedExperts.get_expert_mapping)."""

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


def _tiny_peft_qwen3_moe(checkpoint_wrapped: bool):
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
    if checkpoint_wrapped:
        layers = model.base_model.model.model.layers
        for index, layer in enumerate(layers):
            layers[index] = checkpoint_wrapper(layer)
    return model


def _served_lora_modules() -> set[str]:
    """The module names vLLM's Qwen3MoeForCausalLM reads a LoRA under (q/k/v packed into ``qkv_proj``, per-expert
    gate/down/up packed into ``experts``)."""
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


@pytest.mark.parametrize("checkpoint_wrapped", [False, True])
def test_every_saved_qwen3_moe_lora_tensor_names_a_served_vllm_module(tmp_path, checkpoint_wrapped):
    from safetensors.torch import load_file

    from arctic_platform.model import save_peft_adapters
    from arctic_platform.model.weight_export import PEFT_ADAPTER_DIRNAME

    model = _tiny_peft_qwen3_moe(checkpoint_wrapped)
    save_peft_adapters(model, str(tmp_path))

    adapter_dir = tmp_path / PEFT_ADAPTER_DIRNAME
    saved = load_file(str(adapter_dir / "adapter_model.safetensors"))
    assert {_module_of(name) for name in saved} == _served_lora_modules()
    with open(adapter_dir / "adapter_config.json") as f:
        config = json.load(f)
    # Per-expert modules now: the sampler would read target_parameters as PEFT's 3-D expert layout.
    assert "target_parameters" not in config
    assert config["target_modules"] == sorted({"q_proj", "k_proj", "v_proj", "o_proj", *_PROJECTIONS.values()})


def test_per_expert_lora_carries_each_experts_trained_delta(tmp_path):
    """``B_e @ A_e * alpha / r`` of every saved expert module is that expert's delta in PEFT's merged weights, in
    the [out, in] orientation of vLLM's per-expert projection (GroupedExperts applies ``x @ w[e].T``)."""
    from safetensors.torch import load_file

    from arctic_platform.model import save_peft_adapters
    from arctic_platform.model.weight_export import PEFT_ADAPTER_DIRNAME

    model = _tiny_peft_qwen3_moe(checkpoint_wrapped=False)
    base = {
        name.replace(".base_layer", ""): parameter.detach().clone()
        for name, parameter in model.base_model.model.named_parameters()
        if name.rsplit(".", 1)[-1] in _PROJECTIONS and ".experts." in name
    }
    save_peft_adapters(model, str(tmp_path))
    saved = load_file(str(tmp_path / PEFT_ADAPTER_DIRNAME / "adapter_model.safetensors"))
    merged = dict(copy.deepcopy(model).merge_and_unload().named_parameters())

    scaling = _ALPHA / _RANK
    checked = 0
    for index in range(_LAYERS):
        for parameter, projection in _PROJECTIONS.items():
            weight = f"model.layers.{index}.mlp.experts.{parameter}"
            delta = merged[weight].detach() - base[weight]
            assert delta.abs().amax() > 0
            for expert in range(_EXPERTS):
                prefix = f"base_model.model.model.layers.{index}.mlp.experts.{expert}.{projection}"
                lora_a, lora_b = saved[f"{prefix}.lora_A.weight"], saved[f"{prefix}.lora_B.weight"]
                assert lora_a.shape[0] == _RANK and lora_b.shape[1] == _RANK
                torch.testing.assert_close(lora_b @ lora_a * scaling, delta[expert], rtol=1e-5, atol=1e-6)
                checked += 1
    assert checked == _LAYERS * len(_PROJECTIONS) * _EXPERTS


def test_lora_weight_sync_keeps_the_stacked_qwen3_moe_experts():
    """LoRA weight sync (iter_lora_weights) still sends stacked experts.w1 / w2 / w3, which the sampler packs into
    ``mlp.experts`` (weight_sync/receiver.py); everything else under its served name."""
    from arctic_platform.model.weight_export import iter_lora_weights

    model = _tiny_peft_qwen3_moe(checkpoint_wrapped=True)
    stacked = {f"model.layers.{index}.mlp.experts.{p}" for index in range(_LAYERS) for p in _PROJECTIONS}
    sent = {_module_of(name) for name, _ in iter_lora_weights(model, is_master=True, is_zero3=False)}
    assert sent - stacked == {
        f"model.layers.{index}.self_attn.{name}"
        for index in range(_LAYERS)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    assert stacked <= sent


def test_families_without_the_opt_in_keep_peft_names(tmp_path):
    """A family that does not declare vllm_lora_expert_projections keeps PEFT's 3-D target_parameters pack."""
    from peft import LoraConfig
    from peft import get_peft_model
    from safetensors.torch import load_file

    from arctic_platform.model import save_peft_adapters
    from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts
    from arctic_platform.model.weight_export import PEFT_ADAPTER_DIRNAME

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = GroupedExperts(dim=8, hidden_dim=4, num_experts=2, use_grouped_mm=False)

    class _Family(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = _Block()

    model = get_peft_model(
        _Family(), LoraConfig(r=2, lora_alpha=2, target_parameters=["experts.w1"], init_lora_weights=False)
    )
    save_peft_adapters(model, str(tmp_path))
    adapter_dir = tmp_path / PEFT_ADAPTER_DIRNAME
    assert set(load_file(str(adapter_dir / "adapter_model.safetensors"))) == {
        f"base_model.model.block.experts.lora_{kind}.weight" for kind in "AB"
    }
    with open(adapter_dir / "adapter_config.json") as f:
        assert json.load(f)["target_parameters"] == ["experts.w1"]
