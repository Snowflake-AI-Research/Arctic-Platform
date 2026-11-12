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
"""GLM-5.2/5.3 (glm_moe_dsa) LoRA tensors carry vLLM's module names, in the sampler's adapter pack and in LoRA weight
sync.

vLLM serves GLM-5 as DeepseekV32ForCausalLM: the routed experts take LoRA one expert at a time
(``mlp.experts.{e}.gate_proj|down_proj|up_proj``), the shared expert is ``mlp.shared_experts.gate_proj|up_proj``
(packed into ``gate_up_proj``) and ``down_proj``, q_a / kv_a are packed into ``fused_qkv_a_proj``, and the first dense
layers' MLP is ``mlp.gate_proj|up_proj|down_proj``. Before, PEFT's 3-D pack of the routed experts bound w1 nowhere and
w2 / w3 into the wrong slots, and the shared expert's nested wrappers matched no module."""

import copy
import json

import pytest
import torch

pytest.importorskip("peft")

_EXPERTS = 4
_RANK = 2
_ALPHA = 4
_DENSE_LAYERS = 1
_LAYERS = 3
_PROJECTIONS = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
# The catalog's GLM-5.3 long LoRA targets, without kv_b_proj (vLLM folds it into W_UK / W_UV at load, so a LoRA on it
# binds and does nothing).
_PEFT_CONFIG = {
    "peft_type": "Lora",
    "r": _RANK,
    "lora_alpha": _ALPHA,
    "lora_dropout": 0.0,
    "bias": "none",
    "target_modules": ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "target_parameters": [
        "experts.w1",
        "experts.w2",
        "experts.w3",
        "shared_expert.w1",
        "shared_expert.w2",
        "shared_expert.w3",
    ],
    "init_lora_weights": False,
}


def _tiny_glm(checkpoint_wrapped: bool):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    from arctic_platform.model.implementations.glm52.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )
    from arctic_platform.model.implementations.glm52.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaForCausalLM,
    )
    from arctic_platform.model.patches.peft import apply_peft

    config = GlmMoeDsaConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=_LAYERS,
        first_k_dense_replace=_DENSE_LAYERS,
        mlp_layer_types=["dense", "sparse", "sparse"],
        indexer_types=["full", "full", "shared"],
        num_attention_heads=2,
        num_key_value_heads=2,
        kv_lora_rank=128,
        q_lora_rank=256,
        qk_rope_head_dim=64,
        qk_nope_head_dim=64,
        v_head_dim=64,
        index_n_heads=2,
        index_head_dim=128,
        index_topk=16,
        n_routed_experts=_EXPERTS,
        num_experts_per_tok=2,
        n_shared_experts=1,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        max_position_embeddings=256,
        rope_parameters={"rope_theta": 10000, "rope_type": "default"},
        rms_norm_eps=1e-5,
        pad_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        num_nextn_predict_layers=0,
    )
    torch.manual_seed(0)
    model = GlmMoeDsaForCausalLM(config)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point():
                parameter.normal_(std=0.02)
    model = apply_peft(model, dict(_PEFT_CONFIG))
    if checkpoint_wrapped:
        layers = model.base_model.model.model.layers
        for index, layer in enumerate(layers):
            layers[index] = checkpoint_wrapper(layer)
    return model


def _served_lora_modules() -> set[str]:
    """The module names vLLM's GLM-5 (vllm/models/deepseek_v32, DeepseekV2 MoE) reads a LoRA under, before packing."""
    served: set[str] = set()
    for index in range(_LAYERS):
        prefix = f"model.layers.{index}"
        served |= {f"{prefix}.self_attn.{name}" for name in ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj")}
        if index < _DENSE_LAYERS:
            served |= {f"{prefix}.mlp.{p}" for p in _PROJECTIONS.values()}
        else:
            served |= {f"{prefix}.mlp.shared_experts.{p}" for p in _PROJECTIONS.values()}
            served |= {f"{prefix}.mlp.experts.{e}.{p}" for e in range(_EXPERTS) for p in _PROJECTIONS.values()}
    return served


def _module_of(name: str) -> str:
    assert name.startswith("base_model.model."), name
    module, _, leaf = name.removeprefix("base_model.model.").rpartition(".lora_")
    assert leaf in ("A.weight", "B.weight"), name
    return module


@pytest.mark.parametrize("checkpoint_wrapped", [False, True])
def test_every_saved_glm_lora_tensor_names_a_served_vllm_module(tmp_path, checkpoint_wrapped):
    from safetensors.torch import load_file

    from arctic_platform.model import save_peft_adapters
    from arctic_platform.model.weight_export import PEFT_ADAPTER_DIRNAME

    model = _tiny_glm(checkpoint_wrapped)
    save_peft_adapters(model, str(tmp_path))
    adapter_dir = tmp_path / PEFT_ADAPTER_DIRNAME
    saved = load_file(str(adapter_dir / "adapter_model.safetensors"))
    # Every targeted module, the dense layers' MLP included, and nothing vLLM does not serve.
    assert {_module_of(name) for name in saved} == _served_lora_modules()
    with open(adapter_dir / "adapter_config.json") as f:
        config = json.load(f)
    assert "target_parameters" not in config
    assert not config.get("rank_pattern")


def test_glm_per_expert_and_shared_expert_lora_carry_their_trained_deltas(tmp_path):
    """``B @ A * alpha / r`` of every saved routed-expert and shared-expert module equals that weight's delta in PEFT's
    merged model, in the [out, in] orientation of vLLM's projection."""
    from safetensors.torch import load_file

    from arctic_platform.model import save_peft_adapters
    from arctic_platform.model.weight_export import PEFT_ADAPTER_DIRNAME

    model = _tiny_glm(checkpoint_wrapped=False)
    base = {
        name.replace(".base_layer", ""): parameter.detach().clone()
        for name, parameter in model.base_model.model.named_parameters()
        if ".experts." in name or ".shared_expert." in name
    }
    save_peft_adapters(model, str(tmp_path))
    saved = load_file(str(tmp_path / PEFT_ADAPTER_DIRNAME / "adapter_model.safetensors"))
    merged = dict(copy.deepcopy(model).merge_and_unload().named_parameters())

    def check(module: str, delta: torch.Tensor) -> None:
        lora_a = saved[f"base_model.model.{module}.lora_A.weight"]
        lora_b = saved[f"base_model.model.{module}.lora_B.weight"]
        torch.testing.assert_close(lora_b @ lora_a * (_ALPHA / _RANK), delta, rtol=1e-5, atol=1e-6)

    checked = 0
    for index in range(_DENSE_LAYERS, _LAYERS):
        for parameter, projection in _PROJECTIONS.items():
            routed = f"model.layers.{index}.mlp.experts.{parameter}"
            delta = merged[routed].detach() - base[routed]
            for expert in range(_EXPERTS):
                check(f"model.layers.{index}.mlp.experts.{expert}.{projection}", delta[expert])
                checked += 1
            shared = f"model.layers.{index}.mlp.shared_expert.{parameter}"
            delta = (merged[shared] - base[shared]).detach()
            check(f"model.layers.{index}.mlp.shared_experts.{projection}", delta.reshape(delta.shape[-2:]))
            checked += 1
    assert checked == (_LAYERS - _DENSE_LAYERS) * len(_PROJECTIONS) * (_EXPERTS + 1)


def test_glm_lora_weight_sync_sends_served_vllm_module_names():
    """LoRA weight sync (iter_lora_weights): the stacked experts.w1 / w2 / w3 the sampler packs into ``mlp.experts``
    (weight_sync/receiver.py), and every other LoRA under its served name."""
    from arctic_platform.model.weight_export import iter_lora_weights

    model = _tiny_glm(checkpoint_wrapped=True)
    stacked = {
        f"model.layers.{index}.mlp.experts.{p}" for index in range(_DENSE_LAYERS, _LAYERS) for p in _PROJECTIONS
    }
    sent = {_module_of(name) for name, _ in iter_lora_weights(model, is_master=True, is_zero3=False)}
    assert stacked <= sent
    assert sent - stacked == {name for name in _served_lora_modules() if ".mlp.experts." not in name}


def test_vllm_lora_name_maps_the_shared_expert_only():
    from arctic_platform.model.implementations.glm52.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaForCausalLM,
    )
    from arctic_platform.model.implementations.glm52.models.glm_moe_dsa.modeling_glm_moe_dsa import vllm_lora_name

    assert GlmMoeDsaForCausalLM.vllm_lora_name is vllm_lora_name
    cases = {
        "base_model.model.model.layers.3.mlp.shared_expert.w1.lora_A.weight": (
            "base_model.model.model.layers.3.mlp.shared_experts.gate_proj.lora_A.weight"
        ),
        "model.layers.3.mlp.shared_expert.w2.lora_B.weight": (
            "model.layers.3.mlp.shared_experts.down_proj.lora_B.weight"
        ),
        "model.layers.3.mlp.shared_expert.w3.lora_A.weight": "model.layers.3.mlp.shared_experts.up_proj.lora_A.weight",
        "model.layers.3.mlp.experts.w1.lora_A.weight": "model.layers.3.mlp.experts.w1.lora_A.weight",
        "model.layers.0.mlp.gate_proj.lora_A.weight": "model.layers.0.mlp.gate_proj.lora_A.weight",
        "model.layers.0.self_attn.q_a_proj.lora_A.weight": "model.layers.0.self_attn.q_a_proj.lora_A.weight",
    }
    for name, served in cases.items():
        assert vllm_lora_name(name) == served
