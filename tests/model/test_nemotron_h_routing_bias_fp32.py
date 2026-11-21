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
"""CPU: Nemotron-H's routing bias (e_score_correction_bias) stays the checkpoint's FP32 values, as vLLM
(models/nemotron_h.py:167-168) and Transformers (_keep_in_fp32_modules_strict) keep it: built in FP32 in a BF16 model,
loaded unshifted, kept through DeepSpeed's mixed-precision cast, written to the HF export and state dicts, and the gate
selects with it exactly as Transformers' NemotronHTopkRouter does. Nemotron-3-Nano's biases lie near 57, about 0.01
apart, where BF16 values are 0.25 apart."""

from __future__ import annotations

import json
import types

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig as HFNemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHTopkRouter

from arctic_platform.model.implementations.moe.hf_loader import load_hf_weights
from arctic_platform.model.implementations.moe.layers.moe import NemotronHRouter
from arctic_platform.model.implementations.moe.vllm_weights import named_weight_sync_tensors
from arctic_platform.model.implementations.nemotron_h.configuration_nemotron_h import NemotronHConfig
from arctic_platform.model.implementations.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM

EXPERTS = 8
BIAS_KEY = "model.layers.0.mlp.router.e_score_correction_bias"
HF_BIAS_KEY = "backbone.layers.0.mixer.gate.e_score_correction_bias"


def _checkpoint_bias() -> torch.Tensor:
    """Nemotron-3-Nano-like: near 56.75 with spreads under BF16's 0.25 step there; BF16 collapses them."""
    bias = 56.75 + torch.tensor([0.031, -0.012, 0.004, 0.022, -0.035, 0.017, -0.003, 0.009])
    assert bias.to(torch.bfloat16).unique().numel() == 1
    return bias


def _config() -> NemotronHConfig:
    config = NemotronHConfig(
        hidden_size=16,
        layers_block_type=["moe", "attention"],
        vocab_size=32,
        n_routed_experts=EXPERTS,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        moe_shared_expert_intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    config._attn_implementation = "sdpa"
    return config


def _bf16_model() -> NemotronHForCausalLM:
    # As the trainer builds it (qwen35/model_builder.get_model): from_config with the training dtype.
    return NemotronHForCausalLM._from_config(_config(), dtype=torch.bfloat16)


def test_bias_is_built_in_fp32_in_a_bf16_model():
    model = _bf16_model()
    router = model.model.layers[0].mlp.router
    assert model.lm_head.weight.dtype == torch.bfloat16 and router.gate.dtype == torch.bfloat16
    assert router.e_score_correction_bias.dtype == torch.float32
    with torch.device("meta"):
        meta = NemotronHForCausalLM._from_config(_config(), dtype=torch.bfloat16)
    meta.to_empty(device="cpu")
    assert meta.model.layers[0].mlp.router.e_score_correction_bias.dtype == torch.float32


def _write_hf_checkpoint(path, bias: torch.Tensor) -> dict[str, torch.Tensor]:
    model = NemotronHForCausalLM(_config()).float()
    torch.manual_seed(0)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    NemotronHForCausalLM.convert_to_hf(state)
    state[HF_BIAS_KEY] = bias.clone()
    path.mkdir()
    save_file({name: tensor.contiguous() for name, tensor in state.items()}, path / "model.safetensors")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in state}})
    )
    return state


def test_direct_hf_load_keeps_the_checkpoint_bias_unshifted_in_fp32(tmp_path):
    bias = _checkpoint_bias()
    _write_hf_checkpoint(tmp_path / "hf", bias)
    model = _bf16_model()
    # As qwen35/model_builder.load_dcp_from_hf: the model's own state dict tensors take the converted checkpoint.
    load_hf_weights(model.state_dict(), tmp_path / "hf", model.convert_layer_to_prime)
    loaded = model.model.layers[0].mlp.router.e_score_correction_bias
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, bias)


@pytest.mark.parametrize("buffer_dtype", [None, torch.bfloat16])
def test_deepspeed_mixed_precision_cast_keeps_the_bias_in_fp32(buffer_dtype):
    """DeepSpeed's bf16 module cast (runtime/engine.py _cast_module_mixed_precision) with keep_in_fp32_modules "auto":
    the parameters go to BF16, the bias stays FP32 even when a buffer_dtype casts the other buffers."""
    from deepspeed.runtime.engine import DeepSpeedEngine
    from deepspeed.runtime.keep_in_fp32 import keep_in_fp32_pattern

    model = NemotronHForCausalLM(_config()).float()
    router = model.model.layers[0].mlp.router
    router.e_score_correction_bias.copy_(_checkpoint_bias())
    pattern = keep_in_fp32_pattern(model, "auto", torch.bfloat16)
    assert pattern is not None and pattern.search(BIAS_KEY)
    engine = types.SimpleNamespace(module=model, _warn_listed_params_not_kept=lambda pattern: None)
    DeepSpeedEngine._cast_module_mixed_precision(engine, torch.bfloat16, buffer_dtype, False, pattern)
    assert router.gate.dtype == torch.bfloat16
    assert router.e_score_correction_bias.dtype == torch.float32
    assert torch.equal(router.e_score_correction_bias, _checkpoint_bias())


def test_selection_matches_transformers_router_bit_for_bit():
    """The gate adds the FP32 bias to the FP32 scores, as Transformers' router (and vLLM's grouped top-k) do, so it
    picks their experts; a BF16 copy of the same bias does not."""
    torch.manual_seed(3)
    ours = NemotronHRouter(dim=16, num_experts=EXPERTS, top_k=2, n_group=1, topk_group=1, norm_topk_prob=True)
    ours.init_weights(0.3)
    ours.e_score_correction_bias.copy_(_checkpoint_bias())
    hf_config = HFNemotronHConfig(
        hidden_size=16, n_routed_experts=EXPERTS, num_experts_per_tok=2, n_group=1, topk_group=1, norm_topk_prob=True
    )
    theirs = NemotronHTopkRouter(hf_config)
    theirs.weight.data.copy_(ours.gate.data)
    theirs.e_score_correction_bias.copy_(_checkpoint_bias())
    x = torch.randn(4096, 16)
    _, selected, _ = ours(x)
    _, _, hf_selected = theirs(x)
    assert torch.equal(torch.sort(selected, -1).values, torch.sort(hf_selected, -1).values)
    bf16 = ours.e_score_correction_bias.to(torch.bfloat16).float()
    scores = torch.sigmoid(x @ ours.gate.T)
    bf16_selected = torch.topk(scores + bf16, k=2, dim=-1).indices
    assert not torch.equal(torch.sort(bf16_selected, -1).values, torch.sort(selected, -1).values)


def test_state_dict_and_hf_export_carry_the_fp32_bias_and_weight_sync_leaves_it(tmp_path):
    """The state dict (DeepSpeed's module checkpoint) holds the FP32 bias and a restore keeps it; the HF export stream
    (a weights-only checkpoint: iter_model_weights(..., "hf", include_skipped=True)) writes it under the checkpoint's
    name and values; a weight sync, whose sampler loaded the same checkpoint, leaves it out."""
    bias = _checkpoint_bias()
    _write_hf_checkpoint(tmp_path / "hf", bias)
    model = _bf16_model()
    load_hf_weights(model.state_dict(), tmp_path / "hf", model.convert_layer_to_prime)

    state = model.state_dict()
    assert state[BIAS_KEY].dtype == torch.float32 and torch.equal(state[BIAS_KEY], bias)
    restored = _bf16_model()
    restored.load_state_dict(state)
    assert torch.equal(restored.model.layers[0].mlp.router.e_score_correction_bias, bias)

    exported = dict(named_weight_sync_tensors(model, include_export_buffers=True))
    assert exported[BIAS_KEY].dtype == torch.float32
    layer = {name: tensor for name, tensor in exported.items() if name.startswith("model.layers.0.")}
    NemotronHForCausalLM.convert_layer_to_hf(layer, 0)
    assert torch.equal(layer[HF_BIAS_KEY], bias)
    synced = dict(named_weight_sync_tensors(model, include_export_buffers=False))
    assert BIAS_KEY not in synced
