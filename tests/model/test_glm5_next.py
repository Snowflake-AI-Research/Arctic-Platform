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
import json

import pytest
import torch

from arctic_platform.model import ModelSpec
from arctic_platform.model import ParallelismConfig
from arctic_platform.model import PlatformCapabilities
from arctic_platform.model import resolve_model_profile
from arctic_platform.model.implementations.glm53.converting_glm5_next import convert_hf_layer_to_prime
from arctic_platform.model.implementations.glm53.converting_glm5_next import convert_prime_layer_to_hf
from arctic_platform.model.implementations.glm53.vllm_weights import convert_glm5_next_layer_to_vllm


def _checkpoint(tmp_path, model_type: str) -> str:
    path = tmp_path / model_type
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(path)


def _tiny_config():
    pytest.importorskip("transformers.models.glm5_next")
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig

    text = Glm5NextTextConfig(
        vocab_size=32,
        pad_token_id=0,
        eos_token_id=1,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_rope_head_dim=0,
        qk_nope_head_dim=4,
        v_head_dim=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=4,
        index_kpool=2,
        linear_num_heads=2,
        linear_head_dim=4,
        hc_mult=2,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "deepseek_sparse_attention",
        ],
        mlp_layer_types=["dense", "dense", "dense", "sparse"],
    )
    vision = Glm5NextVisionConfig(
        depth=1,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        out_hidden_size=16,
        projection_intermediate_size=16,
        image_size=14,
        patch_size=14,
    )
    return Glm5NextConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=2,
        video_token_id=3,
        image_start_token_id=4,
        image_end_token_id=5,
        video_start_token_id=6,
        video_end_token_id=7,
    )


def test_glm53_flash_family_dispatch_uses_model_type(tmp_path):
    glm5 = ModelSpec(
        model_path_or_name=_checkpoint(tmp_path, "glm5_next"),
        parallelism=ParallelismConfig(expert_parallel=2),
    )
    glm52 = ModelSpec(
        model_path_or_name=_checkpoint(tmp_path, "glm_moe_dsa"),
        parallelism=ParallelismConfig(expert_parallel=2),
    )

    assert glm5.loader == "glm5_next"
    assert glm52.loader == "glm_moe_dsa"


def test_glm53_custom_model_replaces_only_sparse_moe():
    from arctic_platform.model.implementations.glm53.modeling_glm5_next import Glm5NextForConditionalGenerationPrimeRL
    from arctic_platform.model.implementations.moe.layers.moe import MoE

    config = _tiny_config()
    config.use_grouped_mm = False
    with torch.device("meta"):
        model = Glm5NextForConditionalGenerationPrimeRL(config)

    layers = model.model.language_model.layers
    assert not isinstance(layers[0].mlp, MoE)
    assert isinstance(layers[3].mlp, MoE)
    assert layers[3].mlp.experts.use_grouped_mm is False
    assert layers[3].mlp.experts.swiglu_limit == 10.0
    assert model._is_vlm is True


def test_glm53_tiled_shared_expert_bypasses_wrapped_forward():
    from arctic_platform.model.implementations.glm53.deepspeed_integration import _shared_expert_forward
    from arctic_platform.model.implementations.moe.layers.moe import BCFeedForward

    feed_forward = BCFeedForward(dim=4, hidden_dim=8)
    with torch.no_grad():
        for parameter in feed_forward.parameters():
            parameter.normal_(std=0.1)
    hidden_states = torch.randn(3, 4)
    expected = BCFeedForward.forward(feed_forward, hidden_states)

    def wrapped_forward(_hidden_states):
        raise RecursionError

    feed_forward.forward = wrapped_forward
    torch.testing.assert_close(
        _shared_expert_forward(feed_forward, hidden_states),
        expected,
    )


def test_glm53_vlm_registry_selects_custom_model_and_language_stack():
    from arctic_platform.model.implementations.glm53.modeling_glm5_next import Glm5NextForConditionalGenerationPrimeRL
    from arctic_platform.model.implementations.moe.vlm import get_language_model
    from arctic_platform.model.implementations.moe.vlm import is_vlm_architecture

    config = _tiny_config()
    assert is_vlm_architecture(config)

    with torch.device("meta"):
        model = Glm5NextForConditionalGenerationPrimeRL(config)
    assert get_language_model(model) is model.model.language_model


def test_glm53_reinitializes_vision_rotary_buffer_after_meta_load():
    from arctic_platform.model.implementations.glm53.modeling_glm5_next import Glm5NextForConditionalGenerationPrimeRL

    with torch.device("meta"):
        model = Glm5NextForConditionalGenerationPrimeRL(_tiny_config())
    model.to_empty(device="cpu")
    model.init_buffers_post_meta()

    rotary = model.model.visual.rotary_pos_emb
    if hasattr(rotary, "compute_axial_rope_parameters"):
        expected, _ = rotary.compute_axial_rope_parameters(
            rotary.config,
            rotary.inv_freq.device,
        )
        torch.testing.assert_close(rotary.original_inv_freq, expected)
    else:
        expected = 1.0 / (rotary.theta ** (torch.arange(0, rotary.dim, 2, dtype=torch.float32) / rotary.dim))
    torch.testing.assert_close(rotary.inv_freq, expected)


def test_glm53_weight_sync_includes_router_bias_buffer():
    from arctic_platform.model.implementations.glm53.modeling_glm5_next import Glm5NextForConditionalGenerationPrimeRL
    from arctic_platform.model.implementations.moe.vllm_weights import named_weight_sync_tensors

    with torch.device("meta"):
        model = Glm5NextForConditionalGenerationPrimeRL(_tiny_config())

    names = {name for name, _ in named_weight_sync_tensors(model)}
    prefix = "model.language_model.layers.3.mlp"
    assert f"{prefix}.expert_bias" in names
    assert f"{prefix}.tokens_per_expert" not in names


def test_glm53_hf_prime_conversion_round_trip():
    prefix = "model.language_model.layers.3"
    state = {
        f"{prefix}.hc_attn_fn": torch.randn(8, 32),
        f"{prefix}.hc_attn_base": torch.randn(8),
        f"{prefix}.hc_attn_scale": torch.randn(3),
        f"{prefix}.hc_ffn_fn": torch.randn(8, 32),
        f"{prefix}.hc_ffn_base": torch.randn(8),
        f"{prefix}.hc_ffn_scale": torch.randn(3),
        f"{prefix}.mlp.gate.weight": torch.randn(2, 4),
        f"{prefix}.mlp.gate.e_score_correction_bias": torch.randn(2),
        f"{prefix}.mlp.shared_experts.gate_proj.weight": torch.randn(3, 4),
        f"{prefix}.mlp.shared_experts.gate_proj.weight_scale_inv": torch.rand(1, 1),
        f"{prefix}.mlp.shared_experts.down_proj.weight": torch.randn(4, 3),
        f"{prefix}.mlp.shared_experts.down_proj.weight_scale_inv": torch.rand(1, 1),
        f"{prefix}.mlp.shared_experts.up_proj.weight": torch.randn(3, 4),
        f"{prefix}.mlp.shared_experts.up_proj.weight_scale_inv": torch.rand(1, 1),
    }
    for expert_idx in range(2):
        state[f"{prefix}.mlp.experts.{expert_idx}.gate_proj.weight"] = torch.randn(3, 4)
        state[f"{prefix}.mlp.experts.{expert_idx}.gate_proj.weight_scale_inv"] = torch.rand(1, 1)
        state[f"{prefix}.mlp.experts.{expert_idx}.down_proj.weight"] = torch.randn(4, 3)
        state[f"{prefix}.mlp.experts.{expert_idx}.down_proj.weight_scale_inv"] = torch.rand(1, 1)
        state[f"{prefix}.mlp.experts.{expert_idx}.up_proj.weight"] = torch.randn(3, 4)
        state[f"{prefix}.mlp.experts.{expert_idx}.up_proj.weight_scale_inv"] = torch.rand(1, 1)
    expected = {name: tensor.clone() for name, tensor in state.items()}

    convert_hf_layer_to_prime(state, 3)
    assert f"{prefix}.mlp.experts.w1" in state
    assert f"{prefix}.mlp.experts.w1_scale_inv" in state
    assert f"{prefix}.attn_hc.fn" in state
    convert_prime_layer_to_hf(state, 3)

    assert state.keys() == expected.keys()
    for name, tensor in expected.items():
        torch.testing.assert_close(state[name], tensor)


def test_glm53_vllm_packer_fuses_kda_and_dense_mlp():
    prefix = "model.language_model.layers.0"
    attn = f"{prefix}.self_attn"
    state = {
        f"{attn}.q_proj.weight": torch.randn(8, 4),
        f"{attn}.k_proj.weight": torch.randn(8, 4),
        f"{attn}.v_proj.weight": torch.randn(8, 4),
        f"{attn}.b_proj.weight": torch.randn(2, 4),
        f"{attn}.forget_gate.f_a_proj.weight": torch.randn(3, 4),
        f"{attn}.g_a_proj.weight": torch.randn(3, 4),
        f"{attn}.forget_gate.f_b_proj.weight": torch.randn(8, 3),
        f"{attn}.forget_gate.A_log": torch.randn(2),
        f"{attn}.forget_gate.dt_bias": torch.randn(8),
        f"{attn}.conv1d.weight": torch.randn(24, 1, 4),
        f"{prefix}.mlp.gate_proj.weight": torch.randn(6, 4),
        f"{prefix}.mlp.up_proj.weight": torch.randn(6, 4),
        f"{prefix}.mlp.down_proj.weight": torch.randn(4, 6),
        f"{prefix}.attn_hc.fn": torch.randn(8, 16),
    }

    convert_glm5_next_layer_to_vllm(state, 0)

    assert state[f"{attn}.in_proj_qkvbfg_a.weight"].shape == (32, 4)
    assert state[f"{attn}.q_conv1d.weight"].shape == (8, 1, 4)
    assert f"{attn}.A_log" in state
    assert state[f"{prefix}.mlp.gate_up_proj.weight"].shape == (12, 4)
    assert f"{prefix}.hc_attn_fn" in state


def test_glm53_vllm_packer_fuses_sparse_mla_and_moe():
    prefix = "model.language_model.layers.3"
    attn = f"{prefix}.self_attn"
    state = {
        f"{attn}.q_a_proj.weight": torch.randn(3, 4),
        f"{attn}.kv_a_proj_with_mqa.weight": torch.randn(2, 4),
        f"{attn}.indexer.wk.weight": torch.randn(5, 4),
        f"{attn}.indexer.weights_proj.weight": torch.randn(2, 4),
        f"{prefix}.mlp.router.gate.weight": torch.randn(2, 4),
        f"{prefix}.mlp.expert_bias": torch.randn(2),
        f"{prefix}.mlp.experts.w1": torch.randn(2, 3, 4),
        f"{prefix}.mlp.experts.w2": torch.randn(2, 4, 3),
        f"{prefix}.mlp.experts.w3": torch.randn(2, 3, 4),
        f"{prefix}.mlp.shared_expert.w1": torch.randn(3, 4),
        f"{prefix}.mlp.shared_expert.w2": torch.randn(4, 3),
        f"{prefix}.mlp.shared_expert.w3": torch.randn(3, 4),
    }

    convert_glm5_next_layer_to_vllm(state, 3)

    assert state[f"{attn}.fused_qkv_a_proj.weight"].shape == (5, 4)
    assert state[f"{attn}.indexer.wk_weights_proj.weight"].shape == (7, 4)
    assert state[f"{prefix}.mlp.experts.w13_weight"].shape == (2, 6, 4)
    assert state[f"{prefix}.mlp.shared_experts.gate_up_proj.weight"].shape == (
        6,
        4,
    )
    assert f"{prefix}.mlp.gate.e_score_correction_bias" in state


def test_glm53_native_fp8_training_builds_quantized_modules():
    from arctic_platform.model.implementations.fp8 import BlockFp8Linear
    from arctic_platform.model.implementations.glm53.modeling_glm5_next import Glm5NextForConditionalGenerationPrimeRL

    config = _tiny_config()
    config.quantization_config = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": [f"model.layers.{layer_idx}.self_attn" for layer_idx in range(4)],
    }
    with torch.device("meta"):
        model = Glm5NextForConditionalGenerationPrimeRL(config)

    layers = model.model.language_model.layers
    assert isinstance(layers[0].mlp.gate_proj, BlockFp8Linear)
    assert layers[3].mlp.experts.w1.dtype == torch.float8_e4m3fn
    assert layers[3].mlp.experts.w1_scale_inv.dtype == torch.float32
    assert layers[3].mlp.experts.w1_scale_inv._dss_keep_fp32
    assert isinstance(layers[3].mlp.router.gate, torch.nn.Linear)
    assert not isinstance(layers[3].mlp.router.gate, BlockFp8Linear)


def test_glm53_defaults_to_sparse_mla_and_allows_context_parallelism(tmp_path):
    spec = ModelSpec(
        model_path_or_name=_checkpoint(tmp_path, "glm5_next"),
        parallelism=ParallelismConfig(expert_parallel=2, sequence_parallel=2),
    )
    profile = resolve_model_profile(
        spec,
        PlatformCapabilities.for_accelerator("hopper"),
    )

    assert spec.loader == "glm5_next"
    assert profile.attn_implementation == "sparse_mla"
    assert profile.ep_comm_backend == "uccl"


def test_glm53_sparse_reference_supports_rectangular_context_parallelism():
    from arctic_platform.model.implementations.glm53.context_parallel import _reference_sparse_mla

    torch.manual_seed(0)
    query = torch.randn(1, 2, 2, 4, requires_grad=True)
    key_value = torch.randn(1, 5, 1, 4, requires_grad=True)
    indices = torch.tensor([[[0, 2, -1], [1, 3, -1]]], dtype=torch.int32)

    output = _reference_sparse_mla(
        query,
        key_value,
        indices,
        scale=0.5,
        value_dim=3,
    )
    output.square().sum().backward()

    assert output.shape == (1, 2, 2, 3)
    assert query.grad is not None
    assert key_value.grad is not None


def test_glm53_sparse_indices_are_padded_for_flashmla():
    from arctic_platform.model.implementations.glm53.context_parallel import _pad_sparse_indices

    indices = torch.arange(2051, dtype=torch.int32).view(1, 1, -1)
    padded = _pad_sparse_indices(indices)

    assert padded.shape[-1] == 2176
    torch.testing.assert_close(padded[..., :2051], indices)
    assert torch.all(padded[..., 2051:] == -1)


def test_glm53_sparse_mla_full_forward_backward_without_sdpa():
    from arctic_platform.model.implementations.glm53.context_parallel import apply_context_parallelism
    from arctic_platform.model.implementations.glm53.modeling_glm5_next import Glm5NextForConditionalGenerationPrimeRL

    config = _tiny_config()
    config.use_cache = False
    config.text_config.use_cache = False
    model = Glm5NextForConditionalGenerationPrimeRL(config)
    language_model = model.model.language_model
    apply_context_parallelism(model, cp_size=1, cp_group=None)
    sparse_attention = language_model.layers[-1].self_attn
    hidden_states = torch.randn(1, 4, config.text_config.hidden_size)
    output, _, _ = sparse_attention(
        hidden_states,
        attention_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    output.square().sum().backward()

    assert sparse_attention.q_a_proj.weight.grad is not None
    assert sparse_attention.kv_a_proj_with_mqa.weight.grad is not None
    assert sparse_attention.indexer.wq_b.weight.grad is None
