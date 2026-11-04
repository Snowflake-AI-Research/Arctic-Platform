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
import importlib
import json
import types
from unittest.mock import Mock

import pytest
import torch

from arctic_platform.model import ModelSpec
from arctic_platform.model import ParallelismConfig
from arctic_platform.model.implementations.glm53.converting_glm5_next import convert_hf_layer_to_prime
from arctic_platform.model.implementations.glm53.converting_glm5_next import convert_prime_layer_to_hf
from arctic_platform.model.implementations.glm53.vllm_weights import convert_glm5_next_layer_to_vllm
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal


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


def test_glm53_decoder_forwards_its_physical_layer_routes():
    from arctic_platform.model.implementations.glm53.modeling_glm5_next import Glm5NextForConditionalGenerationPrimeRL

    class SparseAttention(torch.nn.Module):
        def forward(self, hidden_states, **_kwargs):
            return torch.zeros_like(hidden_states), None, None

    def record_routes(self, hidden_states, routed_experts=None):
        self.routed_experts = routed_experts
        return torch.zeros_like(hidden_states)

    config = _tiny_config()
    config.use_grouped_mm = False
    model = Glm5NextForConditionalGenerationPrimeRL(config)
    layer = model.model.language_model.layers[3]
    layer.self_attn = SparseAttention()
    layer.mlp.forward = types.MethodType(record_routes, layer.mlp)
    routes = torch.arange(16).reshape(1, 2, 4, 2)

    layer(torch.randn(1, 2, 2, 16), routed_experts=routes)

    torch.testing.assert_close(layer.mlp.routed_experts, routes[:, :, 3, :])


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


def test_glm53_vllm_packer_fuses_fp8_moe_scale_inv():
    """Native FP8 expert scales must pack to vLLM w13/w2_weight_scale_inv keys."""
    prefix = "model.language_model.layers.3"
    attn = f"{prefix}.self_attn"
    w1_scale = torch.rand(2, 1, 1)
    w3_scale = torch.rand(2, 1, 1)
    w2_scale = torch.rand(2, 1, 1)
    sw1_scale = torch.rand(1, 1)
    sw3_scale = torch.rand(1, 1)
    sw2_scale = torch.rand(1, 1)
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
        f"{prefix}.mlp.experts.w1_scale_inv": w1_scale.clone(),
        f"{prefix}.mlp.experts.w2_scale_inv": w2_scale.clone(),
        f"{prefix}.mlp.experts.w3_scale_inv": w3_scale.clone(),
        f"{prefix}.mlp.shared_expert.w1": torch.randn(3, 4),
        f"{prefix}.mlp.shared_expert.w2": torch.randn(4, 3),
        f"{prefix}.mlp.shared_expert.w3": torch.randn(3, 4),
        f"{prefix}.mlp.shared_expert.w1_scale_inv": sw1_scale.clone(),
        f"{prefix}.mlp.shared_expert.w2_scale_inv": sw2_scale.clone(),
        f"{prefix}.mlp.shared_expert.w3_scale_inv": sw3_scale.clone(),
    }

    convert_glm5_next_layer_to_vllm(state, 3)

    assert f"{prefix}.mlp.experts.w1_scale_inv" not in state
    assert f"{prefix}.mlp.experts.w2_scale_inv" not in state
    assert f"{prefix}.mlp.experts.w3_scale_inv" not in state
    assert not any(key.startswith(f"{prefix}.mlp.shared_expert.") for key in state)
    torch.testing.assert_close(
        state[f"{prefix}.mlp.experts.w13_weight_scale_inv"],
        torch.cat([w1_scale, w3_scale], dim=1),
    )
    torch.testing.assert_close(
        state[f"{prefix}.mlp.experts.w2_weight_scale_inv"],
        w2_scale,
    )
    torch.testing.assert_close(
        state[f"{prefix}.mlp.shared_experts.gate_up_proj.weight_scale_inv"],
        torch.cat([sw1_scale, sw3_scale], dim=0),
    )
    torch.testing.assert_close(
        state[f"{prefix}.mlp.shared_experts.down_proj.weight_scale_inv"],
        sw2_scale,
    )


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

    assert spec.loader == "glm5_next"
    assert spec.attn_implementation == "sparse_mla"


@pytest.mark.parametrize("backend", ["sparse_mla", "flashinfer", "flashmla"])
def test_glm53_spec_accepts_sparse_backend_selection(tmp_path, backend):
    spec = ModelSpec(
        model_path_or_name=_checkpoint(tmp_path, "glm5_next"),
        attn_implementation=backend,
    )
    assert spec.attn_implementation == backend


@pytest.mark.parametrize("backend", ["sparse_mla", "flashinfer", "flashmla"])
def test_glm53_loader_preserves_backend_and_constructs_hf_eager_attention(monkeypatch, backend):
    from arctic_platform.model.implementations.glm53 import deepspeed_integration as integration
    from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions

    config = integration._build_model_config("unused", 1, 1, "bfloat16", backend, Qwen3_5MoeOptions())
    assert config.attn == backend
    get_model = Mock(return_value=torch.nn.Module())
    monkeypatch.setattr(integration.qwen_ds, "_generic_adapter", lambda: types.SimpleNamespace(get_model=get_model))
    model = integration._get_model(config, torch.device("meta"), torch.bfloat16)
    assert model is get_model.return_value
    assert get_model.call_args.args[0].attn == "eager"
    assert config.attn == backend


@pytest.mark.parametrize("backend", ["sparse_mla", "flashinfer", "flashmla"])
def test_glm53_deepspeed_adapter_forwards_backend_at_sp1(monkeypatch, backend):
    from arctic_platform.model.implementations.glm53 import deepspeed_integration as integration

    apply_cp = Mock()
    monkeypatch.setattr(integration, "apply_context_parallelism", apply_cp)
    model = torch.nn.Module()
    integration._adapter(backend).apply_sequence_parallelism(model, 1, None)
    apply_cp.assert_called_once_with(model, 1, None, attn_implementation=backend)


@pytest.mark.parametrize("backend", ["sparse_mla", "flashinfer", "flashmla"])
def test_glm53_deepspeed_loader_uses_model_config_backend(monkeypatch, backend):
    from arctic_platform.model.implementations.glm53 import deepspeed_integration as integration
    from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions

    config = integration._build_model_config("unused", 1, 1, "bfloat16", backend, Qwen3_5MoeOptions())
    model = torch.nn.Module()
    apply_cp = Mock()
    monkeypatch.setattr(integration, "apply_context_parallelism", apply_cp)

    def load(adapter, *_args, **kwargs):
        adapter.apply_sequence_parallelism(model, kwargs["sp_size"], kwargs["sp_group"])
        return model

    monkeypatch.setattr(integration.qwen_ds, "_load_moe_model_for_deepspeed", load)
    assert integration.load_glm5_next_model_for_deepspeed(config, None, None, "unused") is model
    apply_cp.assert_called_once_with(model, 1, None, attn_implementation=backend)


def _assert_sparse_mla_gradient_close(actual, expected, dtype, *, is_kv):
    # FP16 KV scatter sums 4 queries x 64 heads; H200 cancellation residuals reached 2.4e-4.
    # Keep query/BF16 checks strict and bound only FP16 KV's absolute floor to 3e-4.
    atol = 3e-4 if dtype == torch.float16 and is_kv else 1e-6
    torch_assert_close(actual, expected.to(dtype), atol=atol, rtol=torch.finfo(dtype).eps)


@pytest.mark.parametrize("dtype,is_kv", [(torch.float16, True), (torch.float16, False), (torch.bfloat16, True)])
def test_sparse_mla_gradient_tolerance_is_bounded(dtype, is_kv):
    expected = torch.zeros(2, dtype=dtype)
    allowed = 2.4e-4 if dtype == torch.float16 and is_kv else 5e-7
    _assert_sparse_mla_gradient_close(expected + allowed, expected, dtype, is_kv=is_kv)
    rejected = 5e-4 if dtype == torch.float16 and is_kv else 2e-6
    with pytest.raises(AssertionError):
        _assert_sparse_mla_gradient_close(expected + rejected, expected, dtype, is_kv=is_kv)


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


def test_glm53_sparse_reference_tiles_match_dense():
    from arctic_platform.model.implementations.glm53 import context_parallel as cp

    torch.manual_seed(0)
    query = torch.randn(1, 5, 2, 4, requires_grad=True)
    key_value = torch.randn(1, 7, 1, 4, requires_grad=True)
    indices = torch.tensor(
        [[[0, 2, -1, 6], [1, 3, 4, -1], [0, 1, 2, 3], [-1, -1, -1, -1], [5, 5, 0, 2]]],
        dtype=torch.int32,
    )
    scale = 0.5
    value_dim = 3

    dense = cp._reference_sparse_mla_math(query, key_value, indices, scale, value_dim)
    dense.square().sum().backward()
    query_grad = query.grad.detach().clone()
    key_grad = key_value.grad.detach().clone()
    query.grad = None
    key_value.grad = None

    previous = cp._REFERENCE_SPARSE_MLA_TILE_BYTES
    cp._REFERENCE_SPARSE_MLA_TILE_BYTES = 64
    try:
        output = cp._reference_sparse_mla(
            query,
            key_value,
            indices,
            scale=scale,
            value_dim=value_dim,
        )
        output.square().sum().backward()
    finally:
        cp._REFERENCE_SPARSE_MLA_TILE_BYTES = previous

    torch.testing.assert_close(output, dense)
    torch.testing.assert_close(query.grad, query_grad)
    torch.testing.assert_close(key_value.grad, key_grad)


@require_torch_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_glm53_flashinfer_sparse_mla_matches_reference(monkeypatch, dtype):
    pytest.importorskip("flashinfer")
    from arctic_platform.model.implementations.glm53 import context_parallel as cp

    set_seed(0)
    device = "cuda"
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FlashInfer FA3 MLA requires Hopper")
    query = (torch.randn(1, 4, 64, 512, device=device, dtype=dtype) * 0.05).requires_grad_()
    key_value = (torch.randn(1, 33, 1, 512, device=device, dtype=dtype) * 0.05).requires_grad_()
    indices = torch.randint(0, 32, (1, 4, 16), device=device, dtype=torch.int32)
    indices[:, :, -3:] = -1
    indices[0, 0, :] = -1
    scale = 512**-0.5

    assert cp._use_flashinfer(query, key_value)
    reference_query = query.detach().float().requires_grad_()
    reference_kv = key_value.detach().float().requires_grad_()
    dense = cp._reference_sparse_mla_math(reference_query, reference_kv, indices, scale, 512).to(dtype)
    output = cp._flashinfer_sparse_mla_apply(query, key_value, indices, scale=scale, value_dim=512)
    monkeypatch.setattr(cp, "_FLASHINFER_MAX_REQUESTS", 2)
    monkeypatch.setattr(cp, "_REFERENCE_SPARSE_MLA_TILE_BYTES", 16 * 512 * 4)
    chunked = cp._flashinfer_sparse_mla_apply(query, key_value, indices, scale=scale, value_dim=512)
    # Permit one dtype rounding unit per element and a fixed CUDA-cancellation floor.
    for actual in (output, chunked):
        torch_assert_close(actual, dense, atol=1e-4, rtol=torch.finfo(dtype).eps)
        torch_assert_equal(actual[:, 0], torch.zeros_like(actual[:, 0]))
    gradient = torch.randn_like(dense)
    expected_grads = torch.autograd.grad(dense, (reference_query, reference_kv), gradient)
    for actual in (output, chunked):
        actual_grads = torch.autograd.grad(actual, (query, key_value), gradient)
        for index, (actual_grad, expected_grad) in enumerate(zip(actual_grads, expected_grads, strict=True)):
            assert torch.isfinite(actual_grad).all()
            _assert_sparse_mla_gradient_close(actual_grad, expected_grad, dtype, is_kv=index == 1)


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


def test_glm53_sparse_mla_applies_kv_b_proj_lora():
    """The absorbed sparse path must see a kv_b_proj LoRA exactly as if it were merged into the weight."""
    from peft import LoraConfig
    from peft import get_peft_model

    from arctic_platform.model.implementations.glm53.context_parallel import apply_context_parallelism
    from arctic_platform.model.implementations.glm53.modeling_glm5_next import Glm5NextForConditionalGenerationPrimeRL

    set_seed(0)
    config = _tiny_config()
    config.use_cache = False
    config.text_config.use_cache = False
    model = Glm5NextForConditionalGenerationPrimeRL(config)
    apply_context_parallelism(model, cp_size=1, cp_group=None)
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=2, target_modules=["kv_b_proj"]))
    sparse_attention = model.base_model.model.model.language_model.layers[-1].self_attn
    kv_b_proj = sparse_attention.kv_b_proj
    torch.nn.init.normal_(kv_b_proj.lora_B["default"].weight)
    hidden_states = torch.randn(1, 4, config.text_config.hidden_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    output, _, _ = sparse_attention(hidden_states, attention_mask=mask)
    output.square().sum().backward()
    assert kv_b_proj.lora_A["default"].weight.grad.abs().sum() > 0
    assert kv_b_proj.lora_B["default"].weight.grad.abs().sum() > 0

    with torch.no_grad():
        kv_b_proj.base_layer.weight.add_(kv_b_proj.get_delta_weight("default"))
        kv_b_proj.lora_B["default"].weight.zero_()
        merged, _, _ = sparse_attention(hidden_states, attention_mask=mask)
    torch_assert_close(output.detach(), merged)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused indexer path runs on CUDA")
def test_glm53_fused_indexer_selects_the_general_paths_keys():
    """One packed row takes the fused path; a batch of two identical rows takes the general path."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer

    from arctic_platform.model.implementations.glm53.context_parallel import _select_sparse_indices

    config = _tiny_config().text_config
    config.hidden_size, config.q_lora_rank = 64, 32
    config.index_n_heads, config.index_head_dim, config.index_topk, config.index_kpool = 4, 128, 64, 4
    config.index_kpool_always_select_tail = True
    torch.manual_seed(0)
    indexer = Glm5NextTextIndexer(config, layer_idx=3)
    with torch.no_grad():
        indexer.index_kpool_compress_ape.normal_()
        indexer.index_kpool_compress_gate.normal_(std=0.1)
    indexer = indexer.bfloat16().cuda()
    length = 1030  # two packed documents; the second ends inside a pool
    hidden = torch.randn(1, length, config.hidden_size, device="cuda").bfloat16()
    q_resid = torch.randn(1, length, config.q_lora_rank, device="cuda").bfloat16()
    cu_seqlens = torch.tensor([0, 600, length], dtype=torch.int32, device="cuda")

    def select(rows):
        return _select_sparse_indices(
            indexer,
            hidden.expand(rows, -1, -1),
            q_resid.expand(rows, -1, -1),
            torch.ones(rows, length, dtype=torch.bool, device="cuda"),
            cu_seqlens,
            cp_group=None,
            cp_rank=0,
        )[0]

    fused, general = select(1), select(2)

    assert fused.shape == general.shape
    assert torch.equal((fused >= 0).sum(-1), (general >= 0).sum(-1))
    # Same inputs; scores differ only in FP32 accumulation order, so only exact near-ties may swap.
    rows_equal = (fused.sort(-1).values == general.sort(-1).values).all(-1).float().mean().item()
    assert rows_equal > 0.995


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashMLA forward with the fused TileLang backward")
def test_glm53_nope_sparse_mla_matches_reference_through_the_kernels():
    """GLM-5.3-Flash's 512-wide NoPE latent runs the 576-wide kernels with a zero-padded tail."""
    import torch.nn.functional as F

    from arctic_platform.model.implementations.glm52.models.glm_moe_dsa.sparse_mla_attention import (
        sparse_mla_flashmla_fused_apply,
    )
    from arctic_platform.model.implementations.glm52.models.kernels.sparse_mla_flashmla import flash_mla_available
    from arctic_platform.model.implementations.glm53.context_parallel import _reference_sparse_mla

    if not flash_mla_available():
        pytest.skip("flash_mla / vllm._flashmla_C not installed")
    torch.manual_seed(0)
    queries, keys, heads, latent, topk = 256, 1024, 64, 512, 256
    query = (torch.randn(1, queries, heads, latent, device="cuda") * 0.5).bfloat16()
    key_value = (torch.randn(1, keys + 1, 1, latent, device="cuda") * 0.5).bfloat16()
    key_value[:, keys] = 0
    # The queries are the last shard of a longer key sequence, as under context parallelism.
    positions = torch.arange(keys - queries, keys, device="cuda")
    indices = torch.randint(0, keys, (1, queries, topk), device="cuda")
    indices = torch.where(indices <= positions[None, :, None], indices, -1).to(torch.int32)

    q, kv = query.clone().requires_grad_(True), key_value.clone().requires_grad_(True)
    q_ref, kv_ref = query.float().requires_grad_(True), key_value.float().requires_grad_(True)
    sentinel = indices.masked_fill(indices < 0, keys)
    out = sparse_mla_flashmla_fused_apply(F.pad(q, (0, 64)), F.pad(kv, (0, 64)), sentinel.unsqueeze(2), 0.0625)
    out_ref = _reference_sparse_mla(q_ref, kv_ref, indices, scale=0.0625, value_dim=latent)
    grad = torch.randn_like(out_ref)
    out.backward(grad.bfloat16())
    out_ref.backward(grad)

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    assert type(out.grad_fn).__name__ == "_SparseMLATilelangBackward"
    assert rel(out, out_ref) < 1e-2
    assert rel(q.grad, q_ref.grad) < 1e-2
    assert rel(kv.grad[:, :keys], kv_ref.grad[:, :keys]) < 1e-2


@pytest.mark.parametrize("target_format", ["prime", "hf"])
def test_glm53_streaming_checkpoint_matches_whole_state_conversion(tmp_path, monkeypatch, target_format):
    from safetensors.torch import save_file

    from arctic_platform.model.implementations.glm53 import modeling_glm5_next
    from arctic_platform.model.implementations.moe.weights import load_state_dict

    set_seed(0)
    model_class = modeling_glm5_next.Glm5NextForConditionalGenerationPrimeRL
    original = dict(model_class(_tiny_config()).state_dict())
    # Native expert tensors are allocated empty; conversion needs finite, distinct fixture values.
    original = {
        name: torch.arange(weight.numel(), dtype=weight.dtype).reshape(weight.shape) + index
        for index, (name, weight) in enumerate(original.items())
    }
    if target_format == "prime":
        original = model_class.convert_to_hf(original)
    expected = getattr(model_class, f"convert_to_{target_format}")(
        {name: tensor.clone() for name, tensor in original.items()}
    )
    source = tmp_path / "source"
    source.mkdir()
    names = tuple(original)
    for shard_index in range(2):
        save_file(
            {name: original[name].clone().contiguous() for name in names[shard_index::2]},
            source / f"model-{shard_index}.safetensors",
        )

    observed_groups = []
    convert = getattr(model_class, f"convert_to_{target_format}")

    def observe_group(cls, state):
        observed_groups.append(tuple(state))
        return convert(state)

    monkeypatch.setattr(model_class, f"convert_to_{target_format}", classmethod(observe_group))
    destination = tmp_path / "converted"
    getattr(model_class, f"convert_to_{target_format}_checkpoint")(source, destination)

    converted = load_state_dict(destination)
    assert converted.keys() == expected.keys()
    for name, weight in expected.items():
        torch_assert_equal(converted[name], weight)
    assert len(observed_groups) > 4
    for group in observed_groups:
        layer_prefixes = {name.split(".", 4)[3] for name in group if name.startswith("model.language_model.layers.")}
        assert len(layer_prefixes) <= 1
        if not layer_prefixes:
            assert len(group) == 1
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    assert index["weight_map"].keys() == expected.keys()


@pytest.mark.parametrize("groups", [[["a"]], [["a", "b"], ["b"]], [["a", "b"], []], [["a", "b", "c"]]])
def test_checkpoint_conversion_rejects_invalid_dependency_groups(tmp_path, groups):
    from safetensors.torch import save_file

    from arctic_platform.model.implementations.moe.weights import convert_safetensor_checkpoint

    source = tmp_path / "source"
    source.mkdir()
    save_file({"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}, source / "model.safetensors")
    destination = tmp_path / "converted"
    calls = []

    with pytest.raises(ValueError, match="cover every source tensor exactly once"):
        convert_safetensor_checkpoint(source, destination, calls.append, lambda names: groups)

    assert not destination.exists()
    assert calls == []


@pytest.mark.parametrize("failure", ["empty", "overlapping"])
def test_checkpoint_conversion_rejects_invalid_converted_groups(tmp_path, failure):
    from safetensors.torch import save_file

    from arctic_platform.model.implementations.moe.weights import convert_safetensor_checkpoint

    source = tmp_path / "source"
    source.mkdir()
    save_file({"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}, source / "model.safetensors")
    destination = tmp_path / "converted"

    def convert(state):
        tensor = next(iter(state.values()))
        state.clear()
        if failure == "overlapping":
            state["output"] = tensor

    with pytest.raises(ValueError, match="empty or overlapping groups"):
        convert_safetensor_checkpoint(source, destination, convert, lambda names: [(name,) for name in names])

    assert not (destination / "model.safetensors.index.json").exists()


@pytest.mark.parametrize("failure", ["empty-source", "duplicate-source"])
def test_checkpoint_conversion_rejects_invalid_source(tmp_path, failure):
    from safetensors.torch import save_file

    from arctic_platform.model.implementations.moe.weights import convert_safetensor_checkpoint

    source = tmp_path / "source"
    source.mkdir()
    if failure == "duplicate-source":
        for index in range(2):
            save_file({"a": torch.tensor([1.0])}, source / f"model-{index}.safetensors")
    destination = tmp_path / "converted"
    message = "No safetensors weights" if failure == "empty-source" else "Duplicate checkpoint tensor"

    with pytest.raises(ValueError, match=message):
        convert_safetensor_checkpoint(source, destination, lambda state: None, lambda names: [names])

    assert not destination.exists()


def test_checkpoint_conversion_shard_failure_never_publishes_index(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    from arctic_platform.model.implementations.moe import weights

    source = tmp_path / "source"
    source.mkdir()
    save_file({"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}, source / "model.safetensors")
    destination = tmp_path / "converted"
    calls = []

    def save(state, path, **kwargs):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("storage full")
        save_file(state, path, **kwargs)

    monkeypatch.setattr(weights, "save_file", save)
    with pytest.raises(OSError, match="storage full"):
        weights.convert_safetensor_checkpoint(
            source, destination, lambda state: None, lambda names: [(name,) for name in names]
        )

    assert len(calls) == 2
    assert calls[0].is_file()
    assert not (destination / "model.safetensors.index.json").exists()


def test_checkpoint_conversion_uses_source_index_and_converter_return(tmp_path):
    from safetensors.torch import save_file

    from arctic_platform.model.implementations.moe.weights import convert_safetensor_checkpoint
    from arctic_platform.model.implementations.moe.weights import load_state_dict
    from arctic_platform.model.implementations.moe.weights import load_state_dict_keys

    source = tmp_path / "source"
    source.mkdir()
    save_file({"a": torch.tensor([1.0]), "unindexed": torch.tensor([3.0])}, source / "indexed.safetensors")
    save_file({"a": torch.tensor([99.0])}, source / "stray.safetensors")
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "indexed.safetensors"}}))
    assert load_state_dict_keys(source) == ["a"]
    original = load_state_dict(source)
    assert set(original) == {"a"}
    torch_assert_equal(original["a"], torch.tensor([1.0]))
    destination = tmp_path / "converted"
    convert_safetensor_checkpoint(source, destination, lambda state: {"b": state["a"] + 1}, lambda names: [names])
    converted = load_state_dict(destination)
    assert set(converted) == {"b"}
    torch_assert_equal(converted["b"], torch.tensor([2.0]))


def test_unindexed_checkpoint_readers_keep_last_file_wins_on_duplicate_names(tmp_path):
    from safetensors.torch import save_file

    from arctic_platform.model.implementations.moe import weights

    save_file({"a": torch.tensor([1.0]), "first-only": torch.zeros(1)}, tmp_path / "consolidated.safetensors")
    save_file({"a": torch.tensor([2.0]), "last-only": torch.ones(1)}, tmp_path / "model.safetensors")
    assert set(weights.load_state_dict_keys(tmp_path)) == {"a", "first-only", "last-only"}
    loaded = weights.load_state_dict(tmp_path)
    assert set(loaded) == {"a", "first-only", "last-only"}
    torch_assert_equal(loaded["a"], torch.tensor([2.0]))
    with pytest.raises(ValueError, match="Duplicate checkpoint tensor: a"):
        weights.convert_safetensor_checkpoint(
            tmp_path, tmp_path / "converted", lambda state: state, lambda names: [names]
        )


@pytest.mark.parametrize("index", [{}, {"weight_map": []}, {"weight_map": {"a": "../outside.safetensors"}}])
def test_checkpoint_readers_reject_invalid_source_index(tmp_path, index):
    from arctic_platform.model.implementations.moe import weights

    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    readers = [
        weights.load_state_dict_keys,
        weights.load_state_dict,
        lambda source: weights.convert_safetensor_checkpoint(
            source, source / "converted", lambda state: state, lambda names: [names]
        ),
    ]
    for reader in readers:
        with pytest.raises(ValueError, match="Checkpoint"):
            reader(tmp_path)


def test_checkpoint_readers_accept_hub_symlinks_outside_snapshot(tmp_path):
    from safetensors.torch import save_file

    from arctic_platform.model.implementations.moe import weights

    blob = tmp_path / "blobs" / "hash"
    blob.parent.mkdir()
    save_file({"a": torch.tensor([1.0])}, blob)
    source = tmp_path / "snapshots" / "revision"
    source.mkdir(parents=True)
    (source / "shard.safetensors").symlink_to(blob)
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "shard.safetensors"}}))
    assert weights.load_state_dict_keys(source) == ["a"]
    torch_assert_equal(weights.load_state_dict(source)["a"], torch.tensor([1.0]))
    converted = tmp_path / "converted"
    weights.convert_safetensor_checkpoint(source, converted, lambda state: state, lambda names: [names])
    torch_assert_equal(weights.load_state_dict(converted)["a"], torch.tensor([1.0]))


@pytest.mark.parametrize("builder_family", ["qwen35", "glm52"])
@pytest.mark.parametrize("scope", ["node-local", "shared"])
@pytest.mark.parametrize("converter_api", ["checkpoint", "stream"])
def test_glm53_load_selects_streaming_checkpoint_converter(
    tmp_path, monkeypatch, builder_family, scope, converter_api
):
    from types import SimpleNamespace

    from safetensors.torch import save_file

    from arctic_platform.model.implementations.glm53.modeling_glm5_next import Glm5NextForConditionalGenerationPrimeRL
    from arctic_platform.model.implementations.moe.weights import load_state_dict

    builder = importlib.import_module(f"arctic_platform.model.implementations.{builder_family}.model_builder")
    model = Glm5NextForConditionalGenerationPrimeRL(_tiny_config())
    state = {
        name: torch.arange(weight.numel(), dtype=weight.dtype).reshape(weight.shape)
        for name, weight in model.state_dict().items()
    }
    source = tmp_path / "source"
    source.mkdir()
    hf = model.convert_to_hf({name: weight.clone() for name, weight in state.items()})
    save_file({name: weight.clone().contiguous() for name, weight in hf.items()}, source / "model.safetensors")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(hf, "model.safetensors")})
    )
    save_file({"stray.weight": torch.ones(1)}, source / "stray.safetensors")
    cache = tmp_path / "prime"
    monkeypatch.setattr(builder, "resolve_conversion_cache_path", lambda *args: cache)
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", scope)
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_ATTEMPT_ID", "streamed-load-test")
    monkeypatch.setattr(builder, "get_world", lambda: SimpleNamespace(rank=0, local_rank=0, is_master=True))
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(builder, "dcp_load", lambda *args, **kwargs: None)
    monkeypatch.setattr(builder, "_move_buffers_to_cuda", lambda *args: None)
    if builder_family == "glm52":
        monkeypatch.setattr(builder, "_init_buffers_post_meta", lambda *args: None)
    monkeypatch.setattr(builder, "HuggingFaceStorageReader", lambda **kwargs: None)

    def refuse_bulk_load(*args):
        raise AssertionError("GLM5-next conversion must not bulk-load the checkpoint")

    monkeypatch.setattr(builder, "load_state_dict", refuse_bulk_load)
    if converter_api == "stream":
        convert_checkpoint = model.convert_to_prime_checkpoint
        monkeypatch.setattr(model, "stream_convert_to_prime", convert_checkpoint, raising=False)
        monkeypatch.setattr(model, "convert_to_prime_checkpoint", refuse_bulk_load)
    config = SimpleNamespace(
        name=str(source), debug=SimpleNamespace(random_init=False), dequantize_checkpoint=False, fsdp_cpu_offload=True
    )
    builder.load_dcp_from_hf(model, config, SimpleNamespace())
    converted = load_state_dict(cache)
    assert set(converted) == set(state)
    for name, weight in state.items():
        torch_assert_equal(converted[name], weight)


@pytest.mark.parametrize("builder_family", ["qwen35", "glm52"])
def test_shared_conversion_cache_failure_stops_both_loaders_before_barrier(tmp_path, monkeypatch, builder_family):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from safetensors.torch import save_file

    from arctic_platform.model.implementations.glm53.modeling_glm5_next import Glm5NextForConditionalGenerationPrimeRL
    from arctic_platform.model.implementations.moe import conversion_cache

    builder = importlib.import_module(f"arctic_platform.model.implementations.{builder_family}.model_builder")
    models = [Glm5NextForConditionalGenerationPrimeRL(_tiny_config()) for _ in range(2)]
    source = tmp_path / "source"
    source.mkdir()
    hf_name = "model.language_model.layers.3.mlp.experts.0.gate_proj.weight"
    save_file({hf_name: torch.ones(8, 16)}, source / "model.safetensors")
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {hf_name: "model.safetensors"}}))
    cache = tmp_path / "prime"
    local = threading.local()
    converting, reader_observed_lock = threading.Event(), threading.Event()
    barriers = {0: 0, 1: 0}
    original_marker = conversion_cache._read_cache_marker
    original_wait = conversion_cache.wait_for_conversion_cache

    def read_marker(path):
        marker = original_marker(path)
        if path.name.endswith(".in-progress") and marker.get("attempt_id") is not None:
            reader_observed_lock.set()
        return marker

    def fail_conversion(source, destination):
        converting.set()
        assert reader_observed_lock.wait(3), "reader must observe the active writer's attempt marker"
        raise OSError("injected shared conversion failure")

    def barrier():
        barriers[local.rank] += 1

    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", "shared")
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_ATTEMPT_ID", "shared-failure-test")
    monkeypatch.setattr(builder, "resolve_conversion_cache_path", lambda *args: cache)
    monkeypatch.setattr(
        builder, "get_world", lambda: SimpleNamespace(rank=local.rank, local_rank=0, is_master=local.rank == 0)
    )
    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    monkeypatch.setattr(conversion_cache, "_read_cache_marker", read_marker)
    monkeypatch.setattr(
        builder,
        "wait_for_conversion_cache",
        lambda path, **kwargs: original_wait(path, timeout=3, poll_interval=0.01, **kwargs),
    )
    monkeypatch.setattr(models[0], "convert_to_prime_checkpoint", fail_conversion)
    monkeypatch.setattr(
        builder, "dcp_load", lambda *args, **kwargs: pytest.fail("failed conversion must not reach DCP")
    )
    config = SimpleNamespace(
        name=str(source), debug=SimpleNamespace(random_init=False), dequantize_checkpoint=False, fsdp_cpu_offload=True
    )

    def load(rank):
        local.rank = rank
        builder.load_dcp_from_hf(models[rank], config, SimpleNamespace())

    with ThreadPoolExecutor(max_workers=2) as executor:
        writer = executor.submit(load, 0)
        assert converting.wait(3)
        reader = executor.submit(load, 1)
        with pytest.raises(OSError, match="injected shared conversion failure"):
            writer.result(timeout=5)
        with pytest.raises(RuntimeError, match="writer failed"):
            reader.result(timeout=5)
    assert barriers == {0: 1, 1: 1}, "both ranks passed only the pre-conversion barrier"
    assert not cache.exists()
