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

import json

import pytest
import torch

from arctic_platform.model import ModelSpec
from arctic_platform.model import ParallelismConfig
from arctic_platform.model.implementations.qwen38.converting_qwen4_exp import convert_hf_to_prime
from arctic_platform.model.implementations.qwen38.converting_qwen4_exp import convert_prime_to_hf


def _require():
    pytest.importorskip("transformers.models.qwen4_exp")
    global Qwen4ExpConfig, Qwen4ExpTextConfig, Qwen4ExpTextSparseMoeBlock
    global EPShardedEmbedding, Qwen4ExpForConditionalGenerationPrimeRL, Qwen4ExpSparseMoePrimeRL
    global get_custom_vlm_cls, get_model, ModelConfig
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextSparseMoeBlock

    from arctic_platform.model.implementations.qwen35.config import ModelConfig
    from arctic_platform.model.implementations.qwen35.model_builder import get_model
    from arctic_platform.model.implementations.qwen35.models import get_custom_vlm_cls
    from arctic_platform.model.implementations.qwen38.modeling_qwen4_exp import EPShardedEmbedding
    from arctic_platform.model.implementations.qwen38.modeling_qwen4_exp import Qwen4ExpForConditionalGenerationPrimeRL
    from arctic_platform.model.implementations.qwen38.modeling_qwen4_exp import Qwen4ExpSparseMoePrimeRL


def _tiny_config():
    _require()
    text_config = {
        "vocab_size": 32,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_conv_kernel_dim": 2,
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "hc_count": 2,
        "hc_lowrank": 4,
        "ple_layer_ids": [1],
        "ple_embed_dim": 8,
        "ple_conv_kernel_size": 2,
        "ngram_size": 3,
        "heads_per_ngram": 1,
        "ngram_vocab_size_base": 11,
        "make_ngram_vocab_size_divisible_by": 4,
        "split_ngram_parts": 2,
        "indexer_n_heads": 1,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 4,
        "indexer_budget": 4,
        "indexer_compress_ratio": 2,
        "output_gate_type": "sigmoid",
        "eos_token_id": 1,
        "pad_token_id": 0,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 0],
            "mrope_interleaved": True,
        },
    }
    vision_config = {
        "depth": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_heads": 2,
        "out_hidden_size": 16,
        "patch_size": 2,
        "spatial_merge_size": 1,
        "temporal_patch_size": 1,
        "num_position_embeddings": 16,
    }
    config = Qwen4ExpConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=29,
        video_token_id=30,
        vision_start_token_id=27,
        vision_end_token_id=28,
    )
    config.use_grouped_mm = False
    return config


def _checkpoint(tmp_path, model_type: str) -> str:
    path = tmp_path / model_type
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(path)


def test_qwen38_family_dispatch_and_custom_vlm_registration(tmp_path):
    qwen = ModelSpec(
        model_path_or_name=_checkpoint(tmp_path, "qwen4_exp"),
        parallelism=ParallelismConfig(expert_parallel=2),
    )
    qwen35 = ModelSpec(
        model_path_or_name=_checkpoint(tmp_path, "qwen3_5_moe"),
        parallelism=ParallelismConfig(expert_parallel=2),
    )
    from arctic_platform.model.implementations.qwen38.deepspeed_integration import _adapter

    assert qwen.loader == "qwen4_exp"
    assert qwen35.loader == "qwen3_5_moe"
    assert _adapter().extra_weight_iterators == ()


def test_qwen38_custom_vlm_registration():
    config = _tiny_config()
    assert get_custom_vlm_cls(config) is Qwen4ExpForConditionalGenerationPrimeRL


def test_qwen38_replaces_moe_and_shards_ple_embedding():
    _require()
    from arctic_platform.model.implementations.moe.vlm import get_language_model

    model = Qwen4ExpForConditionalGenerationPrimeRL(_tiny_config())
    language_model = get_language_model(model)

    assert model._requires_hf_weight_sync
    assert all(isinstance(layer.mlp, Qwen4ExpSparseMoePrimeRL) for layer in language_model.layers)
    embedding = language_model.layers[0].ple.ple_embedding.ngram_embedding
    assert isinstance(embedding, EPShardedEmbedding)
    assert embedding._dss_shard_on_ep
    assert not embedding.weight.requires_grad
    assert embedding.weight._dss_skip_weight_sync


def test_qwen38_model_builder_selects_custom_vlm(tmp_path):
    _require()
    _tiny_config().save_pretrained(tmp_path)
    model = get_model(
        ModelConfig(
            name=str(tmp_path),
            impl="custom",
            attn="sdpa",
            moe_use_grouped_mm=False,
        ),
        device=torch.device("meta"),
        dtype=torch.float32,
    )

    assert isinstance(model, Qwen4ExpForConditionalGenerationPrimeRL)
    assert model.model.language_model.layers[0].mlp.experts.w1.device.type == "meta"


def test_qwen38_conversion_round_trip_and_ple_concatenation():
    prefix = "model.language_model.layers.0"
    original = {
        f"{prefix}.mlp.experts.gate_up_proj": torch.randn(4, 16, 16),
        f"{prefix}.mlp.experts.down_proj": torch.randn(4, 16, 8),
        f"{prefix}.mlp.gate.weight": torch.randn(4, 16),
        f"{prefix}.mlp.shared_expert.gate_proj.weight": torch.randn(8, 16),
        f"{prefix}.mlp.shared_expert.down_proj.weight": torch.randn(16, 8),
        f"{prefix}.mlp.shared_expert.up_proj.weight": torch.randn(8, 16),
        f"{prefix}.mlp.shared_expert_gate.weight": torch.randn(1, 16),
        f"{prefix}.ple.ple_embedding.ngram_embedding.shard_0.weight": torch.randn(3, 4),
        f"{prefix}.ple.ple_embedding.ngram_embedding.shard_1.weight": torch.randn(2, 4),
    }
    expected_ple = torch.cat(
        [
            original[f"{prefix}.ple.ple_embedding.ngram_embedding.shard_0.weight"],
            original[f"{prefix}.ple.ple_embedding.ngram_embedding.shard_1.weight"],
        ]
    )
    state = {name: tensor.clone() for name, tensor in original.items()}

    convert_hf_to_prime(state)
    assert f"{prefix}.mlp.experts.w1" in state
    assert f"{prefix}.mlp.qwen_shared_expert.w1.weight" in state
    torch.testing.assert_close(
        state[f"{prefix}.ple.ple_embedding.ngram_embedding.weight"],
        expected_ple,
    )

    convert_prime_to_hf(state)
    for name, tensor in original.items():
        if ".ngram_embedding.shard_" not in name:
            torch.testing.assert_close(state[name], tensor)
    torch.testing.assert_close(
        state[f"{prefix}.ple.ple_embedding.ngram_embedding.weight"],
        expected_ple,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DSS MoE kernels require CUDA",
)
def test_qwen38_moe_matches_transformers():
    _require()
    config = Qwen4ExpTextConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["linear_attention"],
        hc_count=2,
        hc_lowrank=4,
        eos_token_id=1,
        pad_token_id=0,
    )
    hf_moe = Qwen4ExpTextSparseMoeBlock(config)
    prime_moe = Qwen4ExpSparseMoePrimeRL(config, use_grouped_mm=False)
    with torch.no_grad():
        for parameter in hf_moe.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    state = {
        f"model.language_model.layers.0.mlp.{name}": tensor.clone() for name, tensor in hf_moe.state_dict().items()
    }
    convert_hf_to_prime(state)
    prime_moe.load_state_dict({name.split(".mlp.", 1)[1]: tensor for name, tensor in state.items()})
    prime_moe.ep_comm_backend = "local"
    prime_moe.experts.forward = prime_moe.experts._forward_deepep

    hidden_states = torch.randn(2, 3, 16, device="cuda")
    hf_moe.cuda().eval()
    prime_moe.cuda().eval()
    torch.testing.assert_close(
        prime_moe(hidden_states),
        hf_moe(hidden_states),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Qwen3.8 hybrid attention requires CUDA",
)
def test_qwen38_full_forward_backward():
    _require()
    from arctic_platform.model.implementations.moe.layers.lm_head import inject_prime_lm_head

    model = Qwen4ExpForConditionalGenerationPrimeRL(_tiny_config())
    for layer in model.model.language_model.layers:
        layer.mlp.ep_comm_backend = "local"
        layer.mlp.experts.forward = layer.mlp.experts._forward_deepep
    inject_prime_lm_head(model, fused_cross_entropy=False)
    model.cuda().train()

    input_ids = torch.tensor([[2, 3, 4, 5]], device="cuda")
    output = model(input_ids=input_ids, labels=input_ids)
    assert output["logits"].shape == (1, 4, 32)
    output["logits"].sum().backward()

    ple_weight = model.model.language_model.layers[0].ple.ple_embedding.ngram_embedding.weight
    assert ple_weight.grad is None
    assert model.model.language_model.layers[0].mlp.experts.w1.grad is not None


def test_qwen38_initializes_meta_buffers():
    _require()
    with torch.device("meta"):
        model = Qwen4ExpForConditionalGenerationPrimeRL(_tiny_config())
    model.to_empty(device="cpu")
    model.init_buffers_post_meta()

    rotary = model.model.language_model.rotary_emb
    assert torch.isfinite(rotary.inv_freq).all()
    torch.testing.assert_close(rotary.inv_freq, rotary.original_inv_freq)
    ple = model.model.language_model.layers[0].ple.ple_embedding
    assert torch.equal(
        ple.ngram_heads_offsets,
        torch.tensor(ple.head_offsets),
    )


def test_qwen38_ep_embedding_masks_nonlocal_rows(monkeypatch):
    _require()
    embedding = EPShardedEmbedding(7, 2)
    embedding.weight = torch.nn.Parameter(torch.tensor([[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]]))
    embedding._ep_rank = 1
    embedding._ep_world_size = 2
    embedding._ep_group = object()
    monkeypatch.setattr(
        "arctic_platform.model.implementations.qwen38.modeling_qwen4_exp.dist_nn.all_reduce",
        lambda tensor, group: tensor,
    )

    output = embedding(torch.tensor([[0, 4, 6]]))
    torch.testing.assert_close(
        output,
        torch.tensor([[[0.0, 0.0], [4.0, 40.0], [6.0, 60.0]]]),
    )


def test_qwen38_hf_sync_excludes_frozen_ple_table(monkeypatch):
    _require()
    from arctic_platform.model.implementations.moe.deepspeed_integration import build_iter_full_hf_weights

    model = Qwen4ExpForConditionalGenerationPrimeRL(_tiny_config())
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    embedding = model.model.language_model.layers[0].ple.ple_embedding.ngram_embedding
    del embedding.weight._dss_skip_weight_sync

    names = {name for name, _ in build_iter_full_hf_weights(model)()}
    assert not any("ngram_embedding.weight" in name for name in names)
    assert "model.language_model.layers.0.mlp.experts.gate_up_proj" in names


def test_qwen38_adapter_uses_sdpa_and_rejects_sequence_parallelism():
    from arctic_platform.model.implementations.qwen38.deepspeed_integration import _build_model_config
    from arctic_platform.model.implementations.qwen38.deepspeed_integration import _validate_parallelism
    from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions

    options = Qwen3_5MoeOptions()
    model_config = _build_model_config(
        "Qwen/Qwen3.8-Flash-Next",
        8,
        1,
        "bfloat16",
        "flash_attention_3",
        options,
    )
    assert model_config.attn == "sdpa"
    with pytest.raises(ValueError, match="must divide 512"):
        _build_model_config(
            "Qwen/Qwen3.8-Flash-Next",
            3,
            1,
            "bfloat16",
            "sdpa",
            options,
        )
    _validate_parallelism(1)
    with pytest.raises(NotImplementedError, match="Sequence parallelism"):
        _validate_parallelism(2)


def test_qwen38_rejects_native_fp8_training():
    _require()
    config = _tiny_config()
    config.quantization_config = {"quant_method": "fp8"}
    with pytest.raises(NotImplementedError, match="Native-FP8"):
        Qwen4ExpForConditionalGenerationPrimeRL(config)
