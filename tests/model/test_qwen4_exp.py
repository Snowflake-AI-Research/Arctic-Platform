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
    assert _adapter().vllm_weight_export is None


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


def test_qwen38_gated_delta_net_output_norms_round_once():
    _require()
    from arctic_platform.model.implementations.gpu import vllm_rounding
    from arctic_platform.model.implementations.moe.vlm import get_language_model

    language_model = get_language_model(Qwen4ExpForConditionalGenerationPrimeRL(_tiny_config()))
    norms = [layer.linear_attn.norm for layer in language_model.layers if layer.layer_type == "linear_attention"]

    # The rounding of each stock class is checked in test_gated_norm_rounding.py; here, that every norm uses it.
    assert len(norms) == 1
    for norm in norms:
        assert vars(norm)["forward"].__func__ is vllm_rounding._rms_norm_gated_once_forward
        assert norm.activation == "sigmoid"


def test_qwen38_decoder_forwards_its_physical_layer_routes():
    _require()
    from arctic_platform.model.implementations.moe.vlm import get_language_model

    class LinearAttention(torch.nn.Module):
        def forward(self, hidden_states, **_kwargs):
            return torch.zeros_like(hidden_states)

    class RouteSpy(torch.nn.Module):
        def forward(self, hidden_states, routed_experts=None):
            self.routed_experts = routed_experts
            return torch.zeros_like(hidden_states)

    model = Qwen4ExpForConditionalGenerationPrimeRL(_tiny_config())
    layer = get_language_model(model).layers[0]
    layer.ple = None
    layer.linear_attn = LinearAttention()
    layer.mlp = RouteSpy()
    routes = torch.arange(8).reshape(1, 2, 2, 2)

    layer(
        torch.randn(1, 2, 32),
        position_embeddings=(torch.ones(1), torch.zeros(1)),
        routed_experts=routes,
    )

    torch.testing.assert_close(layer.mlp.routed_experts, routes[:, :, 0, :])


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


def _indexed_expert_weights(prefix, expert_ids):
    shapes = {"gate_proj": (3, 4), "up_proj": (3, 4), "down_proj": (4, 3)}
    return {
        f"{prefix}.mlp.experts.{expert_idx}.{projection}.weight": (
            torch.arange(shape[0] * shape[1], dtype=torch.bfloat16).reshape(shape) + expert_idx * 16 + projection_idx
        )
        for expert_idx in reversed(expert_ids)
        for projection_idx, (projection, shape) in enumerate(shapes.items())
    }


def test_qwen38_conversion_stacks_indexed_experts_in_numeric_order():
    prefix = "model.language_model.layers.0"
    original = _indexed_expert_weights(prefix, range(12))
    other_layer = _indexed_expert_weights("model.language_model.layers.1", range(2))
    state = {**original, **other_layer}
    state[f"{prefix}.mlp.gate.weight"] = torch.arange(48).reshape(12, 4)

    from arctic_platform.model.implementations.qwen38.converting_qwen4_exp import convert_hf_layer_to_prime

    convert_hf_layer_to_prime(state, 0)
    for prime_projection, hf_projection in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
        expected = torch.stack(
            [original[f"{prefix}.mlp.experts.{expert_idx}.{hf_projection}.weight"] for expert_idx in range(12)]
        )
        torch.testing.assert_close(state[f"{prefix}.mlp.experts.{prime_projection}"], expected, rtol=0, atol=0)
    assert not set(original).intersection(state)
    assert f"{prefix}.mlp.router.gate.weight" in state
    for name, value in other_layer.items():
        assert state[name] is value

    convert_hf_to_prime(state)
    assert "model.language_model.layers.1.mlp.experts.w1" in state
    convert_prime_to_hf(state)
    torch.testing.assert_close(
        state[f"{prefix}.mlp.experts.gate_up_proj"],
        torch.cat(
            [
                torch.stack([original[f"{prefix}.mlp.experts.{idx}.gate_proj.weight"] for idx in range(12)]),
                torch.stack([original[f"{prefix}.mlp.experts.{idx}.up_proj.weight"] for idx in range(12)]),
            ],
            dim=1,
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        state[f"{prefix}.mlp.experts.down_proj"],
        torch.stack([original[f"{prefix}.mlp.experts.{idx}.down_proj.weight"] for idx in range(12)]),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("expert_ids", [[1], [0, 2], [0, 1, 3]])
def test_qwen38_conversion_rejects_noncontiguous_indexed_experts(expert_ids):
    state = _indexed_expert_weights("model.language_model.layers.0", expert_ids)
    original = state.copy()
    with pytest.raises(KeyError, match="Non-contiguous Qwen3.8 experts"):
        convert_hf_to_prime(state)
    assert set(state) == set(original)
    assert all(state[name] is value for name, value in original.items())


@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
def test_qwen38_conversion_rejects_incomplete_indexed_experts(projection):
    prefix = "model.language_model.layers.0"
    state = _indexed_expert_weights(prefix, range(2))
    del state[f"{prefix}.mlp.experts.1.{projection}.weight"]
    original = state.copy()
    with pytest.raises(KeyError, match="Incomplete Qwen3.8 expert"):
        convert_hf_to_prime(state)
    assert set(state) == set(original)
    assert all(state[name] is value for name, value in original.items())


def test_qwen38_conversion_validates_indexed_layers_before_mutating_checkpoint():
    prefix = "model.language_model.layers.0"
    state = _indexed_expert_weights(prefix, range(2))
    state.update(_indexed_expert_weights("model.language_model.layers.1", [0, 2]))
    state[f"{prefix}.ple.ple_embedding.ngram_embedding.shard_0.weight"] = torch.ones(2, 4)
    state[f"{prefix}.ple.ple_embedding.ngram_embedding.shard_1.weight"] = torch.ones(2, 4)
    original = state.copy()
    with pytest.raises(KeyError, match="Non-contiguous Qwen3.8 experts"):
        convert_hf_to_prime(state)
    assert set(state) == set(original)
    assert all(state[name] is value for name, value in original.items())


@pytest.mark.parametrize("packed_projection", ["gate_up_proj", "down_proj", "w1", "w2", "w3"])
def test_qwen38_conversion_rejects_mixed_expert_layouts(packed_projection):
    prefix = "model.language_model.layers.0"
    state = _indexed_expert_weights(prefix, range(2))
    state[f"{prefix}.mlp.experts.{packed_projection}"] = torch.ones(2, 3, 4)
    original = state.copy()
    with pytest.raises(ValueError, match="Mixed Qwen3.8 expert layouts"):
        convert_hf_to_prime(state)
    assert set(state) == set(original)
    assert all(state[name] is value for name, value in original.items())


def test_qwen38_checkpoint_conversion_matches_bulk_across_source_shards(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    from arctic_platform.model.implementations.moe import weights
    from arctic_platform.model.implementations.qwen38.converting_qwen4_exp import convert_hf_checkpoint_to_prime

    source = tmp_path / "source"
    source.mkdir()
    original = {
        **_indexed_expert_weights("model.language_model.layers.0", range(3)),
        **_indexed_expert_weights("model.language_model.layers.1", range(2)),
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight": (
            torch.arange(12).reshape(3, 4)
        ),
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_1.weight": (
            torch.arange(8).reshape(2, 4)
        ),
        "model.language_model.embed_tokens.weight": torch.arange(12).reshape(3, 4),
        "lm_head.weight": torch.arange(16).reshape(4, 4),
    }
    names = list(original)
    for shard in range(3):
        save_file({name: original[name] for name in names[shard::3]}, source / f"shard-{shard}.safetensors")
    expected = {name: tensor.clone() for name, tensor in original.items()}
    convert_hf_to_prime(expected)
    destination = tmp_path / "prime"
    saved_sizes = []
    ordinary_save = weights.save_file

    def save_group(state, path, **kwargs):
        assert not (destination / "model.safetensors.index.json").exists()
        saved_sizes.append(sum(value.numel() * value.element_size() for value in state.values()))
        return ordinary_save(state, path, **kwargs)

    monkeypatch.setattr(weights, "save_file", save_group)
    convert_hf_checkpoint_to_prime(source, destination)
    actual = weights.load_state_dict(destination)
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in expected.values())
    assert len(saved_sizes) == 4
    assert max(saved_sizes) < index["metadata"]["total_size"]
    assert weights.load_state_dict(source).keys() == original.keys()


@pytest.mark.parametrize("invalid_layout", ["expert-gap", "expert-missing", "expert-mixed", "ple-gap"])
def test_qwen38_checkpoint_conversion_rejects_incomplete_groups_before_writing(tmp_path, invalid_layout):
    from safetensors.torch import save_file

    from arctic_platform.model.implementations.qwen38.converting_qwen4_exp import convert_hf_checkpoint_to_prime

    prefix = "model.language_model.layers.0"
    state = _indexed_expert_weights(prefix, [0, 2] if invalid_layout == "expert-gap" else range(2))
    if invalid_layout == "expert-missing":
        del state[f"{prefix}.mlp.experts.1.down_proj.weight"]
    elif invalid_layout == "expert-mixed":
        state[f"{prefix}.mlp.experts.w1"] = torch.ones(2, 3, 4)
    elif invalid_layout == "ple-gap":
        state[f"{prefix}.ple.ple_embedding.ngram_embedding.shard_1.weight"] = torch.ones(2, 4)
    source = tmp_path / "source"
    source.mkdir()
    save_file(state, source / "model.safetensors")
    destination = tmp_path / "prime"
    with pytest.raises((KeyError, ValueError)):
        convert_hf_checkpoint_to_prime(source, destination)
    assert not destination.exists()


def test_qwen38_checkpoint_conversion_failure_never_publishes_index(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    from arctic_platform.model.implementations.moe import weights
    from arctic_platform.model.implementations.qwen38.converting_qwen4_exp import convert_hf_checkpoint_to_prime

    source = tmp_path / "source"
    source.mkdir()
    save_file({"first": torch.ones(2), "second": torch.ones(3)}, source / "model.safetensors")
    destination = tmp_path / "prime"
    calls = 0
    ordinary_save = weights.save_file

    def fail_second_group(state, path, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected shard write failure")
        return ordinary_save(state, path, **kwargs)

    monkeypatch.setattr(weights, "save_file", fail_second_group)
    with pytest.raises(OSError, match="injected shard write"):
        convert_hf_checkpoint_to_prime(source, destination)
    assert calls == 2
    assert not (destination / "model.safetensors.index.json").exists()


@pytest.mark.parametrize("include_skipped", [False, True])
def test_qwen38_full_hf_export_restores_ple_shards(monkeypatch, include_skipped):
    from types import SimpleNamespace

    from arctic_platform.model.implementations.qwen38 import deepspeed_integration as integration

    model = torch.nn.Module()
    model.config = SimpleNamespace(text_config=SimpleNamespace(split_ngram_parts=2))
    name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight"
    tensor = torch.arange(20).reshape(5, 4)

    def ordinary_builder(built_model):
        assert built_model is model

        def iterator(*, include_skipped=False):
            yield name, tensor
            if include_skipped:
                yield "mtp.fc_embedding.weight", torch.ones(2, 2)

        return iterator

    monkeypatch.setattr(integration.qwen_ds, "_build_iter_full_hf_weights_with_mtp", ordinary_builder)
    exported = dict(integration._build_iter_full_hf_weights(model)(include_skipped=include_skipped))
    if not include_skipped:
        assert set(exported) == {name}
        assert exported[name] is tensor
        return
    assert set(exported) == {
        name.replace(".weight", ".shard_0.weight"),
        name.replace(".weight", ".shard_1.weight"),
        "mtp.fc_embedding.weight",
    }
    torch.testing.assert_close(exported[name.replace(".weight", ".shard_0.weight")], tensor[:3], rtol=0, atol=0)
    torch.testing.assert_close(exported[name.replace(".weight", ".shard_1.weight")], tensor[3:], rtol=0, atol=0)


def test_qwen38_full_hf_export_uses_family_export_builder():
    from arctic_platform.model.implementations.qwen38 import deepspeed_integration as integration

    assert integration._adapter().hf_weight_export is integration._build_iter_full_hf_weights


@pytest.mark.parametrize("parts", [0, -1, True, 1.5, "128"])
def test_qwen38_full_hf_export_rejects_invalid_ple_part_count(monkeypatch, parts):
    from types import SimpleNamespace

    from arctic_platform.model.implementations.qwen38 import deepspeed_integration as integration

    model = torch.nn.Module()
    model.config = SimpleNamespace(text_config=SimpleNamespace(split_ngram_parts=parts))
    monkeypatch.setattr(integration.qwen_ds, "_build_iter_full_hf_weights_with_mtp", lambda model: None)
    with pytest.raises(ValueError, match="split_ngram_parts must be a positive integer"):
        integration._build_iter_full_hf_weights(model)


def test_qwen38_preserves_original_mtp_payload(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from safetensors.torch import save_file

    from arctic_platform.model.implementations.qwen35 import mtp_weights

    names = (
        "fc_embedding.weight",
        "fc_hidden.weight",
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
        "hyper_connection_mixer.hc_norm.weight",
        "hyper_connection_mixer.input_mix_weight_down.weight",
        "hyper_connection_mixer.input_mix_weight_up.weight",
    )
    layer_names = (
        "attn_hyper_connection.block_inject_weight.weight",
        "attn_hyper_connection.hc_norm.weight",
        "attn_hyper_connection.input_mix_weight_down.weight",
        "attn_hyper_connection.input_mix_weight_up.weight",
        "mlp_hyper_connection.block_inject_weight.weight",
        "mlp_hyper_connection.hc_norm.weight",
        "mlp_hyper_connection.input_mix_weight_down.weight",
        "mlp_hyper_connection.input_mix_weight_up.weight",
        "self_attn.indexer.index_qk_proj.weight",
        "self_attn.indexer.k_layernorm.weight",
        "self_attn.indexer.q_layernorm.weight",
        "self_attn.k_norm.weight",
        "self_attn.k_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.q_proj.weight",
        "self_attn.v_proj.weight",
        "mlp.gate.weight",
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_expert_gate.weight",
    )
    payload = {
        name: torch.full((2, 3), float(index), dtype=torch.bfloat16)
        for index, name in enumerate(
            [f"mtp.{name}" for name in names] + [f"mtp.layers.0.{name}" for name in layer_names]
        )
    }
    save_file(payload, tmp_path / "model.safetensors")
    model = torch.nn.Module()
    model.config = SimpleNamespace(text_config=SimpleNamespace(model_type="qwen4_exp_text", mtp_num_hidden_layers=1))
    monkeypatch.setattr(mtp_weights.dist, "is_initialized", lambda: False)
    mtp_weights.capture_mtp_weights(model, tmp_path)
    exported = dict(mtp_weights.iter_preserved_mtp_weights(model))
    assert len(exported) == 31
    assert set(exported) == set(payload)
    assert not list(model.parameters()) and not list(model.buffers())
    for name, value in payload.items():
        torch.testing.assert_close(exported[name], value, rtol=0, atol=0)


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
    from arctic_platform.model.implementations.qwen38.qsa_flex import apply_qsa_flex
    from arctic_platform.model.implementations.qwen38.qsa_flex import register_qsa_flex_backend

    config = _tiny_config()
    config.text_config.head_dim = 16
    config.text_config.indexer_head_dim = 8
    config.use_cache = False
    config.text_config.use_cache = False
    register_qsa_flex_backend()
    config._attn_implementation = "qsa_flex"
    config.text_config._attn_implementation = "qsa_flex"
    model = Qwen4ExpForConditionalGenerationPrimeRL(config)
    apply_qsa_flex(model)
    for layer in model.model.language_model.layers:
        layer.mlp.ep_comm_backend = "local"
        layer.mlp.experts.forward = layer.mlp.experts._forward_deepep
    inject_prime_lm_head(model, fused_cross_entropy=False)
    model.bfloat16().cuda().train()

    input_ids = torch.tensor([[2, 3, 4, 5]], device="cuda")
    output = model(input_ids=input_ids, labels=input_ids)
    assert output["logits"].shape == (1, 4, 32)
    output["logits"].sum().backward()

    ple_weight = model.model.language_model.layers[0].ple.ple_embedding.ngram_embedding.weight
    assert ple_weight.grad is None
    assert model.model.language_model.layers[0].mlp.experts.w1.grad is not None
    attention = model.model.language_model.layers[1].self_attn
    assert attention.q_proj.weight.grad is not None
    assert attention.indexer.index_qk_proj.weight.grad is None


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

    def all_to_all_single(output, tensor, **_kwargs):
        if tensor.numel() == 2:
            output.copy_(torch.tensor([0, 2]))
        else:
            output.copy_(tensor[1:])

    monkeypatch.setattr(
        "arctic_platform.model.implementations.qwen38.modeling_qwen4_exp.dist.all_to_all_single",
        all_to_all_single,
    )

    def differentiable_all_to_all(output, tensor, **_kwargs):
        output.zero_()
        output[1:].copy_(tensor)
        return output

    monkeypatch.setattr(
        "arctic_platform.model.implementations.qwen38.modeling_qwen4_exp.dist_nn.all_to_all_single",
        differentiable_all_to_all,
    )

    output = embedding(torch.tensor([[0, 4, 6]]))
    torch.testing.assert_close(
        output,
        torch.tensor([[[0.0, 0.0], [4.0, 40.0], [6.0, 60.0]]]),
    )


def test_qwen38_hf_checkpoint_includes_frozen_ple_table(monkeypatch):
    _require()
    from arctic_platform.model.implementations.moe.deepspeed_integration import build_iter_full_hf_weights

    model = Qwen4ExpForConditionalGenerationPrimeRL(_tiny_config())
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    ngram_name = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.weight"

    iterator = build_iter_full_hf_weights(model)
    sync_names = {name for name, _ in iterator()}
    checkpoint_names = {name for name, _ in iterator(include_skipped=True)}

    assert ngram_name not in sync_names
    assert ngram_name in checkpoint_names
    assert "model.language_model.layers.0.mlp.experts.gate_up_proj" in sync_names


def test_qwen38_adapter_uses_qsa_flex_and_validates_context_parallelism():
    from arctic_platform.model.implementations.qwen38.deepspeed_integration import _build_model_config
    from arctic_platform.model.implementations.qwen38.deepspeed_integration import _validate_parallelism
    from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions

    options = Qwen3_5MoeOptions()
    model_config = _build_model_config(
        "Qwen/Qwen3.8-Flash-Next",
        8,
        1,
        "bfloat16",
        "flex_attention",
        options,
    )
    assert model_config.attn == "qsa_flex"
    with pytest.raises(ValueError, match="non-SDPA"):
        _build_model_config(
            "Qwen/Qwen3.8-Flash-Next",
            8,
            1,
            "bfloat16",
            "sdpa",
            options,
        )
    with pytest.raises(ValueError, match="must divide 512"):
        _build_model_config(
            "Qwen/Qwen3.8-Flash-Next",
            3,
            1,
            "bfloat16",
            "qsa_flex",
            options,
        )
    _validate_parallelism(1)
    with pytest.raises(ValueError, match="process group"):
        _validate_parallelism(2)
    _validate_parallelism(8, object())


def test_qwen38_rank_zero_ple_halo_keeps_collective_backward(monkeypatch):
    from arctic_platform.model.implementations.qwen38 import context_parallel

    source = torch.arange(8.0, requires_grad=True)
    hidden_states = source.view(1, 4, 2)
    monkeypatch.setattr(
        context_parallel.dist_nn,
        "all_gather",
        lambda tensor, group: (tensor * 2, tensor * 3),
    )
    monkeypatch.setattr(context_parallel.dist, "get_rank", lambda group: 0)

    previous = context_parallel._previous_rank_context(
        hidden_states,
        2,
        process_group=object(),
        pad_value=0,
        differentiable=True,
    )
    previous.sum().backward()

    torch.testing.assert_close(previous, torch.zeros_like(previous))
    torch.testing.assert_close(source.grad, torch.zeros_like(source))


def test_qwen38_packed_ple_components_reset_at_position_boundaries():
    _require()
    from arctic_platform.model.implementations.qwen38.modeling_qwen4_exp import Qwen4ExpTextNGramEmbeddingPrimeRL
    from arctic_platform.model.implementations.qwen38.modeling_qwen4_exp import Qwen4ExpTextPLELayerPrimeRL

    config = _tiny_config().text_config
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    first = torch.tensor([[2, 3, 4, 5]])
    second = torch.tensor([[6, 7, 8, 9]])

    ngram = Qwen4ExpTextNGramEmbeddingPrimeRL(
        config,
        config.ple_embed_dim,
        layer_idx=0,
    )
    packed_ngram = ngram(
        torch.cat([first, second], dim=1),
        None,
        position_ids=positions,
    )[:, 4:]
    separate_ngram = ngram(second, None)
    torch.testing.assert_close(packed_ngram, separate_ngram)

    ple = Qwen4ExpTextPLELayerPrimeRL(config, layer_idx=0, ple_layer_index=0)
    with torch.no_grad():
        ple.conv1d.weight.fill_(1)
    first_hidden = torch.ones(1, 4, config.hidden_size * config.hc_count)
    second_hidden = torch.zeros_like(first_hidden)
    packed_conv = ple._short_conv(
        torch.cat([first_hidden, second_hidden], dim=1),
        None,
        position_ids=positions,
    )[:, 4:]
    separate_conv = ple._short_conv(second_hidden, None)
    torch.testing.assert_close(packed_conv, separate_conv)


def test_qwen38_context_parallel_ple_transports_positions(monkeypatch):
    from arctic_platform.model.implementations.qwen38 import context_parallel

    class Embedding(torch.nn.Module):
        context_len = 2
        eos_token_id = 0

        def forward(self, input_ids, past_key_values, position_ids=None):
            self.received_positions = position_ids
            return input_ids.unsqueeze(-1).float()

    class PLE(torch.nn.Module):
        short_conv_state_len = 2

        def __init__(self):
            super().__init__()
            self.ple_embedding = Embedding()

        def _short_conv(self, hidden_states, past_key_values, position_ids=None):
            self.received_positions = position_ids
            return hidden_states

    calls = []

    def previous(tensor, width, *, process_group, pad_value, differentiable):
        calls.append((width, pad_value, differentiable))
        return torch.full_like(tensor[:, :width], pad_value)

    monkeypatch.setattr(context_parallel, "_previous_rank_context", previous)
    ple = PLE()
    context_parallel._adapt_ple_layer(ple, process_group=object())

    positions = torch.tensor([[0, 1, 2]])
    ple.ple_embedding(
        torch.tensor([[3, 4, 5]]),
        None,
        position_ids=positions,
    )
    ple._short_conv(
        torch.randn(1, 3, 4),
        None,
        position_ids=positions,
    )

    expected_positions = torch.tensor([[-1, -1, 0, 1, 2]])
    torch.testing.assert_close(
        ple.ple_embedding.received_positions,
        expected_positions,
    )
    torch.testing.assert_close(ple.received_positions, expected_positions)
    assert calls == [
        (2, 0, False),
        (2, -1, False),
        (2, 0, True),
        (2, -1, False),
    ]


def test_qwen38_supports_selective_activation_checkpointing():
    _require()
    from arctic_platform.model.implementations.moe.layers.checkpointing import get_supported_targets
    from arctic_platform.model.implementations.moe.layers.checkpointing import (
        supports_selective_activation_checkpointing,
    )

    model = Qwen4ExpForConditionalGenerationPrimeRL(_tiny_config())
    linear_layer = model.model.language_model.layers[0]

    assert supports_selective_activation_checkpointing(linear_layer)
    assert {"linear_attn", "routed_experts"} <= get_supported_targets(linear_layer)


def test_qwen38_qsa_route_selection_is_causal():
    from arctic_platform.model.implementations.qwen38.qsa_flex import select_qsa_token_ids

    queries = torch.ones(1, 12, 1, 2)
    keys = torch.ones(1, 6, 1, 2)
    routes = select_qsa_token_ids(
        queries,
        keys,
        torch.tensor([12]),
        token_budget=4,
        compress_ratio=2,
    )

    for position, selected in enumerate(routes[0]):
        valid = selected[selected >= 0]
        if position < 5:
            assert valid.numel() == position + 1
        else:
            assert 4 <= valid.numel() <= 5
        assert torch.all(valid <= position)


def test_qwen38_qsa_route_selection_uses_global_query_offset():
    from arctic_platform.model.implementations.qwen38.qsa_flex import select_qsa_token_ids

    queries = torch.ones(1, 4, 1, 2)
    keys = torch.ones(1, 6, 1, 2)
    routes = select_qsa_token_ids(
        queries,
        keys,
        torch.tensor([12]),
        token_budget=4,
        compress_ratio=2,
        query_offset=8,
    )

    for local_position, selected in enumerate(routes[0]):
        valid = selected[selected >= 0]
        assert torch.all(valid <= local_position + 8)
        assert torch.any(valid > local_position)


def test_qwen38_qsa_compresses_after_gather_for_unaligned_cp_shards(monkeypatch):
    _require()
    from arctic_platform.model.implementations.qwen38 import qsa_flex

    class Indexer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.index_n_heads = 1
            self.index_head_dim = 2
            self.compress_ratio = 2
            self.token_budget = 2
            self.index_qk_proj = torch.nn.Linear(4, 4, bias=False)
            self.q_layernorm = torch.nn.Identity()
            self.k_layernorm = torch.nn.Identity()
            self._cp_group = object()

    indexer = Indexer()
    monkeypatch.setattr(qsa_flex.dist, "get_rank", lambda group: 1)
    monkeypatch.setattr(qsa_flex.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        qsa_flex,
        "_gather_sequence_no_grad",
        lambda tensor, group: torch.cat([tensor, tensor], dim=1),
    )

    def capture_keys(index_queries, compressed_keys, sequence_lengths, **kwargs):
        assert compressed_keys.shape[1] == 3
        return torch.zeros((*index_queries.shape[:2], 3), dtype=torch.int32)

    monkeypatch.setattr(qsa_flex, "select_qsa_token_ids", capture_keys)
    hidden_states = torch.ones(1, 3, 4)
    position_embeddings = (torch.ones(1, 3, 2), torch.zeros(1, 3, 2))

    routes = qsa_flex._qsa_indexer_forward(
        indexer,
        hidden_states,
        position_embeddings,
        torch.ones(1, 3, dtype=torch.bool),
        position_ids=torch.tensor([[3, 4, 5]]),
        cu_seqlens=torch.tensor([0, 6]),
    )

    assert routes.shape == (1, 3, 3)


@pytest.mark.parametrize("fp32_parity", [True, False])
def test_qwen38_sparse_gqa_custom_backward_matches_autograd(fp32_parity):
    from arctic_platform.model.implementations.qwen38.qsa_flex import _sparse_gqa_chunk
    from arctic_platform.model.implementations.qwen38.qsa_flex import _SparseGQAAttention

    torch.manual_seed(7)
    query = torch.randn(1, 3, 4, 2, dtype=torch.double, requires_grad=True)
    key = torch.randn(1, 5, 2, 2, dtype=torch.double, requires_grad=True)
    value = torch.randn(1, 5, 2, 2, dtype=torch.double, requires_grad=True)
    routes = torch.tensor([[[0, 1, -1], [1, 2, 3], [0, 3, 4]]], dtype=torch.int32)

    expected, *_ = _sparse_gqa_chunk(query, key, value, routes, 0.5, fp32_parity)
    expected.sum().backward()
    expected_grads = tuple(tensor.grad.clone() for tensor in (query, key, value))

    actual_inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in (query, key, value))
    actual = _SparseGQAAttention.apply(*actual_inputs, routes, 0.5, 2, fp32_parity)
    actual.sum().backward()

    torch.testing.assert_close(actual, expected.detach(), rtol=0, atol=0)
    for actual_input, expected_grad in zip(actual_inputs, expected_grads, strict=True):
        torch.testing.assert_close(actual_input.grad, expected_grad, rtol=0, atol=1e-12)


def test_qwen38_rejects_native_fp8_training():
    _require()
    config = _tiny_config()
    config.quantization_config = {"quant_method": "fp8"}
    with pytest.raises(NotImplementedError, match="Native-FP8"):
        Qwen4ExpForConditionalGenerationPrimeRL(config)


def test_packed_ple_survives_custom_lm_head():
    """Custom heads bypass the outer HF forward; PLE still isolates packed rows."""
    from transformers import set_seed

    from arctic_platform.model.implementations.moe.layers.lm_head import _patch_model_forward

    config = _tiny_config()
    config.text_config.layer_types = ["linear_attention"] * config.text_config.num_hidden_layers
    config.text_config.use_cache = False
    set_seed(37)
    model = Qwen4ExpForConditionalGenerationPrimeRL(config)

    class Zero(torch.nn.Module):
        def forward(self, hidden_states, *args, **kwargs):
            assert "ple_position_ids" not in kwargs
            return torch.zeros_like(hidden_states)

    class HiddenHead(torch.nn.Module):
        def forward(self, hidden_states, *args, **kwargs):
            return hidden_states

    for layer in model.model.language_model.layers:
        layer.linear_attn = Zero()
        layer.mlp = Zero()
    model.lm_head = HiddenHead()
    _patch_model_forward(model)
    kwargs = dict(attention_mask={"linear_attention": None, "qwen_sparse_attention": None})
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    packed = model(input_ids=ids, position_ids=positions, **kwargs)
    separate = model(input_ids=ids[:, 4:], position_ids=positions[:, 4:], **kwargs)
    torch.testing.assert_close(packed[:, 4:], separate, rtol=0, atol=1e-6)
