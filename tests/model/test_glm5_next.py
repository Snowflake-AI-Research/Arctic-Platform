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
import hashlib
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


def _auxiliary_model(count=1):
    model = torch.nn.Module()
    model.config = types.SimpleNamespace(
        text_config=types.SimpleNamespace(
            model_type="glm5_next_text", num_hidden_layers=2, num_nextn_predict_layers=count
        )
    )
    return model


def _auxiliary_checkpoint(tmp_path):
    from safetensors.torch import save_file

    prefix = "model.language_model.layers.2."
    tensors = {
        prefix + "eh_proj.weight": torch.ones(2, 4, dtype=torch.bfloat16),
        prefix + "enorm.weight": torch.ones(4, dtype=torch.float32),
        prefix + "hnorm.weight": torch.ones(4, dtype=torch.bfloat16),
        prefix + "mlp.experts.0.down_proj.weight": torch.ones(4, 4).to(torch.float8_e4m3fn),
        prefix + "mlp.experts.0.down_proj.weight_scale_inv": torch.ones(1, 1, dtype=torch.float32),
        "model.language_model.layers.0.input_layernorm.weight": torch.zeros(4, dtype=torch.bfloat16),
    }
    save_file(tensors, tmp_path / "model-00001.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model-00001.safetensors")})
    )
    return {name: tensor for name, tensor in tensors.items() if name.startswith(prefix)}


def test_glm53_captures_auxiliary_mixed_dtypes_without_trainables(tmp_path, monkeypatch):
    from arctic_platform.model.implementations.glm53 import auxiliary_weights

    original = _auxiliary_checkpoint(tmp_path)
    model = _auxiliary_model()
    monkeypatch.setattr(auxiliary_weights.dist, "is_initialized", lambda: False)
    auxiliary_weights.capture_auxiliary_weights(model, tmp_path)
    exported = dict(auxiliary_weights.iter_preserved_auxiliary_weights(model))
    assert set(exported) == set(original)
    assert not dict(model.named_parameters()) and not dict(model.named_buffers())
    for name, value in exported.items():
        assert value.dtype == original[name].dtype
        torch_assert_equal(value.view(torch.uint8), original[name].view(torch.uint8))
        assert value.device.type == "cpu"


def test_glm53_auxiliary_capture_rejects_missing_and_changed_shards(tmp_path, monkeypatch):
    from arctic_platform.model.implementations.glm53 import auxiliary_weights

    _auxiliary_checkpoint(tmp_path)
    monkeypatch.setattr(auxiliary_weights.dist, "is_initialized", lambda: False)
    index = tmp_path / "model.safetensors.index.json"
    mapping = json.loads(index.read_text())
    mapping["weight_map"]["model.language_model.layers.2.enorm.weight"] = "missing.safetensors"
    index.write_text(json.dumps(mapping))
    with pytest.raises(RuntimeError, match="Cannot preserve"):
        auxiliary_weights.capture_auxiliary_weights(_auxiliary_model(), tmp_path)


def test_glm53_auxiliary_source_failure_reaches_peers(tmp_path, monkeypatch):
    from arctic_platform.model.implementations.glm53 import auxiliary_weights

    observed = []
    monkeypatch.setattr(auxiliary_weights.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(auxiliary_weights.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(auxiliary_weights.dist, "broadcast_object_list", lambda value, src: observed.append(value[0]))
    with pytest.raises(RuntimeError, match="Cannot preserve"):
        auxiliary_weights.capture_auxiliary_weights(_auxiliary_model(), tmp_path)
    assert observed and observed[0] is not None


def test_glm53_auxiliary_capture_nonmaster_never_opens_source(tmp_path, monkeypatch):
    from arctic_platform.model.implementations.glm53 import auxiliary_weights

    model = _auxiliary_model()
    monkeypatch.setattr(auxiliary_weights.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(auxiliary_weights.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(auxiliary_weights.dist, "broadcast_object_list", lambda value, src: None)
    monkeypatch.setattr(auxiliary_weights, "_capture", lambda *args: pytest.fail("nonmaster read"))
    auxiliary_weights.capture_auxiliary_weights(model, tmp_path)
    assert not hasattr(model, auxiliary_weights.STATE_ATTRIBUTE)


def _reviewed_auxiliary_fixture(tmp_path, inventory="absent"):
    from safetensors.torch import save_file

    original = _auxiliary_checkpoint(tmp_path)
    tensors = {"model.language_model.layers.0.input_layernorm.weight": torch.zeros(4, dtype=torch.bfloat16)}
    if inventory != "absent":
        tensors.update(original)
    if inventory == "partial":
        tensors.pop("model.language_model.layers.2.eh_proj.weight")
    if inventory == "unexpected":
        tensors["model.language_model.layers.3.enorm.weight"] = torch.ones(4)
    save_file(tensors, tmp_path / "model-00001.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model-00001.safetensors")})
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm5_next",
                "text_config": {"model_type": "glm5_next_text", "num_hidden_layers": 2, "num_nextn_predict_layers": 1},
            }
        )
    )
    files = [
        {"path": path.name, "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(tmp_path.iterdir())
    ]
    request = {
        "schema_version": 1,
        "config_id": "glm-reduced-fixture",
        "fixture_locator": "ap-correctness/glm-reduced",
        "reference_path": str((tmp_path / "reference").resolve()),
        "reviewed_hash": "0123456789abcdef",
        "fixture_sha256": (
            hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        ),
        "files": files,
        "source_checkpoint": "zai-org/GLM-5.3-Flash",
        "num_hidden_layers": 2,
        "vision_depth": None,
        "materialization": "pretrained-prefix-reduction",
    }
    config = types.SimpleNamespace(
        name=request["fixture_locator"], debug=types.SimpleNamespace(reduced_fixture_mtp=request)
    )
    return _auxiliary_model(), config, original


def test_glm53_unauthorized_reduced_fixture_still_requires_original_auxiliary_weights(tmp_path, monkeypatch):
    from arctic_platform.debug.reduced_fixture_mtp import maybe_capture_reduced_fixture_mtp

    model, config, _ = _reviewed_auxiliary_fixture(tmp_path)
    config.debug.reduced_fixture_mtp = None
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="original prediction layer is incomplete"):
        maybe_capture_reduced_fixture_mtp(model, tmp_path, config)


def test_glm53_authenticated_reduced_fixture_loads_without_auxiliary_but_full_export_remains_strict(
    tmp_path, monkeypatch
):
    from arctic_platform.debug.reduced_fixture_mtp import maybe_capture_reduced_fixture_mtp
    from arctic_platform.model.implementations.glm53 import auxiliary_weights

    model, config, _ = _reviewed_auxiliary_fixture(tmp_path)
    setattr(model, auxiliary_weights.STATE_ATTRIBUTE, "stale capture")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    maybe_capture_reduced_fixture_mtp(model, tmp_path, config)
    assert not hasattr(model, auxiliary_weights.STATE_ATTRIBUTE)
    with pytest.raises(RuntimeError, match="requires captured original prediction layers"):
        list(auxiliary_weights.iter_preserved_auxiliary_weights(model))


def test_glm53_authenticated_complete_auxiliary_inventory_is_preserved(tmp_path, monkeypatch):
    from arctic_platform.debug.reduced_fixture_mtp import maybe_capture_reduced_fixture_mtp
    from arctic_platform.model.implementations.glm53 import auxiliary_weights

    model, config, original = _reviewed_auxiliary_fixture(tmp_path, "complete")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    maybe_capture_reduced_fixture_mtp(model, tmp_path, config)
    exported = dict(auxiliary_weights.iter_preserved_auxiliary_weights(model))
    assert exported.keys() == original.keys()
    for name, tensor in exported.items():
        torch_assert_equal(tensor.view(torch.uint8), original[name].view(torch.uint8))


@pytest.mark.parametrize("inventory", ["partial", "unexpected"])
def test_glm53_authenticated_reduced_fixture_rejects_partial_or_unexpected_auxiliary_inventory(
    tmp_path, monkeypatch, inventory
):
    from arctic_platform.debug.reduced_fixture_mtp import maybe_capture_reduced_fixture_mtp

    model, config, _ = _reviewed_auxiliary_fixture(tmp_path, inventory)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="partial or unexpected GLM"):
        maybe_capture_reduced_fixture_mtp(model, tmp_path, config)


@pytest.mark.parametrize("change", ["bytes", "count", "family"])
def test_glm53_reduced_fixture_rejects_changed_bytes_or_runtime_configuration(tmp_path, monkeypatch, change):
    from arctic_platform.debug.reduced_fixture_mtp import maybe_capture_reduced_fixture_mtp

    model, config, _ = _reviewed_auxiliary_fixture(tmp_path)
    if change == "bytes":
        with (tmp_path / "model-00001.safetensors").open("ab") as stream:
            stream.write(b"changed")
    elif change == "count":
        model.config.text_config.num_nextn_predict_layers = 2
    else:
        model.config.text_config.model_type = "other"
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="bytes differ|runtime model differs|configured prediction weights"):
        maybe_capture_reduced_fixture_mtp(model, tmp_path, config)


@pytest.mark.parametrize("peer", [["ValueError: peer bytes differ", False], [None, False]])
def test_glm53_reduced_fixture_peer_failure_or_inventory_disagreement_reaches_every_rank(tmp_path, monkeypatch, peer):
    from arctic_platform.debug.reduced_fixture_mtp import maybe_capture_reduced_fixture_mtp

    model, config, _ = _reviewed_auxiliary_fixture(tmp_path)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        torch.distributed, "all_gather_object", lambda rows, local: rows.__setitem__(slice(None), [local, peer])
    )
    with pytest.raises(RuntimeError, match="peer bytes differ|decisions differ"):
        maybe_capture_reduced_fixture_mtp(model, tmp_path, config)


@pytest.mark.parametrize("include_skipped,rank", [(False, 0), (True, 0), (True, 1)])
def test_glm53_full_hf_export_auxiliary_scope(monkeypatch, include_skipped, rank):
    from arctic_platform.model.implementations.glm53 import deepspeed_integration as glm

    model = _auxiliary_model()
    ordinary = torch.ones(4)
    auxiliary = torch.ones(4, dtype=torch.bfloat16)
    monkeypatch.setattr(
        glm.qwen_ds,
        "_build_iter_full_hf_weights",
        lambda model: lambda **kwargs: iter([("trained.weight", ordinary)]),
    )
    monkeypatch.setattr(glm.torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(glm, "iter_preserved_auxiliary_weights", lambda model: iter([("inactive.weight", auxiliary)]))
    exported = dict(glm._build_iter_full_hf_weights_with_auxiliary(model)(include_skipped=include_skipped))
    assert exported["trained.weight"] is ordinary
    assert ("inactive.weight" in exported) == (include_skipped and rank == 0)
    assert glm._adapter().hf_weight_export is glm._build_iter_full_hf_weights_with_auxiliary


def test_glm53_full_hf_export_rejects_live_auxiliary_collision(monkeypatch):
    from arctic_platform.model.implementations.glm53 import deepspeed_integration as glm

    value = torch.ones(4)
    monkeypatch.setattr(
        glm.qwen_ds,
        "_build_iter_full_hf_weights",
        lambda model: lambda **kwargs: iter([("inactive.weight", value)]),
    )
    monkeypatch.setattr(glm.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(glm, "iter_preserved_auxiliary_weights", lambda model: iter([("inactive.weight", value)]))
    with pytest.raises(RuntimeError, match="collides"):
        list(glm._build_iter_full_hf_weights_with_auxiliary(_auxiliary_model())(include_skipped=True))


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


@pytest.mark.parametrize("suffix", ["", ".weight"])
def test_glm53_packed_hf_experts_convert_without_losing_weights(suffix):
    prefix = "model.language_model.layers.3.mlp"
    gate_up = torch.arange(48, dtype=torch.float32).reshape(2, 6, 4)
    down = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    router = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    bias = torch.tensor([0.25, 0.5])
    shared = {
        name: torch.full(shape, value)
        for name, shape, value in (
            ("gate_proj", (3, 4), 1.0),
            ("down_proj", (4, 3), 2.0),
            ("up_proj", (3, 4), 3.0),
        )
    }
    state = {
        f"{prefix}.experts.gate_up_proj{suffix}": gate_up.clone(),
        f"{prefix}.experts.down_proj{suffix}": down.clone(),
        f"{prefix}.gate.weight": router.clone(),
        f"{prefix}.gate.e_score_correction_bias": bias.clone(),
        **{f"{prefix}.shared_experts.{name}.weight": weight.clone() for name, weight in shared.items()},
    }

    convert_hf_layer_to_prime(state, 3)

    expected = {
        f"{prefix}.experts.w1": gate_up[:, :3],
        f"{prefix}.experts.w2": down,
        f"{prefix}.experts.w3": gate_up[:, 3:],
        f"{prefix}.router.gate.weight": router,
        f"{prefix}.expert_bias": bias,
        **{
            f"{prefix}.shared_expert.{prime}": shared[hf]
            for prime, hf in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            )
        },
    }
    assert state.keys() == expected.keys()
    for name, weight in expected.items():
        torch_assert_equal(state[name], weight)


def test_glm53_packed_hf_state_loads_strictly_into_native_model():
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration

    from arctic_platform.model.implementations.glm53.converting_glm5_next import convert_hf_to_prime
    from arctic_platform.model.implementations.glm53.modeling_glm5_next import Glm5NextForConditionalGenerationPrimeRL

    set_seed(0)
    config = _tiny_config()
    source = Glm5NextForConditionalGeneration(config)
    native = Glm5NextForConditionalGenerationPrimeRL(config)
    state = convert_hf_to_prime(source.state_dict())

    native.load_state_dict(state, strict=True)

    for name, weight in native.state_dict().items():
        torch_assert_equal(weight, state[name])


@pytest.mark.parametrize("suffix", ["", ".weight"])
def test_glm53_packed_hf_fp8_scales_and_other_layers_are_preserved(suffix):
    prefix = "model.language_model.layers.3.mlp"
    gate_up = torch.arange(48, dtype=torch.float32).reshape(2, 6, 4).to(torch.float8_e4m3fn)
    down = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3).to(torch.float8_e4m3fn)
    gate_up_scale = torch.arange(12, dtype=torch.float32).reshape(2, 6, 1) + 1
    down_scale = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1) + 1
    shared_scale = torch.full((1, 1), 0.5)
    untouched = {
        "model.language_model.layers.13.mlp.experts.gate_up_proj": gate_up.clone(),
        "model.language_model.layers.3.self_attn.q_proj.weight": torch.ones(4, 4),
        "model.language_model.embed_tokens.weight": torch.ones(8, 4),
    }
    state = {
        **untouched,
        f"{prefix}.experts.gate_up_proj{suffix}": gate_up,
        f"{prefix}.experts.gate_up_proj.weight_scale_inv": gate_up_scale,
        f"{prefix}.experts.down_proj{suffix}": down,
        f"{prefix}.experts.down_proj.weight_scale_inv": down_scale,
        f"{prefix}.gate.weight": torch.ones(2, 4),
        f"{prefix}.gate.e_score_correction_bias": torch.ones(2),
    }
    for name, shape in (("gate_proj", (3, 4)), ("down_proj", (4, 3)), ("up_proj", (3, 4))):
        state[f"{prefix}.shared_experts.{name}.weight"] = torch.ones(shape).to(torch.float8_e4m3fn)
        state[f"{prefix}.shared_experts.{name}.weight_scale_inv"] = shared_scale.clone()

    convert_hf_layer_to_prime(state, 3)

    for name, expected in untouched.items():
        assert state[name] is expected
    for projection, weight, scale in (
        ("w1", gate_up[:, :3], gate_up_scale[:, :3]),
        ("w2", down, down_scale),
        ("w3", gate_up[:, 3:], gate_up_scale[:, 3:]),
    ):
        assert state[f"{prefix}.experts.{projection}"].dtype == torch.float8_e4m3fn
        torch_assert_equal(state[f"{prefix}.experts.{projection}"].float(), weight.float())
        torch_assert_equal(state[f"{prefix}.experts.{projection}_scale_inv"], scale)
        torch_assert_equal(state[f"{prefix}.shared_expert.{projection}_scale_inv"], shared_scale)
    assert not any(name.startswith(f"{prefix}.experts.gate_up_proj") for name in state)


@pytest.mark.parametrize("missing_projection", ["gate_up_proj", "down_proj"])
def test_glm53_packed_hf_fp8_rejects_missing_scales(missing_projection):
    prefix = "model.language_model.layers.3.mlp"
    state = {
        f"{prefix}.experts.gate_up_proj": torch.ones(2, 6, 4).to(torch.float8_e4m3fn),
        f"{prefix}.experts.gate_up_proj.weight_scale_inv": torch.ones(2, 6, 1),
        f"{prefix}.experts.down_proj": torch.ones(2, 4, 3).to(torch.float8_e4m3fn),
        f"{prefix}.experts.down_proj.weight_scale_inv": torch.ones(2, 4, 1),
        f"{prefix}.gate.weight": torch.ones(2, 4),
        f"{prefix}.gate.e_score_correction_bias": torch.ones(2),
        f"{prefix}.shared_experts.gate_proj.weight": torch.ones(3, 4),
        f"{prefix}.shared_experts.down_proj.weight": torch.ones(4, 3),
        f"{prefix}.shared_experts.up_proj.weight": torch.ones(3, 4),
    }
    del state[f"{prefix}.experts.{missing_projection}.weight_scale_inv"]

    with pytest.raises(ValueError, match="missing its scale"):
        convert_hf_layer_to_prime(state, 3)


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
