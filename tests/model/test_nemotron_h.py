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

"""NemotronH on CPU: router replay, and text-only training of a ``nemotron_h_omni`` (Nemotron 3.5 VL) checkpoint.

The omni checkpoint nests the language model under ``language_model.`` beside a vision tower, projector and MTP
head. The text-only model loads only the language model and writes the checkpoint back complete: its own tensors
under their original names, and every other tensor unchanged from the source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

from arctic_platform.model.implementations.gpu.router_replay_recompute import _router_supports_routed_experts
from arctic_platform.model.implementations.gpu.router_replay_recompute import install_self_router_replay
from arctic_platform.model.implementations.moe.layers.moe import ROUTER_REPLAY_FRESH
from arctic_platform.model.implementations.moe.layers.moe import LatentMoE
from arctic_platform.model.implementations.moe.layers.moe import NemotronHRouter
from arctic_platform.model.implementations.moe.vllm_weights import named_weight_sync_tensors
from arctic_platform.model.implementations.nemotron_h.configuration_nemotron_h import NemotronHConfig
from arctic_platform.model.implementations.nemotron_h.configuration_nemotron_h import NemotronHOmniTextConfig
from arctic_platform.model.implementations.nemotron_h.converting_nemotron_h import is_omni_language_model_tensor
from arctic_platform.model.implementations.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM
from arctic_platform.model.implementations.nemotron_h.modeling_nemotron_h import NemotronHOmniTextForCausalLMPrimeRL
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal

LAYERS_BLOCK_TYPE = ["mamba", "moe", "attention", "moe"]
MOE_LAYERS = [1, 3]
TOP_K = 2

# Tensors of the multimodal checkpoint that text-only training does not own.
OMNI_EXTRA_TENSORS = {
    "language_model.mtp.layers.0.mixer.q_proj.weight": (16, 16),
    "language_model.mtp.layers.1.mixer.experts.0.up_proj.weight": (8, 8),
    "vision_model.radio_model.model.patch_generator.embedder.weight": (8, 48),
    "vision_model.radio_model.input_conditioner.norm_mean": (3, 1, 1),
    "mlp1.0.weight": (32,),
    "mlp1.1.weight": (16, 32),
    "vision_projector.vision_final_layernorm.weight": (8,),
}


def _llm_config(**overrides) -> dict:
    """A tiny NemotronH language-model config with the released checkpoint's (Hugging Face) field names."""
    config = dict(
        architectures=["NemotronHForCausalLM"],
        model_type="nemotron_h",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=16,
        layers_block_type=LAYERS_BLOCK_TYPE,
        mtp_layers_block_type=["attention", "moe"],
        num_nextn_predict_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        mamba_num_heads=8,
        mamba_head_dim=2,
        n_groups=8,
        ssm_state_size=4,
        conv_kernel=4,
        chunk_size=8,
        expand=1,
        n_routed_experts=4,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=8,
        moe_shared_expert_intermediate_size=8,
        moe_latent_size=8,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=5.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=11,
        tie_word_embeddings=False,
    )
    config.update(overrides)
    return config


def _omni_config(llm_config: dict, *, text_config_key: str = "llm_config") -> dict:
    """The released Nemotron 3.5 Super VL ``config.json`` layout around a language-model config."""
    return {
        "architectures": ["NemotronH_Omni_Reasoning_V3"],
        "model_type": "nemotron_h_omni",
        text_config_key: llm_config,
        "vision_config": {"model_type": "radio", "video_temporal_patch_size": 2},
        "downsample_ratio": 0.5,
        "img_context_token_id": 18,
        "projector_hidden_size": 32,
        "vit_hidden_size": 8,
        "video_temporal_patch_size": 2,
    }


class _Head(nn.Module):
    """Stands in for the injected PrimeRL LM head, which takes labels and temperature."""

    def forward(self, hidden_states, labels=None, temperature=None):
        return hidden_states


def _model(model_cls=NemotronHOmniTextForCausalLMPrimeRL, config_cls=NemotronHOmniTextConfig):
    set_seed(0)
    model = model_cls(config_cls.from_dict(_llm_config(), attn_implementation="sdpa"))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_()
        for layer_idx in MOE_LAYERS:
            bias = model.model.layers[layer_idx].mlp.router.e_score_correction_bias
            # Loading shifts the bias to a minimum of zero, so a zero-minimum bias round-trips exactly.
            bias.copy_(torch.rand(bias.shape) * 0.1)
            bias.sub_(bias.min())
    return model


def _write_checkpoint(path: Path, tensors: dict[str, torch.Tensor], config: dict) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    items = sorted(tensors.items())
    weight_map = {}
    for shard, chunk in enumerate((items[::2], items[1::2])):
        filename = f"model-{shard}.safetensors"
        save_file({name: tensor.contiguous() for name, tensor in chunk}, path / filename)
        weight_map.update({name: filename for name, _ in chunk})
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


@pytest.fixture
def omni_checkpoint(tmp_path):
    """A tiny ``nemotron_h_omni`` checkpoint and the PrimeRL state its language model holds."""
    model = _model()
    prime_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    hf_state = NemotronHOmniTextForCausalLMPrimeRL.convert_to_hf(dict(prime_state))
    generator = torch.Generator().manual_seed(1)
    for name, shape in OMNI_EXTRA_TENSORS.items():
        hf_state[name] = torch.randn(shape, generator=generator)
    path = tmp_path / "omni"
    _write_checkpoint(path, hf_state, _omni_config(_llm_config()))
    return path, prime_state, hf_state


# --------------------------------------------------------------------------- #
# Router replay
# --------------------------------------------------------------------------- #


def _router(n_group: int = 1, topk_group: int = 1) -> NemotronHRouter:
    set_seed(0)
    router = NemotronHRouter(
        dim=8, num_experts=6, top_k=TOP_K, n_group=n_group, topk_group=topk_group, norm_topk_prob=True
    )
    with torch.no_grad():
        router.gate.normal_()
        router.e_score_correction_bias.copy_(torch.linspace(0.0, 0.3, 6))
    return router


def _routes() -> torch.Tensor:
    return torch.tensor([[5, 4], [0, 1], [3, 2], [1, 0], [2, 5]], dtype=torch.int64)


@pytest.mark.parametrize("n_group,topk_group", [(1, 1), (2, 1)])
def test_router_gathers_replayed_routes(n_group, topk_group):
    router = _router(n_group, topk_group)
    x = torch.randn(5, 8)
    routed = _routes()

    top_scores, experts, num_tokens_per_expert = router(x, routed_experts=routed)

    torch_assert_equal(experts, routed)
    raw = torch.sigmoid(F.linear(x, router.gate)).gather(1, routed)
    torch_assert_close(top_scores, raw / raw.sum(dim=-1, keepdim=True))
    torch_assert_equal(num_tokens_per_expert, torch.bincount(routed.reshape(-1), minlength=6).float())


@pytest.mark.parametrize("n_group,topk_group", [(1, 1), (2, 1)])
def test_router_routes_fresh_tokens_like_replay_off(n_group, topk_group):
    router = _router(n_group, topk_group)
    router.enable_fresh_replay_routing()
    x = torch.randn(5, 8)
    routed = _routes()
    routed[[1, 3]] = ROUTER_REPLAY_FRESH
    fresh = (routed == ROUTER_REPLAY_FRESH).all(dim=1)

    off_scores, off_experts, _ = router(x)
    top_scores, experts, _ = router(x, routed_experts=routed)

    torch_assert_equal(experts[fresh], off_experts[fresh])
    torch_assert_equal(top_scores[fresh], off_scores[fresh])
    torch_assert_equal(experts[~fresh], routed[~fresh])


def test_router_is_wrapped_for_activation_checkpoint_replay():
    block = nn.Module()
    block.router = _router()

    assert _router_supports_routed_experts(block.router)
    assert install_self_router_replay(block) == 1


def test_router_correction_bias_is_exported_but_not_synced():
    block = nn.Module()
    block.router = _router()

    assert "router.e_score_correction_bias" in dict(named_weight_sync_tensors(block, include_export_buffers=True))
    assert "router.e_score_correction_bias" not in dict(named_weight_sync_tensors(block, include_export_buffers=False))


def test_model_hands_each_moe_layer_its_replayed_routes(monkeypatch):
    seen = []

    def moe_forward(self, x, routed_experts=None):
        seen.append(routed_experts)
        return torch.zeros_like(x)

    monkeypatch.setattr(LatentMoE, "forward", moe_forward)
    model = _model(NemotronHForCausalLM, NemotronHConfig)
    model.lm_head = _Head()
    input_ids = torch.randint(0, 32, (1, 6))
    position_ids = torch.arange(6).unsqueeze(0)
    routed = torch.arange(6 * len(LAYERS_BLOCK_TYPE) * TOP_K).reshape(1, 6, len(LAYERS_BLOCK_TYPE), TOP_K) % 4

    with torch.no_grad():
        model(input_ids=input_ids, position_ids=position_ids, routed_experts=routed)
        model(input_ids=input_ids, position_ids=position_ids)

    assert len(seen) == 2 * len(MOE_LAYERS)
    for routes, layer_idx in zip(seen[: len(MOE_LAYERS)], MOE_LAYERS):
        torch_assert_equal(routes, routed[:, :, layer_idx, :])
    assert seen[len(MOE_LAYERS) :] == [None] * len(MOE_LAYERS)


# --------------------------------------------------------------------------- #
# Text-only training of a nemotron_h_omni checkpoint
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text_config_key", ["llm_config", "text_config"])
def test_omni_text_config_reads_the_nested_language_model_config(tmp_path, text_config_key):
    (tmp_path / "config.json").write_text(json.dumps(_omni_config(_llm_config(), text_config_key=text_config_key)))

    config = NemotronHOmniTextConfig.from_omni_checkpoint(str(tmp_path), attn_implementation="sdpa")

    reference = NemotronHConfig.from_dict(_llm_config())
    assert type(config) is NemotronHOmniTextConfig
    assert config.model_type == "nemotron_h"
    assert config._attn_implementation == "sdpa"
    assert config.layers_block_type == LAYERS_BLOCK_TYPE
    for name in ("mamba_expand", "mamba_d_state", "conv_kernel", "n_groups", "moe_latent_size", "n_routed_experts"):
        assert getattr(config, name) == getattr(reference, name), name


def test_omni_text_config_rejects_other_checkpoints(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_llm_config()))

    with pytest.raises(ValueError, match="not a nemotron_h_omni checkpoint"):
        NemotronHOmniTextConfig.from_omni_checkpoint(str(tmp_path))


def test_omni_checkpoint_selects_the_generic_moe_loader(omni_checkpoint):
    from arctic_platform.model import ModelSpec
    from arctic_platform.model import ParallelismConfig

    path, _, _ = omni_checkpoint
    spec = ModelSpec(model_path_or_name=str(path), parallelism=ParallelismConfig(expert_parallel=2))

    assert spec.loader == "generic_moe"


def test_omni_checkpoint_parallelism_metadata_comes_from_the_language_model(omni_checkpoint):
    from transformers import AutoConfig

    from arctic_platform.model.loader import model_parallelism_metadata_from_config

    # The raw config nests the head counts under ``llm_config``, so DSS's SP preflight reads the composite config.
    path, _, _ = omni_checkpoint
    metadata = model_parallelism_metadata_from_config(AutoConfig.from_pretrained(str(path)).to_dict())

    assert metadata.num_attention_heads == 2
    assert metadata.num_key_value_heads == 1
    assert metadata.has_linear_attention is False


def test_omni_full_conversion_keeps_only_the_language_model(omni_checkpoint):
    _, prime_state, hf_state = omni_checkpoint

    converted = NemotronHOmniTextForCausalLMPrimeRL.convert_to_prime(dict(hf_state))

    assert converted.keys() == prime_state.keys()
    for name, tensor in prime_state.items():
        torch_assert_equal(converted[name], tensor)


def test_omni_passthrough_requires_a_source_checkpoint():
    with pytest.raises(RuntimeError, match="not loaded from a checkpoint"):
        next(_model().iter_hf_checkpoint_passthrough())


def test_omni_checkpoint_loads_the_language_model_and_exports_the_whole_checkpoint(
    omni_checkpoint, single_rank_group, monkeypatch
):
    from arctic_platform.model.implementations.moe.deepspeed_integration import build_iter_full_hf_weights
    from arctic_platform.model.implementations.moe.vllm_weights import build_iter_full_vllm_weights
    from arctic_platform.model.implementations.qwen35 import model_builder
    from arctic_platform.model.implementations.qwen35.config import ModelConfig

    monkeypatch.setattr(model_builder, "_move_buffers_to_cuda", lambda *_args: None)
    path, prime_state, hf_state = omni_checkpoint
    config = ModelConfig(name=str(path), attn="sdpa", impl="custom", fsdp_cpu_offload=True)
    model = model_builder.get_model(config, device=torch.device("meta"), dtype=torch.float32)
    assert type(model) is NemotronHOmniTextForCausalLMPrimeRL
    assert model.config.layers_block_type == LAYERS_BLOCK_TYPE

    model_builder.load_dcp_from_hf(model, config, None)

    state = model.state_dict()
    for name, tensor in prime_state.items():
        torch_assert_equal(state[name], tensor)

    complete = dict(build_iter_full_hf_weights(model)(include_skipped=True))
    assert complete.keys() == hf_state.keys()
    for name, tensor in hf_state.items():
        torch_assert_equal(complete[name], tensor)

    synced = dict(build_iter_full_hf_weights(model)())
    assert synced.keys() == {
        name
        for name in hf_state
        if is_omni_language_model_tensor(name) and not name.endswith("e_score_correction_bias")
    }

    served = dict(build_iter_full_vllm_weights(model)())
    assert all(name.startswith("language_model.") for name in served)
    for name in (
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm_f.weight",
        "language_model.lm_head.weight",
        "language_model.model.layers.0.mixer.A",
        "language_model.model.layers.1.mixer.experts.w13_weight",
        "language_model.model.layers.1.mixer.fc1_latent_proj.weight",
        "language_model.model.layers.2.mixer.q_proj.weight",
    ):
        assert name in served, name
