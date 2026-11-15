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
"""Nemotron-H Omni (Nemotron 3.5 Super VL): Arctic's composite against transformers' native model, on CPU.

The reference is transformers' ``NemotronH_Omni_Reasoning_V3`` loaded with ``from_pretrained`` from a checkpoint in
NVIDIA's layout, which Arctic's conversion writes: transformers' own rename rules then check the names, and its
forward checks the image path and the language model. The checkpoint's ``vision_projector.vision_final_layernorm``
is set to identity here, since transformers 5.20 does not load it (Arctic does; see ``converting_nemotron_h_omni``).

On CPU the experts run their local for-loop path, and ``torch.histc`` counts integer expert ids as floats, which it
does on CUDA natively.
"""

from __future__ import annotations

import re

import pytest
import torch

pytest.importorskip("transformers.models.nemotron_h_omni")

from safetensors.torch import save_file  # noqa: E402
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig as HFNemotronHConfig  # noqa: E402
from transformers.models.nemotron_h_omni.configuration_nemotron_h_omni import (  # noqa: E402
    NemotronH_Omni_Reasoning_V3_Config,
)
from transformers.models.nemotron_h_omni.modeling_nemotron_h_omni import NemotronH_Omni_Reasoning_V3  # noqa: E402

from arctic_platform.common.packing import media_spec_for  # noqa: E402
from arctic_platform.model.implementations.moe.layers.lm_head import inject_prime_lm_head  # noqa: E402
from arctic_platform.model.implementations.moe.vlm import get_language_model  # noqa: E402
from arctic_platform.model.implementations.moe.vlm import get_vision_encoder  # noqa: E402
from arctic_platform.model.implementations.nemotron_h.configuration_nemotron_h import NemotronHConfig  # noqa: E402
from arctic_platform.model.implementations.nemotron_h.converting_nemotron_h_omni import (  # noqa: E402
    convert_checkpoint_to_prime,
)
from arctic_platform.model.implementations.nemotron_h.converting_nemotron_h_omni import (  # noqa: E402
    convert_prime_to_checkpoint,
)
from arctic_platform.model.implementations.nemotron_h.modeling_nemotron_h_omni import (  # noqa: E402
    NemotronHOmniForConditionalGenerationPrimeRL,
)
from arctic_platform.model.implementations.qwen35.models import get_custom_vlm_cls  # noqa: E402
from arctic_platform.model.patches.peft import apply_peft  # noqa: E402

IMAGE_TOKEN_ID = 18
TEXT_CONFIG = dict(
    vocab_size=64,
    hidden_size=32,
    layers_block_type=["mamba", "moe", "attention"],
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    mamba_num_heads=4,
    mamba_head_dim=8,
    ssm_state_size=8,
    n_routed_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=16,
    moe_shared_expert_intermediate_size=16,
    n_shared_experts=1,
    moe_latent_size=16,
    num_nextn_predict_layers=1,
    routed_scaling_factor=1.0,
    n_group=1,
    topk_group=1,
    norm_topk_prob=True,
    pad_token_id=0,
    eos_token_id=2,
)
# Arctic's and transformers' configs name the Mamba group count, conv width and chunk size differently.
ARCTIC_MAMBA = dict(mamba_n_groups=1, mamba_d_conv=4, mamba_chunk_size=8)
HF_MAMBA = dict(n_groups=1, conv_kernel=4, chunk_size=8)
VISION_CONFIG = dict(
    model_type="radio",
    hidden_size=32,
    num_hidden_layers=1,
    num_attention_heads=4,
    intermediate_size=64,
    patch_size=4,
    image_size=16,
    video_temporal_patch_size=2,
)
# Two images of 4x6 and 2x2 patches: 6 and 1 tokens after the 2x2 pixel shuffle.
IMAGE_GRIDS_HW = torch.tensor([[4, 6], [2, 2]])
INPUT_IDS = torch.tensor([[1, 5, 7] + [IMAGE_TOKEN_ID] * 6 + [9, 10] + [IMAGE_TOKEN_ID] + [11, 12]])


def _config(*, native: bool = False) -> NemotronH_Omni_Reasoning_V3_Config:
    text = HFNemotronHConfig(**TEXT_CONFIG, **HF_MAMBA) if native else NemotronHConfig(**TEXT_CONFIG, **ARCTIC_MAMBA)
    config = NemotronH_Omni_Reasoning_V3_Config(
        text_config=text,
        vision_config=VISION_CONFIG,
        image_token_id=IMAGE_TOKEN_ID,
        vision_hidden_size=32,
        projector_hidden_size=48,
        downsample_ratio=0.5,
    )
    for sub in (config, config.text_config, config.vision_config):
        sub._attn_implementation = "sdpa"
    config.use_grouped_mm = False
    return config


@pytest.fixture
def cpu_experts(monkeypatch):
    histc = torch.histc

    def integer_histc(values, *args, **kwargs):
        if values.is_floating_point():
            return histc(values, *args, **kwargs)
        return histc(values.float(), *args, **kwargs).to(values.dtype)

    monkeypatch.setattr(torch, "histc", integer_histc)


def _model() -> NemotronHOmniForConditionalGenerationPrimeRL:
    torch.manual_seed(0)
    model = NemotronHOmniForConditionalGenerationPrimeRL(_config()).float()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.2)
    for module in model.modules():
        if hasattr(module, "router") and hasattr(module, "set_ep_comm_backend"):
            module.ep_comm_backend = "local"
            module.experts.forward = module.experts._forward_deepep
    return model


def _pixel_values(grids_hw: torch.Tensor) -> torch.Tensor:
    torch.manual_seed(1)
    vision = _config().vision_config
    return torch.randn(int(grids_hw.prod(-1).sum()), vision.num_channels * vision.patch_size**2)


def _grid_thw(grids_hw: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.ones(grids_hw.shape[0], 1, dtype=torch.long), grids_hw], dim=1)


def _logits(model, **kwargs) -> torch.Tensor:
    return model(input_ids=INPUT_IDS, position_ids=torch.arange(INPUT_IDS.shape[1])[None], **kwargs)["logits"]


def test_registration():
    config = _config()
    assert get_custom_vlm_cls(config) is NemotronHOmniForConditionalGenerationPrimeRL
    spec = media_spec_for("nemotron_h_omni")
    assert spec.positions == "1d" and spec.modalities == ("image",)
    assert spec.placeholder_ids(config) == {"image": IMAGE_TOKEN_ID}
    assert spec.spatial_merge_size(config) == 2

    model = NemotronHOmniForConditionalGenerationPrimeRL(config)
    assert get_vision_encoder(model) is model.model.visual
    assert get_language_model(model) is model.model.language_model
    assert model.config.text_config.use_grouped_mm is False


def test_a_transformers_text_config_is_refused():
    with pytest.raises(TypeError, match="Arctic's NemotronHConfig"):
        NemotronHOmniForConditionalGenerationPrimeRL(_config(native=True))


def test_matches_transformers_with_images(tmp_path, cpu_experts):
    model = _model().eval()
    with torch.no_grad():
        model.model.visual.vision_final_layernorm.weight.fill_(1.0)
        model.model.visual.vision_final_layernorm.bias.zero_()
    checkpoint = convert_prime_to_checkpoint({name: tensor.clone() for name, tensor in model.state_dict().items()})
    _config(native=True).save_pretrained(tmp_path)
    save_file({name: tensor.contiguous() for name, tensor in checkpoint.items()}, str(tmp_path / "model.safetensors"))
    reference, info = NemotronH_Omni_Reasoning_V3.from_pretrained(
        tmp_path, config=_config(native=True), output_loading_info=True, dtype=torch.float32
    )
    # Every converted name is one transformers maps, but the final norm it leaves out.
    assert not info["missing_keys"] and not info["mismatched_keys"]
    assert set(info["unexpected_keys"]) == {
        "vision_projector.vision_final_layernorm.weight",
        "vision_projector.vision_final_layernorm.bias",
    }

    inject_prime_lm_head(model)
    pixel_values = _pixel_values(IMAGE_GRIDS_HW)
    with torch.no_grad():
        expected = reference.eval()(
            input_ids=INPUT_IDS, pixel_values=pixel_values, image_grid_hw=IMAGE_GRIDS_HW
        ).logits
        actual = _logits(model, pixel_values=pixel_values, image_grid_thw=_grid_thw(IMAGE_GRIDS_HW))
        text_only = _logits(model)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(actual, text_only)


def test_checkpoint_conversion_round_trips_whole_and_by_layer():
    state = {name: tensor.clone() for name, tensor in _model().state_dict().items()}
    checkpoint = convert_prime_to_checkpoint(dict(state))

    assert {name.split(".")[0] for name in checkpoint} == {
        "language_model",
        "vision_model",
        "vision_projector",
        "mlp1",
    }
    assert "vision_projector.vision_final_layernorm.weight" in checkpoint
    assert checkpoint["vision_model.radio_model.model.blocks.0.attn.qkv.weight"].shape == (96, 32)
    back = convert_checkpoint_to_prime(dict(checkpoint))
    assert back.keys() == state.keys()
    assert all(torch.equal(back[name], state[name]) for name in state)

    # The per-layer export converts every decoder layer alone, and then the rest (-1).
    def layer(name: str, prefix: str) -> int:
        return int(name.split(".")[3]) if name.startswith(prefix) else -1

    by_layer = {}
    for index in sorted({layer(name, "model.language_model.layers.") for name in state}):
        part = {name: tensor for name, tensor in state.items() if layer(name, "model.language_model.layers.") == index}
        by_layer.update(NemotronHOmniForConditionalGenerationPrimeRL.convert_layer_to_hf(part, index))
    assert by_layer.keys() == checkpoint.keys()
    assert all(torch.equal(by_layer[name], checkpoint[name]) for name in checkpoint)

    by_layer = {}
    for index in sorted({layer(name, "language_model.backbone.layers.") for name in checkpoint}):
        part = {
            name: tensor
            for name, tensor in checkpoint.items()
            if layer(name, "language_model.backbone.layers.") == index
        }
        by_layer.update(NemotronHOmniForConditionalGenerationPrimeRL.convert_layer_to_prime(part, index))
    assert by_layer.keys() == state.keys()


def test_checkpoint_only_weights_are_dropped():
    checkpoint = convert_prime_to_checkpoint(dict(_model().state_dict()))
    checkpoint["language_model.mtp.layers.0.norm.weight"] = torch.ones(32)
    checkpoint["vision_model.radio_model.input_conditioner.norm_mean"] = torch.zeros(3, 1, 1)
    assert convert_checkpoint_to_prime(checkpoint).keys() == _model().state_dict().keys()


def test_a_trainable_vision_tower_joins_a_text_only_backward(cpu_experts):
    model = _model()
    inject_prime_lm_head(model)
    with torch.no_grad():
        expected = _logits(model.eval())

    logits = _logits(model.train())
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    logits.sum().backward()
    trainable = [(name, p) for name, p in model.model.visual.named_parameters() if p.requires_grad]
    # Only the video patch embedder, which images never reach, is frozen.
    assert len(trainable) == len(list(model.model.visual.parameters())) - 1
    for name, parameter in trainable:
        assert parameter.grad is not None, name
        assert not parameter.grad.any(), name

    model.zero_grad(set_to_none=True)
    _logits(
        model, pixel_values=_pixel_values(IMAGE_GRIDS_HW), image_grid_thw=_grid_thw(IMAGE_GRIDS_HW)
    ).sum().backward()
    assert all(parameter.grad is not None for _, parameter in trainable)

    calls = []
    model.model.visual.dummy_forward = lambda: calls.append(1)
    model.model.visual.requires_grad_(False)
    _logits(model)
    assert calls == []


def test_image_inputs_are_checked(cpu_experts):
    model = _model().eval()
    pixel_values = _pixel_values(IMAGE_GRIDS_HW)
    with torch.no_grad(), pytest.raises(ValueError, match="hold 7 image tokens but the images encode to 6"):
        _logits(model, pixel_values=pixel_values[:24], image_grid_thw=_grid_thw(IMAGE_GRIDS_HW[:1]))
    with pytest.raises(ValueError, match=r"rows \[1, h, w\]"):
        _logits(model, pixel_values=pixel_values, image_grid_thw=torch.tensor([[2, 4, 6], [1, 2, 2]]))
    with pytest.raises(ValueError, match="not both"):
        _logits(
            model, pixel_values=pixel_values, image_grid_thw=_grid_thw(IMAGE_GRIDS_HW), image_grid_hw=IMAGE_GRIDS_HW
        )
    with pytest.raises(ValueError, match="passed together"):
        _logits(model, pixel_values=pixel_values)
    with pytest.raises(NotImplementedError, match="images only"):
        _logits(model, pixel_values_videos=torch.zeros(1), video_grid_thw=torch.tensor([[2, 2, 2]]))


def test_lora_leaves_the_image_path_alone():
    model = apply_peft(_model(), {"peft_type": "LORA", "r": 2, "target_modules": "all-linear"})
    adapted = {name.split(".lora_")[0] for name, _ in model.named_parameters() if "lora_" in name}
    assert adapted
    assert not [name for name in adapted if ".visual." in name]
    assert any(re.search(r"language_model\.layers\.2\.self_attn\.q_proj$", name) for name in adapted)
    assert not any(parameter.requires_grad for parameter in model.base_model.model.model.visual.parameters())


def test_model_builder_selects_the_composite(tmp_path):
    from arctic_platform.model.implementations.qwen35.config import ModelConfig
    from arctic_platform.model.implementations.qwen35.model_builder import get_model

    _config().save_pretrained(tmp_path)
    model = get_model(
        ModelConfig(name=str(tmp_path), impl="custom", attn="sdpa", moe_use_grouped_mm=False),
        device=torch.device("meta"),
        dtype=torch.float32,
    )
    assert isinstance(model, NemotronHOmniForConditionalGenerationPrimeRL)
    assert isinstance(model.config.text_config, NemotronHConfig)
    assert model.model.language_model.layers[1].mlp.experts.w1.device.type == "meta"


def test_replayed_routes_reach_each_moe_layer(monkeypatch):
    from arctic_platform.model.implementations.moe.layers.moe import LatentMoE

    seen = []

    def moe_forward(self, x, routed_experts=None):
        seen.append(routed_experts)
        return torch.zeros_like(x)

    monkeypatch.setattr(LatentMoE, "forward", moe_forward)
    model = _model().eval()
    inject_prime_lm_head(model)
    layers = len(TEXT_CONFIG["layers_block_type"])
    routes = torch.arange(INPUT_IDS.numel() * layers * 2).reshape(1, INPUT_IDS.shape[1], layers, 2) % 4
    with torch.no_grad():
        _logits(
            model,
            pixel_values=_pixel_values(IMAGE_GRIDS_HW),
            image_grid_thw=_grid_thw(IMAGE_GRIDS_HW),
            routed_experts=routes,
        )
    (moe_layer,) = [index for index, kind in enumerate(TEXT_CONFIG["layers_block_type"]) if kind == "moe"]
    (replayed,) = seen
    torch.testing.assert_close(replayed, routes[:, :, moe_layer, :], rtol=0, atol=0)


# Checkpoint tensors the model does not hold: a complete export carries them from the source.
CHECKPOINT_ONLY = {
    "language_model.mtp.layers.0.mixer.q_proj.weight": (32, 32),
    "language_model.mtp.layers.1.mixer.experts.0.up_proj.weight": (16, 16),
    "vision_model.radio_model.input_conditioner.norm_mean": (3, 1, 1),
    "vision_model.radio_model.input_conditioner.norm_std": (3, 1, 1),
}


def _write_checkpoint(path, tensors) -> None:
    import json

    path.mkdir()
    _config().save_pretrained(path)
    items = sorted(tensors.items())
    weight_map = {}
    for shard, chunk in enumerate((items[::2], items[1::2])):
        filename = f"model-{shard}.safetensors"
        save_file({name: tensor.contiguous() for name, tensor in chunk}, str(path / filename))
        weight_map.update({name: filename for name, _ in chunk})
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def test_loads_the_checkpoint_and_exports_it_whole(tmp_path, single_rank_group, monkeypatch):
    from arctic_platform.model.implementations.moe.deepspeed_integration import build_iter_full_hf_weights
    from arctic_platform.model.implementations.moe.vllm_weights import build_iter_full_vllm_weights
    from arctic_platform.model.implementations.qwen35 import model_builder
    from arctic_platform.model.implementations.qwen35.config import ModelConfig
    from arctic_platform.model.weight_export import requires_hf_weight_sync

    prime_state = {name: tensor.detach().clone() for name, tensor in _model().state_dict().items()}
    checkpoint = convert_prime_to_checkpoint(dict(prime_state))
    generator = torch.Generator().manual_seed(1)
    checkpoint.update({name: torch.randn(shape, generator=generator) for name, shape in CHECKPOINT_ONLY.items()})
    _write_checkpoint(tmp_path / "omni", checkpoint)

    monkeypatch.setattr(model_builder, "_move_buffers_to_cuda", lambda *_args: None)
    config = ModelConfig(
        name=str(tmp_path / "omni"),
        attn="sdpa",
        impl="custom",
        moe_use_grouped_mm=False,
        fsdp_cpu_offload=True,
        weight_conversion_cache_dir=str(tmp_path / "cache"),
    )
    model = model_builder.get_model(config, device=torch.device("meta"), dtype=torch.float32)
    assert type(model) is NemotronHOmniForConditionalGenerationPrimeRL
    model_builder.load_dcp_from_hf(model, config, None)
    state = model.state_dict()
    assert all(torch.equal(state[name], tensor) for name, tensor in prime_state.items())

    complete = dict(build_iter_full_hf_weights(model)(include_skipped=True))
    assert complete.keys() == checkpoint.keys()
    assert all(torch.equal(complete[name], tensor) for name, tensor in checkpoint.items())
    # Weight sync sends the trained weights in the checkpoint layout, which vLLM's own loader maps.
    assert requires_hf_weight_sync(model)
    synced = dict(build_iter_full_hf_weights(model)())
    # Buffers are constant checkpoint state the sampler loaded with the model.
    buffers = ("e_score_correction_bias", "summary_idxs")
    assert synced.keys() == {name for name in checkpoint if name not in CHECKPOINT_ONLY and not name.endswith(buffers)}
    with pytest.raises(RuntimeError, match="checkpoint layout"):
        dict(build_iter_full_vllm_weights(model)())


def test_parallelism_metadata_comes_from_the_language_model(tmp_path):
    from transformers import AutoConfig

    from arctic_platform.model.loader import model_parallelism_metadata_from_config

    _config().save_pretrained(tmp_path)
    metadata = model_parallelism_metadata_from_config(AutoConfig.from_pretrained(str(tmp_path)).to_dict())
    assert (metadata.num_attention_heads, metadata.num_key_value_heads) == (4, 2)
    assert metadata.has_linear_attention is False


def test_a_complete_export_needs_the_source_checkpoint():
    with pytest.raises(RuntimeError, match="not loaded from a checkpoint"):
        next(_model().iter_hf_checkpoint_passthrough())
