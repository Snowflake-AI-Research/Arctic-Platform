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

"""Nemotron-H Omni (Nemotron 3.5 Super VL) for training: transformers' RADIO tower and projector on Arctic's Nemotron-H.

The image path is transformers' own (``NemotronH_Omni_Reasoning_V3.get_image_features``): the RADIO tower encodes
each image's patches, its features are layer-normalized, pixel-shuffled by ``downsample_ratio`` and projected into
the language model, and each image's embeddings replace its run of ``image_token_id`` tokens. The language model is
Arctic's Nemotron-H, so its experts shard for expert parallelism and its Mamba layers keep packed rows apart.

The tower, its final norm and the projector sit together under ``model.visual``, the module the VLM registry names as
the vision encoder, so adapters and freezing treat the whole image path as one.

Images arrive in ``media.py``'s layout: ``pixel_values`` patches with one ``image_grid_thw`` row ``[1, h, w]`` per
image (the processor's ``image_grid_hw`` with a leading 1). Video is refused: its frames share the image
placeholder and transformers prunes them inside the forward, so their tokens cannot be placed beforehand.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch import nn
from transformers import AutoModel
from transformers.generation import GenerationMixin
from transformers.models.nemotron_h_omni.configuration_nemotron_h_omni import NemotronH_Omni_Reasoning_V3_Config
from transformers.models.nemotron_h_omni.modeling_nemotron_h_omni import NemotronH_Omni_Reasoning_V3
from transformers.models.nemotron_h_omni.modeling_nemotron_h_omni import NemotronH_Omni_Reasoning_V3MultiModalProjector

from arctic_platform.model.implementations.moe.layers.lm_head import PrimeLmOutput

from .configuration_nemotron_h import NemotronHConfig
from .converting_nemotron_h_omni import DROPPED_PREFIXES
from .converting_nemotron_h_omni import convert_checkpoint_to_prime
from .converting_nemotron_h_omni import convert_prime_to_checkpoint
from .converting_nemotron_h_omni import is_checkpoint_state_dict
from .converting_nemotron_h_omni import is_prime_state_dict
from .modeling_nemotron_h import NemotronHModel
from .modeling_nemotron_h import NemotronHPreTrainedModel

_VIDEO_KEYS = ("pixel_values_videos", "video_grid_thw", "video_row_ids")


def _language_model_config(config: NemotronH_Omni_Reasoning_V3_Config) -> NemotronHConfig:
    """The composite's ``text_config``, which must be Arctic's Nemotron-H config, with the attention backend."""
    text_config = config.text_config
    if not isinstance(text_config, NemotronHConfig):
        # transformers' own config renames the layer types and Mamba fields Arctic's layers read.
        raise TypeError(
            f"nemotron_h_omni text_config is {type(text_config).__name__}; load the config after importing "
            "arctic_platform.model so AutoConfig builds Arctic's NemotronHConfig for 'nemotron_h'"
        )
    if getattr(text_config, "_attn_implementation", None) is None:
        text_config._attn_implementation = getattr(config, "_attn_implementation", None)
    # The model builder sets this on the composite config; the language model's layers read their own.
    if hasattr(config, "use_grouped_mm"):
        text_config.use_grouped_mm = config.use_grouped_mm
    return text_config


class NemotronHOmniVisionPrimeRL(nn.Module):
    """RADIO tower, its final norm and the projector: images to language-model embeddings."""

    # transformers' image path, which reads exactly these attribute names.
    get_image_features = NemotronH_Omni_Reasoning_V3.get_image_features
    pixel_shuffle = NemotronH_Omni_Reasoning_V3.pixel_shuffle

    def __init__(self, config: NemotronH_Omni_Reasoning_V3_Config):
        super().__init__()
        self.config = config
        self.vision_model = AutoModel.from_config(config.vision_config)
        self.vision_model.make_preprocessor_external()
        # Checkpoint state a complete export writes back (weight sync leaves buffers out).
        self.vision_model._dss_weight_export_buffers = frozenset({"summary_idxs"})
        if float(config.vision_config.layerscale_value) == 1.0:
            # C-RADIO has no LayerScale: transformers keeps an identity one, which no checkpoint stores.
            for layer in self.vision_model.encoder.layer:
                layer.layer_scale1 = nn.Identity()
                layer.layer_scale2 = nn.Identity()
        video_embedder = getattr(self.vision_model.embeddings, "video_patch_projection", None)
        if video_embedder is not None:
            # Images never reach it: trainable, it would be a parameter no backward ever gives a gradient.
            video_embedder.requires_grad_(False)
        self.vision_final_layernorm = (
            nn.LayerNorm(config.vision_hidden_size, eps=config.vision_config.layer_norm_eps)
            if config.text_config.num_nextn_predict_layers > 0
            else None
        )
        self.multi_modal_projector = NemotronH_Omni_Reasoning_V3MultiModalProjector(
            config.vision_hidden_size * int(1 / config.downsample_ratio) ** 2,
            config.projector_hidden_size,
            config.text_config.hidden_size,
        )

    def forward(self, pixel_values: Tensor, image_grid_hw: Tensor) -> Tensor:
        """Every image's embeddings, concatenated in order: ``[sum(h * w) / merge**2, hidden]``."""
        return torch.cat(self.get_image_features(pixel_values, image_grid_hw, return_dict=True).pooler_output)

    def dummy_forward(self) -> Tensor:
        """The image path on its smallest input: one image of one pixel-shuffle window of zero patches."""
        vision = self.config.vision_config
        merge = int(round(1 / self.config.downsample_ratio))
        reference = self.vision_model.embeddings.patch_projection.weight
        pixel_values = torch.zeros(
            merge * merge, vision.num_channels * vision.patch_size**2, dtype=reference.dtype, device=reference.device
        )
        return self(pixel_values, torch.tensor([[merge, merge]], dtype=torch.long, device=reference.device))


class NemotronHOmniModelPrimeRL(nn.Module):
    """The image path (``visual``) beside Arctic's Nemotron-H language model."""

    def __init__(self, config: NemotronH_Omni_Reasoning_V3_Config):
        super().__init__()
        self.config = config
        self.image_token_id = int(config.image_token_id)
        self.visual = NemotronHOmniVisionPrimeRL(config)
        self.language_model = NemotronHModel(_language_model_config(config))

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[Tensor] = None,
        image_grid_thw: Optional[Tensor] = None,
        image_grid_hw: Optional[Tensor] = None,
        routed_experts: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        videos = sorted(key for key in _VIDEO_KEYS if kwargs.get(key) is not None)
        if videos:
            raise NotImplementedError(f"Nemotron-H Omni training takes images only, not {videos}")
        grid_hw = self._image_grid_hw(image_grid_thw, image_grid_hw)
        if (pixel_values is None) != (grid_hw is None):
            raise ValueError("pixel_values and the image grid must be passed together")
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
        if pixel_values is not None:
            if input_ids is None:
                raise ValueError("image inputs need input_ids to locate the image tokens")
            image_embeds = self.visual(pixel_values, grid_hw).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = input_ids == self.image_token_id
            image_tokens = int(image_mask.sum())
            if image_tokens != image_embeds.shape[0]:
                raise ValueError(
                    f"input_ids hold {image_tokens} image tokens but the images encode to {image_embeds.shape[0]}"
                )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(inputs_embeds), image_embeds
            )
        elif self._needs_dummy_vision_forward():
            # A trainable vision tower must take part in every microbatch's backward so every rank produces (zero)
            # gradients for it and ZeRO's bucketed reduction stays in step. It runs after the embedding, where the
            # real image path runs, so gradients become ready in the same order with or without images.
            inputs_embeds = inputs_embeds + self.visual.dummy_forward().sum().to(inputs_embeds.dtype) * 0
        return self.language_model(
            inputs_embeds=inputs_embeds, position_ids=position_ids, routed_experts=routed_experts
        )

    @staticmethod
    def _image_grid_hw(image_grid_thw: Optional[Tensor], image_grid_hw: Optional[Tensor]) -> Optional[Tensor]:
        if image_grid_thw is None:
            return image_grid_hw
        if image_grid_hw is not None:
            raise ValueError("pass image_grid_thw or image_grid_hw, not both")
        if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3 or bool((image_grid_thw[:, 0] != 1).any()):
            raise ValueError(
                f"Nemotron-H Omni images take image_grid_thw rows [1, h, w], got {image_grid_thw.tolist()}"
            )
        return image_grid_thw[:, 1:]

    def _needs_dummy_vision_forward(self) -> bool:
        if not (self.training and torch.is_grad_enabled()):
            return False
        return any(parameter.requires_grad for parameter in self.visual.parameters())


class NemotronHOmniForConditionalGenerationPrimeRL(NemotronHPreTrainedModel, GenerationMixin):
    """Nemotron-H Omni with Arctic's LM head contract; ``model`` is ``NemotronHOmniModelPrimeRL``."""

    config_class = NemotronH_Omni_Reasoning_V3_Config
    config: NemotronH_Omni_Reasoning_V3_Config
    _tied_weights_keys: dict[str, str] = {}

    def __init__(self, config: NemotronH_Omni_Reasoning_V3_Config, **_kwargs):
        if getattr(config, "quantization_config", None):
            raise NotImplementedError("Nemotron-H Omni training supports the BF16 checkpoint only")
        # Read off the top-level config (by Nemotron-H's _init_weights and the weight loader); the composite config
        # carries them only in text_config unless its config.json repeats them.
        if getattr(config, "initializer_range", None) is None:
            config.initializer_range = config.text_config.initializer_range
        try:
            tied = config.tie_word_embeddings
        except AttributeError:
            tied = None
        if tied is None:
            config.tie_word_embeddings = bool(getattr(config.text_config, "tie_word_embeddings", False))
        if config.tie_word_embeddings:
            raise NotImplementedError("Nemotron-H Omni has an untied lm_head")
        super().__init__(config)
        self.model = NemotronHOmniModelPrimeRL(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self._is_vlm = True
        # vLLM loads this model from checkpoint names (vision tower and projector included), not Arctic's.
        self._requires_hf_weight_sync = True
        self.post_init()

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        temperature: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> PrimeLmOutput:
        if position_ids is None:
            reference = inputs_embeds if inputs_embeds is not None else input_ids
            assert reference is not None, "pass input_ids or inputs_embeds"
            position_ids = torch.arange(reference.shape[1], device=reference.device).unsqueeze(0)
        outputs = self.model(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, **kwargs)
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
        )

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_checkpoint_state_dict(state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_prime_state_dict(state_dict)

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return convert_checkpoint_to_prime(state_dict)

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return convert_prime_to_checkpoint(state_dict)

    # The conversions take any subset of the weights, so one decoder layer's (or the rest, at -1) converts alone.
    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return convert_prime_to_checkpoint(state_dict)

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return convert_checkpoint_to_prime(state_dict)

    def iter_hf_checkpoint_passthrough(self) -> Iterator[tuple[str, Tensor]]:
        """The source checkpoint's tensors this model does not hold (MTP head, input conditioner), unchanged.

        A complete export carries them, so it loads as the checkpoint did, MTP speculative decoding included.
        """
        from safetensors import safe_open

        source = getattr(self, "hf_source_path", None)
        if source is None:
            raise RuntimeError(
                f"{type(self).__name__} was not loaded from a checkpoint, so a complete export cannot carry its "
                "MTP head and input conditioner"
            )
        files = {}
        for path in sorted(Path(source).glob("*.safetensors")):
            with safe_open(path, framework="pt") as handle:
                files.update({name: path for name in handle.keys() if name.startswith(DROPPED_PREFIXES)})
        for name in sorted(files):
            with safe_open(files[name], framework="pt") as handle:
                yield name, handle.get_tensor(name)

    def init_buffers_post_meta(self):
        pass


__all__ = [
    "NemotronHOmniForConditionalGenerationPrimeRL",
    "NemotronHOmniModelPrimeRL",
    "NemotronHOmniVisionPrimeRL",
]
