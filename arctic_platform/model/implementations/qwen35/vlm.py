"""Vision-Language Model (VLM) support utilities.

Central registry for VLM model families. On the DSS text-only path only
``get_language_model`` is exercised (it falls back to ``model.model``), but the
registry and helpers are kept for parity with prime-rl.
"""

from dataclasses import dataclass

import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig


@dataclass(frozen=True)
class VLMModelInfo:
    """Per-model-family VLM architecture metadata."""

    vision_encoder_attr: str
    language_model_attr: str


# Central registry: model_type -> architecture info.
VLM_REGISTRY: dict[str, VLMModelInfo] = {
    "qwen3_vl": VLMModelInfo(vision_encoder_attr="model.visual", language_model_attr="model.language_model"),
    "qwen3_5": VLMModelInfo(vision_encoder_attr="model.visual", language_model_attr="model.language_model"),
    "qwen3_5_moe": VLMModelInfo(vision_encoder_attr="model.visual", language_model_attr="model.language_model"),
    "qwen3_vl_moe": VLMModelInfo(vision_encoder_attr="model.visual", language_model_attr="model.language_model"),
}

# Text-only default
DEFAULT_LAYER_PREFIX = "model.layers."


def get_vision_encoder(model: nn.Module, override: str | None = None) -> nn.Module | None:
    """Get the vision encoder module.

    Checks: config override -> registry. Returns None if not found.
    Raises ValueError on a bad config override.
    """
    if override is not None:
        result = _resolve_attr(model, override)
        if result is None:
            raise ValueError(f"vlm.vision_encoder_attr='{override}' does not resolve on this model")
        return result

    info = _get_model_info(model)
    if info is not None:
        return _resolve_attr(model, info.vision_encoder_attr)

    return None


def get_language_model(model: nn.Module, override: str | None = None) -> nn.Module:
    """Get the language model module (the part with transformer layers).

    Checks: config override -> registry -> model.model (text-only default).
    Raises ValueError on a bad config override.
    """
    if override is not None:
        result = _resolve_attr(model, override)
        if result is None:
            raise ValueError(f"vlm.language_model_attr='{override}' does not resolve on this model")
        return result

    info = _get_model_info(model)
    if info is not None:
        result = _resolve_attr(model, info.language_model_attr)
        if result is not None:
            return result

    # Text-only models: language model is directly at model.model
    return model.model


def is_vlm_architecture(model_config: PretrainedConfig) -> bool:
    """Check if the model config belongs to a known VLM architecture."""
    return _get_model_info_from_config(model_config) is not None


def get_layer_prefix(model_config: PretrainedConfig, override: str | None = None) -> str:
    """Return the weight key prefix for language model layers."""
    if override is not None:
        return override
    info = _get_model_info_from_config(model_config)
    if info is not None:
        return info.language_model_attr + ".layers."
    return DEFAULT_LAYER_PREFIX


def freeze_unused_vision_tower(model: nn.Module, rank: int = 0) -> int:
    """Freeze a VLM vision tower that a text-only job will never activate.

    Qwen3.5 checkpoints ship a composite config, so ``AutoModelForCausalLM``
    instantiates language model plus ViT even for pure-text distillation.
    Those vision params produce no gradients; ZeRO-3 waits on every trainable
    param at the accumulation boundary and can stall. Returns the number of
    parameters frozen (0 when there is no vision tower).
    """
    vision_encoder = get_vision_encoder(model)
    if vision_encoder is None:
        return 0

    frozen = 0
    for param in vision_encoder.parameters():
        if param.requires_grad:
            param.requires_grad_(False)
            frozen += 1
    return frozen


def _get_model_info(model: nn.Module) -> VLMModelInfo | None:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return VLM_REGISTRY.get(model_type) if model_type else None


def _get_model_info_from_config(model_config: PretrainedConfig) -> VLMModelInfo | None:
    model_type = getattr(model_config, "model_type", None)
    return VLM_REGISTRY.get(model_type) if model_type else None


def _resolve_attr(obj, dotted_path: str):
    """Resolve a dotted attribute path like 'model.visual' on an object."""
    for part in dotted_path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj
