## Copies AutoModelForCausalLM from transformers but uses our own custom model.
## Slimmed to register ONLY the qwen3_5_moe family (the other prime-rl model
## families are intentionally not carved out).

from collections import OrderedDict

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig as HFQwen3_5MoeConfig

from .base import PreTrainedModelPrimeRL
from .layers.lm_head import PrimeLmOutput, cast_float_and_contiguous
from .qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM

# Make custom config discoverable by AutoConfig
AutoConfig.register("qwen3_5_moe", HFQwen3_5MoeConfig, exist_ok=True)
AutoConfig.register("qwen3_5_moe_text", Qwen3_5MoeConfig, exist_ok=True)

_CUSTOM_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, OrderedDict())
_CUSTOM_CAUSAL_LM_MAPPING.register(Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM, exist_ok=True)


class AutoModelForCausalLMPrimeRL(_BaseAutoModelClass):
    _model_mapping = _CUSTOM_CAUSAL_LM_MAPPING


AutoModelForCausalLMPrimeRL = auto_class_update(AutoModelForCausalLMPrimeRL, head_doc="causal language modeling")


def supports_custom_impl(model_config: PretrainedConfig) -> bool:
    """Check if the model configuration supports the custom PrimeRL implementation."""
    return type(model_config) in _CUSTOM_CAUSAL_LM_MAPPING


# Mapping from HF composite VLM model_type to custom PrimeRL class.
_CUSTOM_VLM_MAPPING: dict[str, type] = {
    "qwen3_5_moe": Qwen3_5MoeForCausalLM,
}


def get_custom_vlm_cls(model_config: PretrainedConfig) -> type | None:
    """Return the custom PrimeRL VLM class for this config, or None if unsupported."""
    return _CUSTOM_VLM_MAPPING.get(getattr(model_config, "model_type", None))


__all__ = [
    "AutoModelForCausalLMPrimeRL",
    "PreTrainedModelPrimeRL",
    "supports_custom_impl",
    "get_custom_vlm_cls",
    "PrimeLmOutput",
    "cast_float_and_contiguous",
]
