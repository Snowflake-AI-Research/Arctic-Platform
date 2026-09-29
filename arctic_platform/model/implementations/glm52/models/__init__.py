from collections import OrderedDict

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

from arctic_platform.model.implementations.moe.base import PreTrainedModelPrimeRL
from arctic_platform.model.implementations.moe.layers.lm_head import PrimeLmOutput, cast_float_and_contiguous

from .glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

AutoConfig.register("glm_moe_dsa", GlmMoeDsaConfig, exist_ok=True)

_CUSTOM_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, OrderedDict())
_CUSTOM_CAUSAL_LM_MAPPING.register(GlmMoeDsaConfig, GlmMoeDsaForCausalLM, exist_ok=True)


class AutoModelForCausalLMPrimeRL(_BaseAutoModelClass):
    _model_mapping = _CUSTOM_CAUSAL_LM_MAPPING


AutoModelForCausalLMPrimeRL = auto_class_update(AutoModelForCausalLMPrimeRL, head_doc="causal language modeling")


def supports_custom_impl(model_config: PretrainedConfig) -> bool:
    return type(model_config) in _CUSTOM_CAUSAL_LM_MAPPING


_CUSTOM_VLM_MAPPING: dict[str, type] = {}


def get_custom_vlm_cls(model_config: PretrainedConfig) -> type | None:
    return _CUSTOM_VLM_MAPPING.get(getattr(model_config, "model_type", None))


__all__ = [
    "AutoModelForCausalLMPrimeRL",
    "PreTrainedModelPrimeRL",
    "supports_custom_impl",
    "get_custom_vlm_cls",
    "PrimeLmOutput",
    "cast_float_and_contiguous",
]
