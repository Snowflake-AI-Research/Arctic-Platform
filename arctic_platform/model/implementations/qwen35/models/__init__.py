## Copies AutoModelForCausalLM from transformers but uses our own custom model.
## Qwen3.5-MoE stays optional because its kernels may be absent. The generic
## PrimeRL MoE families register unconditionally.

from collections import OrderedDict
import logging

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig as HFQwen3_5MoeConfig

from arctic_platform.model.implementations.afmoe.configuration_afmoe import AfmoeConfig
from arctic_platform.model.implementations.afmoe.modeling_afmoe import AfmoeForCausalLM
from arctic_platform.model.implementations.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from arctic_platform.model.implementations.glm4_moe.modeling_glm4_moe import Glm4MoeForCausalLM
from arctic_platform.model.implementations.minimax_m2.configuration_minimax_m2 import MiniMaxM2Config
from arctic_platform.model.implementations.minimax_m2.modeling_minimax_m2 import MiniMaxM2ForCausalLM
from arctic_platform.model.implementations.moe.base import PreTrainedModelPrimeRL
from arctic_platform.model.implementations.moe.layers.lm_head import PrimeLmOutput, cast_float_and_contiguous
from arctic_platform.model.implementations.nemotron_h.configuration_nemotron_h import NemotronHConfig
from arctic_platform.model.implementations.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM
from arctic_platform.model.implementations.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from arctic_platform.model.implementations.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

logger = logging.getLogger(__name__)

# Make custom config discoverable by AutoConfig
AutoConfig.register("qwen3_5_moe", HFQwen3_5MoeConfig, exist_ok=True)

_CUSTOM_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, OrderedDict())
_CUSTOM_VLM_MAPPING: dict[str, type] = {}
_QWEN3_5_CUSTOM_IMPL_AVAILABLE = False
_QWEN3_5_CUSTOM_IMPL_IMPORT_ERROR: ImportError | None = None

try:
    from .qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM
except ImportError as exc:
    _QWEN3_5_CUSTOM_IMPL_IMPORT_ERROR = exc
    # An absent GPU kernel is the one failure worth tolerating, and only because CPU environments are expected
    # to lack them. Every other error means this implementation is broken, and the Hugging Face model is not a
    # substitute for it: the fused lm_head, the fp32 projection and the packed-sequence boundary contract all
    # live here, so a job that fell back would report healthy numbers measured on different code.
    logger.warning(
        "qwen3_5 custom implementation unavailable; explicit custom model loads will fail: %r",
        exc,
    )
else:
    _QWEN3_5_CUSTOM_IMPL_AVAILABLE = True
    AutoConfig.register("qwen3_5_moe_text", Qwen3_5MoeConfig, exist_ok=True)
    _CUSTOM_CAUSAL_LM_MAPPING.register(Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM, exist_ok=True)
    _CUSTOM_VLM_MAPPING["qwen3_5_moe"] = Qwen3_5MoeForCausalLM


def _register_causal_lm(config_cls: type, model_cls: type, model_type: str) -> None:
    AutoConfig.register(model_type, config_cls, exist_ok=True)
    _CUSTOM_CAUSAL_LM_MAPPING.register(config_cls, model_cls, exist_ok=True)


for _config_cls, _model_cls, _model_type in (
    (Qwen3MoeConfig, Qwen3MoeForCausalLM, "qwen3_moe"),
    (Glm4MoeConfig, Glm4MoeForCausalLM, "glm4_moe"),
    (MiniMaxM2Config, MiniMaxM2ForCausalLM, "minimax_m2"),
    (AfmoeConfig, AfmoeForCausalLM, "afmoe"),
    (NemotronHConfig, NemotronHForCausalLM, "nemotron_h"),
):
    _register_causal_lm(_config_cls, _model_cls, _model_type)


def _register_flash_vlm(module: str, class_name: str, model_type: str) -> None:
    """Register a transformers-backed VLM when that upstream module is installed."""
    import importlib

    try:
        model_cls = getattr(importlib.import_module(module), class_name)
    except ImportError as exc:
        logger.warning("%s custom implementation unavailable: %r", model_type, exc)
        return
    _CUSTOM_VLM_MAPPING[model_type] = model_cls


_register_flash_vlm(
    "arctic_platform.model.implementations.glm53.modeling_glm5_next",
    "Glm5NextForConditionalGenerationPrimeRL",
    "glm5_next",
)
_register_flash_vlm(
    "arctic_platform.model.implementations.qwen38.modeling_qwen4_exp",
    "Qwen4ExpForConditionalGenerationPrimeRL",
    "qwen4_exp",
)


class AutoModelForCausalLMPrimeRL(_BaseAutoModelClass):
    _model_mapping = _CUSTOM_CAUSAL_LM_MAPPING


AutoModelForCausalLMPrimeRL = auto_class_update(AutoModelForCausalLMPrimeRL, head_doc="causal language modeling")


def supports_custom_impl(model_config: PretrainedConfig) -> bool:
    """Check if the model configuration supports the custom PrimeRL implementation."""
    return type(model_config) in _CUSTOM_CAUSAL_LM_MAPPING


def get_custom_vlm_cls(model_config: PretrainedConfig) -> type | None:
    """Return the custom PrimeRL VLM class for this config, or None if unsupported."""
    return _CUSTOM_VLM_MAPPING.get(getattr(model_config, "model_type", None))


def get_custom_impl_import_error() -> ImportError | None:
    """Return the import failure that made the custom implementation unavailable."""
    return _QWEN3_5_CUSTOM_IMPL_IMPORT_ERROR


__all__ = [
    "AutoModelForCausalLMPrimeRL",
    "PreTrainedModelPrimeRL",
    "supports_custom_impl",
    "get_custom_vlm_cls",
    "get_custom_impl_import_error",
    "PrimeLmOutput",
    "cast_float_and_contiguous",
]
