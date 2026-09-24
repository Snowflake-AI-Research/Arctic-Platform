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
## Copies AutoModelForCausalLM from transformers but uses our own custom model.
## Slimmed to register ONLY the qwen3_5_moe family (the other prime-rl model
## families are intentionally not carved out).

from collections import OrderedDict
import logging

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig as HFQwen3_5MoeConfig

from arctic_platform.model.implementations.moe.base import PreTrainedModelPrimeRL
from arctic_platform.model.implementations.moe.layers.lm_head import PrimeLmOutput, cast_float_and_contiguous

logger = logging.getLogger(__name__)

# Make custom config discoverable by AutoConfig.
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


class AutoModelForCausalLMPrimeRL(_BaseAutoModelClass):
    _model_mapping = _CUSTOM_CAUSAL_LM_MAPPING


AutoModelForCausalLMPrimeRL = auto_class_update(AutoModelForCausalLMPrimeRL, head_doc="causal language modeling")


def supports_custom_impl(model_config: PretrainedConfig) -> bool:
    """Check if the model configuration supports the custom PrimeRL implementation."""
    if _QWEN3_5_CUSTOM_IMPL_AVAILABLE:
        return type(model_config) in _CUSTOM_CAUSAL_LM_MAPPING
    return False


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
