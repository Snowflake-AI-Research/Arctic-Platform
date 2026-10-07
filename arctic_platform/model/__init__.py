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
"""Model factory: turn a declarative ModelSpec into a configured nn.Module."""

from arctic_platform._dependency_groups import require_any_dep_group

require_any_dep_group("sft", "rl")

from arctic_platform.model.config import ActivationCheckpointConfig
from arctic_platform.model.config import ActivationOffloadConfig
from arctic_platform.model.config import ActivationOffloadPatch
from arctic_platform.model.config import CompilePatch
from arctic_platform.model.config import LmHeadPatch
from arctic_platform.model.config import ModelSpec
from arctic_platform.model.config import ParallelismConfig
from arctic_platform.model.config import Patches
from arctic_platform.model.config import TiledMlpPatch
from arctic_platform.model.config import ZorroTrainPatch
from arctic_platform.model.factory import build_model
from arctic_platform.model.implementations.moe.config_validation import effective_fused_cross_entropy
from arctic_platform.model.implementations.moe.config_validation import validate_lm_head_fused_ce_config
from arctic_platform.model.implementations.moe.conversion_cache import conversion_cache_is_node_local
from arctic_platform.model.implementations.moe.conversion_cache import conversion_cache_ready
from arctic_platform.model.implementations.moe.conversion_cache import resolve_conversion_cache_path
from arctic_platform.model.implementations.moe.vlm import get_language_model
from arctic_platform.model.implementations.moe.vlm import get_vision_encoder
from arctic_platform.model.loader import LoadedModel
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.loader import ModelParallelismMetadata
from arctic_platform.model.loader import canonical_parameter_name
from arctic_platform.model.loader import finalize_model_for_training
from arctic_platform.model.loader import model_parallelism_metadata_from_config
from arctic_platform.model.loader import register_loader
from arctic_platform.model.loader import select_loader
from arctic_platform.model.patch import apply_patches
from arctic_platform.model.patch import register_patch
from arctic_platform.model.patches.peft import apply_peft
from arctic_platform.model.weight_export import PEFT_ADAPTER_DIRNAME
from arctic_platform.model.weight_export import WeightExportContract
from arctic_platform.model.weight_export import checkpoint_peft_adapter_dir
from arctic_platform.model.weight_export import gather_peft_adapter_state_dict
from arctic_platform.model.weight_export import hf_export_parameter_name
from arctic_platform.model.weight_export import iter_lora_weights
from arctic_platform.model.weight_export import iter_model_weights
from arctic_platform.model.weight_export import pretrained_config_of
from arctic_platform.model.weight_export import pretrained_module_for_hf_save
from arctic_platform.model.weight_export import save_exported_state_dict
from arctic_platform.model.weight_export import save_hf_pretrained
from arctic_platform.model.weight_export import save_peft_adapters
from arctic_platform.model.weight_export import supports_weight_format
from arctic_platform.model.weight_export import validate_lora_sync_trainable_parameters
from arctic_platform.model.weight_export import weight_export_contract

# Import built-in loaders and patches for their registration side effects.
from arctic_platform.model import loaders  # noqa: F401  # isort: skip
from arctic_platform.model import patches  # noqa: F401  # isort: skip

__all__ = [
    "ActivationCheckpointConfig",
    "ActivationOffloadConfig",
    "ActivationOffloadPatch",
    "PEFT_ADAPTER_DIRNAME",
    "WeightExportContract",
    "canonical_parameter_name",
    "checkpoint_peft_adapter_dir",
    "conversion_cache_is_node_local",
    "conversion_cache_ready",
    "effective_fused_cross_entropy",
    "finalize_model_for_training",
    "gather_peft_adapter_state_dict",
    "get_language_model",
    "get_vision_encoder",
    "hf_export_parameter_name",
    "iter_lora_weights",
    "iter_model_weights",
    "LoadedModel",
    "LoaderContext",
    "LmHeadPatch",
    "CompilePatch",
    "ModelParallelismMetadata",
    "ModelSpec",
    "model_parallelism_metadata_from_config",
    "ParallelismConfig",
    "Patches",
    "pretrained_config_of",
    "pretrained_module_for_hf_save",
    "save_hf_pretrained",
    "save_peft_adapters",
    "supports_weight_format",
    "TiledMlpPatch",
    "ZorroTrainPatch",
    "apply_patches",
    "apply_peft",
    "build_model",
    "register_loader",
    "register_patch",
    "resolve_conversion_cache_path",
    "save_exported_state_dict",
    "select_loader",
    "validate_lora_sync_trainable_parameters",
    "validate_lm_head_fused_ce_config",
    "weight_export_contract",
]
