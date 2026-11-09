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
"""Draft configuration for the vendored SpecForge DFlash2 model (written here, not copied from SpecForge).

``build_draft_config`` follows ``AutoDraftModelConfig.from_file`` in SpecForge and adds checks for the settings
that make a trained draft behave differently once it is served.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from transformers import PretrainedConfig

from .specforge.modeling.draft.dflash2 import DFlash2DraftModel

logger = logging.getLogger(__name__)

ARCHITECTURE = "DFlash2DraftModel"
_REQUIRED = (
    "block_size",
    "target_layer_ids",
    "mask_token_id",
    "conv_kernel_size",
    "conv_group_size",
    "selector_rank",
    "selector_top_k",
)


def build_draft_config(raw: dict[str, Any]) -> PretrainedConfig:
    raw = copy.deepcopy(dict(raw))
    architectures = raw.get("architectures") or []
    if architectures != [ARCHITECTURE]:
        raise ValueError(f"draft config architectures must be [{ARCHITECTURE!r}], got {architectures}")

    settings = raw.get("dflash_config") or {}
    missing = [key for key in _REQUIRED if settings.get(key) is None]
    if missing:
        raise ValueError(f"DFlash2 dflash_config is missing {missing}")
    if int(settings["block_size"]) < 2:
        raise ValueError(f"DFlash2 block_size must be >= 2 (an anchor plus a proposal), got {settings['block_size']}")
    layer_ids = [int(i) for i in settings["target_layer_ids"]]
    if not layer_ids or layer_ids != sorted(set(layer_ids)) or layer_ids[0] < 0:
        raise ValueError(
            "DFlash2 target_layer_ids must be non-empty, non-negative and strictly ascending (the serving stack "
            f"concatenates the captured layers in ascending order), got {layer_ids}"
        )
    num_target_layers = raw.get("num_target_layers")
    if num_target_layers is not None and layer_ids[-1] >= int(num_target_layers) - 1:
        # Layer L means the output of decoder layer L. For the last layer HF returns the post-final-norm state, while
        # a serving runtime captures the residual stream before that norm.
        raise ValueError(
            f"DFlash2 target_layer_ids {layer_ids} must stay below the target's last layer "
            f"(num_target_layers={num_target_layers}): that layer's output is post-norm in training but pre-norm "
            "when captured for serving"
        )
    if "sliding_attention" in (raw.get("layer_types") or []) and raw.get("is_causal") is False:
        # Training attends causally inside a block on sliding-window layers and never reads ``is_causal``; a serving
        # runtime that finds an explicit ``false`` makes those layers bidirectional instead.
        logger.warning("dropping is_causal=false from a sliding-window draft config so serving matches training")
        raw.pop("is_causal")

    # as AutoDraftModelConfig.from_file in SpecForge
    if "tie_word_embeddings" in raw:
        raw["tie_word_embeddings"] = False
    if raw.get("draft_vocab_size") is None:
        raw["draft_vocab_size"] = raw.get("vocab_size")
    return DFlash2DraftModel.config_class.from_dict(raw)
