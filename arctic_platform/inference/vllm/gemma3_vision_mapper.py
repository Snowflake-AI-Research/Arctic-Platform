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

"""Map Transformers 5 Gemma 3 vision weights onto vLLM's Siglip wrapper.

Transformers 5 flattens ``SiglipVisionModel``. Trainer parameters are named
``model.vision_tower.embeddings.*``. vLLM still stores those tensors at
``vision_tower.vision_model.embeddings.*``. Checkpoints saved before that
flattening already contain ``vision_model`` and must keep the stock prefix.
"""

from __future__ import annotations

import re
from typing import Any

_FLAT_VISION_TOWER = re.compile(r"^model\.vision_tower\.(?!vision_model\.)")
_STOCK_VISION_PREFIX = "vision_tower."


def install_gemma3_flat_vision_mapper(mapper: Any) -> bool:
    """Insert ``vision_model`` for flattened Gemma 3 vision keys.

    Returns True when the regex was added. A mapper whose vision prefix is
    already rewritten, or which already has this regex, is left alone.
    """
    prefixes = getattr(mapper, "orig_to_new_prefix", None) or {}
    if prefixes.get("model.vision_tower.") != _STOCK_VISION_PREFIX:
        return False
    regexes = dict(getattr(mapper, "orig_to_new_regex", None) or {})
    if any(getattr(pattern, "pattern", None) == _FLAT_VISION_TOWER.pattern for pattern in regexes):
        return False
    regexes[_FLAT_VISION_TOWER] = "vision_tower.vision_model."
    mapper.orig_to_new_regex = regexes
    return True


def ensure_gemma3_vision_mapper_patch() -> None:
    from vllm.model_executor.models.gemma3_mm import Gemma3ForConditionalGeneration

    install_gemma3_flat_vision_mapper(Gemma3ForConditionalGeneration.hf_to_vllm_mapper)
