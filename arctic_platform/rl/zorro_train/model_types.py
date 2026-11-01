# Copyright 2026 Snowflake Inc.
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

"""ZoRRo Train model-type allowlist.

The patcher checks a loaded model's ``config.model_type``. Cortex checks the
checkpoint string before the model exists. Both call
:func:`require_supported_model_type`.
"""

from __future__ import annotations

SUPPORTED_MODEL_TYPES = {
    "qwen3",
    "qwen3_moe",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_next",
}

MODEL_TYPE_ALIASES = {
    "qwen3_5_text": "qwen3_5",
    "qwen3_5_moe_text": "qwen3_5_moe",
}

MODEL_TYPE_TO_TEXT_BACKBONE_CLASS = {
    "qwen3": "Qwen3Model",
    "qwen3_moe": "Qwen3MoeModel",
    "qwen3_5": "Qwen3_5TextModel",
    "qwen3_5_moe": "Qwen3_5MoeTextModel",
    "qwen3_next": "Qwen3NextModel",
}


def _normalize_model_type(model_type: str) -> str:
    return MODEL_TYPE_ALIASES.get(model_type, model_type)


def require_supported_model_type(model_type: str) -> str:
    """Return the normalized ZoRRo type, accepting the text-config aliases."""
    normalized = _normalize_model_type(model_type)
    if normalized not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type={model_type}. Supported model types: {sorted(SUPPORTED_MODEL_TYPES)} "
            f"(aliases: {sorted(MODEL_TYPE_ALIASES.keys())})"
        )
    return normalized
