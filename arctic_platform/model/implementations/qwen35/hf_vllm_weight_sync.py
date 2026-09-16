# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governed permissions and
# limitations under the License.

"""Compatibility re-exports. Conversion lives in Arctic-Inference."""

from __future__ import annotations

from arctic_inference.server.weight_sync.adapters.qwen35 import SyncOp as _SyncOp
from arctic_inference.server.weight_sync.adapters.qwen35 import apply_qwen35_sync_op
from arctic_inference.server.weight_sync.adapters.qwen35 import expected_hf_names_for_text_sync
from arctic_inference.server.weight_sync.adapters.qwen35 import has_unpacked_qwen35_gdn
from arctic_inference.server.weight_sync.adapters.qwen35 import is_optional_frozen_vllm_param
from arctic_inference.server.weight_sync.adapters.qwen35 import pack_qwen35_gdn_layer
from arctic_inference.server.weight_sync.adapters.qwen35 import plan_qwen35_vllm_sync
from arctic_inference.server.weight_sync.adapters.qwen35 import to_vllm_param_name
from arctic_inference.server.weight_sync.adapters.qwen35 import to_vllm_sync_weights


def install_optional_frozen_weight_sync_patch() -> None:
    """Deprecated no-op. Name validation lives on Arctic-Inference's extension."""
    return None


__all__ = [
    "_SyncOp",
    "apply_qwen35_sync_op",
    "expected_hf_names_for_text_sync",
    "has_unpacked_qwen35_gdn",
    "install_optional_frozen_weight_sync_patch",
    "is_optional_frozen_vllm_param",
    "pack_qwen35_gdn_layer",
    "plan_qwen35_vllm_sync",
    "to_vllm_param_name",
    "to_vllm_sync_weights",
]
