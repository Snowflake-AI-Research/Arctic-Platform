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
# See the License for the specific language governing permissions and
# limitations under the License.

"""vLLM worker extension loaded inside EngineCore for weight-sync name checks.

Monkey-patching ``WeightSyncExtension`` in the HTTP/Ray parent process does not
reach the EngineCore subprocess. Register this class via ``worker_extension_cls``
so the optional vision/MTP filter runs where names are validated.
"""

from __future__ import annotations

from arctic_inference.server.weight_sync.receiver import WeightSyncExtension
from arctic_platform.model.implementations.qwen35.hf_vllm_weight_sync import (
    expected_hf_names_for_text_sync,
)

WORKER_EXTENSION_CLS = (
    "arctic_platform.common.weight_sync_extension.TextOnlyWeightSyncExtension"
)


class TextOnlyWeightSyncExtension(WeightSyncExtension):
    """Same as arctic-inference's extension, but missing ``visual.*`` / ``mtp.*`` are ok."""

    def _validate_weight_sync_names(self, model, sender_names, *, context: str = ""):
        from arctic_inference.server.weight_sync import utils as ws_utils

        orig_compute = ws_utils.compute_expected_hf_param_names
        sender_set = {n for n in sender_names if not ws_utils._name_is_non_synced(n)}

        def _compute_expected(module):
            return expected_hf_names_for_text_sync(orig_compute(module), sender_set)

        ws_utils.compute_expected_hf_param_names = _compute_expected
        try:
            return super()._validate_weight_sync_names(model, sender_names, context=context)
        finally:
            ws_utils.compute_expected_hf_param_names = orig_compute
