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

"""Compatibility re-export. The EngineCore extension lives in Arctic-Inference."""

from __future__ import annotations

from arctic_inference.server.weight_sync.receiver import TextOnlyWeightSyncExtension
from arctic_inference.server.weight_sync.receiver import WeightSyncExtension

WORKER_EXTENSION_CLS = "arctic_inference.server.weight_sync.WeightSyncExtension"
WEIGHT_SYNC_POLICY = "text_only"

__all__ = [
    "TextOnlyWeightSyncExtension",
    "WeightSyncExtension",
    "WORKER_EXTENSION_CLS",
    "WEIGHT_SYNC_POLICY",
]
