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

"""Model-config features that change how the independent reference must execute."""

from __future__ import annotations

import json
from pathlib import Path


def uses_mixer_packing(model_path: str | Path) -> bool:
    """Whether packed rows need Qwen hybrid-mixer sequence boundaries.

    Qwen3.5/3.6 configs identify GatedDeltaNet layers as ``linear_attention``. The reference reads that
    declaration from the materialized checkpoint rather than inferring it from a model name.
    """
    config = json.loads((Path(model_path) / "config.json").read_text())
    text_config = config.get("text_config", config)
    return "linear_attention" in text_config.get("layer_types", [])
