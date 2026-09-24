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
"""Shared configuration validation for clients and model code."""

from __future__ import annotations

from typing import Any


def validate_peft_config(peft_config: dict[str, Any] | None) -> dict[str, Any] | None:
    """None disables PEFT; a supplied config must name its adapter type."""
    if peft_config is not None:
        if not isinstance(peft_config, dict):
            raise ValueError("peft_config must be a dict or None")
        peft_type = peft_config.get("peft_type")
        if not isinstance(peft_type, str) or not peft_type.strip():
            raise ValueError("peft_config.peft_type must be a non-empty string; use None to disable PEFT")
    return peft_config
