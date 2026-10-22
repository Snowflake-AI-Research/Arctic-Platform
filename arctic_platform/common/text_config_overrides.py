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

from typing import Any

from transformers import AutoConfig
from transformers import PretrainedConfig


def _update_nested(target: PretrainedConfig | dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict):
            nested = target.get(key) if isinstance(target, dict) else getattr(target, key, None)
            if isinstance(nested, (dict, PretrainedConfig)):
                _update_nested(nested, value)
                continue
        if isinstance(target, dict):
            target[key] = value
        else:
            setattr(target, key, value)


def apply_text_config_overrides(config: PretrainedConfig, overrides: dict[str, Any]) -> None:
    _update_nested(config.get_text_config(), overrides)


def text_config_hf_overrides(model: str, overrides: dict[str, Any]) -> dict[str, Any]:
    config = AutoConfig.from_pretrained(model)
    apply_text_config_overrides(config, overrides)
    text_config = config.get_text_config()
    merged = text_config.to_dict()
    values = {key: merged[key] for key in overrides}
    if text_config is config:
        return values
    attribute = next(key for key, value in vars(config).items() if value is text_config)
    return {attribute: values}
