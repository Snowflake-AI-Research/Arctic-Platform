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
"""Hugging Face model preparation for sequence parallelism."""

from __future__ import annotations

from arctic_platform.model.implementations.gpu.sp.gated_delta_net import apply_gated_delta_net_sequence_parallelism


def configure_transformers_sequence_parallel_model(model, process_group) -> int:
    model_config = getattr(model, "config", None)
    get_text_config = getattr(model_config, "get_text_config", None)
    text_config = get_text_config() if callable(get_text_config) else model_config
    for config in (model_config, text_config):
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False

    return apply_gated_delta_net_sequence_parallelism(model, process_group)
