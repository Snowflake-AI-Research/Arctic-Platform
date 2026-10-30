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
"""In-tree inference rollout configuration."""

import pytest

from arctic_platform.common.utils.server_models import build_model_config

pytest.importorskip("vllm")


def test_in_tree_model_config_maps_fca_and_speculative_decoding():
    config = build_model_config(
        "Qwen/Qwen3-1.7B",
        {},
        arctic_inference_config={
            "zorro_inference": {"enable": True},
            "speculative_decoding": {
                "model": "Qwen/Qwen3-0.6B",
                "num_speculative_tokens": 5,
            },
        },
    )

    assert config.forest_cascade_attn_configs == "{}"
    assert config.speculative_model == "Qwen/Qwen3-0.6B"
    assert config.speculative_config == {
        "method": "arctic",
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": 5,
    }


def test_in_tree_model_config_omits_disabled_rollout_features():
    config = build_model_config(
        "Qwen/Qwen3-1.7B",
        {},
        arctic_inference_config={
            "zorro_inference": {"enable": False},
            "speculative_decoding": {"model": ""},
        },
    )

    assert config.speculative_model is None
    assert config.speculative_config is None
