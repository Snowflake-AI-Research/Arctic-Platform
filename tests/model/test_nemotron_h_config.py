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

import pytest

from arctic_platform.model.implementations.nemotron_h.configuration_nemotron_h import NemotronHConfig


def test_config_saved_by_transformers_keeps_its_layer_types(tmp_path):
    transformers_config = pytest.importorskip("transformers.models.nemotron_h.configuration_nemotron_h")
    layer_types = ["mamba", "moe", "attention", "moe"]
    transformers_config.NemotronHConfig(
        layers_block_type=layer_types, mtp_layers_block_type=["attention", "moe"]
    ).save_pretrained(tmp_path)

    config = NemotronHConfig.from_pretrained(tmp_path)

    assert config.layers_block_type == layer_types
    assert config.mtp_layers_block_type == ["attention", "moe"]
