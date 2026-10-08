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

import torch

from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM


def test_hf_conversion_rewrites_prime_router_in_mixed_layout() -> None:
    router = torch.arange(6).reshape(2, 3)
    gate_up = torch.arange(24).reshape(2, 4, 3)
    down = torch.arange(12).reshape(2, 3, 2)
    state_dict = {
        "model.layers.0.mlp.router.gate.weight": router,
        "model.layers.0.mlp.experts.gate_up_proj": gate_up,
        "model.layers.0.mlp.experts.down_proj": down,
    }

    converted = Qwen3_5MoeForCausalLM.convert_to_hf(state_dict)

    assert set(converted) == {
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    }
    assert converted["model.layers.0.mlp.gate.weight"] is router
    assert converted["model.layers.0.mlp.experts.gate_up_proj"] is gate_up
    assert converted["model.layers.0.mlp.experts.down_proj"] is down
