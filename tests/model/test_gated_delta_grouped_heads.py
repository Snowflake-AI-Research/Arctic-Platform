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

"""q and k stay at the key-head count when the gated-delta kernel groups value heads itself."""

import torch

from arctic_platform.model.implementations.gpu.sp.gated_delta_net import kernel_groups_value_heads
from arctic_platform.model.implementations.gpu.sp.gated_delta_net import repeat_query_key_for_value_heads


def _rule(module: str):
    def rule(*_args, **_kwargs):
        return None

    rule.__module__ = module
    return rule


def test_fla_keeps_query_and_key_at_the_key_heads_and_the_torch_fallback_expands_them():
    fla = _rule("fla.ops.gated_delta_rule")
    fallback = _rule("arctic_platform.model.implementations.qwen35.models.qwen3_5_moe.modeling_qwen3_5_moe")
    query = torch.zeros(1, 4, 2, 8)
    key = torch.zeros(1, 4, 2, 8)

    assert kernel_groups_value_heads(fla)
    assert not kernel_groups_value_heads(fallback)
    assert not kernel_groups_value_heads(None)

    fla_query, fla_key = repeat_query_key_for_value_heads(query, key, 2, 6, fla)
    torch_query, torch_key = repeat_query_key_for_value_heads(query, key, 2, 6, fallback)

    assert fla_query.shape[2] == 2 and fla_key.shape[2] == 2
    assert torch_query.shape[2] == 6 and torch_key.shape[2] == 6
