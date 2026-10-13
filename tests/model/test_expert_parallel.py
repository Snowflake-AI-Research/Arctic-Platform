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

"""Expert-parallel validation and the sequence-parallel-1 group setup."""

from __future__ import annotations

import pytest

from arctic_platform.model import expert_parallel as expert_parallel_mod


def test_ep_group_name_matches_the_deepspeed_registry_key():
    assert expert_parallel_mod.ep_group_name(8) == "ep_size_8"


@pytest.mark.parametrize(
    ("ep_size", "world_size"),
    [(1, 8), (8, 8), (4, 16), (16, 16)],
)
def test_validate_expert_parallel_accepts_divisors(ep_size, world_size):
    expert_parallel_mod.validate_expert_parallel(ep_size, world_size)


def test_validate_expert_parallel_rejects_a_degree_that_does_not_divide_the_world():
    with pytest.raises(ValueError, match="must divide world_size"):
        expert_parallel_mod.validate_expert_parallel(8, 12)


def test_validate_expert_parallel_rejects_zero_stage_3():
    with pytest.raises(ValueError, match="ZeRO stage 1 or 2"):
        expert_parallel_mod.validate_expert_parallel(8, 8, zero_stage=3)


def test_disabled_expert_parallel_skips_the_other_checks():
    expert_parallel_mod.validate_expert_parallel(1, 8, zero_stage=3)


def test_replicated_expert_groups_register_only_when_expert_parallel_is_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        expert_parallel_mod,
        "_create_expert_and_data_parallel",
        lambda ep_size: calls.append(ep_size),
    )

    expert_parallel_mod.setup_replicated_expert_groups(1)
    expert_parallel_mod.setup_replicated_expert_groups(4)

    assert calls == [4]


def test_expert_parallel_group_uses_the_registry_key(monkeypatch):
    seen = []
    monkeypatch.setattr(
        expert_parallel_mod,
        "_expert_parallel_group",
        lambda group_name: seen.append(group_name) or group_name,
    )

    assert expert_parallel_mod.expert_parallel_group(8) == "ep_size_8"
    assert seen == ["ep_size_8"]


def test_split_expert_param_groups_names_groups_before_the_deepspeed_split(monkeypatch):
    seen = {}

    def configure(groups):
        seen["groups"] = groups
        return [{"name": "experts", "params": groups[0]["params"]}]

    monkeypatch.setattr(
        "deepspeed.moe.utils.configure_moe_param_groups",
        configure,
    )
    params = [object()]
    result = expert_parallel_mod.split_expert_param_groups([{"params": params}])

    assert seen["groups"] == [{"params": params, "name": "group_0"}]
    assert result == [{"name": "experts", "params": params}]
