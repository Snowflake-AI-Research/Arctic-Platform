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

"""``{"device": "none"}`` must not count as CPU optimizer offload."""

import pytest

from arctic_platform.client.config import _without_noop_offload
from arctic_platform.common.deepspeed_engine import _optimizer_cpu_offload_enabled


def _cfg(offload):
    return {"zero_optimization": {"stage": 2, "offload_optimizer": offload}}


def _stripper_keeps_optimizer_offload(ds_config: dict) -> bool:
    zero = _without_noop_offload(ds_config).get("zero_optimization") or {}
    return "offload_optimizer" in zero


@pytest.mark.parametrize(
    "offload",
    [
        {"device": "none"},
        {"device": "None"},
        {"device": "NONE"},
        {"device": ""},
        {},
        {"pin_memory": True},
    ],
)
def test_device_none_or_missing_is_offload_off(offload):
    assert _optimizer_cpu_offload_enabled(_cfg(offload)) is False


@pytest.mark.parametrize("offload", [{"device": "cpu"}, {"device": "cpu", "pin_memory": False}])
def test_real_cpu_offload_is_on(offload):
    assert _optimizer_cpu_offload_enabled(_cfg(offload)) is True


@pytest.mark.parametrize("offload", [None, False, ""])
def test_falsy_offload_is_off(offload):
    assert _optimizer_cpu_offload_enabled(_cfg(offload)) is False


def test_missing_zero_block_is_off():
    assert _optimizer_cpu_offload_enabled({}) is False


@pytest.mark.parametrize(
    "offload, enabled",
    [
        ({}, False),
        ({"pin_memory": True}, False),
        ({"device": "none"}, False),
        ({"device": "cpu"}, True),
    ],
)
def test_engine_and_client_stripper_agree(offload, enabled):
    ds_config = _cfg(offload)
    assert _optimizer_cpu_offload_enabled(ds_config) is enabled
    assert _stripper_keeps_optimizer_offload(ds_config) is enabled
