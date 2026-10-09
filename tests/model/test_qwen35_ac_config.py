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

"""Config-shape tests for the AP-owned Qwen3.5 activation-checkpoint models.

Covers the CPU-offload knobs living in a nested ``ActivationOffloadConfig`` (with its own ``enabled``
toggle) and the Pydantic coercion that keeps the wire schema (a plain ``ac_config`` dict) working.
"""

import pytest

from arctic_platform.model.config import ActivationCheckpointConfig
from arctic_platform.model.config import ActivationOffloadConfig


def test_defaults_nest_a_default_offload_config():
    ac = ActivationCheckpointConfig()

    assert ac.mode == "full"
    assert ac.freq == 1
    assert isinstance(ac.offload_config, ActivationOffloadConfig)
    assert ac.offload_config.enabled is False
    assert ac.offload_config.keep_last_n == 1
    assert ac.offload_config.use_streams is True
    assert ac.offload_config.tensor_size_threshold == 1 << 20
    assert ac.offload_config.pin_memory_enabled is True
    assert ac.offload_config.pin_memory_max_size_gib == "auto"
    assert ac.offload_config.pin_memory_bucket_size_mib == 64


def test_default_offload_configs_are_not_shared_between_instances():
    a = ActivationCheckpointConfig()
    b = ActivationCheckpointConfig()

    a.offload_config.keep_last_n = 5

    assert b.offload_config.keep_last_n == 1, "each instance must get its own ActivationOffloadConfig"


def test_wire_dict_coerces_nested_offload_config():
    # ac_config arrives as a plain dict (``ActivationCheckpointConfig(**ac_cfg)``); the nested
    # ``offload_config`` must be coerced from dict into the dataclass.
    ac = ActivationCheckpointConfig(
        **{
            "mode": "full",
            "offload_config": {
                "enabled": True,
                "keep_last_n": 2,
                "use_streams": False,
                "tensor_size_threshold": 4096,
                "pin_memory_enabled": False,
                "pin_memory_max_size_gib": 12.5,
                "pin_memory_bucket_size_mib": 128,
            },
        }
    )

    assert isinstance(ac.offload_config, ActivationOffloadConfig)
    assert ac.offload_config.enabled is True
    assert ac.offload_config.keep_last_n == 2
    assert ac.offload_config.use_streams is False
    assert ac.offload_config.tensor_size_threshold == 4096
    assert ac.offload_config.pin_memory_enabled is False
    assert ac.offload_config.pin_memory_max_size_gib == 12.5
    assert ac.offload_config.pin_memory_bucket_size_mib == 128


def test_offload_config_validates_pin_memory_sizes():
    with pytest.raises(ValueError, match="pin_memory_max_size_gib"):
        ActivationOffloadConfig(pin_memory_max_size_gib=-1)
    with pytest.raises(ValueError, match="pin_memory_bucket_size_mib"):
        ActivationOffloadConfig(pin_memory_bucket_size_mib=0)


def test_already_constructed_offload_config_is_preserved():
    offload_config = ActivationOffloadConfig(keep_last_n=3)
    ac = ActivationCheckpointConfig(offload_config=offload_config)

    assert ac.offload_config is offload_config
    assert ac.offload_config.keep_last_n == 3


def test_offload_enabled_with_selective_mode_raises():
    with pytest.raises(ValueError, match=r"mode='full'"):
        ActivationCheckpointConfig(mode="selective", offload_config=ActivationOffloadConfig(enabled=True))
