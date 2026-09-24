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
"""Torch-free settings and validation for activation CPU offload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal
from typing import Optional

if TYPE_CHECKING:
    from arctic_platform.model.implementations.qwen35.config import ActivationOffloadConfig

_DEFAULT_PIN_MEMORY_BUCKET_SIZE_MIB = 64
_GIB = 1 << 30
_MIB = 1 << 20
PinMemoryMaxSize = float | Literal["auto"]


def validate_pin_memory_max_size_gib(pin_memory_max_size_gib: PinMemoryMaxSize) -> PinMemoryMaxSize:
    if pin_memory_max_size_gib == "auto":
        return pin_memory_max_size_gib
    value = float(pin_memory_max_size_gib)
    if value < 0:
        raise ValueError("pin_memory_max_size_gib must be 'auto' or non-negative")
    return value


def validate_pin_memory_bucket_size_mib(pin_memory_bucket_size_mib: int) -> int:
    value = int(pin_memory_bucket_size_mib)
    if value <= 0:
        raise ValueError("pin_memory_bucket_size_mib must be positive")
    return value


@dataclass
class ActivationOffloadSettings:
    """Runtime knobs for activation CPU offload and its pinned staging cache."""

    keep_last_n: int = 1
    use_streams: bool = True
    tensor_size_threshold: Optional[int] = None
    pin_memory_enabled: bool = True
    pin_memory_max_size_gib: PinMemoryMaxSize = "auto"
    pin_memory_bucket_size_mib: int = _DEFAULT_PIN_MEMORY_BUCKET_SIZE_MIB

    def __post_init__(self) -> None:
        self.keep_last_n = max(0, int(self.keep_last_n))
        self.use_streams = bool(self.use_streams)
        self.pin_memory_enabled = bool(self.pin_memory_enabled)
        self.pin_memory_max_size_gib = validate_pin_memory_max_size_gib(self.pin_memory_max_size_gib)
        self.pin_memory_bucket_size_mib = validate_pin_memory_bucket_size_mib(self.pin_memory_bucket_size_mib)
        if self.tensor_size_threshold is not None:
            self.tensor_size_threshold = int(self.tensor_size_threshold)

    @classmethod
    def from_offload_config(cls, config: ActivationOffloadConfig) -> ActivationOffloadSettings:
        return cls(
            keep_last_n=config.keep_last_n,
            use_streams=config.use_streams,
            tensor_size_threshold=config.tensor_size_threshold,
            pin_memory_enabled=config.pin_memory_enabled,
            pin_memory_max_size_gib=config.pin_memory_max_size_gib,
            pin_memory_bucket_size_mib=config.pin_memory_bucket_size_mib,
        )

    @property
    def pin_memory_bucket_size_bytes(self) -> int:
        return self.pin_memory_bucket_size_mib * _MIB

    @property
    def pin_memory_hard_max_size_bytes(self) -> Optional[int]:
        if self.pin_memory_max_size_gib == "auto":
            return None
        return max(0, int(float(self.pin_memory_max_size_gib) * _GIB))
