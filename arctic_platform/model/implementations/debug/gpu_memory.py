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

"""Retained CUDA capacity guards for model construction and training."""

from __future__ import annotations

import logging
import math
import os
from typing import Any
from typing import Mapping

from arctic_platform.model.implementations.debug.config import training_debug_config

logger = logging.getLogger(__name__)


def gpu_mem_preallocate_bytes(training_config: Mapping[str, Any]) -> int:
    """Validate the per-rank reserve, rounding fractional GiB up to bytes."""
    value = training_debug_config(training_config).get("gpu_mem_preallocate_gib", 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("debug.gpu_mem_preallocate_gib must be a finite non-negative number")
    try:
        size_bytes = value * (1 << 30)
        valid = value >= 0 and math.isfinite(size_bytes)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("debug.gpu_mem_preallocate_gib must be a finite non-negative number")
    return math.ceil(size_bytes)


def gpu_memory_debug_config(training_config: Mapping[str, Any]) -> dict:
    """Select only the capacity guard from provider-neutral or legacy diagnostics."""
    debug = training_debug_config(training_config)
    gpu_mem_preallocate_bytes(training_config)
    return {key: debug[key] for key in ("gpu_mem_preallocate_gib",) if key in debug}


def allocate_gpu_mem_preallocation(debug: Mapping[str, Any], *, device=None):
    """Allocate a live byte tensor; OOM propagates without reducing the reserve."""
    size_bytes = gpu_mem_preallocate_bytes({"debug": debug})
    if size_bytes == 0:
        return None
    import torch

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    allocation = torch.empty(size_bytes, dtype=torch.uint8, device=device)
    logger.info(
        "rank=%s [gpu-mem-preallocate] bytes=%d size=%.3fGiB",
        os.environ.get("RANK", "?"),
        size_bytes,
        size_bytes / (1 << 30),
    )
    return allocation
