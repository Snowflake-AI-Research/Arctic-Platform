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
"""Opt-in freeze of a VLM vision tower that a text-only job will never activate."""

from __future__ import annotations

import logging

import torch.nn as nn

from arctic_platform.model.implementations.qwen35.vlm import freeze_unused_vision_tower
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.patch import register_patch

logger = logging.getLogger(__name__)


@register_patch("freeze_unused_vision_tower")
def apply_freeze_unused_vision_tower(model: nn.Module, ctx: LoaderContext) -> None:
    frozen = freeze_unused_vision_tower(model)
    if frozen:
        logger.info(
            "froze %d vision-tower params (text-only job; unused params produce no grads and stall ZeRO-3)",
            frozen,
        )
