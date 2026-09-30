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
"""Activation offload patch."""

from __future__ import annotations

import torch.nn as nn

from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.patch import register_patch


@register_patch("activation_offload")
def apply_activation_offload(model: nn.Module, ctx: LoaderContext) -> None:
    config = ctx.spec.patches.activation_offload
    if config is None or not config.enabled:
        raise ValueError("activation_offload patch requires an enabled patches.activation_offload config")

    from arctic_platform.model.implementations.gpu.activation_offload import install_activation_offload

    manager = install_activation_offload(model, config=config)
    backbone = getattr(model, "base_model", None)
    if backbone is not None and backbone is not model:
        install_activation_offload(
            backbone,
            config=config,
            manager=manager,
        )
