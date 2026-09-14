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

"""Arctic Platform SFT — client, config, and training processors.

The unified client (``ArcticSFTClient``) imports without ``[sft]`` / DeepSpeed.
Server processors still require ``arctic-platform[sft]`` or ``[rl]``.

Shared DeepSpeed/HTTP/Ray server code lives in ``arctic_platform.common``.
"""

from __future__ import annotations

from typing import Any

from arctic_platform.client.config import ArcticClientConfig
from arctic_platform.client.sft import ArcticSFTClient
from arctic_platform.client.sft import ArcticSFTClientConfig
from arctic_platform.client.sft import merge_sft_step_metrics

__all__ = [
    "ArcticClientConfig",
    "ArcticSFTClient",
    "ArcticSFTClientConfig",
    "LOGIT_LOSS_FNS",
    "SFT_LOSS_FNS",
    "merge_sft_step_metrics",
    "run_sft_pipeline",
    "sft_ce_loss",
    "sft_loss",
]

_SERVER = {
    "LOGIT_LOSS_FNS": ("arctic_platform.sft.processor", "LOGIT_LOSS_FNS"),
    "SFT_LOSS_FNS": ("arctic_platform.sft.processor", "SFT_LOSS_FNS"),
    "run_sft_pipeline": ("arctic_platform.sft.processor", "run_sft_pipeline"),
    "sft_ce_loss": ("arctic_platform.sft.processor", "sft_ce_loss"),
    "sft_loss": ("arctic_platform.sft.processor", "sft_loss"),
}


def __getattr__(name: str) -> Any:
    if name not in _SERVER:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from arctic_platform._dependency_groups import require_any_dep_group

    require_any_dep_group("sft", "rl")
    import importlib

    module_path, attr = _SERVER[name]
    value = getattr(importlib.import_module(module_path), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
