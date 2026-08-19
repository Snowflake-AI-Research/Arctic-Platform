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

"""Shared Arctic Platform training/server infrastructure (protocol-agnostic).

Protocol-specific code lives under ``arctic_platform.rl`` (GRPO/RL); a
forthcoming SFT package will share this stack. This package holds the DeepSpeed
worker, HTTP/Ray servers, Ray cluster helpers, and low-level utils.
"""

from __future__ import annotations

from typing import Any

from arctic_platform._dependency_groups import require_any_dep_group

require_any_dep_group("sft", "rl")

__all__ = [
    "DeepSpeedWorker",
    "init_ray_cluster",
    "primary_ip",
]

# Lazily resolve the heavy exports so `import arctic_platform.common` stays cheap
# (DeepSpeedWorker pulls in ray/deepspeed). PEP 562 module __getattr__ keeps the
# names in __all__ actually accessible instead of being aspirational.
_LAZY = {
    "DeepSpeedWorker": ("arctic_platform.common.deepspeed_worker", "DeepSpeedWorker"),
    "init_ray_cluster": ("arctic_platform.common.ray_cluster", "init_ray_cluster"),
    "primary_ip": ("arctic_platform.common.ray_cluster", "primary_ip"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    return getattr(module, target[1])


def __dir__() -> list[str]:
    return sorted(__all__)
