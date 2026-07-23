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

"""Arctic RL client -- HTTP client for RL training against dss-platform or local server.

Top-level names are exposed via lazy PEP 562 ``__getattr__`` so
lightweight consumers (e.g. ``arctic_platform.rl.tinker_router`` unit
tests without torch/aiohttp on the path) can import the subpackage
without dragging in the full HTTP/Ray client stack. Fully-installed
callers keep the same import surface — the first attribute access
hydrates the submodule as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


_LAZY_ATTRS = {
    "create_arctic_rl_client": ("arctic_platform.rl.client", "create_arctic_rl_client"),
    "ArcticRLClientConfig": ("arctic_platform.rl.config", "ArcticRLClientConfig"),
    "WeightSyncConfig": ("arctic_platform.rl.config", "WeightSyncConfig"),
    "WeightSyncCoordinator": ("arctic_platform.rl.weight_sync", "WeightSyncCoordinator"),
    "run_pipeline": ("arctic_platform.rl.processors", "run_pipeline"),
    "register_loss_fn": ("arctic_platform.rl.processors", "register_loss_fn"),
    "register_post_processor": ("arctic_platform.rl.processors", "register_post_processor"),
    "grpo_loss": ("arctic_platform.rl.processors", "grpo_loss"),
    "pack_sequences": ("arctic_platform.rl.processors", "pack_sequences"),
    "unpack_sequences": ("arctic_platform.rl.processors", "unpack_sequences"),
}


def __getattr__(name: str):
    try:
        module_path, attr = _LAZY_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'arctic_platform.rl' has no attribute {name!r}") from exc
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))


__all__ = list(_LAZY_ATTRS)


if TYPE_CHECKING:
    # Re-export for static analysis — keeps IDE go-to-definition and mypy
    # happy without triggering eager imports at runtime.
    from arctic_platform.rl.client import create_arctic_rl_client  # noqa: F401
    from arctic_platform.rl.config import ArcticRLClientConfig  # noqa: F401
    from arctic_platform.rl.config import WeightSyncConfig  # noqa: F401
    from arctic_platform.rl.processors import grpo_loss  # noqa: F401
    from arctic_platform.rl.processors import pack_sequences  # noqa: F401
    from arctic_platform.rl.processors import register_loss_fn  # noqa: F401
    from arctic_platform.rl.processors import register_post_processor  # noqa: F401
    from arctic_platform.rl.processors import run_pipeline  # noqa: F401
    from arctic_platform.rl.processors import unpack_sequences  # noqa: F401
    from arctic_platform.rl.weight_sync import WeightSyncCoordinator  # noqa: F401
