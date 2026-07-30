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

"""Arctic RL client — HTTP / Ray / Cortex frontends for RL training.

Two symbols are imported eagerly (`ArcticRLClientConfig`, `WeightSyncConfig`)
because they carry no heavy transitive deps — just pydantic. Everything else
is lazy-loaded on first attribute access via `__getattr__` so that

    from arctic_platform.rl import ArcticRLClientConfig
    # or:
    from arctic_platform.rl import create_arctic_rl_client
    create_arctic_rl_client(ArcticRLClientConfig(backend="cortex", ...))

on a *Cortex-only* driver (no vllm / ray / arctic_inference / torch installed)
succeeds. Prior to this, `__init__` eagerly loaded `client.py`, which pulled
`http_client.py` → `http_server.py` → `arctic_inference.server.metrics` →
`vllm`. Cortex users paid that cost even though the Cortex dispatch branch
(see `arctic_platform.rl.client.create_arctic_rl_client`) short-circuits to
`arctic_platform.client` and never touches the on-prem HTTP server code.

Attribute-name → (module, attr) map below defines the public surface:
`__all__` still lists everything, so `from arctic_platform.rl import *`
retrieves the same set as before (each import triggers the lazy load).
"""

from __future__ import annotations

import importlib
from typing import Any

from arctic_platform.rl.config import ArcticRLClientConfig
from arctic_platform.rl.config import WeightSyncConfig

# name -> (submodule dotted path, attribute name on that submodule).
# Kept as data (not code) so the lazy resolver stays a single implementation.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "create_arctic_rl_client": ("arctic_platform.rl.client", "create_arctic_rl_client"),
    "grpo_loss": ("arctic_platform.rl.processors", "grpo_loss"),
    "pack_sequences": ("arctic_platform.rl.processors", "pack_sequences"),
    "register_loss_fn": ("arctic_platform.rl.processors", "register_loss_fn"),
    "register_post_processor": ("arctic_platform.rl.processors", "register_post_processor"),
    "run_pipeline": ("arctic_platform.rl.processors", "run_pipeline"),
    "unpack_sequences": ("arctic_platform.rl.processors", "unpack_sequences"),
    "WeightSyncCoordinator": ("arctic_platform.rl.weight_sync", "WeightSyncCoordinator"),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy loader for the heavy exports.

    Cached in `globals()` after first resolution so subsequent attribute
    accesses hit the normal fast path (no re-import).
    """
    try:
        mod_path, attr = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'arctic_platform.rl' has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(mod_path), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy exports in `dir()` and tab-completion."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "create_arctic_rl_client",
    "ArcticRLClientConfig",
    "WeightSyncConfig",
    "WeightSyncCoordinator",
    "run_pipeline",
    "register_loss_fn",
    "register_post_processor",
    "grpo_loss",
    "pack_sequences",
    "unpack_sequences",
]
