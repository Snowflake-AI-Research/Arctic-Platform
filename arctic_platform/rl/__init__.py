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

"""Arctic RL package.

Heavy imports (HTTP client/server, weight sync) are lazy so
``python -m arctic_platform.common.http_server`` (or the back-compat
``python -m arctic_platform.rl.http_server``) can start a training-only
server without pulling arctic_inference / vLLM at package import time.

SFT symbols remain lazily re-exported for back-compat; prefer
``arctic_platform.sft``.
"""

from __future__ import annotations

from typing import Any

from arctic_platform._dependency_groups import require_any_dep_group

# [sft] is enough: a training-only server must start from here without the
# sampling stack, which is gated at the arctic_inference import sites instead.
require_any_dep_group("sft", "rl")

__all__ = [
    "create_arctic_rl_client",
    "ArcticRLClientConfig",
    "WeightSyncConfig",
    "WeightSyncCoordinator",
    "run_pipeline",
    "run_sft_pipeline",
    "register_loss_fn",
    "register_post_processor",
    "grpo_loss",
    "sft_loss",
    "pack_sequences",
    "unpack_sequences",
]

_LAZY: dict[str, tuple[str, str]] = {
    "create_arctic_rl_client": ("arctic_platform.rl.client", "create_arctic_rl_client"),
    "ArcticRLClientConfig": ("arctic_platform.rl.config", "ArcticRLClientConfig"),
    "WeightSyncConfig": ("arctic_platform.rl.config", "WeightSyncConfig"),
    "WeightSyncCoordinator": ("arctic_platform.rl.weight_sync", "WeightSyncCoordinator"),
    "run_pipeline": ("arctic_platform.rl.processors", "run_pipeline"),
    "run_sft_pipeline": ("arctic_platform.sft", "run_sft_pipeline"),
    "register_loss_fn": ("arctic_platform.rl.processors", "register_loss_fn"),
    "register_post_processor": ("arctic_platform.rl.processors", "register_post_processor"),
    "grpo_loss": ("arctic_platform.rl.processors", "grpo_loss"),
    "sft_loss": ("arctic_platform.sft", "sft_loss"),
    "pack_sequences": ("arctic_platform.rl.processors", "pack_sequences"),
    "unpack_sequences": ("arctic_platform.rl.processors", "unpack_sequences"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr = _LAZY[name]
    import importlib

    mod = importlib.import_module(module_path)
    value = getattr(mod, attr)
    globals()[name] = value
    return value
