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

"""ZoRRO Train prompt deduplication for RL training.

Patchers load on first use so a caller can import the model-type allowlist
without importing torch.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "Qwen3ModelOncePatcher",
    "QwenAttentionOncePatcher",
    "ReconstructionInfo",
    "ZoRRoTrain",
]

_LAZY: dict[str, tuple[str, str]] = {
    "Qwen3ModelOncePatcher": ("arctic_platform.rl.zorro_train.qwen_model_patcher", "Qwen3ModelOncePatcher"),
    "QwenAttentionOncePatcher": ("arctic_platform.rl.zorro_train.qwen_attention_patcher", "QwenAttentionOncePatcher"),
    "ReconstructionInfo": ("arctic_platform.rl.zorro_train.qwen_model_patcher", "ReconstructionInfo"),
    "ZoRRoTrain": ("arctic_platform.rl.zorro_train.zorro_train", "ZoRRoTrain"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_path, attr = _LAZY[name]
    value = getattr(importlib.import_module(module_path), attr)
    globals()[name] = value
    return value
