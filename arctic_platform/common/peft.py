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
"""Validation for the LoRA config subset accepted by weight sync."""

from __future__ import annotations

from typing import Any

PEFT_LORA_DEFAULT_R = 8
PEFT_LORA_DEFAULT_ALPHA = 8
PEFT_LORA_DEFAULT_DROPOUT = 0.0
PEFT_LORA_DEFAULT_BIAS = "none"
DETERMINISTIC_LORA_RECIPE = "arctic-correctness-lora-init-v1"
SUPPORTED_LORA_PEFT_CONFIG_KEYS = frozenset(
    {
        "peft_type",
        "task_type",
        "r",
        "lora_alpha",
        "lora_dropout",
        "bias",
        "target_modules",
        "target_parameters",
        "deterministic_init",
    }
)


def normalize_lora_initialization(value: Any, *, location: str = "peft_config.deterministic_init") -> dict:
    """Validate the opt-in, name-seeded CPU initializer without training imports."""
    if not isinstance(value, dict) or set(value) != {"recipe_version", "global_seed", "dtype"}:
        raise ValueError(f"{location} requires exactly recipe_version, global_seed, and dtype")
    if value["recipe_version"] != DETERMINISTIC_LORA_RECIPE:
        raise ValueError(f"{location}.recipe_version is unsupported")
    seed = value["global_seed"]
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError(f"{location}.global_seed must be an integer in [0, 2**63)")
    if not isinstance(value["dtype"], str) or value["dtype"] not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"{location}.dtype must be float32, float16, or bfloat16")
    return dict(value)


def normalize_lora_peft_config(
    peft_config: dict[str, Any],
    *,
    location: str = "peft_config",
) -> dict[str, Any]:
    """Validate and normalize the LoRA subset supported by weight sync."""
    if not isinstance(peft_config, dict):
        raise ValueError(f"{location} must be an object, got {peft_config!r}")

    unsupported = sorted(set(peft_config) - SUPPORTED_LORA_PEFT_CONFIG_KEYS)
    if unsupported:
        supported = sorted(SUPPORTED_LORA_PEFT_CONFIG_KEYS)
        raise ValueError(f"{location} contains unsupported field(s): {unsupported}. Supported fields: {supported}")

    peft_type = peft_config.get("peft_type")
    if peft_type != "Lora":
        raise ValueError(f"{location}.peft_type must be 'Lora', got {peft_type!r}")

    task_type = peft_config.get("task_type", "CAUSAL_LM")
    if task_type != "CAUSAL_LM":
        raise ValueError(f"{location}.task_type must be 'CAUSAL_LM', got {task_type!r}")

    rank = peft_config.get("r", PEFT_LORA_DEFAULT_R)
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError(f"{location}.r must be a positive integer, got {rank!r}")

    lora_alpha = peft_config.get("lora_alpha", PEFT_LORA_DEFAULT_ALPHA)
    if not isinstance(lora_alpha, int) or isinstance(lora_alpha, bool) or lora_alpha < 0:
        raise ValueError(f"{location}.lora_alpha must be a non-negative integer, got {lora_alpha!r}")

    lora_dropout = peft_config.get("lora_dropout", PEFT_LORA_DEFAULT_DROPOUT)
    if not isinstance(lora_dropout, (int, float)) or isinstance(lora_dropout, bool) or not 0 <= lora_dropout <= 1:
        raise ValueError(f"{location}.lora_dropout must be a number in [0, 1], got {lora_dropout!r}")

    bias = peft_config.get("bias", PEFT_LORA_DEFAULT_BIAS)
    if bias not in {"none", "all", "lora_only"}:
        raise ValueError(f"{location}.bias must be one of 'none', 'all', or 'lora_only', got {bias!r}")
    if bias != "none":
        raise ValueError(f"{location}.bias={bias!r} is not supported by vLLM 0.26 LoRA. Use bias='none'.")

    target_modules = peft_config.get("target_modules", [])
    if isinstance(target_modules, str):
        if not target_modules:
            raise ValueError(f"{location}.target_modules cannot be empty")
    elif isinstance(target_modules, list):
        if any(not isinstance(value, str) or not value for value in target_modules):
            raise ValueError(f"{location}.target_modules must contain only non-empty strings")
        target_modules = list(target_modules)
    else:
        raise ValueError(f"{location}.target_modules must be a string or list of strings, got {target_modules!r}")

    target_parameters = peft_config.get("target_parameters", [])
    if not isinstance(target_parameters, list) or any(
        not isinstance(value, str) or not value for value in target_parameters
    ):
        raise ValueError(f"{location}.target_parameters must be a list of non-empty strings")
    target_parameters = list(target_parameters)

    if not target_modules and not target_parameters:
        raise ValueError(f"{location} must specify target_modules, target_parameters, or both")

    normalized = {
        "peft_type": "Lora",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": float(lora_dropout),
        "bias": bias,
        "target_modules": target_modules,
        "target_parameters": target_parameters,
    }
    if "deterministic_init" in peft_config:
        if target_parameters:
            raise ValueError(f"{location}.deterministic_init supports dense target_modules only")
        normalized["deterministic_init"] = normalize_lora_initialization(
            peft_config["deterministic_init"], location=f"{location}.deterministic_init"
        )
    return normalized
