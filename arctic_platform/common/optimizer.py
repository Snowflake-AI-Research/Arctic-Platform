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
"""Canonical validation for the training optimizer block.

The wire form is flat and uses ``betas``:

    {"name": "AdamW", "lr": 1e-5, "weight_decay": 0.0, "betas": [0.9, 0.95], "eps": 1e-8}

``normalize_optimizer_config`` is the single place aliases are resolved, so every
submission path (direct Ray, Neutrino, the local ArcticRL server) emits the same
block from the same input. It runs client-side before ``createJob`` so a bad
optimizer costs zero GPU-seconds, and again server-side as an independent check.

Normalization only renames and validates; it never fills defaults. The server's
``OptimizerConfig`` owns the defaults, which keeps an explicitly submitted value
distinguishable from a defaulted one. The one exception is ``betas``: supplying
only ``beta1`` or only ``beta2`` fills its partner from :data:`DEFAULT_BETAS`,
since the canonical form is a 2-tuple.
"""

from __future__ import annotations

import math
from typing import Any

# Every key the training worker's optimizer builder actually reads. ``fused``
# selects FusedAdam vs torch.optim.AdamW; the GLM-MoE path forces it off.
SUPPORTED_OPTIMIZER_KEYS = frozenset(
    {
        "name",
        "lr",
        "weight_decay",
        "betas",
        "eps",
        "fused",
    }
)

# Accepted on input, rewritten to the canonical spelling on the wire.
# ``type`` is the DeepSpeed spelling; the worker reads ``name``.
OPTIMIZER_KEY_ALIASES = {"type": "name"}

DEFAULT_BETAS = (0.9, 0.999)

# Documented by ArcticRL as optimizer fields but consumed by nothing. Rejected
# with an actionable message rather than accepted and ignored.
UNSUPPORTED_SCHEDULER_KEYS = frozenset(
    {
        "lr_scheduler",
        "lr_scheduler_type",
        "num_warmup_steps",
        "training_horizon",
        "warmup_ratio",
        "warmup_steps",
        "warmup_steps_proportion",
    }
)

_SCHEDULER_GUIDANCE = (
    "DSS does not consume these names. For client-driven LR (RL/GRPO), pass the "
    "per-step value to the client's step() call. For a server-side DeepSpeed "
    'scheduler, set ds_config["scheduler"].'
)


def normalize_optimizer_config(
    optimizer: Any,
    *,
    location: str = "optimizer",
) -> tuple[dict[str, Any], float | None]:
    """Canonicalize and validate an optimizer block.

    Returns ``(canonical_optimizer, gradient_clipping)``. ``gradient_clipping``
    is accepted inside the optimizer block for backward compatibility but is
    never part of the canonical block: it belongs to the training config, so the
    caller hoists the returned value to wherever that lives. It is ``None`` when
    the input did not carry one.

    Raises ``ValueError`` for unknown keys, unsupported scheduler keys,
    conflicting aliases, and out-of-range values.
    """
    if not isinstance(optimizer, dict):
        raise ValueError(f"{location} must be an object, got {optimizer!r}")

    flat = _flatten_deepspeed_params(optimizer, location=location)
    gradient_clipping = _pop_gradient_clipping(flat, location=location)

    scheduler_keys = sorted(set(flat) & UNSUPPORTED_SCHEDULER_KEYS)
    if scheduler_keys:
        raise ValueError(
            f"{location} contains unsupported scheduler field(s): {scheduler_keys}. {_SCHEDULER_GUIDANCE}"
        )

    betas = _resolve_betas(flat, location=location)
    canonical = _resolve_aliases(flat, location=location)

    unsupported = sorted(set(canonical) - SUPPORTED_OPTIMIZER_KEYS)
    if unsupported:
        raise ValueError(
            f"{location} contains unsupported field(s): {unsupported}. "
            f"Supported fields: {sorted(SUPPORTED_OPTIMIZER_KEYS)}"
        )

    normalized: dict[str, Any] = {}
    if "name" in canonical:
        name = canonical["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{location}.name must be a non-empty string, got {name!r}")
        normalized["name"] = name
    if "lr" in canonical:
        # lr=0 is important for tests, lr<0 is the only invalid value
        normalized["lr"] = _non_negative_float(canonical["lr"], f"{location}.lr")
    if "weight_decay" in canonical:
        normalized["weight_decay"] = _non_negative_float(canonical["weight_decay"], f"{location}.weight_decay")
    if betas is not None:
        normalized["betas"] = betas
    if "eps" in canonical:
        normalized["eps"] = _positive_float(canonical["eps"], f"{location}.eps")
    if "fused" in canonical:
        fused = canonical["fused"]
        if not isinstance(fused, bool):
            raise ValueError(f"{location}.fused must be a boolean, got {fused!r}")
        normalized["fused"] = fused

    return normalized, gradient_clipping


def _flatten_deepspeed_params(optimizer: dict[str, Any], *, location: str) -> dict[str, Any]:
    """Merge a DeepSpeed ``{"type": ..., "params": {...}}`` block to one level.

    The worker reads hyperparameters from the top level, so an unflattened
    DeepSpeed block would drop every value nested under ``params``.
    """
    params = optimizer.get("params")
    if params is None:
        return {key: value for key, value in optimizer.items() if key != "params"}
    if not isinstance(params, dict):
        raise ValueError(f"{location}.params must be an object, got {params!r}")

    flat = {key: value for key, value in optimizer.items() if key != "params"}
    for key, value in params.items():
        if key in flat and flat[key] != value:
            raise ValueError(f"{location} sets {key}={flat[key]!r} and {location}.params.{key}={value!r}. Remove one.")
        flat[key] = value
    return flat


def _pop_gradient_clipping(flat: dict[str, Any], *, location: str) -> float | None:
    if "gradient_clipping" not in flat:
        return None
    value = flat.pop("gradient_clipping")
    if value is None:
        return None
    return _non_negative_float(value, f"{location}.gradient_clipping")


def _resolve_aliases(flat: dict[str, Any], *, location: str) -> dict[str, Any]:
    """Rewrite alias keys to canonical ones, rejecting conflicting duplicates."""
    canonical = {
        key: value
        for key, value in flat.items()
        if key not in OPTIMIZER_KEY_ALIASES and key not in ("beta1", "beta2", "betas")
    }
    for alias, target in OPTIMIZER_KEY_ALIASES.items():
        if alias not in flat:
            continue
        value = flat[alias]
        if target in canonical and canonical[target] != value:
            raise ValueError(
                f"{location} sets both {target}={canonical[target]!r} and {alias}={value!r}. Use {target}."
            )
        canonical[target] = value
    return canonical


def _resolve_betas(flat: dict[str, Any], *, location: str) -> list[float] | None:
    """Resolve ``betas`` / ``beta1`` / ``beta2`` into a validated 2-list."""
    has_betas = "betas" in flat and flat["betas"] is not None
    beta1 = flat.get("beta1")
    beta2 = flat.get("beta2")

    if has_betas:
        betas = _coerce_beta_pair(flat["betas"], location=location)
        # An alias repeating the same value is fine; a differing one is a bug
        # worth surfacing rather than silently picking a winner.
        for index, (alias, value) in enumerate((("beta1", beta1), ("beta2", beta2))):
            if value is None:
                continue
            alias_value = _beta_value(value, f"{location}.{alias}")
            if alias_value != betas[index]:
                raise ValueError(f"{location} sets betas={flat['betas']!r} and {alias}={value!r}. Use betas.")
        return betas

    if beta1 is None and beta2 is None:
        return None
    return [
        _beta_value(beta1, f"{location}.beta1") if beta1 is not None else DEFAULT_BETAS[0],
        _beta_value(beta2, f"{location}.beta2") if beta2 is not None else DEFAULT_BETAS[1],
    ]


def _coerce_beta_pair(value: Any, *, location: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{location}.betas must be a list of two numbers, got {value!r}")
    if len(value) != 2:
        raise ValueError(f"{location}.betas must have exactly two values, got {len(value)}: {value!r}")
    return [
        _beta_value(value[0], f"{location}.betas[0]"),
        _beta_value(value[1], f"{location}.betas[1]"),
    ]


def _beta_value(value: Any, name: str) -> float:
    number = _finite_float(value, name)
    if not 0 <= number < 1:
        raise ValueError(f"{name} must be in [0, 1), got {value!r}")
    return number


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _positive_float(value: Any, name: str) -> float:
    number = _finite_float(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value!r}")
    return number


def _non_negative_float(value: Any, name: str) -> float:
    number = _finite_float(value, name)
    if number < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return number
