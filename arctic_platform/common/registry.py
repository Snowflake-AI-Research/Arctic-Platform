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

"""Shared loss / post-processor registries used by RL and SFT pipelines."""

from __future__ import annotations

import importlib
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Set

POST_PROCESSORS: Dict[str, Callable] = {}
LOSS_FNS: Dict[str, Callable] = {}
PACKED_LOSS_REDUCTION_ATTR = "_arctic_packed_loss_reduction"
SUMMED_METRICS_ATTR = "_arctic_summed_metrics"
LOSS_CAPABILITIES_ATTR = "_arctic_loss_capabilities"

# Metric names a loss fn declared additive via ``register_loss_fn(...,
# summed_metrics=...)``. Flat union rather than per-loss-fn: the reducers in
# ``common.utils.batch`` run on metric dicts that have already been merged
# across microbatches, gradient accumulation, and DP ranks, so no loss-fn
# context survives to the reduction. Names must therefore stay globally unique,
# which the loss-fn prefixes already enforce.
DECLARED_SUMMED_METRICS: Set[str] = set()

# Built-in public names. ``register_*`` may add more; colliding a public name
# with a *different* callable raises. ``_``-prefixed names are test-only and
# may overwrite.
PUBLIC_LOSS_FNS = frozenset(
    {
        "ap_grpo",
        "ap_grpo_echo_v1",
        "grpo",
        "grpo_echo_v1",
        "sft",
        "sft_ce",
        "verl_grpo",
        "causal_cross_entropy",
    }
)
PUBLIC_POST_PROCESSORS = frozenset(
    {
        "identity",
        "compute_entropy_and_logprobs",
        "ap_compute_logprobs",
        "compute_logprobs",
        "compute_entropy",
        "apply_temperature",
    }
)


def _is_public_registry_name(name: str) -> bool:
    return bool(name) and not name.startswith("_")


def _bind_registry(registry: dict, name: str, fn: Callable) -> None:
    """Store *fn* under *name*; refuse a public-name overwrite by another callable."""
    existing = registry.get(name)
    if existing is not None and existing is not fn and _is_public_registry_name(name):
        raise ValueError(f"refusing to overwrite registered {name!r} with a different callable")
    registry[name] = fn


def register_post_processor(name: str):
    """Register a post-forward processor under *name*."""

    def decorator(fn: Callable) -> Callable:
        _bind_registry(POST_PROCESSORS, name, fn)
        return fn

    return decorator


def is_declared_summed_metric(name: str) -> bool:
    """Whether some registered loss fn declared *name* as an additive metric."""
    return name in DECLARED_SUMMED_METRICS


def register_loss_fn(
    name: str,
    *,
    packed_loss_reduction: Callable | None = None,
    summed_metrics: Iterable[str] = (),
):
    """Register a loss function and its optional packed-microbatch contract.

    ``summed_metrics`` names the metrics this loss fn emits that are additive
    across packed microbatches, gradient accumulation, and DP ranks, so the
    reducers sum them instead of averaging. Declaring a metric here is the
    explicit alternative to relying on the ``loss_term_*`` / ``*_sum`` /
    ``*_count`` naming convention, which stays in force for undeclared keys
    (see ``common.utils.batch.metric_is_summed``).

    """
    declared = frozenset(summed_metrics)

    def decorator(fn: Callable) -> Callable:
        if packed_loss_reduction is not None:
            existing = getattr(fn, PACKED_LOSS_REDUCTION_ATTR, None)
            if existing is not None and existing is not packed_loss_reduction:
                raise ValueError(f"refusing to replace packed_loss_reduction on registered {name!r}")
        # Bind first so a refused public-name overwrite cannot leak
        # ``summed_metrics`` into the process-global union.
        _bind_registry(LOSS_FNS, name, fn)
        if packed_loss_reduction is not None:
            setattr(fn, PACKED_LOSS_REDUCTION_ATTR, packed_loss_reduction)
        if declared:
            setattr(fn, SUMMED_METRICS_ATTR, declared | getattr(fn, SUMMED_METRICS_ATTR, frozenset()))
            DECLARED_SUMMED_METRICS.update(declared)
        return fn

    return decorator


def declare_loss_capabilities(*capabilities: str):
    """Attach execution capabilities without changing function registration."""
    declared = frozenset(capabilities)

    def decorator(fn: Callable) -> Callable:
        setattr(
            fn,
            LOSS_CAPABILITIES_ATTR,
            declared | getattr(fn, LOSS_CAPABILITIES_ATTR, frozenset()),
        )
        return fn

    return decorator


def resolve_fn(registry: dict, name: str) -> Callable:
    """Look up *name* in registry; fall back to dotted-path import."""
    if name in registry:
        return registry[name]
    if "." not in name:
        known = sorted(key for key in registry if _is_public_registry_name(key) and "." not in key)
        hint = ""
        if name in {"cortex_grpo", "cortex_grpo_echo_v1"}:
            hint = " (did you mean 'grpo' / 'grpo_echo_v1'? AP variants are 'ap_grpo' / 'ap_grpo_echo_v1')"
        elif name == "cortex_compute_logprobs":
            hint = " (did you mean 'compute_logprobs'? AP variant is 'ap_compute_logprobs')"
        raise ValueError(f"unknown registry name {name!r}{hint}; known: {known}")
    module_path, fn_name = name.rsplit(".", 1)
    fn = getattr(importlib.import_module(module_path), fn_name)
    _bind_registry(registry, name, fn)
    return fn


# Back-compat alias used by older call sites.
_resolve_fn = resolve_fn
