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
"""Registry and pipeline for optimizations applied to a model after load."""

from __future__ import annotations

from typing import Any
from typing import Callable

from arctic_platform.model.loader import LoadedModel
from arctic_platform.model.loader import LoaderContext

# An optimization mutates the model in place given (model, config, ctx), where
# config is the matching field on ModelSpec.optimizations.
Optimization = Callable[[Any, Any, LoaderContext], None]

# Canonical apply order; they interact, so this is the single source of truth for
# sequencing (e.g. fp32_lm_head before chunked_lm_head, compile last). Every
# registered optimization must appear here.
OPTIMIZATION_ORDER: tuple[str, ...] = ("liger",)

_OPTIMIZATIONS: dict[str, Optimization] = {}


def register_optimization(name: str) -> Callable[[Optimization], Optimization]:
    """Register an optimization by name (must match an ``Optimizations`` field and be in OPTIMIZATION_ORDER)."""

    def decorator(fn: Optimization) -> Optimization:
        assert name in OPTIMIZATION_ORDER, f"optimization {name!r} missing from OPTIMIZATION_ORDER"
        assert name not in _OPTIMIZATIONS, f"optimization {name!r} already registered"
        _OPTIMIZATIONS[name] = fn
        return fn

    return decorator


def apply_optimizations(loaded: LoadedModel, ctx: LoaderContext) -> None:
    """Apply enabled optimizations in canonical order, skipping ones the loader already applied."""
    for name in OPTIMIZATION_ORDER:
        if name in loaded.applied_optimizations:
            continue
        config = getattr(ctx.spec.optimizations, name, None)
        if config is None or config is False:
            continue
        _OPTIMIZATIONS[name](loaded.model, config, ctx)
