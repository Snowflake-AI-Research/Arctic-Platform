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
"""Registry and pipeline for patches applied to a model after load."""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

from arctic_platform.model.loader import LoadedModel
from arctic_platform.model.loader import LoaderContext

# Mutates the model in place. A patch reads its own settings (and any sibling
# patches it cares about) off ctx.spec.patches.
Patch = Callable[[nn.Module, LoaderContext], None]


# Canonical apply order; patches interact, so this is the single source of truth for
# sequencing (e.g. fp32_lm_head before chunked_lm_head, compile last). Every
# registered patch must appear here.
#
# Where does a new patch go? Patches form a dependency graph: some read or wrap
# structures that others create or replace. Place a patch after everything it
# depends on and before anything that depends on it. Independent patches can go in
# any relative order; when unsure, put structural rewrites early and whole-model
# wrappers (e.g. torch.compile) last.
PATCH_ORDER: tuple[str, ...] = ("liger",)

_PATCHES: dict[str, Patch] = {}


def register_patch(name: str) -> Callable[[Patch], Patch]:
    """Register a patch by name (must match a ``Patches`` field and be in PATCH_ORDER)."""

    def decorator(fn: Patch) -> Patch:
        assert name in PATCH_ORDER, f"patch {name!r} missing from PATCH_ORDER"
        assert name not in _PATCHES, f"patch {name!r} already registered"
        _PATCHES[name] = fn
        return fn

    return decorator


def apply_patches(loaded: LoadedModel, ctx: LoaderContext) -> None:
    """Apply enabled patches in canonical order, recording each on ``loaded.applied_patches``.

    Patches the loader already applied are skipped, so a second call is a no-op.
    """
    applied = set(loaded.applied_patches)
    for name in PATCH_ORDER:
        if name in applied:
            continue
        if not getattr(ctx.spec.patches, name, None):
            continue
        _PATCHES[name](loaded.model, ctx)
        applied.add(name)
    loaded.applied_patches = frozenset(applied)
