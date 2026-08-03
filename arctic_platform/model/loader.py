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
"""Loaders: how a base module is built and its weights materialized."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

import torch.nn as nn

from arctic_platform.model.config import ModelSpec

if TYPE_CHECKING:
    from transformers import PretrainedConfig


@functools.lru_cache(maxsize=None)
def _load_hf_config(model_path_or_name: str) -> PretrainedConfig | None:
    """Load (and memoize) the HuggingFace config, or None if the name/path isn't an HF model."""
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(model_path_or_name)
    except (OSError, ValueError):
        return None


@dataclass
class LoaderContext:
    """Everything a loader needs to build a model."""

    spec: ModelSpec
    parallel_groups: Any | None = None

    @property
    def hf_config(self) -> PretrainedConfig | None:
        """The model's HuggingFace config, or None if the name/path isn't an HF model. Parsed once."""
        return _load_hf_config(self.spec.model_path_or_name)


@dataclass
class LoadedModel:
    """A built model and the patches already applied to it."""

    model: nn.Module
    applied_patches: frozenset[str] = field(default_factory=frozenset)


Loader = Callable[[LoaderContext], LoadedModel]
Matcher = Callable[[LoaderContext], bool]


@dataclass
class _LoaderEntry:
    fn: Loader
    matches: Matcher | None


_LOADERS: dict[str, _LoaderEntry] = {}
_DEFAULT_LOADER: str | None = None


def register_loader(name: str, matches: Matcher | None = None, default: bool = False) -> Callable[[Loader], Loader]:
    """Register a loader by name. Optionally give it a ``matches`` predicate or mark it the ``default``."""

    def decorator(fn: Loader) -> Loader:
        global _DEFAULT_LOADER
        assert name not in _LOADERS, f"loader {name!r} already registered"
        if default:
            assert _DEFAULT_LOADER is None, f"default loader already registered: {_DEFAULT_LOADER!r}"
            _DEFAULT_LOADER = name
        _LOADERS[name] = _LoaderEntry(fn=fn, matches=matches)
        return fn

    return decorator


def is_registered_loader(name: str) -> bool:
    return name in _LOADERS


def resolve_loader_name(spec: ModelSpec) -> str:
    """Resolve which loader a spec should use: single matching predicate, else the default."""
    ctx = LoaderContext(spec=spec)
    matched = [name for name, entry in _LOADERS.items() if entry.matches is not None and entry.matches(ctx)]
    assert len(matched) <= 1, f"multiple loaders match: {matched}; set spec.loader to disambiguate"
    if len(matched) == 1:
        return matched[0]

    assert _DEFAULT_LOADER is not None, "no default loader registered"
    return _DEFAULT_LOADER


def select_loader(ctx: LoaderContext) -> Loader:
    """Return the loader for a spec whose ``loader`` has already been resolved."""
    name = ctx.spec.loader
    assert name is not None, "spec.loader must be resolved before select_loader"
    return _LOADERS[name].fn
