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

"""Tests for loader selection and the optimization pipeline. These run on CPU with stub loaders."""

from __future__ import annotations

import types

import pytest
import torch.nn as nn
from pydantic import ValidationError

from arctic_platform.model import LoadedModel
from arctic_platform.model import LoaderContext
from arctic_platform.model import ModelSpec
from arctic_platform.model import build_model
from arctic_platform.model import register_loader
from arctic_platform.model import register_optimization
from arctic_platform.model import apply_optimizations
from arctic_platform.model import factory as factory_mod
from arctic_platform.model import loader as loader_mod
from arctic_platform.model import optimization as opt_mod


@pytest.fixture(autouse=True)
def _restore_registries():
    """Snapshot the global loader/optimization registries and restore them after each test."""
    loaders = dict(loader_mod._LOADERS)
    default = loader_mod._DEFAULT_LOADER
    opts = dict(opt_mod._OPTIMIZATIONS)
    order = opt_mod.OPTIMIZATION_ORDER
    yield
    loader_mod._LOADERS.clear()
    loader_mod._LOADERS.update(loaders)
    loader_mod._DEFAULT_LOADER = default
    opt_mod._OPTIMIZATIONS.clear()
    opt_mod._OPTIMIZATIONS.update(opts)
    opt_mod.OPTIMIZATION_ORDER = order


def _ctx(**optimization_flags) -> LoaderContext:
    """A LoaderContext whose spec exposes only the optimization flags a test cares about."""
    optimizations = types.SimpleNamespace(**optimization_flags)
    return LoaderContext(spec=types.SimpleNamespace(optimizations=optimizations))


def _register(name, *, matches=None, default=False):
    """Register a stub loader; the autouse fixture restores the registry afterwards."""

    @register_loader(name, matches=matches, default=default)
    def _loader(ctx: LoaderContext) -> LoadedModel:
        return LoadedModel(model=nn.Identity())

    return _loader


class TestLoaderSelection:
    @pytest.fixture(autouse=True)
    def _empty_registry(self):
        """Start each loader test from an empty registry so nothing depends on the real loaders."""
        loader_mod._LOADERS.clear()
        loader_mod._DEFAULT_LOADER = None

    def test_unset_loader_resolves_to_registered_default(self):
        """An unset loader is populated with whatever loader is registered as the default."""
        _register("base", default=True)
        assert ModelSpec(model_path="x").loader == "base"

    def test_explicit_loader_overrides_default(self):
        _register("base", default=True)
        _register("other")
        assert ModelSpec(model_path="x", loader="other").loader == "other"

    def test_unknown_loader_rejected(self):
        _register("base", default=True)
        with pytest.raises(ValidationError):
            ModelSpec(model_path="x", loader="nope")

    def test_duplicate_default_rejected(self):
        """A second default loader is rejected at registration, not selection."""
        _register("base", default=True)
        with pytest.raises(AssertionError, match="default loader already registered"):
            _register("also_default", default=True)

    def test_matching_predicate_wins_over_default(self, monkeypatch):
        """A loader whose predicate matches the model config is chosen over the default."""
        fake_config = types.SimpleNamespace(model_type="special")
        monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *a, **k: fake_config)

        _register("base", default=True)
        _register("special", matches=lambda ctx: getattr(ctx.model_config, "model_type", "") == "special")

        assert ModelSpec(model_path="x").loader == "special"

    def test_multiple_matches_rejected(self, monkeypatch):
        """Two matching predicates are ambiguous and rejected."""
        monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *a, **k: types.SimpleNamespace())

        _register("base", default=True)
        _register("m1", matches=lambda ctx: True)
        _register("m2", matches=lambda ctx: True)

        with pytest.raises(ValidationError, match="multiple loaders match"):
            ModelSpec(model_path="x")

    def test_build_model_runs_resolved_loader_then_optimizations(self, monkeypatch):
        """build_model builds via the resolved loader and hands the result to the optimization pipeline."""
        built = nn.Linear(1, 1)

        @register_loader("fake")
        def _fake(ctx: LoaderContext) -> LoadedModel:
            return LoadedModel(model=built)

        optimized = []
        monkeypatch.setattr(factory_mod, "apply_optimizations", lambda loaded, ctx: optimized.append(loaded))

        loaded = build_model(ModelSpec(model_path="x", loader="fake"))

        assert loaded.model is built
        assert optimized == [loaded]


class TestOptimizationPipeline:
    def test_register_requires_order_membership(self):
        """An optimization missing from OPTIMIZATION_ORDER cannot be registered."""
        with pytest.raises(AssertionError, match="missing from OPTIMIZATION_ORDER"):

            @register_optimization("not_in_order")
            def _opt(model, config, ctx) -> None:
                pass

    def test_apply_respects_order_and_skips_disabled(self, monkeypatch):
        """Enabled optimizations run in OPTIMIZATION_ORDER; disabled (False/None) ones are skipped."""
        monkeypatch.setattr(opt_mod, "OPTIMIZATION_ORDER", ("a", "b", "c"))
        calls = []
        for name in ("a", "b", "c"):
            opt_mod._OPTIMIZATIONS[name] = lambda model, config, ctx, n=name: calls.append(n)

        apply_optimizations(LoadedModel(model=nn.Identity()), _ctx(a=True, b=False, c={"vocab": 1}))

        assert calls == ["a", "c"]

    def test_apply_skips_loader_applied(self, monkeypatch):
        """Optimizations already reported by the loader are not re-applied."""
        monkeypatch.setattr(opt_mod, "OPTIMIZATION_ORDER", ("a",))
        calls = []
        opt_mod._OPTIMIZATIONS["a"] = lambda model, config, ctx: calls.append("a")

        loaded = LoadedModel(model=nn.Identity(), applied_optimizations=frozenset({"a"}))
        apply_optimizations(loaded, _ctx(a=True))

        assert calls == []
