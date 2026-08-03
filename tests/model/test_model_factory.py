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

"""Tests for loader selection and the patch pipeline. These run on CPU with stub loaders."""

from __future__ import annotations

import types

import pytest
import torch.nn as nn
from pydantic import ValidationError

from arctic_platform.model import LoadedModel
from arctic_platform.model import LoaderContext
from arctic_platform.model import ModelSpec
from arctic_platform.model import apply_patches
from arctic_platform.model import build_model
from arctic_platform.model import factory as factory_mod
from arctic_platform.model import loader as loader_mod
from arctic_platform.model import patch as patch_mod
from arctic_platform.model import register_loader
from arctic_platform.model import register_patch


@pytest.fixture(autouse=True)
def _restore_registries():
    """Snapshot the global loader/patch registries and restore them after each test."""
    loaders = dict(loader_mod._LOADERS)
    default = loader_mod._DEFAULT_LOADER
    patches = dict(patch_mod._PATCHES)
    order = patch_mod.PATCH_ORDER
    yield
    loader_mod._LOADERS.clear()
    loader_mod._LOADERS.update(loaders)
    loader_mod._DEFAULT_LOADER = default
    patch_mod._PATCHES.clear()
    patch_mod._PATCHES.update(patches)
    patch_mod.PATCH_ORDER = order


def _ctx(**patch_flags) -> LoaderContext:
    """A LoaderContext whose spec exposes only the patch flags a test cares about."""
    patches = types.SimpleNamespace(**patch_flags)
    return LoaderContext(spec=types.SimpleNamespace(patches=patches))


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

    def test_build_model_runs_resolved_loader_then_patches(self, monkeypatch):
        """build_model builds via the resolved loader and hands the result to the patch pipeline."""
        built = nn.Linear(1, 1)

        @register_loader("fake")
        def _fake(ctx: LoaderContext) -> LoadedModel:
            return LoadedModel(model=built)

        patched = []
        monkeypatch.setattr(factory_mod, "apply_patches", lambda loaded, ctx: patched.append(loaded))

        loaded = build_model(ModelSpec(model_path="x", loader="fake"))

        assert loaded.model is built
        assert patched == [loaded]


class TestPatchPipeline:
    def test_registry_matches_config_and_order(self):
        """Built-in patches line up: PATCH_ORDER == Patches fields == registered patches."""
        from arctic_platform.model.config import Patches

        order = set(patch_mod.PATCH_ORDER)
        assert order == set(Patches.model_fields)
        assert order == set(patch_mod._PATCHES)

    def test_register_requires_order_membership(self):
        """A patch missing from PATCH_ORDER cannot be registered."""
        with pytest.raises(AssertionError, match="missing from PATCH_ORDER"):

            @register_patch("not_in_order")
            def _patch(model, ctx) -> None:
                pass

    def test_apply_respects_order_and_skips_disabled(self, monkeypatch):
        """Enabled patches run in PATCH_ORDER; disabled (False/None) ones are skipped."""
        monkeypatch.setattr(patch_mod, "PATCH_ORDER", ("a", "b", "c"))
        calls = []
        for name in ("a", "b", "c"):
            patch_mod._PATCHES[name] = lambda model, ctx, n=name: calls.append(n)

        apply_patches(LoadedModel(model=nn.Identity()), _ctx(a=True, b=False, c={"vocab": 1}))

        assert calls == ["a", "c"]

    def test_apply_skips_loader_applied(self, monkeypatch):
        """Patches already reported by the loader are not re-applied."""
        monkeypatch.setattr(patch_mod, "PATCH_ORDER", ("a",))
        calls = []
        patch_mod._PATCHES["a"] = lambda model, ctx: calls.append("a")

        loaded = LoadedModel(model=nn.Identity(), applied_patches=frozenset({"a"}))
        apply_patches(loaded, _ctx(a=True))

        assert calls == []
