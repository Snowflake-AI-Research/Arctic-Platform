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

import sys
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
    loader_mod._load_hf_config.cache_clear()


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
        assert ModelSpec(model_path_or_name="x").loader == "base"

    def test_explicit_loader_overrides_default(self):
        _register("base", default=True)
        _register("other")
        assert ModelSpec(model_path_or_name="x", loader="other").loader == "other"

    def test_unknown_loader_rejected(self):
        _register("base", default=True)
        with pytest.raises(ValidationError):
            ModelSpec(model_path_or_name="x", loader="nope")

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
        _register("special", matches=lambda ctx: getattr(ctx.hf_config, "model_type", "") == "special")

        assert ModelSpec(model_path_or_name="x").loader == "special"

    def test_multiple_matches_rejected(self, monkeypatch):
        """Two matching predicates are ambiguous and rejected."""
        monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *a, **k: types.SimpleNamespace())

        _register("base", default=True)
        _register("m1", matches=lambda ctx: True)
        _register("m2", matches=lambda ctx: True)

        with pytest.raises(ValidationError, match="multiple loaders match"):
            ModelSpec(model_path_or_name="x")

    def test_build_model_runs_resolved_loader_then_patches(self, monkeypatch):
        """build_model builds via the resolved loader and hands the result to the patch pipeline."""
        built = nn.Linear(1, 1)

        @register_loader("fake")
        def _fake(ctx: LoaderContext) -> LoadedModel:
            return LoadedModel(model=built)

        patched = []
        monkeypatch.setattr(factory_mod, "apply_patches", lambda loaded, ctx: patched.append(loaded))

        loaded = build_model(ModelSpec(model_path_or_name="x", loader="fake"))

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

    def test_apply_records_applied_and_is_idempotent(self, monkeypatch):
        """apply_patches records what it applied; a second call re-applies nothing."""
        monkeypatch.setattr(patch_mod, "PATCH_ORDER", ("a", "b"))
        calls = []
        for name in ("a", "b"):
            patch_mod._PATCHES[name] = lambda model, ctx, n=name: calls.append(n)

        loaded = LoadedModel(model=nn.Identity())
        ctx = _ctx(a=True, b=False)
        apply_patches(loaded, ctx)
        assert loaded.applied_patches == frozenset({"a"})

        apply_patches(loaded, ctx)
        assert calls == ["a"]


class TestFromDsWorkerConfig:
    def test_defaults_preserve_worker_behavior(self):
        spec = ModelSpec.from_ds_worker_config("Qwen/Qwen3-1.7B", {"attn_implementation": "flash_attention_2"})

        assert spec.model_path_or_name == "Qwen/Qwen3-1.7B"
        assert spec.dtype == "bfloat16"
        assert spec.attn_implementation == "flash_attention_2"
        assert spec.patches.liger is False
        assert spec.patches.gradient_checkpointing is True  # bridge default
        assert spec.patches.zorro_train is None  # None disables (empty patch would be truthy)

    def test_requires_attn_implementation(self):
        with pytest.raises(ValueError, match="requires attn_implementation"):
            ModelSpec.from_ds_worker_config("Qwen/Qwen3-1.7B", {})
        with pytest.raises(ValueError, match="requires attn_implementation"):
            ModelSpec.from_ds_worker_config("Qwen/Qwen3-1.7B", {"attn_implementation": None})

    def test_accepts_any_flash_attention_backend(self):
        for impl in ("flash_attention_2", "flash_attention_3", "flash_attention_4"):
            spec = ModelSpec.from_ds_worker_config("Qwen/Qwen3-1.7B", {"attn_implementation": impl})
            assert spec.attn_implementation == impl

    def test_generic_patches_default_gc_off(self):
        from arctic_platform.model.config import Patches

        assert Patches().gradient_checkpointing is False
        assert ModelSpec(model_path_or_name="x").attn_implementation is None

    def test_liger_and_gc_flags_map(self):
        spec = ModelSpec.from_ds_worker_config(
            "x",
            {"use_liger": True, "enable_gradient_checkpointing": False, "attn_implementation": "sdpa"},
        )
        assert spec.patches.liger is True
        assert spec.patches.gradient_checkpointing is False
        assert spec.attn_implementation == "sdpa"
        assert spec.patches.zorro_train is None

    def test_zorro_enabled_maps_fields(self):
        spec = ModelSpec.from_ds_worker_config(
            "x",
            {
                "attn_implementation": "flash_attention_2",
                "zorro_train_enable": True,
                "response_len": 1024,
                "max_token_len": 16384,
                "rollout_n": 8,
                "temperature": 1.0,
                "use_unpad": True,
                "world_size": 2,
                "logits_optimization": "memory",
                "logits_optimization_peak_mem_size_in_gib": 8,
                "logits_compute_from_fp32_inputs": True,
                "logits_compute_in_fp32": True,
            },
        )
        z = spec.patches.zorro_train
        assert z is not None
        assert z.response_len == 1024
        assert z.max_token_len == 16384
        assert z.rollout_n == 8
        assert z.temperature == 1.0
        assert z.world_size == 2
        assert z.logits_optimization == "memory"
        assert z.logits_optimization_peak_mem_size_in_gib == 8
        assert z.logits_compute_from_fp32_inputs is True
        assert z.logits_compute_in_fp32 is True

    def test_zorro_uses_pydantic_defaults_for_omitted_fields(self):
        from arctic_platform.model.config import ZorroTrainPatch

        spec = ModelSpec.from_ds_worker_config(
            "x",
            {"attn_implementation": "flash_attention_2", "zorro_train_enable": True, "rollout_n": 4},
        )
        z = spec.patches.zorro_train
        assert z is not None
        assert z.rollout_n == 4
        # Omitted keys come from ZorroTrainPatch field defaults (single source of truth).
        defaults = ZorroTrainPatch()
        assert z.use_unpad == defaults.use_unpad
        assert z.logits_optimization == defaults.logits_optimization
        assert z.logits_optimization_peak_mem_size_in_gib == defaults.logits_optimization_peak_mem_size_in_gib
        assert z.logits_compute_from_fp32_inputs == defaults.logits_compute_from_fp32_inputs
        assert z.logits_compute_in_fp32 == defaults.logits_compute_in_fp32


class TestZorroAndGcPatches:
    def test_gradient_checkpointing_patch_calls_enable(self):
        from arctic_platform.model.patches.gradient_checkpointing import apply_gradient_checkpointing

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.enabled = False

            def gradient_checkpointing_enable(self):
                self.enabled = True

        model = _Model()
        apply_gradient_checkpointing(model, _ctx(gradient_checkpointing=True))
        assert model.enabled is True

    def test_gradient_checkpointing_skipped_via_apply_patches(self, monkeypatch):
        monkeypatch.setattr(patch_mod, "PATCH_ORDER", ("gradient_checkpointing",))
        calls = []
        patch_mod._PATCHES["gradient_checkpointing"] = lambda model, ctx: calls.append("gc")

        apply_patches(LoadedModel(model=nn.Identity()), _ctx(gradient_checkpointing=False))
        assert calls == []
        apply_patches(LoadedModel(model=nn.Identity()), _ctx(gradient_checkpointing=True))
        assert calls == ["gc"]

    def test_zorro_patch_builds_patcher_and_attaches(self, monkeypatch):
        from arctic_platform.model.config import ZorroTrainPatch
        from arctic_platform.model.patches.zorro_train import apply_zorro_train

        constructed = {}

        class _FakePatcher:
            def __init__(self, model, **kwargs):
                constructed["model"] = model
                constructed["kwargs"] = kwargs
                self.patched = False

            def patch_forward(self):
                self.patched = True

        fake_mod = types.ModuleType("arctic_platform.rl.zorro_train.qwen_model_patcher")
        fake_mod.Qwen3ModelOncePatcher = _FakePatcher
        fake_mod.get_supported_model_type = lambda model: "qwen3"
        monkeypatch.setitem(sys.modules, "arctic_platform.rl.zorro_train.qwen_model_patcher", fake_mod)

        model = nn.Identity()
        model.config = types.SimpleNamespace(model_type="qwen3")
        settings = ZorroTrainPatch(response_len=1024, rollout_n=8, world_size=2, logits_optimization="memory")
        apply_zorro_train(model, _ctx(zorro_train=settings))

        assert constructed["model"] is model
        assert constructed["kwargs"]["response_len"] == 1024
        assert constructed["kwargs"]["rollout_n"] == 8
        assert constructed["kwargs"]["world_size"] == 2
        assert constructed["kwargs"]["logits_optimization"] == "memory"
        assert isinstance(model._arctic_zorro_once_patcher, _FakePatcher)
        assert model._arctic_zorro_once_patcher.patched is True

    def test_zorro_patch_rejects_non_qwen3(self, monkeypatch):
        from arctic_platform.model.config import ZorroTrainPatch
        from arctic_platform.model.patches.zorro_train import apply_zorro_train
        from arctic_platform.rl.zorro_train.qwen_model_patcher import get_supported_model_type

        # Keep the real guard; stub only the patcher constructor so we never patch.
        fake_mod = types.ModuleType("arctic_platform.rl.zorro_train.qwen_model_patcher")
        fake_mod.Qwen3ModelOncePatcher = object
        fake_mod.get_supported_model_type = get_supported_model_type
        monkeypatch.setitem(sys.modules, "arctic_platform.rl.zorro_train.qwen_model_patcher", fake_mod)

        model = nn.Identity()
        model.config = types.SimpleNamespace(model_type="llama")
        with pytest.raises(ValueError, match="Unsupported model_type=llama"):
            apply_zorro_train(model, _ctx(zorro_train=ZorroTrainPatch(response_len=1024, rollout_n=8, world_size=1)))
