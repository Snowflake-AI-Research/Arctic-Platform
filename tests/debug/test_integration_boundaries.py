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

"""Main-only debug composition preserves disabled calls and concrete source identities."""

import ast
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from arctic_platform.debug import lora
from arctic_platform.debug import training


def test_disabled_reduced_fixture_hook_neither_imports_prediction_code_nor_mutates_model(monkeypatch):
    model = SimpleNamespace(_ap_preserved_mtp_weights=object(), _ap_preserved_glm_auxiliary_weights=object())
    before = dict(model.__dict__)
    original_import = __import__

    def reject_prediction_import(name, *args, **kwargs):
        if name.startswith(
            ("arctic_platform.model.implementations.glm53", "arctic_platform.model.implementations.qwen35")
        ):
            pytest.fail(f"disabled fixture observer imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", reject_prediction_import)
    training.maybe_capture_reduced_fixture_mtp(model, Path("/unused"), SimpleNamespace(debug=SimpleNamespace()))
    assert model.__dict__ == before


def test_disabled_participation_leaves_actor_binding_untouched(monkeypatch):
    monkeypatch.delenv(training.CAPTURE_ENV, raising=False)
    original = object()
    manager = SimpleNamespace(_ACTOR_CLS={"training": original})
    with training.maybe_select_participation_coordinator(manager, "training", original, object()):
        assert manager._ACTOR_CLS["training"] is original
    assert manager._ACTOR_CLS["training"] is original


def test_enabled_participation_requires_the_real_correctness_harness_before_actor_replacement(monkeypatch):
    monkeypatch.setenv(training.CAPTURE_ENV, "1")
    original = object()
    manager = SimpleNamespace(_ACTOR_CLS={"training": original})
    native_import = training.importlib.import_module
    requested = []

    def require_existing_harness(name):
        if name.startswith("arctic_platform.correctness"):
            requested.append(name)
            if name == "arctic_platform.correctness.harness.lora":
                raise ModuleNotFoundError(name)
        return native_import(name)

    monkeypatch.setattr(training.importlib, "import_module", require_existing_harness)
    with pytest.raises(ModuleNotFoundError, match="arctic_platform.correctness.harness.lora"):
        with training.maybe_select_participation_coordinator(manager, "training", original, object()):
            pytest.fail("an absent correctness harness admitted capture")
    assert requested == ["arctic_platform.correctness.harness.lora"]
    assert manager._ACTOR_CLS["training"] is original


def test_inventory_lazy_imports_select_the_relocated_debug_library():
    for helper in (lora.maybe_observe_model_materialization, lora.maybe_observe_checkpoint_reader):
        imports = [
            item
            for item in ast.walk(ast.parse(inspect.getsource(helper)))
            if isinstance(item, ast.ImportFrom)
            and any(alias.name == "observe_native_model_inventory" for alias in item.names)
        ]
        assert len(imports) == 1
        assert imports[0].module == "arctic_platform.debug.initialization_memory"


def test_disabled_materialization_and_reader_keep_original_objects(monkeypatch):
    monkeypatch.setattr(lora, "_model_construction_request", lambda: None)
    sentinel = object()
    assert lora.maybe_observe_model_materialization(sentinel, sentinel, sentinel, sentinel, sentinel) is None
    assert lora.maybe_observe_checkpoint_reader(sentinel, sentinel, sentinel, sentinel, sentinel) is sentinel


def test_selected_lora_exporter_hash_binds_the_actual_debug_integrated_native_file():
    from arctic_platform.model import weight_export

    path = Path(inspect.getsourcefile(weight_export._iter_param_wrapper_expert_loras))
    assert lora._EXPORT_SHA == hashlib.sha256(path.read_bytes()).hexdigest()
    assert inspect.getsourcefile(weight_export.iter_lora_weights) == str(path)
