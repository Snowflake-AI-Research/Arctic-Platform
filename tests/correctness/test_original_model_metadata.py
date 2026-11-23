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

"""CPU controls for the packaged, immutable original-model reference metadata."""

import hashlib
import json
from dataclasses import FrozenInstanceError
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from arctic_platform.correctness.models import original


def test_original_models_reviewed_metadata_pins_are_unchanged():
    raw = json.loads(original.ORIGINAL_MODELS_PATH.read_text(encoding="utf-8"))
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == "5782f162479a253a4e4ce36cba57ecd7b1c6101560abd0502f23b0433fe0da73"


def test_original_models_load_every_reference_field_independently_of_cwd(monkeypatch, tmp_path):
    raw = json.loads(original.ORIGINAL_MODELS_PATH.read_text(encoding="utf-8"))
    monkeypatch.chdir(tmp_path)
    models = original._load_original_models()
    assert models == original.ORIGINAL_MODELS
    assert set(models) == set(raw)
    for name, metadata in models.items():
        actual = {field.name: getattr(metadata, field.name) for field in fields(metadata)}
        actual["dtype_bytes"] = dict(metadata.dtype_bytes)
        assert json.loads(json.dumps(actual)) == raw[name]
        assert isinstance(metadata.source_checkpoints, tuple)
        assert isinstance(metadata.additional_config_ids, tuple)
        assert metadata.vocabulary_shape is None or isinstance(metadata.vocabulary_shape, tuple)


@pytest.mark.parametrize("name", original.ORIGINAL_MODELS)
def test_original_models_reference_metadata_remains_immutable(name):
    metadata = original.ORIGINAL_MODELS[name]
    with pytest.raises(TypeError):
        original.ORIGINAL_MODELS[name] = metadata
    with pytest.raises(FrozenInstanceError):
        metadata.revision = "unreviewed"
    with pytest.raises(TypeError):
        metadata.dtype_bytes["BF16"] = 4
    with pytest.raises(TypeError):
        metadata.source_checkpoints[0] = "unreviewed"
    if metadata.additional_config_ids:
        with pytest.raises(TypeError):
            metadata.additional_config_ids[0] = "unreviewed"
    if metadata.vocabulary_shape is not None:
        with pytest.raises(TypeError):
            metadata.vocabulary_shape[0] = 1


def test_original_models_loader_uses_data_instead_of_a_python_model_table(tmp_path):
    record = json.loads(original.ORIGINAL_MODELS_PATH.read_text(encoding="utf-8"))["Qwen/Qwen3-8B"]
    record.update(model_name="unit/model", source_checkpoints=["unit/model"])
    path = tmp_path / "original-models.json"
    path.write_text(json.dumps({"unit/model": record}), encoding="utf-8")
    models = original._load_original_models(path)
    assert set(models) == {"unit/model"}
    assert models["unit/model"].source_checkpoints == ("unit/model",)
    assert models["unit/model"].revision == record["revision"]


@pytest.mark.parametrize("name", original.ORIGINAL_MODELS)
def test_original_models_preserve_source_aliases_and_config_restrictions(name):
    metadata = original.ORIGINAL_MODELS[name]
    config_ids = (metadata.config_id or "any-config", *metadata.additional_config_ids)
    for config_id in config_ids:
        cfg = SimpleNamespace(model_name=name, config_id=config_id)
        for source in metadata.source_checkpoints:
            spec = SimpleNamespace(model=SimpleNamespace(source_checkpoint=source))
            assert original._original_model(cfg, spec) is metadata
        spec = SimpleNamespace(model=SimpleNamespace(source_checkpoint="unreviewed/source"))
        with pytest.raises(ValueError, match="local full model mode requires"):
            original._original_model(cfg, spec)
    if metadata.config_id is not None:
        cfg = SimpleNamespace(model_name=name, config_id="unreviewed-config")
        spec = SimpleNamespace(model=SimpleNamespace(source_checkpoint=metadata.source_checkpoints[0]))
        with pytest.raises(ValueError, match="local full model mode requires"):
            original._original_model(cfg, spec)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_name", "different/model"),
        ("source_checkpoints", []),
        ("source_checkpoints", [""]),
        ("source_checkpoints", "unit/model"),
        ("revision", 123),
        ("tensor_count", "311"),
        ("tensor_count", True),
        ("dtype_bytes", {"BF16": "2"}),
        ("vocabulary_shape", [151936, "1024"]),
        ("additional_config_ids", [123]),
        ("layer_pattern", "["),
        ("unknown_field", "unsupported"),
    ],
)
def test_original_models_loader_rejects_invalid_metadata(tmp_path, field, value):
    raw = json.loads(original.ORIGINAL_MODELS_PATH.read_text(encoding="utf-8"))
    record = raw["Qwen/Qwen3-0.6B"]
    record[field] = value
    path = tmp_path / "original-models.json"
    path.write_text(json.dumps({"Qwen/Qwen3-0.6B": record}), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot load original model reference metadata") as error:
        original._load_original_models(path)
    assert str(path) in str(error.value)


@pytest.mark.parametrize(
    "raw",
    ["{", "[]", "{}", '{"model": {}, "model": {}}', '{"model": NaN}', '{"model": {"revision": "a", "revision": "b"}}'],
)
def test_original_models_loader_rejects_invalid_registry_json(tmp_path, raw):
    path = tmp_path / "original-models.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="cannot load original model reference metadata"):
        original._load_original_models(path)


def test_original_models_loader_reports_missing_reference_file(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="cannot load original model reference metadata") as error:
        original._load_original_models(path)
    assert str(path) in str(error.value)
    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_original_models_reference_file_is_inside_the_correctness_package():
    assert original.ORIGINAL_MODELS_PATH == (
        Path(original.__file__).resolve().parents[1] / "reference_data" / "original-models.json"
    )
