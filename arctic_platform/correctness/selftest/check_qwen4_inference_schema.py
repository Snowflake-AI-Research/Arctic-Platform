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

"""Header-only Qwen4 integer-buffer export controls; no model or GPU initialization."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.utils import hub

from arctic_platform.correctness.checks import inference_checkpoint_loss as check
from arctic_platform.correctness.harness import inference_export as export

PREFIX = "model.language_model.layers.0.ple.ple_embedding."


@pytest.fixture
def config():
    return {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "dtype": "bfloat16",
        "transformers_version": "5.17.0",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "use_cache": True,
            "rope_parameters": {"rope_theta": 1000000.0},
            "ple_config": {"ngram_orders": [2, 3], "ngram_vocab_sizes": [8, 16]},
        },
        "vision_config": {"model_type": "qwen4_exp_vision", "depth": 2},
    }


def test_full_nested_config_preserved_without_mutating_source(config):
    original = copy.deepcopy(config)
    target = copy.deepcopy(config)
    target["transformers_version"] = "5.18.0"
    target["text_config"]["use_cache"] = False
    assert export.validate_qwen4_full_config(config, target) is None
    assert config == original


@pytest.mark.parametrize(
    "path",
    [
        ("model_type",),
        ("architectures",),
        ("dtype",),
        ("text_config", "model_type"),
        ("text_config", "hidden_size"),
        ("text_config", "num_hidden_layers"),
        ("text_config", "rope_parameters", "rope_theta"),
        ("text_config", "ple_config", "ngram_orders"),
        ("text_config", "ple_config", "ngram_vocab_sizes"),
        ("vision_config", "model_type"),
        ("vision_config", "depth"),
    ],
)
@pytest.mark.parametrize("failure", ["changed", "missing"])
def test_full_config_rejects_changed_or_missing_original_semantics(config, path, failure):
    target = copy.deepcopy(config)
    container = target
    for name in path[:-1]:
        container = container[name]
    if failure == "missing":
        container.pop(path[-1])
    else:
        container[path[-1]] = None
    with pytest.raises(ValueError, match="configuration (differs|field missing)"):
        export.validate_qwen4_full_config(config, target)


@pytest.mark.parametrize("path", [(), ("text_config",), ("vision_config",), ("text_config", "ple_config")])
def test_full_config_rejects_remote_code_at_each_nested_level(config, path):
    target = copy.deepcopy(config)
    container = target
    for name in path:
        container = container[name]
    container["auto_map"] = {"AutoConfig": "remote.Configuration"}
    with pytest.raises(ValueError, match="without remote code"):
        export.validate_qwen4_full_config(config, target)


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_full_config_cache_setting_must_remain_an_explicit_boolean(config, value):
    target = copy.deepcopy(config)
    target["text_config"]["use_cache"] = value
    with pytest.raises(ValueError, match="explicit boolean"):
        export.validate_qwen4_full_config(config, target)


def test_full_config_rejects_json_scalar_type_substitution(config):
    target = copy.deepcopy(config)
    target["text_config"]["num_hidden_layers"] = True
    with pytest.raises(ValueError, match="configuration differs"):
        export.validate_qwen4_full_config(config, target)


@pytest.mark.parametrize("path", [(), ("text_config",), ("vision_config",), ("text_config", "ple_config")])
@pytest.mark.parametrize("name", ["quantization_config", "unknown_model_option", "extra_remote_config"])
def test_full_config_rejects_export_only_semantic_fields(config, path, name):
    target = copy.deepcopy(config)
    container = target
    for key in path:
        container = container[key]
    container[name] = {"auto_map": {"AutoConfig": "remote.Configuration"}}
    with pytest.raises(ValueError, match="unexpected Qwen4 configuration fields"):
        export.validate_qwen4_full_config(config, target)


def test_full_config_accepts_only_optional_serializer_and_boolean_cache_fields(config):
    target = copy.deepcopy(config)
    target.pop("transformers_version")
    target["use_cache"] = False
    target["vision_config"]["transformers_version"] = "5.18.0"
    target["vision_config"]["use_cache"] = True
    assert export.validate_qwen4_full_config(config, target) is None


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_full_config_new_cache_setting_must_be_boolean(config, value):
    target = copy.deepcopy(config)
    target["use_cache"] = value
    with pytest.raises(ValueError, match="explicit boolean"):
        export.validate_qwen4_full_config(config, target)


@pytest.fixture
def schema():
    return {
        "model.language_model.layers.0.norm.weight": ["BF16", [4]],
        "model.language_model.embed_tokens.weight": ["BF16", [8, 4]],
        **{PREFIX + name: ["I64", [2]] for name in export.QWEN4_INTEGER_BUFFERS},
    }


def test_original_integer_buffer_headers_preserve_dtype_shape_and_bytes(schema):
    original = copy.deepcopy(schema)
    assert export.validate_qwen4_full_schema(schema, copy.deepcopy(schema)) == 120
    assert export.schema_bytes({PREFIX + "layer_multipliers": ["I64", []]}) == 8
    assert schema == original


@pytest.mark.parametrize("name", sorted(export.QWEN4_INTEGER_BUFFERS))
@pytest.mark.parametrize("failure", ["missing", "float", "shape"])
def test_each_integer_buffer_is_required_without_cast_or_reshape(schema, name, failure):
    target = copy.deepcopy(schema)
    key = PREFIX + name
    if failure == "missing":
        target.pop(key)
    elif failure == "float":
        target[key][0] = "BF16"
    else:
        target[key][1] = [1, 2]
    with pytest.raises(ValueError, match="exact original"):
        export.validate_qwen4_full_schema(schema, target)


@pytest.mark.parametrize("failure", ["extra", "rename", "floating-weight", "unrecognized-integer", "buffer-source"])
def test_qwen4_full_inventory_remains_strict(schema, failure):
    target = copy.deepcopy(schema)
    if failure == "extra":
        target["unexpected.weight"] = ["BF16", [1]]
    elif failure == "rename":
        target[PREFIX + "renamed"] = target.pop(PREFIX + "layer_multipliers")
    elif failure == "floating-weight":
        target["model.language_model.layers.0.norm.weight"][0] = "F32"
    elif failure == "unrecognized-integer":
        schema["unrecognized.integer"] = ["I64", [2]]
        target["unrecognized.integer"] = ["I64", [2]]
    else:
        schema[PREFIX + "layer_multipliers"][0] = "F32"
        target[PREFIX + "layer_multipliers"][0] = "BF16"
    with pytest.raises(ValueError):
        export.validate_qwen4_full_schema(schema, target)


def test_floating_source_projection_preserves_integer_buffers(schema):
    target = copy.deepcopy(schema)
    schema["model.language_model.layers.0.norm.weight"][0] = "F32"
    assert export.validate_qwen4_full_schema(schema, target) == 120


@pytest.mark.parametrize("failure", [None, "text", "vision", "ple", "remote-code"])
def test_local_qwen4_export_validates_real_mixed_headers(tmp_path, monkeypatch, config, failure):
    source = tmp_path / "source"
    target = tmp_path / "target"
    values = {
        "model.language_model.layers.0.norm.weight": torch.ones(4, dtype=torch.bfloat16),
        **{
            PREFIX + name: torch.tensor([2**53 + 1, 2**53 + 3], dtype=torch.int64)
            for name in export.QWEN4_INTEGER_BUFFERS
        },
    }
    for root in (source, target):
        root.mkdir()
        (root / "config.json").write_text(json.dumps(config))
        save_file(values, str(root / "weights.safetensors"))
        index = {
            "metadata": {"total_size": sum(value.numel() * value.element_size() for value in values.values())},
            "weight_map": {name: "weights.safetensors" for name in values},
        }
        (root / "model.safetensors.index.json").write_text(json.dumps(index))
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(**config))
    monkeypatch.setattr(hub, "cached_file", lambda model, name, **kwargs: str(source / name))
    ctx = SimpleNamespace(cfg=SimpleNamespace(model_name="Qwen/synthetic-qwen4"))
    before = {name: value.clone() for name, value in values.items()}
    if failure is None:
        assert check.validate_local_inference_export(ctx, target) == target
    else:
        exported = copy.deepcopy(config)
        if failure == "text":
            exported["text_config"]["rope_parameters"]["rope_theta"] = 1.0
        elif failure == "vision":
            exported["vision_config"]["depth"] = 1
        elif failure == "ple":
            exported["text_config"]["ple_config"]["ngram_vocab_sizes"] = [16, 8]
        else:
            exported["text_config"]["auto_map"] = {"AutoConfig": "remote.Configuration"}
        (target / "config.json").write_text(json.dumps(exported))
        with pytest.raises(ValueError):
            check.validate_local_inference_export(ctx, target)
    for name, value in values.items():
        assert torch.equal(value, before[name])
