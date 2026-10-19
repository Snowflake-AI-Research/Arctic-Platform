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

"""Header-only HF export controls for dense and MoE text projections."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import safetensors

from arctic_platform.correctness.harness import inference_export as projection


@pytest.fixture(params=["qwen3_5", "qwen3_5_moe"])
def case(request):
    family = request.param
    original_class, text_type, exported_class = projection.FAMILIES[family]
    text = {
        "model_type": text_type,
        "num_hidden_layers": 2,
        "hidden_size": 4,
        "intermediate_size": 8,
        "vocab_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "tie_word_embeddings": False,
        "dtype": "bfloat16",
        "use_cache": True,
        "layer_types": ["linear_attention", "full_attention"],
        "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25},
        "linear_conv_kernel_dim": 4,
        "mtp_num_hidden_layers": 1,
    }
    if family.endswith("_moe"):
        text["num_experts"] = 4
        text["num_experts_per_tok"] = 2
    source = {
        "model_type": family,
        "architectures": [original_class],
        "text_config": text,
        "vision_config": {"depth": 1},
    }
    exported = {
        **copy.deepcopy(text),
        "architectures": [exported_class],
        "transformers_version": "synthetic-only",
        "use_cache": False,
    }
    schema = {
        "model.language_model.layers.0.norm.weight": ["F32", [4]],
        "model.language_model.layers.1.norm.weight": ["BF16", [4]],
        "model.language_model.embed_tokens.weight": ["BF16", [8, 4]],
        "lm_head.weight": ["BF16", [8, 4]],
        "model.visual.block.weight": ["BF16", [4, 4]],
        "mtp.fc.weight": ["BF16", [4, 8]],
    }
    if family.endswith("_moe"):
        schema["model.language_model.layers.0.mlp.experts.gate_up_proj"] = ["BF16", [4, 16, 4]]
    output = {
        name: ["BF16", list(row[1])]
        for name, row in schema.items()
        if name.startswith("model.language_model.") or name == "lm_head.weight"
    }
    return source, exported, schema, output


def test_named_projection_uses_original_not_job12_dimensions(case):
    original = copy.deepcopy(case)
    receipt = projection.validate_text_projection(*case)
    assert receipt["text_tensor_count"] == len(case[3])
    assert receipt["source_tensor_count"] == len(case[2])
    assert receipt["excluded_vision_tensor_count"] == 1
    assert receipt["excluded_mtp_tensor_count"] == 1
    assert receipt["text_layer_count"] == 2
    assert receipt["runtime_admitted"] is False
    assert receipt["tensor_payloads_read"] is False
    assert receipt["numerical_pass"] is None
    assert case == original


@pytest.mark.parametrize("family", ["qwen3", "qwen3_moe", "qwen3_5_text", "qwen3_5_moe_text"])
def test_ordinary_family_delegates_unchanged(family):
    assert projection.validate_text_projection({"model_type": family}, {}, None, None) is None


def test_unchanged_composite_delegates_without_mutation(case):
    source = case[0]
    exported = copy.deepcopy(source)
    original = copy.deepcopy((source, exported))
    assert projection.validate_text_projection(source, exported, None, None) is None
    assert (source, exported) == original


def test_scalar_text_tensor_retains_exact_shape_and_bytes(case):
    source, exported, schema, output = case
    name = "model.language_model.layers.0.scale"
    schema[name] = ["F32", []]
    output[name] = ["BF16", []]
    receipt = projection.validate_text_projection(source, exported, schema, output)
    assert receipt["export_payload_bytes"] == projection.schema_bytes(output)
    assert projection.schema_bytes({name: ["BF16", []]}) == 2
    output[name][1] = [1]
    with pytest.raises(ValueError, match="shapes"):
        projection.validate_text_projection(source, exported, schema, output)


def test_unknown_projection_rejected(case):
    source, exported, schema, output = case
    source["model_type"] = "unknown_composite"
    with pytest.raises(ValueError, match="unknown"):
        projection.validate_text_projection(source, exported, schema, output)


@pytest.mark.parametrize(
    "failure",
    [
        "source-class",
        "export-class",
        "text-family",
        "reduced-layers",
        "reduced-width",
        "remote-code",
        "rope-drift",
        "added-field",
        "missing-field",
        "bool-dimension",
        "cache-type",
        "nonfinite-config",
    ],
)
def test_configuration_fail_closed(case, failure):
    source, exported, _, _ = case
    if failure == "source-class":
        source["architectures"] = ["UnrelatedForConditionalGeneration"]
    elif failure == "export-class":
        exported["architectures"] = source["architectures"]
    elif failure == "text-family":
        exported["model_type"] = "qwen3"
    elif failure == "reduced-layers":
        exported["num_hidden_layers"] = 1
    elif failure == "reduced-width":
        exported["hidden_size"] = 2
    elif failure == "remote-code":
        exported["auto_map"] = {"AutoModel": "untrusted.Model"}
    elif failure == "rope-drift":
        exported["rope_parameters"]["rope_theta"] = 10000
    elif failure == "added-field":
        exported["unreviewed_switch"] = True
    elif failure == "missing-field":
        exported.pop("linear_conv_kernel_dim")
    elif failure == "bool-dimension":
        exported["num_key_value_heads"] = True
    elif failure == "cache-type":
        exported["use_cache"] = 1
    else:
        exported["rope_parameters"]["rope_theta"] = float("nan")
    with pytest.raises(ValueError):
        projection.validate_text_projection(*case)


@pytest.mark.parametrize(
    "failure",
    [
        "missing-text",
        "extra-text",
        "vision-export",
        "mtp-export",
        "rename",
        "same-byte-shape",
        "fp32",
        "bool-shape",
        "unknown-source",
        "source-layer-missing",
    ],
)
def test_header_schema_coverage_fail_closed(case, failure):
    _, _, source, exported = case
    name = "model.language_model.layers.1.norm.weight"
    if failure == "missing-text":
        exported.pop(name)
    elif failure == "extra-text":
        exported["model.language_model.extra.weight"] = ["BF16", [4]]
    elif failure == "vision-export":
        exported["model.visual.block.weight"] = ["BF16", [4, 4]]
    elif failure == "mtp-export":
        exported["mtp.fc.weight"] = ["BF16", [4, 8]]
    elif failure == "rename":
        exported["model.layers.1.norm.weight"] = exported.pop(name)
    elif failure == "same-byte-shape":
        exported[name][1] = [2, 2]
    elif failure == "fp32":
        exported[name] = ["F32", [4]]
    elif failure == "bool-shape":
        exported[name][1] = [True, 4]
    elif failure == "unknown-source":
        source["unreviewed.weight"] = ["BF16", [4]]
    else:
        source.pop(name)
        exported.pop(name)
    with pytest.raises(ValueError):
        projection.validate_text_projection(*case)


def test_additional_model_fields_are_not_exempt(case):
    source, exported, schema, output = case
    field = "num_experts" if source["model_type"].endswith("_moe") else "linear_conv_kernel_dim"
    exported[field] = 2
    with pytest.raises(ValueError, match=field):
        projection.validate_text_projection(source, exported, schema, output)


@pytest.fixture
def headers(tmp_path, monkeypatch):
    root = tmp_path.resolve() / "weights"
    root.mkdir()
    path = root / "weights.safetensors"
    path.write_bytes(b"synthetic-header-only")
    schema = {"model.norm.weight": ["BF16", [4]]}
    keys = ["model.norm.weight"]

    def safe_open(filename, **kwargs):
        assert Path(filename) == path
        assert kwargs == {"framework": "pt", "device": "cpu"}

        class Headers:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def keys(self):
                return list(keys)

            def get_slice(self, name):
                dtype, shape = schema[name]
                return SimpleNamespace(get_dtype=lambda: dtype, get_shape=lambda: list(shape))

            def get_tensor(self, name):
                pytest.fail("header admission must not materialize tensors")

        return Headers()

    monkeypatch.setattr(safetensors, "safe_open", safe_open)
    index = {"weight_map": {"model.norm.weight": path.name}, "metadata": {"total_size": 8}}
    return SimpleNamespace(root=root, path=path, schema=schema, keys=keys, index=index)


@pytest.mark.parametrize("count", [8, 8.0])
def test_header_inventory_without_values(headers, count):
    headers.index["metadata"]["total_size"] = count
    assert projection.read_inventory(headers.root, headers.index) == headers.schema


def test_scalar_header_inventory_without_values(headers):
    headers.schema["model.norm.weight"][1] = []
    headers.index["metadata"]["total_size"] = 2
    assert projection.read_inventory(headers.root, headers.index) == headers.schema


@pytest.mark.parametrize("count", [True, float("nan"), float("inf"), 8.5])
def test_nonintegral_or_nonfinite_index_bytes_rejected(headers, count):
    headers.index["metadata"]["total_size"] = count
    with pytest.raises(ValueError, match="byte inventory"):
        projection.read_inventory(headers.root, headers.index)


@pytest.mark.parametrize("failure", ["unsafe-shard", "missing-key", "duplicate-key", "wrong-bytes", "symlink"])
def test_index_header_fail_closed(headers, failure):
    if failure == "unsafe-shard":
        headers.index["weight_map"]["model.norm.weight"] = "../escape.safetensors"
    elif failure == "missing-key":
        headers.keys.clear()
    elif failure == "duplicate-key":
        headers.keys.append("model.norm.weight")
    elif failure == "wrong-bytes":
        headers.index["metadata"]["total_size"] = 9
    else:
        stored = headers.path.with_suffix(".stored")
        headers.path.rename(stored)
        headers.path.symlink_to(stored)
    with pytest.raises(ValueError):
        projection.read_inventory(headers.root, headers.index)


def test_duplicate_json_rejected(tmp_path):
    path = tmp_path / "index.json"
    path.write_text('{"weight_map":{},"weight_map":{}}')
    with pytest.raises(ValueError, match="duplicate"):
        projection.read_export_json(path)
