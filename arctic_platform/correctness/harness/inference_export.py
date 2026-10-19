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

"""Named-family text-export validation; source asset admission stays with the caller."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

FAMILIES = {
    "qwen3_5": ("Qwen3_5ForConditionalGeneration", "qwen3_5_text", "Qwen3_5ForCausalLM"),
    "qwen3_5_moe": ("Qwen3_5MoeForConditionalGeneration", "qwen3_5_moe_text", "Qwen3_5MoeForCausalLM"),
}
DIMENSION_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "vocab_size",
    "num_attention_heads",
    "num_key_value_heads",
)
QWEN4_INTEGER_BUFFERS = frozenset({"layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"})


def schema_sha(schema):
    raw = json.dumps(schema, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def schema_bytes(schema):
    if not isinstance(schema, dict) or not schema:
        raise ValueError("nonempty tensor header schema required")
    total = 0
    for name, row in schema.items():
        if not isinstance(name, str) or not name or not isinstance(row, list) or len(row) != 2:
            raise ValueError("invalid tensor header schema")
        dtype, shape = row
        if (
            dtype not in ("BF16", "F32", "I64")
            or (dtype == "I64" and name.rsplit(".", 1)[-1] not in QWEN4_INTEGER_BUFFERS)
            or not isinstance(shape, list)
            or any(type(dim) is not int or dim <= 0 for dim in shape)
        ):
            raise ValueError("invalid tensor dtype or shape")
        total += math.prod(shape) * {"BF16": 2, "F32": 4, "I64": 8}[dtype]
    return total


def validate_qwen4_full_schema(source_schema, export_schema):
    """Preserve original Qwen4 integer buffers while exporting floating weights as BF16."""
    schema_bytes(source_schema)
    schema_bytes(export_schema)
    projected = {}
    for name, (dtype, shape) in source_schema.items():
        integer_buffer = name.rsplit(".", 1)[-1] in QWEN4_INTEGER_BUFFERS
        if integer_buffer and dtype != "I64":
            raise ValueError("original Qwen4 integer buffer must retain I64 dtype: " + name)
        if dtype == "I64" and not integer_buffer:
            raise ValueError("unexpected original Qwen4 integer tensor: " + name)
        projected[name] = ["I64" if integer_buffer else "BF16", shape]
    if export_schema != projected:
        raise ValueError("exact original Qwen4 names, shapes and integer buffer dtypes required")
    return schema_bytes(export_schema)


def _json_identity(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_qwen4_full_config(source_config, export_config):
    """Preserve the original composite configuration, including nested text and vision settings."""
    if (
        not isinstance(source_config, dict)
        or not isinstance(export_config, dict)
        or source_config.get("model_type") != "qwen4_exp"
        or source_config.get("architectures") != ["Qwen4ExpForConditionalGeneration"]
    ):
        raise ValueError("original Qwen4 composite configuration required")
    _json_identity(source_config)
    _json_identity(export_config)

    def compare(original, exported, prefix):
        if not isinstance(exported, dict) or exported.get("auto_map"):
            raise ValueError("local Qwen4 configuration without remote code required: " + prefix)
        extra = set(exported) - set(original) - {"transformers_version", "use_cache"}
        if extra:
            raise ValueError("unexpected Qwen4 configuration fields: " + prefix + ", ".join(sorted(extra)))
        if "use_cache" in exported and type(exported["use_cache"]) is not bool:
            raise ValueError("explicit boolean Qwen4 cache configuration required: " + prefix + "use_cache")
        for field, value in original.items():
            name = prefix + field
            if field == "transformers_version":
                continue
            if field == "use_cache":
                if type(value) is not bool or type(exported.get(field)) is not bool:
                    raise ValueError("explicit boolean Qwen4 cache configuration required: " + name)
                continue
            if field not in exported:
                raise ValueError("original Qwen4 configuration field missing: " + name)
            if isinstance(value, dict):
                compare(value, exported[field], name + ".")
            elif _json_identity(value) != _json_identity(exported[field]):
                raise ValueError("original Qwen4 configuration differs: " + name)

    compare(source_config, export_config, "")


def validate_text_projection(source_config, export_config, source_schema, export_schema):
    """Return None for ordinary exports; validate named text projections against admitted originals."""
    if not isinstance(source_config, dict) or not isinstance(export_config, dict):
        raise ValueError("raw configuration objects required")
    if source_config.get("model_type") == export_config.get("model_type"):
        return None
    family = FAMILIES.get(source_config.get("model_type"))
    if family is None:
        if isinstance(source_config.get("text_config"), dict):
            raise ValueError("unknown composite-to-text projection")
        return None
    source_architecture, text_type, export_architecture = family
    text = source_config.get("text_config")
    if (
        source_config.get("architectures") != [source_architecture]
        or not isinstance(text, dict)
        or text.get("model_type") != text_type
        or export_config.get("model_type") != text_type
        or export_config.get("architectures") != [export_architecture]
        or "text_config" in export_config
        or "vision_config" in export_config
        or any(config.get("auto_map") for config in (source_config, text, export_config))
    ):
        raise ValueError("named original composite-to-text configuration required")
    _json_identity(source_config)
    _json_identity(export_config)
    for field in DIMENSION_FIELDS:
        if (
            type(text.get(field)) is not int
            or text[field] <= 0
            or type(export_config.get(field)) is not int
            or export_config[field] != text[field]
        ):
            raise ValueError("full original text dimension differs: " + field)
    if (
        type(text.get("tie_word_embeddings")) is not bool
        or export_config.get("tie_word_embeddings") is not text["tie_word_embeddings"]
        or text.get("dtype") != "bfloat16"
        or export_config.get("dtype") != "bfloat16"
    ):
        raise ValueError("original text embedding and BF16 configuration required")
    # These are the observed serializer/cache differences, not model-field exemptions.
    allowed = {"architectures", "transformers_version", "use_cache"}
    if set(export_config) - set(text) - allowed:
        raise ValueError("unexpected text export configuration fields")
    for field, value in text.items():
        if field not in allowed and (
            field not in export_config or _json_identity(export_config[field]) != _json_identity(value)
        ):
            raise ValueError("original text configuration differs: " + field)
    if type(text.get("use_cache")) is not bool or type(export_config.get("use_cache")) is not bool:
        raise ValueError("explicit boolean cache configuration required")
    source_bytes = schema_bytes(source_schema)
    text_names = {
        name for name in source_schema if name.startswith("model.language_model.") or name == "lm_head.weight"
    }
    vision_names = {name for name in source_schema if name.startswith("model.visual.")}
    mtp_names = {name for name in source_schema if name.startswith("mtp.")}
    if not text_names or set(source_schema) - text_names - vision_names - mtp_names:
        raise ValueError("source tensors outside original text, named vision and MTP namespaces")
    layers = {
        int(match[1]) for name in text_names if (match := re.match(r"model\.language_model\.layers\.(\d+)\.", name))
    }
    if layers != set(range(text["num_hidden_layers"])):
        raise ValueError("all original text layers required")
    for name in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
        if source_schema.get(name, [None, None])[1] != [text["vocab_size"], text["hidden_size"]]:
            raise ValueError("full original vocabulary tensor shape required")
    if not isinstance(export_schema, dict) or set(export_schema) != text_names:
        raise ValueError("exact original text inventory required; missing or extra export tensor")
    projected = {name: ["BF16", source_schema[name][1]] for name in text_names}
    if export_schema != projected:
        raise ValueError("exact original text shapes and BF16 projection required")
    export_bytes = schema_bytes(export_schema)
    return {
        "contract": source_config["model_type"] + "_composite_to_full_original_text",
        "schema_contract_valid": True,
        "source_schema_sha256": schema_sha(source_schema),
        "export_schema_sha256": schema_sha(export_schema),
        "source_tensor_count": len(source_schema),
        "text_tensor_count": len(text_names),
        "excluded_vision_tensor_count": len(vision_names),
        "excluded_mtp_tensor_count": len(mtp_names),
        "source_payload_bytes": source_bytes,
        "export_payload_bytes": export_bytes,
        "text_layer_count": text["num_hidden_layers"],
        "tensor_payloads_read": False,
        "runtime_admitted": False,
        "numerical_pass": None,
    }


def _fingerprint(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate export metadata key: " + key)
        result[key] = value
    return result


def read_export_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular HF metadata required")
    before = _fingerprint(path)
    with path.open("rb") as stream:
        raw = stream.read(16 * 1024**2 + 1)
    if len(raw) > 16 * 1024**2 or _fingerprint(path) != before:
        raise ValueError("oversized or concurrently changed HF metadata")
    result = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(result, dict):
        raise ValueError("HF metadata object required")
    return result


def read_inventory(root, index):
    """Validate index/header coverage without materializing tensor values."""
    from safetensors import safe_open

    root = Path(root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise ValueError("canonical regular HF directory required")
    if not isinstance(index, dict):
        raise ValueError("HF index object required")
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("nonempty HF tensor index required")
    for tensor, shard in mapping.items():
        if (
            not isinstance(tensor, str)
            or not tensor
            or not isinstance(shard, str)
            or Path(shard).name != shard
            or not shard.endswith(".safetensors")
        ):
            raise ValueError("invalid indexed tensor or shard path")
    shards = set(mapping.values())
    if {path.name for path in root.glob("*.safetensors")} != shards:
        raise ValueError("weights inventory differs from its shard index")
    schema = {}
    for name in sorted(shards):
        path = root / name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
            raise ValueError("regular contained HF weight shard required")
        before = _fingerprint(path)
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            keys = list(stream.keys())
            if (
                len(keys) != len(set(keys))
                or set(keys) != {tensor for tensor, shard in mapping.items() if shard == name}
                or schema.keys() & set(keys)
            ):
                raise ValueError("HF tensor headers disagree with the index")
            for tensor in keys:
                view = stream.get_slice(tensor)
                schema[tensor] = [view.get_dtype(), view.get_shape()]
        if _fingerprint(path) != before:
            raise ValueError("HF tensor header changed during admission")
    total = schema_bytes(schema)
    metadata = index.get("metadata")
    expected = metadata.get("total_size") if isinstance(metadata, dict) else None
    # An original admitted index may encode an integral byte count as a JSON float.
    if type(expected) not in (int, float) or expected != total:
        raise ValueError("HF header byte inventory disagrees with the index")
    return schema
