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

"""Pinned original-model admission; registry membership does not confer test applicability."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from arctic_platform.client import CortexConfig

from ..harness.config import LoadedConfig
from ..harness.spec import TestSpec
from ..harness.workdir import require_data_fast_path


@dataclass(frozen=True)
class OriginalModelMetadata:
    model_name: str
    config_id: str | None
    source_checkpoints: tuple[str, ...]
    revision: str
    config_sha256: str
    index_sha256: str | None
    layer_pattern: str
    tensor_count: int | None
    shard_count: int | None
    payload_bytes: int | None
    vocabulary_shape: tuple[int, ...] | None
    schema_sha256: str | None
    dtype_bytes: Mapping[str, int]


ORIGINAL_MODELS: Mapping[str, OriginalModelMetadata] = MappingProxyType(
    {
        "Qwen/Qwen3-8B": OriginalModelMetadata(
            model_name="Qwen/Qwen3-8B",
            config_id=None,
            source_checkpoints=("Qwen/Qwen3-8B",),
            revision="b968826d9c46dd6066d109eabc6255188de91218",
            config_sha256="f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
            index_sha256=None,
            layer_pattern=r"model\.layers\.(\d+)\.",
            tensor_count=None,
            shard_count=None,
            payload_bytes=None,
            vocabulary_shape=None,
            schema_sha256=None,
            dtype_bytes=MappingProxyType({}),
        ),
        "Qwen/Qwen3.8-27B": OriginalModelMetadata(
            model_name="Qwen/Qwen3.8-27B",
            config_id="qwen3.8-27b-b200-train-sft-8gpus-2k",
            source_checkpoints=("Qwen/Qwen3.8-27B", "/data-fast/base-models/Qwen/Qwen3.8-27B"),
            revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            config_sha256="191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
            index_sha256="77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
            layer_pattern=r"model\.language_model\.layers\.(\d+)\.",
            tensor_count=1199,
            shard_count=18,
            payload_bytes=55562855904,
            vocabulary_shape=(248320, 5120),
            schema_sha256="9c0eb904ed294a62882adb287b81e36fdb48e5fb8e006ee25b084341ac78fd55",
            dtype_bytes=MappingProxyType({"BF16": 2, "F32": 4}),
        ),
        "Qwen/Qwen3.8-Flash-Next": OriginalModelMetadata(
            model_name="Qwen/Qwen3.8-Flash-Next",
            config_id="qwen3.8-flash-next-h200-train-sft-full-16gpus-sp16-256k",
            source_checkpoints=(
                "Qwen/Qwen3.8-Flash-Next",
                (
                    "/checkpoint/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/"
                    "snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540"
                ),
            ),
            revision="de4b8e4d43b917e7706784d8bb445c9af86a3540",
            config_sha256="889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b",
            index_sha256="99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de",
            layer_pattern=r"model\.language_model\.layers\.(\d+)\.",
            tensor_count=1658,
            shard_count=131,
            payload_bytes=359999963128,
            vocabulary_shape=(248320, 2560),
            schema_sha256="1d55a6d3f64b1b2b004302864c0e74e11871ddf9f3481b10fc21fc8b0ec53d26",
            dtype_bytes=MappingProxyType({"BF16": 2, "I64": 8}),
        ),
    }
)


def _original_model(cfg: LoadedConfig, spec: TestSpec) -> OriginalModelMetadata:
    metadata = ORIGINAL_MODELS.get(cfg.model_name)
    if (
        metadata is None
        or (metadata.config_id is not None and metadata.config_id != cfg.config_id)
        or spec.model.source_checkpoint not in metadata.source_checkpoints
    ):
        raise ValueError("local full model mode requires a supported pinned original source")
    return metadata


def validate_local_full_model(backend: CortexConfig, cfg: LoadedConfig, spec: TestSpec) -> Path:
    """Resolve the explicitly supported original model offline, without inventing a reduced binding."""
    from safetensors import safe_open
    from transformers.utils.hub import cached_file

    from ..harness.hosted import require_local_cortex_gateway

    require_local_cortex_gateway(backend)
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("DSS_ALLOW_REMOTE_MODELS") != "1":
        raise ValueError("local full model requires offline HF cache and DSS_ALLOW_REMOTE_MODELS=1")
    metadata = _original_model(cfg, spec)
    revision = metadata.revision
    config_sha256 = metadata.config_sha256
    index_sha256 = metadata.index_sha256
    layer_pattern = metadata.layer_pattern
    tensor_count, shard_count, payload_bytes = metadata.tensor_count, metadata.shard_count, metadata.payload_bytes
    vocabulary_shape = list(metadata.vocabulary_shape) if metadata.vocabulary_shape is not None else None
    schema_sha256 = metadata.schema_sha256
    dtype_bytes = metadata.dtype_bytes
    cache = require_data_fast_path(os.environ.get("HF_HUB_CACHE", ""), label="offline full model cache")
    config_file = cached_file(cfg.model_name, "config.json", local_files_only=True)
    if not isinstance(config_file, str):
        raise ValueError("original full model config is not present in the offline cache")
    root = require_data_fast_path(Path(config_file).parent, label="offline full model snapshot")
    if root != cache / ("models--" + cfg.model_name.replace("/", "--")) / "snapshots" / revision:
        raise ValueError("offline full model snapshot does not match the pinned original revision")

    def readable(relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("offline full model has an unsafe relative asset path")
        resolved = require_data_fast_path(root / path, label="offline full model asset")
        if not resolved.is_relative_to(cache) or not resolved.is_file() or resolved.stat().st_size == 0:
            raise ValueError("offline full model asset is missing, empty or outside its cache")
        return resolved

    config_bytes = readable("config.json").read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != config_sha256:
        raise ValueError("offline full model config does not match the pinned original config")
    config = json.loads(config_bytes)
    readable("tokenizer_config.json")
    readable("tokenizer.json")
    index_bytes = readable("model.safetensors.index.json").read_bytes()
    if index_sha256 is not None and hashlib.sha256(index_bytes).hexdigest() != index_sha256:
        raise ValueError("offline full model index does not match the pinned original index")
    index = json.loads(index_bytes)
    mapping = index.get("weight_map") if isinstance(index, dict) else None
    if (
        not isinstance(mapping, dict)
        or not mapping
        or not all(
            isinstance(name, str) and name and isinstance(shard, str) and shard.endswith(".safetensors")
            for name, shard in mapping.items()
        )
    ):
        raise ValueError("offline full model requires an indexed safetensors checkpoint")
    if index_sha256 is not None and (
        len(mapping) != tensor_count
        or index.get("metadata", {}).get("total_size") != payload_bytes
        or set(mapping.values())
        != {f"model-{i:05d}-of-{shard_count:05d}.safetensors" for i in range(1, shard_count + 1)}
    ):
        raise ValueError("offline full model index is missing original tensors or shards")
    keys = set()
    vocabulary_shapes = {}
    tensor_bytes = 0
    tensor_schema = {}
    for shard in sorted(set(mapping.values())):
        with safe_open(readable(shard), framework="pt", device="cpu") as stream:
            actual = set(stream.keys())
            if actual != {name for name, filename in mapping.items() if filename == shard} or keys & actual:
                raise ValueError("offline full model safetensors keys do not match the complete weight index")
            keys.update(actual)
            if index_sha256 is not None:
                # Read tensor headers, not the full weights or GPU tensors.
                for name in actual:
                    view = stream.get_slice(name)
                    shape = view.get_shape()
                    dtype = view.get_dtype()
                    tensor_schema[name] = [dtype, shape]
                    if dtype not in dtype_bytes:
                        raise ValueError("offline full model tensor dtype differs from the original source")
                    elements = 1
                    for size in shape:
                        elements *= size
                    tensor_bytes += elements * dtype_bytes[dtype]
                    if name in {"lm_head.weight", "model.language_model.embed_tokens.weight"}:
                        vocabulary_shapes[name] = shape
    layers = {int(match.group(1)) for name in keys if (match := re.match(layer_pattern, name)) is not None}
    text_config = config.get("text_config", config)
    if layers != set(range(text_config["num_hidden_layers"])):
        raise ValueError("offline full model checkpoint is missing original transformer layers")
    if index_sha256 is not None:
        vision_layers = {
            int(match.group(1))
            for name in keys
            if (match := re.match(r"model\.visual\.blocks\.(\d+)\.", name)) is not None
        }
        if (
            vision_layers != set(range(config["vision_config"]["depth"]))
            or tensor_bytes != payload_bytes
            or vocabulary_shapes
            != {
                "lm_head.weight": vocabulary_shape,
                "model.language_model.embed_tokens.weight": vocabulary_shape,
            }
        ):
            raise ValueError("offline full model checkpoint is missing original vision or vocabulary tensors")
        schema_bytes = json.dumps(tensor_schema, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "ascii"
        )
        if hashlib.sha256(schema_bytes).hexdigest() != schema_sha256:
            raise ValueError("offline full model tensor shapes or dtypes differ from the pinned original source")
    return root
