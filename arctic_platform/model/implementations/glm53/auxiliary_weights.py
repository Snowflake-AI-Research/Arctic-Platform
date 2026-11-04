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

"""Retain inactive GLM prediction layers for complete HF checkpoint export."""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn

from arctic_platform.model.implementations.qwen35.mtp_weights import _file_identity
from arctic_platform.model.implementations.qwen35.mtp_weights import _unique_json_object

STATE_ATTRIBUTE = "_ap_preserved_glm_auxiliary_weights"


@dataclass(frozen=True)
class _PreservedShardWeight:
    name: str
    path: Path
    identity: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: torch.dtype


def _layer_prefixes(config) -> tuple[str, ...]:
    text = getattr(config, "text_config", config)
    if getattr(text, "model_type", None) != "glm5_next_text":
        return ()
    count = getattr(text, "num_nextn_predict_layers", 0)
    layers = getattr(text, "num_hidden_layers", None)
    if type(count) is not int or count < 0 or type(layers) is not int or layers <= 0:
        raise ValueError("GLM layer and prediction-layer counts must be nonnegative integers")
    return tuple(f"model.language_model.layers.{index}." for index in range(layers, layers + count))


def _capture(source_path: Path, prefixes: tuple[str, ...]):
    index_path = source_path / "model.safetensors.index.json"
    index_identity = None
    mapping: dict[str, str] = {}
    if index_path.exists():
        metadata = index_path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("GLM auxiliary checkpoint index must be a regular file")
        index_identity = (index_path, _file_identity(metadata))
        with index_path.open(encoding="utf-8") as stream:
            index = json.load(stream, object_pairs_hook=_unique_json_object)
        indexed_mapping = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(indexed_mapping, dict) or not all(isinstance(name, str) for name in indexed_mapping):
            raise ValueError("GLM auxiliary checkpoint requires an indexed tensor inventory")
        mapping = indexed_mapping
    else:
        for shard in sorted(source_path.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as reader:
                for name in reader.keys():
                    if name in mapping:
                        raise ValueError(f"Duplicate original GLM tensor: {name}")
                    mapping[name] = shard.name
    expected = {name for name in mapping if name.startswith(prefixes)}
    for prefix in prefixes:
        required = {prefix + suffix for suffix in ("eh_proj.weight", "enorm.weight", "hnorm.weight")}
        if not required <= expected:
            raise ValueError(f"GLM original prediction layer is incomplete: {prefix}")
    shards: dict[Path, list[str]] = {}
    for name in sorted(expected):
        filename = mapping[name]
        if (
            not isinstance(filename, str)
            or not filename
            or "\\" in filename
            or Path(filename).is_absolute()
            or ".." in Path(filename).parts
            or Path(filename).suffix != ".safetensors"
        ):
            raise ValueError(f"Unsafe original GLM shard path: {filename!r}")
        shards.setdefault(source_path / filename, []).append(name)
    preserved = []
    for shard, names in shards.items():
        metadata = shard.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Original GLM shard is not a regular file: {shard}")
        identity = _file_identity(metadata)
        with safe_open(shard, framework="pt", device="cpu") as reader:
            for name in names:
                # get_tensor creates a mapped view; retain only metadata, never the inactive layer's payload.
                tensor = reader.get_tensor(name)
                preserved.append(_PreservedShardWeight(name, shard, identity, tuple(tensor.shape), tensor.dtype))
        if _file_identity(shard.stat()) != identity:
            raise ValueError(f"Original GLM shard changed during capture: {shard}")
    if index_identity is not None and _file_identity(index_path.stat()) != index_identity[1]:
        raise ValueError("GLM auxiliary checkpoint index changed during capture")
    return tuple(sorted(expected)), tuple(sorted(preserved, key=lambda weight: weight.name)), index_identity


def capture_auxiliary_weights(model: nn.Module, source_path: Path) -> None:
    """Record original inactive-layer locations on rank zero before HF conversion."""
    model.__dict__.pop(STATE_ATTRIBUTE, None)
    prefixes = _layer_prefixes(getattr(model, "config", None))
    if not prefixes:
        return
    distributed = dist.is_initialized()
    is_master = not distributed or dist.get_rank() == 0
    state = None
    failure: list[str | None] = [None]
    cause = None
    if is_master:
        try:
            names, weights, index_identity = _capture(source_path, prefixes)
            state = (prefixes, names, weights, index_identity)
        except Exception as exc:
            cause = exc
            failure[0] = f"{type(exc).__name__}: {exc}"
    if distributed:
        dist.broadcast_object_list(failure, src=0)
    if failure[0] is not None:
        raise RuntimeError(f"Cannot preserve original GLM prediction layers: {failure[0]}") from cause
    if is_master:
        setattr(model, STATE_ATTRIBUTE, state)


def validate_preserved_auxiliary_weights(model: nn.Module) -> None:
    """Check immutable source generations without reading or retaining tensor payloads."""
    if model.__dict__.get("_ap_weights_random_initialized", False):
        return
    prefixes = _layer_prefixes(getattr(model, "config", None))
    if not prefixes:
        return
    state = getattr(model, STATE_ATTRIBUTE, None)
    if state is None or len(state) != 4 or state[0] != prefixes:
        raise RuntimeError("Full GLM HF export requires captured original prediction layers")
    _, names, weights, index_identity = state
    actual = tuple(weight.name for weight in weights)
    if actual != names or len(set(actual)) != len(actual) or not all(name.startswith(prefixes) for name in actual):
        raise RuntimeError("Preserved GLM prediction-layer inventory changed")
    if index_identity is not None and _file_identity(index_identity[0].stat()) != index_identity[1]:
        raise RuntimeError("Preserved GLM prediction-layer index changed")
    for weight in weights:
        if _file_identity(weight.path.stat()) != weight.identity:
            raise RuntimeError(f"Preserved GLM prediction shard changed: {weight.path}")


def iter_preserved_auxiliary_weights(model: nn.Module):
    """Map one original tensor at a time; the original checkpoint must remain immutable through export."""
    validate_preserved_auxiliary_weights(model)
    state = getattr(model, STATE_ATTRIBUTE, None)
    if model.__dict__.get("_ap_weights_random_initialized", False) or not _layer_prefixes(
        getattr(model, "config", None)
    ):
        return
    if state is None:
        raise RuntimeError("Full GLM HF export requires captured original prediction layers")
    for weight in state[2]:
        with safe_open(weight.path, framework="pt", device="cpu") as reader:
            tensor = reader.get_tensor(weight.name)
            if tuple(tensor.shape) != weight.shape or tensor.dtype != weight.dtype:
                raise RuntimeError(f"Preserved GLM prediction tensor changed layout or dtype: {weight.name}")
            yield weight.name, tensor
