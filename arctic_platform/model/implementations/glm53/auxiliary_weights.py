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
import os
import stat
from pathlib import Path

import torch.distributed as dist
from torch import nn

from arctic_platform.model.implementations.qwen35.mtp_weights import _capture_original_weights
from arctic_platform.model.implementations.qwen35.mtp_weights import _file_identity
from arctic_platform.model.implementations.qwen35.mtp_weights import _unique_json_object

STATE_ATTRIBUTE = "_ap_preserved_glm_auxiliary_weights"


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
    fd = os.open(index_path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or _file_identity(os.stat(index_path)) != _file_identity(metadata):
            raise ValueError("GLM auxiliary checkpoint index must be a stable regular file")
        with os.fdopen(os.dup(fd), encoding="utf-8") as stream:
            index = json.load(stream, object_pairs_hook=_unique_json_object)
        mapping = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(mapping, dict) or not all(isinstance(name, str) for name in mapping):
            raise ValueError("GLM auxiliary checkpoint requires an indexed tensor inventory")
        expected = {name for name in mapping if name.startswith(prefixes)}
        for prefix in prefixes:
            required = {prefix + suffix for suffix in ("eh_proj.weight", "enorm.weight", "hnorm.weight")}
            if not required <= expected:
                raise ValueError(f"GLM original prediction layer is incomplete: {prefix}")
        preserved = _capture_original_weights(source_path, expected, prefixes=prefixes)
        if _file_identity(os.fstat(fd)) != _file_identity(metadata) or _file_identity(
            os.stat(index_path)
        ) != _file_identity(metadata):
            raise ValueError("GLM auxiliary checkpoint index changed during capture")
        return tuple(sorted(expected)), tuple(sorted(preserved, key=lambda weight: weight.name))
    finally:
        os.close(fd)


def capture_auxiliary_weights(model: nn.Module, source_path: Path) -> None:
    """Capture the original inactive layers on rank zero before HF conversion."""
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
            names, weights = _capture(source_path, prefixes)
            state = (prefixes, names, weights)
        except Exception as exc:
            cause = exc
            failure[0] = f"{type(exc).__name__}: {exc}"
    if distributed:
        dist.broadcast_object_list(failure, src=0)
    if failure[0] is not None:
        raise RuntimeError(f"Cannot preserve original GLM prediction layers: {failure[0]}") from cause
    if is_master:
        setattr(model, STATE_ATTRIBUTE, state)


def iter_preserved_auxiliary_weights(model: nn.Module):
    """Export the captured schema unchanged, without registering trainable tensors."""
    prefixes = _layer_prefixes(getattr(model, "config", None))
    if not prefixes:
        return
    state = getattr(model, STATE_ATTRIBUTE, None)
    if state is None or state[0] != prefixes:
        raise RuntimeError("Full GLM HF export requires captured original prediction layers")
    _, names, weights = state
    actual = tuple(weight.name for weight in weights)
    if actual != names or len(set(actual)) != len(actual) or not all(name.startswith(prefixes) for name in actual):
        raise RuntimeError("Preserved GLM prediction-layer inventory changed")
    for weight in weights:
        tensor = weight.tensor
        if (
            tuple(tensor.shape) != weight.shape
            or tensor.dtype != weight.dtype
            or tensor.device.type != "cpu"
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise RuntimeError(f"Preserved GLM prediction tensor changed layout or dtype: {weight.name}")
        yield weight.name, tensor
