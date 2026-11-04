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

from copy import copy
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from arctic_platform.model.implementations.moe.preserved_weights import PreservedWeight as _PreservedShardWeight
from arctic_platform.model.implementations.moe.preserved_weights import capture_original_weights
from arctic_platform.model.implementations.moe.preserved_weights import file_identity
from arctic_platform.model.implementations.moe.preserved_weights import open_preserved_weight
from arctic_platform.model.implementations.moe.preserved_weights import read_preserved_weight

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


def _required_layer_names(config, prefix: str, inventory: set[str]) -> set[str]:
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer

    text = copy(getattr(config, "text_config", config))
    index = int(prefix.rstrip(".").split(".")[-1])
    # Prediction layers extend the final decoder layout when the HF config lists only active layers.
    for attribute in ("layer_types", "mlp_layer_types", "indexer_types"):
        values = list(getattr(text, attribute))
        setattr(text, attribute, values + [values[-1]] * max(0, index + 1 - len(values)))
    with torch.device("meta"):
        layer = Glm5NextTextDecoderLayer(text, index)
    suffixes = set(layer.state_dict())
    for stem, target in (("attn_hc", "hc_attn"), ("ffn_hc", "hc_ffn")):
        for field in ("fn", "base", "scale"):
            name = f"{stem}.{field}"
            if name in suffixes:
                suffixes.remove(name)
                suffixes.add(f"{target}_{field}")
    # Shipped checkpoints may store fused experts or one projection per expert.
    packed = "mlp.experts.gate_up_proj"
    if packed in suffixes and prefix + packed not in inventory:
        suffixes.difference_update({packed, "mlp.experts.down_proj"})
        suffixes.update(
            f"mlp.experts.{expert}.{projection}.weight"
            for expert in range(text.n_routed_experts)
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
    return {prefix + suffix for suffix in suffixes | {"eh_proj.weight", "enorm.weight", "hnorm.weight"}}


def _capture(source_path: Path, prefixes: tuple[str, ...], config):
    preserved: list[_PreservedShardWeight] = capture_original_weights(source_path, None, prefixes=prefixes)
    expected = {weight.name for weight in preserved}
    for prefix in prefixes:
        required = _required_layer_names(config, prefix, expected)
        if not required <= expected:
            raise ValueError(f"GLM original prediction layer is incomplete: missing={sorted(required - expected)}")
    index_identity = preserved[0].index_identity if preserved else None
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
            names, weights, index_identity = _capture(source_path, prefixes, model.config)
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
    if index_identity is not None and file_identity(index_identity[0].stat()) != index_identity[1]:
        raise RuntimeError("Preserved GLM prediction-layer index changed")
    for weight in weights:
        with open_preserved_weight(weight):
            pass


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
        yield weight.name, read_preserved_weight(weight)


def preserved_auxiliary_names(model: nn.Module):
    state = getattr(model, STATE_ATTRIBUTE, None)
    if model.__dict__.get("_ap_weights_random_initialized", False) or state is None:
        return ()
    return state[1]
