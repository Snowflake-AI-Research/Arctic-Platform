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

"""Preserve inactive Qwen prediction weights for full HF checkpoint export."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from arctic_platform.model.implementations.moe.preserved_weights import PreservedWeight as _PreservedWeight
from arctic_platform.model.implementations.moe.preserved_weights import (
    capture_original_weights as _capture_original_weights,
)
from arctic_platform.model.implementations.moe.preserved_weights import open_preserved_weight
from arctic_platform.model.implementations.moe.preserved_weights import read_preserved_weight

_STATE_ATTRIBUTE = "_ap_preserved_mtp_weights"
_BASE_NAMES = (
    "fc.weight",
    "norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)
_LAYER_NAMES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate.weight",
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
    "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert.down_proj.weight",
    "mlp.shared_expert_gate.weight",
)
_QWEN4_BASE_NAMES = (
    "fc_embedding.weight",
    "fc_hidden.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
    "hyper_connection_mixer.hc_norm.weight",
    "hyper_connection_mixer.input_mix_weight_down.weight",
    "hyper_connection_mixer.input_mix_weight_up.weight",
)
_QWEN4_LAYER_NAMES = (
    "attn_hyper_connection.block_inject_weight.weight",
    "attn_hyper_connection.hc_norm.weight",
    "attn_hyper_connection.input_mix_weight_down.weight",
    "attn_hyper_connection.input_mix_weight_up.weight",
    "mlp_hyper_connection.block_inject_weight.weight",
    "mlp_hyper_connection.hc_norm.weight",
    "mlp_hyper_connection.input_mix_weight_down.weight",
    "mlp_hyper_connection.input_mix_weight_up.weight",
    "self_attn.indexer.index_qk_proj.weight",
    "self_attn.indexer.k_layernorm.weight",
    "self_attn.indexer.q_layernorm.weight",
    *_LAYER_NAMES[2:],
)


@dataclass(frozen=True)
class _PreservedExport:
    weights: Iterable[tuple[str, torch.Tensor]]
    names: frozenset[str]


def _configured_names(model: nn.Module) -> set[str]:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    model_type = getattr(text_config, "model_type", None)
    if model_type == "qwen3_5_moe_text":
        base_names, layer_names = _BASE_NAMES, _LAYER_NAMES
    elif model_type == "qwen4_exp_text":
        base_names, layer_names = _QWEN4_BASE_NAMES, _QWEN4_LAYER_NAMES
    else:
        return set()
    count = getattr(text_config, "mtp_num_hidden_layers", 0)
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("Qwen mtp_num_hidden_layers must be a nonnegative integer")
    if count == 0:
        return set()
    return {f"mtp.{name}" for name in base_names} | {
        f"mtp.layers.{index}.{name}" for index in range(count) for name in layer_names
    }


def capture_mtp_weights(model: nn.Module, source_path: Path) -> None:
    """Record rank zero's immutable source locations without retaining tensor payloads."""
    model.__dict__.pop(_STATE_ATTRIBUTE, None)
    expected = _configured_names(model)
    if not expected:
        return

    distributed = dist.is_initialized()
    is_master = not distributed or dist.get_rank() == 0
    preserved: list[_PreservedWeight] = []
    failure: list[str | None] = [None]
    cause = None
    if is_master:
        try:
            preserved = _capture_original_weights(source_path, expected)
        except Exception as exc:
            cause = exc
            failure[0] = f"{type(exc).__name__}: {exc}"

    # A source failure on the serializer rank must not strand peers in the following DCP barrier.
    if distributed:
        dist.broadcast_object_list(failure, src=0)
    if failure[0] is not None:
        raise RuntimeError(f"Cannot preserve original Qwen MTP weights: {failure[0]}") from cause
    if is_master:
        setattr(model, _STATE_ATTRIBUTE, tuple(sorted(preserved, key=lambda weight: weight.name)))


def prepare_preserved_weights_for_export(model: nn.Module, iterator, *, description: str, validate, names):
    """Validate rank zero's sidecar before any ordinary export collective."""
    distributed = dist.is_initialized()
    is_master = not distributed or dist.get_rank() == 0
    weights = _PreservedExport((), frozenset())
    failure: list[str | None] = [None]
    cause = None
    if is_master:
        try:
            validate(model)
            weights = _PreservedExport(iterator(model), frozenset(names(model)))
        except Exception as exc:
            cause = exc
            failure[0] = f"{type(exc).__name__}: {exc}"
    if distributed:
        dist.broadcast_object_list(failure, src=0)
    if failure[0] is not None:
        raise RuntimeError(f"Cannot export preserved {description}: {failure[0]}") from cause
    return weights


def finish_preserved_weight_export(preserved, emitted: set[str], *, description: str):
    """Reach completion on sidecar errors or close; every rank must drain or close the export iterator."""
    distributed = dist.is_initialized()
    is_master = not distributed or dist.get_rank() == 0
    failure: list[str | None] = [None]
    cause = None
    if is_master:
        try:
            collisions = preserved.names & emitted
            if collisions:
                raise RuntimeError(f"{description} tensor collides with live HF export: {sorted(collisions)}")
        except Exception as exc:
            cause = exc
            failure[0] = f"{type(exc).__name__}: {exc}"
    if distributed:
        dist.broadcast_object_list(failure, src=0)
    if failure[0] is not None:
        raise RuntimeError(f"Cannot finish preserved {description} export: {failure[0]}") from cause
    try:
        if is_master:
            for name, tensor in preserved.weights:
                yield name, tensor
    except BaseException as exc:
        cause = exc
        failure[0] = f"{type(exc).__name__}: sidecar export interrupted: {exc}"
        if not isinstance(exc, Exception):
            raise
    finally:
        if distributed:
            dist.broadcast_object_list(failure, src=0)
    if failure[0] is not None:
        raise RuntimeError(f"Cannot finish preserved {description} export: {failure[0]}") from cause


def preserved_mtp_names(model: nn.Module):
    if model.__dict__.get("_ap_weights_random_initialized", False):
        return ()
    return (weight.name for weight in getattr(model, _STATE_ATTRIBUTE, ()))


def validate_preserved_mtp_weights(model: nn.Module) -> None:
    """Validate metadata and original source generations before ordinary export collectives."""
    if model.__dict__.get("_ap_weights_random_initialized", False):
        return
    expected = _configured_names(model)
    if not expected:
        return
    preserved = getattr(model, _STATE_ATTRIBUTE, ())
    names = [weight.name for weight in preserved]
    if len(names) != len(set(names)) or set(names) != expected:
        raise RuntimeError("Full Qwen HF export requires captured original MTP weights")
    for weight in preserved:
        with open_preserved_weight(weight):
            pass


def iter_preserved_mtp_weights(model: nn.Module):
    """Map only the current exported tensor; the original checkpoint must remain immutable."""
    validate_preserved_mtp_weights(model)
    if model.__dict__.get("_ap_weights_random_initialized", False) or not _configured_names(model):
        return
    for weight in getattr(model, _STATE_ATTRIBUTE, ()):
        yield weight.name, read_preserved_weight(weight)
