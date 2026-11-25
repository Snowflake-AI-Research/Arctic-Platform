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

"""Marks: properties a plugin attaches to its tensors, which the platform reads at fixed points.

``mark(tensor_or_module, **properties)`` is the one way a plugin tells the platform how to treat a tensor. A mark on
a module applies to every tensor under it; a deeper module's mark, then a tensor's own mark, overrides it. Mark a
module rather than a buffer: moving a model between devices or dtypes replaces buffer objects, and a mark set on the
old object would be lost.

The platform reads marks when it casts the model's dtype (:func:`cast_model`), builds optimizer groups
(:func:`optimizer_groups`), streams weights to the sampler (``weight_sync``) and sets up LoRA
(:func:`lora_target_modules`).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal
from typing import TypeVar

import torch
from pydantic import BaseModel
from pydantic import ConfigDict
from torch import nn

MARKS_ATTR = "_ap_marks"
# The platform's own tensor attributes, which marks feed so the training stack treats plugin tensors like its own:
# arctic_platform.model.implementations.fp8.KEEP_FP32_ATTR (exempt from the bf16 cast at DeepSpeed init) and the
# weight-sync skip flag. Spelled out rather than imported, which would pull in DeepSpeed.
KEEP_FP32_ATTR = "_dss_keep_fp32"
SKIP_WEIGHT_SYNC_ATTR = "_dss_skip_weight_sync"

T = TypeVar("T", torch.Tensor, nn.Module)


class Marks(BaseModel):
    """The properties a tensor can be marked with (contract 1.1). Unset properties take the platform default."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dtype: Literal["fp32"] | None = None
    """``"fp32"``: the tensor stays fp32 when the model is cast to a lower precision."""
    weight_sync: bool = True
    """``False``: kept in checkpoints, left out of the stream to the sampler (trainer-only state)."""
    weight_decay: bool | None = None
    """Whether the optimizer decays this parameter; by default, parameters with two or more dimensions."""
    lora_target: bool = False
    """The module is a LoRA target (modules only). Plugin LoRA is not supported yet (``peft``); this records the
    targets, whose adapters will reach vLLM under the plugin's Hugging Face-to-vLLM name map."""
    frozen: bool = False
    """The parameter is not trained."""


def mark(obj: T, **properties) -> T:
    """Mark a tensor, or every tensor of a module, with ``properties`` (see :class:`Marks`); returns ``obj``."""
    if isinstance(obj, torch.Tensor) and "lora_target" in properties:
        raise ValueError("lora_target marks a module, not a tensor")
    merged = {**getattr(obj, MARKS_ATTR, {}), **properties}
    Marks.model_validate(merged)  # rejects unknown properties and bad values
    setattr(obj, MARKS_ATTR, merged)
    return obj


def skip_weight_sync(obj: T) -> T:
    """``mark(obj, weight_sync=False)``: keep it in checkpoints but out of the weight sync."""
    return mark(obj, weight_sync=False)


def _tensors(module: nn.Module, prefix: str) -> Iterator[tuple[str, torch.Tensor]]:
    # Every name of a shared or tied tensor, as state_dict lists them.
    yield from module.named_parameters(prefix=prefix, remove_duplicate=False)
    yield from module.named_buffers(prefix=prefix, remove_duplicate=False)


def tensor_marks(model: nn.Module) -> dict[str, Marks]:
    """The resolved marks of every parameter and buffer of ``model`` (persistent or not), by name."""
    properties: dict[str, dict] = {name: {} for name, _ in _tensors(model, "")}
    for path, module in model.named_modules(remove_duplicate=False):  # parents before children
        module_marks = getattr(module, MARKS_ATTR, None)
        if module_marks:
            for name, _ in _tensors(module, path):
                properties[name].update(module_marks)
    tensors: dict[int, list[str]] = {}
    for name, tensor in _tensors(model, ""):
        properties[name].update(getattr(tensor, MARKS_ATTR, {}))
        tensors.setdefault(id(tensor), []).append(name)
    for names in tensors.values():  # one tensor under several names (tied weights) must resolve to one set of marks
        if any(properties[name] != properties[names[0]] for name in names):
            raise ValueError(f"{names} are one tensor with different marks: {[properties[name] for name in names]}")
    return {name: Marks.model_validate(values) for name, values in properties.items()}


def _owned(model: nn.Module) -> Iterator[tuple[nn.Module, str, str, bool]]:
    """``(owner, local name, full name, is_parameter)`` for every tensor directly owned by each module."""
    for path, module in model.named_modules(remove_duplicate=False):
        prefix = f"{path}." if path else ""
        for name in list(module._parameters):
            if module._parameters[name] is not None:
                yield module, name, prefix + name, True
        for name in list(module._buffers):
            if module._buffers[name] is not None:
                yield module, name, prefix + name, False


def apply_marks(model: nn.Module) -> None:
    """Write marks into the tensor attributes the training stack reads: fp32 exemption, weight-sync skip, and
    ``requires_grad`` for frozen parameters. Idempotent; the platform calls it wherever tensors are created or
    replaced: after building, after materializing, and after preparing parallelism."""
    marks = tensor_marks(model)
    for owner, name, full, is_parameter in _owned(model):
        tensor = owner._parameters[name] if is_parameter else owner._buffers[name]
        if marks[full].dtype == "fp32":
            setattr(tensor, KEEP_FP32_ATTR, True)
        if not marks[full].weight_sync:
            setattr(tensor, SKIP_WEIGHT_SYNC_ATTR, True)
        if is_parameter and marks[full].frozen:
            tensor.requires_grad_(False)


def cast_model(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    """Cast every floating parameter and buffer to ``dtype``, except tensors marked ``dtype="fp32"``; returns
    ``model``. Parameters stay the same objects, so their attributes survive."""
    marks = tensor_marks(model)
    for owner, name, full, is_parameter in _owned(model):
        tensor = owner._parameters[name] if is_parameter else owner._buffers[name]
        if not tensor.is_floating_point():
            continue
        target = torch.float32 if marks[full].dtype == "fp32" else dtype
        if is_parameter:
            tensor.data = tensor.data.to(target)
        else:
            owner._buffers[name] = tensor.to(target)
    return model


def weight_sync_skipped(model: nn.Module) -> set[str]:
    """Names of the state tensors the weight sync leaves out."""
    marks = tensor_marks(model)
    return {name for name in model.state_dict() if not marks[name].weight_sync}


def optimizer_groups(model: nn.Module) -> list[dict]:
    """Named optimizer param groups of the trainable parameters. ``decay`` groups take the optimizer's own weight
    decay and ``no_decay`` groups none; a parameter decays if its ``weight_decay`` mark says so, by default if it has
    two or more dimensions. fp32-marked parameters get groups of their own (``*_fp32``): ZeRO and DeepSpeed's bf16
    optimizer flatten each group into one buffer of one dtype."""
    marks = tensor_marks(model)
    groups: dict[str, list[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:  # frozen, by mark or otherwise
            continue
        decays = marks[name].weight_decay if marks[name].weight_decay is not None else param.ndim >= 2
        key = ("decay" if decays else "no_decay") + ("_fp32" if marks[name].dtype == "fp32" else "")
        groups.setdefault(key, []).append(param)
    return [
        {"name": key, "params": params, **({} if key.startswith("decay") else {"weight_decay": 0.0})}
        for key, params in sorted(groups.items())
    ]


def lora_target_modules(model: nn.Module) -> list[str]:
    """Paths of the modules marked ``lora_target``. Recorded only: plugin LoRA is not supported yet."""
    return sorted(path for path, module in model.named_modules() if getattr(module, MARKS_ATTR, {}).get("lora_target"))
