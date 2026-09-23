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
"""Shared PEFT model construction, before optimizer or DeepSpeed initialization."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    import torch
    from torch import nn


def is_fp8_lora_dtype(dtype: Any) -> bool:
    import torch

    return dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


def model_has_fp8_weights(model: nn.Module) -> bool:
    return any(is_fp8_lora_dtype(param.dtype) for _, param in model.named_parameters())


def is_peft_lora_param(name: str, param: Any) -> bool:
    """Return whether this is a trainable LoRA A/B adapter parameter."""
    return bool(getattr(param, "requires_grad", False)) and (".lora_A." in name or ".lora_B." in name)


def cast_lora_adapters_off_fp8(model: nn.Module, dtype: torch.dtype | None = None) -> int:
    """Keep LoRA A/B in the optimization dtype while preserving frozen FP8 weights."""
    import torch
    from torch import nn

    dtype = torch.bfloat16 if dtype is None else dtype
    count = 0
    for name, param in list(model.named_parameters()):
        if not is_peft_lora_param(name, param) or param.dtype == dtype:
            continue
        parts = name.split(".")
        module = model
        for part in parts[:-1]:
            module = getattr(module, part)
        setattr(module, parts[-1], nn.Parameter(param.detach().to(dtype=dtype), requires_grad=param.requires_grad))
        count += 1
    return count


def _resolve_peft_config_class(peft_module: Any, peft_config: dict[str, Any]) -> Any:
    peft_type = peft_config.get("peft_type")
    if not isinstance(peft_type, str) or not peft_type:
        raise ValueError("peft_config.peft_type must be a non-empty string")
    config_class_name = f"{peft_type}Config"
    if not hasattr(peft_module, config_class_name):
        raise ValueError(f"Unsupported PEFT type {peft_type!r}: peft.{config_class_name} is not available")
    return getattr(peft_module, config_class_name)


def apply_peft(
    model: nn.Module,
    peft_config: dict[str, Any] | None,
    *,
    optimization_dtype: torch.dtype | None = None,
) -> nn.Module:
    """Wrap a model with PEFT and enable backward through frozen input embeddings.

    ``peft_type`` names a PEFT config class without the ``Config`` suffix (e.g.
    ``Lora``). FP8 bases disable PEFT's adapter autocast and cast trainable A/B
    tensors to ``optimization_dtype`` (default bf16) before optimizer flat buffers are built.
    Model-specific expert tagging and tiled-forward hooks belong to the caller.
    """
    if not peft_config:
        return model

    import peft

    config = _resolve_peft_config_class(peft, peft_config)(**peft_config)
    if model_has_fp8_weights(model):
        model = peft.get_peft_model(model, config, autocast_adapter_dtype=False)
        cast_lora_adapters_off_fp8(model, optimization_dtype)
    else:
        model = peft.get_peft_model(model, config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model
