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
"""PEFT wrapping after model patches and before optimizer construction."""

from __future__ import annotations

import re
from typing import Any

import torch
from torch import nn

from arctic_platform.common.config import validate_peft_config
from arctic_platform.model.implementations.moe.vlm import get_vision_encoder
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.patch import register_patch


def is_fp8_lora_dtype(dtype: Any) -> bool:
    return dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


def model_has_fp8_weights(model: nn.Module) -> bool:
    return any(is_fp8_lora_dtype(param.dtype) for _, param in model.named_parameters())


def is_peft_lora_param(name: str, param: Any) -> bool:
    """Return whether this is a trainable LoRA A/B adapter parameter."""
    return bool(getattr(param, "requires_grad", False)) and (".lora_A." in name or ".lora_B." in name)


def cast_trainable_params_off_fp8(model: nn.Module, dtype: torch.dtype | None = None) -> int:
    """Cast trainables before optimizer construction, preserving frozen FP8 weights and parameter ties."""
    dtype = torch.bfloat16 if dtype is None else dtype
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("FP8 PEFT optimization_dtype must be float16, bfloat16, or float32")
    count = 0
    for _, param in model.named_parameters():
        if not param.requires_grad or param.dtype == dtype:
            continue
        param.data = param.data.to(dtype=dtype)
        count += 1
    return count


def cast_lora_adapters_off_fp8(model: nn.Module, dtype: torch.dtype | None = None) -> int:
    """Compatibility entry point; includes trainable biases, modules_to_save and other adapter types."""
    return cast_trainable_params_off_fp8(model, dtype)


def install_unfused_expert_lora_activation() -> int:
    """Avoid materializing a fused ``[experts, out, in]`` LoRA delta."""
    try:
        from peft.tuners.lora.layer import ParamWrapper
    except ImportError:
        return 0
    if getattr(ParamWrapper, "_ap_skip_fused_expert_delta", False):
        return 1

    from contextlib import contextmanager

    original = ParamWrapper._activate_lora
    ParamWrapper._ap_activate_lora_original = original

    @contextmanager
    def activate_lora(self, active_adapters):
        if getattr(self, "num_experts", 1) > 1:
            yield
            return
        with original(self, active_adapters):
            yield

    ParamWrapper._activate_lora = activate_lora
    ParamWrapper._ap_skip_fused_expert_delta = True
    return 1


def attach_unfused_expert_lora_factors(model: nn.Module) -> int:
    """Attach expert LoRA A/B factors to AP grouped-expert modules."""
    try:
        from peft.tuners.lora.layer import ParamWrapper
    except ImportError:
        return 0

    attached = 0
    for module in model.modules():
        if not isinstance(module, ParamWrapper):
            continue
        parameter_name = getattr(module, "parameter_name", None)
        if parameter_name not in ("w1", "w2", "w3") or getattr(module, "num_experts", 1) <= 1:
            continue
        adapter = getattr(module, "active_adapter", "default")
        if isinstance(adapter, (list, tuple)):
            if len(adapter) > 1:
                raise NotImplementedError(
                    f"unfused expert LoRA supports one active adapter, got {list(adapter)} on {parameter_name}"
                )
            adapter = adapter[0] if adapter else "default"
        if adapter not in module.lora_A or adapter not in module.lora_B:
            continue
        base = module.get_base_layer()
        factors = getattr(base, "_ap_lora_ab", None)
        if factors is None:
            factors = {}
            base._ap_lora_ab = factors
        factors[parameter_name] = (
            module.lora_A[adapter].weight,
            module.lora_B[adapter].weight,
            int(module.r[adapter]),
            float(module.scaling[adapter]),
        )
        attached += 1
    return attached


def _resolve_peft_config_class(peft_module: Any, peft_config: dict[str, Any]) -> Any:
    validate_peft_config(peft_config)
    peft_type = peft_config.get("peft_type")
    registry = getattr(peft_module, "PEFT_TYPE_TO_CONFIG_MAPPING", {})
    if peft_type in registry:
        return registry[peft_type]
    config_class_name = f"{peft_type}Config"
    if not hasattr(peft_module, config_class_name):
        raise ValueError(f"Unsupported PEFT type {peft_type!r}: peft.{config_class_name} is not available")
    return getattr(peft_module, config_class_name)


def exclude_vision_encoder(model: nn.Module, peft_config: dict[str, Any]) -> dict[str, Any]:
    """Return ``peft_config`` with the vision encoder's modules added to PEFT's ``exclude_modules``.

    Adapters train the language model only; the vision tower stays frozen whatever the targets. Text-only models
    get ``peft_config`` back unchanged. PEFT matches a string ``exclude_modules`` as a full-name regex and a list by
    name suffix, so a caller's exclusion is folded into one regex alongside the vision prefix. PEFT does not apply
    ``exclude_modules`` to ``target_parameters``, so a suffix that also matches vision parameters is replaced by the
    full names of its language-model matches.
    """
    vision = get_vision_encoder(model)
    if vision is None:
        return peft_config
    prefix = next((name for name, module in model.named_modules() if module is vision), None)
    if not prefix:
        return peft_config
    patterns = [rf"{re.escape(prefix)}(?:\..*)?"]
    excluded = peft_config.get("exclude_modules")
    if isinstance(excluded, str):
        patterns.append(excluded)
    elif excluded:
        # ``LoraConfig.to_dict()`` gives a set; sort it so every rank builds (and saves) the same regex.
        names = sorted(excluded) if isinstance(excluded, (set, frozenset)) else excluded
        patterns.extend(rf"(?:.*\.)?{re.escape(name)}" for name in names)
    peft_config = {**peft_config, "exclude_modules": "|".join(f"(?:{pattern})" for pattern in patterns)}
    targets = peft_config.get("target_parameters")
    if targets:
        # PEFT's own match for target_parameters: the full name, or a suffix after a dot.
        matched = [
            name
            for name, _ in model.named_parameters()
            if any(name == target or name.endswith(f".{target}") for target in targets)
        ]
        kept = [name for name in matched if not name.startswith(f"{prefix}.")]
        if len(kept) < len(matched):
            if not kept:
                raise ValueError(f"peft.target_parameters={list(targets)} only match the frozen vision tower {prefix}")
            peft_config["target_parameters"] = kept
    return peft_config


def apply_peft(
    model: nn.Module,
    peft_config: dict[str, Any] | None,
    *,
    optimization_dtype: torch.dtype | None = None,
) -> nn.Module:
    """Wrap a model with PEFT and enable backward through frozen input embeddings.

    ``peft_type`` accepts PEFT's canonical registry names (e.g. ``LORA``) and
    legacy config-class names (e.g. ``Lora``). FP8 bases disable PEFT's adapter autocast and cast all trainable
    tensors to ``optimization_dtype`` (default bf16) before optimizer flat buffers are built.
    A model with a vision encoder gets no adapters on it, whatever the targets (see ``exclude_vision_encoder``).
    Expert integration is installed by the registered patch after PEFT wraps the
    fully transformed model.
    """
    validate_peft_config(peft_config)
    if peft_config is None:
        return model

    import peft

    peft_config = exclude_vision_encoder(model, peft_config)
    config = _resolve_peft_config_class(peft, peft_config)(**peft_config)
    if model_has_fp8_weights(model):
        model = peft.get_peft_model(model, config, autocast_adapter_dtype=False)
        cast_trainable_params_off_fp8(model, optimization_dtype)
    else:
        model = peft.get_peft_model(model, config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    from arctic_platform.model.patches.tiled_mlp import register_tiled_mlp_peft_parameter_wrappers

    register_tiled_mlp_peft_parameter_wrappers(model)
    return model


@register_patch("peft")
def apply_peft_patch(model: nn.Module, ctx: LoaderContext) -> nn.Module:
    dtype = torch.bfloat16 if ctx.spec.dtype == "auto" else getattr(torch, ctx.spec.dtype)
    fp8_base = model_has_fp8_weights(model)
    wrapped = apply_peft(model, ctx.spec.patches.peft, optimization_dtype=dtype)
    peft_config = ctx.spec.patches.peft or {}
    if fp8_base and peft_config.get("target_parameters"):
        install_unfused_expert_lora_activation()
        attached = attach_unfused_expert_lora_factors(wrapped)
        if attached == 0:
            raise RuntimeError("peft.target_parameters is set but no expert LoRA factors were attached")

    ep_size = ctx.spec.parallelism.expert_parallel
    if ep_size > 1 and peft_config.get("target_parameters"):
        from arctic_platform.model.implementations.moe.deepspeed_integration import (
            tag_expert_lora_adapters_for_deepspeed,
        )

        tag_expert_lora_adapters_for_deepspeed(wrapped, f"ep_size_{ep_size}")
    return wrapped
