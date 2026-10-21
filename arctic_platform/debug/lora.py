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

"""Debug-only deterministic LoRA initialization and its opt-in checks."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from torch import nn

    from arctic_platform.model.config import ParallelismConfig

DETERMINISTIC_LORA_RECIPE = "arctic-correctness-lora-init-v1"
_SEED_DOMAIN = "arctic-correctness-lora-tensor-seed-v1"


def normalize_lora_initialization(value: Any, *, location: str = "peft_config.deterministic_init") -> dict[str, Any]:
    """Validate the name-seeded initializer without importing training dependencies."""
    if not isinstance(value, dict) or set(value) != {"recipe_version", "global_seed", "dtype"}:
        raise ValueError(f"{location} requires exactly recipe_version, global_seed, and dtype")
    if value["recipe_version"] != DETERMINISTIC_LORA_RECIPE:
        raise ValueError(f"{location}.recipe_version is unsupported")
    global_seed = value["global_seed"]
    if type(global_seed) is not int or not 0 <= global_seed < 2**63:
        raise ValueError(f"{location}.global_seed must be an integer in [0, 2**63)")
    if not isinstance(value["dtype"], str) or value["dtype"] not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"{location}.dtype must be float32, float16, or bfloat16")
    return dict(value)


def maybe_normalize_lora_initialization(
    peft_config: dict[str, Any], lora_config: dict[str, Any], *, location: str
) -> dict[str, Any]:
    """Add validated debug options to the base LoRA config only when requested."""
    if "deterministic_init" not in peft_config:
        return lora_config
    if lora_config["target_parameters"]:
        raise ValueError(f"{location}.deterministic_init supports dense target_modules only")
    lora_config["deterministic_init"] = normalize_lora_initialization(
        peft_config["deterministic_init"], location=f"{location}.deterministic_init"
    )
    return lora_config


def maybe_prepare_lora_initialization(
    model: nn.Module, peft_config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Validate the opt-in and strip debug fields before constructing PEFT's config."""
    if "deterministic_init" not in peft_config:
        return peft_config, None

    from arctic_platform.common.peft import normalize_lora_peft_config

    validation_config = dict(peft_config)
    # The public DSS spelling is Lora; direct AP callers may use PEFT's LORA.
    if validation_config.get("peft_type") == "LORA":
        validation_config["peft_type"] = "Lora"
    initialization_config = normalize_lora_peft_config(validation_config)["deterministic_init"]
    if getattr(getattr(model, "config", None), "model_type", None) != "qwen3":
        raise ValueError("deterministic LoRA initialization currently supports dense Qwen3 only")
    adapter_config = dict(peft_config)
    del adapter_config["deterministic_init"]
    return adapter_config, initialization_config


def maybe_validate_lora_parallelism(peft_config: dict[str, Any] | None, parallelism: ParallelismConfig) -> None:
    """Reject unsupported debug initialization layouts before PEFT wrapping."""
    if peft_config is None or peft_config.get("deterministic_init") is None:
        return
    if parallelism.expert_parallel != 1 or parallelism.sequence_parallel != 1:
        raise ValueError("deterministic LoRA initialization currently requires dense EP1/SP1")


def maybe_initialize_lora_factors(model: nn.Module, initialization_config: dict[str, Any] | None) -> int:
    """Initialize existing LoRA factors only when the debug option was supplied."""
    if initialization_config is None:
        return 0
    return initialize_lora_factors(model, initialization_config)


def _canonical_name(name: str) -> str:
    parts = [part for part in name.split(".") if part != "_checkpoint_wrapped_module"]
    if len(parts) >= 4 and parts[-3] in {"lora_A", "lora_B"} and parts[-2] == "default":
        del parts[-2]
    if (
        len(parts) < 3
        or any(not part or any(character.isspace() for character in part) for part in parts)
        or parts[-2] not in {"lora_A", "lora_B"}
        or parts[-1] != "weight"
        or any(part.startswith("lora_") for part in parts[:-2])
    ):
        raise ValueError(f"deterministic LoRA initialization requires default A/B weights: {name!r}")
    return ".".join(parts)


def _tensor_seed(global_seed: int, name: str) -> int:
    payload = json.dumps(
        {"domain": _SEED_DOMAIN, "global_seed": global_seed, "name": name},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def initialize_lora_factors(model: nn.Module, options: dict[str, Any]) -> int:
    """Fill existing parameters before optimizer creation, without global RNG draws."""
    initialization_config = normalize_lora_initialization(options)
    if sys.byteorder != "little":
        raise ValueError("deterministic LoRA recipe requires a little-endian host")

    import torch

    factor_dtype = getattr(torch, initialization_config["dtype"])
    lora_parameters: dict[str, nn.Parameter] = {}
    parameter_ids: set[int] = set()
    for raw_name, parameter in model.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad:
            continue
        name = _canonical_name(raw_name)
        if name in lora_parameters or id(parameter) in parameter_ids:
            raise ValueError("deterministic LoRA initialization does not support duplicate or tied factors")
        if (
            parameter.device.type == "meta"
            or parameter.layout != torch.strided
            or parameter.ndim != 2
            or any(size <= 0 for size in parameter.shape)
            or parameter.dtype != factor_dtype
            or hasattr(parameter, "ds_id")
            or getattr(parameter, "allreduce", True) is False
        ):
            raise ValueError(f"{name}: deterministic LoRA requires materialized, unsharded {factor_dtype} factors")
        lora_parameters[name] = parameter
        parameter_ids.add(id(parameter))
    if not lora_parameters:
        raise ValueError("deterministic LoRA initialization requires at least one complete A/B pair")
    for name, parameter in lora_parameters.items():
        prefix, factor, _ = name.rsplit(".", 2)
        partner_factor = "lora_B" if factor == "lora_A" else "lora_A"
        partner = lora_parameters.get(f"{prefix}.{partner_factor}.weight")
        if partner is None:
            raise ValueError(f"{name}: deterministic LoRA factor is missing its A/B partner")
        lora_a, lora_b = (parameter, partner) if factor == "lora_A" else (partner, parameter)
        if lora_a.shape[0] != lora_b.shape[1] or lora_a.dtype != lora_b.dtype:
            raise ValueError(f"{prefix}: deterministic LoRA A/B rank or dtype differs")

    # Generate each complete factor on CPU before copying to its existing storage.
    with torch.no_grad():
        for name, parameter in sorted(lora_parameters.items()):
            generated_factor = torch.empty(tuple(parameter.shape), dtype=torch.float32, device="cpu")
            if name.endswith(".lora_A.weight"):
                generator = torch.Generator(device="cpu").manual_seed(
                    _tensor_seed(initialization_config["global_seed"], name)
                )
                torch.nn.init.kaiming_uniform_(
                    generated_factor, a=math.sqrt(5.0), mode="fan_in", nonlinearity="leaky_relu", generator=generator
                )
            else:
                generated_factor.zero_()
            parameter.copy_(generated_factor.to(dtype=factor_dtype))
    return len(lora_parameters)
