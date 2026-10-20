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

# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in initialization of materialized, unsharded default LoRA factors."""

from __future__ import annotations

import hashlib
import json
import math
import sys

import torch
from torch import nn

from arctic_platform.common.peft import normalize_lora_initialization

_SEED_DOMAIN = "arctic-correctness-lora-tensor-seed-v1"


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


def _tensor_seed(seed: int, name: str) -> int:
    payload = json.dumps(
        {"domain": _SEED_DOMAIN, "global_seed": seed, "name": name},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def initialize_lora_factors(model: nn.Module, options: dict) -> int:
    """Fill existing parameters before optimizer creation, without global RNG draws."""
    options = normalize_lora_initialization(options)
    if sys.byteorder != "little":
        raise ValueError("deterministic LoRA recipe requires a little-endian host")
    dtype = getattr(torch, options["dtype"])
    parameters: dict[str, nn.Parameter] = {}
    for raw_name, parameter in model.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad:
            continue
        name = _canonical_name(raw_name)
        if name in parameters or any(parameter is previous for previous in parameters.values()):
            raise ValueError("deterministic LoRA initialization does not support duplicate or tied factors")
        if (
            parameter.device.type == "meta"
            or parameter.layout != torch.strided
            or parameter.ndim != 2
            or any(size <= 0 for size in parameter.shape)
            or parameter.dtype != dtype
            or hasattr(parameter, "ds_id")
            or getattr(parameter, "allreduce", True) is False
        ):
            raise ValueError(f"{name}: deterministic LoRA requires materialized, unsharded {dtype} factors")
        parameters[name] = parameter
    if not parameters:
        raise ValueError("deterministic LoRA initialization requires at least one complete A/B pair")
    for name, parameter in parameters.items():
        prefix, factor, _ = name.rsplit(".", 2)
        partner = parameters.get(f"{prefix}.{'lora_B' if factor == 'lora_A' else 'lora_A'}.weight")
        if partner is None:
            raise ValueError(f"{name}: deterministic LoRA factor is missing its A/B partner")
        a, b = (parameter, partner) if factor == "lora_A" else (partner, parameter)
        if a.shape[0] != b.shape[1] or a.dtype != b.dtype:
            raise ValueError(f"{prefix}: deterministic LoRA A/B rank or dtype differs")

    # Generate each complete factor on CPU before copying to its existing storage.
    with torch.no_grad():
        for name, parameter in sorted(parameters.items()):
            tensor = torch.empty(tuple(parameter.shape), dtype=torch.float32, device="cpu")
            if name.endswith(".lora_A.weight"):
                generator = torch.Generator(device="cpu").manual_seed(_tensor_seed(options["global_seed"], name))
                torch.nn.init.kaiming_uniform_(
                    tensor, a=math.sqrt(5.0), mode="fan_in", nonlinearity="leaky_relu", generator=generator
                )
            else:
                tensor.zero_()
            parameter.copy_(tensor.to(dtype=dtype))
    return len(parameters)
