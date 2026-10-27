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

"""Initialize registered native LoRA factors without changing reference suppliers."""

from __future__ import annotations

from pathlib import Path

from .shared_lora_adapter import _read_adapter
from .shared_lora_adapter import _tensor_identity
from .shared_lora_adapter import _topology
from .shared_lora_adapter import adapter_identity


def _adapter_parameters(model) -> dict:
    import torch
    from peft.tuners.lora.layer import LoraLayer

    from .lora import canonical_name

    parameters = {}
    seen = set()
    storage_seen = set()
    base_parameters = set()
    base_storage = set()
    factors = {"lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_magnitude_vector"}
    for name, buffer in model.named_buffers(remove_duplicate=False):
        if factors.intersection(name.split(".")):
            raise ValueError("shared adapter requires registered default dense A/B parameters")
        if buffer.device.type != "meta" and buffer.layout == torch.strided:
            base_storage.add((buffer.device, buffer.untyped_storage().data_ptr()))
    for name, parameter in model.named_parameters(remove_duplicate=False):
        parts = name.split(".")
        if not factors.intersection(parts):
            base_parameters.add(id(parameter))
            if parameter.device.type != "meta" and parameter.layout == torch.strided:
                base_storage.add((parameter.device, parameter.untyped_storage().data_ptr()))
            continue
        if len(parts) < 4 or parts[-3] not in {"lora_A", "lora_B"} or parts[-2:] != ["default", "weight"]:
            raise ValueError("shared adapter requires registered default dense A/B parameters")
        layer = model.get_submodule(".".join(parts[:-3]))
        if (
            not isinstance(layer, LoraLayer)
            or set(layer.lora_A) != {"default"}
            or set(layer.lora_B) != {"default"}
            or getattr(layer, parts[-3])["default"].weight is not parameter
        ):
            raise ValueError("shared adapter requires registered default dense A/B parameters")
        canonical = canonical_name(name)
        if canonical in parameters or id(parameter) in seen:
            raise ValueError("shared adapter contains duplicate canonical tensors or aliased parameters")
        if parameter.device.type == "meta" or parameter.layout != torch.strided or not parameter.is_contiguous():
            raise ValueError(f"shared adapter requires materialized contiguous target storage: {name}")
        storage = (parameter.device, parameter.untyped_storage().data_ptr())
        if storage in storage_seen:
            raise ValueError("shared adapter contains aliased parameter storage")
        parameters[canonical] = parameter
        seen.add(id(parameter))
        storage_seen.add(storage)
    if seen.intersection(base_parameters) or storage_seen.intersection(base_storage):
        raise ValueError("shared adapter parameters alias base-model storage")
    return parameters


def initialize_adapter(model, path: str | Path, expected_identity: dict) -> None:
    """Transactionally copy exact native factors before optimizer construction."""
    import torch

    from .lora import canonical_name

    state, identity = _read_adapter(path)
    if identity != expected_identity:
        raise ValueError("shared adapter changed after reference admission")
    configs = getattr(model, "peft_config", {})
    if set(configs) != {"default"} or _topology(configs["default"].to_dict()) != identity["peft_config"]:
        raise ValueError("target PEFT topology differs from the shared adapter")
    # Checkpoint wrappers strip state-dict paths but retain registered parameter paths.
    current = _adapter_parameters(model)
    before = _tensor_identity(current)
    expected = identity["tensors"]
    if [{key: entry[key] for key in ("name", "shape", "dtype")} for entry in before] != [
        {key: entry[key] for key in ("name", "shape", "dtype")} for entry in expected
    ]:
        raise ValueError("shared adapter target tensor coverage, shapes or dtypes differ")
    if adapter_identity(path) != expected_identity:
        raise ValueError("shared adapter changed before loading")
    admitted = {canonical_name(name): tensor for name, tensor in state.items()}
    originals = {name: parameter.detach().clone() for name, parameter in current.items()}
    try:
        with torch.no_grad():
            for name, parameter in current.items():
                parameter.copy_(admitted[name])
        installed = _adapter_parameters(model)
        if (
            set(installed) != set(current)
            or any(installed[name] is not parameter for name, parameter in current.items())
            or _tensor_identity(installed) != expected
        ):
            raise ValueError("shared adapter target tensors differ after loading")
        if adapter_identity(path) != expected_identity:
            raise ValueError("shared adapter changed during loading")
    except BaseException as primary:
        # Restore every factor even when a copy or postflight fails partway through.
        with torch.no_grad():
            for name, parameter in current.items():
                try:
                    parameter.copy_(originals[name])
                except BaseException as rollback:
                    primary.add_note(f"shared adapter factor rollback failed for {name}: {rollback!r}")
        try:
            restored = _adapter_parameters(model)
            if (
                set(restored) != set(current)
                or any(restored[name] is not parameter for name, parameter in current.items())
                or _tensor_identity(restored) != before
            ):
                raise ValueError("shared adapter factor rollback identity differs")
        except BaseException as rollback:
            primary.add_note(f"shared adapter factor rollback verification failed: {rollback!r}")
        raise
