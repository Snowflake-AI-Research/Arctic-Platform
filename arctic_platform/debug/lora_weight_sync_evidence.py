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

"""Collectively retain the native LoRA stream without relabeling base holdings."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from arctic_platform.debug.ep_weight_sync_evidence import global_inventory
from arctic_platform.debug.weight_sync_evidence import source_files
from arctic_platform.debug.weight_sync_evidence import tensor_manifest
from arctic_platform.debug.weight_sync_evidence import verify_sources

_EXPORT_SHA = "d25d1db0186d228c1b3ac7b7735d7d1368399589c9f13bc7f5383a6390368c10"


def _admit(admitted, *objects):
    selected = source_files(*objects)
    if (
        not isinstance(admitted, dict)
        or not admitted
        or any(admitted.get(path) != sha for path, sha in selected.items())
    ):
        raise ValueError("LoRA selected source origins are not caller-admitted")
    verify_sources(admitted)
    return dict(admitted)


def _verify_wire_tensor(name, tensor, records):
    """Check native dim-zero gathering against normalized actual local bytes."""
    first = records[0]["wire_inputs"][name]
    members = first["ep_ranks"]
    members = [0] if members is None else members
    shards = [records[rank]["wire_inputs"][name]["manifest"] for rank in members]
    shape = shards[0]["shape"]
    actual = tensor_manifest(tensor)
    if (
        not shape
        or actual["dtype"] != shards[0]["dtype"]
        or actual["shape"]
        != [
            sum(shard["shape"][0] for shard in shards),
            *shape[1:],
        ]
    ):
        raise ValueError(f"native LoRA output geometry differs from actual ordered shards: {name}")
    offset = 0
    for shard in shards:
        length = shard["shape"][0]
        if tensor_manifest(tensor.narrow(0, offset, length)) != shard:
            raise ValueError(f"native LoRA output bytes differ from an actual ordered shard: {name}")
        offset += length


def _validate_wire_records(records, inventory):
    by_rank = {record["rank"]: record for record in records}
    ordered = [by_rank[rank] for rank in range(len(records))]
    if any(
        record.get("adapter_names") != ordered[0].get("adapter_names")
        or record.get("adapter_rank") != ordered[0].get("adapter_rank")
        for record in ordered
    ):
        raise ValueError("native LoRA adapter selection differs across training ranks")
    inputs = ordered[0].get("wire_inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("LoRA preparation requires actual adapter wire inputs")
    adapter_names = set()
    for wire_name, entry in inputs.items():
        if not isinstance(wire_name, str) or not wire_name or not isinstance(entry, dict):
            raise ValueError("invalid adapter wire name or entry")
        name = entry.get("parameter")
        if name not in inventory["parameters"] or name in adapter_names:
            raise ValueError("native LoRA wire mapping duplicates or invents a parameter")
        adapter_names.add(name)
        for record in ordered:
            current = record.get("wire_inputs")
            if (
                not isinstance(current, dict)
                or list(current) != list(inputs)
                or record.get("wire_order") != list(current)
            ):
                raise ValueError("native LoRA wire order differs across training ranks")
            current_entry = current[wire_name]
            if not isinstance(current_entry, dict) or current_entry.get("parameter") != name:
                raise ValueError("native LoRA wire-to-parameter binding differs across ranks")
            parameter = record["parameters"][name]
            manifest = current_entry.get("manifest")
            if (
                not isinstance(manifest, dict)
                or manifest.get("numel") != parameter["numel"]
                or manifest.get("dtype") != parameter["dtype"]
                or current_entry.get("ep_ranks") != parameter["ep_ranks"]
                or not parameter["requires_grad"]
                or type(manifest.get("shape")) is not list
                or not manifest["shape"]
                or any(type(size) is not int or size <= 0 for size in manifest["shape"])
                or math.prod(manifest["shape"]) != manifest["numel"]
                or not isinstance(manifest.get("sha256"), str)
                or len(manifest["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in manifest["sha256"])
            ):
                raise ValueError("native LoRA normalization does not preserve actual adapter geometry")
            base = entry["manifest"]
            if any(manifest.get(key) != base.get(key) for key in ("shape", "dtype", "numel")):
                raise ValueError("normalized LoRA shard geometry differs across ranks")
            if parameter["ep_ranks"] is None and manifest.get("sha256") != base.get("sha256"):
                raise ValueError("replicated native LoRA bytes disagree across training ranks")
        if entry.get("ep_ranks") != ordered[0]["parameters"][name]["ep_ranks"]:
            raise ValueError("normalized LoRA wire group differs from its actual parameter")
    trainable = {name for name, entry in inventory["parameters"].items() if entry["requires_grad"]}
    if adapter_names != trainable:
        raise ValueError("native LoRA stream does not cover every actual trainable adapter")
    return ordered


@dataclass
class PreparedLoRAWeights:
    rank: int
    weights: list
    manifest: dict
    inventory: dict
    rank_records: list
    sources: dict
    metadata_sha256: str = field(init=False)

    def __post_init__(self):
        self.metadata_sha256 = self._digest()

    def _digest(self):
        return hashlib.sha256(
            json.dumps(
                [
                    self.rank,
                    [name for name, _ in self.weights],
                    self.manifest,
                    self.inventory,
                    self.rank_records,
                    self.sources,
                ],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def verify(self):
        if self.metadata_sha256 != self._digest():
            raise RuntimeError("prepared native LoRA metadata changed after export")
        if (
            len(self.weights) != len(self.manifest)
            or {name: tensor_manifest(value) for name, value in self.weights} != self.manifest
        ):
            raise RuntimeError("prepared native LoRA tensor names or bytes changed")
        _validate_wire_records(self.rank_records, self.inventory)
        verify_sources(self.sources)

    def receipt(self):
        self.verify()
        adapter_names = {entry["parameter"] for entry in self.rank_records[0]["wire_inputs"].values()}
        return {
            "rank": self.rank,
            "weight_format": "lora",
            "emitted_manifest": self.manifest,
            "global_inventory": self.inventory,
            "rank_records": self.rank_records,
            "adapter_parameter_names": sorted(adapter_names),
            "non_wire_parameter_names": sorted(set(self.inventory["parameters"]) - adapter_names),
            "non_wire_buffers": self.inventory["buffers"],
            "metadata_sha256": self.metadata_sha256,
            "source_files": self.sources,
            "unchanged_native_iterator": True,
            "native_shard_bytes_checked": self.rank == 0,
            "native_iterator_completed": True,
            "receiver_installation_admitted": False,
            "serving_request_admitted": False,
            "router_application_admitted": False,
            "runtime_admitted": False,
            "capacity_admitted": False,
            "parameter_only_judge_admitted": False,
            "recipe_admitted": False,
        }


def prepare_collective_lora_weights(
    model,
    *,
    weight_format: str,
    is_zero3: bool,
    max_weight_bytes: int,
    admitted_sources: dict,
    expected_world_size: int = 8,
    expected_ep_size: int = 8,
) -> PreparedLoRAWeights:
    """All eight training ranks prepare once, before any scheduled sender.

    The unchanged native iterator owns adapter naming, expert reshaping and
    collective calls. This helper checks the emitted bytes, not a replacement
    packing algorithm. Frozen base parameters and buffers stay non-wire.
    """
    if weight_format != "lora" or is_zero3 is not False:
        raise ValueError("native LoRA preparation requires explicit lora and non-ZeRO3")
    if type(max_weight_bytes) is not int or max_weight_bytes <= 0:
        raise ValueError("native LoRA preparation requires a positive emitted-byte limit")
    if any(type(size) is not int or size != 8 for size in (expected_world_size, expected_ep_size)):
        raise ValueError("this native LoRA preparation is restricted to training8/EP8")
    sources = _admit(admitted_sources, prepare_collective_lora_weights, global_inventory)

    import deepspeed.utils.groups as ds_groups
    import torch.distributed as dist
    from peft.tuners.lora.layer import ParamWrapper

    from arctic_platform.model import weight_export as native
    from arctic_platform.model.patches.peft import is_peft_lora_param

    exporter = native.iter_lora_weights
    if _EXPORT_SHA not in source_files(exporter).values():
        raise ValueError("native LoRA exporter source differs from the selected recipe")
    selected = (
        type(model),
        model.named_parameters,
        model.named_buffers,
        model.named_modules,
        model.modules,
        exporter,
        native._iter_param_wrapper_expert_loras,
        native._reshape_expert_lora_for_ep,
        native._expert_lora_vllm_key,
        native._peft_lora_parameter_to_vllm_name,
        native._peft_adapter_rank,
        native._is_ep_sharded_parameter,
        native._expert_parallel_group,
        native.validate_lora_sync_trainable_parameters,
        native.is_peft_model,
        is_peft_lora_param,
        ParamWrapper,
        dist.is_initialized,
        dist.get_rank,
        dist.get_world_size,
        dist.all_gather,
        dist.all_gather_object,
        dist.get_process_group_ranks,
        ds_groups._get_expert_parallel_group,
    )
    sources = _admit(sources, *selected)
    if not dist.is_initialized() or dist.get_world_size() != expected_world_size:
        raise ValueError("actual LoRA training process group does not match training8")
    rank = dist.get_rank()
    if type(rank) is not int or not 0 <= rank < expected_world_size:
        raise ValueError("actual LoRA training rank is outside training8")
    if not native.is_peft_model(model) or getattr(model, "_is_vlm", False):
        raise ValueError("native LoRA preparation requires the selected non-VLM PEFT model")
    configs = model.peft_config
    if not isinstance(configs, dict) or len(configs) != 1:
        raise ValueError("native LoRA preparation requires one unambiguous adapter")
    config_objects = dict(configs)
    lora_rank = native._peft_adapter_rank(model)

    def local_state():
        native.validate_lora_sync_trainable_parameters(model)
        objects = {"parameters": dict(model.named_parameters()), "buffers": dict(model.named_buffers())}
        parameter_names = {id(value): name for name, value in objects["parameters"].items()}
        if len(parameter_names) != len(objects["parameters"]):
            raise ValueError("LoRA parameter aliases require a separately supported native binding")
        inventories = {}
        for kind, values in objects.items():
            inventories[kind] = {}
            for name, value in values.items():
                if hasattr(value, "ds_id"):
                    raise ValueError("native LoRA preparation does not admit ZeRO3 views")
                members = None
                if getattr(value, "allreduce", True) is False:
                    if not native._is_ep_sharded_parameter(value):
                        raise ValueError("expert tensor has no native EP group binding")
                    members = list(dist.get_process_group_ranks(native._expert_parallel_group(value)))
                    if len(members) != expected_ep_size or rank not in members:
                        raise ValueError("actual LoRA expert group does not match EP8")
                inventories[kind][name] = {
                    **tensor_manifest(value),
                    "requires_grad": value.requires_grad,
                    "ep_ranks": members,
                }
        inputs = {}
        expert_ids = set()

        def add(name, parameter, local):
            parameter_name = parameter_names.get(id(parameter))
            if (
                parameter_name is None
                or not is_peft_lora_param(parameter_name, parameter)
                or not isinstance(name, str)
                or not name
                or name in inputs
            ):
                raise ValueError("native LoRA export selected an ambiguous or non-adapter parameter")
            inputs[name] = {
                "parameter": parameter_name,
                "manifest": tensor_manifest(local),
                "ep_ranks": inventories["parameters"][parameter_name]["ep_ranks"],
            }

        for module in model.modules():
            if isinstance(module, ParamWrapper) and (
                len(module.lora_A) != 1
                or len(module.lora_B) != 1
                or list(module.lora_A) != list(configs)
                or list(module.lora_B) != list(configs)
            ):
                raise ValueError("native expert adapter bucket selection is ambiguous")
        for prefix, weight_name, kind, parameter in native._iter_param_wrapper_expert_loras(model):
            if id(parameter) in expert_ids:
                raise ValueError("native expert LoRA iterator selected an adapter twice")
            expert_ids.add(id(parameter))
            prefix = native.strip_activation_checkpoint_segments(prefix).replace(".base_layer", "")
            name = native._expert_lora_vllm_key(prefix, weight_name, kind)
            normalized = native._reshape_expert_lora_for_ep(parameter.data.contiguous(), kind, lora_rank, label=name)
            add(name, parameter, normalized)
        for name, parameter in objects["parameters"].items():
            if is_peft_lora_param(name, parameter) and id(parameter) not in expert_ids:
                add(native._peft_lora_parameter_to_vllm_name(name), parameter, parameter.data)
        return objects, inventories, inputs

    before_objects, before, inputs = local_state()
    record = {
        "rank": rank,
        "world_size": expected_world_size,
        "adapter_names": list(configs),
        "adapter_rank": lora_rank,
        **before,
        "wire_inputs": inputs,
        "wire_order": list(inputs),
    }
    records: list[Any] = [None] * expected_world_size
    dist.all_gather_object(records, record)
    inventory = global_inventory(records)
    records = _validate_wire_records(records, inventory)
    if records[rank] != record:
        raise ValueError("actual local LoRA record was not retained by the collective")
    adapter_names = {entry["parameter"] for entry in inputs.values()}
    wire_bytes = sum(
        inventory["parameters"][name]["global_numel"] * before_objects["parameters"][name].element_size()
        for name in adapter_names
    )
    if wire_bytes > max_weight_bytes:
        raise ValueError("actual native adapter holdings exceed the emitted-byte limit")
    verify_sources(sources)
    weights, manifest = [], {}
    emitted_bytes = 0
    for name, tensor in exporter(model, is_master=rank == 0, is_zero3=is_zero3):
        if rank != 0:
            raise RuntimeError("native LoRA iterator emitted on a non-sender training rank")
        if name not in inputs or name in manifest:
            raise ValueError("native LoRA iterator emitted an unknown or duplicate wire name")
        _verify_wire_tensor(name, tensor, records)
        emitted_bytes += tensor.numel() * tensor.element_size()
        if emitted_bytes > max_weight_bytes:
            raise ValueError("actual native LoRA stream exceeds the emitted-byte limit")
        value = tensor.detach().clone()
        weights.append((name, value))
        manifest[name] = tensor_manifest(value)
    if rank == 0 and (list(manifest) != list(inputs) or emitted_bytes != wire_bytes):
        raise ValueError("native LoRA iterator did not cover the complete ordered adapter byte domain")
    after_objects, after, after_inputs = local_state()
    if (
        before != after
        or inputs != after_inputs
        or native.iter_lora_weights is not exporter
        or model.peft_config is not configs
        or list(configs) != list(config_objects)
        or any(configs[name] is not config for name, config in config_objects.items())
        or native._peft_adapter_rank(model) != lora_rank
        or any(
            set(after_objects[kind]) != set(values)
            or any(after_objects[kind][name] is not value for name, value in values.items())
            for kind, values in before_objects.items()
        )
    ):
        raise RuntimeError("native LoRA parameter, buffer, adapter selection or bytes changed during export")
    _admit(sources, *selected)
    return PreparedLoRAWeights(rank, weights, manifest, inventory, records, sources)


def send_prepared_lora_weights(sender, prepared: PreparedLoRAWeights):
    """Use the actual sender once; adapter holdings never stand in for base holdings."""
    if not isinstance(prepared, PreparedLoRAWeights) or prepared.rank != 0 or not prepared.weights:
        raise ValueError("only the actual rank-zero retained LoRA stream can be sent")
    prepared.verify()
    sources = _admit(prepared.sources, sender.send, send_prepared_lora_weights)
    send = sender.send
    result = send(prepared.weights)
    if not isinstance(result, dict):
        raise TypeError("actual LoRA sender did not return a dictionary")
    if sender.send != send:
        raise RuntimeError("actual LoRA sender callable changed during dispatch")
    prepared.verify()
    verify_sources(sources)
    parameters = prepared.inventory["parameters"]
    adapter_names = {entry["parameter"] for entry in prepared.rank_records[0]["wire_inputs"].values()}

    def holdings(names):
        return {"numel": sum(parameters[name]["global_numel"] for name in names), "count": len(names)}

    return {
        **result,
        "weight_format": "lora",
        "sent_manifest": prepared.manifest,
        "trainer_parameters": holdings(parameters),
        "trainer_adapter_parameters": holdings(adapter_names),
        "trainer_non_wire_parameters": holdings(set(parameters) - adapter_names),
        "trainer_buffers": {
            "numel": sum(entry["global_numel"] for entry in prepared.inventory["buffers"].values()),
            "count": len(prepared.inventory["buffers"]),
            "wire_numel": 0,
        },
        "collective_preparation": prepared.receipt(),
    }
