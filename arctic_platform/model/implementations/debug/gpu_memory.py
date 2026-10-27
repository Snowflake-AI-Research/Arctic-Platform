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

"""Retained CUDA capacity guards for model construction and training."""

from __future__ import annotations

import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence

from arctic_platform.model.implementations.debug.config import training_debug_config

logger = logging.getLogger(__name__)


def _nonnegative_integer(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def tensor_storage_bytes(shape: Sequence[int], itemsize: int) -> int:
    """Bytes of a complete storage shape, not a view or a checkpoint file."""
    if _nonnegative_integer(itemsize, "itemsize") == 0:
        raise ValueError("itemsize must be positive")
    return math.prod(_nonnegative_integer(dim, "dimension") for dim in shape) * itemsize


def observe_native_model_inventory(
    model, *, retained_objects: list | None = None, checkpoint_destinations: dict | None = None
) -> dict:
    """Read registered metadata without materialization or admission.

    For before/after comparison, keep one ``retained_objects`` list alive through
    both calls and the intervening callback. Its references prevent identity
    reuse; the returned JSON contains only process-local identities and metadata.

    An optional flat ``checkpoint_destinations`` dictionary is observed using
    the same storage ledger. Detached destination objects can share model
    backing storage; this observation does not prove destination completeness.
    """
    import torch
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor import Partial
    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor import Shard

    if not isinstance(model, torch.nn.Module):
        raise ValueError("actual torch module required for native inventory")
    if retained_objects is not None and type(retained_objects) is not list:
        raise ValueError("ordinary retained_objects list required")
    if checkpoint_destinations is not None and type(checkpoint_destinations) is not dict:
        raise ValueError("ordinary checkpoint_destinations dictionary required")
    checkpoint_items = () if checkpoint_destinations is None else tuple(checkpoint_destinations.items())
    if any(type(name) is not str or not name for name, _ in checkpoint_items):
        raise ValueError("named checkpoint destination tensors required")

    retained = [] if retained_objects is None else retained_objects
    retained_ids = {id(value) for value in retained}
    storage_ids = {}
    storages: list[dict] = []
    registrations, modules, empty = [], [], []
    parameters: list[dict] = []
    buffers: list[dict] = []

    def class_identity(value):
        cls = type(value)
        return {"module": cls.__module__, "qualname": cls.__qualname__}

    def object_identity(value, kind):
        key = id(value)
        if key not in retained_ids:
            retained_ids.add(key)
            retained.append(value)
        return f"{kind}:{key:x}"

    def tensor_metadata(tensor):
        if type(tensor) not in (torch.Tensor, torch.nn.Parameter, DTensor):
            raise ValueError("unsupported native tensor subclass")
        if tensor.layout != torch.strided or tensor.is_quantized or tensor.is_nested:
            raise ValueError("unsupported native tensor layout")
        shape, stride = list(tensor.shape), list(tensor.stride())
        if any(type(value) is not int or value < 0 for value in (*shape, *stride)):
            raise ValueError("concrete native tensor shape and stride required")
        row = {
            "object_id": object_identity(tensor, "tensor"),
            "class": class_identity(tensor),
            "dtype": str(tensor.dtype),
            "shape": shape,
            "stride": stride,
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad,
            "element_size": tensor.element_size(),
            "logical_bytes": tensor_storage_bytes(shape, tensor.element_size()),
            "storage": None,
            "dtensor": None,
        }
        if type(tensor) is DTensor:
            mesh = tensor.device_mesh
            if (
                type(mesh) is not DeviceMesh
                or type(mesh.mesh) is not torch.Tensor
                or mesh.mesh.device.type != "cpu"
                or mesh.mesh.dtype not in (torch.int32, torch.int64)
                or mesh.mesh.layout != torch.strided
                or type(mesh.device_type) is not str
                or (
                    mesh.mesh_dim_names is not None
                    and (
                        type(mesh.mesh_dim_names) is not tuple
                        or any(type(name) is not str for name in mesh.mesh_dim_names)
                    )
                )
            ):
                raise ValueError("actual CPU device-mesh membership metadata required")
            placements = []
            for placement in tensor.placements:
                placement_id = object_identity(placement, "placement")
                if type(placement) is Shard:
                    if type(placement.dim) is not int or not -len(shape) <= placement.dim < len(shape):
                        raise ValueError("concrete native DTensor shard dimension required")
                    placements.append({"object_id": placement_id, "kind": "Shard", "dim": placement.dim})
                elif type(placement) is Replicate:
                    placements.append({"object_id": placement_id, "kind": "Replicate"})
                elif type(placement) is Partial and type(placement.reduce_op) is str:
                    placements.append({"object_id": placement_id, "kind": "Partial", "reduce_op": placement.reduce_op})
                else:
                    raise ValueError("unsupported native DTensor placement")
            if len(placements) != mesh.mesh.ndim:
                raise ValueError("native mesh and placements differ")
            local = tensor._local_tensor
            if type(local) not in (torch.Tensor, torch.nn.Parameter):
                raise ValueError("unsupported native DTensor local tensor")
            row["dtensor"] = {
                "spec_id": object_identity(tensor._spec, "spec"),
                "mesh": {
                    "object_id": object_identity(mesh, "mesh"),
                    "members_object_id": object_identity(mesh.mesh, "tensor"),
                    "device_type": mesh.device_type,
                    "shape": list(mesh.mesh.shape),
                    "members": mesh.mesh.tolist(),
                    "dim_names": None if mesh.mesh_dim_names is None else list(mesh.mesh_dim_names),
                },
                "placements": placements,
                "local": tensor_metadata(local),
            }
        elif not tensor.is_meta:
            storage = tensor.untyped_storage()
            key = (str(storage.device), storage._cdata)
            storage_id = f"storage:{storage._cdata:x}"
            object_identity(storage, "storage_wrapper")
            if key not in storage_ids:
                storage_ids[key] = len(storages)
                storages.append({"storage_id": storage_id, "device": str(storage.device), "nbytes": storage.nbytes()})
            elif storages[storage_ids[key]]["nbytes"] != storage.nbytes():
                raise ValueError("native backing storage changed during inventory")
            offset = tensor.storage_offset()
            row["storage"] = {
                "storage_id": storage_id,
                "offset_elements": offset,
                "offset_bytes": offset * tensor.element_size(),
                "nbytes": storage.nbytes(),
            }
        return row

    # Walk registrations directly so custom enumeration cannot drop aliases or buffers.
    pending: list[tuple[str, Any, frozenset[int]]] = [("", model, frozenset())]
    while pending:
        prefix, module, ancestors = pending.pop()
        if not isinstance(module, torch.nn.Module) or id(module) in ancestors:
            raise ValueError("unsupported or cyclic native module registration")
        state = object.__getattribute__(module, "__dict__")
        tables = tuple(state.get(name) for name in ("_parameters", "_buffers", "_modules"))
        nonpersistent = state.get("_non_persistent_buffers_set")
        if any(type(table) is not dict for table in tables) or type(nonpersistent) is not set:
            raise ValueError("ordinary native registration dictionaries required")
        if not nonpersistent <= tables[1].keys():
            raise ValueError("nonpersistent native buffer registration differs")
        owner_id = object_identity(module, "module")
        modules.append({"name": prefix[:-1], "object_id": owner_id, "class": class_identity(module)})
        for table in (*tables, nonpersistent):
            object_identity(table, "registration")
        captured = tuple(tuple(table.items()) for table in tables)
        registrations.append((module, tables, captured, nonpersistent, frozenset(nonpersistent)))
        for kind, items, destination in (
            ("parameter", captured[0], parameters),
            ("buffer", captured[1], buffers),
        ):
            for name, tensor in items:
                if type(name) is not str or not name or "." in name:
                    raise ValueError("ordinary native registration name required")
                row = {
                    "name": prefix + name,
                    "owner_id": owner_id,
                    "owner_class": class_identity(module),
                    "persistent": None if kind == "parameter" else name not in nonpersistent,
                }
                if tensor is None:
                    empty.append({"kind": kind, **row})
                else:
                    destination.append({**row, "tensor": tensor_metadata(tensor)})
        for name, child in reversed(captured[2]):
            if type(name) is not str or not name or "." in name:
                raise ValueError("ordinary native module registration name required")
            if child is not None:
                pending.append((prefix + name + ".", child, ancestors | {id(module)}))

    destinations = {}
    if checkpoint_destinations is not None:
        object_identity(checkpoint_destinations, "checkpoint_destinations")
        destinations = {name: tensor_metadata(tensor) for name, tensor in checkpoint_items}
        after = tuple(checkpoint_destinations.items())
        if len(after) != len(checkpoint_items) or any(
            name != prior_name or value is not prior_value
            for (name, value), (prior_name, prior_value) in zip(after, checkpoint_items, strict=True)
        ):
            raise ValueError("checkpoint destinations changed during inventory")

    for module, tables, captured, nonpersistent, names in registrations:
        state = object.__getattribute__(module, "__dict__")
        if (
            any(state.get(name) is not table for name, table in zip(("_parameters", "_buffers", "_modules"), tables))
            or state.get("_non_persistent_buffers_set") is not nonpersistent
            or nonpersistent != names
        ):
            raise ValueError("native registrations changed during inventory")
        for table, before in zip(tables, captured, strict=True):
            after = tuple(table.items())
            if len(after) != len(before) or any(
                name != prior_name or value is not prior_value
                for (name, value), (prior_name, prior_value) in zip(after, before, strict=True)
            ):
                raise ValueError("native registrations changed during inventory")
    observed = {
        "model_class": class_identity(model),
        "modules": modules,
        "parameters": parameters,
        "buffers": buffers,
        "storages": storages,
        "empty_registrations": empty,
        "alias_identity_scope": "process-local",
        "snapshot_only": True,
        "capacity_admitted": False,
        "runtime_admitted": False,
    }
    if checkpoint_destinations is not None:
        observed["checkpoint_destinations"] = destinations
    return observed


@dataclass(frozen=True)
class CheckpointTensor:
    """One original tensor header; dtype width excludes separate scale tensors."""

    file_name: str
    shape: tuple[int, ...]
    itemsize: int


@dataclass(frozen=True)
class DCPReadRequest:
    """Source slice from an actual selected load plan, before planner resolution."""

    tensor_name: str
    offsets: tuple[int, ...]
    lengths: tuple[int, ...]


def checkpoint_header_inventory(
    shards: Mapping[str, Mapping[str, Any]],
    *,
    weight_map: Mapping[str, str],
    dtype_widths: Mapping[str, int],
) -> dict[str, CheckpointTensor]:
    """Validate captured header shapes against the real index, without payload IO.

    ``dtype_widths`` must come from the selected dtype recipe. Header/index
    consistency proves neither payload bytes nor installed loader behavior.
    In particular, official BF16 headers cannot replace original FP8 headers.
    """
    if not shards or not weight_map or not dtype_widths:
        raise ValueError("captured headers, original index weight_map and dtype widths required")
    if set(shards) != set(weight_map.values()):
        raise ValueError("captured shard set differs from original index")
    result = {}
    for file_name, shard in shards.items():
        header_size = _nonnegative_integer(shard["header_size_bytes"], "header size")
        payload_size = _nonnegative_integer(shard["file_size_bytes"], "file size") - 8 - header_size
        if payload_size < 0 or not shard["tensors"]:
            raise ValueError("invalid captured shard header")
        intervals = []
        for name, tensor in shard["tensors"].items():
            if name in result or weight_map.get(name) != file_name:
                raise ValueError("tensor key duplicate or original index mapping differs")
            dtype = tensor["dtype"]
            if dtype not in dtype_widths:
                raise ValueError(f"unresolved dtype width: {dtype}")
            shape = tuple(tensor["shape"])
            size_bytes = tensor_storage_bytes(shape, dtype_widths[dtype])
            offsets = tensor["data_offsets"]
            if len(offsets) != 2:
                raise ValueError("tensor data offsets must have two bounds")
            start, end = (_nonnegative_integer(value, "data offset") for value in offsets)
            if end - start != size_bytes or end > payload_size:
                raise ValueError("tensor dtype/shape bytes differ from captured data span")
            intervals.append((start, end))
            result[name] = CheckpointTensor(file_name, shape, dtype_widths[dtype])
        position = 0
        for start, end in sorted(intervals):
            if start != position:
                raise ValueError("captured tensor spans overlap or leave a payload gap")
            position = end
        if position != payload_size:
            raise ValueError("captured tensors do not cover shard payload")
    if set(result) != set(weight_map):
        raise ValueError("captured tensor key set differs from original index")
    return result


def saved_header_ledger(
    shards: Mapping[str, Mapping[str, Any]],
    *,
    dtype_widths: Mapping[str, int],
    source_identity: Mapping[str, str],
) -> dict:
    """Validate saved header spans without claiming original-index consistency.

    The identity is a caller recipe, including model_id, revision and the saved
    receipt_sha256. No file or payload is read. ``derived_shard_map`` describes
    the saved headers only; it MUST NOT replace the genuine index weight_map
    required by checkpoint_header_inventory.
    """
    if not shards or not dtype_widths:
        raise ValueError("saved headers and explicit dtype widths required")
    if (
        not all(source_identity.get(key) for key in ("model_id", "revision", "receipt_sha256"))
        or any(not isinstance(value, str) or not value for value in source_identity.values())
        or re.fullmatch(r"[0-9a-f]{64}", source_identity["receipt_sha256"]) is None
    ):
        raise ValueError("model, revision and saved receipt SHA256 recipe required")
    headers = {}
    dtypes = {}
    per_dtype: dict[str, dict[str, int]] = {}
    file_bytes = 0
    header_bytes = 0
    for file_name, shard in shards.items():
        if not isinstance(file_name, str) or not file_name:
            raise ValueError("named saved shard required")
        header_size = _nonnegative_integer(shard["header_size_bytes"], "header size")
        file_size = _nonnegative_integer(shard["file_size_bytes"], "file size")
        payload_size = file_size - 8 - header_size
        if payload_size < 0 or not shard["tensors"]:
            raise ValueError("invalid saved shard header")
        intervals = []
        for name, tensor in shard["tensors"].items():
            if not isinstance(name, str) or not name or name in headers:
                raise ValueError("unique named saved tensor required")
            dtype = tensor["dtype"]
            if dtype not in dtype_widths:
                raise ValueError(f"unresolved saved dtype width: {dtype}")
            shape = tuple(tensor["shape"])
            size_bytes = tensor_storage_bytes(shape, dtype_widths[dtype])
            offsets = tensor["data_offsets"]
            if len(offsets) != 2:
                raise ValueError("saved data offsets must have two bounds")
            start, end = (_nonnegative_integer(value, "data offset") for value in offsets)
            if end - start != size_bytes or end > payload_size:
                raise ValueError("saved dtype/shape bytes differ from data span")
            intervals.append((start, end))
            headers[name] = CheckpointTensor(file_name, shape, dtype_widths[dtype])
            dtypes[name] = dtype
            totals = per_dtype.setdefault(dtype, {"tensor_count": 0, "payload_bytes": 0})
            totals["tensor_count"] += 1
            totals["payload_bytes"] += size_bytes
        position = 0
        for start, end in sorted(intervals):
            if start != position:
                raise ValueError("saved tensor spans overlap or leave a payload gap")
            position = end
        if position != payload_size:
            raise ValueError("saved tensors do not cover shard payload")
        file_bytes += file_size
        header_bytes += 8 + header_size
    return {
        "schema": "saved-header-ledger-v1",
        "source_identity": dict(source_identity),
        "headers": headers,
        "dtypes": dtypes,
        "derived_shard_map": {name: header.file_name for name, header in headers.items()},
        "tensor_count": len(headers),
        "shard_count": len(shards),
        "tensor_payload_bytes": sum(row["payload_bytes"] for row in per_dtype.values()),
        "dtype_totals": per_dtype,
        "file_bytes": file_bytes,
        "header_and_prefix_bytes": header_bytes,
        "source_identity_bound": False,
        "original_index_bound": False,
        "payload_identity_bound": False,
        "capacity_admitted": False,
        "runtime_admitted": False,
        "unresolved_inputs": ["genuine_original_index_weight_map", "original_payload_identity"],
    }


def glm_bf16_conversion_projection(ledger: Mapping[str, Any]) -> dict:
    """Project selected GLM rename/stack/cat metadata without tensor allocation.

    Only the BF16/F32, unscaled branch of glm53.convert_hf_layer_to_prime is
    represented. Reject collisions rather than silently overwriting a source
    tensor. Output layouts have no invented converted-cache file association.
    This source recipe is neither an imported-converter binding nor a native
    destination inventory; source and converted namespaces remain distinct.
    """
    if ledger.get("schema") != "saved-header-ledger-v1" or not ledger.get("headers"):
        raise ValueError("saved header ledger required")
    headers = ledger["headers"]
    dtypes = dict(ledger["dtypes"])
    if set(headers) != set(dtypes):
        raise ValueError("saved header/dtype key sets differ")
    layouts = {}
    for name, header in headers.items():
        if dtypes[name] not in ("BF16", "F32") or header.itemsize != {"BF16": 2, "F32": 4}[dtypes[name]]:
            raise ValueError("GLM projection supports only explicit BF16/F32 widths")
        if name.endswith(("_scale_inv", ".weight_scale_inv")):
            raise ValueError("scaled GLM checkpoint is outside the BF16 projection")
        tensor_storage_bytes(header.shape, header.itemsize)
        layouts[name] = (header.shape, header.itemsize)
    source_bytes = _layout_bytes(layouts)
    lineage = {name: (name,) for name in layouts}
    packed = {}
    sharded_names = []
    expert_counts = {}

    def rename(old, new):
        if old not in layouts:
            return
        if new in layouts:
            raise ValueError(f"GLM converted key collision: {new}")
        layouts[new] = layouts.pop(old)
        dtypes[new] = dtypes.pop(old)
        lineage[new] = lineage.pop(old)

    def pack(keys, target, *, stack):
        if target in layouts:
            raise ValueError(f"GLM converted key collision: {target}")
        if any(key not in layouts for key in keys):
            raise ValueError(f"missing GLM projection input: {target}")
        shapes = [layouts[key][0] for key in keys]
        dtype = dtypes[keys[0]]
        if dtype != "BF16" or any(dtypes[key] != dtype for key in keys):
            raise ValueError("GLM packed weights require identical BF16 dtypes")
        if any(not shape or any(dim == 0 for dim in shape) for shape in shapes):
            raise ValueError("GLM packing requires nonempty positive tensor shapes")
        if stack:
            if any(shape != shapes[0] for shape in shapes):
                raise ValueError("GLM expert projection shapes differ")
            shape = (len(keys), *shapes[0])
        else:
            if any(shape[1:] != shapes[0][1:] for shape in shapes):
                raise ValueError("GLM convolution trailing shapes differ")
            shape = (sum(shape[0] for shape in shapes), *shapes[0][1:])
        sources = tuple(source for key in keys for source in lineage[key])
        for key in keys:
            del layouts[key], dtypes[key], lineage[key]
        layouts[target] = (shape, 2)
        dtypes[target] = dtype
        lineage[target] = sources
        packed[target] = layouts[target]

    layer_pattern = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
    layers = sorted({int(match[1]) for name in layouts if (match := layer_pattern.match(name))})
    for layer in layers:
        prefix = f"model.language_model.layers.{layer}"
        for source, target in (
            ("hc_attn_fn", "attn_hc.fn"),
            ("hc_attn_base", "attn_hc.base"),
            ("hc_attn_scale", "attn_hc.scale"),
            ("hc_ffn_fn", "ffn_hc.fn"),
            ("hc_ffn_base", "ffn_hc.base"),
            ("hc_ffn_scale", "ffn_hc.scale"),
        ):
            rename(f"{prefix}.{source}", f"{prefix}.{target}")
        attn = f"{prefix}.self_attn"
        conv_keys = [f"{attn}.{name}_conv1d.weight" for name in ("q", "k", "v")]
        present = [key in layouts for key in conv_keys]
        if any(present):
            if not all(present):
                raise ValueError("incomplete GLM convolution triple")
            pack(conv_keys, f"{attn}.conv1d.weight", stack=False)
        for source, target in (
            ("A_log", "forget_gate.A_log"),
            ("dt_bias", "forget_gate.dt_bias"),
            ("f_a_proj.weight", "forget_gate.f_a_proj.weight"),
            ("f_b_proj.weight", "forget_gate.f_b_proj.weight"),
        ):
            rename(f"{attn}.{source}", f"{attn}.{target}")
        expert_prefix = f"{prefix}.mlp.experts."
        expert_ids = sorted(
            {
                int(part)
                for name in layouts
                if name.startswith(expert_prefix) and (part := name[len(expert_prefix) :].split(".", 1)[0]).isdigit()
            }
        )
        if not expert_ids:
            continue
        if expert_ids != list(range(len(expert_ids))):
            raise ValueError("non-contiguous GLM expert indices")
        expert_counts[layer] = len(expert_ids)
        for target, source in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
            target_name = f"{expert_prefix}{target}"
            pack([f"{expert_prefix}{expert}.{source}.weight" for expert in expert_ids], target_name, stack=True)
            sharded_names.append(target_name)
        shapes = [layouts[f"{expert_prefix}{name}"][0] for name in ("w1", "w2", "w3")]
        if len(shapes[0]) != 3 or shapes[0] != shapes[2] or shapes[1] != (shapes[0][0], shapes[0][2], shapes[0][1]):
            raise ValueError("GLM gate/up/down matrix shapes are inconsistent")
        rename(f"{prefix}.mlp.gate.weight", f"{prefix}.mlp.router.gate.weight")
        rename(f"{prefix}.mlp.gate.e_score_correction_bias", f"{prefix}.mlp.expert_bias")
        for target, source in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
            old = f"{prefix}.mlp.shared_experts.{source}.weight"
            if old in layouts and dtypes[old] != "BF16":
                raise ValueError("GLM shared expert projection requires BF16")
            rename(old, f"{prefix}.mlp.shared_expert.{target}")
    if _layout_bytes(layouts) != source_bytes:
        raise ValueError("GLM metadata conversion changed total payload bytes")
    return {
        "schema": "glm-bf16-conversion-projection-v1",
        "source_identity": dict(ledger["source_identity"]),
        "layouts": layouts,
        "dtypes": dtypes,
        "source_names": lineage,
        "packed_layouts": packed,
        "sharded_tensor_names": tuple(sharded_names),
        "checkpoint_expert_counts": expert_counts,
        "source_payload_bytes": source_bytes,
        "converted_payload_bytes": _layout_bytes(layouts),
        "packed_payload_bytes": _layout_bytes(packed),
        "converter_recipe": "glm53.convert_hf_layer_to_prime:BF16-unscaled",
        "converted_cache_bound": False,
        "native_inventory_complete": False,
        "capacity_admitted": False,
        "runtime_admitted": False,
        "unresolved_inputs": [
            *ledger["unresolved_inputs"],
            "selected_converter_callable_binding",
            "actual_converted_cache_headers_and_files",
            "actual_native_destination_and_buffer_inventory",
        ],
    }


def glm_ep_dcp_projection(conversion: Mapping[str, Any], *, ep_size: int, ep_rank: int) -> dict:
    """Emit conditional full-chunk DCP rows for an equal BF16 expert partition.

    ``ep_rank`` is an expert-mesh coordinate, not an inferred global rank.
    Only known stacked routed weights use Shard(0); every other converted
    tensor is conditionally replicated. No SP division or actual cache file,
    native destination, installed readplan or usable limit is asserted.
    """
    if conversion.get("schema") != "glm-bf16-conversion-projection-v1" or not conversion.get("layouts"):
        raise ValueError("GLM BF16 conversion projection required")
    if _nonnegative_integer(ep_size, "ep_size") == 0:
        raise ValueError("ep_size must be positive")
    if _nonnegative_integer(ep_rank, "ep_rank") >= ep_size:
        raise ValueError("ep_rank exceeds expert partition")
    sharded = set(conversion["sharded_tensor_names"])
    if not sharded.issubset(conversion["layouts"]):
        raise ValueError("sharded projection key is missing")
    destination = {}
    requests = []
    for name, (shape, itemsize) in conversion["layouts"].items():
        tensor_storage_bytes(shape, itemsize)
        offsets = (0,) * len(shape)
        lengths = tuple(shape)
        if name in sharded:
            if len(shape) != 3 or shape[0] == 0 or shape[0] % ep_size:
                raise ValueError("GLM expert count must divide equally across EP")
            local_experts = shape[0] // ep_size
            offsets = (ep_rank * local_experts, 0, 0)
            lengths = (local_experts, *shape[1:])
        destination[name] = (lengths, itemsize)
        requests.append(DCPReadRequest(name, offsets, lengths))
    return {
        "source_identity": dict(conversion["source_identity"]),
        "ep_size": ep_size,
        "ep_coordinate": ep_rank,
        "destination_layouts": destination,
        "requests": requests,
        "named_destination_bytes": _layout_bytes(destination),
        "native_buffers_bytes": None,
        "native_workspace_bytes": None,
        "usable_limits": None,
        "conditional_only": True,
        "actual_readplan_bound": False,
        "native_inventory_complete": False,
        "capacity_admitted": False,
        "runtime_admitted": False,
        "unresolved_inputs": [
            *conversion["unresolved_inputs"],
            "actual_expert_mesh_and_destination_placement",
            "actual_selected_dcp_readplan",
            "installed_reader_and_native_workspace",
            "actual_usable_resource_limits",
        ],
    }


def hf_read_slice_envelope(
    headers: Mapping[str, CheckpointTensor],
    requests: Sequence[DCPReadRequest],
    *,
    thread_count: int,
) -> dict:
    """Derive the sequential-per-file HF reader's concurrent slice payload.

    The candidate reader loops serially inside each file and assigns at most
    ``min(thread_count, files)`` files to threads. This is NOT a DCP CPU/GPU
    peak: mapped pages, planner tensors, copies and native retention are extra.
    Installed reader/planner/safetensors source and callable binding must be
    supplied separately; this arithmetic never asserts that binding.
    """
    if _nonnegative_integer(thread_count, "thread_count") == 0:
        raise ValueError("thread_count must be positive")
    if not headers or not requests:
        raise ValueError("original tensor headers and selected read requests are required")
    for name, header in headers.items():
        if not name or not header.file_name:
            raise ValueError("named original tensor and file required")
        tensor_storage_bytes(header.shape, header.itemsize)
    per_file: dict[str, int] = {}
    for request in requests:
        if request.tensor_name not in headers:
            raise ValueError(f"missing original header: {request.tensor_name}")
        header = headers[request.tensor_name]
        if not (len(request.offsets) == len(request.lengths) == len(header.shape)):
            raise ValueError("read request dimensionality differs from original header")
        for offset, length, size in zip(request.offsets, request.lengths, header.shape):
            _nonnegative_integer(offset, "offset")
            _nonnegative_integer(length, "length")
            if offset + length > size:
                raise ValueError("read request exceeds original tensor")
        size_bytes = tensor_storage_bytes(request.lengths, header.itemsize)
        per_file[header.file_name] = max(per_file.get(header.file_name, 0), size_bytes)
    concurrency = min(thread_count, len(per_file))
    return {
        "algorithm": "hf-sequential-requests-per-file",
        "concurrent_files": concurrency,
        "per_file_slice_bytes": per_file,
        "slice_payload_bytes": sum(sorted(per_file.values(), reverse=True)[:concurrency]),
        "installed_reader_bound": False,
        "complete_dcp_peak_bytes": None,
        "unresolved_terms": (
            "safe_open_mapped_and_backend_residency",
            "planner_resolution_cpu_and_gpu_staging",
            "copy_native_workspace_and_allocator_retention",
        ),
    }


@dataclass(frozen=True)
class MemoryLifetime:
    """One resource-local term alive through an inclusive phase interval.

    ``size_bytes=None`` is unresolved, never zero. ``last_phase=None`` keeps
    the term live through the final phase because release is not established.
    ``storage_id`` deduplicates only explicitly identified complete aliases
    on the same resource, not replicas on other ranks/devices.
    """

    name: str
    resource: str
    size_bytes: int | None
    first_phase: str
    last_phase: str | None = None
    storage_id: str | None = None


def _layout_bytes(layouts: Mapping[str, tuple[Sequence[int], int]] | None) -> int | None:
    if layouts is None:
        return None
    return sum(tensor_storage_bytes(shape, itemsize) for shape, itemsize in layouts.values())


def _native_backing_storage_terms(inventory, *, name, scope, resources, first_phase):
    if (
        inventory.get("alias_identity_scope") != "process-local"
        or inventory.get("snapshot_only") is not True
        or not scope
        or not isinstance(scope, str)
        or not resources
    ):
        raise ValueError("native snapshot, owning-process birth scope and device resources required")
    storages = {}
    for row in inventory["storages"]:
        key = (row["device"], row["storage_id"])
        if any(not isinstance(value, str) or not value for value in key) or key in storages or key[0] not in resources:
            raise ValueError("unique native storage and actual device resource required")
        _nonnegative_integer(row["nbytes"], "backing storage bytes")
        storages[key] = row
    referenced = set()

    def local_tensor(tensor):
        if tensor["dtensor"] is not None:
            tensor = tensor["dtensor"]["local"]
            if tensor["dtensor"] is not None:
                raise ValueError("ordinary local DTensor backing tensor required")
        shape, stride, width = tensor["shape"], tensor["stride"], tensor["element_size"]
        logical_bytes = tensor_storage_bytes(shape, width)
        if len(shape) != len(stride) or tensor["logical_bytes"] != logical_bytes:
            raise ValueError("native local tensor dimensions or logical bytes differ")
        for value in stride:
            _nonnegative_integer(value, "native stride")
        storage = tensor["storage"]
        if storage is None:
            raise ValueError("materialized complete native backing storage required")
        key = (tensor["device"], storage["storage_id"])
        if key not in storages or storages[key]["nbytes"] != storage["nbytes"]:
            raise ValueError("native tensor backing storage differs from full ledger")
        offset = _nonnegative_integer(storage["offset_elements"], "native storage offset")
        if storage["offset_bytes"] != offset * width:
            raise ValueError("native tensor storage byte offset differs")
        span = 0 if not logical_bytes else 1 + sum((dim - 1) * step for dim, step in zip(shape, stride))
        if (offset + span) * width > storage["nbytes"]:
            raise ValueError("native tensor view exceeds full backing storage")
        referenced.add(key)
        return tensor

    for kind in ("parameters", "buffers"):
        for row in inventory[kind]:
            local_tensor(row["tensor"])
    destinations = {
        key: {"local_tensor": local_tensor(tensor), "dtensor_metadata": tensor["dtensor"]}
        for key, tensor in inventory["checkpoint_destinations"].items()
    }
    if referenced != storages.keys():
        raise ValueError("native full storage ledger has unreferenced entries")
    terms = [
        MemoryLifetime(
            f"{name}.backing_storage.{index}",
            resources[device],
            row["nbytes"],
            first_phase,
            storage_id=f"{scope}:{device}:{storage_id}",
        )
        for index, ((device, storage_id), row) in enumerate(storages.items())
    ]
    return terms, destinations


def _finalized_read_requests(plan, reader_storage, cache_path, headers, destinations):
    if not reader_storage or cache_path is None or not headers or not destinations:
        raise ValueError("selected reader files, cache path, headers and native destinations required")
    root = Path(cache_path)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("absolute selected cache path required")
    requests, rows = [], []
    for item in plan.items:
        if item.type.name != "TENSOR":
            raise ValueError("only tensor reads are supported by the HF reader arithmetic")
        source, destination = item.storage_index, item.dest_index
        if source.fqn not in headers or destination.fqn not in destinations:
            raise ValueError("finalized source or destination tensor is missing")
        file_path = Path(reader_storage[source].relative_path)
        if file_path.is_absolute():
            file_path = file_path.relative_to(root)
        if not file_path.parts or ".." in file_path.parts or file_path.as_posix() != headers[source.fqn].file_name:
            raise ValueError("selected reader file differs from actual cache header")
        for index in (source, destination):
            if index.offset is not None:
                for offset in index.offset:
                    _nonnegative_integer(offset, "metadata index offset")
            if index.index is not None:
                _nonnegative_integer(index.index, "metadata index hint")
        destination_metadata = destinations[destination.fqn]
        local = destination_metadata["local_tensor"]
        if destination_metadata["dtensor_metadata"] is None and destination.offset is not None:
            if tuple(destination.offset) != (0,) * len(local["shape"]):
                raise ValueError("ordinary tensor destination requires an absent or zero index offset")
        offsets, lengths = tuple(item.dest_offsets), tuple(item.lengths)
        if len(offsets) != len(lengths) or len(lengths) != len(local["shape"]):
            raise ValueError("finalized destination dimensionality differs from actual local tensor")
        for offset, length, size in zip(offsets, lengths, local["shape"]):
            _nonnegative_integer(offset, "destination offset")
            _nonnegative_integer(length, "destination length")
            if offset + length > size:
                raise ValueError("finalized destination slice exceeds actual local tensor")
        requests.append(DCPReadRequest(source.fqn, tuple(item.storage_offsets), lengths))
        rows.append(
            {
                "source_fqn": source.fqn,
                "source_index_offset": None if source.offset is None else tuple(source.offset),
                "source_index_hint": source.index,
                "source_offsets": tuple(item.storage_offsets),
                "file_name": file_path.as_posix(),
                "destination_fqn": destination.fqn,
                "destination_index_offset": None if destination.offset is None else tuple(destination.offset),
                "destination_index_hint": destination.index,
                "destination_offsets": offsets,
                "lengths": lengths,
                "destination_device": local["device"],
                "destination_storage": dict(local["storage"]),
                "destination_shape": tuple(local["shape"]),
                "destination_stride": tuple(local["stride"]),
                "destination_element_size": local["element_size"],
                "destination_dtensor": destination_metadata["dtensor_metadata"],
                "destination_resolution_bound": False,
            }
        )
    return requests, rows


def dcp_load_phase_terms(
    source_headers: Mapping[str, CheckpointTensor] | None,
    load_headers: Mapping[str, CheckpointTensor] | None,
    requests: Sequence[DCPReadRequest] | None,
    destination_layouts: Mapping[str, tuple[Sequence[int], int]] | None,
    *,
    name: str,
    cpu_resource: str,
    gpu_resource: str,
    destination_resource: str,
    first_phase: str,
    reader_threads: int,
    reader_binding: Mapping[str, Any] | None,
    cache_miss: bool | None,
    conversion_writer: bool | None,
    packed_layouts: Mapping[str, tuple[Sequence[int], int]] | None,
    saved_layouts: Mapping[str, tuple[Sequence[int], int]] | None,
    cache_release_phase: str | None,
    read_release_phase: str | None,
    limits: Mapping[str, int] | None,
    native_inventory: Mapping[str, Any] | None = None,
    storage_scope: str | None = None,
    storage_resources: Mapping[str, str] | None = None,
    read_plan: Any | None = None,
    reader_storage: Mapping[Any, Any] | None = None,
    cache_path: str | os.PathLike | None = None,
) -> dict:
    """Produce named load lifetimes from caller-bound original/converted layouts.

    Source headers describe the full original conversion dictionary. Load
    headers describe the actual DCP source, which can be a converted cache
    with DIFFERENT keys/shapes. Destination layouts are full rank-local
    storages, not global shapes divided by SP/EP. Count aliases conservatively
    unless an external inventory establishes shared complete storage.

    Cache-hit/nonwriter zeros are conditional on those explicit caller inputs;
    they do not attest cache identity or writer placement. A supplied reader
    descriptor is retained as a source recipe, not installed-callable proof.
    Native/backend/allocator terms stay unknown and block the finite gate.

    Optional reader-callback inputs replace the destination aggregate with
    complete observed backing storages, including nonloaded buffers/adapters.
    ``storage_scope`` is a caller recipe derived from owning-process birth,
    not proof of it. ``read_plan`` supplies source requests and retains local
    destination metadata without calling the planner or reading payloads.
    Destination lookup/DTensor chunk resolution remains separately unbound.
    """
    if not all(isinstance(value, str) and value for value in (name, cpu_resource, gpu_resource, first_phase)):
        raise ValueError("named load scope, resources and first phase required")
    if cpu_resource == gpu_resource or destination_resource not in (cpu_resource, gpu_resource):
        raise ValueError("distinct CPU/GPU resources and actual destination resource required")
    if any(value is not None and type(value) is not bool for value in (cache_miss, conversion_writer)):
        raise ValueError("cache branch and writer selection must be bool or unresolved None")
    if _nonnegative_integer(reader_threads, "reader_threads") == 0:
        raise ValueError("reader_threads must be positive")
    missing = ["installed_reader_planner_native_binding", "original_and_cache_payload_identity"]
    native_terms, destinations, request_rows = None, None, None
    if native_inventory is None:
        if storage_scope is not None or storage_resources is not None or read_plan is not None:
            raise ValueError("native inventory required for storage scope and finalized plan")
    else:
        if destination_layouts is not None:
            raise ValueError("full native storages replace, not duplicate, destination layouts")
        if not storage_resources or any(
            value not in (cpu_resource, gpu_resource) for value in storage_resources.values()
        ):
            raise ValueError("actual native storage resources must belong to this CPU/GPU load scope")
        native_terms, destinations = _native_backing_storage_terms(
            native_inventory, name=name, scope=storage_scope, resources=storage_resources, first_phase=first_phase
        )
        missing.extend(
            (
                "actual_storage_owner_and_device_resource_binding",
                "complete_constructor_and_conversion_lifetimes",
                "selected_planner_destination_resolution",
            )
        )
    if read_plan is None:
        if reader_storage is not None or cache_path is not None:
            raise ValueError("finalized read plan required for reader file bindings")
    else:
        if requests is not None:
            raise ValueError("finalized plan replaces, not duplicates, supplied source requests")
        requests, request_rows = _finalized_read_requests(
            read_plan, reader_storage, cache_path, load_headers, destinations
        )
    if source_headers is None:
        missing.append("complete_original_headers")
    if load_headers is None:
        missing.append("actual_dcp_source_headers")
    if destination_layouts is None and native_terms is None:
        missing.append("actual_rank_local_destination_layouts")
    if requests is None:
        missing.append("actual_selected_readplan")
    if reader_binding is None:
        missing.append("selected_reader_source_recipe")
    if limits is None:
        missing.append("actual_usable_resource_limits")
    elif cpu_resource not in limits or gpu_resource not in limits:
        raise ValueError("actual CPU/GPU resource limits required")
    else:
        for value in limits.values():
            _nonnegative_integer(value, "resource limit")
    if source_headers is not None and not source_headers:
        raise ValueError("original header inventory cannot be empty")
    if load_headers is not None and not load_headers:
        raise ValueError("DCP source header inventory cannot be empty")
    inactive_writer = cache_miss is False or conversion_writer is False
    active_writer = cache_miss is True and conversion_writer is True
    source_bytes = (
        None
        if source_headers is None
        else sum(tensor_storage_bytes(header.shape, header.itemsize) for header in source_headers.values())
    )
    cache_sizes = (
        0 if inactive_writer else source_bytes if active_writer else None,
        0 if inactive_writer else _layout_bytes(packed_layouts) if active_writer else None,
        0 if inactive_writer else _layout_bytes(saved_layouts) if active_writer else None,
    )
    slice_bytes = None
    if load_headers is not None and requests is not None:
        slice_bytes = hf_read_slice_envelope(load_headers, requests, thread_count=reader_threads)[
            "slice_payload_bytes"
        ]
    rows = [
        ("conversion_source", cpu_resource, cache_sizes[0], cache_release_phase),
        ("conversion_pack", cpu_resource, cache_sizes[1], cache_release_phase),
        ("save_contiguous_copies", cpu_resource, cache_sizes[2], cache_release_phase),
        ("save_native", cpu_resource, 0 if inactive_writer else None, None),
        ("read_slice_payload", cpu_resource, slice_bytes, read_release_phase),
        ("mapped_backend_residency", cpu_resource, None, None),
        ("planner_cpu_staging", cpu_resource, None, None),
        ("planner_gpu_staging", gpu_resource, None, None),
        ("copy_native_cpu_and_allocator", cpu_resource, None, None),
        ("copy_native_gpu_and_allocator", gpu_resource, None, None),
    ]
    destination_terms = (
        [MemoryLifetime(f"{name}.destination", destination_resource, _layout_bytes(destination_layouts), first_phase)]
        if native_terms is None
        else native_terms
    )
    result = {
        "terms": (
            destination_terms
            + [
                MemoryLifetime(f"{name}.{suffix}", resource, size_bytes, first_phase, release)
                for suffix, resource, size_bytes, release in rows
            ]
        ),
        "unresolved_inputs": missing,
        "reader_recipe": dict(reader_binding) if reader_binding is not None else None,
        "reader_recipe_is_installed_binding": False,
        "limits": dict(limits) if limits is not None else None,
        "capacity_admitted": False,
        "runtime_admitted": False,
    }
    if native_terms is not None:
        result["storage_scope_is_owner_binding"] = False
        result["destination_request_rows"] = request_rows
    return result


def initialization_phase_plan(
    phases: Sequence[str],
    terms: Sequence[MemoryLifetime],
    *,
    limits: Mapping[str, int],
    unresolved_inputs: Sequence[str],
) -> dict:
    """Compose simultaneous host/device terms; no native admission is implied.

    Feed actual TRAIN8/SAMPLE8 rank-local layouts into this domain calculation.
    Training SP does not divide replicated base storage; sampler wire factors
    are not divided by TP. Assign CPU terms to shared host resources and GPU
    terms to distinct device resources. Unknown release stays overlapping.
    Limits are usable remaining bytes, after other allocation owners/reserves.
    Even a fully finite result is arithmetic, not an installed-runtime verifier.
    """
    if not phases or any(not isinstance(phase, str) or not phase for phase in phases):
        raise ValueError("named ordered phases are required")
    if len(set(phases)) != len(phases):
        raise ValueError("duplicate phase")
    if not terms or not limits:
        raise ValueError("explicit terms and resource limits are required")
    if any(not isinstance(value, str) or not value for value in unresolved_inputs):
        raise ValueError("unresolved inputs must be named")
    indices = {phase: index for index, phase in enumerate(phases)}
    for resource, limit in limits.items():
        if not isinstance(resource, str) or not resource:
            raise ValueError("named resource required")
        _nonnegative_integer(limit, "resource limit")
    intervals = []
    names = set()
    aliases: dict[tuple[str, str], int] = {}
    for term in terms:
        if not isinstance(term.name, str) or not term.name or term.name in names:
            raise ValueError("unique named memory terms are required")
        names.add(term.name)
        if term.resource not in limits:
            raise ValueError(f"resource limit missing: {term.resource}")
        if term.first_phase not in indices or (term.last_phase is not None and term.last_phase not in indices):
            raise ValueError("memory lifetime uses an unknown phase")
        start = indices[term.first_phase]
        end = len(phases) - 1 if term.last_phase is None else indices[term.last_phase]
        if end < start:
            raise ValueError("memory lifetime ends before it starts")
        if term.size_bytes is not None:
            _nonnegative_integer(term.size_bytes, "term size")
        if term.storage_id is not None:
            if not isinstance(term.storage_id, str) or not term.storage_id or term.size_bytes is None:
                raise ValueError("alias identity requires named complete known storage")
            alias = (term.resource, term.storage_id)
            if alias in aliases and aliases[alias] != term.size_bytes:
                raise ValueError("aliased complete storage sizes differ")
            aliases[alias] = term.size_bytes
        intervals.append((term, start, end))
    unused = set(limits).difference(term.resource for term in terms)
    if unused:
        raise ValueError(f"resource has no inventory terms: {sorted(unused)}")
    rows = []
    known_peaks = dict.fromkeys(limits, 0)
    unknowns: dict[str, set[str]] = defaultdict(set)
    for index, phase in enumerate(phases):
        for resource in limits:
            live = [term for term, start, end in intervals if start <= index <= end and term.resource == resource]
            known = {}
            unresolved = []
            for term in live:
                if term.size_bytes is None:
                    unresolved.append(term.name)
                    unknowns[resource].add(term.name)
                else:
                    key = ("alias", term.storage_id) if term.storage_id is not None else ("term", term.name)
                    known[key] = term.size_bytes
            subtotal = sum(known.values())
            known_peaks[resource] = max(known_peaks[resource], subtotal)
            rows.append(
                {
                    "phase": phase,
                    "resource": resource,
                    "known_bytes": subtotal,
                    "peak_bound_bytes": None if unresolved else subtotal,
                    "unresolved_terms": unresolved,
                    "known_terms_exceed_limit": subtotal > limits[resource],
                }
            )
    return {
        "phases": rows,
        "resources": {
            resource: {
                "known_peak_bytes": peak,
                "peak_bound_bytes": None if unknowns[resource] else peak,
                "unresolved_terms": sorted(unknowns[resource]),
                "limit_bytes": limits[resource],
                "known_terms_exceed_limit": peak > limits[resource],
            }
            for resource, peak in known_peaks.items()
        },
        "unresolved_inputs": list(unresolved_inputs),
        "all_named_terms_finite": not any(unknowns.values()),
        "capacity_admitted": False,
        "runtime_admitted": False,
    }


def require_finite_initialization_plan(
    phases: Sequence[str],
    terms: Sequence[MemoryLifetime],
    *,
    limits: Mapping[str, int],
    unresolved_inputs: Sequence[str],
) -> dict:
    """Reject incomplete/oversized arithmetic without granting native authority.

    The installed-reader binding belongs in ``unresolved_inputs`` until the
    selected worker genuinely binds it. Callers must not treat this function
    as Peirce's native ``verify_capacity`` callback: source/runtime/placement
    and complete-inventory validation are still separate obligations.
    """
    plan = initialization_phase_plan(phases, terms, limits=limits, unresolved_inputs=unresolved_inputs)
    if plan["unresolved_inputs"]:
        raise ValueError(f"unresolved initialization inputs: {plan['unresolved_inputs']}")
    for resource, row in plan["resources"].items():
        if row["unresolved_terms"]:
            raise ValueError(f"unresolved memory terms on {resource}: {row['unresolved_terms']}")
        if row["known_terms_exceed_limit"]:
            raise ValueError(f"initialization memory exceeds limit on {resource}")
    return plan


def gpu_mem_preallocate_bytes(training_config: Mapping[str, Any]) -> int:
    """Validate the per-rank reserve, rounding fractional GiB up to bytes."""
    value = training_debug_config(training_config).get("gpu_mem_preallocate_gib", 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("debug.gpu_mem_preallocate_gib must be a finite non-negative number")
    try:
        size_bytes = value * (1 << 30)
        valid = value >= 0 and math.isfinite(size_bytes)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("debug.gpu_mem_preallocate_gib must be a finite non-negative number")
    return math.ceil(size_bytes)


def gpu_memory_debug_config(training_config: Mapping[str, Any]) -> dict:
    """Select only the capacity guard from provider-neutral or legacy diagnostics."""
    debug = training_debug_config(training_config)
    gpu_mem_preallocate_bytes(training_config)
    return {key: debug[key] for key in ("gpu_mem_preallocate_gib",) if key in debug}


def allocate_gpu_mem_preallocation(debug: Mapping[str, Any], *, device=None):
    """Allocate a live byte tensor; OOM propagates without reducing the reserve."""
    size_bytes = gpu_mem_preallocate_bytes({"debug": debug})
    if size_bytes == 0:
        return None
    import torch

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    allocation = torch.empty(size_bytes, dtype=torch.uint8, device=device)
    logger.info(
        "rank=%s [gpu-mem-preallocate] bytes=%d size=%.3fGiB",
        os.environ.get("RANK", "?"),
        size_bytes,
        size_bytes / (1 << 30),
    )
    return allocation
