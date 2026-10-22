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

"""Collective preparation using actual family-owned weight exporters."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from dataclasses import field
from types import FunctionType
from typing import Any

from arctic_platform.debug.weight_sync_evidence import source_files
from arctic_platform.debug.weight_sync_evidence import tensor_manifest
from arctic_platform.debug.weight_sync_evidence import verify_sources


def global_inventory(records: list[dict]) -> dict:
    """Bind full rank coverage and count EP shards once, not once per DP replica."""
    if not isinstance(records, list) or not records:
        raise ValueError("collective inventory requires every actual training rank")
    world = len(records)
    by_rank = {}
    for record in records:
        if (
            not isinstance(record, dict)
            or type(record.get("rank")) is not int
            or not 0 <= record["rank"] < world
            or type(record.get("world_size")) is not int
            or record["world_size"] != world
            or record["rank"] in by_rank
        ):
            raise ValueError("collective rank identities are incomplete or duplicated")
        by_rank[record["rank"]] = record
    result = {}
    for kind in ("parameters", "buffers"):
        first = by_rank[0].get(kind)
        if not isinstance(first, dict) or any(
            not isinstance(record.get(kind), dict) or set(record[kind]) != set(first) for record in records
        ):
            raise ValueError(f"{kind} inventories differ across training ranks")
        inventory = {}
        for name, base in first.items():
            if not isinstance(name, str) or not name or not isinstance(base, dict):
                raise ValueError("inventory tensor names must be nonempty strings")
            groups = set()
            sharded = base.get("ep_ranks") is not None
            for rank in range(world):
                entry = by_rank[rank][kind][name]
                if not isinstance(entry, dict):
                    raise ValueError("inventory tensor records must be dictionaries")
                shape = entry.get("shape")
                if (
                    not isinstance(shape, list)
                    or any(type(size) is not int or size <= 0 for size in shape)
                    or type(entry.get("numel")) is not int
                    or math.prod(shape) != entry["numel"]
                    or not isinstance(entry.get("dtype"), str)
                    or type(entry.get("requires_grad")) is not bool
                    or not isinstance(entry.get("sha256"), str)
                    or len(entry["sha256"]) != 64
                    or any(character not in "0123456789abcdef" for character in entry["sha256"])
                ):
                    raise ValueError("invalid actual tensor geometry")
                if any(entry.get(key) != base.get(key) for key in ("shape", "numel", "dtype", "requires_grad")):
                    raise ValueError(f"rank-local tensor geometry differs: {name}")
                members = entry.get("ep_ranks")
                if (members is not None) != sharded:
                    raise ValueError(f"EP classification differs across ranks: {name}")
                if members is not None:
                    if (
                        not isinstance(members, list)
                        or not members
                        or any(type(member) is not int or not 0 <= member < world for member in members)
                        or rank not in members
                        or len(set(members)) != len(members)
                    ):
                        raise ValueError(f"invalid actual EP group ordering: {name}")
                    groups.add(tuple(members))
            if sharded:
                sizes = {len(group) for group in groups}
                covered: set[int] = set()
                for group in groups:
                    if covered.intersection(group):
                        raise ValueError(f"overlapping or reordered EP group reports: {name}")
                    for member in group:
                        if tuple(by_rank[member][kind][name]["ep_ranks"]) != group:
                            raise ValueError(f"EP group membership disagrees: {name}")
                    covered.update(group)
                if covered != set(range(world)) or len(sizes) != 1:
                    raise ValueError(f"EP groups do not partition the training allocation: {name}")
                # Rank zero's actual EP group owns one complete copy of this tensor.
                selected = by_rank[0][kind][name]["ep_ranks"]
                count = sum(by_rank[rank][kind][name]["numel"] for rank in selected)
            else:
                count = base["numel"]
            inventory[name] = {
                "local_shape": base["shape"],
                "local_numel": base["numel"],
                "global_numel": count,
                "dtype": base["dtype"],
                "requires_grad": base.get("requires_grad"),
                "ep_groups": [list(group) for group in sorted(groups)],
            }
        result[kind] = inventory
    return result


def _bind_sources(admitted: dict, selected: dict) -> dict:
    if (
        not isinstance(admitted, dict)
        or not admitted
        or any(admitted.get(path) != sha for path, sha in selected.items())
    ):
        raise ValueError("selected source origins do not match caller-admitted sources")
    verify_sources(admitted)
    return dict(admitted)


def pack_layer_with_evidence(converter, layer_sd: dict, layer_idx: int, *, layer_prefix=None):
    """Observe the actual converter's lossless cat/split operations and bytes.

    Only whole inputs, concatenation and dimension-zero input splits are
    supported. Unknown arithmetic/mutations fail before dispatch. No element
    IDs, tensor-sized provenance storage or alternate packing recipe is used.
    """
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    if not isinstance(layer_sd, dict) or not layer_sd:
        raise ValueError("packing requires the actual nonempty gathered layer")
    sources = source_files(converter, pack_layer_with_evidence, TorchDispatchMode)
    verify_sources(sources)
    inputs = dict(layer_sd)
    before = {name: tensor_manifest(value) for name, value in inputs.items()}
    plans = {}
    held = {}
    for name, value in inputs.items():
        if not isinstance(name, str) or not name or value.numel() <= 0 or id(value) in plans:
            raise ValueError("packing input names and tensor objects must be unique and nonempty")
        plans[id(value)] = ("input", name)
        held[id(value)] = value

    def plan(value):
        if id(value) not in plans or held[id(value)] is not value:
            raise ValueError("converter emitted or consumed an untracked tensor")
        return plans[id(value)]

    class ObservePacking(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is torch.ops.aten.cat.default:
                values = args[0] if args else kwargs["tensors"]
                dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
                children = [(plan(value), tuple(value.shape)) for value in values]
                if not children:
                    raise ValueError("empty packing concatenation")
                dim %= len(values[0].shape)
                result = func(*args, **kwargs)
                plans[id(result)] = ("cat", dim, children)
                held[id(result)] = result
                return result
            if func in (torch.ops.aten.split.Tensor, torch.ops.aten.split_with_sizes.default):
                value = args[0] if args else kwargs["self"]
                dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
                origin = plan(value)
                if origin[0] != "input" or dim % value.ndim != 0:
                    raise ValueError("packing only supports dimension-zero splits of actual inputs")
                result = func(*args, **kwargs)
                start = 0
                for part in result:
                    stop = start + part.shape[0]
                    plans[id(part)] = ("slice", origin[1], start, stop)
                    held[id(part)] = part
                    start = stop
                return result
            raise ValueError(f"unsupported packing operation before dispatch: {func}")

    with torch.no_grad(), ObservePacking():
        output = converter(layer_sd, layer_idx, layer_prefix=layer_prefix)
    if output is not layer_sd:
        raise ValueError("native converter did not return the actual packing dictionary")
    if before != {name: tensor_manifest(value) for name, value in inputs.items()}:
        raise RuntimeError("converter changed the actual gathered input bytes")
    coverage: dict[str, list[tuple[int, int]]] = {name: [] for name in inputs}
    wire = {}

    def check(actual, expression, pieces):
        kind = expression[0]
        if kind == "cat":
            _, dim, children = expression
            start = 0
            for child, shape in children:
                check(actual.narrow(dim, start, shape[dim]), child, pieces)
                start += shape[dim]
            if start != actual.shape[dim]:
                raise ValueError("packing concatenation geometry changed")
            return
        name = expression[1]
        source = inputs[name]
        size = source.shape[0] if source.ndim else 1
        start, stop = (0, size) if kind == "input" else expression[2:]
        expected = source if kind == "input" else source.narrow(0, start, stop - start)
        if tensor_manifest(actual) != tensor_manifest(expected):
            raise RuntimeError("actual packed byte region differs from its source tensor")
        coverage[name].append((start, stop))
        pieces.append({"source": name, "dim0_start": start, "dim0_stop": stop, **tensor_manifest(expected)})

    for name, value in output.items():
        if not isinstance(name, str) or not name:
            raise ValueError("converter emitted an empty or non-string wire name")
        pieces: list[dict] = []
        check(value, plan(value), pieces)
        wire[name] = {"manifest": tensor_manifest(value), "pieces": pieces}
    for name, regions in coverage.items():
        cursor = 0
        for start, stop in sorted(regions):
            if start != cursor or stop <= start:
                raise ValueError(f"converter dropped or duplicated input byte regions: {name}")
            cursor = stop
        size = inputs[name].shape[0] if inputs[name].ndim else 1
        if cursor != size:
            raise ValueError(f"converter did not emit every actual input element: {name}")
    verify_sources(sources)
    return {"inputs": before, "wire": wire, "source_files": sources}


def verify_gathered_layer(layer_sd: dict, wire_sources: dict) -> dict:
    """Bind the native builder's actual dim-zero gather to each rank's bytes."""
    result = {}
    for name, value in layer_sd.items():
        if name not in wire_sources:
            raise ValueError("native gathered tensor has no actual inventory origin")
        shards = wire_sources[name]["ordered_shards"]
        if not shards:
            raise ValueError("native gathered tensor has no actual shard records")
        shape = list(shards[0]["shape"])
        expected_shape = list(shape)
        if len(shards) > 1:
            if not shape:
                raise ValueError("native dim-zero EP gather cannot gather scalar shards")
            expected_shape[0] *= len(shards)
        if list(value.shape) != expected_shape:
            raise ValueError("native gathered tensor geometry differs from its actual shards")
        actual = []
        for index, entry in enumerate(shards):
            part = value if len(shards) == 1 else value.narrow(0, index * shape[0], shape[0])
            descriptor = tensor_manifest(part)
            if descriptor != {key: entry[key] for key in ("shape", "dtype", "numel", "sha256")}:
                raise ValueError(f"native gathered shard bytes do not bind actual rank holdings: {name}")
            actual.append(descriptor)
        result[name] = actual
    return result


def _observed_iterator(model, builder, converter, proofs, wire_sources):
    """Reuse native iterator code/globals with only its converter cell observed."""
    native = builder(model)
    if not isinstance(native, FunctionType) or native.__globals__ is not builder.__globals__:
        raise ValueError("selected builder does not expose the source-bound native iterator")
    freevars = native.__code__.co_freevars
    if "convert_layer" not in freevars or native.__closure__ is None:
        raise ValueError("native iterator has no explicit converter closure")
    cells = list(native.__closure__)
    index = freevars.index("convert_layer")
    if cells[index].cell_contents is not converter:
        raise ValueError("native iterator selected a different actual converter")

    def observe(layer_sd, layer_idx, *, layer_prefix=None):
        gathered = verify_gathered_layer(layer_sd, wire_sources)
        proof = pack_layer_with_evidence(converter, layer_sd, layer_idx, layer_prefix=layer_prefix)
        proof["gathered_shards"] = gathered
        proofs.append(proof)
        return layer_sd

    def make_cell(value):
        return (lambda: value).__closure__[0]

    cells[index] = make_cell(observe)
    observed = FunctionType(native.__code__, native.__globals__, native.__name__, native.__defaults__, tuple(cells))
    observed.__kwdefaults__ = native.__kwdefaults__
    return observed()


@dataclass
class PreparedWeights:
    """Actor-local actual tensors; not a runtime/capacity/admission receipt."""

    rank: int
    weight_format: str
    weights: list
    manifest: dict
    inventory: dict
    rank_records: list
    sources: dict
    packing: list
    wire_sources: dict
    metadata_sha256: str = field(init=False)

    def __post_init__(self):
        self.metadata_sha256 = self._digest()

    def _digest(self):
        return hashlib.sha256(
            json.dumps(
                [
                    self.rank,
                    self.weight_format,
                    self.manifest,
                    self.inventory,
                    self.rank_records,
                    self.sources,
                    self.packing,
                    self.wire_sources,
                ],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def verify(self):
        if self.metadata_sha256 != self._digest():
            raise RuntimeError("prepared collective metadata changed after actual export")
        if {name: tensor_manifest(value) for name, value in self.weights} != self.manifest:
            raise RuntimeError("prepared actual tensor bytes or names changed after export")
        if len(self.weights) != len(self.manifest):
            raise RuntimeError("prepared actual wire names are duplicated")
        verify_sources(self.sources)

    def receipt(self):
        self.verify()
        return {
            "rank": self.rank,
            "weight_format": self.weight_format,
            "emitted_manifest": self.manifest,
            "global_inventory": self.inventory,
            "rank_records": self.rank_records,
            "source_files": self.sources,
            "metadata_sha256": self.metadata_sha256,
            "packing": self.packing,
            "wire_sources": self.wire_sources,
            "non_wire_buffers": {
                name: entry
                for name, entry in self.inventory["buffers"].items()
                if not any(
                    origin["kind"] == "buffers" and origin["name"] == name for origin in self.wire_sources.values()
                )
            },
            "iterator_binding": {
                "native_code_and_globals_reused": True,
                "observed_closure_cells": ["convert_layer"],
                "unchanged_native_iterator": False,
            },
            "receiver_format_admitted": False,
            "wire_replica_values_checked": True,
            "router_application_admitted": False,
            "capacity_admitted": False,
        }


def prepare_collective_weights(
    model,
    *,
    weight_format: str,
    max_weight_bytes: int,
    admitted_sources: dict,
    expected_world_size: int = 8,
    expected_ep_size: int = 8,
    is_zero3: bool = False,
    checkpoint_dir: str | None = None,
) -> PreparedWeights:
    """Every training rank must call this once before any sender dispatch.

    The selected full-weight exporter owns gathering dimensions and packing.
    LoRA is deliberately not supported here. All emitted weights remain
    actor-local; only rank zero can send them.
    """
    import deepspeed.utils.groups as ds_groups
    import torch.distributed as dist
    from torch.utils._python_dispatch import TorchDispatchMode

    from arctic_platform.model.implementations.glm53.vllm_weights import convert_glm5_next_layer_to_vllm
    from arctic_platform.model.implementations.moe.vllm_weights import build_iter_full_vllm_weights
    from arctic_platform.model.implementations.moe.vllm_weights import named_weight_sync_tensors
    from arctic_platform.model.implementations.moe.vllm_weights import vllm_layer_converter
    from arctic_platform.model.weight_export import weight_export_contract

    if weight_format != "vllm" or is_zero3 is not False or checkpoint_dir is not None:
        raise ValueError("this source-supported GLM path requires explicit vllm, non-ZeRO3, no checkpoint fallback")
    if type(max_weight_bytes) is not int or max_weight_bytes <= 0:
        raise ValueError("a positive explicit emitted-weight byte limit is required")
    if (
        expected_world_size != 8
        or expected_ep_size != 8
        or any(type(size) is not int for size in (expected_world_size, expected_ep_size))
    ):
        raise ValueError("this canonical preparation is restricted to the explicit EP8 training domain")
    if not dist.is_initialized():
        raise RuntimeError("actual training process group is not initialized")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != expected_world_size:
        raise ValueError("actual training world does not match the EP8 request")
    contract = weight_export_contract(model)
    builder = getattr(contract, weight_format, None) if contract is not None else None
    converter = vllm_layer_converter(model)
    if builder is not build_iter_full_vllm_weights or converter is not convert_glm5_next_layer_to_vllm:
        raise ValueError("the actual registered family exporter/converter is not the selected GLM vLLM route")
    if getattr(model, "peft_config", None) or getattr(model, "_is_vlm", False):
        raise ValueError("GLM EP8 preparation does not reinterpret LoRA or VLM export semantics")
    selected = source_files(
        type(model),
        type(model).named_parameters,
        type(model).named_buffers,
        type(model).get_submodule,
        builder,
        converter,
        named_weight_sync_tensors,
        vllm_layer_converter,
        weight_export_contract,
        prepare_collective_weights,
        TorchDispatchMode,
        dist.all_gather,
        dist.all_gather_object,
        dist.get_process_group_ranks,
        ds_groups._get_expert_parallel_group,
    )
    sources = _bind_sources(admitted_sources, selected)
    if (
        "20806894cebc1050d877351bf67764bab44dcf6ac8fd8bbaef56c10027b651a1" not in source_files(builder).values()
        or "543c34347f9d5f67e35a6e9cdcc3f5d47bc45c608751ff78ffd1369ee4975a5a" not in source_files(converter).values()
    ):
        raise ValueError("actual native gather/GLM packing source differs from this selected recipe")

    def local_inventory():
        objects = {"parameters": dict(model.named_parameters()), "buffers": dict(model.named_buffers())}
        output = {}
        for kind, tensors in objects.items():
            output[kind] = {}
            for name, tensor in tensors.items():
                if hasattr(tensor, "ds_id"):
                    raise ValueError("post-gather ZeRO3 tensor lifetimes are unsupported")
                members = None
                if getattr(tensor, "allreduce", True) is False:
                    if not isinstance(getattr(tensor, "group_name", None), str) or not tensor.group_name:
                        raise ValueError("expert shard has no actual named process group")
                    group = ds_groups._get_expert_parallel_group(tensor.group_name)
                    members = list(dist.get_process_group_ranks(group))
                    if len(members) != expected_ep_size:
                        raise ValueError("actual expert process group is not EP8")
                output[kind][name] = {
                    **tensor_manifest(tensor),
                    "requires_grad": tensor.requires_grad,
                    "ep_ranks": members,
                }
        selected_tensors = list(named_weight_sync_tensors(model))
        if len({name for name, _ in selected_tensors}) != len(selected_tensors):
            raise ValueError("native exporter selected duplicate tensor names")
        for name, tensor in selected_tensors:
            if not any(objects[kind].get(name) is tensor for kind in objects):
                raise ValueError("native exporter selected an unregistered tensor")
        names = [name for name, _ in selected_tensors]
        if not set(objects["parameters"]).issubset(names):
            raise ValueError("native exporter excludes a registered parameter")
        return objects, output, names

    before_objects, local, names = local_inventory()
    if not before_objects["parameters"]:
        raise ValueError("canonical GLM export requires actual registered parameters")
    verify_sources(sources)
    rank_record = {"rank": rank, "world_size": world, "wire_inputs": names, **local}
    records: list[Any] = [None] * world
    dist.all_gather_object(records, rank_record)
    inventory = global_inventory(records)
    by_rank = {record["rank"]: record for record in records}
    if any(record.get("wire_inputs") != names for record in records):
        raise ValueError("native exporter input order differs across actual training ranks")
    wire_sources = {}
    for kind in inventory:
        for name in inventory[kind]:
            if name not in names:
                continue
            entry = by_rank[0][kind][name]
            members = entry["ep_ranks"]
            if members is None and any(record[kind][name]["sha256"] != entry["sha256"] for record in records):
                raise ValueError(f"actual replicated wire-input bytes disagree across ranks: {name}")
            normalized = name.replace("._checkpoint_wrapped_module", "")
            if normalized in wire_sources:
                raise ValueError("activation-checkpoint unwrapping collides in the native exporter")
            wire_sources[normalized] = {
                "name": name,
                "kind": kind,
                "ordered_shards": [
                    {"rank": member, **by_rank[member][kind][name]}
                    for member in (members if members is not None else [0])
                ],
            }
    wire_bytes = sum(
        inventory[kind][name]["global_numel"] * before_objects[kind][name].element_size()
        for kind in inventory
        for name in inventory[kind]
        if name in names
    )
    if wire_bytes > max_weight_bytes:
        raise ValueError("actual lossless wire inventory exceeds its explicit emitted-weight byte limit")
    weights = []
    manifest = {}
    packing: list[dict] = []
    emitted_bytes = 0
    iterator = _observed_iterator(model, builder, converter, packing, wire_sources)
    for name, tensor in iterator:
        if rank != 0:
            raise RuntimeError("family exporter emitted weights on a non-sender training rank")
        if not isinstance(name, str) or not name or name in manifest:
            raise ValueError("family exporter emitted a duplicate or empty wire name")
        emitted_bytes += tensor.numel() * tensor.element_size()
        if emitted_bytes > max_weight_bytes:
            raise ValueError("actual emitted stream exceeds its explicit byte limit")
        # Exporters may yield temporary views: retain a private value copy.
        value = tensor.detach().clone()
        weights.append((name, value))
        manifest[name] = tensor_manifest(value)
    if rank == 0:
        full_inputs = {}
        for proof in packing:
            for name, entry in proof["inputs"].items():
                if name in full_inputs:
                    raise ValueError("native layer grouping emitted an input twice")
                full_inputs[name] = entry
            sources = _bind_sources(sources, proof["source_files"])
        if set(full_inputs) != set(wire_sources):
            raise ValueError("native packing did not cover the exact registered wire-input domain")
        packed_manifest = {name: entry["manifest"] for proof in packing for name, entry in proof["wire"].items()}
        if packed_manifest != manifest or emitted_bytes != wire_bytes:
            raise ValueError("actual native emitted stream differs from its complete packing byte proof")
    after_objects, after, after_names = local_inventory()
    if (
        local != after
        or names != after_names
        or weight_export_contract(model) is not contract
        or any(
            set(before_objects[kind]) != set(after_objects[kind])
            or any(after_objects[kind][name] is not tensor for name, tensor in before_objects[kind].items())
            for kind in before_objects
        )
    ):
        raise RuntimeError("actual parameter/buffer identity, bytes, geometry or exporter changed during export")
    if vllm_layer_converter(model) is not converter:
        raise RuntimeError("actual family converter selection changed during export")
    verify_sources(sources)
    return PreparedWeights(rank, weight_format, weights, manifest, inventory, records, sources, packing, wire_sources)


def send_prepared_weights(sender, prepared: PreparedWeights):
    """Send exactly the freshly retained stream, with truthful separate holdings."""
    if not isinstance(prepared, PreparedWeights) or prepared.rank != 0 or not prepared.weights:
        raise ValueError("only the actual rank-zero nonempty prepared stream can be sent")
    prepared.verify()
    sources = _bind_sources(prepared.sources, source_files(type(sender).send, send_prepared_weights))
    selected_send = sender.send
    before = {name: tensor_manifest(value) for name, value in prepared.weights}
    if len(before) != len(prepared.weights) or before != prepared.manifest:
        raise RuntimeError("prepared actual tensor bytes or wire names changed before sending")
    result = selected_send(prepared.weights)
    if not isinstance(result, dict):
        raise TypeError("actual sender did not return a dictionary")
    if before != {name: tensor_manifest(value) for name, value in prepared.weights}:
        raise RuntimeError("prepared actual tensor bytes changed while sending")
    if sender.send != selected_send:
        raise RuntimeError("actual sender callable changed while sending")
    prepared.verify()
    verify_sources(sources)
    parameters = prepared.inventory["parameters"]
    buffers = prepared.inventory["buffers"]
    return {
        **result,
        "sent_manifest": before,
        "trainer_parameters": {
            "numel": sum(entry["global_numel"] for entry in parameters.values()),
            "count": len(parameters),
        },
        "trainer_buffers": {
            "numel": sum(entry["global_numel"] for entry in buffers.values()),
            "count": len(buffers),
            "wire_numel": sum(
                buffers[entry["name"]]["global_numel"]
                for entry in prepared.wire_sources.values()
                if entry["kind"] == "buffers"
            ),
        },
        "collective_preparation": prepared.receipt(),
    }


def bind_receiver_results(
    prepared: PreparedWeights,
    rank_results: list[dict],
    *,
    expected_sampler_ranks: int,
    admitted_sources: dict,
) -> dict:
    """Bind retained actual stream bytes to every caller-held native rank result.

    This consumes the existing dispatcher's receiver evidence, not a new load.
    Native rank identity, format selection, runtime and capacity remain the
    operations owner's admission responsibilities.
    """
    if not isinstance(prepared, PreparedWeights) or prepared.rank != 0 or not prepared.weights:
        raise ValueError("receiver binding requires the actual rank-zero prepared stream")
    if (
        type(expected_sampler_ranks) is not int
        or expected_sampler_ranks <= 0
        or not isinstance(rank_results, list)
        or len(rank_results) != expected_sampler_ranks
    ):
        raise ValueError("receiver binding requires every selected sampling rank")
    sources = _bind_sources(admitted_sources, {**prepared.sources, **source_files(bind_receiver_results)})
    prepared.verify()
    actual = {name: tensor_manifest(value) for name, value in prepared.weights}
    if len(actual) != len(prepared.weights) or actual != prepared.manifest:
        raise RuntimeError("prepared stream changed before receiver binding")
    records = []
    for index, result in enumerate(rank_results):
        record = result.get("weight_sync_verification") if isinstance(result, dict) else None
        if (
            not isinstance(result, dict)
            or result.get("status") != "done"
            or not isinstance(record, dict)
            or record.get("status") != "ok"
        ):
            raise ValueError("a selected sampling rank supplied no successful actual byte evidence")
        origins = record.get("evidence_sources")
        if not isinstance(origins, dict) or not origins:
            raise ValueError("receiver evidence does not bind actual selected loader sources")
        _bind_sources(sources, origins)
        received = record.get("received")
        if not isinstance(received, dict) or set(received) != set(actual):
            raise ValueError("receiver wire names differ from the actual prepared stream")
        reached = set()
        for name, entry in received.items():
            if not isinstance(entry, dict) or any(entry.get(key) != value for key, value in actual[name].items()):
                raise ValueError("receiver actual bytes/shape/dtype/count differ from the prepared stream")
            destinations = entry.get("destinations")
            if (
                not isinstance(destinations, list)
                or not destinations
                or any(not isinstance(destination, str) or not destination for destination in destinations)
            ):
                raise ValueError("an actual received tensor reached no native parameter")
            reached.update(destinations)
        destinations = record.get("destinations")
        if not isinstance(destinations, dict) or set(destinations) != reached:
            raise ValueError("receiver destination coverage does not bind every observed write")
        for value in destinations.values():
            if (
                not isinstance(value, dict)
                or type(value.get("uncovered")) is not int
                or value["uncovered"] != 0
                or type(value.get("mismatched")) is not int
                or value["mismatched"] != 0
            ):
                raise ValueError("native receiver shadow coverage/bytes are not complete")
        for field_name, empty in (
            ("untouched_parameters", {}),
            ("unverifiable_destinations", []),
            ("non_parameter_destinations", []),
        ):
            if field_name not in record or type(record[field_name]) is not type(empty) or record[field_name] != empty:
                raise ValueError("native receiver has unproved or untouched destinations")
        records.append({"result_index": index, "actual_result": result})
    verify_sources(sources)
    prepared.verify()
    if actual != {name: tensor_manifest(value) for name, value in prepared.weights}:
        raise RuntimeError("prepared stream changed during receiver binding")
    return {
        "sent_manifest": actual,
        "actual_rank_results": records,
        "packing": prepared.packing,
        "wire_sources": prepared.wire_sources,
        "source_files": sources,
        "wire_bytes_and_shadow_coverage_bound": True,
        "native_rank_identity_admitted": False,
        "receiver_format_admitted": False,
        "router_application_admitted": False,
        "capacity_admitted": False,
        "overall_admission": False,
    }
