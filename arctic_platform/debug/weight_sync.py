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

"""Weight transfer, collective inventories, and sampler verification."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import json
import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from types import FunctionType
from typing import Any
from uuid import uuid4

# Weight Sync Evidence

VERIFY_ENV = "ARCTIC_INFERENCE_WEIGHT_SYNC_VERIFY"


def enabled() -> bool:
    return os.environ.get(VERIFY_ENV) == "1"


def tensor_manifest(value) -> dict:
    import torch

    if value.layout != torch.strided or value.device.type == "meta":
        raise ValueError("weight-sync manifest requires materialized strided tensors")
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "numel": value.numel(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def source_files(*objects) -> dict[str, str]:
    """Hash the actual selected callable origins, including partials."""
    result = {str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for obj in objects:
        while isinstance(obj, functools.partial):
            obj = obj.func
        path = inspect.getsourcefile(obj)
        if path is None:
            raise ValueError("weight loader has no inspectable source origin")
        actual = Path(path).resolve(strict=True)
        result[str(actual)] = hashlib.sha256(actual.read_bytes()).hexdigest()
    return result


def verify_sources(pins: dict[str, str]) -> None:
    for path, expected in pins.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"weight-sync source changed: {path}")


@contextmanager
def maybe_observe_weight_send(worker, weights):
    """Observe a production-owned send without calling the transport."""
    evidence = {}
    sender = worker._weight_sender
    if not enabled():
        yield evidence
        return
    parameters = list(worker.engine.module.named_parameters())
    if any(hasattr(value, "ds_id") for _, value in parameters):
        raise ValueError("weight-sync evidence does not admit post-gather ZeRO3 tensor views")
    if any(getattr(value, "allreduce", True) is False for _, value in parameters):
        raise ValueError("a single sender cannot prove complete expert-parallel trainer holdings")
    origins = source_files(type(sender).send, type(worker).get_weights)
    manifest = {}
    for name, value in weights:
        if not isinstance(name, str) or not name or name in manifest:
            raise ValueError("weight-sync names must be nonempty and unique")
        manifest[name] = tensor_manifest(value)
    verify_sources(origins)
    yield evidence
    verify_sources(origins)
    if manifest != {name: tensor_manifest(value) for name, value in weights}:
        raise RuntimeError("actually sent tensors changed during weight sync")
    evidence.update(
        sent_manifest=manifest,
        trainer_parameters={
            "numel": sum(value.numel() for _, value in parameters),
            "count": len(parameters),
        },
        evidence_sources=origins,
    )


def maybe_annotate_weight_sync_result(result, evidence):
    if not evidence:
        return result
    if not isinstance(result, dict):
        raise TypeError("actual weight sender did not return a dictionary")
    return {**result, **evidence}


def _storage(value):
    return (str(value.device), value.untyped_storage().data_ptr(), value.untyped_storage().nbytes())


class ShadowVerifier:
    """Replay supported ATen loaders into private storages, never into live NaNs.

    Unknown/custom operations and writes outside registered parameters or
    newly allocated temporaries fail before dispatch. Shared parameter storages,
    nonfloating parameters and quantized loaders are deliberately unsupported.
    """

    def __init__(self, model, *, max_shadow_bytes: int):
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_map

        self.model = model
        self.parameters = dict(model.named_parameters())
        self.aliases = {name: id(value) for name, value in model.named_parameters(remove_duplicate=False)}
        self.layouts = {
            name: (tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
            for name, value in self.parameters.items()
        }
        self.storage_names: dict[tuple[str, int, int], str] = {}
        total = 0
        if not self.parameters:
            raise ValueError("weight-sync model has no parameters")
        for name, value in self.parameters.items():
            if (
                value.device.type == "meta"
                or value.layout != torch.strided
                or not value.is_floating_point()
                or not value.is_contiguous()
                or value.storage_offset() != 0
                or value.numel() == 0
                or value.untyped_storage().nbytes() != value.numel() * value.element_size()
            ):
                raise ValueError(f"unsupported shadow parameter layout: {name}")
            key = _storage(value)
            if key in self.storage_names:
                raise ValueError("shared parameter storages require an explicit supported loader")
            self.storage_names[key] = name
            total += value.numel() * value.element_size()
        if type(max_shadow_bytes) is not int or not 0 < total <= max_shadow_bytes:
            raise ValueError(f"shadow allocation {total} exceeds the explicit {max_shadow_bytes} byte limit")
        self.sources = source_files(
            model.load_weights,
            TorchDispatchMode,
            tree_map,
            *[value.weight_loader for value in self.parameters.values() if hasattr(value, "weight_loader")],
        )
        verify_sources(self.sources)
        self.shadows = {name: torch.full_like(value, float("nan")) for name, value in self.parameters.items()}
        self.shadow_names = {_storage(value): name for name, value in self.shadows.items()}
        self.received: dict[str, dict] = {}
        self.touched: set[str] = set()

    def _check_inventory(self):
        current = dict(self.model.named_parameters())
        aliases = {name: id(value) for name, value in self.model.named_parameters(remove_duplicate=False)}
        if (
            aliases != self.aliases
            or set(current) != set(self.parameters)
            or any(
                current[name] is not value
                or self.storage_names.get(_storage(value)) != name
                or (tuple(value.shape), tuple(value.stride()), value.dtype, value.device) != self.layouts[name]
                for name, value in self.parameters.items()
            )
        ):
            raise RuntimeError("weight loader changed the registered parameter identities or storage")

    @contextmanager
    def observe_load(self, name, value, *, loader):
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_flatten
        from torch.utils._pytree import tree_map

        if not isinstance(name, str) or not name or name in self.received:
            raise ValueError("received tensor names must be nonempty and unique")
        self._check_inventory()
        verify_sources(self.sources)
        self.sources.update(source_files(loader))
        verify_sources(self.sources)
        entry = tensor_manifest(value)
        verifier = self

        class Observe(TorchDispatchMode):
            def __init__(self, shadow):
                super().__init__()
                self.shadow = shadow
                self.owned = {_storage(tensor) for tensor in verifier.shadows.values()} if shadow else set()
                self.written = set()

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if func.namespace not in {"aten", "prims"}:
                    raise ValueError(f"unobservable native weight-loader operation: {func}")
                flat, _ = tree_flatten((args, kwargs))
                inputs = {_storage(t) for t in flat if isinstance(t, torch.Tensor)}

                def substitute(tensor):
                    if not isinstance(tensor, torch.Tensor):
                        return tensor
                    destination = verifier.storage_names.get(_storage(tensor))
                    if destination is None or not self.shadow:
                        return tensor
                    parameter = verifier.parameters[destination]
                    if tensor.dtype != parameter.dtype:
                        raise ValueError("reinterpretation of parameter storage is unsupported")
                    return verifier.shadows[destination].as_strided(
                        tensor.shape, tensor.stride(), tensor.storage_offset()
                    )

                for index, argument in enumerate(func._schema.arguments):
                    if argument.alias_info is None or not argument.alias_info.is_write:
                        continue
                    actual = args[index] if index < len(args) else kwargs.get(argument.name)
                    tensors, _ = tree_flatten(actual)
                    for tensor in tensors:
                        if not isinstance(tensor, torch.Tensor):
                            continue
                        key = _storage(tensor)
                        destination = verifier.storage_names.get(key)
                        if self.shadow:
                            destination = destination or verifier.shadow_names.get(key)
                        if destination is not None:
                            self.written.add(destination)
                        elif key not in self.owned:
                            raise ValueError("weight loader writes an input, buffer or unowned storage")
                call_args = tree_map(substitute, args)
                call_kwargs = tree_map(substitute, kwargs)
                result = func(*call_args, **call_kwargs)
                outputs, _ = tree_flatten(result)
                for tensor in outputs:
                    if isinstance(tensor, torch.Tensor):
                        key = _storage(tensor)
                        if key not in inputs and key not in verifier.storage_names:
                            self.owned.add(key)
                return result

        with torch.no_grad(), Observe(False) as live:
            yield
        # Each replay receives private input too; loaders must not mutate the wire tensor.
        with torch.no_grad(), Observe(True) as replay:
            loader(name, value.detach().clone())
        self._check_inventory()
        verify_sources(self.sources)
        if live.written != replay.written or tensor_manifest(value) != entry:
            raise RuntimeError("shadow loader destination set or received bytes changed")
        entry["destinations"] = sorted(live.written)
        self.received[name] = entry
        self.touched.update(live.written)

    def load(self, name, value, *, loader=None):
        """Debug-only convenience for a live load and its private shadow replay."""
        if loader is None:

            def loader(source_name, tensor):
                return self.model.load_weights([(source_name, tensor)])

        with self.observe_load(name, value, loader=loader):
            return loader(name, value)

    def finish(self):
        import torch

        self._check_inventory()
        verify_sources(self.sources)
        destinations = {}
        for name in sorted(self.touched):
            live = self.parameters[name].detach()
            shadow = self.shadows[name]
            uncovered = torch.isnan(shadow)
            byte_diff = (
                live.reshape(-1).view(torch.uint8).reshape(-1, live.element_size())
                != shadow.reshape(-1).view(torch.uint8).reshape(-1, shadow.element_size())
            ).any(dim=1)
            destinations[name] = {
                "uncovered": int(uncovered.sum().item()),
                "mismatched": int((byte_diff & ~uncovered.reshape(-1)).sum().item()),
            }
        return {
            "status": "ok",
            "received": self.received,
            "destinations": destinations,
            "unverifiable_destinations": [],
            "non_parameter_destinations": [],
            "untouched_parameters": {
                name: value.numel() for name, value in self.parameters.items() if name not in self.touched
            },
            "evidence_sources": self.sources,
        }


@functools.lru_cache(maxsize=1)
def receiver_extension():
    """Resolve the optional external dependency only in an enabled sampling worker."""
    from arctic_inference.server.weight_sync.receiver import WeightSyncExtension

    class WeightSyncEvidenceExtension(WeightSyncExtension):
        def sync_weights(self, *args, **kwargs):
            if not enabled():
                return super().sync_weights(*args, **kwargs)
            if self.model_config.quantization:
                raise ValueError("quantized weight-loader shadow evidence is not implemented")
            self._ap_weight_verification = None
            origins = source_files(WeightSyncExtension.sync_weights, WeightSyncExtension._load_batched)
            verify_sources(origins)
            result = super().sync_weights(*args, **kwargs)
            verify_sources(origins)
            if result.get("status") == "engine_ready":
                return result
            if self._ap_weight_verification is None:
                raise RuntimeError("receiver completed without dispatcher-observed verification")
            return {**result, "weight_sync_verification": self._ap_weight_verification}

        def _load_batched(self, model, engine, received_names=None):
            if not enabled():
                return super()._load_batched(model, engine, received_names)
            limit = int(os.environ.get("ARCTIC_PLATFORM_WEIGHT_SYNC_SHADOW_MAX_BYTES", str(1 << 30)))
            verifier = ShadowVerifier(model, max_shadow_bytes=limit)
            verifier.sources.update(source_files(type(engine).receive_weights))
            verify_sources(verifier.sources)
            loaded = 0
            for name, value in engine.receive_weights():
                verifier.load(name, value)
                if received_names is not None:
                    received_names.append(name)
                loaded += 1
            self._ap_weight_verification = verifier.finish()
            return loaded

        def _load_direct(self, model, engine, received_names=None):
            if not enabled():
                return super()._load_direct(model, engine, received_names)
            raise ValueError("direct parameter writer needs a separate supported shadow observer")

        def _load_direct_zero_copy(self, model, engine):
            if not enabled():
                return super()._load_direct_zero_copy(model, engine)
            raise ValueError("zero-copy receive bypasses observable shadow loads")

    return WeightSyncEvidenceExtension


def __getattr__(name):
    if name == "WeightSyncEvidenceExtension":
        return receiver_extension()
    raise AttributeError(name)


# Ep Weight Sync Evidence


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


# Broadcast Weight Sync Evidence

_SEND = ContextVar("native_broadcast_weight_evidence", default=None)
TRAINING_OPTION = "arctic_platform_weight_sync_verify"
SHADOW_MAX_BYTES_ENV = "ARCTIC_PLATFORM_WEIGHT_SYNC_SHADOW_MAX_BYTES"


def maybe_configure_broadcast_training(ds_config):
    """Consume the opt-in before DeepSpeed sees it in a dedicated trainer process."""
    if TRAINING_OPTION not in ds_config:
        return ds_config
    configured = ds_config[TRAINING_OPTION]
    if type(configured) is not bool:
        raise ValueError(f"{TRAINING_OPTION} must be a boolean")
    result = {key: value for key, value in ds_config.items() if key != TRAINING_OPTION}
    if configured:
        os.environ[VERIFY_ENV] = "1"
    return result


def maybe_configure_broadcast_sampling(sampling, *, verify):
    """Carry verification through the native sampler actor and vLLM child environments."""
    if not verify:
        return sampling
    vllm = deepcopy(sampling.vllm)
    extra_env = vllm.get("extra_env")
    if extra_env is not None and not isinstance(extra_env, dict):
        raise ValueError("weight-sync extra_env must be an object")
    extra_env = dict(extra_env or {})
    if extra_env.get(VERIFY_ENV, "1") != "1":
        raise ValueError("conflicting weight-sync verification environment")
    extra_env[VERIFY_ENV] = "1"
    limits = []
    for value in (extra_env.get(SHADOW_MAX_BYTES_ENV), os.environ.get(SHADOW_MAX_BYTES_ENV)):
        if value is None:
            continue
        try:
            limit = int(value) if isinstance(value, str) else 0
        except ValueError:
            limit = 0
        if limit <= 0:
            raise ValueError(f"{SHADOW_MAX_BYTES_ENV} must be a positive integer byte limit")
        limits.append(limit)
    if len(set(limits)) > 1:
        raise ValueError("conflicting weight-sync shadow byte limits")
    if limits:
        extra_env[SHADOW_MAX_BYTES_ENV] = str(limits[0])
    vllm["extra_env"] = extra_env
    return sampling.model_copy(update={"vllm": vllm}, deep=True)


@contextmanager
def weight_sync_verification_environment():
    """The legacy local client inherits this opt-in; never leak it to later checks."""
    previous = os.environ.get(VERIFY_ENV)
    os.environ[VERIFY_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(VERIFY_ENV, None)
        else:
            os.environ[VERIFY_ENV] = previous


@contextmanager
def maybe_observe_broadcast_send(sender, weights, send):
    """Wrap the stream; the caller still executes its native transport."""
    evidence = {}
    if not enabled():
        yield weights, evidence
        return
    if _SEND.get() is not None:
        raise RuntimeError("nested broadcast weight verification is unsupported")
    state = {"trainer_parameters": None, "trainer_inventory": None}
    token = _SEND.set(state)
    sources = source_files(send, type(sender), maybe_observe_broadcast_send)
    manifest = {}

    def observed():
        for name, value in weights:
            if not isinstance(name, str) or not name or name in manifest:
                raise ValueError("broadcast tensor names must be nonempty and unique")
            entry = tensor_manifest(value)
            manifest[name] = entry
            yield name, value
            if tensor_manifest(value) != entry:
                raise RuntimeError("actually sent broadcast tensor changed during transfer")

    try:
        verify_sources(sources)
        yield observed(), evidence
        verify_sources(sources)
        if not manifest or state["trainer_parameters"] is None:
            raise RuntimeError("broadcast sender has no complete collective trainer inventory")
        sources.update(state["evidence_sources"])
        verify_sources(sources)
        evidence.update(
            sent_manifest=manifest,
            trainer_parameters=state["trainer_parameters"],
            trainer_inventory=state["trainer_inventory"],
            evidence_sources=sources,
        )
    finally:
        _SEND.reset(token)


def maybe_bind_broadcast_trainer_inventory(model, *, is_master, is_zero3):
    """All exporter ranks bind actual EP holdings; count DP replicas only once."""
    if not enabled():
        return
    import torch.distributed as dist

    pass

    if is_zero3:
        raise ValueError("native broadcast evidence does not admit ZeRO3 tensor views")
    if not dist.is_initialized():
        raise RuntimeError("native broadcast evidence requires the actual training process group")
    rank, world = dist.get_rank(), dist.get_world_size()
    sources = source_files(
        maybe_bind_broadcast_trainer_inventory,
        type(model).named_parameters,
        dist.all_gather_object,
        global_inventory,
    )
    parameters = {}
    error = None
    try:
        if is_master != (rank == 0):
            raise ValueError("exporter master does not match the actual training rank")
        for name, value in model.named_parameters():
            if hasattr(value, "ds_id"):
                raise ValueError("native broadcast evidence does not admit ZeRO3 tensor views")
            members = None
            if getattr(value, "allreduce", True) is False:
                import deepspeed.utils.groups as groups

                group_name = getattr(value, "group_name", None)
                if not isinstance(group_name, str) or not group_name:
                    raise ValueError("expert shard has no actual named process group")
                group = groups._get_expert_parallel_group(group_name)
                members = list(dist.get_process_group_ranks(group))
                sources.update(source_files(groups._get_expert_parallel_group, dist.get_process_group_ranks))
            parameters[name] = {
                **tensor_manifest(value),
                "requires_grad": value.requires_grad,
                "ep_ranks": members,
            }
        if not parameters:
            raise ValueError("actual trainer has no registered parameters")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    verify_sources(sources)
    local = {"rank": rank, "world_size": world, "parameters": parameters, "buffers": {}, "error": error}
    records = [None] * world
    dist.all_gather_object(records, local)
    errors = [record.get("error") for record in records if isinstance(record, dict) and record.get("error")]
    if errors:
        raise ValueError("collective trainer inventory failed: " + "; ".join(errors))
    inventory = global_inventory(records)
    _verify_trainer_replicas(records, inventory)
    verify_sources(sources)
    state = _SEND.get()
    if is_master:
        if state is None or state["trainer_parameters"] is not None:
            raise RuntimeError("trainer inventory is not bound to exactly one active broadcast")
        state.update(
            trainer_parameters={
                "numel": sum(entry["global_numel"] for entry in inventory["parameters"].values()),
                "count": len(inventory["parameters"]),
            },
            trainer_inventory=inventory,
            evidence_sources=sources,
        )


def _verify_trainer_replicas(records, inventory):
    by_rank = {record["rank"]: record for record in records}
    for name, entry in inventory["parameters"].items():
        groups = entry["ep_groups"] or [[rank] for rank in sorted(by_rank)]
        for shards in zip(*groups):
            if len({by_rank[rank]["parameters"][name]["sha256"] for rank in shards}) != 1:
                raise ValueError(f"actual trainer replica bytes disagree: {name}")


def maybe_admit_broadcast_receiver(receiver, *, engine_only):
    if not enabled():
        return
    if not engine_only and receiver.model_config.quantization:
        raise ValueError("quantized native broadcast shadow evidence is not implemented")


def maybe_begin_broadcast_load(receiver, model, engine):
    if not enabled():
        return None
    if receiver.model_config.quantization:
        raise ValueError("quantized native broadcast shadow evidence is not implemented")
    limit = int(os.environ.get(SHADOW_MAX_BYTES_ENV, str(1 << 30)))
    verifier = ShadowVerifier(model, max_shadow_bytes=limit)
    verifier.sources.update(source_files(maybe_begin_broadcast_load, type(engine).receive_weights))
    verify_sources(verifier.sources)
    return verifier


def maybe_bind_broadcast_loader(verifier, writer):
    if verifier is not None:
        verify_sources(verifier.sources)
        verifier.sources.update(source_files(type(writer)))
        verify_sources(verifier.sources)


@contextmanager
def maybe_observe_broadcast_tensor(verifier, loader, name, value):
    if verifier is None:
        yield
        return
    with verifier.observe_load(name, value, loader=loader):
        yield


def maybe_broadcast_receiver_result(result, verifier, *, rank):
    if verifier is None:
        return result
    return {
        **result,
        "rank": rank,
        "weight_sync_verification": {**verifier.finish(), "rank": rank},
    }


def maybe_collect_broadcast_verification(results, *, engine_only, llm, rank_offset):
    """Preserve native results and reject missing or duplicated sampler ranks."""
    if not enabled() or engine_only:
        return results
    expected_ranks = llm.vllm_config.parallel_config.world_size
    if (
        type(expected_ranks) is not int
        or expected_ranks < 1
        or type(rank_offset) is not int
        or not isinstance(results, list)
        or len(results) != expected_ranks
        or any(
            not isinstance(row, dict) or not isinstance(row.get("weight_sync_verification"), dict) for row in results
        )
        or any(type(row.get("rank")) is not int for row in results)
        or {row["rank"] for row in results} != set(range(rank_offset, rank_offset + expected_ranks))
        or any(
            type(row["weight_sync_verification"].get("rank")) is not int
            or row["weight_sync_verification"]["rank"] != row["rank"]
            for row in results
        )
    ):
        raise RuntimeError("native broadcast did not return every actual sampler rank's verification")
    return [
        {
            **results[0],
            "all_rank_weight_sync_verification": [row["weight_sync_verification"] for row in results],
        },
        *results[1:],
    ]


# Rl Sampler Evidence

_EXTENSION = "arctic_platform.debug.weight_sync.WeightSyncEvidenceExtension"


def maybe_replica_pool(pool_class):
    """Keep the actual external pool/scheduler; inject only its public worker protocol."""
    if not enabled():
        return pool_class
    return functools.partial(pool_class, worker_cls=_worker_actor())


def _worker_actor():
    import ray

    environment = {VERIFY_ENV: "1"}
    limit_key = "ARCTIC_PLATFORM_WEIGHT_SYNC_SHADOW_MAX_BYTES"
    if limit_key in os.environ:
        environment[limit_key] = os.environ[limit_key]
    return ray.remote(runtime_env={"env_vars": environment})(SamplerEvidenceWorker)


async def maybe_sync_weights(receive_task, send_tasks):
    received, *sent = await asyncio.gather(receive_task, *send_tasks)
    return maybe_sync_weights_result(received, sent)


def maybe_sync_weights_result(received, sent):
    """Format completed receipts without changing the server's transfer awaiters."""
    if not enabled():
        return {"status": "ok"}
    if not isinstance(received, dict) or not isinstance(received.get("workers"), list) or not received["workers"]:
        raise RuntimeError("external pool did not return actual worker sync results")
    for worker in received["workers"]:
        if not isinstance(worker, dict) or not worker.get("all_rank_weight_sync_verification"):
            raise RuntimeError("a sampling worker returned no all-rank byte verification")
    return {
        "status": "ok",
        "weight_format": "nccl",
        "targets": [{"send": sent, "recv": received}],
    }


class SamplerEvidenceWorker:
    """External ReplicaPool worker protocol, without unwrapping its Ray actor.

    The external TransferSchedule, sender, NCCL engine, receiver superclass and
    scheduler remain selected. A per-actor lock excludes generate/sleep during
    sync. This worker does not implement native router replay or speculative sync.
    """

    def __init__(self):
        self.llm = None
        self.state = "uninitialized"
        self._lock = asyncio.Lock()
        self._origins = None
        self._tp = None

    async def initialize(self, engine_kwargs, extra_env=None):
        if self.llm is not None:
            raise RuntimeError("sampler evidence worker is already initialized")
        if extra_env:
            os.environ.update(extra_env)
        if not enabled():
            raise RuntimeError("sampler evidence opt-in must propagate into the actual worker")
        import vllm.plugins
        from arctic_inference.envs import arctic_inference_effective_enabled
        from arctic_inference.server.metrics import RingStatLogger

        vllm.plugins.load_general_plugins()
        from vllm.v1.engine.async_llm import AsyncLLM

        if arctic_inference_effective_enabled():
            from arctic_inference.vllm.args import ArcticAsyncEngineArgs as EngineArgs
        else:
            from vllm.engine.arg_utils import AsyncEngineArgs as EngineArgs

        kwargs = dict(engine_kwargs)
        existing = kwargs.get("worker_extension_cls")
        if existing not in (None, _EXTENSION, "arctic_inference.server.weight_sync.WeightSyncExtension"):
            raise ValueError("sampler evidence cannot replace another selected worker extension")
        if kwargs.get("speculative_config"):
            raise ValueError("speculative weight-sync evidence is not implemented")
        if os.environ.get("VLLM_BATCH_INVARIANT") == "1":
            kwargs.setdefault("attention_config", {"backend": "FLASH_ATTN"})
        kwargs["worker_extension_cls"] = _EXTENSION
        # Unlike a compatibility fallback, never silently remove engine recipe fields.
        args = EngineArgs(**kwargs)
        self._tp = args.tensor_parallel_size
        if type(self._tp) is not int or self._tp < 1:
            raise ValueError("native engine tensor-parallel size must be positive")
        if self._tp == 1:
            raise ValueError("native TP1 selects an unsupported direct weight loader")
        self._origins = source_files(type(self), EngineArgs, AsyncLLM.from_vllm_config)
        verify_sources(self._origins)
        from arctic_platform.debug.lora import maybe_admit_sampler_construction
        from arctic_platform.debug.lora import maybe_verify_sampler_construction

        maybe_admit_sampler_construction(self, args, AsyncLLM.from_vllm_config, engine_kwargs)
        self.llm = AsyncLLM.from_vllm_config(args.create_engine_config(), stat_loggers=[RingStatLogger])
        try:
            maybe_verify_sampler_construction(self)
            verify_sources(self._origins)
        except BaseException:
            self.shutdown()
            raise
        self.state = "ready"

    def _ready(self):
        if self.llm is None or self.state != "ready":
            raise RuntimeError(f"sampler not ready: {self.state}")
        verify_sources(self._origins)

    async def generate(self, prompt, sampling_params):
        # Native routing/cache/application must be supplied, not simulated by a marker.
        if any(key.startswith("dss_") for key in sampling_params):
            raise ValueError("external native router replay producer/transport/application is not implemented")
        from vllm import SamplingParams

        async with self._lock:
            self._ready()
            params = SamplingParams(**dict(sampling_params))
            request_id = str(uuid4())
            prompt_input = {"prompt_token_ids": prompt} if isinstance(prompt, list) else prompt
            final = None
            try:
                async for output in self.llm.generate(prompt_input, params, request_id=request_id):
                    final = output
            except BaseException:
                await self.llm.abort(request_id)
                raise
            verify_sources(self._origins)
            if final is None or not final.outputs:
                raise RuntimeError("empty generation output")
            choice = final.outputs[0]
            result = {
                "text": choice.text,
                "token_ids": list(choice.token_ids),
                "finish_reason": choice.finish_reason,
                "prompt_len": len(final.prompt_token_ids or []),
                "generation_len": len(choice.token_ids),
                "prefix_cache_len": int(getattr(final, "num_cached_tokens", None) or 0),
            }
            for key, positions in (("prompt_logprobs", final.prompt_logprobs), ("logprobs", choice.logprobs)):
                if positions is not None:
                    result[key] = [
                        (
                            None
                            if position is None
                            else {
                                token: (
                                    {"logprob": value.logprob, "rank": value.rank}
                                    if hasattr(value, "logprob")
                                    else {"logprob": float(value), "rank": None}
                                )
                                for token, value in position.items()
                            }
                        )
                        for position in positions
                    ]
            return result

    async def sync_weights(
        self,
        master_addr,
        master_port,
        rank_offset,
        world_size,
        bucket_size=256 * 1024 * 1024,
        engine_only=False,
        direct_mode=False,
        reverse=False,
    ):
        async with self._lock:
            self._ready()
            results = await self.llm.collective_rpc(
                "sync_weights",
                args=(
                    master_addr,
                    master_port,
                    rank_offset,
                    world_size,
                    bucket_size,
                    engine_only,
                    direct_mode,
                    reverse,
                ),
            )
            verify_sources(self._origins)
            if not isinstance(results, list) or len(results) != self._tp:
                raise RuntimeError("collective RPC did not return every selected TP rank")
            if engine_only:
                if any(not isinstance(row, dict) or row.get("status") != "engine_ready" for row in results):
                    raise RuntimeError("native receiver engine initialization failed")
                return {"status": "engine_ready", "rank_results": results}
            if any(
                not isinstance(row, dict)
                or row.get("status") != "done"
                or not isinstance(row.get("weight_sync_verification"), dict)
                for row in results
            ):
                raise RuntimeError("a native TP rank did not return receiver byte/shadow evidence")
            return {
                "status": "done",
                "rank_results": results,
                "all_rank_weight_sync_verification": [row["weight_sync_verification"] for row in results],
            }

    async def sleep(self, level=1, offload_weights=False):
        async with self._lock:
            self._ready()
            if offload_weights:
                await self.llm.collective_rpc("offload_model_weights")
            await self.llm.collective_rpc("sleep", kwargs={"level": level})
            verify_sources(self._origins)
            self.state = "sleeping"
            return {"status": "sleeping", "level": level}

    async def wake_up(self, tags=None, restore_weights=False):
        async with self._lock:
            if self.state == "ready":
                return {"status": "already_ready"}
            if self.state not in {"sleeping", "weights_ready"}:
                raise RuntimeError(f"cannot wake sampler in {self.state}")
            verify_sources(self._origins)
            await self.llm.collective_rpc("wake_up", kwargs={"tags": tags})
            if restore_weights:
                await self.llm.collective_rpc("backload_model_weights")
            verify_sources(self._origins)
            self.state = "weights_ready" if set(tags or []) == {"weights"} else "ready"
            return {"status": self.state}

    async def reset_prefix_cache(self):
        async with self._lock:
            if self.llm is None:
                raise RuntimeError("sampler is not initialized")
            verify_sources(self._origins)
            await self.llm.reset_prefix_cache()
            verify_sources(self._origins)
            return {"status": "prefix_cache_reset"}

    async def close_weight_sync(self):
        async with self._lock:
            if self.llm is None:
                raise RuntimeError("sampler is not initialized")
            # Resource cleanup is not conditional on a fresh source-admission check.
            results = await self.llm.collective_rpc("close_weight_sync")
            return {"status": "ok", "rank_results": results}

    async def compute_weight_norm(self):
        async with self._lock:
            self._ready()
            results = await self.llm.collective_rpc("compute_weight_norm")
            verify_sources(self._origins)
            squared = sum(row["sq_sum"] for row in results)
            return {
                "norm": squared**0.5,
                "sq_sum": squared,
                "num_params": sum(row["num_params"] for row in results),
            }

    def is_healthy(self):
        return self.llm is not None and self.state in {"ready", "sleeping", "weights_ready"}

    def get_state(self):
        return self.state

    def get_stats(self):
        from arctic_inference.server.metrics import get_collector

        return {"state": self.state, "pid": os.getpid(), **get_collector().latest()}

    def drain_metrics(self):
        from arctic_inference.server.metrics import get_collector

        return {"pid": os.getpid(), "snapshots": get_collector().drain_snapshots()}

    def set_replica_id(self, replica_id):
        from arctic_inference.server.metrics import get_collector

        get_collector().set_replica_id(int(replica_id))

    def pid(self):
        return os.getpid()

    def shutdown(self):
        if self.llm is not None:
            self.llm.shutdown()
            self.llm = None
        self.state = "uninitialized"
