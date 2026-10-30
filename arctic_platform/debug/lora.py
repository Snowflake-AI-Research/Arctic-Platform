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

"""LoRA adapters, training, receiver verification, and sampling."""

from __future__ import annotations

import asyncio
import base64
import functools
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import marshal
import math
import os
import re
import stat
import sys
from contextlib import contextmanager
from contextlib import nullcontext
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from types import CodeType
from types import MethodType
from typing import Any
from uuid import uuid4

from arctic_platform.debug import weight_sync
from arctic_platform.debug.weight_sync import VERIFY_ENV
from arctic_platform.debug.weight_sync import SamplerEvidenceWorker
from arctic_platform.debug.weight_sync import enabled
from arctic_platform.debug.weight_sync import global_inventory
from arctic_platform.debug.weight_sync import source_files
from arctic_platform.debug.weight_sync import tensor_manifest
from arctic_platform.debug.weight_sync import verify_sources

# Adapter Storage

NATIVE_WEIGHTS_SHA256 = "767c82ea9d3a302f14046af43c638d41389a6b5136b02b08ead6ab9128b9191c"


@dataclass(frozen=True)
class ObservedLoRAOperation:
    value: object
    evidence: dict


def _adapter_storage_require(condition, message):
    if not condition:
        raise ValueError(message)


def _sources(admitted_sources, *objects):
    _adapter_storage_require(enabled(), "LoRA storage evidence must be explicitly enabled")
    observed = source_files(_sources, *objects)
    _adapter_storage_require(
        type(admitted_sources) is dict
        and observed.keys() <= admitted_sources.keys()
        and all(admitted_sources[path] == digest for path, digest in observed.items()),
        "every selected adapter callable requires an admitted source identity",
    )
    verify_sources(admitted_sources)
    return dict(admitted_sources)


def _storage(tensor):
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr(), storage.nbytes()


def _full_tensor(tensor):
    import torch

    _adapter_storage_require(
        isinstance(tensor, torch.Tensor)
        and tensor.layout == torch.strided
        and tensor.device.type != "meta"
        and tensor.is_floating_point()
        and tensor.numel() > 0
        and tensor.is_contiguous()
        and tensor.storage_offset() == 0
        and tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size(),
        "adapter evidence requires a nonempty floating tensor owning its full contiguous storage",
    )
    return tensor.numel() * tensor.element_size()


def _finite_tensors(tensors):
    import torch

    _adapter_storage_require(
        all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors),
        "adapter evidence requires finite input tensors",
    )


def _budget(total, maximum):
    _adapter_storage_require(
        type(maximum) is int and 0 < total <= maximum, "private adapter shadow bytes exceed the explicit limit"
    )


def _receipt(kind, origins, **fields):
    return {
        "schema": "observed-lora-native-operation-v1",
        "operation": kind,
        **fields,
        "evidence_sources": origins,
        "complete_adapter_inventory_verified": False,
        "tensor_parallel_slot_routing_verified": False,
        "adapter_storage_verified": False,
        "adapter_application_verified": False,
        "full_parameter_coverage": False,
        "capacity_admitted": False,
        "numerical": None,
        "native_python_calls": 1,
        "shadow_budget_scope": (
            "retained private tensor storages; excludes manifests, coverage masks and runtime scratch"
        ),
    }


def _native_weight_origin(method):
    function = getattr(method, "__func__", method)
    return (
        getattr(function, "__module__", None) == "vllm.lora.lora_weights"
        and NATIVE_WEIGHTS_SHA256 in source_files(method).values()
    )


def _factors(weights):
    import torch

    packed = type(weights.lora_a) is list and type(weights.lora_b) is list
    _adapter_storage_require(
        packed or (isinstance(weights.lora_a, torch.Tensor) and isinstance(weights.lora_b, torch.Tensor)),
        "only actual dense factors or ordered packed factor lists are supported",
    )
    a = weights.lora_a if packed else [weights.lora_a]
    b = weights.lora_b if packed else [weights.lora_b]
    scales = weights.scaling if packed else [weights.scaling]
    _adapter_storage_require(
        type(scales) is list
        and len(a) == len(b) == len(scales)
        and a
        and all(type(scale) in (float, int) and math.isfinite(scale) for scale in scales),
        "complete finite scaling and A/B factors are required; missing factors are unsupported",
    )
    _adapter_storage_require(
        type(weights.rank) is int and weights.rank > 0, "actual positive native adapter rank required"
    )
    total, storages, values = 0, set(), {}
    for index, (left, right) in enumerate(zip(a, b)):
        for kind, tensor in (("A", left), ("B", right)):
            total += _full_tensor(tensor)
            key = _storage(tensor)
            _adapter_storage_require(
                key not in storages, "shared factor storage requires a separate source-derived transform"
            )
            storages.add(key)
            _adapter_storage_require(tensor.ndim in (2, 3), "only matrix or stacked-expert factors are supported")
            rank_axis = -2 if kind == "A" else -1
            _adapter_storage_require(
                tensor.shape[rank_axis] == weights.rank,
                "factor rank differs from the actual native adapter rank",
            )
            values[f"{index}.{kind}"] = tensor
    return values, list(scales), total


def observe_stacked_moe_packing(loras, module_name, *, pack, max_shadow_bytes, admitted_sources):
    """Invoke the selected stacked pack once and verify ordered tensor aliases."""
    _adapter_storage_require(
        type(loras) in (list, tuple) and len(loras) == 3, "exactly ordered w1/w2/w3 weights required"
    )
    _adapter_storage_require(type(module_name) is str and module_name, "actual native packed module name required")
    _adapter_storage_require(
        getattr(pack, "__name__", None) == "pack_moe_stacked" and isinstance(getattr(pack, "__self__", None), type),
        "the selected native stacked classmethod is required",
    )
    origins = _sources(admitted_sources, pack, *[type(weights) for weights in loras])
    factors, before, total = {}, [], 0
    rank, alpha, scaling = loras[0].rank, loras[0].lora_alpha, loras[0].scaling
    for index, weights in enumerate(loras):
        _adapter_storage_require(
            weights.module_name == f"{module_name}.w{index + 1}"
            and weights.rank == rank
            and weights.lora_alpha == alpha
            and weights.scaling == scaling,
            "stacked native projection order, rank, alpha and scaling must agree",
        )
        values, scales, size = _factors(weights)
        _adapter_storage_require(
            len(scales) == 1 and all(tensor.ndim == 3 for tensor in values.values()), "stacked factors required"
        )
        total += size
        _budget(total, max_shadow_bytes)
        _finite_tensors(values.values())
        factors.update({f"w{index + 1}.{kind[-1]}": tensor for kind, tensor in values.items()})
        before.append({name: tensor_manifest(tensor) for name, tensor in values.items()})
    _budget(total, max_shadow_bytes)
    _adapter_storage_require(
        len({_storage(tensor) for tensor in factors.values()}) == 6, "cross-projection aliases are unsupported"
    )
    _adapter_storage_require(
        getattr(pack.__self__, "pack_moe_stacked") == pack,
        "selected native stacked classmethod changed before invocation",
    )
    result = pack(loras, module_name)
    verify_sources(origins)
    _adapter_storage_require(
        type(result) is pack.__self__
        and result.module_name == module_name
        and result.rank == rank
        and result.lora_alphas == [alpha] * 3
        and result.scaling == [scaling] * 3
        and type(result.lora_a) is list
        and type(result.lora_b) is list
        and len(result.lora_a) == len(result.lora_b) == 3
        and getattr(pack.__self__, "pack_moe_stacked") == pack,
        "native stacked packing changed its selected method or metadata",
    )
    for index, weights in enumerate(loras):
        _adapter_storage_require(
            result.lora_a[index] is factors[f"w{index + 1}.A"]
            and result.lora_b[index] is factors[f"w{index + 1}.B"]
            and weights.lora_a is factors[f"w{index + 1}.A"]
            and weights.lora_b is factors[f"w{index + 1}.B"]
            and weights.module_name == f"{module_name}.w{index + 1}"
            and weights.rank == rank
            and weights.lora_alpha == alpha
            and weights.scaling == scaling,
            "native stacked packing did not retain exact ordered factor aliases",
        )
        _adapter_storage_require(
            before[index] == {"0.A": tensor_manifest(weights.lora_a), "0.B": tensor_manifest(weights.lora_b)},
            "native stacked packing changed input factor bytes",
        )
    return ObservedLoRAOperation(
        result,
        _receipt(
            "stacked_moe_packing",
            origins,
            module_name=module_name,
            projection_order=["w1", "w2", "w3"],
            rank=rank,
            lora_alpha=alpha,
            scaling=[scaling] * 3,
            packed_factors={name: tensor_manifest(tensor) for name, tensor in factors.items()},
            factor_bytes=total,
            private_shadow_bytes=0,
            ordered_aliases_verified=True,
            matched_retained_native_weights_source=_native_weight_origin(pack),
        ),
    )


def observe_lora_scaling(weights, *, max_shadow_bytes, admitted_sources):
    """Observe one actual optimize call against private same-dtype B factors."""
    import torch

    optimize = weights.optimize
    origins = _sources(admitted_sources, optimize, type(weights))
    factors, scales, total = _factors(weights)
    b_values = {name: tensor for name, tensor in factors.items() if name.endswith(".B")}
    shadow_bytes = sum(tensor.numel() * tensor.element_size() for tensor in b_values.values())
    _budget(shadow_bytes, max_shadow_bytes)
    _finite_tensors(factors.values())
    metadata = (weights.module_name, weights.rank, weights.lora_alpha)
    packed_alphas = list(weights.lora_alphas) if type(weights.lora_a) is list else None
    before = {name: tensor_manifest(tensor) for name, tensor in factors.items()}
    layouts = {
        name: (id(tensor), _storage(tensor), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
        for name, tensor in factors.items()
    }
    with torch.no_grad():
        expected = {name: tensor.detach().clone() for name, tensor in b_values.items()}
        for name, tensor in expected.items():
            scale = scales[int(name.split(".")[0])]
            if scale != 1:
                tensor.mul_(scale)
        result = optimize()
    verify_sources(origins)
    _adapter_storage_require(
        result is weights
        and weights.optimize == optimize
        and (weights.module_name, weights.rank, weights.lora_alpha) == metadata
        and (packed_alphas is None or weights.lora_alphas == packed_alphas),
        "native optimize replaced its object, metadata or selected method",
    )
    current, after_scales, after_bytes = _factors(weights)
    _finite_tensors(current.values())
    _adapter_storage_require(
        after_bytes == total and all(scale == 1 for scale in after_scales), "native optimize metadata differs"
    )
    _adapter_storage_require(
        current.keys() == factors.keys()
        and all(
            (id(tensor), _storage(tensor), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype) == layouts[name]
            for name, tensor in current.items()
        ),
        "native optimize changed factor identities, layouts or storage",
    )
    after = {name: tensor_manifest(tensor) for name, tensor in current.items()}
    _adapter_storage_require(
        all(
            after[name] == (tensor_manifest(expected[name]) if name in expected else before[name]) for name in current
        ),
        "native optimize bytes differ from unchanged A and same-dtype scaled B",
    )
    return ObservedLoRAOperation(
        result,
        _receipt(
            "factor_scaling",
            origins,
            module_name=weights.module_name,
            input_factors=before,
            optimized_factors=after,
            input_scaling=scales,
            output_scaling=after_scales,
            factor_bytes=total,
            private_shadow_bytes=shadow_bytes,
            same_dtype_scaling_bytes_verified=True,
            matched_retained_native_weights_source=_native_weight_origin(optimize),
        ),
    )


def _storage_tensor(owner, path):
    _adapter_storage_require(
        type(path) in (tuple, list)
        and path
        and type(path[0]) is str
        and path[0]
        and all(type(index) is int and index >= 0 for index in path[1:]),
        "storage paths require one actual attribute followed by nonnegative list indices",
    )
    fields = vars(owner)
    name = path[0]
    if name in fields:
        value = fields[name]
    else:
        registered = [fields.get(key, {}) for key in ("_buffers", "_parameters")]
        matches = [mapping[name] for mapping in registered if name in mapping]
        _adapter_storage_require(len(matches) == 1, "actual storage attribute is absent or ambiguous")
        value = matches[0]
    for index in path[1:]:
        _adapter_storage_require(
            type(value) in (tuple, list) and index < len(value), "actual storage list index is absent"
        )
        value = value[index]
    return value


def observe_lora_storage_write(
    setter, *args, storage_paths, factor_tensors, max_shadow_bytes, admitted_sources, **kwargs
):
    """Mirror ATen operations from one bound set_lora call, not the Python call.

    ``storage_paths`` selects actual full native storage allocations, not
    caller-created destinations. Every write view is retained with its offset,
    shape and strides. Uncovered elements remain explicit and are not admission.
    """
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    from torch.utils._pytree import tree_map

    owner = getattr(setter, "__self__", None)
    _adapter_storage_require(
        owner is not None
        and not isinstance(owner, type)
        and getattr(setter, "__name__", None) == "set_lora"
        and owner.set_lora == setter,
        "actual bound native layer set_lora method required",
    )
    origins = _sources(admitted_sources, setter, TorchDispatchMode, tree_flatten, tree_map)
    _adapter_storage_require(
        type(storage_paths) is dict
        and storage_paths
        and type(factor_tensors) is dict
        and factor_tensors
        and all(type(name) is str and name for name in (*storage_paths, *factor_tensors)),
        "nonempty named actual storage paths and factor tensors required",
    )
    paths = {name: tuple(path) for name, path in storage_paths.items()}
    destinations = {name: _storage_tensor(owner, path) for name, path in paths.items()}
    seeds = {
        **{("factor", name): tensor for name, tensor in factor_tensors.items()},
        **{("destination", name): tensor for name, tensor in destinations.items()},
    }
    sizes = {name: _full_tensor(tensor) for name, tensor in seeds.items()}
    total = sum(sizes.values()) + sum(size for name, size in sizes.items() if name[0] == "destination")
    _budget(total, max_shadow_bytes)
    _finite_tensors(seeds.values())
    keys = {_storage(tensor): name for name, tensor in seeds.items()}
    _adapter_storage_require(
        len(keys) == len(seeds), "aliased factors or destination allocations require explicit supported mapping"
    )
    initial = {name: tensor_manifest(tensor) for name, tensor in factor_tensors.items()}
    layouts = {
        name: (id(tensor), _storage(tensor), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
        for name, tensor in destinations.items()
    }
    with torch.no_grad():
        baselines = {name: tensor.detach().clone() for name, tensor in destinations.items()}
        shadows = {
            _storage(tensor): tensor.detach().clone() if name[0] == "factor" else torch.full_like(tensor, float("nan"))
            for name, tensor in seeds.items()
        }
    live_storages = {_storage(tensor) for tensor in seeds.values()}
    live_bases = {_storage(tensor): tensor for tensor in seeds.values()}
    factor_keys = {_storage(tensor): name for name, tensor in factor_tensors.items()}
    destination_keys = {_storage(tensor): name for name, tensor in destinations.items()}
    dependencies = {key: {factor_keys[key]} if key in factor_keys else set() for key in live_storages}
    dtype_by_key = {_storage(tensor): tensor.dtype for tensor in seeds.values()}
    writes, consumed = [], set()
    private_bytes = total

    class Observe(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            nonlocal private_bytes
            kwargs = kwargs or {}
            _adapter_storage_require(
                func.namespace in {"aten", "prims"}, f"unobservable native adapter operation: {func}"
            )
            _adapter_storage_require(
                torch.Tag.nondeterministic_seeded not in func.tags,
                "random adapter operations cannot be mirrored as deterministic loading",
            )
            flat, _ = tree_flatten((args, kwargs))
            tensors = [value for value in flat if isinstance(value, torch.Tensor)]
            input_keys = {_storage(tensor) for tensor in tensors}
            _adapter_storage_require(input_keys <= live_storages, "native setter read an unbound tensor storage")
            inherited = set().union(*(dependencies[key] for key in input_keys))
            pending, write_keys = [], set()
            for index, argument in enumerate(func._schema.arguments):
                if argument.alias_info is None or not argument.alias_info.is_write:
                    continue
                actual = args[index] if index < len(args) else kwargs.get(argument.name)
                operands, _ = tree_flatten(actual)
                for tensor in operands:
                    if not isinstance(tensor, torch.Tensor):
                        continue
                    key = _storage(tensor)
                    _adapter_storage_require(
                        key not in factor_keys, "native setter cannot overwrite bound adapter factors"
                    )
                    _adapter_storage_require(key in live_storages, "native setter wrote an unowned tensor storage")
                    write_keys.add(key)
                    if key in destination_keys:
                        pending.append(
                            {
                                "operator": str(func),
                                "destination": destination_keys[key],
                                "shape": list(tensor.shape),
                                "stride": list(tensor.stride()),
                                "storage_offset_elements": tensor.storage_offset(),
                                "numel": tensor.numel(),
                                "logical_write_bytes": tensor.numel() * tensor.element_size(),
                                "factor_dependencies": sorted(inherited),
                            }
                        )

            def substitute(tensor):
                if not isinstance(tensor, torch.Tensor):
                    return tensor
                key = _storage(tensor)
                _adapter_storage_require(
                    tensor.dtype == dtype_by_key[key], "native setter reinterpreted tracked storage dtype"
                )
                mirror = shadows[key]
                return mirror.as_strided(tensor.shape, tensor.stride(), tensor.storage_offset())

            mirror_args = tree_map(substitute, args)
            mirror_kwargs = tree_map(substitute, kwargs)
            result = func(*args, **kwargs)
            actual, _ = tree_flatten(result)
            new = {}
            for tensor in actual:
                if isinstance(tensor, torch.Tensor):
                    key = _storage(tensor)
                    if key not in live_storages:
                        new[key] = tensor.untyped_storage().nbytes()
            _budget(private_bytes + sum(new.values()), max_shadow_bytes)
            replay = func(*mirror_args, **mirror_kwargs)
            mirrored, _ = tree_flatten(replay)
            _adapter_storage_require(len(actual) == len(mirrored), "operator mirror return structure differs")
            for tensor, mirror in zip(actual, mirrored):
                _adapter_storage_require(
                    isinstance(tensor, torch.Tensor) == isinstance(mirror, torch.Tensor), "mirror type differs"
                )
                if not isinstance(tensor, torch.Tensor):
                    _adapter_storage_require(
                        type(tensor) is type(mirror) and tensor == mirror, "operator mirror scalar differs"
                    )
                    continue
                key = _storage(tensor)
                _adapter_storage_require(
                    tensor.shape == mirror.shape
                    and tensor.stride() == mirror.stride()
                    and tensor.dtype == mirror.dtype
                    and tensor.storage_offset() == mirror.storage_offset(),
                    "operator mirror tensor layout differs",
                )
                if key not in live_storages:
                    _adapter_storage_require(
                        tensor.storage_offset() == 0
                        and tensor.is_contiguous()
                        and tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size(),
                        "new native scratch requires full contiguous tracked storage",
                    )
                    live_storages.add(key)
                    live_bases[key] = tensor
                    shadows[key] = mirror
                    dtype_by_key[key] = tensor.dtype
                    dependencies[key] = set(inherited)
                    private_bytes += new[key]
                else:
                    _adapter_storage_require(
                        _storage(mirror) == _storage(shadows[key]), "operator mirror storage alias differs"
                    )
            for key in write_keys:
                dependencies[key].update(inherited)
            for entry in pending:
                consumed.update(entry["factor_dependencies"])
            writes.extend(pending)
            return result

    with torch.no_grad(), Observe():
        result = setter(*args, **kwargs)
    verify_sources(origins)
    _adapter_storage_require(
        owner.set_lora == setter
        and all(
            _storage_tensor(owner, paths[name]) is tensor
            and (id(tensor), _storage(tensor), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
            == layouts[name]
            for name, tensor in destinations.items()
        ),
        "native setter changed its callable or actual storage identities/layouts",
    )
    _adapter_storage_require(
        initial == {name: tensor_manifest(tensor) for name, tensor in factor_tensors.items()},
        "bound adapter factor bytes changed during native setter",
    )
    _adapter_storage_require(
        writes and consumed == factor_tensors.keys(), "not every bound adapter factor reaches an observed write"
    )
    for entry in writes:
        live = destinations[entry["destination"]]
        shadow = shadows[_storage(live)]
        for tensor in (live, shadow):
            written = tensor.as_strided(entry["shape"], entry["stride"], entry["storage_offset_elements"])
            _finite_tensors([written])
    coverage = {}
    for name, live in destinations.items():
        shadow = shadows[_storage(live)]
        covered = ~torch.isnan(shadow.reshape(-1))
        finite = torch.isfinite(shadow.reshape(-1))
        different = (
            live.detach().reshape(-1).view(torch.uint8).reshape(-1, live.element_size())
            != shadow.reshape(-1).view(torch.uint8).reshape(-1, live.element_size())
        ).any(dim=1)
        mismatch = int((different & covered).sum().item())
        untouched_different = (
            live.detach().reshape(-1).view(torch.uint8).reshape(-1, live.element_size())
            != baselines[name].reshape(-1).view(torch.uint8).reshape(-1, live.element_size())
        ).any(dim=1)
        unobserved = int((untouched_different & ~covered).sum().item())
        _adapter_storage_require(
            mismatch == 0 and unobserved == 0 and bool(finite[covered].all().item()),
            "native adapter write differs from private mirror or changed uncovered storage",
        )
        coverage[name] = {
            "storage_bytes": live.untyped_storage().nbytes(),
            "full_numel": live.numel(),
            "covered_elements": int(covered.sum().item()),
            "uncovered_elements": int((~covered).sum().item()),
            "mismatched_covered_elements": mismatch,
            "changed_uncovered_elements": unobserved,
            "actual_storage": tensor_manifest(live),
            "private_shadow": tensor_manifest(shadow),
        }
    return ObservedLoRAOperation(
        result,
        _receipt(
            "individual_native_storage_write",
            origins,
            selected_layer_class=f"{type(owner).__module__}.{type(owner).__qualname__}",
            storage_paths={name: list(path) for name, path in paths.items()},
            factor_inputs=initial,
            factor_names_reaching_observed_writes=sorted(consumed),
            writes=writes,
            destinations=coverage,
            private_shadow_bytes=private_bytes,
            observed_write_shadow_verified=True,
            elementwise_factor_origin_verified=False,
            native_python_call_replayed=False,
        ),
    )


# Adapter Binding

NATIVE_MODEL_SHA256 = "57d0c10b374a29e2b4fda826dfd9657712394c97ddf6a8099783f718ae2860a4"
NATIVE_PARSER_SHA256 = "49e8ff84a4725c963f685b4828eec1f70066969b3e46ca65c0be784df692a83f"
NATIVE_MAPPER_SHA256 = "7171b6347b3cbab213df299fa317094ac6c9b2b397daf6a2ec93ff3a152b0e75"


def _adapter_binding_require(condition, message):
    if not condition:
        raise ValueError(message)


def _config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def _source_binding(admitted_sources, *objects):
    _adapter_binding_require(enabled(), "adapter tensor binding must be explicitly enabled")
    observed = source_files(_source_binding, ObservedLoRAOperation, *objects)
    _adapter_binding_require(
        type(admitted_sources) is dict
        and observed.keys() <= admitted_sources.keys()
        and all(admitted_sources[path] == digest for path, digest in observed.items()),
        "every actual adapter builder/parser/mapper callable needs admitted source identity",
    )
    verify_sources(admitted_sources)
    return dict(admitted_sources)


def _mapper_calls(root):
    if root is None:
        return []
    calls = [
        type(root),
        root.get_rename_mapper,
        root._map_name,
        root.map_name,
        root._map_name_with_shard,
    ]
    for renaming in root.orig_to_new_renaming:
        calls.append(renaming.rename_source_key)
    return calls


def _project_names(tensors, *, prefix, remap_key, parse_name, mapper):
    projected, remapped, occupied = {}, set(), set()
    for name in tensors:
        _adapter_binding_require(
            type(name) is str and name.endswith((".lora_A.weight", ".lora_B.weight")),
            "only original received A/B weight names are supported",
        )
        rewritten = remap_key(name, prefix)
        _adapter_binding_require(
            type(rewritten) is str and rewritten and rewritten not in remapped, "AP name remapping collided"
        )
        remapped.add(rewritten)
        parsed = parse_name(rewritten, mapper)
        _adapter_binding_require(
            type(parsed) is tuple
            and len(parsed) == 2
            and type(parsed[0]) is str
            and parsed[0]
            and type(parsed[1]) is bool,
            "actual native parser did not return a module name and factor kind",
        )
        module, is_a = parsed
        kind = "A" if is_a else "B"
        _adapter_binding_require((module, kind) not in occupied, "native parser/mapper collided on an adapter factor")
        occupied.add((module, kind))
        projected[name] = {"rewritten": rewritten, "module": module, "kind": kind}
    return projected


def _expected_modules(projected):
    modules, groups = {}, {}
    for wire_name, entry in projected.items():
        modules.setdefault(entry["module"], {})[entry["kind"]] = wire_name
    _adapter_binding_require(
        all(set(parts) == {"A", "B"} for parts in modules.values()), "every native module requires both A and B"
    )
    order = list(modules)
    for name in modules:
        for index in (1, 2, 3):
            if name == f"experts.w{index}" or name.endswith(f".experts.w{index}"):
                group = name[:-3]
                groups.setdefault(group, {})[index] = name
                break
    expected = {name: {"packed": False, "parts": [parts]} for name, parts in modules.items()}
    for group, parts in groups.items():
        _adapter_binding_require(set(parts) == {1, 2, 3}, "native stacked expert group requires exactly w1/w2/w3")
        _adapter_binding_require(group not in expected, "packed expert module collides with a received module")
        projections = [parts[index] for index in (1, 2, 3)]
        expected[group] = {"packed": True, "parts": [modules[name] for name in projections]}
        order.append(group)
        for name in projections:
            del expected[name]
            order.remove(name)
    return expected, order


def _factor_bytes(tensors, dtype, maximum):
    import torch

    _adapter_binding_require(
        type(maximum) is int and maximum > 0, "explicit positive received-factor byte bound required"
    )
    total = 0
    for tensor in tensors.values():
        _adapter_binding_require(
            isinstance(tensor, torch.Tensor)
            and tensor.layout == torch.strided
            and tensor.device.type == "cpu"
            and tensor.dtype == dtype
            and tensor.numel() > 0
            and tensor.is_contiguous()
            and tensor.storage_offset() == 0
            and tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size(),
            "received factors require full contiguous CPU storage in the exact requested dtype; casts unsupported",
        )
        total += tensor.numel() * tensor.element_size()
        _adapter_binding_require(total <= maximum, "received adapter factors exceed the explicit binding byte limit")
    _adapter_binding_require(
        all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors.values()),
        "received adapter factors must be finite",
    )
    return total


def _verify_built(result, *, expected, order, manifests, rank, alpha, dtype, identity):
    import torch

    _adapter_binding_require(
        type(result.id) is int
        and result.id == identity
        and type(result.rank) is int
        and result.rank == rank
        and type(result.loras) is dict
        and list(result.loras) == order
        and not result.modules_to_save
        and result.is_3d_lora_weight is False,
        "native built adapter identity, ordered module domain or unsupported full/3D view differs",
    )
    bound = {}
    scaling = alpha / rank
    for name, domain in expected.items():
        weights = result.loras[name]
        _adapter_binding_require(
            weights.module_name == name and type(weights.rank) is int and weights.rank == rank,
            "native factor module name/rank differs",
        )
        if domain["packed"]:
            _adapter_binding_require(
                type(weights.lora_a) is list
                and type(weights.lora_b) is list
                and len(weights.lora_a) == len(weights.lora_b) == 3
                and weights.lora_alpha == 0
                and weights.lora_alphas == [alpha] * 3
                and weights.scaling == [scaling] * 3,
                "native stacked expert lists/order/alpha/scaling differ",
            )
            a, b = weights.lora_a, weights.lora_b
        else:
            _adapter_binding_require(
                weights.lora_alpha == alpha and weights.scaling == scaling,
                "native dense alpha/scaling differs from the exact builder recipe",
            )
            a, b = [weights.lora_a], [weights.lora_b]
        for index, parts in enumerate(domain["parts"]):
            for kind, tensor in (("A", a[index]), ("B", b[index])):
                _adapter_binding_require(
                    isinstance(tensor, torch.Tensor)
                    and tensor.device.type == "cpu"
                    and tensor.dtype == dtype
                    and tensor.ndim == (3 if domain["packed"] else 2)
                    and tensor.shape[-2 if kind == "A" else -1] == rank,
                    "native factor shape/rank/dtype/device differs; no transpose or cast is inferred",
                )
                wire_name = parts[kind]
                actual = tensor_manifest(tensor)
                _adapter_binding_require(
                    actual == manifests[wire_name],
                    "native built factor bytes differ from the original received tensor",
                )
                bound[wire_name] = {
                    "native_module": name,
                    "factor_kind": kind,
                    "packed_projection": f"w{index + 1}" if domain["packed"] else None,
                    "native_factor": actual,
                }
    _adapter_binding_require(bound.keys() == manifests.keys(), "received adapter factors were ignored or duplicated")
    return bound


def observe_lora_build(
    builder,
    *,
    lora_int_id,
    tensors,
    lora_config,
    device,
    dtype,
    model,
    native_model_class,
    parse_name,
    remap_key,
    served_prefix_for_model,
    max_binding_bytes,
    admitted_sources,
):
    """Replace the caller's builder invocation once, preserving all its arguments."""
    import torch

    _adapter_binding_require(type(tensors) is dict and tensors, "genuine retained received tensor dictionary required")
    _adapter_binding_require(type(lora_int_id) is int and lora_int_id > 0, "actual positive adapter identity required")
    _adapter_binding_require(
        type(lora_config) is dict
        and type(lora_config.get("r")) is int
        and lora_config["r"] > 0
        and type(lora_config.get("lora_alpha")) is int
        and lora_config["lora_alpha"] > 0
        and lora_config.get("bias") == "none"
        and not lora_config.get("use_rslora")
        and not lora_config.get("use_dora")
        and not lora_config.get("modules_to_save"),
        "exact supported unscaled adapter-only builder recipe required",
    )
    _adapter_binding_require(
        (device == "cpu" or isinstance(device, torch.device) and device.type == "cpu" and device.index is None)
        and dtype in (torch.bfloat16, torch.float16, torch.float32),
        "actual CPU builder and explicit floating dtype required",
    )
    globals_ = getattr(builder, "__globals__", {})
    packing = globals_.get("_pack_fused_expert_loras")
    constructor = native_model_class.from_lora_tensors
    constructor_globals = constructor.__func__.__globals__
    _adapter_binding_require(
        globals_.get("_remap_lora_key") is remap_key
        and globals_.get("_served_lm_module_prefix") is served_prefix_for_model
        and callable(packing)
        and constructor.__self__ is native_model_class
        and constructor_globals.get("parse_fine_tuned_lora_name") is parse_name,
        "actual builder helper and native constructor/parser bindings required",
    )
    root_mapper = getattr(model, "hf_to_vllm_mapper", None) if model is not None else None
    root_calls = _mapper_calls(root_mapper)
    selected = [
        builder,
        remap_key,
        served_prefix_for_model,
        packing,
        native_model_class,
        constructor,
        parse_name,
        *root_calls,
    ]
    constructor_symbols = {}
    for symbol in ("LoRALayerWeights", "PEFTHelper"):
        _adapter_binding_require(
            symbol in constructor_globals, "native constructor factor/config source binding is absent"
        )
        constructor_symbols[symbol] = constructor_globals[symbol]
        selected.append(constructor_symbols[symbol])
    if model is not None:
        selected.append(type(model))
        if hasattr(model, "named_modules"):
            selected.append(model.named_modules)
    origins = _source_binding(admitted_sources, *selected)
    config_sha256 = _config_digest(lora_config)
    prefix = served_prefix_for_model(model)
    _adapter_binding_require(type(prefix) is str and prefix, "actual served model prefix required")
    mapper = root_mapper.get_rename_mapper() if root_mapper is not None else None
    mapper_calls = _mapper_calls(mapper)
    if mapper is not None:
        origins = _source_binding(origins, *mapper_calls)
        _adapter_binding_require(
            not mapper.orig_to_new_stacked, "actual LoRA rename mapper cannot retain stacked weight sharding"
        )
    projection = _project_names(tensors, prefix=prefix, remap_key=remap_key, parse_name=parse_name, mapper=mapper)
    expected, order = _expected_modules(projection)
    total = _factor_bytes(tensors, dtype, max_binding_bytes)
    manifests = {name: tensor_manifest(tensor) for name, tensor in tensors.items()}
    identities = {name: id(tensor) for name, tensor in tensors.items()}
    verify_sources(origins)
    result = builder(
        lora_int_id=lora_int_id,
        tensors=tensors,
        lora_config=lora_config,
        device=device,
        dtype=dtype,
        model=model,
    )
    verify_sources(origins)
    _adapter_binding_require(type(result) is native_model_class, "native builder returned a different adapter class")
    _adapter_binding_require(
        globals_.get("_remap_lora_key") is remap_key
        and globals_.get("_served_lm_module_prefix") is served_prefix_for_model
        and globals_.get("_pack_fused_expert_loras") is packing
        and native_model_class.from_lora_tensors == constructor
        and constructor_globals.get("parse_fine_tuned_lora_name") is parse_name
        and all(constructor_globals.get(name) is obj for name, obj in constructor_symbols.items())
        and (getattr(model, "hf_to_vllm_mapper", None) if model is not None else None) is root_mapper
        and _mapper_calls(root_mapper) == root_calls
        and _config_digest(lora_config) == config_sha256
        and served_prefix_for_model(model) == prefix
        and identities == {name: id(tensor) for name, tensor in tensors.items()}
        and manifests == {name: tensor_manifest(tensor) for name, tensor in tensors.items()},
        "actual builder bindings/config/prefix or original received tensor identities/bytes changed",
    )
    current_mapper = root_mapper.get_rename_mapper() if root_mapper is not None else None
    current_calls = _mapper_calls(current_mapper)
    # Rename-only mapper copies are fresh objects; compare their selected code,
    # not bound-method receiver identity.
    _adapter_binding_require(
        [getattr(call, "__func__", call) for call in current_calls]
        == [getattr(call, "__func__", call) for call in mapper_calls],
        "actual rename mapper selected different implementations during native build",
    )
    _source_binding(origins, *current_calls)
    _adapter_binding_require(
        (current_mapper is None or not current_mapper.orig_to_new_stacked)
        and _project_names(tensors, prefix=prefix, remap_key=remap_key, parse_name=parse_name, mapper=current_mapper)
        == projection,
        "actual rename mapper changed the received factor domain during native build",
    )
    bound = _verify_built(
        result,
        expected=expected,
        order=order,
        manifests=manifests,
        rank=lora_config["r"],
        alpha=lora_config["lora_alpha"],
        dtype=dtype,
        identity=lora_int_id,
    )
    verify_sources(origins)
    matched = (
        native_model_class.__module__ == "vllm.lora.lora_model"
        and native_model_class.__name__ == "LoRAModel"
        and NATIVE_MODEL_SHA256 in source_files(native_model_class).values()
        and getattr(parse_name, "__module__", None) == "vllm.lora.utils"
        and NATIVE_PARSER_SHA256 in source_files(parse_name).values()
        and (
            root_mapper is None
            or type(root_mapper).__module__ == "vllm.model_executor.models.utils"
            and NATIVE_MAPPER_SHA256 in source_files(type(root_mapper)).values()
        )
    )
    return ObservedLoRAOperation(
        result,
        {
            "schema": "observed-lora-native-build-binding-v1",
            "coverage_domain": "actual built loras A/B factors, not served parameters or manager slots",
            "lora_int_id": lora_int_id,
            "rank": lora_config["r"],
            "lora_alpha": lora_config["lora_alpha"],
            "config_sha256": config_sha256,
            "dtype": str(dtype).removeprefix("torch."),
            "received_bytes": total,
            "wire_order": list(tensors),
            "built_module_order": list(result.loras),
            "served_prefix": prefix,
            "name_projection": projection,
            "factor_binding": bound,
            "original_received": manifests,
            "all_received_adapter_factors_bound": True,
            "ignored_received_tensors": [],
            "extra_built_tensors": [],
            "casts_or_transposes_inferred": False,
            "built_factor_byte_correspondence_verified": True,
            "built_factor_alias_identity_required": False,
            "mapper_projection_unchanged": True,
            "native_builder_calls": 1,
            "native_model_reconstructed": False,
            "matched_retained_model_parser_mapper_sources": matched,
            "evidence_sources": origins,
            "runtime_admitted": False,
            "manager_layer_coverage_verified": False,
            "tensor_parallel_slot_routing_verified": False,
            "adapter_storage_verified": False,
            "adapter_application_verified": False,
            "full_parameter_coverage": False,
            "capacity_admitted": False,
            "numerical": None,
        },
    )


# Receiver Binding

_MANAGER_METHODS = ("add_adapter", "activate_adapter", "get_adapter", "remove_adapter")


def _receiver_binding_require(condition, message):
    if not condition:
        raise ValueError(message)


def _origins(admitted_sources, *callables):
    observed = source_files(_origins, *callables)
    _receiver_binding_require(
        type(admitted_sources) is dict
        and observed
        and observed.keys() <= admitted_sources.keys()
        and all(admitted_sources[path] == digest for path, digest in observed.items()),
        "actual receiver/manager callbacks require admitted source identities",
    )
    verify_sources(admitted_sources)
    return dict(admitted_sources)


@dataclass
class OwnedLoRAReceiverBinding:
    receiver: object
    runner: object
    native_model: object
    wrapper: object
    manager: object
    adapter_id: int
    callbacks: dict
    sources: dict
    adapter: object = None
    closed: bool = False
    events: list = field(default_factory=list)

    def guard(self):
        _receiver_binding_require(not self.closed, "receiver binding is already closed")
        verify_sources(self.sources)
        _receiver_binding_require(
            self.receiver.model_runner is self.runner
            and self.runner.model is self.native_model
            and self.runner.lora_manager is self.wrapper
            and self.wrapper._adapter_manager is self.manager
            and all(getattr(self.manager, name) == method for name, method in self.callbacks.items()),
            "actual receiver/manager identity or captured callbacks changed",
        )

    def reserve(self, adapter):
        """Own the exact model before any manager method can mutate native state."""
        self.guard()
        _receiver_binding_require(self.adapter is None, "this receiver binding already reserved an adapter")
        _receiver_binding_require(
            getattr(adapter, "id", None) == self.adapter_id, "built adapter id differs from the owned id"
        )
        _receiver_binding_require(
            self.callbacks["get_adapter"](self.adapter_id) is None, "cannot replace an existing adapter"
        )
        owned = getattr(self.receiver, "_ap_owned_lora_installations", None)
        if owned is None:
            owned = self.receiver._ap_owned_lora_installations = {}
        _receiver_binding_require(type(owned) is dict and self.adapter_id not in owned, "adapter id is already owned")
        owned[self.adapter_id] = (
            self.manager,
            adapter,
            self.callbacks["get_adapter"],
            self.callbacks["remove_adapter"],
        )
        self.adapter = adapter
        self.events.append("reserved")

    def install(self, adapter, *, installer):
        """Call the selected installer once; retain ownership on any rejection."""
        _receiver_binding_require(self.adapter is None, "native installation may only be attempted once")
        self.sources = _origins(self.sources, installer)
        self.reserve(adapter)
        added, activated = installer(self.manager, adapter, self.adapter_id)
        self.guard()
        _receiver_binding_require(
            added is True and activated is True and self.callbacks["get_adapter"](self.adapter_id) is adapter,
            "native manager did not retain the exact added and activated adapter",
        )
        self.events.append("installed")
        return {
            "schema": "owned-live-lora-manager-installation-v1",
            "adapter_id": self.adapter_id,
            "added": added,
            "activated": activated,
            "captured_manager_callbacks": list(self.callbacks),
            "evidence_sources": dict(self.sources),
            "native_slot_binding_admitted": False,
            "adapter_storage_verified": False,
            "adapter_application_verified": False,
            "capacity_admitted": False,
            "numerical": None,
        }

    def cleanup(self):
        """Use captured owned callbacks even after receiver or source drift."""
        if self.closed:
            return {"owned_adapter_empty": True, "already_closed": True}
        if self.adapter is not None:
            current = self.callbacks["get_adapter"](self.adapter_id)
            _receiver_binding_require(
                current is None or current is self.adapter, "refusing to remove a different owner's adapter"
            )
            if current is self.adapter:
                self.callbacks["remove_adapter"](self.adapter_id)
            _receiver_binding_require(
                self.callbacks["get_adapter"](self.adapter_id) is None, "owned adapter remains after removal"
            )
            owned = getattr(self.receiver, "_ap_owned_lora_installations", {})
            expected = (
                self.manager,
                self.adapter,
                self.callbacks["get_adapter"],
                self.callbacks["remove_adapter"],
            )
            _receiver_binding_require(
                type(owned) is dict and owned.get(self.adapter_id) == expected,
                "owned receiver installation ticket changed",
            )
            del owned[self.adapter_id]
        self.closed = True
        self.events.append("cleaned")
        return {"owned_adapter_empty": True, "already_closed": False}


def capture_receiver_manager(receiver, *, adapter_id, admitted_sources):
    """Capture the existing receiver manager, never an imported replacement."""
    _receiver_binding_require(enabled(), "LoRA receiver evidence must be explicitly enabled")
    _receiver_binding_require(type(adapter_id) is int and adapter_id > 0, "positive actual adapter id required")
    runner = receiver.model_runner
    wrapper = getattr(runner, "lora_manager", None)
    manager = getattr(wrapper, "_adapter_manager", None)
    _receiver_binding_require(
        manager is not None and runner.model is not None, "actual loaded receiver manager/model required"
    )
    callbacks = {name: getattr(manager, name) for name in _MANAGER_METHODS}
    _receiver_binding_require(
        all(inspect.ismethod(method) and method.__self__ is manager for method in callbacks.values()),
        "native adapter operations must be bound to the exact live manager",
    )
    sources = _origins(admitted_sources, capture_receiver_manager, type(manager), *callbacks.values())
    binding = OwnedLoRAReceiverBinding(
        receiver,
        runner,
        runner.model,
        wrapper,
        manager,
        adapter_id,
        callbacks,
        sources,
    )
    binding.guard()
    _receiver_binding_require(callbacks["get_adapter"](adapter_id) is None, "adapter id is already installed")
    return binding


@contextmanager
def owned_receiver_binding(receiver, *, adapter_id, admitted_sources, teardown):
    """Always attempt captured teardown, including after admission/cleanup errors."""
    _receiver_binding_require(callable(teardown), "captured receiver/gateway teardown callback required")
    binding, primary = None, None
    try:
        sources = _origins(admitted_sources, teardown)
        binding = capture_receiver_manager(receiver, adapter_id=adapter_id, admitted_sources=sources)
        yield binding
    except BaseException as error:
        primary = error
        raise
    finally:
        secondary = []
        if binding is not None:
            try:
                binding.cleanup()
            except BaseException as error:
                secondary.append(("owned adapter cleanup", error))
        try:
            teardown()
        except BaseException as error:
            secondary.append(("captured receiver/gateway teardown", error))
        if primary is not None:
            for phase, cleanup_error in secondary:
                primary.add_note(phase + " also failed: " + repr(cleanup_error))
        elif secondary:
            phase, cleanup_error = secondary[0]
            for later_phase, later in secondary[1:]:
                cleanup_error.add_note(later_phase + " also failed: " + repr(later))
            cleanup_error.add_note("failure phase: " + phase)
            raise cleanup_error


# Lora Weight Sync Evidence

_EXPORT_SHA = "677781f51ac8fe1c28cb39787a78689a0c04ccefb77971ad13f626eb32ca8940"


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


# Receiver

LORA_EXTENSION = "arctic_platform.debug.lora.LoRAWeightSyncExtension"


def _receiver_require(condition, message):
    if not condition:
        raise ValueError(message)


def engine_kwargs_for_lora(engine_kwargs, *, rank):
    """Bind the explicitly LoRA-enabled engine to this named receiver extension."""
    _receiver_require(enabled(), "LoRA sampler evidence must be explicitly enabled")
    _receiver_require(type(rank) is int and rank > 0, "actual positive configured LoRA rank required")
    kwargs = dict(engine_kwargs)
    _receiver_require(kwargs.get("enable_lora") is True, "the selected engine must enable LoRA explicitly")
    _receiver_require(
        type(kwargs.get("max_lora_rank")) is int and kwargs["max_lora_rank"] >= rank,
        "the selected engine must admit the configured adapter rank",
    )
    _receiver_require(not kwargs.get("speculative_config"), "speculative adapter sync is not implemented")
    _receiver_require(
        kwargs.get("worker_extension_cls")
        in (
            None,
            LORA_EXTENSION,
            "arctic_inference.server.weight_sync.WeightSyncExtension",
        ),
        "cannot replace a different selected receiver extension",
    )
    kwargs["worker_extension_cls"] = LORA_EXTENSION
    return kwargs


def _bind_sources(admitted_sources, *callables):
    observed = source_files(_bind_sources, *callables)
    _receiver_require(
        type(admitted_sources) is dict
        and observed
        and observed.keys() <= admitted_sources.keys()
        and all(admitted_sources[path] == digest for path, digest in observed.items()),
        "every actual adapter callable needs an admitted source identity",
    )
    # Include the whole supplied operation closure, not just its selected callables.
    verify_sources(admitted_sources)
    return dict(admitted_sources)


def cleanup_owned_loras(receiver):
    """Remove only captured owned adapter objects, even after source drift."""
    owned = getattr(receiver, "_ap_owned_lora_installations", {})
    errors = []
    for identity, (manager, model, get_adapter, remove_adapter) in list(owned.items()):
        try:
            current = get_adapter(identity)
            if current is not None:
                _receiver_require(current is model, "refusing to remove a different owner's adapter")
                remove_adapter(identity)
            _receiver_require(get_adapter(identity) is None, "owned adapter remains after cleanup")
            del owned[identity]
        except BaseException as error:
            errors.append(error)
    if errors:
        raise RuntimeError("owned adapter cleanup failed: " + "; ".join(str(error) for error in errors))
    return {"owned_adapters_empty": not owned}


def receive_and_install_lora(
    receiver, engine, *, lora_int_id, lora_name, lora_config, max_received_bytes, admitted_sources
):
    """Clone the real reusable wire buffers, then use native adapter build/install.

    No base-weight loader or parameter-only shadow judge is used. The returned
    record describes actual wire bytes and installation, not application proof.
    """
    _receiver_require(enabled(), "LoRA receiver evidence must be explicitly enabled")
    _receiver_require(
        type(lora_int_id) is int and lora_int_id > 0 and type(lora_name) is str and lora_name,
        "positive adapter id and nonempty name required",
    )
    _receiver_require(
        type(max_received_bytes) is int and max_received_bytes > 0,
        "an explicit finite received-adapter byte limit is required",
    )
    # The optional inference backend belongs to the selected receiver, not the model factory.
    native = importlib.import_module("arctic_platform.inference.server.weight_sync.receiver")
    _build_lora_model = native._build_lora_model
    _install_lora_adapter = native._install_lora_adapter
    normalize_lora_peft_config = native.normalize_lora_peft_config

    runner = receiver.model_runner
    wrapper = getattr(runner, "lora_manager", None)
    manager = getattr(wrapper, "_adapter_manager", None)
    _receiver_require(manager is not None, "actual enabled native LoRA adapter manager required")
    _receiver_require(not receiver.model_config.quantization, "quantized adapter evidence is not implemented")
    manager_calls = {
        name: getattr(manager, name) for name in ("add_adapter", "activate_adapter", "get_adapter", "remove_adapter")
    }
    origins = _bind_sources(
        admitted_sources,
        type(engine).receive_weights,
        _build_lora_model,
        _install_lora_adapter,
        normalize_lora_peft_config,
        *manager_calls.values(),
    )
    config = normalize_lora_peft_config(lora_config, location="lora_config")
    _receiver_require(
        manager.get_adapter(lora_int_id) is None,
        "existing adapter id cannot be replaced without owned eviction evidence",
    )
    tensors, received = {}, {}
    size = 0
    for name, value in engine.receive_weights():
        _receiver_require(
            type(name) is str and name and name not in received, "adapter wire names must be nonempty and unique"
        )
        _receiver_require(".lora_A." in name or ".lora_B." in name, "base-model weights are not adapter weights")
        size += value.numel() * value.element_size()
        _receiver_require(size <= max_received_bytes, "adapter bytes exceed the explicit receive limit")
        received[name] = tensor_manifest(value)
        clone = value.detach().to(device="cpu", copy=True).contiguous()
        _receiver_require(tensor_manifest(clone) == received[name], "retained adapter copy differs from wire bytes")
        tensors[name] = clone
        verify_sources(origins)
    _receiver_require(received, "empty adapter transfer cannot install a policy")
    model = _build_lora_model(
        lora_int_id=lora_int_id,
        tensors=tensors,
        lora_config=config,
        device="cpu",
        dtype=receiver.model_config.dtype,
        model=runner.model,
    )
    verify_sources(origins)
    owned = getattr(receiver, "_ap_owned_lora_installations", None)
    if owned is None:
        owned = receiver._ap_owned_lora_installations = {}
    _receiver_require(lora_int_id not in owned, "an owned adapter id is already reserved")
    owned[lora_int_id] = (manager, model, manager_calls["get_adapter"], manager_calls["remove_adapter"])
    try:
        _receiver_require(
            all(getattr(manager, name) == method for name, method in manager_calls.items()),
            "selected adapter manager callables changed before installation",
        )
        added, activated = _install_lora_adapter(manager, model, lora_int_id)
        _receiver_require(
            all(getattr(manager, name) == method for name, method in manager_calls.items()),
            "selected adapter manager callables changed during installation",
        )
        _receiver_require(
            added is True and activated is True and manager_calls["get_adapter"](lora_int_id) is model,
            "actual manager must retain the exact built and activated adapter",
        )
        verify_sources(origins)
    except BaseException as error:
        try:
            cleanup_owned_loras(receiver)
        except BaseException as cleanup_error:
            error.add_note("owned adapter cleanup also failed: " + repr(cleanup_error))
        raise
    return {
        "schema": "explicit-external-lora-installation-v1",
        "status": "adapter_installed",
        "weight_format": "lora",
        "lora_int_id": lora_int_id,
        "lora_name": lora_name,
        "lora_added": added,
        "lora_activated": activated,
        "is_3d_lora_weight": bool(getattr(model, "is_3d_lora_weight", False)),
        "received": received,
        "received_bytes": size,
        "evidence_sources": origins,
        "adapter_storage_verified": False,
        "adapter_application_verified": False,
        "full_parameter_coverage": False,
        "capacity_admitted": False,
        "numerical": None,
    }


def serving_lora_request(installation, *, admitted_sources):
    """Use the installed id without allowing vLLM to reload a placeholder path."""
    _receiver_require(enabled(), "LoRA serving evidence must be explicitly enabled")
    _receiver_require(
        installation.get("schema") == "explicit-external-lora-installation-v1"
        and installation.get("status") == "adapter_installed"
        and installation.get("lora_added") is True
        and installation.get("lora_activated") is True
        and type(installation.get("lora_int_id")) is int
        and installation["lora_int_id"] > 0
        and type(installation.get("lora_name")) is str
        and installation["lora_name"],
        "a genuine installation record is required before selecting the adapter",
    )
    from vllm.lora.request import LoRARequest

    origins = _bind_sources(admitted_sources, LoRARequest)
    _receiver_require(
        installation["evidence_sources"].items() <= origins.items(),
        "serving must retain every installation source identity",
    )
    request = LoRARequest(
        lora_name=installation["lora_name"],
        lora_int_id=installation["lora_int_id"],
        lora_path="arctic-nccl://" + installation["lora_name"],
        load_inplace=False,
        is_3d_lora_weight=installation["is_3d_lora_weight"],
    )
    _receiver_require(
        request.lora_int_id == installation["lora_int_id"]
        and request.lora_name == installation["lora_name"]
        and request.load_inplace is False,
        "the actual serving request must select the installed adapter without disk reload",
    )
    verify_sources(origins)
    return request


@functools.lru_cache(maxsize=1)
def receiver_extension():
    """Keep the external receiver superclass and its pairwise NCCL engine."""
    from arctic_inference.server.weight_sync.receiver import WeightSyncExtension

    class LoRAWeightSyncExtension(WeightSyncExtension):
        def close_weight_sync(self):
            error = None
            try:
                result = cleanup_owned_loras(self)
            except BaseException as cleanup_error:
                error = cleanup_error
            try:
                engine_result = super().close_weight_sync()
            except BaseException as engine_error:
                if error is None:
                    raise
                error.add_note("external engine cleanup also failed: " + repr(engine_error))
            if error is not None:
                raise error
            return {**result, "external_engine_cleanup": engine_result}

        def sync_lora_weights(
            self,
            master_addr,
            master_port,
            rank_offset,
            world_size,
            *,
            lora_int_id,
            lora_name,
            lora_config,
            max_received_bytes,
            admitted_sources,
            bucket_size=256 * 1024 * 1024,
            engine_only=False,
        ):
            _receiver_require(enabled(), "the explicit LoRA extension must be enabled on the actual TP rank")
            _receiver_require(
                type(bucket_size) is int and bucket_size > 0, "positive native NCCL bucket size required"
            )
            origins = _bind_sources(admitted_sources, type(self)._get_or_create_engine, receive_and_install_lora)
            engine = self._get_or_create_engine(
                master_addr,
                master_port,
                bucket_size,
                False,
                engine_attr="_ws_engine",
                key_attr="_ws_engine_key",
            )
            verify_sources(origins)
            if engine_only:
                return {
                    "status": "engine_ready",
                    "weight_format": "lora",
                    "rank": 1,
                    "world_size": 2,
                    "adapter_storage_verified": False,
                    "capacity_admitted": False,
                }
            return receive_and_install_lora(
                self,
                engine,
                lora_int_id=lora_int_id,
                lora_name=lora_name,
                lora_config=lora_config,
                max_received_bytes=max_received_bytes,
                admitted_sources=origins,
            )

    return LoRAWeightSyncExtension


# Sampler


def _sampler_request(request):
    _receiver_require(
        type(request) is dict
        and set(request)
        == {
            "lora_int_id",
            "lora_name",
            "lora_config",
            "max_received_bytes",
            "admitted_sources",
        },
        "exact explicit adapter worker request required",
    )
    _receiver_require(
        type(request["lora_int_id"]) is int
        and request["lora_int_id"] > 0
        and type(request["lora_name"]) is str
        and request["lora_name"],
        "positive owned adapter identity required",
    )
    _receiver_require(
        type(request["lora_config"]) is dict
        and type(request["lora_config"].get("r")) is int
        and request["lora_config"]["r"] > 0,
        "actual positive configured adapter rank required",
    )
    _receiver_require(
        type(request["max_received_bytes"]) is int and request["max_received_bytes"] > 0,
        "finite explicit adapter receive budget required",
    )
    _receiver_require(
        type(request["admitted_sources"]) is dict and request["admitted_sources"],
        "actual installed source identities must be supplied",
    )
    return deepcopy(request)


def make_lora_replica_pool(pool_class, request):
    """Use the external pool's public worker_cls without changing its scheduler."""
    _receiver_require(enabled(), "LoRA sampler evidence must be explicitly enabled")
    request = _sampler_request(request)

    _bind_sources(
        request["admitted_sources"],
        pool_class,
        LoRASamplerEvidenceWorker,
        SamplerEvidenceWorker,
        receive_and_install_lora,
    )
    import ray

    class OwnedLoRAWorker(LoRASamplerEvidenceWorker):
        def __init__(self):
            super().__init__(request)

    actor = ray.remote(runtime_env={"env_vars": {VERIFY_ENV: "1"}})(OwnedLoRAWorker)
    return functools.partial(pool_class, worker_cls=actor)


class LoRASamplerEvidenceWorker(SamplerEvidenceWorker):
    """Reuse lifecycle/metrics, with explicit adapter install and request selection.

    The first transfer is single-use. A second update needs a separate owned
    eviction protocol; it cannot silently replace the installed adapter.
    """

    def __init__(self, request):
        super().__init__()
        self._request = _sampler_request(request)
        self._installation = None
        self._lora_request = None
        self._sync_attempted = False

    async def initialize(self, engine_kwargs, extra_env=None):
        _receiver_require(self.llm is None, "LoRA sampler is already initialized")
        if extra_env:
            os.environ.update(extra_env)
        _receiver_require(enabled(), "LoRA opt-in must reach the actual sampler worker")
        verify_sources(self._request["admitted_sources"])
        import vllm.plugins
        from arctic_inference.envs import arctic_inference_effective_enabled
        from arctic_inference.server.metrics import RingStatLogger

        vllm.plugins.load_general_plugins()
        from vllm.v1.engine.async_llm import AsyncLLM

        if arctic_inference_effective_enabled():
            from arctic_inference.vllm.args import ArcticAsyncEngineArgs as EngineArgs
        else:
            from vllm.engine.arg_utils import AsyncEngineArgs as EngineArgs

        kwargs = engine_kwargs_for_lora(engine_kwargs, rank=self._request["lora_config"]["r"])
        if os.environ.get("VLLM_BATCH_INVARIANT") == "1":
            kwargs.setdefault("attention_config", {"backend": "FLASH_ATTN"})
        args = EngineArgs(**kwargs)
        _receiver_require(
            type(args.tensor_parallel_size) is int and args.tensor_parallel_size == 8,
            "this original LoRA sampler requires the actual SAMPLE8 engine",
        )
        self._tp = args.tensor_parallel_size
        self._origins = _bind_sources(
            self._request["admitted_sources"],
            type(self),
            SamplerEvidenceWorker,
            EngineArgs,
            args.create_engine_config,
            AsyncLLM.from_vllm_config,
            AsyncLLM.collective_rpc,
            AsyncLLM.generate,
            AsyncLLM.abort,
            AsyncLLM.shutdown,
            serving_lora_request,
        )

        maybe_admit_sampler_construction(self, args, AsyncLLM.from_vllm_config, engine_kwargs)
        self.llm = AsyncLLM.from_vllm_config(args.create_engine_config(), stat_loggers=[RingStatLogger])
        try:
            maybe_verify_sampler_construction(self)
            verify_sources(self._origins)
        except BaseException as error:
            try:
                self.shutdown()
            except BaseException as cleanup_error:
                error.add_note("native LoRA engine shutdown also failed: " + repr(cleanup_error))
            raise
        self.state = "ready"

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
            _receiver_require(
                self.llm is not None and self.state in {"ready", "weights_ready"}, "actual awake LoRA sampler required"
            )
            verify_sources(self._origins)
            _receiver_require(
                not direct_mode and not reverse and rank_offset == 1 and world_size == 2,
                "only the retained external pairwise adapter transport is supported",
            )
            _receiver_require(not self._sync_attempted, "adapter transfer is single-use; no implicit replacement")
            if not engine_only:
                self._sync_attempted = True
            options = {**deepcopy(self._request), "bucket_size": bucket_size, "engine_only": engine_only}
            results = await self.llm.collective_rpc(
                "sync_lora_weights",
                args=(master_addr, master_port, rank_offset, world_size),
                kwargs=options,
            )
            verify_sources(self._origins)
            _receiver_require(
                type(results) is list and len(results) == self._tp,
                "actual adapter results from every selected TP rank required",
            )
            if engine_only:
                _receiver_require(
                    all(
                        type(row) is dict
                        and row.get("status") == "engine_ready"
                        and row.get("weight_format") == "lora"
                        for row in results
                    ),
                    "an adapter receiver engine failed initialization",
                )
                return {"status": "engine_ready", "weight_format": "lora", "rank_results": results}
            for row in results:
                _receiver_require(
                    type(row) is dict
                    and row.get("schema") == "explicit-external-lora-installation-v1"
                    and row.get("status") == "adapter_installed"
                    and row.get("weight_format") == "lora"
                    and row.get("lora_int_id") == self._request["lora_int_id"]
                    and row.get("lora_name") == self._request["lora_name"]
                    and row.get("lora_added") is True
                    and row.get("lora_activated") is True
                    and type(row.get("received")) is dict
                    and row["received"]
                    and row.get("adapter_storage_verified") is False
                    and row.get("adapter_application_verified") is False
                    and row.get("full_parameter_coverage") is False
                    and row.get("capacity_admitted") is False,
                    "a TP rank did not retain genuine adapter installation/nonadmission evidence",
                )
                _receiver_require(
                    row["evidence_sources"].items() <= self._origins.items(),
                    "TP rank installation sources differ from the admitted operation",
                )
                _receiver_require(
                    row["received"] == results[0]["received"]
                    and row["is_3d_lora_weight"] == results[0]["is_3d_lora_weight"],
                    "actual replicated adapter bytes or serving layout differ across TP ranks",
                )
            request = serving_lora_request(results[0], admitted_sources=self._origins)
            self._installation = deepcopy(results)
            self._lora_request = request
            return {
                "status": "done",
                "weight_format": "lora",
                "rank_results": results,
                "all_rank_adapter_installation": results,
                "adapter_storage_verified": False,
                "adapter_application_verified": False,
                "full_parameter_coverage": False,
                "capacity_admitted": False,
                "numerical": None,
            }

    async def generate(self, prompt, sampling_params):
        _receiver_require(
            not any(key.startswith("dss_") for key in sampling_params),
            "native router replay remains unsupported by this explicit adapter worker",
        )
        from vllm import SamplingParams

        async with self._lock:
            self._ready()
            _receiver_require(
                self._installation is not None and self._lora_request is not None,
                "actual installed adapter required before LoRA serving",
            )
            origins = _bind_sources(self._origins, SamplingParams, self.llm.generate, self.llm.abort)
            params = SamplingParams(**dict(sampling_params))
            request_id = str(uuid4())
            prompt_input = {"prompt_token_ids": prompt} if isinstance(prompt, list) else prompt
            final = None
            try:
                async for output in self.llm.generate(
                    prompt_input,
                    params,
                    request_id=request_id,
                    lora_request=self._lora_request,
                ):
                    final = output
            except BaseException as error:
                try:
                    await self.llm.abort(request_id)
                except BaseException as abort_error:
                    error.add_note("native LoRA request abort also failed: " + repr(abort_error))
                raise
            verify_sources(origins)
            _receiver_require(final is not None and final.outputs, "empty native LoRA generation output")
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

    async def close_weight_sync(self):
        try:
            return await super().close_weight_sync()
        finally:
            self._installation = None
            self._lora_request = None

    def shutdown(self):
        try:
            super().shutdown()
        finally:
            self._installation = None
            self._lora_request = None


# Selected Receiver Callbacks

_LORA_MANAGER_METHODS = ("add_adapter", "activate_adapter", "_set_adapter", "get_adapter", "remove_adapter")
_LORA_LAYER_METHODS = ("set_lora", "forward")
_SELECTED_PREPARATION_KEYS = {
    "distributions",
    "output",
    "max_source_bytes",
    "max_layers",
    "max_total_source_bytes",
    "observer",
    "builder",
    "installer",
    "normalize_config",
    "native_model_class",
    "parse_name",
    "remap_key",
    "served_prefix_for_model",
}


def _lora_source_identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _stable_lora_callback_source(path, cap):
    path = Path(path)
    _receiver_require(
        path.is_absolute() and path.resolve(strict=True) == path and path.is_file() and (not path.is_symlink()),
        "canonical regular callback source required",
    )
    before = path.stat()
    _receiver_require(0 < before.st_size <= cap, "callback source byte cap exceeded")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read(cap + 1)
        after = os.fstat(stream.fileno())
    _receiver_require(
        _lora_source_identity(before)
        == _lora_source_identity(opened)
        == _lora_source_identity(after)
        == _lora_source_identity(path.stat())
        and len(raw) == before.st_size,
        "callback source changed during read",
    )
    return raw


def _lora_distribution_origin(filename, distributions, cap):
    """Use exact caller-selected distributions, never a package tree search."""
    filename = Path(filename)
    matches = []
    for distribution in distributions:
        root = Path(distribution.locate_file("")).resolve(strict=True)
        if not filename.is_relative_to(root):
            continue
        member = filename.relative_to(root).as_posix()
        for row in distribution.files or ():
            if str(row) != member:
                continue
            _receiver_require(
                row.hash is not None
                and row.hash.mode == "sha256"
                and (type(row.size) is int)
                and (0 < row.size <= cap),
                "callback requires a bounded SHA256 distribution RECORD entry",
            )
            digest = base64.urlsafe_b64decode(row.hash.value + "=" * (-len(row.hash.value) % 4))
            _receiver_require(len(digest) == 32, "callback RECORD digest must be SHA256")
            raw = _stable_lora_callback_source(filename, cap)
            _receiver_require(
                len(raw) == row.size and hashlib.sha256(raw).digest() == digest,
                "actual callback source differs from its distribution RECORD",
            )
            matches.append(
                {
                    "distribution": distribution.metadata["Name"],
                    "version": distribution.version,
                    "root": str(root),
                    "member": member,
                    "path": str(filename),
                    "sha256": digest.hex(),
                    "size_bytes": row.size,
                }
            )
    _receiver_require(len(matches) == 1, "one exact selected distribution must own each callback source")
    return matches[0]


def _lora_callback_origin(method, owner, distributions, cap):
    _receiver_require(
        inspect.ismethod(method) and method.__self__ is owner and (not isinstance(owner, type)),
        "callback must be bound to the actual selected object",
    )
    function = method.__func__
    _receiver_require(inspect.isfunction(function), "native Python callback body required")
    filename = inspect.getsourcefile(function)
    _receiver_require(
        filename is not None and function.__code__.co_filename == filename,
        "actual callback code/source origin differs",
    )
    return {
        "module": function.__module__,
        "qualname": function.__qualname__,
        "code_sha256": hashlib.sha256(marshal.dumps(function.__code__)).hexdigest(),
        "loaded_code_equivalence_to_source_proven": False,
        "source": _lora_distribution_origin(filename, distributions, cap),
    }


def _lora_adapter_layers(model):
    """Enumerate only the actual model's registered native adapter layers."""
    named = list(model.named_modules())
    _receiver_require(
        named and len({name for name, _ in named}) == len(named), "unique actual model module names required"
    )
    selected = {}
    for name, layer in named:
        descriptor = inspect.getattr_static(type(layer), "set_lora", None)
        if descriptor is None:
            continue
        _receiver_require(
            inspect.isfunction(descriptor), "selected adapter setter cannot be a property or dynamic descriptor"
        )
        _receiver_require(type(name) is str and name, "selected adapter layer requires a nonempty actual module name")
        setter = getattr(layer, "set_lora")
        _receiver_require(
            inspect.ismethod(setter) and setter.__self__ is layer,
            "selected adapter layer setter must be bound to its actual module",
        )
        selected[name] = layer
    _receiver_require(
        selected and len({id(layer) for layer in selected.values()}) == len(selected),
        "nonempty unaliased actual adapter layer inventory required",
    )
    return selected


@dataclass
class SelectedLoRACallbacks:
    receiver: object
    runner: object
    model: object
    wrapper: object
    manager: object
    callbacks: dict
    layers: dict
    layer_callbacks: dict
    distributions: tuple
    max_source_bytes: int
    sources: dict
    evidence: dict

    def guard(self):
        verify_sources(self.sources)
        _receiver_require(
            self.receiver.model_runner is self.runner
            and self.runner.model is self.model
            and (self.runner.lora_manager is self.wrapper)
            and (self.wrapper._adapter_manager is self.manager),
            "actual receiver/manager identity changed",
        )
        _receiver_require(
            all((getattr(self.manager, name) == callback for name, callback in self.callbacks.items())),
            "selected manager callback changed",
        )
        current = _lora_adapter_layers(self.model)
        _receiver_require(
            current.keys() == self.layers.keys()
            and all((current[name] is layer for name, layer in self.layers.items())),
            "selected adapter layer inventory changed",
        )
        for name, methods in self.layer_callbacks.items():
            _receiver_require(
                all((getattr(self.layers[name], key) == method for key, method in methods.items())),
                "selected layer callback changed",
            )
        actual = self.origins()
        _receiver_require(actual == self.evidence["callback_origins"], "selected callback code/RECORD origin changed")
        verify_sources(self.sources)
        return actual

    def origins(self):
        return {
            "manager": {
                name: _lora_callback_origin(method, self.manager, self.distributions, self.max_source_bytes)
                for name, method in self.callbacks.items()
            },
            "layers": {
                name: {
                    key: _lora_callback_origin(method, self.layers[name], self.distributions, self.max_source_bytes)
                    for key, method in methods.items()
                }
                for name, methods in self.layer_callbacks.items()
            },
        }


def capture_selected_lora_callbacks(receiver, *, distributions, admitted_sources, max_source_bytes, max_layers):
    """Capture loaded objects only; never invoke a factory, install or forward."""
    _receiver_require(enabled(), "selected LoRA callback capture must be explicitly enabled")
    _receiver_require(
        type(max_source_bytes) is int and max_source_bytes > 0 and (type(max_layers) is int) and (max_layers > 0),
        "positive explicit capture bounds required",
    )
    _receiver_require(type(distributions) is tuple and distributions, "actual selected distributions required")
    verify_sources(admitted_sources)
    own_sources = source_files(capture_selected_lora_callbacks, SelectedLoRACallbacks)
    _receiver_require(
        own_sources.keys() <= admitted_sources.keys()
        and all((admitted_sources[path] == digest for path, digest in own_sources.items())),
        "actual capture helper source must be admitted",
    )
    runner = receiver.model_runner
    model = runner.model
    wrapper = runner.lora_manager
    manager = wrapper._adapter_manager
    _receiver_require(model is not None and manager is not None, "already-loaded actual native receiver required")
    callbacks = {name: getattr(manager, name) for name in _LORA_MANAGER_METHODS}
    layers = _lora_adapter_layers(model)
    _receiver_require(len(layers) <= max_layers, "selected layer count exceeds explicit capture bound")
    layer_callbacks = {
        name: {key: getattr(layer, key) for key in _LORA_LAYER_METHODS} for name, layer in layers.items()
    }
    captured = SelectedLoRACallbacks(
        receiver,
        runner,
        model,
        wrapper,
        manager,
        callbacks,
        layers,
        layer_callbacks,
        distributions,
        max_source_bytes,
        dict(admitted_sources),
        {},
    )
    origins = captured.origins()
    native = [
        *origins["manager"].values(),
        *(method for methods in origins["layers"].values() for method in methods.values()),
    ]
    for row in native:
        pointer = row["source"]
        previous = captured.sources.setdefault(pointer["path"], pointer["sha256"])
        _receiver_require(previous == pointer["sha256"], "selected origin conflicts with caller-admitted source")
    captured.evidence = {
        "schema": "job3-selected-lora-callback-source-inventory-v1",
        "callback_origins": origins,
        "evidence_sources": dict(captured.sources),
        "selected_layer_names": list(layers),
        "manager_layer_correspondence_verified": False,
        "transitive_kernel_source_closure_complete": False,
        "loaded_code_equivalence_to_source_proven": False,
        "model_constructed": False,
        "callbacks_executed": False,
        "runtime_admitted": False,
        "adapter_storage_verified": False,
        "adapter_application_verified": False,
        "capacity_admitted": False,
        "numerical": None,
    }
    captured.guard()
    return captured


def _read_lora_callback_source(path, *, sha256, size_bytes):
    path = Path(path)
    _receiver_require(
        path.is_absolute() and path.resolve(strict=True) == path and path.is_file() and (not path.is_symlink()),
        "canonical selected source required",
    )
    before = path.stat()
    _receiver_require(before.st_size == size_bytes, "selected source size changed")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read(size_bytes + 1)
        after = os.fstat(stream.fileno())
    _receiver_require(
        _lora_source_identity(before)
        == _lora_source_identity(opened)
        == _lora_source_identity(after)
        == _lora_source_identity(path.stat())
        and len(raw) == size_bytes
        and (hashlib.sha256(raw).hexdigest() == sha256),
        "selected callback source changed during read",
    )
    return raw


def publish_lora_callback_sources(ticket, *, output, max_total_source_bytes):
    """Retain only source bodies selected by the actual guarded callback ticket.

    The receipt is input for deriving slot/kernel semantics, never evidence
    that the manager selected those slots or that any adapter was applied.
    """
    _receiver_require(
        type(max_total_source_bytes) is int and max_total_source_bytes > 0,
        "positive total publication byte cap required",
    )
    own = source_files(publish_lora_callback_sources)
    _receiver_require(
        own.keys() <= ticket.sources.keys() and all((ticket.sources[path] == digest for path, digest in own.items())),
        "actual publisher source must be in the captured operation closure",
    )
    origins = ticket.guard()
    _receiver_require(
        ticket.evidence["schema"] == "job3-selected-lora-callback-source-inventory-v1",
        "actual selected callback capture schema required",
    )
    output = Path(output)
    _receiver_require(
        output.is_absolute()
        and output.resolve(strict=True) == output
        and output.is_dir()
        and (not output.is_symlink())
        and (stat.S_IMODE(output.stat().st_mode) == 448),
        "canonical private caller-owned publication directory required",
    )
    _receiver_require(not any(output.iterdir()), "publication must not replace an earlier owner's source packet")
    native = [
        *origins["manager"].values(),
        *(row for methods in origins["layers"].values() for row in methods.values()),
    ]
    sources = {}
    for row in native:
        selected = row["source"]
        existing = sources.setdefault(selected["path"], selected)
        _receiver_require(existing == selected, "selected callbacks disagree on one source file identity")
    total = sum((row["size_bytes"] for row in sources.values()))
    _receiver_require(0 < total <= max_total_source_bytes, "selected callback packet byte cap exceeded")
    snapshots = []
    for filename, row in sorted(sources.items()):
        raw = _read_lora_callback_source(filename, sha256=row["sha256"], size_bytes=row["size_bytes"])
        destination = output / (row["sha256"] + ".py")
        _receiver_require(
            not destination.exists(), "distinct selected sources must have distinct private snapshot names"
        )
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 384)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _receiver_require(
            _read_lora_callback_source(destination, sha256=row["sha256"], size_bytes=row["size_bytes"]) == raw,
            "private selected source snapshot changed",
        )
        snapshots.append(
            {
                "installed": dict(row),
                "snapshot": {"path": str(destination), "sha256": row["sha256"], "size_bytes": row["size_bytes"]},
            }
        )
    _receiver_require(ticket.guard() == origins, "selected callback inventory changed during publication")
    receipt = {
        "schema": "job3-selected-lora-callback-private-source-packet-v1",
        "callback_origins": origins,
        "selected_layer_names": list(ticket.layers),
        "sources": snapshots,
        "total_source_bytes": total,
        "source_before_after_equal": True,
        "evidence_sources": dict(ticket.sources),
        "manager_layer_correspondence_verified": False,
        "transitive_kernel_source_closure_complete": False,
        "loaded_code_equivalence_to_source_proven": False,
        "callbacks_executed": False,
        "model_constructed": False,
        "runtime_admitted": False,
        "adapter_storage_verified": False,
        "adapter_application_verified": False,
        "capacity_admitted": False,
        "numerical": None,
    }
    destination = output / "callback-source-packet.json"
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 384)
    with os.fdopen(fd, "w") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "path": str(destination),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "size_bytes": destination.stat().st_size,
    }


def _verify_lora_build_evidence(operation, *, lora_int_id, received, size, sources):
    _receiver_require(
        type(operation) is ObservedLoRAOperation
        and operation.evidence.get("schema") == "observed-lora-native-build-binding-v1"
        and (operation.evidence.get("lora_int_id") == lora_int_id)
        and (operation.evidence.get("original_received") == received)
        and (operation.evidence.get("received_bytes") == size)
        and (operation.evidence.get("all_received_adapter_factors_bound") is True)
        and (operation.evidence.get("built_factor_byte_correspondence_verified") is True)
        and (operation.evidence.get("matched_retained_model_parser_mapper_sources") is True)
        and (operation.evidence.get("native_builder_calls") == 1)
        and (operation.evidence.get("evidence_sources") == sources),
        "source-matched one-build evidence must bind every actual received factor",
    )


@contextmanager
def owned_received_lora(
    receiver,
    engine,
    *,
    lora_int_id,
    lora_name,
    lora_config,
    max_received_bytes,
    admitted_sources,
    observer,
    builder,
    installer,
    normalize_config,
    native_model_class,
    parse_name,
    remap_key,
    served_prefix_for_model,
    teardown,
):
    with owned_receiver_binding(
        receiver, adapter_id=lora_int_id, admitted_sources=admitted_sources, teardown=teardown
    ) as binding:
        _receiver_require(type(lora_name) is str and lora_name, "nonempty owned adapter name required")
        _receiver_require(
            type(max_received_bytes) is int and max_received_bytes > 0, "explicit positive received byte cap required"
        )
        _receiver_require(not receiver.model_config.quantization, "quantized adapter operation is not supported")
        sources = _bind_sources(
            binding.sources,
            engine.receive_weights,
            observer,
            builder,
            installer,
            normalize_config,
            native_model_class,
            parse_name,
            remap_key,
            served_prefix_for_model,
        )
        config = normalize_config(lora_config, location="lora_config")
        tensors, received, size = ({}, {}, 0)
        for name, value in engine.receive_weights():
            binding.guard()
            _receiver_require(
                type(name) is str and name and (name not in received), "wire tensor names must be nonempty and unique"
            )
            _receiver_require(
                name.endswith((".lora_A.weight", ".lora_B.weight")),
                "only original adapter A/B wire factors are supported",
            )
            size += value.numel() * value.element_size()
            _receiver_require(size <= max_received_bytes, "received adapter byte cap exceeded")
            received[name] = tensor_manifest(value)
            clone = value.detach().to(device="cpu", copy=True).contiguous()
            _receiver_require(
                tensor_manifest(clone) == received[name], "retained adapter copy differs from actual wire bytes"
            )
            tensors[name] = clone
            verify_sources(sources)
        _receiver_require(received, "empty adapter transfer cannot install a policy")
        binding.guard()
        operation = observer(
            builder,
            lora_int_id=lora_int_id,
            tensors=tensors,
            lora_config=config,
            device="cpu",
            dtype=receiver.model_config.dtype,
            model=binding.native_model,
            native_model_class=native_model_class,
            parse_name=parse_name,
            remap_key=remap_key,
            served_prefix_for_model=served_prefix_for_model,
            max_binding_bytes=max_received_bytes,
            admitted_sources=sources,
        )
        verify_sources(sources)
        binding.guard()
        _verify_lora_build_evidence(operation, lora_int_id=lora_int_id, received=received, size=size, sources=sources)
        installation = binding.install(operation.value, installer=installer)
        yield {
            "schema": "job3-owned-received-lora-operation-v2",
            "status": "adapter_installed",
            "weight_format": "lora",
            "lora_int_id": lora_int_id,
            "lora_name": lora_name,
            "received": deepcopy(received),
            "received_bytes": size,
            "wire_to_built": deepcopy(operation.evidence),
            "manager_installation": installation,
            "evidence_sources": sources,
            "adapter_storage_verified": False,
            "adapter_application_verified": False,
            "full_parameter_coverage": False,
            "capacity_admitted": False,
            "numerical": None,
        }
        binding.guard()
        verify_sources(sources)


@contextmanager
def owned_selected_received_lora(
    receiver,
    engine,
    *,
    distributions,
    output,
    max_source_bytes,
    max_layers,
    max_total_source_bytes,
    admitted_sources,
    teardown,
    **operation,
):
    attempted, primary = (False, None)

    def close_once():
        nonlocal attempted
        if not attempted:
            attempted = True
            teardown()

    try:
        _receiver_require(callable(teardown), "captured receiver teardown required")
        sources = _bind_sources(
            admitted_sources, close_once, teardown, capture_selected_lora_callbacks, publish_lora_callback_sources
        )
        ticket = capture_selected_lora_callbacks(
            receiver,
            distributions=distributions,
            admitted_sources=sources,
            max_source_bytes=max_source_bytes,
            max_layers=max_layers,
        )
        pointer = publish_lora_callback_sources(ticket, output=output, max_total_source_bytes=max_total_source_bytes)
        ticket.guard()
        with owned_received_lora(
            receiver, engine, admitted_sources=ticket.sources, teardown=close_once, **operation
        ) as result:
            ticket.guard()
            yield {
                **result,
                "schema": "job3-owned-selected-received-lora-operation-v2",
                "selected_callback_source_packet": pointer,
                "manager_layer_correspondence_verified": False,
                "transitive_kernel_source_closure_complete": False,
                "loaded_code_equivalence_to_source_proven": False,
                "runtime_admitted": False,
            }
            ticket.guard()
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            close_once()
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note("captured selected receiver teardown also failed: " + repr(error))


def make_selected_receiver_extension(base_extension, *, prepare, admitted_sources):
    """Extend the actual supplied external receiver, without constructing it.

    ``prepare(receiver)`` supplies exact selected builder/parser and private
    publication options. Native constructor/capacity admission and engine
    extension selection remain the useful caller's responsibility.
    """
    _receiver_require(enabled(), "selected LoRA RPC requires explicit evidence opt-in")
    _receiver_require(
        isinstance(base_extension, type) and callable(prepare),
        "actual external receiver class and explicit preparation callback required",
    )
    operation = owned_selected_received_lora
    get_engine = base_extension._get_or_create_engine
    close_engine = base_extension.close_weight_sync
    sources = _bind_sources(
        admitted_sources,
        make_selected_receiver_extension,
        prepare,
        base_extension,
        get_engine,
        close_engine,
        operation,
    )

    class SelectedLoRAReceiver(base_extension):

        def close_weight_sync(self):
            scope = getattr(self, "_ap_selected_rpc_scope", None)
            self._ap_selected_rpc_scope = None
            if scope is not None:
                scope.__exit__(None, None, None)
            else:
                self._close_captured_engine()
            return {
                "owned_adapter_scope_closed": True,
                "external_engine_cleanup_attempted": True,
                "capacity_admitted": False,
                "numerical": None,
            }

        def _close_captured_engine(self):
            if not getattr(self, "_ap_selected_engine_close_attempted", False):
                self._ap_selected_engine_close_attempted = True
                return close_engine(self)
            return None

        def sync_lora_weights(
            self,
            master_addr,
            master_port,
            rank_offset,
            world_size,
            *,
            lora_int_id,
            lora_name,
            lora_config,
            max_received_bytes,
            admitted_sources,
            bucket_size=256 * 1024 * 1024,
            engine_only=False,
        ):
            _receiver_require(enabled(), "selected LoRA RPC requires explicit evidence opt-in")
            _receiver_require(
                not getattr(self, "_ap_selected_engine_close_attempted", False),
                "closed selected receiver cannot reopen its transport",
            )
            _receiver_require(
                not getattr(self, "_ap_selected_sync_attempted", False), "selected adapter transfer is single-use"
            )
            _receiver_require(
                rank_offset == 1
                and world_size == 2
                and (type(bucket_size) is int)
                and (bucket_size > 0)
                and (type(engine_only) is bool),
                "retained external pairwise transport and positive bucket required",
            )
            try:
                _receiver_require(admitted_sources == sources, "RPC sources differ from the frozen caller closure")
                verify_sources(sources)
                _receiver_require(
                    type(self)._get_or_create_engine is get_engine, "selected native engine factory changed"
                )
                engine = get_engine(
                    self,
                    master_addr,
                    master_port,
                    bucket_size,
                    False,
                    engine_attr="_ws_engine",
                    key_attr="_ws_engine_key",
                )
                verify_sources(sources)
                if engine_only:
                    return {
                        "status": "engine_ready",
                        "weight_format": "lora",
                        "rank": 1,
                        "world_size": 2,
                        "adapter_storage_verified": False,
                        "capacity_admitted": False,
                    }
                self._ap_selected_sync_attempted = True
                options = prepare(self)
                _receiver_require(
                    type(options) is dict and set(options) == _SELECTED_PREPARATION_KEYS,
                    "exact selected receiver preparation options required",
                )
                verify_sources(sources)
                scope = operation(
                    self,
                    engine,
                    **options,
                    lora_int_id=lora_int_id,
                    lora_name=lora_name,
                    lora_config=deepcopy(lora_config),
                    max_received_bytes=max_received_bytes,
                    admitted_sources=sources,
                    teardown=self._close_captured_engine,
                )
                result = scope.__enter__()
                self._ap_selected_rpc_scope = scope
                manager = self.model_runner.lora_manager._adapter_manager
                adapter = manager.get_adapter(lora_int_id)
                _receiver_require(adapter is not None, "selected scoped adapter disappeared before RPC return")
                verify_sources(sources)
                return {
                    **deepcopy(result),
                    "schema": "explicit-external-lora-installation-v1",
                    "lora_added": result["manager_installation"]["added"],
                    "lora_activated": result["manager_installation"]["activated"],
                    "is_3d_lora_weight": bool(getattr(adapter, "is_3d_lora_weight", False)),
                    "selected_scope_retained_for_serving": True,
                    "runtime_admitted": False,
                }
            except BaseException as error:
                scope = getattr(self, "_ap_selected_rpc_scope", None)
                self._ap_selected_rpc_scope = None
                try:
                    if scope is None:
                        self._close_captured_engine()
                    else:
                        scope.__exit__(type(error), error, error.__traceback__)
                except BaseException as cleanup_error:
                    if cleanup_error is not error:
                        error.add_note("selected RPC cleanup also failed: " + repr(cleanup_error))
                raise

    return SelectedLoRAReceiver


# Selected

PATH_ENV = "ARCTIC_PLATFORM_SELECTED_LORA_BOOTSTRAP"
SHA_ENV = "ARCTIC_PLATFORM_SELECTED_LORA_BOOTSTRAP_SHA256"
SELECTED_LORA_EXTENSION = "arctic_platform.debug.lora.SelectedLoRAWeightSyncExtension"


def _bytes(path):
    path = Path(path)
    _receiver_require(
        path.is_absolute() and path.is_file() and not path.is_symlink() and path.resolve(strict=True) == path,
        "canonical regular bootstrap/source path required",
    )

    def identity(info):
        return (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    before = path.stat()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read()
        closed = os.fstat(stream.fileno())
    _receiver_require(
        identity(before) == identity(opened) == identity(closed) == identity(path.stat()),
        "bootstrap/source changed while reading",
    )
    return raw


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _receiver_require(key not in result, "duplicate bootstrap JSON key")
        result[key] = value
    return result


def bootstrap_descriptor(pointer):
    """Read caller-staged bytes, not a live singleton or an inferred import path."""
    _receiver_require(
        type(pointer) is dict
        and set(pointer) == {"path", "sha256"}
        and type(pointer["sha256"]) is str
        and re.fullmatch("[0-9a-f]{64}", pointer["sha256"]) is not None,
        "exact bootstrap path/digest required",
    )
    raw = _bytes(pointer["path"])
    _receiver_require(hashlib.sha256(raw).hexdigest() == pointer["sha256"], "bootstrap digest changed")
    body = json.loads(raw, object_pairs_hook=_unique_object)
    _receiver_require(
        type(body) is dict
        and set(body) == {"schema", "receiver", "prepare", "admitted_sources"}
        and body["schema"] == "job3-selected-lora-bootstrap-v1",
        "exact selected bootstrap schema required",
    )
    sources = body["admitted_sources"]
    _receiver_require(
        type(sources) is dict and sources,
        "actual nonempty bootstrap source closure required",
    )
    verify_sources(sources)
    observed = source_files(bootstrap_descriptor, LoRASamplerEvidenceWorker, engine_kwargs_for_lora)
    _receiver_require(
        observed.items() <= sources.items(),
        "bootstrap, worker and scoped receiver sources must be admitted",
    )
    for key in ("receiver", "prepare"):
        item = body[key]
        _receiver_require(
            type(item) is dict
            and set(item) == {"module", "name", "source"}
            and type(item["module"]) is str
            and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", item["module"]) is not None
            and type(item["name"]) is str
            and re.fullmatch(r"[A-Za-z_]\w*", item["name"]) is not None
            and type(item["source"]) is str
            and item["source"] in sources,
            "explicit top-level selected symbol and admitted source required",
        )
    _receiver_require(_bytes(pointer["path"]) == raw, "bootstrap changed during source verification")
    return body


def _symbol(item, sources):
    verify_sources(sources)
    module = importlib.import_module(item["module"])
    _receiver_require(
        inspect.getsourcefile(module) == item["source"],
        "selected module resolved to a different source",
    )
    # Avoid module __getattr__ selecting an unrecorded replacement.
    value = vars(module).get(item["name"])
    _receiver_require(
        value is not None and inspect.getsourcefile(value) == item["source"],
        "selected symbol source changed",
    )
    verify_sources(sources)
    return value


def selected_receiver_extension(pointer):
    """Resolve the supplied native class without constructing a model/receiver."""
    _receiver_require(enabled(), "selected receiver bootstrap requires explicit opt-in")
    body = bootstrap_descriptor(pointer)
    pointer = dict(pointer)
    sources = body["admitted_sources"]
    receiver = _symbol(body["receiver"], sources)
    prepare = _symbol(body["prepare"], sources)
    _receiver_require(
        isinstance(receiver, type) and inspect.isfunction(prepare),
        "actual receiver class and function required",
    )
    scoped = make_selected_receiver_extension(receiver, prepare=prepare, admitted_sources=sources)

    class SelectedLoRAWeightSyncExtension(scoped):
        def sync_lora_weights(self, *args, **kwargs):
            try:
                _receiver_require(
                    bootstrap_descriptor(pointer) == body,
                    "selected bootstrap changed before RPC",
                )
            except BaseException as error:
                try:
                    self.close_weight_sync()
                except BaseException as cleanup_error:
                    error.add_note("selected bootstrap cleanup also failed: " + repr(cleanup_error))
                raise
            # The scoped receiver owns its failures, including non-destructive
            # rejection of a duplicate transfer while its first scope is live.
            result = super().sync_lora_weights(*args, **kwargs)
            try:
                _receiver_require(
                    bootstrap_descriptor(pointer) == body,
                    "selected bootstrap changed during RPC",
                )
                return {
                    **result,
                    "selected_bootstrap_sha256": pointer["sha256"],
                    "selected_bootstrap_runtime_admitted": False,
                }
            except BaseException as error:
                try:
                    self.close_weight_sync()
                except BaseException as cleanup_error:
                    error.add_note("selected bootstrap cleanup also failed: " + repr(cleanup_error))
                raise

    return SelectedLoRAWeightSyncExtension


def _selected_extension_attribute(name):
    if name != "SelectedLoRAWeightSyncExtension":
        raise AttributeError(name)
    _receiver_require(
        PATH_ENV in os.environ and SHA_ENV in os.environ,
        "actual engine worker lacks the owned bootstrap",
    )
    return selected_receiver_extension({"path": os.environ[PATH_ENV], "sha256": os.environ[SHA_ENV]})


def selected_engine_kwargs(engine_kwargs, *, rank):
    supplied = dict(engine_kwargs)
    if supplied.get("worker_extension_cls") == SELECTED_LORA_EXTENSION:
        supplied["worker_extension_cls"] = None
    kwargs = engine_kwargs_for_lora(supplied, rank=rank)
    kwargs["worker_extension_cls"] = SELECTED_LORA_EXTENSION
    return kwargs


def make_selected_lora_replica_pool(pool_class, request, pointer):
    _receiver_require(enabled(), "selected sampler requires explicit evidence opt-in")
    request = _sampler_request(request)
    body = bootstrap_descriptor(pointer)
    pointer = dict(pointer)
    _receiver_require(
        body["admitted_sources"] == request["admitted_sources"],
        "request and bootstrap closures differ",
    )
    _bind_sources(
        request["admitted_sources"],
        pool_class,
        SelectedLoRASamplerWorker,
        make_selected_lora_replica_pool,
    )
    import ray

    class OwnedSelectedLoRAWorker(SelectedLoRASamplerWorker):
        def __init__(self):
            super().__init__(request, pointer)

    settings = {VERIFY_ENV: "1", PATH_ENV: pointer["path"], SHA_ENV: pointer["sha256"]}
    actor = ray.remote(runtime_env={"env_vars": settings})(OwnedSelectedLoRAWorker)
    return functools.partial(pool_class, worker_cls=actor)


class SelectedLoRASamplerWorker(LoRASamplerEvidenceWorker):
    """Preserve serving/cleanup; require actual TP replies from the selected bridge."""

    def __init__(self, request, pointer):
        super().__init__(request)
        body = bootstrap_descriptor(pointer)
        _receiver_require(
            body["admitted_sources"] == self._request["admitted_sources"],
            "bootstrap/request closure mismatch",
        )
        self._bootstrap_pointer = dict(pointer)
        self._bootstrap_environment = None

    def _restore_bootstrap_environment(self):
        previous = self._bootstrap_environment
        self._bootstrap_environment = None
        if previous is not None:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    async def initialize(self, engine_kwargs, extra_env=None):
        _receiver_require(
            self.llm is None and self._bootstrap_environment is None,
            "selected sampler already initialized",
        )
        settings = {
            PATH_ENV: self._bootstrap_pointer["path"],
            SHA_ENV: self._bootstrap_pointer["sha256"],
        }
        _receiver_require(
            all(os.environ.get(key) in (None, value) for key, value in settings.items())
            and (not extra_env or all(extra_env.get(key, value) == value for key, value in settings.items())),
            "another owner's bootstrap environment cannot be replaced",
        )
        self._bootstrap_environment = {key: os.environ.get(key) for key in settings}
        try:
            if extra_env:
                os.environ.update(extra_env)
            os.environ.update(settings)
            _receiver_require(enabled(), "selected sampler opt-in must reach the worker")
            bootstrap_descriptor(self._bootstrap_pointer)
            import vllm.plugins
            from arctic_inference.envs import arctic_inference_effective_enabled
            from arctic_inference.server.metrics import RingStatLogger

            vllm.plugins.load_general_plugins()
            from vllm.v1.engine.async_llm import AsyncLLM

            if arctic_inference_effective_enabled():
                from arctic_inference.vllm.args import ArcticAsyncEngineArgs as EngineArgs
            else:
                from vllm.engine.arg_utils import AsyncEngineArgs as EngineArgs
            kwargs = selected_engine_kwargs(engine_kwargs, rank=self._request["lora_config"]["r"])
            if os.environ.get("VLLM_BATCH_INVARIANT") == "1":
                kwargs.setdefault("attention_config", {"backend": "FLASH_ATTN"})
            args = EngineArgs(**kwargs)
            _receiver_require(
                type(args.tensor_parallel_size) is int and args.tensor_parallel_size == 8,
                "actual SAMPLE8 required",
            )
            self._tp = args.tensor_parallel_size
            self._origins = _bind_sources(
                self._request["admitted_sources"],
                type(self),
                SamplerEvidenceWorker,
                EngineArgs,
                args.create_engine_config,
                AsyncLLM.from_vllm_config,
                AsyncLLM.collective_rpc,
                AsyncLLM.generate,
                AsyncLLM.abort,
                AsyncLLM.shutdown,
                serving_lora_request,
            )

            maybe_admit_sampler_construction(self, args, AsyncLLM.from_vllm_config, engine_kwargs)
            self.llm = AsyncLLM.from_vllm_config(args.create_engine_config(), stat_loggers=[RingStatLogger])
            maybe_verify_sampler_construction(self)
            verify_sources(self._origins)
            bootstrap_descriptor(self._bootstrap_pointer)
            self.state = "ready"
        except BaseException as error:
            try:
                if self.llm is not None:
                    self.shutdown()
            except BaseException as cleanup_error:
                error.add_note("selected sampler shutdown also failed: " + repr(cleanup_error))
            finally:
                self._restore_bootstrap_environment()
            raise

    async def sync_weights(self, *args, **kwargs):
        try:
            bootstrap_descriptor(self._bootstrap_pointer)
            result = await super().sync_weights(*args, **kwargs)
            rows = result["rank_results"]
            _receiver_require(
                len(rows) == 8
                and all(
                    row.get("selected_bootstrap_sha256") == self._bootstrap_pointer["sha256"]
                    and row.get("selected_bootstrap_runtime_admitted") is False
                    and (kwargs.get("engine_only") is True or row.get("selected_scope_retained_for_serving") is True)
                    for row in rows
                ),
                "every actual TP rank must report the same selected bootstrap",
            )
            bootstrap_descriptor(self._bootstrap_pointer)
            return result
        except BaseException as error:
            try:
                await self.close_weight_sync()
            except BaseException as cleanup_error:
                error.add_note("selected sampler transport cleanup also failed: " + repr(cleanup_error))
            raise

    def shutdown(self):
        try:
            super().shutdown()
        finally:
            self._restore_bootstrap_environment()


# Receiver Preparation

PREPARATION_FILENAME = "receiver-preparation.json"
BOUND_KEYS = ("max_source_bytes", "max_layers", "max_total_source_bytes")


def _receiver_preparation_require(value, message):
    if not value:
        raise ValueError(message)


def preparation_settings():
    """Consume settings sealed into the same bootstrap's source/data closure."""
    _receiver_preparation_require(enabled(), "selected receiver preparation requires explicit opt-in")
    _receiver_preparation_require(
        PATH_ENV in os.environ and SHA_ENV in os.environ,
        "actual selected worker bootstrap required",
    )
    pointer = {"path": os.environ[PATH_ENV], "sha256": os.environ[SHA_ENV]}
    body = bootstrap_descriptor(pointer)
    path = Path(pointer["path"]).with_name(PREPARATION_FILENAME)
    sources = body["admitted_sources"]
    _receiver_preparation_require(str(path) in sources, "preparation settings must be bootstrap-bound")
    raw = _bytes(path)
    _receiver_preparation_require(
        hashlib.sha256(raw).hexdigest() == sources[str(path)],
        "preparation settings changed",
    )
    settings = json.loads(raw, object_pairs_hook=_unique_object)
    _receiver_preparation_require(
        type(settings) is dict
        and set(settings) == {"schema", "publication_root", "distribution_names", *BOUND_KEYS}
        and settings["schema"] == "job3-selected-lora-preparation-v1",
        "exact selected preparation settings required",
    )
    _receiver_preparation_require(
        all(type(settings[key]) is int and settings[key] > 0 for key in BOUND_KEYS),
        "explicit positive publication bounds required",
    )
    names = settings["distribution_names"]
    _receiver_preparation_require(
        type(names) is list
        and names
        and all(type(name) is str and name for name in names)
        and len(set(names)) == len(names),
        "explicit distinct selected distributions required",
    )
    _receiver_preparation_require(
        type(settings["publication_root"]) is str,
        "explicit private publication root required",
    )
    verify_sources(sources)
    _receiver_preparation_require(_bytes(path) == raw, "preparation settings changed during resolution")
    return settings, dict(sources)


def _actual_components():
    # These imports run on the already-selected native worker, never at module load.
    from vllm.distributed import get_world_group
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.utils import parse_fine_tuned_lora_name

    from arctic_platform.common.peft import normalize_lora_peft_config

    # Resolve the optional server only on the selected native worker.
    native = importlib.import_module("arctic_platform.inference.server.weight_sync.receiver")

    return {
        "observer": observe_lora_build,
        "builder": native._build_lora_model,
        "installer": native._install_lora_adapter,
        "normalize_config": normalize_lora_peft_config,
        "native_model_class": LoRAModel,
        "parse_name": parse_fine_tuned_lora_name,
        "remap_key": native._remap_lora_key,
        "served_prefix_for_model": native._served_lm_module_prefix,
    }, get_world_group


def _rank_output(root, rank):
    root = Path(root)
    _receiver_preparation_require(
        root.is_absolute()
        and root.is_relative_to("/data-fast")
        and root.resolve(strict=True) == root
        and root.is_dir()
        and not root.is_symlink()
        and stat.S_IMODE(root.stat().st_mode) == 0o700,
        "canonical private FAST publication root required",
    )
    output = root / f"tp-rank-{rank}"
    output.mkdir(mode=0o700, exist_ok=False)
    # A newly created NFS directory can inherit setgid; never chmod an old path.
    output.chmod(0o700)
    _receiver_preparation_require(
        output.resolve(strict=True) == output and stat.S_IMODE(output.stat().st_mode) == 0o700,
        "new rank publication directory is not private",
    )
    return output


def prepare_selected_lora_receiver(receiver):
    """Supply exact receiver options from the selected worker's imports."""
    settings, sources = preparation_settings()
    components, get_world_group = _actual_components()
    observed = source_files(
        prepare_selected_lora_receiver,
        _actual_components,
        get_world_group,
        *components.values(),
    )
    _receiver_preparation_require(
        observed.items() <= sources.items(),
        "actual selected builder/parser/rank sources must be admitted",
    )
    verify_sources(sources)
    group = get_world_group()
    _receiver_preparation_require(
        type(group.rank) is int and type(group.world_size) is int and group.world_size == 8 and 0 <= group.rank < 8,
        "actual original SAMPLE8 rank required",
    )
    _receiver_preparation_require(
        receiver.model_runner.model is not None and receiver.model_runner.lora_manager._adapter_manager is not None,
        "already-loaded selected receiver and manager required",
    )
    distributions = tuple(importlib.metadata.distribution(name) for name in settings["distribution_names"])
    verify_sources(sources)
    output = _rank_output(settings["publication_root"], group.rank)
    verify_sources(sources)
    return {
        **components,
        "distributions": distributions,
        "output": output,
        **{key: settings[key] for key in BOUND_KEYS},
    }


# Training


def _training_request(request):
    _receiver_require(
        type(request) is dict and set(request) == {"max_weight_bytes", "admitted_sources"},
        "exact collective LoRA preparation request required",
    )
    _receiver_require(
        type(request["max_weight_bytes"]) is int and request["max_weight_bytes"] > 0,
        "finite positive native adapter byte limit required",
    )
    _receiver_require(
        type(request["admitted_sources"]) is dict and request["admitted_sources"],
        "actual operation source closure required before collective preparation",
    )
    return deepcopy(request)


def prepare_lora_worker(worker, request):
    """Run on every live native training actor; tensors never leave that actor."""
    _receiver_require(enabled(), "collective LoRA preparation must be explicitly enabled")
    request = _training_request(request)
    _receiver_require(
        type(worker.rank) is int and 0 <= worker.rank < 8 and worker.world_size == 8,
        "actual original eight-rank training actor required",
    )
    engine = worker.engine
    _receiver_require(
        engine is not None and engine.module is not None, "actual initialized native engine.module required"
    )
    _receiver_require(
        not getattr(worker, "_ap_lora_prepare_attempted", False) and not hasattr(worker, "_ap_owned_lora_preparation"),
        "native adapter preparation is single-use; no unchanged collective retry",
    )
    model = engine.module
    origins = _bind_sources(
        request["admitted_sources"],
        prepare_lora_worker,
        prepare_collective_lora_weights,
        send_prepared_lora_weights,
        type(model),
        model.named_parameters,
        model.named_buffers,
    )
    worker._ap_lora_prepare_attempted = True
    prepared = prepare_collective_lora_weights(
        model,
        weight_format="lora",
        is_zero3=False,
        max_weight_bytes=request["max_weight_bytes"],
        admitted_sources=origins,
        expected_world_size=8,
        expected_ep_size=8,
    )
    _receiver_require(
        isinstance(prepared, PreparedLoRAWeights) and prepared.rank == worker.rank,
        "actual native preparation returned a different training rank",
    )
    _receiver_require(worker.engine is engine and engine.module is model, "native model changed during preparation")
    receipt = deepcopy(prepared.receipt())
    _bind_sources(origins, prepare_lora_worker, prepare_collective_lora_weights, send_prepared_lora_weights)
    worker._ap_owned_lora_preparation = {
        "engine": engine,
        "model": model,
        "prepared": prepared,
        "request": request,
        "send_attempted": False,
    }
    return receipt


def send_lora_worker(worker):
    """Send the captured stream with the current owned native sender, once."""
    _receiver_require(enabled(), "prepared LoRA send must be explicitly enabled")
    slot = getattr(worker, "_ap_owned_lora_preparation", None)
    _receiver_require(type(slot) is dict, "actual actor-local collective preparation required before send")
    _receiver_require(
        worker.rank == 0 and slot["prepared"].rank == 0,
        "only actual global rank zero may send the native adapter stream",
    )
    _receiver_require(
        worker.engine is slot["engine"] and worker.engine.module is slot["model"],
        "prepared stream belongs to a different native engine or model",
    )
    _receiver_require(not slot["send_attempted"], "prepared native send is single-use")
    sender = worker._weight_sender
    _receiver_require(sender is not None, "actual initialized native weight sender required")
    origins = _bind_sources(slot["prepared"].sources, send_lora_worker, send_prepared_lora_weights, sender.send)
    slot["send_attempted"] = True
    result = send_prepared_lora_weights(sender, slot["prepared"])
    _receiver_require(
        worker._weight_sender is sender and worker._ap_owned_lora_preparation is slot,
        "owned sender or preparation changed during native send",
    )
    _receiver_require(
        worker.engine is slot["engine"] and worker.engine.module is slot["model"],
        "native model changed during send",
    )
    _bind_sources(origins, send_lora_worker, send_prepared_lora_weights, sender.send)
    return result


def release_lora_worker(worker, slot):
    """Drop only the captured private tensors; native actor teardown stays separate."""
    _receiver_require(
        slot is not None and getattr(worker, "_ap_owned_lora_preparation", None) is slot,
        "refusing to release another owner's preparation",
    )
    del worker._ap_owned_lora_preparation
    return {
        "owned_preparation_released": True,
        "native_sender_destroyed": False,
        "native_process_group_destroyed": False,
    }


async def prepare_lora_training_workers(workers, request):
    """Await all eight public actor preparations before returning send readiness."""
    _receiver_require(enabled(), "collective LoRA preparation must be explicitly enabled")
    request = _training_request(request)
    _receiver_require(
        type(workers) is list and len(workers) == 8 and len({id(worker) for worker in workers}) == 8,
        "exact eight distinct current training actor handles required",
    )
    methods = [worker.prepare_lora_weights.remote for worker in workers]
    _bind_sources(request["admitted_sources"], prepare_lora_training_workers, prepare_lora_worker)
    pending = []
    errors = []
    for method in methods:
        try:
            pending.append(method(deepcopy(request)))
        except BaseException as error:
            errors.append(error)
    completed = await asyncio.gather(*pending, return_exceptions=True)
    errors.extend(value for value in completed if isinstance(value, BaseException))
    if errors:
        preparation_error = RuntimeError("one or more native training-rank preparations failed")
        for cause in errors:
            preparation_error.add_note(repr(cause))
        raise preparation_error from errors[0]
    _receiver_require(len(completed) == 8, "all eight actual training-rank preparations must complete")
    for rank, receipt in enumerate(completed):
        _receiver_require(
            type(receipt) is dict
            and type(receipt.get("rank")) is int
            and receipt["rank"] == rank
            and receipt.get("weight_format") == "lora"
            and receipt.get("native_iterator_completed") is True
            and receipt.get("unchanged_native_iterator") is True
            and receipt.get("native_shard_bytes_checked") is (rank == 0),
            "actual completed native preparation from each ordered training rank required",
        )
        _receiver_require(
            type(receipt.get("rank_records")) is list
            and len(receipt["rank_records"]) == 8
            and [row.get("rank") for row in receipt["rank_records"]] == list(range(8)),
            "every preparation must retain all eight actual rank records",
        )
        _receiver_require(
            receipt.get("source_files") == request["admitted_sources"],
            "training preparation sources differ from the current operation",
        )
        for key in (
            "global_inventory",
            "rank_records",
            "adapter_parameter_names",
            "non_wire_parameter_names",
            "non_wire_buffers",
        ):
            _receiver_require(
                key in receipt and receipt[key] == completed[0].get(key),
                "collective native inventory or adapter selection differs across training ranks",
            )
        for key in (
            "receiver_installation_admitted",
            "serving_request_admitted",
            "router_application_admitted",
            "runtime_admitted",
            "capacity_admitted",
            "parameter_only_judge_admitted",
            "recipe_admitted",
        ):
            _receiver_require(receipt.get(key) is False, "preparation cannot promote native or judge admission")
    _bind_sources(request["admitted_sources"], prepare_lora_training_workers, prepare_lora_worker)
    return {
        "schema": "all-training-ranks-lora-prepared-v1",
        "rank_preparations": completed,
        "all_training_ranks_prepared": True,
        "private_streams_remain_actor_local": True,
        "native_send_started": False,
        "capacity_admitted": False,
        "parameter_only_judge_admitted": False,
        "numerical": None,
    }


# Sync

REQUEST_ENV = "ARCTIC_PLATFORM_LORA_SYNC_REQUEST"
DIGEST_ENV = "ARCTIC_PLATFORM_LORA_SYNC_REQUEST_SHA256"
INITIALIZATION_ENV = "ARCTIC_PLATFORM_INITIALIZATION_REQUEST"
INITIALIZATION_SHA_ENV = "ARCTIC_PLATFORM_INITIALIZATION_REQUEST_SHA256"
MODEL_CONSTRUCTION_ENV = "ARCTIC_PLATFORM_MODEL_CONSTRUCTION_REQUEST"
MODEL_CONSTRUCTION_SHA_ENV = "ARCTIC_PLATFORM_MODEL_CONSTRUCTION_REQUEST_SHA256"
_MODEL_CONSTRUCTOR_FRAME = ContextVar("arctic_platform_owned_model_constructor_frame", default=None)


def _initialization_request(
    *,
    environment=None,
    path_env=INITIALIZATION_ENV,
    digest_env=INITIALIZATION_SHA_ENV,
    schema="explicit-native-initialization-request-v1",
):
    settings = os.environ if environment is None else environment
    path, digest = settings.get(path_env), settings.get(digest_env)
    if path is None and digest is None:
        return None
    _receiver_require(
        settings.get(VERIFY_ENV) == "1",
        "initialization request requires explicit evidence opt-in",
    )
    _receiver_require(
        type(path) is str
        and Path(path).is_relative_to("/data-fast")
        and type(digest) is str
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        "selected initialization requires an explicit FAST request and SHA256",
    )

    raw = _bytes(path)
    _receiver_require(hashlib.sha256(raw).hexdigest() == digest, "initialization request changed")
    request = json.loads(raw, object_pairs_hook=_unique_object)
    fields = {"schema", "admitted_sources", "runtime", "runtime_options", "verify_capacity", "capacity_request"}
    if schema == "explicit-native-model-construction-request-v1":
        fields.add("factory_route")
    _receiver_require(
        type(request) is dict and set(request) == fields and request["schema"] == schema,
        "exact source-bound initialization request required",
    )
    sources = request["admitted_sources"]
    _receiver_require(type(sources) is dict and sources, "initialization requires actual source origins")
    for source, expected in sources.items():
        _receiver_require(
            type(source) is str
            and type(expected) is str
            and len(expected) == 64
            and all(character in "0123456789abcdef" for character in expected)
            and hashlib.sha256(_bytes(source)).hexdigest() == expected,
            "initialization source identity differs",
        )
    for key in ("runtime", "verify_capacity"):
        item = request[key]
        _receiver_require(
            type(item) is dict
            and set(item) == {"module", "name", "source"}
            and type(item["module"]) is str
            and all(part.isidentifier() for part in item["module"].split("."))
            and type(item["name"]) is str
            and item["name"].isidentifier()
            and item["source"] in sources,
            "explicit admitted initialization provider required",
        )
    options = request["runtime_options"]
    _receiver_require(
        type(options) is dict
        and set(options) == {"release_bound", "wrapper_sources"}
        and type(options["release_bound"]) is dict
        and set(options["release_bound"]) == {"path", "sha256"}
        and type(options["release_bound"]["path"]) is str
        and type(options["release_bound"]["sha256"]) is str
        and sources.get(options["release_bound"]["path"]) == options["release_bound"]["sha256"]
        and type(options["wrapper_sources"]) is dict
        and options["wrapper_sources"].items() <= sources.items()
        and type(request["capacity_request"]) is dict
        and request["capacity_request"],
        "explicit existing runtime release and complete capacity supplier request required",
    )
    _bind_sources(sources, _initialization_request, _bytes, _unique_object)
    _receiver_require(_bytes(path) == raw, "initialization request changed during admission")
    return {"path": path, "sha256": digest}, request


def _initialization_code(actual, sources):

    path = actual.co_filename
    _receiver_require(path in sources, "initialization callable origin is not admitted")
    code = compile(_bytes(path), path, "exec", dont_inherit=True, optimize=sys.flags.optimize)
    pending, matches = [code], []
    while pending:
        current = pending.pop()
        if current.co_qualname == actual.co_qualname:
            matches.append(current)
        pending.extend(value for value in current.co_consts if isinstance(value, CodeType))
    _receiver_require(
        len(matches) == 1 and actual == matches[0] and actual.co_linetable == matches[0].co_linetable,
        "loaded initialization callable differs from admitted source",
    )


def _initialization_function(function, sources):
    _receiver_require(inspect.isfunction(function), "source-derived initialization Python callable required")
    _receiver_require(
        function.__code__.co_filename == inspect.getsourcefile(function)
        and function.__code__.co_qualname == function.__qualname__,
        "loaded initialization callable differs from admitted source",
    )
    _initialization_code(function.__code__, sources)


def _initialization_slot(selected, subject, *surfaces, request_reader=_initialization_request):

    pointer, request = selected
    sources = request["admitted_sources"]
    runtime_class = _symbol(request["runtime"], sources)
    capacity = _symbol(request["verify_capacity"], sources)
    _receiver_require(
        inspect.isclass(runtime_class) and inspect.isfunction(capacity) and capacity.__closure__ is None,
        "explicit native runtime class and top-level capacity verifier required",
    )
    _bind_sources(
        sources,
        _initialization_slot,
        _symbol,
        runtime_class,
        runtime_class.guard,
        runtime_class.verify_engine,
        capacity,
        request_reader,
        *surfaces,
    )
    for function in (capacity, runtime_class.__init__, runtime_class.guard, runtime_class.verify_engine):
        _initialization_function(function, sources)
    options = request["runtime_options"]
    runtime = runtime_class(deepcopy(options["release_bound"]), wrapper_sources=deepcopy(options["wrapper_sources"]))
    slot = {
        "pointer": pointer,
        "request": request,
        "request_reader": request_reader,
        "runtime_class": runtime_class,
        "runtime": runtime,
        "capacity": capacity,
        "capacity_code": capacity.__code__,
        "capacity_defaults": deepcopy((capacity.__defaults__, capacity.__kwdefaults__)),
        "subject": subject,
        "status": "armed",
        "observations": [],
        "runtime_methods": {
            name: (
                getattr(runtime_class, name),
                getattr(runtime_class, name).__code__,
                deepcopy((getattr(runtime_class, name).__defaults__, getattr(runtime_class, name).__kwdefaults__)),
            )
            for name in ("guard", "verify_engine")
        },
    }
    _guard_initialization(slot)
    return slot


def _guard_initialization(slot):

    _receiver_require(
        slot["request_reader"]() == (slot["pointer"], slot["request"]),
        "armed initialization request or source closure changed",
    )
    request = slot["request"]
    _receiver_require(
        _symbol(request["runtime"], request["admitted_sources"]) is slot["runtime_class"]
        and _symbol(request["verify_capacity"], request["admitted_sources"]) is slot["capacity"]
        and slot["capacity"].__code__ is slot["capacity_code"],
        "captured initialization provider changed",
    )
    _receiver_require(
        (slot["capacity"].__defaults__, slot["capacity"].__kwdefaults__) == slot["capacity_defaults"],
        "captured capacity verifier defaults changed",
    )
    _require_initialization_runtime_methods(slot)
    slot["runtime"].guard()
    _require_initialization_runtime_methods(slot)


def _require_initialization_runtime_methods(slot):
    for name, (function, code, defaults) in slot["runtime_methods"].items():
        method = getattr(slot["runtime"], name)
        _receiver_require(
            inspect.ismethod(method)
            and method.__self__ is slot["runtime"]
            and method.__func__ is function
            and function.__code__ is code,
            "captured runtime verification method changed",
        )
        _receiver_require(
            (function.__defaults__, function.__kwdefaults__) == defaults,
            "captured runtime verification method defaults changed",
        )


def _verify_initialization_capacity(slot, phase, subject):
    _guard_initialization(slot)
    request = deepcopy(slot["request"]["capacity_request"])
    result = slot["capacity"](request, phase=phase, subject=dict(subject), runtime=slot["runtime"])
    _receiver_require(
        type(result) is dict
        and set(result) == {"phase", "request", "evidence"}
        and result["phase"] == phase
        and request == result["request"] == slot["request"]["capacity_request"]
        and type(result["evidence"]) is dict
        and result["evidence"],
        "actual phase-bound capacity supplier evidence required",
    )
    _guard_initialization(slot)
    observation = {
        "schema": "native-initialization-hook-observation-v1",
        "phase": phase,
        "provider_evidence": deepcopy(result["evidence"]),
        "capacity_admitted": False,
        "numerical": None,
    }
    slot["observations"].append(observation)
    return observation


def _model_construction_request():
    return _initialization_request(
        path_env=MODEL_CONSTRUCTION_ENV,
        digest_env=MODEL_CONSTRUCTION_SHA_ENV,
        schema="explicit-native-model-construction-request-v1",
    )


def _model_construction_route(frames, route, sources):
    _receiver_require(
        type(route) is dict and set(route) == {"factory", "caller", "constructor"},
        "explicit factory, loader and constructor route required",
    )
    for name, frame in zip(("factory", "caller", "constructor"), frames, strict=True):
        descriptor = route[name]
        _receiver_require(
            type(descriptor) is dict
            and set(descriptor) == {"module", "qualname", "source"}
            and frame is not None
            and descriptor
            == {
                "module": frame.f_globals.get("__name__"),
                "qualname": frame.f_code.co_qualname,
                "source": frame.f_code.co_filename,
            },
            "actual model construction route differs from selected caller",
        )
        _initialization_code(frame.f_code, sources)


def _guard_model_constructor_frame(binding):
    owner_class, original, wrapper = binding["class"], binding["original"], binding["wrapper"]
    _receiver_require(
        vars(owner_class).get("__init__") is wrapper
        and original.__code__ is binding["original_code"]
        and wrapper.__code__ is binding["wrapper_code"]
        and getattr(wrapper, "__wrapped__", None) is original,
        "owned native constructor binding changed",
    )
    closure = dict(zip(wrapper.__code__.co_freevars, wrapper.__closure__ or (), strict=True))
    _receiver_require(
        "original" in closure
        and "worker_class" in closure
        and closure["original"].cell_contents is original
        and closure["worker_class"].cell_contents is owner_class,
        "owned native constructor closure changed",
    )
    caller = binding["frame"]
    _receiver_require(
        caller.f_code is binding["wrapper_code"]
        and caller.f_locals.get("worker") is binding["worker"]
        and caller.f_locals.get("original") is original
        and caller.f_locals.get("worker_class") is owner_class,
        "actual owned native wrapper frame required",
    )
    current = sys.modules.get(owner_class.__module__)
    for component in owner_class.__qualname__.split("."):
        _receiver_require(current is not None and component in vars(current), "actual native class origin required")
        current = vars(current)[component]
    _receiver_require(current is owner_class, "native module no longer owns the constructor class")
    _receiver_require(
        (original.__defaults__, original.__kwdefaults__, wrapper.__defaults__, wrapper.__kwdefaults__)
        == binding["defaults"],
        "owned native constructor defaults changed",
    )
    _receiver_require(
        _model_construction_request() == binding["selected"],
        "owned native constructor request or sources changed",
    )
    sources = binding["selected"][1]["admitted_sources"]
    _bind_sources(
        sources,
        owned_model_constructor_frame,
        _OwnedModelConstructorFrame,
        _guard_model_constructor_frame,
        _model_constructor_binding,
        owner_class,
        original,
        wrapper,
    )
    _initialization_function(original, sources)
    _initialization_code(wrapper.__code__, sources)


class _OwnedModelConstructorFrame:
    def __init__(self, binding):
        self.binding = binding
        self.token = None

    def __enter__(self):
        _receiver_require(
            self.token is None and _MODEL_CONSTRUCTOR_FRAME.get() is None,
            "nested or reused native constructor ownership rejected",
        )
        _guard_model_constructor_frame(self.binding)
        self.token = _MODEL_CONSTRUCTOR_FRAME.set(self.binding)
        self.binding["active"] = True

    def __exit__(self, kind, error, traceback):
        try:
            if kind is None:
                _guard_model_constructor_frame(self.binding)
        finally:
            self.binding["active"] = False
            _MODEL_CONSTRUCTOR_FRAME.reset(self.token)
            self.binding["frame"] = None


def owned_model_constructor_frame(worker, *, constructor, wrapper, admitted_sources):
    """Bind the exact owning wrapper frame while it calls its native constructor."""
    selected = _model_construction_request()
    if selected is None:
        return nullcontext()
    owner_class = type(worker)
    _receiver_require(
        type(owner_class) is type
        and inspect.isfunction(constructor)
        and inspect.isfunction(wrapper)
        and constructor.__qualname__ == owner_class.__qualname__ + ".__init__"
        and constructor.__module__ == owner_class.__module__
        and type(admitted_sources) is dict
        and admitted_sources == selected[1]["admitted_sources"],
        "exact native constructor and selected source closure required",
    )
    frame = inspect.currentframe()
    try:
        binding = {
            "class": owner_class,
            "worker": worker,
            "original": constructor,
            "original_code": constructor.__code__,
            "wrapper": wrapper,
            "wrapper_code": wrapper.__code__,
            "frame": frame.f_back,
            "selected": selected,
            "defaults": deepcopy(
                (constructor.__defaults__, constructor.__kwdefaults__, wrapper.__defaults__, wrapper.__kwdefaults__)
            ),
            "active": False,
        }
        _guard_model_constructor_frame(binding)
        return _OwnedModelConstructorFrame(binding)
    finally:
        del frame


def _model_constructor_binding(owner, constructor, sources):
    installed = type(owner).__init__
    _receiver_require(inspect.isfunction(installed), "actual native constructor owner differs from selected code")
    if installed.__code__ is constructor.f_code:
        _initialization_function(installed, sources)
        return installed, installed, None
    binding = _MODEL_CONSTRUCTOR_FRAME.get()
    _receiver_require(
        binding is not None
        and binding["active"]
        and binding["worker"] is owner
        and binding["class"] is type(owner)
        and binding["original_code"] is constructor.f_code
        and constructor.f_back is binding["frame"]
        and sources == binding["selected"][1]["admitted_sources"],
        "actual native constructor owner differs from selected code",
    )
    _guard_model_constructor_frame(binding)
    return binding["original"], binding["wrapper"], binding


def maybe_admit_model_construction(spec, parallel_groups):
    selected = _model_construction_request()
    if selected is None:
        return None
    from arctic_platform.model.config import ModelSpec

    frame = inspect.currentframe()
    factory = frame.f_back
    caller = factory.f_back if factory is not None else None
    constructor = caller.f_back if caller is not None else None
    try:
        sources = selected[1]["admitted_sources"]
        frames = (factory, caller, constructor)
        route = selected[1]["factory_route"]
        _model_construction_route(frames, route, sources)
        _receiver_require(type(spec) is ModelSpec, "actual native ModelSpec required before model construction")
        _receiver_require(
            parallel_groups is None or type(parallel_groups) is dict, "actual selected parallel groups required"
        )
        owner, config = constructor.f_locals.get("self"), constructor.f_locals.get("cfg")
        _receiver_require(
            owner is not None and type(config) is dict, "actual native constructor owner and cfg required"
        )
        constructor_binding = _model_constructor_binding(owner, constructor, sources)
        build_model = factory.f_globals.get("build_model")
        _receiver_require(
            inspect.isfunction(build_model) and build_model.__code__ is factory.f_code,
            "actual native build_model callable required",
        )
        spec_before, config_before = deepcopy(spec), deepcopy(config)
        groups_before = None if parallel_groups is None else dict(parallel_groups)
        subject = {
            "worker": owner,
            "config": config,
            "model_spec": spec,
            "parallel_groups": parallel_groups,
            "build_model": build_model,
            "factory_route": deepcopy(route),
        }
        slot = _initialization_slot(
            selected,
            subject,
            maybe_admit_model_construction,
            _model_construction_route,
            _model_constructor_binding,
            ModelSpec,
            type(owner),
            *constructor_binding[:2],
            build_model,
            request_reader=_model_construction_request,
        )
        result = _verify_initialization_capacity(slot, "before-build-model", subject)
        _model_construction_route(frames, route, sources)
        _receiver_require(
            spec == spec_before and config == config_before, "capacity verifier changed native spec or cfg"
        )
        _receiver_require(
            (parallel_groups is None and groups_before is None)
            or (
                type(parallel_groups) is dict
                and parallel_groups.keys() == groups_before.keys()
                and all(parallel_groups[key] is value for key, value in groups_before.items())
            ),
            "capacity verifier changed native parallel groups",
        )
        _receiver_require(
            factory.f_globals.get("build_model") is build_model
            and build_model.__code__ is factory.f_code
            and all(
                actual is captured
                for actual, captured in zip(
                    _model_constructor_binding(owner, constructor, sources),
                    constructor_binding,
                    strict=True,
                )
            ),
            "capacity verifier replaced native factory or constructor code",
        )
        return result
    finally:
        del frame, factory, caller, constructor


def maybe_observe_model_materialization(model, adapter, config, parallel_dims, ep_mesh):
    """Supply the actual post-sharding model before destination allocation."""
    selected = _model_construction_request()
    if selected is None:
        return None
    from arctic_platform.model.config import ModelSpec
    from arctic_platform.model.implementations.debug.gpu_memory import observe_native_model_inventory

    frame = inspect.currentframe()
    load_caller = frame.f_back
    factory = caller = constructor = None
    try:
        sources = selected[1]["admitted_sources"]
        descriptor = selected[1]["factory_route"]["factory"]
        active = load_caller
        matches = []
        while active is not None:
            if descriptor == {
                "module": active.f_globals.get("__name__"),
                "qualname": active.f_code.co_qualname,
                "source": active.f_code.co_filename,
            }:
                matches.append(active)
            active = active.f_back
        _receiver_require(len(matches) == 1, "one active owning model factory required before materialization")
        factory = matches[0]
        caller = factory.f_back
        constructor = caller.f_back if caller is not None else None
        frames = (factory, caller, constructor)
        _model_construction_route(frames, selected[1]["factory_route"], sources)
        _initialization_code(load_caller.f_code, sources)
        owner, native_config = constructor.f_locals.get("self"), constructor.f_locals.get("cfg")
        spec, groups = factory.f_locals.get("spec"), factory.f_locals.get("parallel_groups")
        _receiver_require(
            owner is not None and type(native_config) is dict and type(spec) is ModelSpec,
            "actual native constructor owner, cfg and ModelSpec required before materialization",
        )
        _receiver_require(groups is None or type(groups) is dict, "actual selected parallel groups required")
        binding = _model_constructor_binding(owner, constructor, sources)
        _receiver_require(
            load_caller.f_locals.get("model") is model
            and load_caller.f_locals.get("adapter") is adapter
            and load_caller.f_locals.get("config") is config
            and load_caller.f_locals.get("parallel_dims") is parallel_dims
            and load_caller.f_locals.get("ep_mesh") is ep_mesh,
            "actual loader model, adapter, config and placement required",
        )
        load = adapter.load_dcp_from_hf
        _initialization_function(load, sources)
        load_code, load_defaults = load.__code__, deepcopy((load.__defaults__, load.__kwdefaults__))
        build_model = factory.f_globals.get("build_model")
        _receiver_require(
            inspect.isfunction(build_model) and build_model.__code__ is factory.f_code,
            "actual native build_model callable required",
        )
        _bind_sources(
            sources, maybe_observe_model_materialization, observe_native_model_inventory, type(model), ModelSpec
        )
        _initialization_function(observe_native_model_inventory, sources)
        retained = []
        inventory = observe_native_model_inventory(model, retained_objects=retained)
        before = deepcopy((adapter, config, parallel_dims, spec, native_config))
        groups_before = None if groups is None else dict(groups)
        subject = {
            "worker": owner,
            "model": model,
            "model_spec": spec,
            "native_config": native_config,
            "parallel_groups": groups,
            "adapter": adapter,
            "config": config,
            "parallel_dims": parallel_dims,
            "ep_mesh": ep_mesh,
            "native_inventory": deepcopy(inventory),
            "build_model": build_model,
            "factory_route": deepcopy(selected[1]["factory_route"]),
        }
        slot = _initialization_slot(
            selected,
            subject,
            maybe_observe_model_materialization,
            observe_native_model_inventory,
            type(model),
            ModelSpec,
            build_model,
            load,
            *binding[:2],
            request_reader=_model_construction_request,
        )
        result = _verify_initialization_capacity(slot, "before-model-materialization", subject)
        _model_construction_route(frames, selected[1]["factory_route"], sources)
        _receiver_require(
            (adapter, config, parallel_dims, spec, native_config) == before,
            "capacity verifier changed the selected loading recipe",
        )
        _receiver_require(
            factory.f_globals.get("build_model") is build_model
            and build_model.__code__ is factory.f_code
            and factory.f_locals.get("spec") is spec
            and factory.f_locals.get("parallel_groups") is groups
            and constructor.f_locals.get("cfg") is native_config
            and constructor.f_locals.get("self") is owner
            and all(
                load_caller.f_locals.get(name) is value
                for name, value in (
                    ("model", model),
                    ("adapter", adapter),
                    ("config", config),
                    ("parallel_dims", parallel_dims),
                    ("ep_mesh", ep_mesh),
                )
            ),
            "capacity verifier changed the active model factory or loader inputs",
        )
        _receiver_require(
            (groups is None and groups_before is None)
            or (
                type(groups) is dict
                and groups.keys() == groups_before.keys()
                and all(groups[key] is value for key, value in groups_before.items())
            ),
            "capacity verifier changed native parallel groups",
        )
        _receiver_require(
            adapter.load_dcp_from_hf is load
            and load.__code__ is load_code
            and (load.__defaults__, load.__kwdefaults__) == load_defaults,
            "capacity verifier changed the selected checkpoint loader",
        )
        _receiver_require(
            observe_native_model_inventory(model, retained_objects=retained) == inventory,
            "capacity verifier changed native parameters, buffers or storage",
        )
        _receiver_require(
            all(
                actual is captured
                for actual, captured in zip(
                    _model_constructor_binding(owner, constructor, sources),
                    binding,
                    strict=True,
                )
            ),
            "capacity verifier changed the owning native constructor",
        )
        _bind_sources(sources, maybe_observe_model_materialization, observe_native_model_inventory, type(model), load)
        _initialization_function(load, sources)
        _initialization_function(observe_native_model_inventory, sources)
        return result
    finally:
        del frame, load_caller, factory, caller, constructor


def maybe_observe_checkpoint_reader(reader, model, state_dict, config, parallel_dims):
    """Observe one selected instance's finalized plan before its original reads."""
    selected = _model_construction_request()
    if selected is None:
        return reader
    from arctic_platform.model.implementations.debug.gpu_memory import observe_native_model_inventory

    frame = inspect.currentframe()
    loader = frame.f_back
    sources = selected[1]["admitted_sources"]
    try:
        _initialization_code(loader.f_code, sources)
        _receiver_require(
            all(
                loader.f_locals.get(name) is value
                for name, value in (
                    ("model", model),
                    ("state_dict", state_dict),
                    ("config", config),
                    ("parallel_dims", parallel_dims),
                )
            ),
            "actual checkpoint builder inputs required",
        )
        _receiver_require(type(state_dict) is dict and state_dict, "actual filtered checkpoint destinations required")
        original = reader.read_data
        _receiver_require(
            inspect.ismethod(original)
            and original.__self__ is reader
            and "read_data" not in vars(reader)
            and original.__func__ is type(reader).read_data,
            "unwrapped selected reader instance required",
        )
        _initialization_function(original.__func__, sources)
        function, code = original.__func__, original.__func__.__code__
        defaults = deepcopy((function.__defaults__, function.__kwdefaults__))
        _bind_sources(sources, maybe_observe_checkpoint_reader, observe_native_model_inventory, function)
        descriptor = selected[1]["factory_route"]["factory"]
        active, matches = loader, []
        while active is not None:
            if descriptor == {
                "module": active.f_globals.get("__name__"),
                "qualname": active.f_code.co_qualname,
                "source": active.f_code.co_filename,
            }:
                matches.append(active)
            active = active.f_back
        _receiver_require(len(matches) == 1, "one active owning model factory required for checkpoint reader")
        factory = matches[0]
        caller = factory.f_back
        constructor = caller.f_back if caller is not None else None
        frames = (factory, caller, constructor)
        _model_construction_route(frames, selected[1]["factory_route"], sources)
        owner = constructor.f_locals.get("self")
        binding = _model_constructor_binding(owner, constructor, sources)
        spec, groups = factory.f_locals.get("spec"), factory.f_locals.get("parallel_groups")
        native_config = constructor.f_locals.get("cfg")
        build_model = factory.f_globals.get("build_model")
        _receiver_require(
            inspect.isfunction(build_model) and build_model.__code__ is factory.f_code,
            "actual native build_model callable required for checkpoint reader",
        )
        recipe = deepcopy((config, parallel_dims, spec, native_config))
        group_items = None if groups is None else tuple(groups.items())
        state_items = tuple(state_dict.items())
        cache_path, threads = reader.path, reader.thread_count
        status = "armed"

        def guard():
            _receiver_require(_model_construction_request() == selected, "selected checkpoint request changed")
            _model_construction_route(frames, selected[1]["factory_route"], sources)
            current = inspect.currentframe()
            active = current.f_back
            try:
                while active is not None and active is not loader:
                    active = active.f_back
                _receiver_require(active is loader, "selected checkpoint builder is no longer active")
            finally:
                del current, active
            _receiver_require(
                factory.f_globals.get("build_model") is build_model
                and build_model.__code__ is factory.f_code
                and factory.f_locals.get("spec") is spec
                and factory.f_locals.get("parallel_groups") is groups
                and constructor.f_locals.get("self") is owner
                and constructor.f_locals.get("cfg") is native_config
                and all(
                    loader.f_locals.get(name) is value
                    for name, value in (
                        ("model", model),
                        ("state_dict", state_dict),
                        ("config", config),
                        ("parallel_dims", parallel_dims),
                    )
                )
                and (config, parallel_dims, spec, native_config) == recipe,
                "selected checkpoint construction recipe changed",
            )
            _receiver_require(
                (groups is None and group_items is None)
                or (
                    type(groups) is dict
                    and tuple(groups) == tuple(name for name, _ in group_items)
                    and all(groups[name] is value for name, value in group_items)
                ),
                "selected checkpoint parallel groups changed",
            )
            _receiver_require(
                type(state_dict) is dict
                and tuple(state_dict) == tuple(name for name, _ in state_items)
                and all(state_dict[name] is value for name, value in state_items),
                "selected checkpoint destination binding changed",
            )
            _receiver_require(
                reader.read_data is installed
                and type(reader).read_data is function
                and inspect.ismethod(original)
                and original.__self__ is reader
                and original.__func__ is function
                and function.__code__ is code
                and (function.__defaults__, function.__kwdefaults__) == defaults
                and reader.path == cache_path
                and reader.thread_count == threads,
                "selected checkpoint reader binding, cache or concurrency changed",
            )
            _receiver_require(
                all(
                    actual is captured
                    for actual, captured in zip(
                        _model_constructor_binding(owner, constructor, sources), binding, strict=True
                    )
                ),
                "selected checkpoint native constructor changed",
            )
            _initialization_function(function, sources)
            _initialization_function(observe_native_model_inventory, sources)

        def observed_read(actual_reader, plan, planner):
            nonlocal status
            _receiver_require(
                actual_reader is reader and status == "armed", "selected checkpoint reader is single-use"
            )
            status = "observing"
            guard()
            _receiver_require(
                type(planner.state_dict) is dict
                and planner.state_dict.keys() == state_dict.keys()
                and all(planner.state_dict[name] is value for name, value in state_items),
                "selected planner must retain the actual filtered destination tensors",
            )
            planner_dict = planner.state_dict
            methods = {}
            for name in ("resolve_tensor", "commit_tensor", "finish_plan"):
                method = getattr(planner, name)
                _receiver_require(
                    inspect.ismethod(method) and method.__self__ is planner, "actual selected planner required"
                )
                _initialization_function(method.__func__, sources)
                methods[name] = (
                    method.__func__,
                    method.__func__.__code__,
                    deepcopy((method.__func__.__defaults__, method.__func__.__kwdefaults__)),
                )
            retained = []
            inventory = observe_native_model_inventory(
                model, retained_objects=retained, checkpoint_destinations=state_dict
            )
            before_plan, before_storage = deepcopy(plan), deepcopy(reader.storage_data)
            subject = {
                "worker": owner,
                "model": model,
                "model_spec": spec,
                "native_config": native_config,
                "parallel_groups": groups,
                "config": config,
                "parallel_dims": parallel_dims,
                "reader": reader,
                "planner": planner,
                "load_plan": plan,
                "checkpoint_destinations": state_dict,
                "native_inventory": deepcopy(inventory),
                "reader_storage": deepcopy(before_storage),
                "cache_path": cache_path,
                "reader_threads": threads,
                "factory_route": deepcopy(selected[1]["factory_route"]),
            }
            slot = _initialization_slot(
                selected,
                subject,
                maybe_observe_checkpoint_reader,
                observe_native_model_inventory,
                guard,
                observed_read,
                function,
                *[item[0] for item in methods.values()],
                build_model,
                *binding[:2],
                request_reader=_model_construction_request,
            )
            _verify_initialization_capacity(slot, "before-checkpoint-read", subject)
            guard()
            _receiver_require(
                plan == before_plan and reader.storage_data == before_storage,
                "capacity verifier changed the finalized checkpoint plan or file bindings",
            )
            _receiver_require(
                planner.state_dict is planner_dict
                and planner_dict.keys() == state_dict.keys()
                and all(planner_dict[name] is value for name, value in state_items)
                and all(
                    getattr(planner, name).__self__ is planner
                    and getattr(planner, name).__func__ is method
                    and method.__code__ is captured_code
                    and (method.__defaults__, method.__kwdefaults__) == captured_defaults
                    for name, (method, captured_code, captured_defaults) in methods.items()
                ),
                "capacity verifier changed the selected checkpoint planner",
            )
            _receiver_require(
                observe_native_model_inventory(model, retained_objects=retained, checkpoint_destinations=state_dict)
                == inventory,
                "capacity verifier changed checkpoint backing storage",
            )
            _initialization_function(guard, sources)
            _initialization_function(observed_read, sources)
            status = "reading"
            try:
                return original(plan, planner)
            finally:
                status = "finished"

        installed = MethodType(observed_read, reader)
        reader.read_data = installed
        return reader
    finally:
        del frame


def maybe_admit_worker_construction(worker, spec, ds_config, parallel_groups, build_model, initialize):
    selected = _initialization_request()
    if selected is None:
        return None
    _receiver_require(not hasattr(worker, "_ap_initialization_admission"), "worker initialization is single-use")
    subject = {
        "worker": worker,
        "model_spec": spec,
        "ds_config": ds_config,
        "parallel_groups": parallel_groups,
        "build_model": build_model,
        "initialize": initialize,
    }
    slot = _initialization_slot(
        selected,
        subject,
        maybe_admit_worker_construction,
        maybe_verify_worker_engine,
        type(worker),
        worker.initialize,
        worker.destroy,
        build_model,
        initialize,
    )
    slot["destroy"] = worker.destroy
    slot["destroy_code"] = slot["destroy"].__func__.__code__
    slot["destroy_defaults"] = deepcopy(
        (slot["destroy"].__func__.__defaults__, slot["destroy"].__func__.__kwdefaults__)
    )
    slot["worker_initialize"] = worker.initialize.__func__
    slot["ds_config"] = deepcopy(ds_config)
    result = _verify_initialization_capacity(slot, "before-build-model", subject)
    _receiver_require(ds_config == slot["ds_config"], "capacity verifier changed the native training recipe")
    _receiver_require(_worker_cleanup_matches(slot, worker), "capacity verifier changed worker cleanup")
    worker._ap_initialization_admission = slot
    return result


def _worker_cleanup_matches(slot, worker):
    destroy = getattr(worker, "destroy", None)
    captured = slot["destroy"]
    return (
        inspect.ismethod(destroy)
        and destroy.__self__ is worker
        and destroy.__func__ is captured.__func__
        and captured.__func__.__code__ is slot["destroy_code"]
        and (captured.__func__.__defaults__, captured.__func__.__kwdefaults__) == slot["destroy_defaults"]
    )


def maybe_verify_worker_engine(worker):
    slot = getattr(worker, "_ap_initialization_admission", None)
    if slot is None:
        _receiver_require(_initialization_request() is None, "selected engine has no armed pre-build admission")
        return None
    _receiver_require(
        slot["subject"]["worker"] is worker and slot["status"] == "armed", "actual armed worker required"
    )
    engine = worker.engine
    try:
        _guard_initialization(slot)
        _receiver_require(
            worker.initialize.__func__ is slot["worker_initialize"] and _worker_cleanup_matches(slot, worker),
            "captured worker lifecycle changed",
        )
        _receiver_require(slot["subject"]["ds_config"] == slot["ds_config"], "native training recipe changed")
        runtime_result = slot["runtime"].verify_engine(
            engine,
            sp_info=worker.get_sequence_parallel_info(),
            sp_group=worker.sp_group,
        )
        _receiver_require(
            type(runtime_result) is dict and runtime_result, "actual selected engine verification required"
        )
        result = _verify_initialization_capacity(
            slot,
            "after-engine",
            {**slot["subject"], "engine": engine},
        )
        _receiver_require(worker.engine is engine, "capacity verifier replaced the actual native engine")
        _receiver_require(slot["subject"]["ds_config"] == slot["ds_config"], "native training recipe changed")
        _receiver_require(_worker_cleanup_matches(slot, worker), "capacity verifier changed worker cleanup")
        slot["runtime_result"] = runtime_result
        slot["status"] = "verified"
        return result
    except BaseException as error:
        slot["status"] = "failed"
        # A source-file change does not invalidate still-owned cleanup code.
        if worker.engine is engine:
            try:
                _receiver_require(
                    _worker_cleanup_matches(slot, worker), "worker cleanup changed; owned cleanup not attempted"
                )
                slot["destroy"]()
            except BaseException as cleanup_error:
                error.add_note("owned worker initialization cleanup also failed: " + repr(cleanup_error))
        raise


def maybe_admit_sampler_construction(worker, args, constructor, engine_kwargs):
    selected = _initialization_request()
    if selected is None:
        return None
    _receiver_require(not hasattr(worker, "_ap_initialization_admission"), "sampler initialization is single-use")
    subject = {"worker": worker, "engine_args": args, "constructor": constructor, "engine_kwargs": engine_kwargs}
    slot = _initialization_slot(
        selected,
        subject,
        maybe_admit_sampler_construction,
        maybe_verify_sampler_construction,
        type(worker),
        worker.initialize,
        type(args),
        args.create_engine_config,
        constructor,
    )
    recipe = deepcopy(engine_kwargs)
    args_state = deepcopy(vars(args))
    result = _verify_initialization_capacity(slot, "before-sampler-constructor", subject)
    _receiver_require(
        engine_kwargs == recipe and vars(args) == args_state,
        "capacity verifier changed the native sampler recipe",
    )
    worker._ap_initialization_admission = slot
    return result


def maybe_verify_sampler_construction(worker):
    slot = getattr(worker, "_ap_initialization_admission", None)
    if slot is None:
        _receiver_require(_initialization_request() is None, "selected sampler has no armed constructor admission")
        return None
    _receiver_require(
        slot["subject"]["worker"] is worker and slot["status"] == "armed" and worker.llm is not None,
        "actual newly constructed sampler required",
    )
    _guard_initialization(slot)
    slot["status"] = "verified"
    return {"source_after_verified": True, "capacity_admitted": False, "numerical": None}


def selected_lora_request():
    """Read the caller-staged request, not a discovered or fabricated recipe."""
    path, expected = os.environ.get(REQUEST_ENV), os.environ.get(DIGEST_ENV)
    if path is None and expected is None:
        return None
    _receiver_require(enabled(), "pinned LoRA sync request requires explicit evidence opt-in")
    _receiver_require(
        type(path) is str
        and type(expected) is str
        and len(expected) == 64
        and all(character in "0123456789abcdef" for character in expected),
        "exact caller-staged request path and SHA256 required",
    )
    selected = Path(path)
    _receiver_require(
        selected.is_absolute()
        and selected.is_relative_to("/data-fast")
        and selected.is_file()
        and not selected.is_symlink()
        and selected.resolve(strict=True) == selected,
        "canonical regular FAST LoRA request required",
    )
    before = selected.stat()
    raw = selected.read_bytes()
    after = selected.stat()

    def identity(info):
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    _receiver_require(
        identity(before) == identity(after) and hashlib.sha256(raw).hexdigest() == expected,
        "caller-staged LoRA request changed",
    )
    request = json.loads(raw)
    _receiver_require(
        type(request) is dict
        and set(request) == {"schema", "training", "sampling"}
        and request["schema"] == "explicit-native-lora-sync-request-v1"
        and type(request["training"]) is dict
        and type(request["sampling"]) is dict
        and request["training"].get("admitted_sources") == request["sampling"].get("admitted_sources"),
        "exact shared source-bound training and sampling request required",
    )
    return request


def maybe_legacy_lora_replica_pool(pool_class):
    request = selected_lora_request()
    if request is None:
        from arctic_platform.debug.weight_sync import maybe_replica_pool

        return maybe_replica_pool(pool_class)

    return make_lora_replica_pool(pool_class, request["sampling"])


def maybe_prepare_lora_worker(worker, request):
    selected = selected_lora_request()
    _receiver_require(
        selected is not None and selected["training"] == request, "actual staged actor preparation request required"
    )
    return prepare_lora_worker(worker, request)


def send_prepared_lora_worker(worker):
    selected = selected_lora_request()
    _receiver_require(selected is not None, "staged LoRA request required for the debug worker")
    slot = getattr(worker, "_ap_owned_lora_preparation", None)
    _receiver_require(
        type(slot) is dict and slot["request"] == selected["training"],
        "native preparation differs from current staged request",
    )
    sender = worker._weight_sender
    binding = slot.get("sender_binding")
    _receiver_require(
        type(binding) is dict
        and binding["sender"] is sender
        and sender.group_id == 0
        and sender.targets == binding["targets"]
        and sender.ports == binding["ports"],
        "actual native sender targets or ports differ from captured initialization",
    )
    result = send_lora_worker(worker)
    _receiver_require(
        worker._weight_sender is sender and sender.targets == binding["targets"] and sender.ports == binding["ports"],
        "actual native sender routing changed during transfer",
    )
    return result


def maybe_lora_training_worker(worker_class):
    """Add debug RPCs before Ray registration only for an explicitly selected run."""
    if selected_lora_request() is None:
        return worker_class

    class LoRATrainingWorker(worker_class):
        def prepare_lora_weights(self, request):
            return maybe_prepare_lora_worker(self, request)

        def send_weights(self):
            return send_prepared_lora_worker(self)

    return LoRATrainingWorker


def maybe_record_lora_sender(worker, group, schedule, base_port):
    request = selected_lora_request()
    if request is None:
        return None
    slot = getattr(worker, "_ap_owned_lora_preparation", None)
    _receiver_require(
        type(slot) is dict and slot["request"] == request["training"] and worker.rank == 0,
        "actual prepared global-rank-zero actor required before sender initialization",
    )
    _receiver_require(
        group.group_id == 0
        and group.sender_train_rank == 0
        and group.targets == [(0, rank) for rank in range(8)]
        and schedule.training_gpus == 8
        and schedule.inference_tp == 8
        and schedule.inference_replicas == 1
        and type(base_port) is int
        and 0 < base_port <= 65528,
        "actual full8 sender schedule and finite native ports required",
    )
    sender = worker._weight_sender
    _receiver_require(
        sender is not None and "sender_binding" not in slot, "actual single initialized native sender required"
    )
    _bind_sources(
        request["training"]["admitted_sources"],
        maybe_record_lora_sender,
        type(sender),
        type(group),
        type(schedule),
        schedule.port_for_target,
    )
    targets = list(group.targets)
    ports = [schedule.port_for_target(replica, tp, base_port) for replica, tp in targets]
    _receiver_require(
        sender.group_id == group.group_id
        and sender.targets == targets
        and sender.ports == ports
        and ports == list(range(base_port, base_port + 8)),
        "actual initialized sender differs from the retained native target and port recipe",
    )
    slot["sender_binding"] = {"sender": sender, "targets": targets, "ports": ports}
    return {"sender_train_rank": 0, "targets": targets, "ports": ports, "training_gpus": 8, "inference_tp": 8}


def maybe_release_lora_worker(worker):
    slot = getattr(worker, "_ap_owned_lora_preparation", None)
    if slot is not None:
        return release_lora_worker(worker, slot)
    return None


def maybe_project_lora_transfer_schedule(native):
    request = selected_lora_request()
    if request is None:
        return native
    _receiver_require(
        native.training_sharding == "dp"
        and native.training_gpus == 8
        and native.inference_replicas == 1
        and native.inference_tp == 8,
        "actual original TRAIN8 and SAMPLE8 schedule geometry required",
    )
    _receiver_require(
        len(native.groups) == 8
        and [group.sender_train_rank for group in native.groups] == list(range(8))
        and all(group.world_size == 2 for group in native.groups),
        "actual native eight-sender default schedule required before LoRA projection",
    )
    targets = [target for group in native.groups for target in group.targets]
    _receiver_require(
        targets == [(0, rank) for rank in range(8)],
        "every native TP target must occur exactly once in retained native order",
    )
    origins = request["training"]["admitted_sources"]
    schedule_class = type(native)
    group_class = type(native.groups[0])
    _bind_sources(origins, maybe_project_lora_transfer_schedule, schedule_class.build, schedule_class, group_class)
    # A prepared actor-local list is replayable; the native sender consumes it for each target.
    native.groups = [group_class(group_id=0, sender_train_rank=0, targets=targets)]
    _receiver_require(
        native.training_gpus == 8
        and native.inference_tp == 8
        and native.inference_replicas == 1
        and native.groups[0].targets == targets
        and native.groups[0].world_size == 2,
        "projected sender policy must retain actual native allocation geometry and targets",
    )
    _bind_sources(origins, maybe_project_lora_transfer_schedule, schedule_class, group_class)
    return native


async def maybe_prepare_lora_training(server, workers, pool, schedule):
    request = selected_lora_request()
    if request is None:
        return None
    _receiver_require(
        server.weight_sync_ready is False
        and not getattr(server, "_ap_lora_prepare_attempted", False)
        and not hasattr(server, "_ap_lora_training_preparation"),
        "native LoRA transfer is single-use; no unchanged collective or target retry",
    )
    _receiver_require(
        pool.num_replicas == 1
        and pool.tp_size == 8
        and len(schedule.groups) == 1
        and schedule.groups[0].sender_train_rank == 0
        and schedule.groups[0].world_size == 2
        and schedule.groups[0].replica_ids == [0],
        "actual original one-replica SAMPLE8 rank-zero pairwise schedule required",
    )
    origins = request["training"]["admitted_sources"]
    _bind_sources(origins, maybe_prepare_lora_training, type(server), type(pool), pool.sync_weights, type(schedule))
    server._ap_lora_prepare_attempted = True
    prepared = await prepare_lora_training_workers(workers, request["training"])
    _receiver_require(selected_lora_request() == request, "staged request changed during native preparation")
    _bind_sources(origins, maybe_prepare_lora_training, type(server), type(pool), pool.sync_weights, type(schedule))
    server._ap_lora_training_preparation = prepared
    return prepared


def maybe_lora_max_param_bytes(server, native_max):
    if selected_lora_request() is None:
        return native_max
    prepared = getattr(server, "_ap_lora_training_preparation", None)
    _receiver_require(
        type(prepared) is dict and prepared["all_training_ranks_prepared"] is True, "all8 preparation required"
    )
    manifest = prepared["rank_preparations"][0].get("emitted_manifest")
    _receiver_require(type(manifest) is dict and manifest, "actual retained rank-zero adapter manifest required")
    widths = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8}
    sizes = []
    for value in manifest.values():
        _receiver_require(
            type(value) is dict
            and value.get("dtype") in widths
            and type(value.get("numel")) is int
            and value["numel"] > 0,
            "actual positive floating adapter tensor geometry required",
        )
        sizes.append(value["numel"] * widths[value["dtype"]])
    return max(sizes)


def maybe_lora_sync_result(server, result, received, sent):
    request = selected_lora_request()
    if request is None:
        return {**result, **weight_sync.maybe_sync_weights_result(received, sent)}
    _receiver_require(
        len(sent) == 1
        and type(sent[0]) is dict
        and sent[0].get("weight_format") == "lora"
        and type(received) is dict
        and type(received.get("workers")) is list
        and len(received["workers"]) == 1,
        "actual single native LoRA sender and sampler receipt required",
    )
    worker = received["workers"][0]
    _receiver_require(
        type(worker) is dict
        and worker.get("weight_format") == "lora"
        and type(worker.get("all_rank_adapter_installation")) is list
        and len(worker["all_rank_adapter_installation"]) == 8
        and worker.get("adapter_storage_verified") is False
        and worker.get("adapter_application_verified") is False
        and worker.get("full_parameter_coverage") is False
        and worker.get("capacity_admitted") is False,
        "allTP adapter installation is not storage/application/full-model/capacity admission",
    )
    _receiver_require(selected_lora_request() == request, "staged request changed during native transfer")
    _bind_sources(request["training"]["admitted_sources"], maybe_lora_sync_result)
    return {
        **result,
        "weight_format": "lora",
        "targets": [{"send": sent, "recv": received}],
        "training_preparation": server._ap_lora_training_preparation,
        "adapter_storage_verified": False,
        "adapter_application_verified": False,
        "full_parameter_coverage": False,
        "capacity_admitted": False,
        "parameter_only_judge_admitted": False,
        "numerical": None,
    }


# Routing


def _pointer():
    path, digest = os.environ.get(PATH_ENV), os.environ.get(SHA_ENV)
    if path is None and digest is None:
        return None
    _receiver_require(
        type(path) is str and type(digest) is str and re.fullmatch("[0-9a-f]{64}", digest) is not None,
        "selected bootstrap requires both exact path and SHA256",
    )
    root = Path(path)
    _receiver_require(
        root.is_absolute()
        and root.is_relative_to("/data-fast")
        and root.is_file()
        and not root.is_symlink()
        and root.resolve(strict=True) == root,
        "selected bootstrap requires a canonical regular FAST path",
    )
    return {"path": path, "sha256": digest}


def maybe_lora_replica_pool(pool_class):
    """Preserve the existing route unless an exact selected bootstrap is staged."""
    reader, maker = selected_lora_request, make_selected_lora_replica_pool
    request = reader()
    pointer = _pointer()
    if pointer is None:
        return maybe_legacy_lora_replica_pool(pool_class)
    _receiver_require(request is not None, "selected bootstrap requires its staged LoRA request")
    origins = deepcopy(request["sampling"]["admitted_sources"])
    callbacks = (maybe_lora_replica_pool, _pointer, reader, maker)
    _bind_sources(origins, *callbacks)
    original = deepcopy(request)
    result = maker(pool_class, deepcopy(request["sampling"]), dict(pointer))
    _receiver_require(
        selected_lora_request is reader
        and make_selected_lora_replica_pool is maker
        and reader() == original
        and _pointer() == pointer,
        "selected LoRA request or bootstrap pointer changed during pool selection",
    )
    _bind_sources(origins, *callbacks)
    return result


# Staging

CONFIG_ID = "glm5.3-flash-h200-train-rl-lora-16gpus-sp8-256k"
MODEL = "zai-org/GLM-5.3-Flash"
PREPARE_MODULE = "arctic_platform.debug.lora"
PREPARE_NAME = "prepare_selected_lora_receiver"


def _staging_require(value, message):
    if not value:
        raise ValueError(message)


def _private_directory(path):
    path = Path(path)
    _staging_require(
        path.is_absolute()
        and path.is_relative_to("/data-fast")
        and path.resolve(strict=True) == path
        and path.is_dir()
        and not path.is_symlink()
        and stat.S_IMODE(path.stat().st_mode) == 0o700,
        "canonical existing private FAST staging directory required",
    )
    return path


def _new_directory(path):
    path.mkdir(mode=0o700, exist_ok=False)
    path.chmod(0o700)
    return _private_directory(path)


def _write_json(path, body):
    raw = (json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _staging_require(
        path.read_bytes() == raw and stat.S_IMODE(path.stat().st_mode) == 0o600,
        "private staged bytes changed",
    )
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _config(cfg, sources):
    _staging_require(
        cfg.config_id == CONFIG_ID
        and cfg.is_rl is True
        and cfg.model_name == cfg.native.model_name == MODEL
        and cfg.n_gpus == cfg.native.training_gpus == cfg.native.sampling_gpus == 8
        and cfg.native.log_prob_gpus == 0
        and cfg.training["sp_size"] == cfg.training["ep_size"] == 8
        and cfg.training["peft_config"]["r"] == 32,
        "actual canonical LoRA TRAIN8+SAMPLE8 LoadedConfig required",
    )
    config_path = str(Path(cfg.path).resolve(strict=True))
    _staging_require(config_path in sources, "actual LoadedConfig bytes must be source-bound")
    verify_sources(sources)
    return deepcopy(cfg.training["peft_config"])


def stage_selected_lora_request(
    cfg,
    *,
    directory,
    receiver,
    admitted_sources,
    adapter_id,
    adapter_name,
    max_weight_bytes,
    max_received_bytes,
    publication_bounds,
    distribution_names,
):
    """Return pinned environment after caller-owned inputs are staged exactly once.

    Byte limits are attempt limits, not a constructor or transient-capacity proof.
    The useful caller still verifies these exact files on every consuming node.
    """
    _staging_require(enabled(), "explicit LoRA staging opt-in required")
    sources = deepcopy(admitted_sources)
    _staging_require(type(sources) is dict and sources, "actual selected source closure required")
    observed = source_files(stage_selected_lora_request, bootstrap_descriptor)
    _staging_require(
        observed.items() <= sources.items(),
        "staging/bootstrap sources must be admitted",
    )
    lora_config = _config(cfg, sources)
    receiver = deepcopy(receiver)
    _staging_require(
        type(receiver) is dict
        and set(receiver) == {"module", "name", "source"}
        and type(receiver["module"]) is str
        and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", receiver["module"])
        and type(receiver["name"]) is str
        and re.fullmatch(r"[A-Za-z_]\w*", receiver["name"])
        and type(receiver["source"]) is str
        and receiver["source"] in sources,
        "explicit selected native receiver descriptor required",
    )
    _staging_require(
        type(adapter_id) is int
        and adapter_id > 0
        and type(adapter_name) is str
        and adapter_name
        and all(type(value) is int and value > 0 for value in (max_weight_bytes, max_received_bytes)),
        "positive owned adapter identity and explicit byte limits required",
    )
    _staging_require(
        max_received_bytes >= max_weight_bytes,
        "receive byte limit cannot be smaller than prepared byte limit",
    )
    bounds = deepcopy(publication_bounds)
    _staging_require(
        type(bounds) is dict
        and set(bounds) == set(BOUND_KEYS)
        and all(type(value) is int and value > 0 for value in bounds.values()),
        "exact explicit positive private-publication bounds required",
    )
    _staging_require(
        bounds["max_total_source_bytes"] >= bounds["max_source_bytes"],
        "total publication limit cannot be smaller than one source limit",
    )
    names = deepcopy(distribution_names)
    _staging_require(
        type(names) is list
        and names
        and all(type(name) is str and name for name in names)
        and len(names) == len(set(names)),
        "exact distinct caller-selected distributions required",
    )
    directory = _private_directory(directory)
    _staging_require(not any(directory.iterdir()), "staging directory must be fresh and empty")
    prepare_source = str(Path(__file__).with_name("lora.py").resolve(strict=True))
    _staging_require(prepare_source in sources, "actual preparation callback source must be admitted")
    verify_sources(sources)
    publication = _new_directory(directory / "callback-sources")
    settings = _write_json(
        directory / PREPARATION_FILENAME,
        {
            "schema": "job3-selected-lora-preparation-v1",
            "publication_root": str(publication),
            "distribution_names": names,
            **bounds,
        },
    )
    _staging_require(settings["path"] not in sources, "cannot replace an admitted settings identity")
    sources[settings["path"]] = settings["sha256"]
    bootstrap_body = {
        "schema": "job3-selected-lora-bootstrap-v1",
        "receiver": receiver,
        "prepare": {
            "module": PREPARE_MODULE,
            "name": PREPARE_NAME,
            "source": prepare_source,
        },
        "admitted_sources": sources,
    }
    bootstrap = _write_json(directory / "bootstrap.json", bootstrap_body)
    _staging_require(bootstrap_descriptor(bootstrap) == bootstrap_body, "staged bootstrap rejected")
    request_body = {
        "schema": "explicit-native-lora-sync-request-v1",
        "training": {"max_weight_bytes": max_weight_bytes, "admitted_sources": sources},
        "sampling": {
            "lora_int_id": adapter_id,
            "lora_name": adapter_name,
            "lora_config": lora_config,
            "max_received_bytes": max_received_bytes,
            "admitted_sources": sources,
        },
    }
    request = _write_json(directory / "request.json", request_body)
    verify_sources(sources)
    return {
        "schema": "job3-selected-lora-staged-inputs-v1",
        "settings": settings,
        "bootstrap": bootstrap,
        "request": request,
        "admitted_sources": sources,
        "environment": {
            VERIFY_ENV: "1",
            REQUEST_ENV: request["path"],
            DIGEST_ENV: request["sha256"],
            PATH_ENV: bootstrap["path"],
            SHA_ENV: bootstrap["sha256"],
        },
        "training_gpus": 8,
        "sampling_gpus": 8,
        "limits_are_capacity_proof": False,
        "runtime_admitted": False,
        "capacity_admitted": False,
        "native_engine_constructed": False,
        "numerical": None,
    }


def __getattr__(name):
    if name == "LoRAWeightSyncExtension":
        return receiver_extension()
    if name == "SelectedLoRAWeightSyncExtension":
        return _selected_extension_attribute(name)
    raise AttributeError(name)
