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

"""Byte evidence around actual adapter transforms and individual storage writes.

These operations do not prove a complete native layer inventory, tensor-parallel
slot routing, or kernel application. The caller must derive that binding from
the selected native sources. No native Python call is replayed or monkeypatched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from arctic_platform.debug.weight_sync_evidence import enabled
from arctic_platform.debug.weight_sync_evidence import source_files
from arctic_platform.debug.weight_sync_evidence import tensor_manifest
from arctic_platform.debug.weight_sync_evidence import verify_sources

NATIVE_WEIGHTS_SHA256 = "767c82ea9d3a302f14046af43c638d41389a6b5136b02b08ead6ab9128b9191c"


@dataclass(frozen=True)
class ObservedLoRAOperation:
    value: object
    evidence: dict


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sources(admitted_sources, *objects):
    _require(enabled(), "LoRA storage evidence must be explicitly enabled")
    observed = source_files(_sources, *objects)
    _require(
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

    _require(
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

    _require(
        all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors),
        "adapter evidence requires finite input tensors",
    )


def _budget(total, maximum):
    _require(type(maximum) is int and 0 < total <= maximum, "private adapter shadow bytes exceed the explicit limit")


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
    _require(
        packed or (isinstance(weights.lora_a, torch.Tensor) and isinstance(weights.lora_b, torch.Tensor)),
        "only actual dense factors or ordered packed factor lists are supported",
    )
    a = weights.lora_a if packed else [weights.lora_a]
    b = weights.lora_b if packed else [weights.lora_b]
    scales = weights.scaling if packed else [weights.scaling]
    _require(
        type(scales) is list
        and len(a) == len(b) == len(scales)
        and a
        and all(type(scale) in (float, int) and math.isfinite(scale) for scale in scales),
        "complete finite scaling and A/B factors are required; missing factors are unsupported",
    )
    _require(type(weights.rank) is int and weights.rank > 0, "actual positive native adapter rank required")
    total, storages, values = 0, set(), {}
    for index, (left, right) in enumerate(zip(a, b)):
        for kind, tensor in (("A", left), ("B", right)):
            total += _full_tensor(tensor)
            key = _storage(tensor)
            _require(key not in storages, "shared factor storage requires a separate source-derived transform")
            storages.add(key)
            _require(tensor.ndim in (2, 3), "only matrix or stacked-expert factors are supported")
            rank_axis = -2 if kind == "A" else -1
            _require(
                tensor.shape[rank_axis] == weights.rank,
                "factor rank differs from the actual native adapter rank",
            )
            values[f"{index}.{kind}"] = tensor
    return values, list(scales), total


def observe_stacked_moe_packing(loras, module_name, *, pack, max_shadow_bytes, admitted_sources):
    """Invoke the selected stacked pack once and verify ordered tensor aliases."""
    _require(type(loras) in (list, tuple) and len(loras) == 3, "exactly ordered w1/w2/w3 weights required")
    _require(type(module_name) is str and module_name, "actual native packed module name required")
    _require(
        getattr(pack, "__name__", None) == "pack_moe_stacked" and isinstance(getattr(pack, "__self__", None), type),
        "the selected native stacked classmethod is required",
    )
    origins = _sources(admitted_sources, pack, *[type(weights) for weights in loras])
    factors, before, total = {}, [], 0
    rank, alpha, scaling = loras[0].rank, loras[0].lora_alpha, loras[0].scaling
    for index, weights in enumerate(loras):
        _require(
            weights.module_name == f"{module_name}.w{index + 1}"
            and weights.rank == rank
            and weights.lora_alpha == alpha
            and weights.scaling == scaling,
            "stacked native projection order, rank, alpha and scaling must agree",
        )
        values, scales, size = _factors(weights)
        _require(len(scales) == 1 and all(tensor.ndim == 3 for tensor in values.values()), "stacked factors required")
        total += size
        _budget(total, max_shadow_bytes)
        _finite_tensors(values.values())
        factors.update({f"w{index + 1}.{kind[-1]}": tensor for kind, tensor in values.items()})
        before.append({name: tensor_manifest(tensor) for name, tensor in values.items()})
    _budget(total, max_shadow_bytes)
    _require(len({_storage(tensor) for tensor in factors.values()}) == 6, "cross-projection aliases are unsupported")
    _require(
        getattr(pack.__self__, "pack_moe_stacked") == pack,
        "selected native stacked classmethod changed before invocation",
    )
    result = pack(loras, module_name)
    verify_sources(origins)
    _require(
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
        _require(
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
        _require(
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
    _require(
        result is weights
        and weights.optimize == optimize
        and (weights.module_name, weights.rank, weights.lora_alpha) == metadata
        and (packed_alphas is None or weights.lora_alphas == packed_alphas),
        "native optimize replaced its object, metadata or selected method",
    )
    current, after_scales, after_bytes = _factors(weights)
    _finite_tensors(current.values())
    _require(after_bytes == total and all(scale == 1 for scale in after_scales), "native optimize metadata differs")
    _require(
        current.keys() == factors.keys()
        and all(
            (id(tensor), _storage(tensor), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype) == layouts[name]
            for name, tensor in current.items()
        ),
        "native optimize changed factor identities, layouts or storage",
    )
    after = {name: tensor_manifest(tensor) for name, tensor in current.items()}
    _require(
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
    _require(
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
        _require(len(matches) == 1, "actual storage attribute is absent or ambiguous")
        value = matches[0]
    for index in path[1:]:
        _require(type(value) in (tuple, list) and index < len(value), "actual storage list index is absent")
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
    _require(
        owner is not None
        and not isinstance(owner, type)
        and getattr(setter, "__name__", None) == "set_lora"
        and owner.set_lora == setter,
        "actual bound native layer set_lora method required",
    )
    origins = _sources(admitted_sources, setter, TorchDispatchMode, tree_flatten, tree_map)
    _require(
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
    _require(len(keys) == len(seeds), "aliased factors or destination allocations require explicit supported mapping")
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
            _require(func.namespace in {"aten", "prims"}, f"unobservable native adapter operation: {func}")
            _require(
                torch.Tag.nondeterministic_seeded not in func.tags,
                "random adapter operations cannot be mirrored as deterministic loading",
            )
            flat, _ = tree_flatten((args, kwargs))
            tensors = [value for value in flat if isinstance(value, torch.Tensor)]
            input_keys = {_storage(tensor) for tensor in tensors}
            _require(input_keys <= live_storages, "native setter read an unbound tensor storage")
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
                    _require(key not in factor_keys, "native setter cannot overwrite bound adapter factors")
                    _require(key in live_storages, "native setter wrote an unowned tensor storage")
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
                _require(tensor.dtype == dtype_by_key[key], "native setter reinterpreted tracked storage dtype")
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
            _require(len(actual) == len(mirrored), "operator mirror return structure differs")
            for tensor, mirror in zip(actual, mirrored):
                _require(isinstance(tensor, torch.Tensor) == isinstance(mirror, torch.Tensor), "mirror type differs")
                if not isinstance(tensor, torch.Tensor):
                    _require(type(tensor) is type(mirror) and tensor == mirror, "operator mirror scalar differs")
                    continue
                key = _storage(tensor)
                _require(
                    tensor.shape == mirror.shape
                    and tensor.stride() == mirror.stride()
                    and tensor.dtype == mirror.dtype
                    and tensor.storage_offset() == mirror.storage_offset(),
                    "operator mirror tensor layout differs",
                )
                if key not in live_storages:
                    _require(
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
                    _require(_storage(mirror) == _storage(shadows[key]), "operator mirror storage alias differs")
            for key in write_keys:
                dependencies[key].update(inherited)
            for entry in pending:
                consumed.update(entry["factor_dependencies"])
            writes.extend(pending)
            return result

    with torch.no_grad(), Observe():
        result = setter(*args, **kwargs)
    verify_sources(origins)
    _require(
        owner.set_lora == setter
        and all(
            _storage_tensor(owner, paths[name]) is tensor
            and (id(tensor), _storage(tensor), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
            == layouts[name]
            for name, tensor in destinations.items()
        ),
        "native setter changed its callable or actual storage identities/layouts",
    )
    _require(
        initial == {name: tensor_manifest(tensor) for name, tensor in factor_tensors.items()},
        "bound adapter factor bytes changed during native setter",
    )
    _require(writes and consumed == factor_tensors.keys(), "not every bound adapter factor reaches an observed write")
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
        _require(
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
