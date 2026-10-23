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

"""Bind every received adapter factor to one actual native builder result.

The wrapper calls the existing builder once. Name projections use its actual AP
helpers and the native parser/rename mapper, not fabricated native tensor names.
This is a pre-installation binding, not TP storage or serving application proof.
"""

from __future__ import annotations

import hashlib
import json

from arctic_platform.debug.lora_adapter_storage_evidence import ObservedLoRAOperation
from arctic_platform.debug.weight_sync_evidence import enabled
from arctic_platform.debug.weight_sync_evidence import source_files
from arctic_platform.debug.weight_sync_evidence import tensor_manifest
from arctic_platform.debug.weight_sync_evidence import verify_sources

NATIVE_MODEL_SHA256 = "57d0c10b374a29e2b4fda826dfd9657712394c97ddf6a8099783f718ae2860a4"
NATIVE_PARSER_SHA256 = "49e8ff84a4725c963f685b4828eec1f70066969b3e46ca65c0be784df692a83f"
NATIVE_MAPPER_SHA256 = "7171b6347b3cbab213df299fa317094ac6c9b2b397daf6a2ec93ff3a152b0e75"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def _source_binding(admitted_sources, *objects):
    _require(enabled(), "adapter tensor binding must be explicitly enabled")
    observed = source_files(_source_binding, ObservedLoRAOperation, *objects)
    _require(
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
        _require(
            type(name) is str and name.endswith((".lora_A.weight", ".lora_B.weight")),
            "only original received A/B weight names are supported",
        )
        rewritten = remap_key(name, prefix)
        _require(type(rewritten) is str and rewritten and rewritten not in remapped, "AP name remapping collided")
        remapped.add(rewritten)
        parsed = parse_name(rewritten, mapper)
        _require(
            type(parsed) is tuple
            and len(parsed) == 2
            and type(parsed[0]) is str
            and parsed[0]
            and type(parsed[1]) is bool,
            "actual native parser did not return a module name and factor kind",
        )
        module, is_a = parsed
        kind = "A" if is_a else "B"
        _require((module, kind) not in occupied, "native parser/mapper collided on an adapter factor")
        occupied.add((module, kind))
        projected[name] = {"rewritten": rewritten, "module": module, "kind": kind}
    return projected


def _expected_modules(projected):
    modules, groups = {}, {}
    for wire_name, entry in projected.items():
        modules.setdefault(entry["module"], {})[entry["kind"]] = wire_name
    _require(all(set(parts) == {"A", "B"} for parts in modules.values()), "every native module requires both A and B")
    order = list(modules)
    for name in modules:
        for index in (1, 2, 3):
            if name == f"experts.w{index}" or name.endswith(f".experts.w{index}"):
                group = name[:-3]
                groups.setdefault(group, {})[index] = name
                break
    expected = {name: {"packed": False, "parts": [parts]} for name, parts in modules.items()}
    for group, parts in groups.items():
        _require(set(parts) == {1, 2, 3}, "native stacked expert group requires exactly w1/w2/w3")
        _require(group not in expected, "packed expert module collides with a received module")
        projections = [parts[index] for index in (1, 2, 3)]
        expected[group] = {"packed": True, "parts": [modules[name] for name in projections]}
        order.append(group)
        for name in projections:
            del expected[name]
            order.remove(name)
    return expected, order


def _factor_bytes(tensors, dtype, maximum):
    import torch

    _require(type(maximum) is int and maximum > 0, "explicit positive received-factor byte bound required")
    total = 0
    for tensor in tensors.values():
        _require(
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
        _require(total <= maximum, "received adapter factors exceed the explicit binding byte limit")
    _require(
        all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors.values()),
        "received adapter factors must be finite",
    )
    return total


def _verify_built(result, *, expected, order, manifests, rank, alpha, dtype, identity):
    import torch

    _require(
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
        _require(
            weights.module_name == name and type(weights.rank) is int and weights.rank == rank,
            "native factor module name/rank differs",
        )
        if domain["packed"]:
            _require(
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
            _require(
                weights.lora_alpha == alpha and weights.scaling == scaling,
                "native dense alpha/scaling differs from the exact builder recipe",
            )
            a, b = [weights.lora_a], [weights.lora_b]
        for index, parts in enumerate(domain["parts"]):
            for kind, tensor in (("A", a[index]), ("B", b[index])):
                _require(
                    isinstance(tensor, torch.Tensor)
                    and tensor.device.type == "cpu"
                    and tensor.dtype == dtype
                    and tensor.ndim == (3 if domain["packed"] else 2)
                    and tensor.shape[-2 if kind == "A" else -1] == rank,
                    "native factor shape/rank/dtype/device differs; no transpose or cast is inferred",
                )
                wire_name = parts[kind]
                actual = tensor_manifest(tensor)
                _require(
                    actual == manifests[wire_name],
                    "native built factor bytes differ from the original received tensor",
                )
                bound[wire_name] = {
                    "native_module": name,
                    "factor_kind": kind,
                    "packed_projection": f"w{index + 1}" if domain["packed"] else None,
                    "native_factor": actual,
                }
    _require(bound.keys() == manifests.keys(), "received adapter factors were ignored or duplicated")
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

    _require(type(tensors) is dict and tensors, "genuine retained received tensor dictionary required")
    _require(type(lora_int_id) is int and lora_int_id > 0, "actual positive adapter identity required")
    _require(
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
    _require(
        (device == "cpu" or isinstance(device, torch.device) and device.type == "cpu" and device.index is None)
        and dtype in (torch.bfloat16, torch.float16, torch.float32),
        "actual CPU builder and explicit floating dtype required",
    )
    globals_ = getattr(builder, "__globals__", {})
    packing = globals_.get("_pack_fused_expert_loras")
    constructor = native_model_class.from_lora_tensors
    constructor_globals = constructor.__func__.__globals__
    _require(
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
        _require(symbol in constructor_globals, "native constructor factor/config source binding is absent")
        constructor_symbols[symbol] = constructor_globals[symbol]
        selected.append(constructor_symbols[symbol])
    if model is not None:
        selected.append(type(model))
        if hasattr(model, "named_modules"):
            selected.append(model.named_modules)
    origins = _source_binding(admitted_sources, *selected)
    config_sha256 = _config_digest(lora_config)
    prefix = served_prefix_for_model(model)
    _require(type(prefix) is str and prefix, "actual served model prefix required")
    mapper = root_mapper.get_rename_mapper() if root_mapper is not None else None
    mapper_calls = _mapper_calls(mapper)
    if mapper is not None:
        origins = _source_binding(origins, *mapper_calls)
        _require(not mapper.orig_to_new_stacked, "actual LoRA rename mapper cannot retain stacked weight sharding")
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
    _require(type(result) is native_model_class, "native builder returned a different adapter class")
    _require(
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
    _require(
        [getattr(call, "__func__", call) for call in current_calls]
        == [getattr(call, "__func__", call) for call in mapper_calls],
        "actual rename mapper selected different implementations during native build",
    )
    _source_binding(origins, *current_calls)
    _require(
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
