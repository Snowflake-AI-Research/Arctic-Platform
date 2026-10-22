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


"Local gateway optimizer snapshots and source-bound ZeRO2 decoding."

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from arctic_platform.client import CortexConfig

from .dss_driver import ArcticJob
from .dss_driver import GatewaySession
from .dss_driver import StepResult
from .dss_driver import fwd_bwd_step
from .hosted import require_local_cortex_gateway
from .hosted import save_and_resolve_checkpoint_id
from .storage import require_data_fast_path

# Only these upstream checkpoint schema classes may extend torch's restricted loader.
_SCHEMA_GLOBALS = {
    "deepspeed.utils.tensor_fragment.fragment_address",
    "deepspeed.runtime.fp16.loss_scaler.LossScaler",
    "deepspeed.runtime.fp16.loss_scaler.DynamicLossScaler",
    "deepspeed.runtime.zero.config.ZeroStageEnum",
}
_OPTIM_FILE = re.compile(r"(?:bf16_|fp16_)?zero_pp_rank_(\d+)_mp_rank_00_optim_states\.pt")
_NATIVE_OPTIM_FILE = re.compile(r"bf16_zero_pp_rank_(\d+)_mp_rank_(\d+)_optim_states\.pt")
_NATIVE_SOURCE_ROLES = {
    "expert_checkpointer",
    "base_checkpointer",
    "training",
    "zero",
    "zero_base",
    "engine",
}


def _native_sources(layout: dict) -> None:
    sources = layout.get("source_files")
    if not isinstance(sources, dict) or set(sources) != _NATIVE_SOURCE_ROLES:
        raise ValueError("native layout requires the complete selected producer source files")
    for descriptor in sources.values():
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
            raise ValueError("invalid native producer source descriptor")
        path, expected = descriptor["path"], descriptor["sha256"]
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or Path(path).is_symlink()
            or not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        ):
            raise ValueError("invalid native producer source descriptor")
        with Path(path).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("native producer source drift")


def _native_layout(layout: dict) -> dict:
    if not isinstance(layout, dict):
        raise ValueError("native layout must be a dictionary")
    # Own a plain immutable-by-comparison copy, not caller-owned mutable topology.
    layout = json.loads(json.dumps(layout))
    if layout.get("schema") != "native-ep-zero2-layout-v1":
        raise ValueError("unsupported native optimizer layout schema")
    for key in ("world_size", "ep_size", "sp_size", "model_mp_world_size"):
        if type(layout.get(key)) is not int or layout[key] < 1:
            raise ValueError(f"native layout requires a positive {key}")
    world, ep, sp = (layout[key] for key in ("world_size", "ep_size", "sp_size"))
    if ep not in (8, 16) or world % ep or world % sp or layout["model_mp_world_size"] not in (1, sp):
        raise ValueError("unsupported native EP/SP topology")
    if type(layout.get("model_ep_rank")) is not int or not 0 <= layout["model_ep_rank"] < ep:
        raise ValueError("native layout requires the model writer's actual EP rank")
    mp_ranks = _sequence(layout.get("optimizer_mp_ranks"), world, "native optimizer MP ranks")
    if any(type(rank) is not int or rank < 0 or rank >= sp for rank in mp_ranks):
        raise ValueError("invalid native optimizer MP ranks")
    groups = layout.get("expert_data_parallel_groups")
    if not isinstance(groups, dict):
        raise ValueError("native layout requires actual ordered expert DP groups")
    for name, routes in groups.items():
        if not isinstance(name, str) or not name:
            raise ValueError("invalid expert optimizer group name")
        routes = _sequence(routes, ep, "expert DP routes")
        ranks = []
        for route in routes:
            route = _sequence(route, world // ep, "expert DP rank order")
            if any(type(rank) is not int or rank < 0 or rank >= world for rank in route):
                raise ValueError("invalid expert DP rank order")
            ranks.extend(route)
        if sorted(ranks) != list(range(world)):
            raise ValueError("expert DP routes must cover every rank exactly once")
        if 0 not in routes[layout["model_ep_rank"]]:
            raise ValueError("expert DP routes disagree with the model writer EP rank")
    config = layout.get("ds_config")
    zero = config.get("zero_optimization") if isinstance(config, dict) else None
    if not isinstance(zero, dict) or type(zero.get("stage")) is not int or zero["stage"] != 2:
        raise ValueError("native layout requires ZeRO2 ds_config")
    bf16 = config.get("bf16")
    if not isinstance(bf16, dict) or any(
        bf16.get(key) is not True for key in ("enabled", "bf16_master_weights_and_grads", "bf16_optimizer_states")
    ):
        raise ValueError("native route requires the original BF16 master/state recipe")
    tp = config.get("tensor_parallel", {})
    if tp and (not isinstance(tp, dict) or tp.get("tp_size", 1) != 1):
        raise ValueError("native tensor parallel checkpoints are unsupported")
    optimizer = config.get("optimizer")
    if (
        not isinstance(optimizer, dict)
        or not isinstance(optimizer.get("type"), str)
        or optimizer["type"].strip().lower() != "adamw"
    ):
        raise ValueError("native optimizer requires AdamW")
    _native_sources(layout)
    return layout


def _native_checkpoint_files(root: Path, layout: dict) -> tuple[Path, list[Path], list[Path]]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("checkpoint root must be a local directory")
    models = list(root.rglob("*_model_states.pt"))
    if len(models) != 1 or models[0].name != "mp_rank_00_model_states.pt":
        raise ValueError("native checkpoint requires exactly one replicated mp_rank_00 model")
    model = models[0]
    optimizers, experts = {}, {}
    for index, path in enumerate(root.rglob("*")):
        if index >= 4096:
            raise ValueError("native checkpoint inventory is too large")
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("native checkpoint contains symlinks or escaped entries")
        if not path.is_dir() and not path.is_file():
            raise ValueError("native checkpoint contains nonregular entries")
        if not path.is_file() or path.suffix != ".pt" or path == model:
            continue
        if path.parent != model.parent or path.stat().st_size == 0:
            raise ValueError("native checkpoint contains empty or nested tensor files")
        match = _NATIVE_OPTIM_FILE.fullmatch(path.name)
        if match:
            rank, mp = map(int, match.groups())
            if rank >= layout["world_size"] or rank in optimizers or mp != layout["optimizer_mp_ranks"][rank]:
                raise ValueError("native optimizer rank inventory or MP filename mismatch")
            optimizers[rank] = path
            continue
        match = re.fullmatch(r"experts_ep_rank_(\d{3})\.pt", path.name)
        if not match or int(match.group(1)) in experts:
            raise ValueError(f"unsupported native checkpoint tensor file: {path.name}")
        experts[int(match.group(1))] = path
    if sorted(optimizers) != list(range(layout["world_size"])) or sorted(experts) != list(range(layout["ep_size"])):
        raise ValueError("native checkpoint requires every optimizer and expert rank")
    return model, [optimizers[rank] for rank in sorted(optimizers)], [experts[rank] for rank in sorted(experts)]


def _checkpoint_hashes(paths: list[Path]) -> dict[str, str]:
    hashes = {}
    for path in paths:
        with path.open("rb") as stream:
            hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


def _native_input_hashes(root: Path) -> dict[str, str]:
    return _checkpoint_hashes(sorted(path for path in root.rglob("*") if path.is_file()))


def _load(path: Path) -> dict:
    globals_needed = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    if globals_needed - _SCHEMA_GLOBALS:
        raise ValueError(f"unsupported checkpoint globals in {path.name}: {sorted(globals_needed - _SCHEMA_GLOBALS)}")
    allowed = []
    for name in sorted(globals_needed):
        module, _, attribute = name.rpartition(".")
        allowed.append(getattr(importlib.import_module(module), attribute))
    with torch.serialization.safe_globals(allowed):
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint {path.name} must contain a dictionary")
    return state


def _integer(value, description: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.is_complex() or value.dtype == torch.bool:
            raise ValueError(f"{description} must be a nonnegative integer")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{description} must be a nonnegative integer")
    if int(value) != value or value < 0:
        raise ValueError(f"{description} must be a nonnegative integer")
    return int(value)


def _sequence(value, count: int, description: str) -> list:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{description} must contain {count} groups")
    return value


def _fp32_vector(value, length: int, description: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.ndim != 1
        or value.numel() != length
        or value.layout != torch.strided
    ):
        raise ValueError(f"{description} must be a dense FP32 vector of length {length}")
    return value


def _bf16_vector(value, length: int, description: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.bfloat16
        or value.ndim != 1
        or value.numel() != length
        or value.layout != torch.strided
    ):
        raise ValueError(f"{description} must be a dense BF16 vector of length {length}")
    return value


def _config(config, description: str) -> None:
    if not isinstance(config, dict):
        raise ValueError(f"{description} requires ds_config")
    zero = config.get("zero_optimization", {})
    if not isinstance(zero, dict) or _integer(zero.get("stage"), "configured ZeRO stage") != 2:
        raise ValueError("only dense ZeRO2 checkpoints are supported")
    optimizer = config.get("optimizer")
    if (
        not isinstance(optimizer, dict)
        or not isinstance(optimizer.get("type"), str)
        or optimizer["type"].strip().lower() != "adamw"
    ):
        raise ValueError(f"{description} requires optimizer type AdamW")
    ep = config.get("expert_parallel", {})
    if ep and (
        not isinstance(ep, dict)
        or ep.get("enabled", False)
        or ep.get("ep_size", 1) != 1
        or ep.get("autoep_size", 1) != 1
    ):
        raise ValueError("expert parallel checkpoints are unsupported")
    if config.get("moe"):
        raise ValueError("expert parallel checkpoints are unsupported")
    tp = config.get("tensor_parallel", {})
    if tp and (not isinstance(tp, dict) or tp.get("tp_size", 1) != 1):
        raise ValueError("tensor parallel checkpoints are unsupported")


def _checkpoint_files(root: Path) -> tuple[Path, list[Path]]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("checkpoint root must be a local directory")
    models = list(root.rglob("*_model_states.pt"))
    if len(models) != 1 or models[0].name != "mp_rank_00_model_states.pt":
        raise ValueError("checkpoint requires exactly one mp_rank_00_model_states.pt")
    model = models[0]
    ranks = {}
    for path in root.rglob("*.pt"):
        if path.is_symlink() or path.parent != model.parent:
            raise ValueError("checkpoint contains unsupported nested or symlinked tensor files")
        if path == model:
            continue
        match = _OPTIM_FILE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unsupported checkpoint tensor file: {path.name}")
        rank = int(match.group(1))
        if rank in ranks:
            raise ValueError(f"duplicate checkpoint DP rank {rank}")
        ranks[rank] = path
    if not ranks or sorted(ranks) != list(range(len(ranks))):
        raise ValueError("checkpoint has missing or noncontiguous DP ranks")
    return model, [ranks[rank] for rank in range(len(ranks))]


@dataclass
class _Snapshot:
    step: int
    shapes: list[dict[str, tuple[int, ...]]]
    shards: list[dict]
    groups: list[list[dict]]
    partition_sizes: list[int]
    native_layout: dict | None = None
    rank_routes: list[list[list[int]]] | None = None
    expert_shards: list[dict] | None = None
    checkpoint_hashes: dict[str, str] | None = None


def _snapshot(
    root: Path,
    *,
    expected_optimizer_config: dict | None,
    expected_learning_rate: float | None,
    expected_native_layout: dict | None = None,
) -> _Snapshot:
    native = expected_native_layout
    experts, input_hashes, expert_files = None, None, []
    if native is None:
        model_file, optim_files = _checkpoint_files(root)
    else:
        model_file, optim_files, expert_files = _native_checkpoint_files(root, native)
        input_hashes = _native_input_hashes(root)
    model = _load(model_file)
    world_size = len(optim_files)
    step = _integer(model.get("global_steps"), "engine global_steps")
    if _integer(model.get("dp_world_size"), "dp_world_size") != world_size:
        raise ValueError("checkpoint has missing DP ranks or a dp_world_size mismatch")
    if _integer(model.get("mp_world_size"), "mp_world_size") != (
        1 if native is None else native["model_mp_world_size"]
    ):
        raise ValueError("model parallel checkpoints are unsupported")
    if model.get("num_experts") or model.get("ds_autoep_layers") or model.get("moe_layer_info"):
        raise ValueError("expert parallel checkpoints are unsupported")
    model_config = model.get("ds_config")
    if native is not None and model_config != native["ds_config"]:
        raise ValueError("native checkpoint does not match the selected ds_config")
    tag = re.fullmatch(r"global_step(\d+)", model_file.parent.name)
    if tag and int(tag.group(1)) != step:
        raise ValueError("checkpoint tag does not match engine global_steps")
    shape_groups = model.get("param_shapes")
    if not isinstance(shape_groups, list) or not shape_groups:
        raise ValueError("checkpoint requires nonempty ordered param_shapes")
    shapes = []
    names = set()
    module = model.get("module")
    if native is not None:
        if not isinstance(module, dict):
            raise ValueError("native checkpoint requires its replicated module payload")
        experts = [_load(path) for path in expert_files]
        for shard in experts:
            if not shard or set(shard) != set(experts[0]):
                raise ValueError("inconsistent expert sidefile parameter coverage")
            for name, value in shard.items():
                reference = module.get(name)
                if (
                    not isinstance(name, str)
                    or not isinstance(value, torch.Tensor)
                    or value.dtype != torch.bfloat16
                    or value.layout != torch.strided
                    or value.ndim < 1
                    or value.numel() == 0
                    or not isinstance(reference, torch.Tensor)
                    or value.shape != reference.shape
                    or reference.dtype != value.dtype
                    or not bool(torch.isfinite(value).all())
                ):
                    raise ValueError("invalid expert sidefile tensor or module shape")
        if any(not torch.equal(value, module[name]) for name, value in experts[native["model_ep_rank"]].items()):
            raise ValueError("rank-zero expert sidefile/module mismatch")
    for group in shape_groups:
        if not isinstance(group, dict) or not group:
            raise ValueError("param_shapes contains an empty or invalid group")
        entries = {}
        for name, shape in group.items():
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("param_shapes has invalid or duplicate parameter names")
            if not isinstance(shape, (list, tuple, torch.Size)):
                raise ValueError(f"invalid parameter shape for {name}")
            dimensions = tuple(_integer(dim, f"{name} shape") for dim in shape)
            if math.prod(dimensions) == 0:
                raise ValueError(f"empty parameter shape for {name}")
            if isinstance(module, dict) and name in module and tuple(module[name].shape) != dimensions:
                raise ValueError(f"model/optimizer shape mismatch for {name}")
            entries[name] = dimensions
            names.add(name)
        shapes.append(entries)
    # The BF16 module payload is not a master-weight source and is no longer needed.
    del model, module
    shards = []
    groups = []
    partition_sizes = []
    rank_routes = []
    count = len(shapes)
    for rank, path in enumerate(optim_files):
        state = _load(path)
        optimizer = state.get("optimizer_state_dict")
        if not isinstance(optimizer, dict):
            raise ValueError("checkpoint requires optimizer_state_dict")
        if _integer(optimizer.get("zero_stage"), "ZeRO stage") != 2:
            raise ValueError("only dense ZeRO2 checkpoints are supported")
        if native is None:
            _config(model_config, "model checkpoint")
            _config(state.get("ds_config"), "optimizer checkpoint")
        else:
            if state.get("ds_config") != model_config:
                raise ValueError("native optimizer ds_config mismatch")
            optimizer_type = model_config.get("optimizer", {}).get("type")
            if not isinstance(optimizer_type, str) or optimizer_type.strip().lower() != "adamw":
                raise ValueError("native optimizer requires AdamW")
        if optimizer.get("overflow"):
            raise ValueError("overflowed optimizer checkpoint is unsupported")
        counts = _sequence(optimizer.get("partition_count"), count, "partition_count")
        if native is None and any(_integer(value, "partition_count") != world_size for value in counts):
            raise ValueError("missing DP ranks or expert parallel partition_count")
        paddings = _sequence(optimizer.get("group_paddings"), count, "group_paddings")
        masters = _sequence(optimizer.get("single_partition_of_fp32_groups"), count, "FP32 master groups")
        mappings = _sequence(optimizer.get("param_slice_mappings"), count, "param_slice_mappings")
        if "param_alignment_paddings" in optimizer:
            alignment = _sequence(optimizer["param_alignment_paddings"], count, "param_alignment_paddings")
            for index, values in enumerate(alignment):
                values = _sequence(values, len(shapes[index]), "parameter alignment padding")
                if any(_integer(value, "parameter alignment padding") != 0 for value in values):
                    raise ValueError("nonzero parameter alignment padding is unsupported")
        base = optimizer.get("base_optimizer_state")
        if not isinstance(base, dict) or not isinstance(base.get("state"), dict):
            raise ValueError("checkpoint requires nonelastic base_optimizer_state")
        param_groups = _sequence(base.get("param_groups"), count, "base optimizer param_groups")
        ids = []
        for index, group in enumerate(param_groups):
            if not isinstance(group, dict) or (native is None and group.get("moe")) or group.get("amsgrad"):
                raise ValueError("unsupported optimizer param group (EP or AMSGrad)")
            if native is None:
                partition_rank, group_world = rank, world_size
            else:
                expert_group = group.get("moe", False)
                if type(expert_group) is not bool:
                    raise ValueError("native optimizer moe flag must be boolean")
                routes = (
                    native["expert_data_parallel_groups"].get(group.get("name"))
                    if expert_group
                    else [list(range(world_size))]
                )
                if not routes:
                    raise ValueError("native expert optimizer group has no actual ordered DP route")
                if rank == 0:
                    rank_routes.append(routes)
                elif routes != rank_routes[index]:
                    raise ValueError("native optimizer group ownership drift")
                route = next(route for route in routes if rank in route)
                partition_rank, group_world = route.index(rank), len(route)
                if _integer(counts[index], "partition_count") != group_world:
                    raise ValueError("native optimizer partition_count disagrees with actual DP route")
                shape_names = set(shapes[index])
                if expert_group != bool(shape_names & set(experts[0])) or (
                    expert_group and not shape_names.issubset(experts[0])
                ):
                    raise ValueError("native expert/dense parameter ownership mismatch")
            params = _sequence(group.get("params"), 1, "optimizer group params")
            param_id = _integer(params[0], "optimizer parameter ID")
            if param_id in ids:
                raise ValueError("duplicate optimizer parameter ID")
            ids.append(param_id)
            if expected_optimizer_config is not None:
                for key, default in (("betas", [0.9, 0.999]), ("eps", 1e-8), ("weight_decay", 0.0)):
                    actual = group.get(key)
                    expected = expected_optimizer_config.get(key, default)
                    if key == "betas":
                        actual = tuple(actual) if isinstance(actual, (list, tuple)) else actual
                        expected = tuple(expected)
                    if actual != expected:
                        raise ValueError(f"optimizer config mismatch for {key}")
                if not group.get("bias_correction", True):
                    raise ValueError("optimizer bias_correction must be enabled")
            if expected_learning_rate is not None and group.get("lr") != expected_learning_rate:
                raise ValueError("optimizer learning rate mismatch")
            if rank:
                keys = ("betas", "eps", "weight_decay", "bias_correction", "lr")
                if native is not None:
                    keys += ("moe", "name")
                for key in keys:
                    if group.get(key) != groups[0][index].get(key):
                        raise ValueError(f"inconsistent optimizer rank config for {key}")
            if not isinstance(mappings[index], dict):
                raise ValueError("invalid param_slice_mappings")
            master = masters[index]
            if not isinstance(master, torch.Tensor):
                raise ValueError("FP32 master groups must contain tensors")
            padding = _integer(paddings[index], "group padding")
            # ZeRO2 state_dict removes master padding but retains full nonelastic Adam state partitions.
            length = master.numel() + padding
            vector_check = _fp32_vector if native is None else _bf16_vector
            vector_check(master, length - padding, "optimizer master partition")
            if native is not None and not bool(torch.isfinite(master).all()):
                raise ValueError("nonfinite native optimizer master")
            if rank == 0:
                if length == 0:
                    raise ValueError("empty FP32 partition")
                partition_sizes.append(length)
            if length != partition_sizes[index]:
                raise ValueError("inconsistent FP32 partition sizes")
            total = sum(math.prod(shape) for shape in shapes[index].values())
            wanted_length = max(0, min(length, total - partition_rank * length))
            if padding != length - wanted_length:
                raise ValueError("FP32 partition padding/parameter shape mismatch")
            if master.numel() != wanted_length:
                raise ValueError("FP32 partition padding/parameter shape mismatch")
            group_state = base["state"].get(param_id, {})
            if not isinstance(group_state, dict):
                raise ValueError("invalid Adam optimizer state")
            if not group_state and step == 0:
                if _integer(group.get("step", 0), "baseline optimizer step") != 0:
                    raise ValueError("baseline optimizer step mismatch")
            else:
                # FusedAdam's group.step is only a legacy initialization seed, not its current counter.
                if _integer(group_state.get("step"), "Adam optimizer step") != step:
                    raise ValueError("engine/Adam optimizer step mismatch")
                for key in ("exp_avg", "exp_avg_sq"):
                    vector_check(group_state.get(key), length, f"Adam {key} partition")
                    if native is not None and not bool(torch.isfinite(group_state[key]).all()):
                        raise ValueError(f"nonfinite native Adam {key}")
            expected_mapping = {}
            offset = 0
            for name, shape in shapes[index].items():
                numel = math.prod(shape)
                start = max(offset, partition_rank * length)
                end = min(offset + numel, (partition_rank + 1) * length)
                if start < end:
                    expected_mapping[name] = (start - partition_rank * length, end - start)
                offset += numel
            if set(mappings[index]) != set(expected_mapping):
                raise ValueError("missing or unexpected parameter slice mappings")
            for name, (start, numel) in expected_mapping.items():
                address = mappings[index][name]
                if (
                    _integer(getattr(address, "start", None), "fragment start") != start
                    or _integer(getattr(address, "numel", None), "fragment numel") != numel
                ):
                    raise ValueError(f"parameter slice mapping mismatch for {name}")
        if set(base["state"]) - set(ids):
            raise ValueError("unexpected optimizer state groups")
        shards.append(optimizer)
        groups.append(param_groups)
    for index, shape_group in enumerate(shapes):
        total = sum(math.prod(shape) for shape in shape_group.values())
        partitions = world_size if native is None else len(rank_routes[index][0])
        if total > partitions * partition_sizes[index]:
            raise ValueError("missing FP32 parameter partitions")
    snapshot = _Snapshot(
        step,
        shapes,
        shards,
        groups,
        partition_sizes,
        native,
        rank_routes if native is not None else None,
        experts,
        input_hashes,
    )
    if native is not None:
        used_groups = {group["name"] for group in groups[0] if group.get("moe")}
        if used_groups != set(native["expert_data_parallel_groups"]):
            raise ValueError("native layout has missing or unused expert optimizer routes")
        for index, shape_group in enumerate(shapes):
            if groups[0][index].get("moe"):
                for name in shape_group:
                    actual = _parameter(snapshot, index, name, "fp32_master")
                    expected = torch.cat([shard[name] for shard in experts], dim=0)
                    if not torch.equal(actual, expected):
                        raise ValueError("expert optimizer master/sidefile ownership mismatch")
        if _native_input_hashes(root) != input_hashes:
            raise ValueError("native optimizer checkpoint input drift")
    return snapshot


def _parameter(snapshot: _Snapshot, group_index: int, name: str, key: str) -> torch.Tensor:
    shape = snapshot.shapes[group_index][name]
    routes = [list(range(len(snapshot.shards)))] if snapshot.rank_routes is None else snapshot.rank_routes[group_index]
    values = []
    for route in routes:
        dtype = torch.float32 if snapshot.native_layout is None else torch.bfloat16
        result = torch.empty(math.prod(shape), dtype=dtype)
        offset = 0
        for rank in route:
            shard, param_groups = snapshot.shards[rank], snapshot.groups[rank]
            address = shard["param_slice_mappings"][group_index].get(name)
            if address is None:
                continue
            if key == "fp32_master":
                vector = shard["single_partition_of_fp32_groups"][group_index]
            else:
                param_id = param_groups[group_index]["params"][0]
                vector = shard["base_optimizer_state"]["state"][param_id][key]
            numel = int(address.numel)
            fragment = vector.narrow(0, int(address.start), numel)
            if not bool(torch.isfinite(fragment).all()):
                raise ValueError(f"nonfinite {name}::{key}")
            result.narrow(0, offset, numel).copy_(fragment)
            offset += numel
        if offset != result.numel():
            raise ValueError(f"incomplete reconstruction for {name}::{key}")
        values.append(result.reshape(shape))
    return values[0] if len(values) == 1 else torch.cat(values, dim=0)


def decode_optimizer_capture(
    before_dir: Path,
    after_dir: Path,
    output_dir: Path,
    *,
    expected_optimizer_config: dict | None = None,
    expected_learning_rate: float | None = None,
    expected_native_layout: dict | None = None,
) -> Path:
    """Validate one real optimizer step and write named comparison tensors.

    Input roots may contain one checkpoint tag plus metadata/export receipts. Output must be new.
    The manifest retains the actual post-step counter; the existing single-step comparison requires step 1.
    Native layouts must come from the owner's selected runtime and ordered process groups. Source hashes
    protect those supplied bytes, not their runtime provenance. BF16 moments remain BF16; updates are the
    FP32 difference of the actual saved BF16 masters, never of model sidefiles.
    """
    output_dir = require_data_fast_path(output_dir, label="optimizer artifact output")
    before_dir, after_dir = Path(before_dir), Path(after_dir)
    native = None if expected_native_layout is None else _native_layout(expected_native_layout)
    if expected_optimizer_config is not None:
        name = expected_optimizer_config.get("name", "AdamW")
        if not isinstance(name, str) or name.strip().lower() != "adamw":
            raise ValueError("expected optimizer type must be AdamW")
    if output_dir.exists():
        raise ValueError("optimizer artifact output directory must not already exist")
    for source in (before_dir, after_dir):
        if output_dir.resolve().is_relative_to(source.resolve()):
            raise ValueError("optimizer artifact output must be outside checkpoint inputs")
    options = {} if native is None else {"expected_native_layout": native}
    before = _snapshot(
        before_dir, expected_optimizer_config=expected_optimizer_config, expected_learning_rate=None, **options
    )
    after = _snapshot(
        after_dir,
        expected_optimizer_config=expected_optimizer_config,
        expected_learning_rate=expected_learning_rate,
        **options,
    )
    if after.step != before.step + 1:
        raise ValueError("optimizer capture requires exactly one step between snapshots")
    if (
        before.shapes != after.shapes
        or [list(group) for group in before.shapes] != [list(group) for group in after.shapes]
        or len(before.shards) != len(after.shards)
        or before.partition_sizes != after.partition_sizes
    ):
        raise ValueError("optimizer checkpoint parameter shape/order or partition drift")
    for before_groups, after_groups in zip(before.groups, after.groups):
        for left, right in zip(before_groups, after_groups):
            for key in ("betas", "eps", "weight_decay", "bias_correction"):
                if left.get(key) != right.get(key):
                    raise ValueError(f"optimizer config drift for {key}")
    if native is not None:
        _verify_native_inputs(before_dir, before)
        _verify_native_inputs(after_dir, after)
        _native_sources(native)
    output_dir.mkdir(parents=True)
    try:
        entries = []
        for group_index, shapes in enumerate(after.shapes):
            for name, shape in shapes.items():
                master_before = _parameter(before, group_index, name, "fp32_master")
                master_after = _parameter(after, group_index, name, "fp32_master")
                if native is not None:
                    master_before, master_after = master_before.float(), master_after.float()
                master_after.sub_(master_before)
                if not bool(torch.isfinite(master_after).all()):
                    raise ValueError(f"nonfinite {name}::parameter_update")
                filename = f"{len(entries):06d}.pt"
                torch.save(
                    {
                        "parameter_update": master_after,
                        "exp_avg": _parameter(after, group_index, name, "exp_avg"),
                        "exp_avg_sq": _parameter(after, group_index, name, "exp_avg_sq"),
                    },
                    output_dir / filename,
                )
                entries.append({"name": name, "file": filename, "shape": list(master_after.shape)})
                del master_before, master_after
        manifest = output_dir / "manifest.json"
        provenance = {}
        if native is not None:
            _verify_native_inputs(before_dir, before)
            _verify_native_inputs(after_dir, after)
            _native_sources(native)
            active_experts = {
                name
                for index, shapes in enumerate(after.shapes)
                if after.groups[0][index].get("moe")
                for name in shapes
            }
            provenance = {
                "native_layout": native,
                "source_dtypes": {"master": "bfloat16", "exp_avg": "bfloat16", "exp_avg_sq": "bfloat16"},
                "parameter_update_recipe": "FP32 difference of saved BF16 optimizer masters",
                "expert_sidefiles_are_master_sources": False,
                "frozen_expert_parameters": sorted(set(after.expert_shards[0]) - active_experts),
                "runtime_provenance_is_caller_owned": True,
            }
        manifest.write_text(
            json.dumps(
                {
                    "step": after.step,
                    "before": {"engine_global_step": before.step},
                    "after": {"engine_global_step": after.step},
                    "parameters": entries,
                    **provenance,
                },
                indent=2,
            )
            + "\n"
        )
        if native is not None:
            _native_sources(native)
            _verify_native_inputs(before_dir, before)
            _verify_native_inputs(after_dir, after)
    except BaseException:
        shutil.rmtree(output_dir)
        raise
    return manifest


def _verify_native_inputs(root: Path, snapshot: _Snapshot) -> None:
    _native_checkpoint_files(root, snapshot.native_layout)
    if _native_input_hashes(root) != snapshot.checkpoint_hashes:
        raise ValueError("native optimizer checkpoint input drift")


def _save_local_snapshot(
    client,
    root: Path,
    *,
    step: int,
    expected_dp_world_size: int,
    expected_native_layout: dict | None = None,
) -> Path:

    root = require_data_fast_path(root, label="local optimizer checkpoint")
    if root.exists() or root.is_symlink():
        raise FileExistsError("local optimizer checkpoint must be new")
    result = client.save_checkpoint(path=str(root), checkpoint_type="resumable", export_hf=False)
    if not isinstance(result, dict):
        raise ValueError("local optimizer save requires checkpoint metadata")
    reported = result.get("checkpoint_path")
    tag = f"global_step{step}"
    if (
        not isinstance(reported, str)
        or reported != str(root)
        or require_data_fast_path(reported, label="local optimizer checkpoint receipt") != root
        or result.get("checkpoint_tag") != tag
        or type(result.get("version")) is not int
        or result["version"] != step
        or result.get("stage_path")
    ):
        raise ValueError("local optimizer checkpoint receipt does not match its requested path and step")
    for index, path in enumerate(root.rglob("*")):
        if index >= 4096 or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("local optimizer checkpoint has unsupported entries or symlinks")
    expert_files = []
    if expected_native_layout is None:
        model, shards = _checkpoint_files(root)
    else:
        _native_sources(expected_native_layout)
        model, shards, expert_files = _native_checkpoint_files(root, expected_native_layout)
    if model.parent != root / tag or len(shards) != expected_dp_world_size:
        raise ValueError("local optimizer checkpoint has incorrect tag or DP shard count")
    if any(path.stat().st_size == 0 for path in [model, *shards, *expert_files]):
        raise ValueError("local optimizer checkpoint contains empty tensor files")
    return root


def capture_optimizer_step(
    session: GatewaySession,
    job: ArcticJob,
    body: dict,
    *,
    learning_rate: float,
    output_dir: Path,
    expected_dp_world_size: int,
    optimizer_config: dict,
    local_checkpoint: bool = False,
    expected_native_layout: dict | None = None,
) -> StepResult:
    """Save immediately around the existing step; decode without worker-local diagnostics."""
    if not isinstance(session.backend, CortexConfig):
        raise ValueError("managed optimizer capture requires CortexConfig")
    transport = job.client.transport
    job_id = transport.job_id
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("managed optimizer capture requires an initialized parent job")
    if type(expected_dp_world_size) is not int or expected_dp_world_size < 1:
        raise ValueError("expected_dp_world_size must be a positive integer")
    native = None if expected_native_layout is None else _native_layout(expected_native_layout)
    if native is not None and (not local_checkpoint or native["world_size"] != expected_dp_world_size):
        raise ValueError("native EP capture requires local full-rank checkpoints and matching world size")
    child_id = None
    if native is not None:
        child_id = job.client.jobs.training
        if not isinstance(child_id, str) or not child_id:
            raise ValueError("native EP capture requires its initialized training child")
    if local_checkpoint:
        require_local_cortex_gateway(session.backend)
        if job.client.config.backend != session.backend:
            raise ValueError("local optimizer capture requires the same verified loopback client backend")

    output_dir = require_data_fast_path(output_dir, label="managed optimizer capture output")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "states"
    if destination.exists():
        raise FileExistsError("managed optimizer capture output already exists")
    with TemporaryDirectory(prefix="snapshots-", dir=output_dir) as scratch:
        scratch = Path(scratch)
        options = {} if native is None else {"expected_native_layout": native}
        if local_checkpoint:
            before = _save_local_snapshot(
                job.client, scratch / "before", step=0, expected_dp_world_size=expected_dp_world_size, **options
            )
            if native is not None:
                baseline = _snapshot(
                    before,
                    expected_optimizer_config=optimizer_config,
                    expected_learning_rate=None,
                    **options,
                )
                if baseline.step != 0:
                    raise ValueError("native optimizer capture must start at step zero")
                del baseline
                _native_sources(native)
                if transport.job_id != job_id or job.client.jobs.training != child_id:
                    raise ValueError("native optimizer parent/training child changed")
            result = fwd_bwd_step(session, job, body, learning_rate=learning_rate)
            after = _save_local_snapshot(
                job.client, scratch / "after", step=1, expected_dp_world_size=expected_dp_world_size, **options
            )
            if native is not None and (transport.job_id != job_id or job.client.jobs.training != child_id):
                raise ValueError("native optimizer parent/training child changed")
        else:
            from .inference_export import export_checkpoint

            before_id = save_and_resolve_checkpoint_id(job.client, job_id, "resumable")
            result = fwd_bwd_step(session, job, body, learning_rate=learning_rate)
            after_id = save_and_resolve_checkpoint_id(job.client, job_id, "resumable")
            if before_id == after_id:
                raise ValueError("managed optimizer snapshots must have distinct durable checkpoint IDs")
            before = export_checkpoint(
                job.client, job_id, before_id, scratch / "before", expected_dp_world_size=expected_dp_world_size
            )
            after = export_checkpoint(
                job.client, job_id, after_id, scratch / "after", expected_dp_world_size=expected_dp_world_size
            )
        decoded = scratch / "decoded"
        manifest = Path(
            decode_optimizer_capture(
                before,
                after,
                decoded,
                expected_optimizer_config=optimizer_config,
                expected_learning_rate=learning_rate,
                **options,
            )
        )
        if not manifest.is_file() or not manifest.resolve().is_relative_to(decoded.resolve()):
            raise ValueError("managed optimizer decoder must return its output manifest")
        step = json.loads(manifest.read_text()).get("step")
        if type(step) is not int or step != 1:
            raise ValueError("managed optimizer artifact must describe exactly step 1")
        if native is not None:
            _native_sources(native)
            if transport.job_id != job_id or job.client.jobs.training != child_id:
                raise ValueError("native optimizer parent/training child changed")
        relative_manifest = manifest.relative_to(decoded)
        decoded.rename(destination)
    return replace(result, optimizer_state_manifest=str(destination / relative_manifest))
