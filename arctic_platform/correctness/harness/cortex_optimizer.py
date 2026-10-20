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


"Local gateway optimizer snapshots and dense ZeRO2 decoding."

from __future__ import annotations

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


def _snapshot(
    root: Path, *, expected_optimizer_config: dict | None, expected_learning_rate: float | None
) -> _Snapshot:
    model_file, optim_files = _checkpoint_files(root)
    model = _load(model_file)
    world_size = len(optim_files)
    step = _integer(model.get("global_steps"), "engine global_steps")
    if _integer(model.get("dp_world_size"), "dp_world_size") != world_size:
        raise ValueError("checkpoint has missing DP ranks or a dp_world_size mismatch")
    if _integer(model.get("mp_world_size"), "mp_world_size") != 1:
        raise ValueError("model parallel checkpoints are unsupported")
    if model.get("num_experts") or model.get("ds_autoep_layers") or model.get("moe_layer_info"):
        raise ValueError("expert parallel checkpoints are unsupported")
    model_config = model.get("ds_config")
    tag = re.fullmatch(r"global_step(\d+)", model_file.parent.name)
    if tag and int(tag.group(1)) != step:
        raise ValueError("checkpoint tag does not match engine global_steps")
    shape_groups = model.get("param_shapes")
    if not isinstance(shape_groups, list) or not shape_groups:
        raise ValueError("checkpoint requires nonempty ordered param_shapes")
    shapes = []
    names = set()
    module = model.get("module")
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
    count = len(shapes)
    for rank, path in enumerate(optim_files):
        state = _load(path)
        optimizer = state.get("optimizer_state_dict")
        if not isinstance(optimizer, dict):
            raise ValueError("checkpoint requires optimizer_state_dict")
        if _integer(optimizer.get("zero_stage"), "ZeRO stage") != 2:
            raise ValueError("only dense ZeRO2 checkpoints are supported")
        _config(model_config, "model checkpoint")
        _config(state.get("ds_config"), "optimizer checkpoint")
        if optimizer.get("overflow"):
            raise ValueError("overflowed optimizer checkpoint is unsupported")
        counts = _sequence(optimizer.get("partition_count"), count, "partition_count")
        if any(_integer(value, "partition_count") != world_size for value in counts):
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
            if not isinstance(group, dict) or group.get("moe") or group.get("amsgrad"):
                raise ValueError("unsupported optimizer param group (EP or AMSGrad)")
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
                for key in ("betas", "eps", "weight_decay", "bias_correction", "lr"):
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
            _fp32_vector(master, length - padding, "FP32 master partition")
            if rank == 0:
                if length == 0:
                    raise ValueError("empty FP32 partition")
                partition_sizes.append(length)
            if length != partition_sizes[index]:
                raise ValueError("inconsistent FP32 partition sizes")
            total = sum(math.prod(shape) for shape in shapes[index].values())
            wanted_length = max(0, min(length, total - rank * length))
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
                    _fp32_vector(group_state.get(key), length, f"Adam {key} partition")
            expected_mapping = {}
            offset = 0
            for name, shape in shapes[index].items():
                numel = math.prod(shape)
                start = max(offset, rank * length)
                end = min(offset + numel, (rank + 1) * length)
                if start < end:
                    expected_mapping[name] = (start - rank * length, end - start)
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
        if total > world_size * partition_sizes[index]:
            raise ValueError("missing FP32 parameter partitions")
    return _Snapshot(step, shapes, shards, groups, partition_sizes)


def _parameter(snapshot: _Snapshot, group_index: int, name: str, key: str) -> torch.Tensor:
    shape = snapshot.shapes[group_index][name]
    result = torch.empty(math.prod(shape), dtype=torch.float32)
    offset = 0
    for shard, param_groups in zip(snapshot.shards, snapshot.groups):
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
    return result.reshape(shape)


def decode_optimizer_capture(
    before_dir: Path,
    after_dir: Path,
    output_dir: Path,
    *,
    expected_optimizer_config: dict | None = None,
    expected_learning_rate: float | None = None,
) -> Path:
    """Validate one real optimizer step and write named FP32 comparison tensors.

    Input roots may contain one checkpoint tag plus metadata/export receipts. Output must be new.
    The manifest retains the actual post-step counter; the existing single-step comparison requires step 1.
    """
    output_dir = require_data_fast_path(output_dir, label="optimizer artifact output")
    before_dir, after_dir = Path(before_dir), Path(after_dir)
    if expected_optimizer_config is not None:
        name = expected_optimizer_config.get("name", "AdamW")
        if not isinstance(name, str) or name.strip().lower() != "adamw":
            raise ValueError("expected optimizer type must be AdamW")
    if output_dir.exists():
        raise ValueError("optimizer artifact output directory must not already exist")
    for source in (before_dir, after_dir):
        if output_dir.resolve().is_relative_to(source.resolve()):
            raise ValueError("optimizer artifact output must be outside checkpoint inputs")
    before = _snapshot(before_dir, expected_optimizer_config=expected_optimizer_config, expected_learning_rate=None)
    after = _snapshot(
        after_dir,
        expected_optimizer_config=expected_optimizer_config,
        expected_learning_rate=expected_learning_rate,
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
    output_dir.mkdir(parents=True)
    try:
        entries = []
        for group_index, shapes in enumerate(after.shapes):
            for name, shape in shapes.items():
                master_before = _parameter(before, group_index, name, "fp32_master")
                master_after = _parameter(after, group_index, name, "fp32_master")
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
                entries.append({"name": name, "file": filename, "shape": list(shape)})
                del master_before, master_after
        manifest = output_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "step": after.step,
                    "before": {"engine_global_step": before.step},
                    "after": {"engine_global_step": after.step},
                    "parameters": entries,
                },
                indent=2,
            )
            + "\n"
        )
    except BaseException:
        shutil.rmtree(output_dir)
        raise
    return manifest


def _save_local_snapshot(client, root: Path, *, step: int, expected_dp_world_size: int) -> Path:

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
    model, shards = _checkpoint_files(root)
    if model.parent != root / tag or len(shards) != expected_dp_world_size:
        raise ValueError("local optimizer checkpoint has incorrect tag or DP shard count")
    if any(path.stat().st_size == 0 for path in [model, *shards]):
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
        if local_checkpoint:
            before = _save_local_snapshot(
                job.client, scratch / "before", step=0, expected_dp_world_size=expected_dp_world_size
            )
            result = fwd_bwd_step(session, job, body, learning_rate=learning_rate)
            after = _save_local_snapshot(
                job.client, scratch / "after", step=1, expected_dp_world_size=expected_dp_world_size
            )
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
            )
        )
        if not manifest.is_file() or not manifest.resolve().is_relative_to(decoded.resolve()):
            raise ValueError("managed optimizer decoder must return its output manifest")
        step = json.loads(manifest.read_text()).get("step")
        if type(step) is not int or step != 1:
            raise ValueError("managed optimizer artifact must describe exactly step 1")
        relative_manifest = manifest.relative_to(decoded)
        decoded.rename(destination)
    return replace(result, optimizer_state_manifest=str(destination / relative_manifest))
