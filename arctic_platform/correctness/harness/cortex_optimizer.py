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

import contextlib
import fcntl
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import shutil
import stat
import uuid
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


_COLLECTOR_STATE = "_correctness_native_optimizer_collector"
_CLEANUP_STATE = "_correctness_native_optimizer_cleanup"
_SCOPE_STATE = "_correctness_native_optimizer_scope"


def verify_native_callback(callback, descriptor: dict) -> None:
    if not callable(callback) or not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise ValueError("native optimizer callback requires an exact source descriptor")
    selected = callback
    if hasattr(callback, "__wrapped__"):
        wrapper = contextlib.contextmanager(lambda: None)
        if (
            callback.__code__ is not wrapper.__code__
            or callback.__globals__ is not contextlib.__dict__
            or len(callback.__closure__ or ()) != 1
            or callback.__closure__[0].cell_contents is not callback.__wrapped__
        ):
            raise ValueError("native optimizer callback has an unsupported wrapper")
        selected = callback.__wrapped__
    path = Path(descriptor["path"])
    if (
        not path.is_absolute()
        or path.is_symlink()
        or inspect.getsourcefile(selected) != str(path)
        or not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]
    ):
        raise ValueError("native optimizer callback source drift or origin mismatch")


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


def _native_runtime_sources(worker, sources: dict) -> None:
    _native_sources({"source_files": sources})
    checkpointer, engine = worker._checkpointer, worker._engine
    zero = engine.optimizer
    selected = {
        "training": type(worker).save,
        "expert_checkpointer": type(checkpointer).save_resumable,
        "base_checkpointer": type(checkpointer).save_adapters,
        "engine": type(engine).save_checkpoint,
        "zero": type(zero).state_dict,
        "zero_base": type(zero).backward,
    }
    for role, function in selected.items():
        if inspect.getsourcefile(function) != sources[role]["path"]:
            raise ValueError(f"native selected source origin mismatch: {role}")
    _native_sources({"source_files": sources})


def _native_worker_layout(worker, target: Path, tag: str, sources: dict) -> dict:
    import deepspeed.utils.groups as ds_groups
    import torch.distributed as dist

    _native_runtime_sources(worker, sources)
    engine, zero = worker._engine, worker._engine.optimizer
    dp_order = dist.get_process_group_ranks(zero.dp_process_group)
    world, global_rank = dist.get_world_size(), dist.get_rank()
    if sorted(dp_order) != list(range(world)) or worker.rank != global_rank or worker.world_size != world:
        raise ValueError("native collector requires the complete actual checkpoint DP group")
    dp_rank = dist.get_rank(group=zero.dp_process_group)
    if dp_order[dp_rank] != global_rank:
        raise ValueError("native checkpoint DP rank order mismatch")
    items, ep_rank, edp_rank = worker._checkpointer._expert_items_and_ranks(engine)
    names = {parameter.group_name for _, parameter in items}
    if len(names) != 1:
        raise ValueError("native expert sidefile requires one selected expert group")
    expert_group = ds_groups._get_expert_parallel_group(next(iter(names)))
    ep_size = dist.get_world_size(group=expert_group)
    groups = []
    param_groups = zero.optimizer.param_groups
    if len(param_groups) != len(zero.real_dp_process_group):
        raise ValueError("native optimizer group/route count mismatch")
    for index, group in enumerate(param_groups):
        ordered = dist.get_process_group_ranks(zero.real_dp_process_group[index])
        ranks = [dp_order.index(rank) for rank in ordered]
        is_moe = group.get("moe", False)
        if type(is_moe) is not bool:
            raise ValueError("native optimizer MoE group marker must be boolean")
        group_ep_rank = None
        if is_moe:
            name = group.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("native optimizer requires named expert groups")
            group_ep_rank = dist.get_rank(group=ds_groups._get_expert_parallel_group(name))
            if group_ep_rank != ep_rank or ranks.index(dp_rank) != edp_rank:
                raise ValueError("native expert optimizer route differs from sidefile ownership")
        elif ranks != list(range(world)):
            raise ValueError("native dense optimizer requires checkpoint DP order")
        groups.append({"name": group.get("name"), "moe": is_moe, "ranks": ranks, "ep_rank": group_ep_rank})
    filename = Path(engine._get_zero_ckpt_name(str(target), tag))
    match = _NATIVE_OPTIM_FILE.fullmatch(filename.name)
    if filename.parent != target / tag or not match or int(match[1]) != dp_rank:
        raise ValueError("native optimizer filename disagrees with its actual DP rank")
    result = {
        "global_rank": global_rank,
        "checkpoint_rank": dp_rank,
        "dp_order": dp_order,
        "world_size": world,
        "ep_size": ep_size,
        "sp_size": worker.sp_size,
        "model_mp_world_size": engine.mp_world_size,
        "ep_rank": ep_rank,
        "edp_rank": edp_rank,
        "optimizer_mp_rank": int(match[2]),
        "optimizer_filename": filename.name,
        "groups": groups,
        "ds_config": engine._config._param_dict,
        "source_files": sources,
    }
    return json.loads(json.dumps(result, allow_nan=False))


def _claim_native_checkpoint_root(state, target):
    lease = state.get("checkpoint_lease")
    if lease is None:
        return
    parent = Path(lease["checkpoint_parent"])
    if target == parent or not target.is_relative_to(parent):
        raise ValueError("native checkpoint root escaped its admitted parent")
    if any(path.is_symlink() for path in (target, *target.parents)):
        raise ValueError("native checkpoint ownership refuses symlinks")
    control = parent / f".native-optimizer-{lease['scope_nonce']}"
    marker = {"schema": "native-optimizer-root-owner-v1", **lease}
    control.mkdir(parents=True, exist_ok=True)
    if control.is_symlink() or not control.is_dir():
        raise ValueError("native checkpoint control must be a regular directory")
    descriptor = os.open(control / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+b") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("native checkpoint control lock must be a regular file")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not target.exists():
            target.mkdir(parents=True)
            identity = (target.stat().st_dev, target.stat().st_ino)
            pending = {"identity": identity, "marker_identity": None}
            state.setdefault("pending_roots", {})[str(target)] = pending
            with (target / ".native-optimizer-owner.json").open("x") as stream:
                metadata = os.fstat(stream.fileno())
                pending["marker_identity"] = (metadata.st_dev, metadata.st_ino)
                stream.write(json.dumps(marker, allow_nan=False) + "\n")
        ownership = target / ".native-optimizer-owner.json"
        if target.is_symlink() or ownership.is_symlink() or json.loads(ownership.read_text()) != marker:
            raise ValueError("native checkpoint root is not owned by this scope")
        identity = (target.stat().st_dev, target.stat().st_ino)
        previous = state["roots"].get(str(target))
        if previous is not None and previous != identity:
            raise ValueError("native checkpoint root identity changed")
        state["roots"][str(target)] = identity
        state.get("pending_roots", {}).pop(str(target), None)


def arm_native_optimizer_collector(
    worker,
    *,
    parent_job_id: str,
    training_child_id: str,
    source_files: dict,
    checkpoint_lease: dict | None = None,
    nonce: str | None = None,
) -> str:
    """Arm one actual worker; transport and parent-to-child resolution belong to the caller."""
    if (
        not isinstance(parent_job_id, str)
        or not parent_job_id
        or not isinstance(training_child_id, str)
        or not training_child_id
        or worker.job_id != training_child_id
        or _COLLECTOR_STATE in vars(worker)
        or _CLEANUP_STATE in vars(worker)
    ):
        raise ValueError("native collector requires its actual owned training child and an unarmed worker")
    sources = json.loads(json.dumps(source_files, allow_nan=False))
    _native_runtime_sources(worker, sources)
    if checkpoint_lease is not None:
        checkpoint_lease = json.loads(json.dumps(checkpoint_lease, allow_nan=False))
        if (
            set(checkpoint_lease) != {"checkpoint_parent", "scope_nonce", "parent_job_id", "training_child_id"}
            or checkpoint_lease["parent_job_id"] != parent_job_id
            or checkpoint_lease["training_child_id"] != training_child_id
            or re.fullmatch(r"[0-9a-f]{32}", checkpoint_lease["scope_nonce"]) is None
        ):
            raise ValueError("native checkpoint lease must bind its exact parent/child and scope nonce")
        require_data_fast_path(checkpoint_lease["checkpoint_parent"], label="native checkpoint lease parent")
    checkpointer, engine = worker._checkpointer, worker._engine
    original = checkpointer.save_resumable
    missing = object()
    previous = vars(checkpointer).get("save_resumable", missing)
    nonce = uuid.uuid4().hex if nonce is None else nonce
    if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
        raise ValueError("native collector requires an exact hexadecimal nonce")

    def save(selected_engine, target_dir, tag):
        target = require_data_fast_path(target_dir, label="native collector checkpoint")
        step = engine.global_steps
        if (
            selected_engine is not engine
            or worker._engine is not engine
            or worker._checkpointer is not checkpointer
            or worker.job_id != training_child_id
            or type(step) is not int
            or step not in (0, 1)
            or tag != f"global_step{step}"
        ):
            raise ValueError("native collector save scope or step changed")
        before = _native_worker_layout(worker, target, tag, sources)
        _claim_native_checkpoint_root(vars(worker)[_COLLECTOR_STATE], target)
        result = original(selected_engine, str(target), tag)
        if worker.job_id != training_child_id or worker._engine is not engine or engine.global_steps != step:
            raise ValueError("native collector training child/engine changed during save")
        after = _native_worker_layout(worker, target, tag, sources)
        if before != after:
            raise ValueError("native collector layout changed during save")
        paths = [target / tag / after["optimizer_filename"]]
        if after["checkpoint_rank"] == 0:
            paths.append(target / tag / "mp_rank_00_model_states.pt")
        if after["edp_rank"] == 0:
            paths.append(target / tag / f"experts_ep_rank_{after['ep_rank']:03d}.pt")
        if any(path.is_symlink() or not path.is_file() or not path.stat().st_size for path in paths):
            raise ValueError("native collector is missing its actual written checkpoint files")
        receipt = {
            "schema": "native-ep-zero2-worker-save-v1",
            "parent_job_id": parent_job_id,
            "training_child_id": training_child_id,
            "step": step,
            "tag": tag,
            "checkpoint_root": str(target),
            "nonce": nonce,
            "layout": after,
            "written_files": {path.name: digest for path, digest in zip(paths, _checkpoint_hashes(paths).values())},
            "written_file_sizes": {path.name: path.stat().st_size for path in paths},
        }
        _native_runtime_sources(worker, sources)
        directory = target / tag / "native_optimizer_layout"
        if directory.is_symlink():
            raise ValueError("native collector receipt directory must not be a symlink")
        directory.mkdir(exist_ok=True)
        with (directory / f"rank{after['checkpoint_rank']:05d}.json").open("x") as stream:
            stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        _native_runtime_sources(worker, sources)
        vars(worker)[_COLLECTOR_STATE]["saves"][(str(target), step)] = {
            "receipt": receipt,
            "identities": {
                path.name: tuple(
                    getattr(path.stat(), field)
                    for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
                )
                for path in paths
            },
        }
        return result

    vars(worker)[_COLLECTOR_STATE] = {
        "nonce": nonce,
        "checkpointer": checkpointer,
        "hook": save,
        "saves": {},
        "checkpoint_lease": checkpoint_lease,
        "roots": {},
        "pending_roots": {},
        "restore": lambda: (
            vars(checkpointer).pop("save_resumable")
            if previous is missing
            else setattr(checkpointer, "save_resumable", previous)
        ),
    }
    checkpointer.save_resumable = save
    return nonce


def _close_native_scope_worker(worker, nonce):
    state = vars(worker).get(_COLLECTOR_STATE)
    if state is None:
        closed = vars(worker).get(_CLEANUP_STATE)
        if closed is None:
            return {"lease_owned": False, "roots": {}, "pending_roots": {}}
        if not isinstance(closed, dict) or closed["nonce"] != nonce:
            raise ValueError("native scope close requires its exact cached nonce")
    else:
        closed = {
            "nonce": nonce,
            "lease": state["checkpoint_lease"],
            "roots": dict(state["roots"]),
            "pending_roots": dict(state.get("pending_roots", {})),
        }
        finish_native_optimizer_collector(worker, nonce)
        vars(worker)[_CLEANUP_STATE] = closed
    return {
        "lease_owned": isinstance(closed["lease"], dict),
        "roots": closed["roots"],
        "pending_roots": closed.get("pending_roots", {}),
    }


def _cleanup_native_scope_worker(worker, nonce, roots, pending_roots=None):
    pending_roots = {} if pending_roots is None else pending_roots
    closed = vars(worker).get(_CLEANUP_STATE)
    if closed is None and not roots and not pending_roots and _COLLECTOR_STATE not in vars(worker):
        return {"hooks_restored": True, "owned_roots_absent": True}
    if not isinstance(closed, dict) or closed["nonce"] != nonce or not isinstance(closed["lease"], dict):
        raise ValueError("native cleanup requires the closed worker's cached lease")
    lease = closed["lease"]
    parent = Path(lease["checkpoint_parent"])
    marker = {"schema": "native-optimizer-root-owner-v1", **lease}
    for raw, pending in pending_roots.items():
        root = Path(raw)
        if (
            root == parent
            or not root.is_relative_to(parent)
            or any(path.is_symlink() for path in (root, *root.parents))
        ):
            raise ValueError("native cleanup refuses escaped or symlink pending roots")
        if not root.exists():
            continue
        if (root.stat().st_dev, root.stat().st_ino) != tuple(pending["identity"]):
            raise ValueError("native cleanup pending root identity changed")
        ownership = root / ".native-optimizer-owner.json"
        contents = list(root.iterdir())
        if contents:
            if contents != [ownership] or ownership.is_symlink() or not ownership.is_file():
                raise ValueError("native cleanup preserves foreign pending root contents")
            if pending["marker_identity"] is None or (ownership.stat().st_dev, ownership.stat().st_ino) != tuple(
                pending["marker_identity"]
            ):
                raise ValueError("native cleanup pending marker identity changed")
            ownership.unlink()
        root.rmdir()
    for raw, identity in roots.items():
        root = Path(raw)
        if (
            root == parent
            or not root.is_relative_to(parent)
            or any(path.is_symlink() for path in (root, *root.parents))
        ):
            raise ValueError("native cleanup refuses escaped or symlink roots")
        if not root.exists():
            continue
        if (root.stat().st_dev, root.stat().st_ino) != tuple(identity):
            raise ValueError("native cleanup root identity changed")
        ownership = root / ".native-optimizer-owner.json"
        if ownership.is_symlink() or json.loads(ownership.read_text()) != marker:
            raise ValueError("native cleanup preserves a foreign ownership marker")
        for index, path in enumerate(root.rglob("*")):
            if index >= 4096 or path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise ValueError("native cleanup preserves unsupported root contents")
        shutil.rmtree(root)
    control = parent / f".native-optimizer-{lease['scope_nonce']}"
    if control.exists():
        if control.is_symlink() or {path.name for path in control.iterdir()} != {"lock"}:
            raise ValueError("native cleanup preserves foreign control contents")
        lock = control / "lock"
        if lock.is_symlink() or not lock.is_file():
            raise ValueError("native cleanup control lock changed")
        lock.unlink()
        control.rmdir()
    del vars(worker)[_CLEANUP_STATE]
    return {"hooks_restored": True, "owned_roots_absent": all(not Path(raw).exists() for raw in roots)}


def _native_scope_sources(request):
    if request.get("schema") not in ("native-optimizer-scope-request-v1", "native-optimizer-owner-split-request-v1"):
        raise ValueError("unsupported native optimizer scope request")
    _native_sources({"source_files": request["source_files"]})
    roles = {"gateway_snowapi", "gateway", "manager", "ray_get", "ray_actor"}
    if request["schema"] == "native-optimizer-owner-split-request-v1":
        roles.add("ray_is_initialized")
    if not isinstance(request.get("route_sources"), dict) or set(request["route_sources"]) != roles:
        raise ValueError("native scope requires its complete selected route sources")
    verify_native_callback(make_native_optimizer_scope, request["helper_source"])
    for descriptor in request["route_sources"].values():
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
            raise ValueError("native scope source descriptor differs")
        path = Path(descriptor["path"])
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]
        ):
            raise ValueError("native scope selected route source drift")
    require_data_fast_path(request["checkpoint_parent"], label="native scope checkpoint parent")
    if (
        type(request.get("reserve_bytes")) is not int
        or request["reserve_bytes"] < 0
        or type(request.get("rpc_timeout_seconds")) not in (int, float)
        or not math.isfinite(request["rpc_timeout_seconds"])
        or not 0 < request["rpc_timeout_seconds"] <= 300
        or not isinstance(request.get("allocation_hosts"), list)
        or not request["allocation_hosts"]
        or any(not isinstance(host, str) or not host for host in request["allocation_hosts"])
        or len(set(request["allocation_hosts"])) != len(request["allocation_hosts"])
    ):
        raise ValueError("native scope requires exact allocation hosts and bounded explicit budgets")
    if request["schema"] == "native-optimizer-owner-split-request-v1":
        if type(request.get("expected_world_size")) is not int or request["expected_world_size"] <= 0:
            raise ValueError("native split scope requires its explicit expected worker count")
        descriptor = request.get("verify_parent_binding_source")
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
            raise ValueError("native split scope requires its exact parent verifier source")
        path = Path(descriptor["path"])
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]
        ):
            raise ValueError("native split parent verifier source drift")


def _native_scope_actor_workers(owner, child_id, training_source):
    verify_native_callback(type(owner).save, training_source)
    original_class = type(owner).__ray_actor_class__
    if owner.save.__func__ is not original_class.save or owner._info.job_id != child_id:
        raise ValueError("native scope actual training actor selection differs")
    hosts = [node.host for node in owner._info.layout for _ in node.gpu_ids]
    workers = list(owner._workers)
    if not workers or list(owner._rank_hosts) != hosts or len(workers) != len(hosts):
        raise ValueError("native scope actual worker/rank-host layout differs")
    actor_ids = [worker._actor_id.hex() for worker in workers]
    if len(set(actor_ids)) != len(workers):
        raise ValueError("native scope worker actor handles must be distinct")
    verify_native_callback(type(owner).save, training_source)
    return {"workers": workers, "rank_hosts": hosts, "actor_ids": actor_ids}


def _native_scope_origin(callback, descriptor):
    if not callable(callback) or inspect.getsourcefile(callback) != descriptor["path"]:
        raise ValueError("native scope actual selected RPC origin differs")


def _native_scope_arm_worker(worker, *, nonce, rank, world_size, lease, source_files, helper_source):
    verify_native_callback(_native_scope_arm_worker, helper_source)
    if worker.rank != rank or worker.world_size != world_size:
        raise ValueError("native scope worker rank/world differs from its actual actor route")
    armed = arm_native_optimizer_collector(
        worker,
        parent_job_id=lease["parent_job_id"],
        training_child_id=lease["training_child_id"],
        source_files=source_files,
        checkpoint_lease=lease,
        nonce=nonce,
    )
    try:
        verify_native_callback(_native_scope_arm_worker, helper_source)
        _native_runtime_sources(worker, source_files)
    except BaseException:
        _close_native_scope_worker(worker, nonce)
        raise
    return armed


def _native_scope_save_receipt(worker, nonce, root, step, helper_source):
    verify_native_callback(_native_scope_save_receipt, helper_source)
    state = vars(worker).get(_COLLECTOR_STATE)
    if not isinstance(state, dict) or state["nonce"] != nonce:
        raise ValueError("native scope receipt requires the exact armed worker")
    saved = state["saves"].get((str(root), step))
    if saved is None:
        raise ValueError("native scope requires each actual completed worker save")
    receipt = json.loads(json.dumps(saved["receipt"], allow_nan=False))
    _native_runtime_sources(worker, receipt["layout"]["source_files"])
    verify_native_callback(_native_scope_save_receipt, helper_source)
    return receipt


def make_native_optimizer_scope(request, *, gateway_snowapi, gateway, manager, ray):
    """Bind caller-held current MAIN objects; never discover another allocation or create a receiver."""
    request = json.loads(json.dumps(request, allow_nan=False))
    _native_scope_sources(request)
    manager_module = importlib.import_module("dss.ray_dss.zones.manager")
    for module, name, role in (
        (gateway_snowapi, "dss.ray_dss.gateway_snowapi", "gateway_snowapi"),
        (gateway, "dss.ray_dss.gateway_server", "gateway"),
        (manager_module, "dss.ray_dss.zones.manager", "manager"),
    ):
        if module is not importlib.import_module(name) or module.__file__ != request["route_sources"][role]["path"]:
            raise ValueError("native scope requires the actual selected MAIN modules")
    if type(manager) is not manager_module.ZoneManager or manager.zone_type.value != "training":
        raise ValueError("native scope requires its actual current MAIN training manager")
    verify_native_callback(manager.get_job, request["route_sources"]["manager"])
    _native_scope_origin(ray.get, request["route_sources"]["ray_get"])
    if ray is not importlib.import_module("ray"):
        raise ValueError("native scope requires the current installed Ray module")
    get = ray.get
    timeout = request["rpc_timeout_seconds"]

    def route(session, job):
        _native_scope_sources(request)
        require_local_cortex_gateway(session.backend)
        if job.client.config.backend != session.backend or job.job_type != "training":
            raise ValueError("native scope requires the same local training client")
        parent_id = job.client.transport.job_id
        parent = gateway_snowapi._parents.get(parent_id)
        if (
            type(parent) is not gateway_snowapi.ParentRecord
            or parent.parent_id != parent_id
            or parent.state != "running"
            or parent.cancel_requested
        ):
            raise ValueError("native scope requires its actual running gateway parent")
        child = parent.children.get("training")
        if type(child) is not gateway_snowapi.ChildRecord or child.role != "training" or not child.job_id:
            raise ValueError("native scope requires the parent's actual training child, not a role token")
        zone_id = gateway._jobs.get(child.job_id)
        if zone_id != manager.zone_id or zone_id not in gateway._zones:
            raise ValueError("native scope actual gateway child zone differs from its training manager")
        verify_native_callback(manager.get_job, request["route_sources"]["manager"])
        info = manager.get_job(child.job_id)
        if info.job_id != child.job_id or info.zone_id != zone_id:
            raise ValueError("native scope actual manager job differs")
        actor = info.actor
        _native_scope_origin(actor.__ray_call__.remote, request["route_sources"]["ray_actor"])
        actual = get(
            actor.__ray_call__.remote(_native_scope_actor_workers, child.job_id, request["source_files"]["training"]),
            timeout=timeout,
        )
        if (
            set(actual["rank_hosts"]) != set(request["allocation_hosts"])
            or actual["rank_hosts"] != [node.host for node in info.layout for _ in node.gpu_ids]
            or len(actual["workers"]) != session.slots
        ):
            raise ValueError("native scope actual workers must exactly cover the admitted allocation layout")
        for worker in actual["workers"]:
            _native_scope_origin(worker.__ray_call__.remote, request["route_sources"]["ray_actor"])
        _native_scope_sources(request)
        return parent_id, child.job_id, actor._actor_id.hex(), actual

    @contextlib.contextmanager
    def scope(session, job):
        with _native_owned_worker_scope(request, lambda: route(session, job), get, client=job.client) as supplier:
            yield supplier

    _native_scope_sources(request)
    return scope


@contextlib.contextmanager
def _native_owned_worker_scope(request, route, get, *, client=None, cleanup_state=None):
    parent_id, child_id, actor_id, actual = route()
    timeout = request["rpc_timeout_seconds"]
    lease = {
        "checkpoint_parent": request["checkpoint_parent"],
        "scope_nonce": uuid.uuid4().hex,
        "parent_job_id": parent_id,
        "training_child_id": child_id,
    }
    # Cache dispatch and nonces before submitting arm, including missing acknowledgements.
    owned, pending = [], []
    complete = False
    if client is not None:
        if _SCOPE_STATE in vars(client) or "shutdown" in vars(client):
            raise ValueError("native scope requires an unmodified client shutdown lifecycle")
        shutdown = client.shutdown

        def guarded_shutdown(*args, **kwargs):
            if not complete:
                raise RuntimeError("native optimizer owned cleanup incomplete; child release is blocked")
            return shutdown(*args, **kwargs)

        client.shutdown = guarded_shutdown
        vars(client)[_SCOPE_STATE] = {"cleanup": None, "shutdown": guarded_shutdown}

    def cleanup():
        nonlocal complete
        if complete:
            return
        errors, hosts, failed_hosts = [], {}, set()
        for row in owned:
            try:
                closed = get(row["dispatch"](_close_native_scope_worker, row["nonce"]), timeout=timeout)
                host = hosts.setdefault(row["host"], {"leader": None, "roots": {}, "pending_roots": {}})
                if closed["lease_owned"] and host["leader"] is None:
                    host["leader"] = row
                for root, identity in closed["roots"].items():
                    merged = host["roots"]
                    if root in merged and tuple(merged[root]) != tuple(identity):
                        raise ValueError("native cleanup same-host root identities disagree")
                    merged[root] = identity
                for root, pending_root in closed["pending_roots"].items():
                    merged = host["pending_roots"]
                    if root in merged and merged[root] != pending_root:
                        raise ValueError("native cleanup same-host pending identities disagree")
                    merged[root] = pending_root
            except BaseException as exc:
                failed_hosts.add(row["host"])
                errors.append(exc)
        for row in owned:
            if row["host"] in failed_hosts:
                continue
            host = hosts[row["host"]]
            try:
                result = get(
                    row["dispatch"](
                        _cleanup_native_scope_worker,
                        row["nonce"],
                        host["roots"] if row is host["leader"] else {},
                        host["pending_roots"] if row is host["leader"] else {},
                    ),
                    timeout=timeout,
                )
                if result != {"hooks_restored": True, "owned_roots_absent": True}:
                    raise ValueError("native owned worker cleanup acknowledgement differs")
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise BaseExceptionGroup("native optimizer owned cleanup incomplete; do not release workers", errors)
        if client is not None:
            if vars(client).get("shutdown") is not guarded_shutdown:
                raise ValueError("native scope preserves a foreign client shutdown hook")
            del vars(client)["shutdown"]
            del vars(client)[_SCOPE_STATE]
        complete = True
        if cleanup_state is not None:
            cleanup_state["complete"] = True

    if client is not None:
        vars(client)[_SCOPE_STATE]["cleanup"] = cleanup
    if cleanup_state is not None:
        cleanup_state.update(cleanup=cleanup, complete=False)
    try:
        for rank, (worker, host) in enumerate(zip(actual["workers"], actual["rank_hosts"])):
            nonce = uuid.uuid4().hex
            dispatch = worker.__ray_call__.remote
            owned.append({"dispatch": dispatch, "nonce": nonce, "host": host})
            reference = dispatch(
                _native_scope_arm_worker,
                nonce=nonce,
                rank=rank,
                world_size=len(actual["workers"]),
                lease=lease,
                source_files=request["source_files"],
                helper_source=request["helper_source"],
            )
            pending.append((reference, nonce))
        for reference, nonce in pending:
            if get(reference, timeout=timeout) != nonce:
                raise ValueError("native scope actual worker arm acknowledgement differs")

        def stable_route():
            current = route()
            if (
                current[:3] != (parent_id, child_id, actor_id)
                or current[3]["actor_ids"] != actual["actor_ids"]
                or current[3]["rank_hosts"] != actual["rank_hosts"]
            ):
                raise ValueError("native scope current parent/child/worker route changed")

        def supplier(root, *, step):
            stable_route()
            receipts = [
                get(
                    row["dispatch"](
                        _native_scope_save_receipt, row["nonce"], str(root), step, request["helper_source"]
                    ),
                    timeout=timeout,
                )
                for row in owned
            ]
            by_checkpoint_rank = {}
            for rank, (receipt, row) in enumerate(zip(receipts, owned)):
                checkpoint_rank = receipt["layout"]["checkpoint_rank"]
                if receipt["layout"]["global_rank"] != rank or checkpoint_rank in by_checkpoint_rank:
                    raise ValueError("native scope saved worker rank binding differs")
                by_checkpoint_rank[checkpoint_rank] = row
            receipts.sort(key=lambda receipt: receipt["layout"]["checkpoint_rank"])

            def read_worker(rank, filename, *, offset, length):
                _native_scope_sources(request)
                row = by_checkpoint_rank[rank]
                return get(
                    row["dispatch"](
                        read_native_optimizer_checkpoint,
                        row["nonce"],
                        str(root),
                        step=step,
                        filename=filename,
                        offset=offset,
                        length=length,
                    ),
                    timeout=timeout,
                )

            consolidate_native_optimizer_checkpoints(
                root,
                receipts,
                step=step,
                read_worker=read_worker,
                read_worker_source=request["helper_source"],
                reserve_bytes=request["reserve_bytes"],
            )
            result = assemble_native_optimizer_layout(
                root,
                step=step,
                parent_job_id=parent_id,
                training_child_id=child_id,
                expected_world_size=len(owned),
                source_files=request["source_files"],
            )
            stable_route()
            _native_scope_sources(request)
            return result

        yield supplier
        stable_route()
    finally:
        cleanup()


def _native_split_request(request):
    _native_scope_sources(request)
    if request["schema"] != "native-optimizer-owner-split-request-v1":
        raise ValueError("native owner split requires its exact request schema")
    return hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def resolve_native_optimizer_parent(request, parent_id, *, gateway_snowapi, gateway):
    """Resolve only in the actual gateway owner; allocation fields remain caller expectations."""
    request = json.loads(json.dumps(request, allow_nan=False))
    digest = _native_split_request(request)
    for module, name, role in (
        (gateway_snowapi, "dss.ray_dss.gateway_snowapi", "gateway_snowapi"),
        (gateway, "dss.ray_dss.gateway_server", "gateway"),
    ):
        if module is not importlib.import_module(name) or module.__file__ != request["route_sources"][role]["path"]:
            raise ValueError("native parent resolution requires the actual gateway owner modules")
    parent = gateway_snowapi._parents.get(parent_id)
    if (
        not isinstance(parent_id, str)
        or not parent_id
        or type(parent) is not gateway_snowapi.ParentRecord
        or parent.parent_id != parent_id
        or parent.state != "running"
        or parent.cancel_requested
    ):
        raise ValueError("native parent resolution requires its actual running gateway parent")
    child = parent.children.get("training")
    if type(child) is not gateway_snowapi.ChildRecord or child.role != "training" or not child.job_id:
        raise ValueError("native parent resolution requires the actual training child")
    zone_id = gateway._jobs.get(child.job_id)
    if not isinstance(zone_id, str) or not zone_id or zone_id not in gateway._zones:
        raise ValueError("native parent resolution requires the actual gateway child zone")
    result = {
        "schema": "native-optimizer-parent-binding-v1",
        "request_sha256": digest,
        "parent_job_id": parent_id,
        "training_child_id": child.job_id,
        "zone_id": zone_id,
        "allocation_expectation": {
            "hosts": request["allocation_hosts"],
            "world_size": request["expected_world_size"],
        },
    }
    _native_split_request(request)
    return result


class _NativeOptimizerZoneScope:
    def __init__(self, context, state):
        self._context, self._state = context, state
        self._entered = False

    def __enter__(self):
        if self._entered:
            raise ValueError("native zone scope cannot be entered twice")
        self._entered = True
        return self._context.__enter__()

    def __exit__(self, *exception):
        return self._context.__exit__(*exception)

    def cleanup(self):
        """Retry only cached owned cleanup, even after arm/close failed or admission drifted."""
        callback = self._state.get("cleanup")
        if callback is None:
            if self.cleanup_complete:
                return
            raise ValueError("native zone scope has no cached owned worker cleanup")
        callback()

    @property
    def cleanup_complete(self):
        return self._state.get("complete", False)


def open_native_optimizer_zone_scope(request, binding, *, manager, ray, verify_parent_binding):
    """Arm and collect in the actual zone owner; caller owns IPC and the client shutdown fence."""
    request = json.loads(json.dumps(request, allow_nan=False))
    binding = json.loads(json.dumps(binding, allow_nan=False))
    digest = _native_split_request(request)
    if (
        set(binding)
        != {"schema", "request_sha256", "parent_job_id", "training_child_id", "zone_id", "allocation_expectation"}
        or binding["schema"] != "native-optimizer-parent-binding-v1"
        or binding["request_sha256"] != digest
        or any(
            not isinstance(binding[key], str) or not binding[key]
            for key in ("parent_job_id", "training_child_id", "zone_id")
        )
        or binding["allocation_expectation"]
        != {"hosts": request["allocation_hosts"], "world_size": request["expected_world_size"]}
    ):
        raise ValueError("native zone scope requires its exact source-bound gateway binding")
    manager_module = importlib.import_module("dss.ray_dss.zones.manager")
    if (
        manager_module.__file__ != request["route_sources"]["manager"]["path"]
        or type(manager) is not manager_module.ZoneManager
        or manager.zone_type.value != "training"
        or manager.zone_id != binding["zone_id"]
    ):
        raise ValueError("native zone scope requires its actual owning training manager")
    _native_scope_origin(ray.get, request["route_sources"]["ray_get"])
    if ray is not importlib.import_module("ray"):
        raise ValueError("native zone scope requires canonical initialized owner Ray")
    verify_native_callback(ray.is_initialized, request["route_sources"]["ray_is_initialized"])
    if ray.is_initialized() is not True:
        raise ValueError("native zone scope requires already initialized owner Ray")
    verify_native_callback(ray.is_initialized, request["route_sources"]["ray_is_initialized"])
    verify_native_callback(verify_parent_binding, request["verify_parent_binding_source"])

    def route():
        _native_split_request(request)
        verify_native_callback(ray.is_initialized, request["route_sources"]["ray_is_initialized"])
        if ray.is_initialized() is not True:
            raise ValueError("native zone scope owner Ray is no longer initialized")
        verify_native_callback(verify_parent_binding, request["verify_parent_binding_source"])
        fresh = verify_parent_binding(json.loads(json.dumps(binding, allow_nan=False)))
        verify_native_callback(verify_parent_binding, request["verify_parent_binding_source"])
        if type(fresh) is not dict or fresh != binding:
            raise ValueError("native zone scope current gateway parent binding changed")
        verify_native_callback(manager.get_job, request["route_sources"]["manager"])
        if manager.zone_id != binding["zone_id"] or manager.zone_type.value != "training":
            raise ValueError("native zone scope actual manager identity changed")
        info = manager.get_job(binding["training_child_id"])
        if info.job_id != binding["training_child_id"] or info.zone_id != binding["zone_id"]:
            raise ValueError("native zone scope actual manager job differs")
        actor = info.actor
        _native_scope_origin(actor.__ray_call__.remote, request["route_sources"]["ray_actor"])
        actual = ray.get(
            actor.__ray_call__.remote(
                _native_scope_actor_workers, binding["training_child_id"], request["source_files"]["training"]
            ),
            timeout=request["rpc_timeout_seconds"],
        )
        if (
            set(actual["rank_hosts"]) != set(request["allocation_hosts"])
            or actual["rank_hosts"] != [node.host for node in info.layout for _ in node.gpu_ids]
            or len(actual["workers"]) != request["expected_world_size"]
        ):
            raise ValueError("native zone scope actual workers differ from allocation expectation")
        for worker in actual["workers"]:
            _native_scope_origin(worker.__ray_call__.remote, request["route_sources"]["ray_actor"])
        verify_native_callback(manager.get_job, request["route_sources"]["manager"])
        _native_split_request(request)
        return binding["parent_job_id"], binding["training_child_id"], actor._actor_id.hex(), actual

    state = {"complete": True}
    return _NativeOptimizerZoneScope(_native_owned_worker_scope(request, route, ray.get, cleanup_state=state), state)


def finish_native_optimizer_collector(worker, nonce: str) -> None:
    """Restore the owned hook without making cleanup depend on fresh source admission."""
    state = vars(worker).get(_COLLECTOR_STATE)
    if not isinstance(state, dict) or nonce != state["nonce"]:
        raise ValueError("native collector cleanup requires its exact armed nonce")
    if vars(state["checkpointer"]).get("save_resumable") is not state["hook"]:
        raise ValueError("native collector owned hook changed; refusing to overwrite another hook")
    state["restore"]()
    del vars(worker)[_COLLECTOR_STATE]


def read_native_optimizer_checkpoint(worker, nonce, root, *, step, filename, offset, length) -> dict:
    """Read bounded bytes only from this armed worker's already completed owned save."""
    state = vars(worker).get(_COLLECTOR_STATE)
    if not isinstance(state, dict) or state["nonce"] != nonce:
        raise ValueError("native checkpoint reader requires its armed nonce")
    saved = state["saves"].get((str(root), step))
    if saved is None or filename not in saved["identities"]:
        raise ValueError("native checkpoint reader requires its exact completed save file")
    if type(offset) is not int or offset < 0 or type(length) is not int or not 0 < length <= 8 << 20:
        raise ValueError("native checkpoint reader requires a bounded byte range")
    path = Path(root) / f"global_step{step}" / filename
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")

    def identity():
        if path.is_symlink():
            raise ValueError("native checkpoint reader refuses symlinks")
        return tuple(getattr(path.stat(), field) for field in fields)

    if identity() != saved["identities"][filename]:
        raise ValueError("native checkpoint reader file identity drift")
    with path.open("rb") as stream:
        stream.seek(offset)
        raw = stream.read(length)
    if identity() != saved["identities"][filename]:
        raise ValueError("native checkpoint reader file identity drift")
    return {
        "offset": offset,
        "size_bytes": saved["receipt"]["written_file_sizes"][filename],
        "sha256": saved["receipt"]["written_files"][filename],
        "bytes": raw,
    }


def consolidate_native_optimizer_checkpoints(
    root, worker_receipts, *, step, read_worker, read_worker_source, reserve_bytes
) -> None:
    """Copy only witnessed worker files through the caller's admitted same-allocation transport."""
    root = require_data_fast_path(root, label="native checkpoint consolidation")
    verify_native_callback(read_worker, read_worker_source)
    if type(reserve_bytes) is not int or reserve_bytes < 0 or type(step) is not int or step not in (0, 1):
        raise ValueError("native consolidation requires an explicit disk reserve and valid step")
    if not isinstance(worker_receipts, list) or not worker_receipts:
        raise ValueError("native consolidation requires actual worker save receipts")
    tag = root / f"global_step{step}"
    if tag.is_symlink():
        raise ValueError("native consolidation checkpoint tag symlink")
    selected = []
    names = set()
    for rank, row in enumerate(worker_receipts):
        if (
            row.get("schema") != "native-ep-zero2-worker-save-v1"
            or row.get("checkpoint_root") != str(root)
            or row.get("step") != step
            or row.get("tag") != tag.name
            or row["layout"]["checkpoint_rank"] != rank
            or set(row["written_files"]) != set(row["written_file_sizes"])
        ):
            raise ValueError("native consolidation worker save binding mismatch")
        for name, digest in row["written_files"].items():
            size = row["written_file_sizes"][name]
            if (
                Path(name).name != name
                or name in names
                or type(size) is not int
                or size < 1
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ValueError("native consolidation duplicate or invalid file descriptor")
            names.add(name)
            selected.append((rank, tag / name, size, digest))
    needed = 0
    for _, path, size, digest in selected:
        if path.is_symlink():
            raise ValueError("native consolidation destination symlink")
        if path.exists():
            if not path.is_file() or path.stat().st_size != size or _checkpoint_hashes([path])[str(path)] != digest:
                raise ValueError("native consolidation preserves mismatching existing files")
        else:
            needed += size
    if shutil.disk_usage(root).free < needed + reserve_bytes:
        raise ValueError("native consolidation has insufficient fresh disk budget")
    verify_native_callback(read_worker, read_worker_source)
    tag.mkdir(exist_ok=True)
    for rank, path, size, digest in selected:
        if path.exists():
            continue
        hasher = hashlib.sha256()
        with path.open("xb") as stream:
            offset = 0
            while offset < size:
                verify_native_callback(read_worker, read_worker_source)
                length = min(8 << 20, size - offset)
                block = read_worker(rank, path.name, offset=offset, length=length)
                if (
                    not isinstance(block, dict)
                    or block.get("offset") != offset
                    or block.get("size_bytes") != size
                    or block.get("sha256") != digest
                    or type(block.get("bytes")) is not bytes
                    or len(block["bytes"]) != length
                ):
                    raise ValueError("native consolidation reader returned mismatching bytes")
                stream.write(block["bytes"])
                hasher.update(block["bytes"])
                offset += length
        if hasher.hexdigest() != digest:
            raise ValueError("native consolidation file digest mismatch")
    directory = tag / "native_optimizer_layout"
    if directory.is_symlink():
        raise ValueError("native consolidation receipt directory symlink")
    directory.mkdir(exist_ok=True)
    for rank, row in enumerate(worker_receipts):
        path = directory / f"rank{rank:05d}.json"
        if path.is_symlink():
            raise ValueError("native consolidation receipt symlink")
        if path.exists():
            if json.loads(path.read_text()) != row:
                raise ValueError("native consolidation preserves mismatching existing receipt")
        else:
            with path.open("x") as stream:
                stream.write(json.dumps(row, indent=2, allow_nan=False) + "\n")
    verify_native_callback(read_worker, read_worker_source)
    for _, path, size, digest in selected:
        if path.stat().st_size != size or _checkpoint_hashes([path])[str(path)] != digest:
            raise ValueError("native consolidation file changed after copy")


def assemble_native_optimizer_layout(
    root: Path, *, step: int, parent_job_id: str, training_child_id: str, expected_world_size: int, source_files: dict
) -> dict:
    """Assemble complete same-allocation receipts after caller-owned peer file consolidation."""
    root = require_data_fast_path(root, label="native optimizer receipt root")
    if type(step) is not int or step not in (0, 1) or type(expected_world_size) is not int or expected_world_size < 1:
        raise ValueError("invalid native optimizer receipt step or world size")
    _native_sources({"source_files": source_files})
    tag = f"global_step{step}"
    directory = root / tag / "native_optimizer_layout"
    paths = sorted(directory.glob("*.json"))
    if directory.is_symlink() or len(paths) != expected_world_size:
        raise ValueError("native collector requires every worker receipt")
    rows = []
    for rank, path in enumerate(paths):
        if path.name != f"rank{rank:05d}.json" or path.is_symlink() or path.stat().st_size > 1 << 20:
            raise ValueError("native collector receipt inventory mismatch")
        row = json.loads(path.read_text())
        expected = {
            "schema": "native-ep-zero2-worker-save-v1",
            "parent_job_id": parent_job_id,
            "training_child_id": training_child_id,
            "step": step,
            "tag": tag,
            "checkpoint_root": str(root),
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("native collector receipt child/step/root binding mismatch")
        if not isinstance(row.get("nonce"), str) or re.fullmatch(r"[0-9a-f]{32}", row["nonce"]) is None:
            raise ValueError("native collector requires actual armed nonce")
        layout = row["layout"]
        if layout["checkpoint_rank"] != rank or layout["world_size"] != expected_world_size:
            raise ValueError("native collector receipt rank/world mismatch")
        if set(row["written_files"]) != set(row["written_file_sizes"]):
            raise ValueError("native collector file size coverage mismatch")
        for name, digest in row["written_files"].items():
            path = root / tag / name
            if (
                Path(name).name != name
                or path.is_symlink()
                or path.stat().st_size != row["written_file_sizes"][name]
                or _checkpoint_hashes([path])[str(path)] != digest
            ):
                raise ValueError("native collector written checkpoint file drift")
        rows.append(row)
    first = rows[0]["layout"]
    common = ("dp_order", "world_size", "ep_size", "sp_size", "model_mp_world_size", "ds_config", "source_files")
    if first["source_files"] != source_files or sorted(first["dp_order"]) != list(range(expected_world_size)):
        raise ValueError("native collector selected source or DP order mismatch")
    groups = {}
    for rank, row in enumerate(rows):
        layout = row["layout"]
        if any(layout[key] != first[key] for key in common) or layout["global_rank"] != first["dp_order"][rank]:
            raise ValueError("native collector worker layout disagreement")
        if len(layout["groups"]) != len(first["groups"]):
            raise ValueError("native collector optimizer group count disagreement")
        written = {layout["optimizer_filename"]}
        if rank == 0:
            written.add("mp_rank_00_model_states.pt")
        if layout["edp_rank"] == 0:
            written.add(f"experts_ep_rank_{layout['ep_rank']:03d}.pt")
        if set(row["written_files"]) != written:
            raise ValueError("native collector requires exact worker write coverage")
        for index, group in enumerate(layout["groups"]):
            base = first["groups"][index]
            if (group["name"], group["moe"]) != (base["name"], base["moe"]):
                raise ValueError("native collector optimizer group order disagreement")
            if group["moe"]:
                routes = groups.setdefault(group["name"], [None] * first["ep_size"])
                ep_rank = group["ep_rank"]
                if type(ep_rank) is not int or not 0 <= ep_rank < first["ep_size"]:
                    raise ValueError("native collector expert rank mismatch")
                if group["ranks"].index(rank) != layout["edp_rank"] or ep_rank != layout["ep_rank"]:
                    raise ValueError("native collector expert DP order disagreement")
                if routes[ep_rank] is not None and routes[ep_rank] != group["ranks"]:
                    raise ValueError("native collector expert replicas disagree")
                routes[ep_rank] = group["ranks"]
            elif group["ranks"] != list(range(expected_world_size)):
                raise ValueError("native collector dense DP order disagreement")
    result = _native_layout(
        {
            "schema": "native-ep-zero2-layout-v1",
            **{key: first[key] for key in ("world_size", "ep_size", "sp_size", "model_mp_world_size", "ds_config")},
            "model_ep_rank": first["ep_rank"],
            "optimizer_mp_ranks": [row["layout"]["optimizer_mp_rank"] for row in rows],
            "expert_data_parallel_groups": groups,
            "source_files": source_files,
            "worker_bindings": {
                "parent_job_id": parent_job_id,
                "training_child_id": training_child_id,
                "nonces": [row["nonce"] for row in rows],
            },
        }
    )
    _native_checkpoint_files(root, result)
    _native_sources(result)
    return result


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
    native_layout_supplier=None,
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
    if native_layout_supplier is not None:
        expected_native_layout = _native_layout(native_layout_supplier(root, step=step))
        if expected_native_layout["world_size"] != expected_dp_world_size:
            raise ValueError("native supplier world size mismatch")
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
        raise ValueError(
            "local optimizer checkpoint metadata mismatch: "
            f"expected tag={tag!r}, observed tag={model.parent.relative_to(root).as_posix()!r}; "
            f"expected DP shard count={expected_dp_world_size}, observed DP shard count={len(shards)}"
        )
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
    native_layout_supplier=None,
    native_layout_supplier_source: dict | None = None,
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
    if native_layout_supplier is None:
        if native_layout_supplier_source is not None:
            raise ValueError("native supplier source requires its selected callback")
    else:
        if expected_native_layout is not None or not local_checkpoint:
            raise ValueError("native layout supplier requires an exclusive local capture route")
        verify_native_callback(native_layout_supplier, native_layout_supplier_source)
        native_layout_supplier_source = dict(native_layout_supplier_source)
    native = None if expected_native_layout is None else _native_layout(expected_native_layout)
    if native is not None and (not local_checkpoint or native["world_size"] != expected_dp_world_size):
        raise ValueError("native EP capture requires local full-rank checkpoints and matching world size")
    child_id = training_selector = None
    if native is not None or native_layout_supplier is not None:
        training_selector = job.client.jobs.training
        if not isinstance(training_selector, str) or not training_selector:
            raise ValueError("native EP capture requires its initialized training child")
        if native_layout_supplier is None:
            child_id = training_selector
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
        save_options = options
        if native_layout_supplier is not None:

            def supply(root, *, step):
                nonlocal native, child_id
                verify_native_callback(native_layout_supplier, native_layout_supplier_source)
                actual = native_layout_supplier(root, step=step)
                verify_native_callback(native_layout_supplier, native_layout_supplier_source)
                actual = _native_layout(actual)
                bindings = actual.get("worker_bindings")
                if (
                    actual["world_size"] != expected_dp_world_size
                    or type(bindings) is not dict
                    or set(bindings) != {"parent_job_id", "training_child_id", "nonces"}
                    or bindings["parent_job_id"] != job_id
                    or type(bindings["training_child_id"]) is not str
                    or not bindings["training_child_id"]
                    or (child_id is not None and bindings["training_child_id"] != child_id)
                    or type(bindings["nonces"]) is not list
                    or len(bindings["nonces"]) != expected_dp_world_size
                    or any(
                        type(nonce) is not str or re.fullmatch(r"[0-9a-f]{32}", nonce) is None
                        for nonce in bindings["nonces"]
                    )
                    or len(set(bindings["nonces"])) != expected_dp_world_size
                ):
                    raise ValueError("native supplier requires exact actual parent/child/worker bindings")
                if transport.job_id != job_id or job.client.jobs.training != training_selector:
                    raise ValueError("native optimizer parent/training child changed during supply")
                child_id = bindings["training_child_id"]
                if native is not None and actual != native:
                    raise ValueError("native supplier changed worker/layout bindings between saves")
                if (
                    assemble_native_optimizer_layout(
                        root,
                        step=step,
                        parent_job_id=job_id,
                        training_child_id=child_id,
                        expected_world_size=expected_dp_world_size,
                        source_files=actual["source_files"],
                    )
                    != actual
                ):
                    raise ValueError("native supplier layout differs from its completed worker save receipts")
                verify_native_callback(native_layout_supplier, native_layout_supplier_source)
                native = actual
                return actual

            save_options = {"native_layout_supplier": supply}
        if local_checkpoint:
            before = _save_local_snapshot(
                job.client, scratch / "before", step=0, expected_dp_world_size=expected_dp_world_size, **save_options
            )
            options = {} if native is None else {"expected_native_layout": native}
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
                if transport.job_id != job_id or job.client.jobs.training != training_selector:
                    raise ValueError("native optimizer parent/training child changed")
            if native_layout_supplier is not None:
                verify_native_callback(native_layout_supplier, native_layout_supplier_source)
            result = fwd_bwd_step(session, job, body, learning_rate=learning_rate)
            if native_layout_supplier is not None:
                verify_native_callback(native_layout_supplier, native_layout_supplier_source)
                _native_sources(native)
                if transport.job_id != job_id or job.client.jobs.training != training_selector:
                    raise ValueError("native optimizer parent/training child changed during target")
            after = _save_local_snapshot(
                job.client, scratch / "after", step=1, expected_dp_world_size=expected_dp_world_size, **save_options
            )
            if native is not None and (transport.job_id != job_id or job.client.jobs.training != training_selector):
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
        input_hashes = (
            None if native_layout_supplier is None else (_native_input_hashes(before), _native_input_hashes(after))
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
            if transport.job_id != job_id or job.client.jobs.training != training_selector:
                raise ValueError("native optimizer parent/training child changed")
        if native_layout_supplier is not None:
            verify_native_callback(native_layout_supplier, native_layout_supplier_source)
            record = json.loads(manifest.read_text())
            record["native_save_receipts"] = {
                label: [
                    json.loads(path.read_text())
                    for path in sorted((root / tag / "native_optimizer_layout").glob("*.json"))
                ]
                for label, root, tag in (("before", before, "global_step0"), ("after", after, "global_step1"))
            }
            if any(len(rows) != expected_dp_world_size for rows in record["native_save_receipts"].values()):
                raise ValueError("native capture requires retained complete worker save receipts")
            for root, step in ((before, 0), (after, 1)):
                if (
                    assemble_native_optimizer_layout(
                        root,
                        step=step,
                        parent_job_id=job_id,
                        training_child_id=child_id,
                        expected_world_size=expected_dp_world_size,
                        source_files=native["source_files"],
                    )
                    != native
                ):
                    raise ValueError("native capture save receipts changed before publication")
            manifest.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            if (_native_input_hashes(before), _native_input_hashes(after)) != input_hashes:
                raise ValueError("native capture checkpoint input drift before publication")
            _native_sources(native)
            verify_native_callback(native_layout_supplier, native_layout_supplier_source)
            if transport.job_id != job_id or job.client.jobs.training != training_selector:
                raise ValueError("native optimizer parent/training child changed before publication")
        relative_manifest = manifest.relative_to(decoded)
        decoded.rename(destination)
    return replace(result, optimizer_state_manifest=str(destination / relative_manifest))
