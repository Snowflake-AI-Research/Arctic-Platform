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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0


"Worker-side optimizer snapshots and checkpoint tensor fingerprints."

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any
from typing import Iterator

import torch
import torch.distributed as dist


def tensor_digest(tensor: torch.Tensor, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    if chunk_bytes < tensor.element_size():
        raise ValueError("chunk_bytes must hold at least one tensor element")
    digest = hashlib.sha256()

    def update(value):
        if value.numel() * value.element_size() <= chunk_bytes:
            raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(raw))
            return
        # Split by logical dimensions to bound copies even for noncontiguous tensors.
        dimension = next(index for index, size in enumerate(value.shape) if size > 1)
        midpoint = value.shape[dimension] // 2
        update(value.narrow(dimension, 0, midpoint))
        update(value.narrow(dimension, midpoint, value.shape[dimension] - midpoint))

    update(tensor)
    return digest.hexdigest()


def model_state_manifest(engine, rank: int) -> dict:
    model = engine.module
    persistent_buffers = {
        f"{prefix}.{name}" if prefix else name
        for prefix, module in model.named_modules()
        for name in module._buffers
        if name not in module._non_persistent_buffers_set
    }
    entries = []
    for kind, tensors in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in sorted(tensors):
            entry = {
                "name": name,
                "kind": kind,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "requires_grad": bool(tensor.requires_grad),
                "group_name": getattr(tensor, "group_name", None),
                "allreduce": getattr(tensor, "allreduce", True),
                "sha256": tensor_digest(tensor),
            }
            if kind == "buffer":
                entry["persistent"] = name in persistent_buffers
            entries.append(entry)
    return {
        "rank": int(rank),
        "engine": {
            "global_steps": int(engine.global_steps),
            "has_moe_layers": bool(engine.has_moe_layers),
            "num_experts": list(engine.num_experts),
        },
        "tensors": entries,
    }


class ModelStateCaptureMixin:
    """Record local tensors immediately before save and after load when opted in."""

    _model_state_output_dir = None

    def _capture_model_state(self, phase: str) -> None:
        if self._model_state_output_dir is None:
            return
        import deepspeed

        manifest = model_state_manifest(self.engine, self.rank)
        manifest["deepspeed_version"] = deepspeed.__version__
        directory = Path(self._model_state_output_dir) / phase
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"rank-{self.rank:05d}.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"MODEL_STATE phase={phase} rank={self.rank} tensors={len(manifest['tensors'])} "
            f"has_moe_layers={manifest['engine']['has_moe_layers']} "
            f"step={manifest['engine']['global_steps']} deepspeed={manifest['deepspeed_version']}",
            flush=True,
        )

    def save_checkpoint(self, path: str, export_hf: bool = False) -> dict:
        self._capture_model_state("before-save")
        return super().save_checkpoint(path, export_hf=export_hf)

    def load_checkpoint(self, path: str) -> int:
        restored_step = super().load_checkpoint(path)
        self._capture_model_state("after-load")
        return restored_step


def compare_model_state(left: Path, right: Path, *, ranks: int) -> dict:
    expected = {f"rank-{rank:05d}.json" for rank in range(ranks)}
    for directory in (left, right):
        if {path.name for path in directory.glob("rank-*.json")} != expected:
            raise RuntimeError(f"incomplete model-state rank coverage in {directory}")
    differences = []
    compared_tensors = 0
    engines = []
    for rank in range(ranks):
        filename = f"rank-{rank:05d}.json"
        before, after = (json.loads((directory / filename).read_text()) for directory in (left, right))
        if before["rank"] != rank or after["rank"] != rank:
            raise RuntimeError(f"model-state rank mismatch in {filename}")
        engines.append({"rank": rank, "before": before["engine"], "after": after["engine"]})
        for field in ("engine", "deepspeed_version"):
            if before.get(field) != after.get(field):
                differences.append({"rank": rank, "kind": "metadata", "name": field, "fields": [field]})
        before_entries, after_entries = (
            {(entry["kind"], entry["name"]): entry for entry in manifest["tensors"]} for manifest in (before, after)
        )
        for kind, name in sorted(before_entries.keys() | after_entries.keys()):
            first, second = before_entries.get((kind, name)), after_entries.get((kind, name))
            compared_tensors += 1
            if first is None or second is None:
                fields = ["missing"]
            else:
                fields = sorted(key for key in first.keys() | second.keys() if first.get(key) != second.get(key))
            if fields:
                differences.append({"rank": rank, "kind": kind, "name": name, "fields": fields})
    return {
        "exact": not differences,
        "ranks": ranks,
        "compared_tensors": compared_tensors,
        "engines": engines,
        "differences": differences,
    }


_OPTIMIZER_CAPTURE_WORKER: Any | None = None


def _optimizer_capture_debug_config(job_config: dict) -> dict[str, Any]:
    debug_config: dict[str, Any] = {}
    for section in ("training_config", "ds_worker_config"):
        config = job_config.get(section)
        if config is None:
            continue
        if not isinstance(config, dict):
            raise TypeError(f"{section} must be a dict")
        debug = config.get("debug")
        if debug is None:
            continue
        if not isinstance(debug, dict):
            raise TypeError(f"{section}.debug must be a dict")
        debug_config.update(debug)
    return debug_config


def _publish_optimizer_capture(workers: list[Any], manifest: str) -> None:
    import ray

    path = Path(manifest)
    if not path.is_absolute():
        raise ValueError("optimizer-state manifest must be an absolute path")
    if not workers:
        raise RuntimeError("optimizer-state capture has no training workers")
    hosts = list(
        dict.fromkeys(
            node["NodeManagerAddress"] for node in ray.nodes() if node.get("Alive") and node.get("NodeManagerAddress")
        )
    )
    addresses = ray.get([worker.routable_ip.remote() for worker in workers])
    unknown = sorted(set(addresses) - set(hosts))
    if unknown:
        raise RuntimeError(f"worker addresses {unknown} are outside the Ray node addresses {hosts}")
    representatives = {}
    for worker, address in zip(workers, addresses):
        representatives.setdefault(address, worker)
    # Training workers are rank-ordered; only rank zero writes these captures.
    peers = [host for host in hosts if host != addresses[0]]
    files = ray.get(workers[0].export_node_checkpoint.remote(str(path.parent), peers))
    if path.name not in files:
        raise FileNotFoundError(f"optimizer-state manifest was not exported: {path}")
    ray.get([worker.require_checkpoint_files.remote(str(path.parent), files) for worker in representatives.values()])


def _optimizer_capture_worker_class() -> Any:
    global _OPTIMIZER_CAPTURE_WORKER
    if _OPTIMIZER_CAPTURE_WORKER is not None:
        return _OPTIMIZER_CAPTURE_WORKER

    import ray

    from arctic_platform.common import deepspeed_worker as deepspeed_worker_module

    base_worker = deepspeed_worker_module.DeepSpeedWorker.__ray_metadata__.modified_class

    @ray.remote
    class OptimizerCaptureDeepSpeedWorker(ModelStateCaptureMixin, base_worker):
        """Add optimizer-state artifact capture for correctness harness jobs only."""

        def initialize(self, master_addr: str, job_config: dict) -> bool:
            debug_config = _optimizer_capture_debug_config(job_config)
            optimizer_state_output_dir = debug_config.get("optimizer_state_output_dir")
            if optimizer_state_output_dir is not None:
                optimizer_state_output_dir = os.fspath(optimizer_state_output_dir)
                if not os.path.isabs(optimizer_state_output_dir):
                    raise ValueError("debug.optimizer_state_output_dir must be an absolute path")
            self._optimizer_state_output_dir = optimizer_state_output_dir
            model_state_output_dir = debug_config.get("model_state_output_dir")
            if model_state_output_dir is not None:
                model_state_output_dir = os.fspath(model_state_output_dir)
                if not os.path.isabs(model_state_output_dir):
                    raise ValueError("debug.model_state_output_dir must be an absolute path")
            self._model_state_output_dir = model_state_output_dir
            return super().initialize(master_addr, job_config)

        def _globalize_expert_optimizer_tensor(self, param: Any, tensor: torch.Tensor) -> torch.Tensor:
            group_name = getattr(param, "group_name", None)
            if group_name is None or getattr(param, "allreduce", True) is not False or tensor.dim() < 2:
                return tensor

            import deepspeed.utils.groups as ds_groups

            local = tensor.contiguous()
            ep_group = ds_groups._get_expert_parallel_group(group_name)
            shards = [torch.empty_like(local) for _ in range(dist.get_world_size(group=ep_group))]
            dist.all_gather(shards, local, group=ep_group)
            return torch.cat(shards, dim=0)

        def _optimizer_step_values(self) -> list[int]:
            values: set[int] = set()
            optimizer = getattr(self.engine, "optimizer", None)
            seen: set[int] = set()
            while optimizer is not None and id(optimizer) not in seen:
                seen.add(id(optimizer))
                state = getattr(optimizer, "state", None)
                if isinstance(state, dict):
                    for entry in state.values():
                        if not isinstance(entry, dict) or "step" not in entry:
                            continue
                        step = entry["step"]
                        values.add(int(step.item() if torch.is_tensor(step) else step))
                optimizer = getattr(optimizer, "optimizer", None)
            return sorted(values)

        def _full_adam_moments(self, param: torch.nn.Parameter) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            from deepspeed.utils import safe_get_full_optimizer_state

            if hasattr(param, "_hp_mapping"):
                mapping = param._hp_mapping
                fragments = getattr(mapping, "optim_fragment", None) or {}
                initialized = torch.tensor(
                    int(mapping is not None and "exp_avg" in fragments and "exp_avg_sq" in fragments),
                    device=param.device,
                )
                dist.all_reduce(initialized, op=dist.ReduceOp.MAX, group=param._dp_group)
                if not bool(initialized.item()):
                    return None, None
                return safe_get_full_optimizer_state(param, "exp_avg"), safe_get_full_optimizer_state(
                    param, "exp_avg_sq"
                )
            try:
                return safe_get_full_optimizer_state(param, "exp_avg"), safe_get_full_optimizer_state(
                    param, "exp_avg_sq"
                )
            except ValueError as error:
                if "not found in optimizer state fragment" not in str(error):
                    raise
                return None, None

        def _snapshot_optimizer_state_parameters(self) -> None:
            if self._optimizer_state_output_dir is None:
                return

            from deepspeed.utils import safe_get_full_fp32_param
            from deepspeed.utils import safe_get_full_grad

            output_dir = Path(self._optimizer_state_output_dir)
            before_dir = output_dir / "before"
            optimizer = getattr(self.engine, "optimizer", None)
            relink_optimizer_state = getattr(optimizer, "_lazy_init_hp_params_optimizer_state", None)
            if callable(relink_optimizer_state):
                optimizer._hp_optimizer_states_linked = False
                relink_optimizer_state()
            if self.rank == 0:
                shutil.rmtree(output_dir, ignore_errors=True)
                before_dir.mkdir(parents=True)
            dist.barrier()
            model = self.engine.module if hasattr(self.engine, "module") else self.engine
            accumulated_gradients = self._bf16_accumulated_gradients()
            inactive_parameters = (
                {
                    id(parameter)
                    for parameters, has_gradients in zip(
                        getattr(optimizer, "bf16_groups", []),
                        getattr(optimizer, "fp32_groups_has_gradients", []),
                        strict=True,
                    )
                    for parameter, has_gradient in zip(parameters, has_gradients, strict=True)
                    if not has_gradient and parameter.grad is None
                }
                if accumulated_gradients is not None
                else set()
            )
            for index, (name, param) in enumerate(sorted(model.named_parameters(), key=lambda item: item[0])):
                if not param.requires_grad:
                    continue
                full_parameter = safe_get_full_fp32_param(param)
                if full_parameter is None:
                    raise RuntimeError(f"optimizer-state capture found no FP32 master for {name}")
                full_parameter = self._globalize_expert_optimizer_tensor(param, full_parameter)
                gradient = (
                    accumulated_gradients.get(id(param))
                    if accumulated_gradients is not None
                    else safe_get_full_grad(param)
                )
                if gradient is None and id(param) not in inactive_parameters:
                    raise RuntimeError(f"optimizer-state capture found no reduced gradient for {name}")
                if gradient is not None:
                    gradient = self._globalize_expert_optimizer_tensor(param, gradient)
                exp_avg, exp_avg_sq = self._full_adam_moments(param)
                if exp_avg is not None:
                    exp_avg = self._globalize_expert_optimizer_tensor(param, exp_avg)
                    exp_avg_sq = self._globalize_expert_optimizer_tensor(param, exp_avg_sq)
                if self.rank == 0:
                    torch.save(
                        {
                            "name": name,
                            "parameter": param.detach().cpu(),
                            "fp32_master": full_parameter.detach().cpu(),
                            "gradient": None if gradient is None else gradient.detach().cpu(),
                            "exp_avg": None if exp_avg is None else exp_avg.detach().cpu(),
                            "exp_avg_sq": None if exp_avg_sq is None else exp_avg_sq.detach().cpu(),
                        },
                        before_dir / f"{index:06d}.pt",
                    )
            if self.rank == 0:
                (before_dir / "state.json").write_text(
                    json.dumps(
                        {
                            "engine_global_step": int(self.engine.global_steps),
                            "optimizer_steps": self._optimizer_step_values(),
                            "learning_rates": [float(value) for value in self.engine.get_lr()],
                        },
                        indent=2,
                    )
                    + "\n"
                )

        def _optimizer_state_artifact_metrics(self) -> dict[str, Any]:
            if self._optimizer_state_output_dir is None:
                return {}

            from deepspeed.utils import safe_get_full_fp32_param

            output_dir = Path(self._optimizer_state_output_dir)
            before_dir = output_dir / "before"
            optimizer = getattr(self.engine, "optimizer", None)
            relink_optimizer_state = getattr(optimizer, "_lazy_init_hp_params_optimizer_state", None)
            if callable(relink_optimizer_state):
                optimizer._hp_optimizer_states_linked = False
                relink_optimizer_state()
            model = self.engine.module if hasattr(self.engine, "module") else self.engine
            entries = []
            for index, (name, param) in enumerate(sorted(model.named_parameters(), key=lambda item: item[0])):
                if not param.requires_grad:
                    continue
                full_parameter = safe_get_full_fp32_param(param)
                exp_avg, exp_avg_sq = self._full_adam_moments(param)
                if exp_avg is None or exp_avg_sq is None:
                    if self.rank == 0:
                        (before_dir / f"{index:06d}.pt").unlink(missing_ok=True)
                    continue
                if full_parameter is None:
                    raise RuntimeError(f"optimizer-state capture found no post-step FP32 master for {name}")
                full_parameter = self._globalize_expert_optimizer_tensor(param, full_parameter)
                exp_avg = self._globalize_expert_optimizer_tensor(param, exp_avg)
                exp_avg_sq = self._globalize_expert_optimizer_tensor(param, exp_avg_sq)
                if self.rank == 0:
                    before = torch.load(before_dir / f"{index:06d}.pt", map_location="cpu", weights_only=True)
                    filename = f"{index:06d}.pt"
                    torch.save(
                        {
                            "parameter": before["parameter"],
                            "fp32_master": before["fp32_master"],
                            "gradient": before["gradient"],
                            "exp_avg_before": before["exp_avg"],
                            "exp_avg_sq_before": before["exp_avg_sq"],
                            "parameter_update": full_parameter.detach().cpu() - before["fp32_master"],
                            "exp_avg": exp_avg.detach().cpu(),
                            "exp_avg_sq": exp_avg_sq.detach().cpu(),
                        },
                        output_dir / filename,
                    )
                    entries.append({"name": name, "file": filename, "shape": list(full_parameter.shape)})
                    (before_dir / f"{index:06d}.pt").unlink()
            dist.barrier()
            if self.rank != 0:
                return {}
            before_state = json.loads((before_dir / "state.json").read_text())
            (before_dir / "state.json").unlink()
            before_dir.rmdir()
            manifest_path = output_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "step": self.engine.global_steps,
                        "before": before_state,
                        "after": {
                            "engine_global_step": int(self.engine.global_steps),
                            "optimizer_steps": self._optimizer_step_values(),
                            "learning_rates": [float(value) for value in self.engine.get_lr()],
                        },
                        "parameters": entries,
                    },
                    indent=2,
                )
                + "\n"
            )
            return {"optimizer_state_manifest": str(manifest_path)}

        def step(self, learning_rate: float | None = None) -> dict:
            self._snapshot_optimizer_state_parameters()
            result = super().step(learning_rate=learning_rate)
            result.setdefault("metrics", {}).update(self._optimizer_state_artifact_metrics())
            return result

    _OPTIMIZER_CAPTURE_WORKER = OptimizerCaptureDeepSpeedWorker
    return _OPTIMIZER_CAPTURE_WORKER


@contextlib.contextmanager
def _patched_deepspeed_worker_bindings(capture_worker: Any) -> Iterator[None]:
    from arctic_platform.common import deepspeed_worker as deepspeed_worker_module

    modules = [deepspeed_worker_module]
    for module_name in ("arctic_platform.common.ray_server", "arctic_platform.rl.ray_server"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "DeepSpeedWorker"):
            modules.append(module)

    originals = [(module, module.DeepSpeedWorker) for module in modules]
    for module, _ in originals:
        module.DeepSpeedWorker = capture_worker
    try:
        yield
    finally:
        for module, original in reversed(originals):
            module.DeepSpeedWorker = original


@contextlib.contextmanager
def optimizer_capture_worker() -> Iterator[None]:
    """Install the correctness worker overlay for jobs created inside the context."""
    from arctic_platform.common import ray_server

    # Load the compatibility re-export before patching so lazy clients cannot retain an overlay.
    compatibility_server = importlib.import_module("arctic_platform.rl.ray_server")
    server_modules = [ray_server, compatibility_server]
    originals = [(module, module.ArcticRLRayServerState, module.ArcticRLRayServer) for module in server_modules]
    capture_worker = _optimizer_capture_worker_class()
    original_server_state = ray_server.ArcticRLRayServerState
    original_server = ray_server.ArcticRLRayServer

    class OptimizerCaptureServerState(original_server_state):
        async def initialize(self, job_config_dict):
            # Ray imports the product server by reference in a fresh actor process.
            with _patched_deepspeed_worker_bindings(_optimizer_capture_worker_class()):
                return await super().initialize(job_config_dict)

    class OptimizerCaptureServer(original_server):
        async def step(self, job_id, body=None):
            result = await super().step(job_id, body)
            manifests = result.get("metrics", {}).get("optimizer_state_manifest", [])
            if isinstance(manifests, str):
                manifests = [manifests]
            for manifest in dict.fromkeys(manifests):
                _publish_optimizer_capture(self.training_workers, manifest)
            return result

    with _patched_deepspeed_worker_bindings(capture_worker):
        for module in server_modules:
            module.ArcticRLRayServerState = OptimizerCaptureServerState
            module.ArcticRLRayServer = OptimizerCaptureServer
        try:
            yield
        finally:
            for module, server_state, server in reversed(originals):
                module.ArcticRLRayServer = server
                module.ArcticRLRayServerState = server_state
