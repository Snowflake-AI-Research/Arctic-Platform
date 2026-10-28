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


"""Parameter alignment, participation, and optimizer capture."""

from __future__ import annotations

import contextlib
import dis
import fcntl
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
import types
import uuid
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from tempfile import TemporaryDirectory
from tempfile import TemporaryFile
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.distributed as dist

from arctic_platform.client import CortexConfig
from arctic_platform.correctness.harness.cortex import UnsafeCleanupError
from arctic_platform.correctness.harness.gateway import ArcticJob
from arctic_platform.correctness.harness.gateway import GatewaySession
from arctic_platform.correctness.harness.gateway import StepResult
from arctic_platform.correctness.harness.gateway import fwd_bwd_step
from arctic_platform.correctness.harness.gateway import require_local_cortex_gateway
from arctic_platform.correctness.harness.gateway import save_and_resolve_checkpoint_id
from arctic_platform.correctness.harness.storage import require_data_fast_path
from arctic_platform.debug.training import merge_worker_receipts

# Names

_CONTAINER_SEGMENTS = frozenset(
    {
        "module",
        "_forward_module",
        "_orig_mod",
        "_checkpoint_wrapped_module",
        "model",
        "language_model",
        "text_model",
        "base_model",
        "transformer",
    }
)

_ROUTED_EXPERT = re.compile(r"^(?P<prefix>.*\.mlp\.experts)\.(?P<weight>w[123])$")
_SHARED_EXPERT = re.compile(
    r"^(?P<layer>(?:.*\.)?layers\.\d+)\.(?:mlp\.)?(?:qwen_)?shared_expert\.(?P<weight>w[123])\.weight$"
)
_BARE_SHARED_EXPERT = re.compile(r"^(?P<layer>(?:.*\.)?layers\.\d+)\.mlp\.shared_expert\.(?P<weight>w[123])$")
_SHARED_NAMES = {
    "w1": "gate_proj.weight",
    "w2": "down_proj.weight",
    "w3": "up_proj.weight",
}


@dataclass
class _Entry:
    originals: List[str]
    value: float


def normalize(name: str) -> str:
    return ".".join(part for part in name.split(".") if part not in _CONTAINER_SEGMENTS)


def _shared_expert_name(name: str) -> Optional[str]:
    match = _SHARED_EXPERT.match(name)
    container = "shared_expert"
    if match is None:
        # GLM-5.3 uses bare Prime projections and a plural HF shared-experts container.
        match = _BARE_SHARED_EXPERT.match(name)
        container = "shared_experts"
    if match is None:
        return None
    return f"{match.group('layer')}.mlp.{container}.{_SHARED_NAMES[match.group('weight')]}"


def _insert(entries: Dict[str, _Entry], key: str, original: str, value: float) -> None:
    existing = entries.get(key)
    if existing is None:
        entries[key] = _Entry([original], float(value))
        return
    if existing.value == float(value):
        existing.originals.append(original)
        return
    # Do not let normalization silently overwrite two different gradients. The synthetic key cannot match
    # anything on the other side and therefore turns the collision into an explicit coverage failure.
    entries[f"{key}::alias::{original}"] = _Entry([original], float(value))


def _normalized(values: Dict[str, float]) -> Dict[str, _Entry]:
    entries: Dict[str, _Entry] = {}
    for original, value in values.items():
        _insert(entries, normalize(original), original, float(value))
    return entries


def _canonical_dss(values: Dict[str, float]) -> Dict[str, _Entry]:
    source = _normalized(values)
    canonical: Dict[str, _Entry] = {}
    routed: Dict[str, Dict[str, _Entry]] = {}

    for key, entry in source.items():
        match = _ROUTED_EXPERT.match(key)
        if match:
            weight = match.group("weight")
            prefix = match.group("prefix")
            if weight in {"w1", "w3"}:
                routed.setdefault(prefix, {})[weight] = entry
                continue
            _insert(canonical, f"{prefix}.down_proj", entry.originals[0], entry.value)
            canonical[f"{prefix}.down_proj"].originals.extend(entry.originals[1:])
            continue

        mapped = _shared_expert_name(key)
        if mapped is not None:
            _insert(canonical, mapped, entry.originals[0], entry.value)
            canonical[mapped].originals.extend(entry.originals[1:])
            continue

        if ".mlp.router.gate.weight" in key:
            mapped = key.replace(".mlp.router.gate.weight", ".mlp.gate.weight")
        elif ".shared_expert_gate.weight" in key and ".mlp.shared_expert_gate.weight" not in key:
            mapped = key.replace(".shared_expert_gate.weight", ".mlp.shared_expert_gate.weight")
        else:
            mapped = key
        _insert(canonical, mapped, entry.originals[0], entry.value)
        canonical[mapped].originals.extend(entry.originals[1:])

    for prefix, pieces in routed.items():
        if set(pieces) == {"w1", "w3"}:
            w1, w3 = pieces["w1"], pieces["w3"]
            key = f"{prefix}.gate_up_proj"
            canonical[key] = _Entry(w1.originals + w3.originals, math.hypot(w1.value, w3.value))
        else:
            for weight, entry in pieces.items():
                canonical[f"{prefix}.{weight}"] = entry
    return canonical


@dataclass(frozen=True)
class CanonicalTensorGroup:
    """Raw Arctic Platform tensor names that form one Hugging Face semantic parameter."""

    originals: Tuple[str, ...]
    concatenate_dim: Optional[int] = None


def canonical_dss_tensor_groups(names: Iterable[str]) -> Dict[str, CanonicalTensorGroup]:
    """Group Arctic Platform tensor names by their Hugging Face parameter representation."""
    canonical: Dict[str, CanonicalTensorGroup] = {}
    routed: Dict[str, Dict[str, str]] = {}

    for original in names:
        key = normalize(original)
        match = _ROUTED_EXPERT.match(key)
        if match:
            weight = match.group("weight")
            prefix = match.group("prefix")
            if weight in {"w1", "w3"}:
                routed.setdefault(prefix, {})[weight] = key
            else:
                canonical[f"{prefix}.down_proj"] = CanonicalTensorGroup((key,))
            continue

        mapped = _shared_expert_name(key)
        if mapped is not None:
            canonical[mapped] = CanonicalTensorGroup((original,))
            continue

        if ".mlp.router.gate.weight" in key:
            mapped = key.replace(".mlp.router.gate.weight", ".mlp.gate.weight")
        elif ".shared_expert_gate.weight" in key and ".mlp.shared_expert_gate.weight" not in key:
            mapped = key.replace(".shared_expert_gate.weight", ".mlp.shared_expert_gate.weight")
        else:
            mapped = key
        canonical[mapped] = CanonicalTensorGroup((key,))

    for prefix, pieces in routed.items():
        if set(pieces) == {"w1", "w3"}:
            canonical[f"{prefix}.gate_up_proj"] = CanonicalTensorGroup(
                (pieces["w1"], pieces["w3"]),
                concatenate_dim=1,
            )
        else:
            for weight, original in pieces.items():
                canonical[f"{prefix}.{weight}"] = CanonicalTensorGroup((original,))
    return canonical


def _originals(entries: Dict[str, _Entry], keys: set[str]) -> List[str]:
    return sorted(original for key in keys for original in entries[key].originals)


def align(
    dss: Dict[str, float], reference: Dict[str, float]
) -> Tuple[List[Tuple[str, float, float]], List[str], List[str]]:
    """Pair every semantic tensor and return pairs plus uncovered raw names on either side."""
    dss_by_key = _canonical_dss(dss)
    ref_by_key = _normalized(reference)
    shared = sorted(set(dss_by_key) & set(ref_by_key))
    pairs = [(key, dss_by_key[key].value, ref_by_key[key].value) for key in shared]
    only_dss_keys = set(dss_by_key) - set(ref_by_key)
    only_ref_keys = set(ref_by_key) - set(dss_by_key)
    return pairs, _originals(dss_by_key, only_dss_keys), _originals(ref_by_key, only_ref_keys)


STATE_KEYS = ("parameter_update", "exp_avg", "exp_avg_sq")
COMPARISON_STATE_KEYS = ("parameter_update_residual", "exp_avg", "exp_avg_sq")
OPTIMIZER_NORM_CRITERION = "optimizer-state-l2-norm-v1"


def validate_optimizer_norm_summary(summary: dict) -> dict:
    """Return the canonical parameter mapping after validating the complete step-one norm schema."""
    if not isinstance(summary, dict) or summary.get("criterion") != OPTIMIZER_NORM_CRITERION:
        raise ValueError(f"optimizer norms require criterion {OPTIMIZER_NORM_CRITERION}")
    if type(summary.get("step")) is not int or summary["step"] != 1:
        raise ValueError("optimizer norms must describe step 1")
    if summary.get("state_keys") != list(COMPARISON_STATE_KEYS):
        raise ValueError("optimizer norms have incomplete or incompatible state coverage")
    parameters = summary.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("optimizer norms have no parameter coverage")
    counts = {
        "parameter_count": len(parameters),
        "optimizer_values_count": len(parameters) * len(COMPARISON_STATE_KEYS),
    }
    for key, expected in counts.items():
        if type(summary.get(key)) is not int or summary[key] != expected:
            raise ValueError(f"optimizer norms have inconsistent {key}")
    for name, values in parameters.items():
        if not isinstance(name, str) or not name or normalize(name) != name:
            raise ValueError("optimizer norms require nonempty canonical parameter names")
        if not isinstance(values, dict) or set(values) != {"shape", *COMPARISON_STATE_KEYS}:
            raise ValueError(f"optimizer norms for {name} have incomplete or incompatible state coverage")
        shape = values["shape"]
        if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
            raise ValueError(f"optimizer norms for {name} have an invalid shape")
        for state in COMPARISON_STATE_KEYS:
            value = values[state]
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise ValueError(f"optimizer norm for {name}::{state} must be finite and nonnegative")
    return parameters


# Optimizer Capture


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
            self._optimizer_participation_capture = None
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
            adapter = debug_config.get("shared_peft_adapter")
            if adapter is None:
                return super().initialize(master_addr, job_config)
            from arctic_platform.correctness.harness.lora import worker_adapter_scope

            with worker_adapter_scope(adapter["object_ref"], adapter["identity"]):
                return super().initialize(master_addr, job_config)

        def _arm_optimizer_participation(self) -> None:
            if (
                self._optimizer_state_output_dir is None
                or getattr(self, "_optimizer_participation_capture", None) is not None
            ):
                return

            if self.engine.global_steps != 0:
                raise ValueError("optimizer participation must be armed before the first backward")
            model = self.engine.module
            decoder_getter = getattr(model, "get_decoder", None)
            head_getter = getattr(model, "get_output_embeddings", None)
            decoder = decoder_getter() if callable(decoder_getter) else None
            head = head_getter() if callable(head_getter) else getattr(model, "lm_head", None)
            capture = ParticipationCapture(model, decoder=decoder, head=head)
            try:
                self._optimizer_participation_before = {
                    "participation": capture.result(),
                    "candidate_weights": {
                        name: weight_identity(capture.parameters[name].detach()) for name in capture.candidates
                    },
                }
            except BaseException:
                capture.close()
                raise
            self._optimizer_participation_capture = capture

        def _close_optimizer_participation(self) -> None:
            capture = getattr(self, "_optimizer_participation_capture", None)
            if capture is not None:
                capture.close()
                self._optimizer_participation_capture = None

        def forward_backward(self, batch: dict) -> dict:
            try:
                if self._optimizer_state_output_dir is not None:
                    self._optimizer_collective_check(self._arm_optimizer_participation)
                return super().forward_backward(batch)
            except BaseException:
                self._close_optimizer_participation()
                raise

        def _optimizer_collective_check(self, callback, *, rank_zero: bool = False):
            # Share local validation and file errors before peers enter another collective.
            result, error = None, None
            try:
                if not rank_zero or self.rank == 0:
                    result = callback()
            except Exception as caught:
                error = caught
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, None if error is None else f"{type(error).__name__}: {error}")
            if error is not None:
                raise error
            for rank, message in enumerate(errors):
                if message is not None:
                    raise RuntimeError(f"optimizer capture failed on rank {rank}: {message}")
            return result

        def _optimizer_participation(self) -> tuple[dict, dict]:

            def observe():
                capture = getattr(self, "_optimizer_participation_capture", None)
                if capture is None:
                    raise ValueError("optimizer capture requires genuine pre-backward participation")
                return {
                    "participation": capture.result(),
                    "candidate_weights": {
                        name: weight_identity(capture.parameters[name].detach()) for name in capture.candidates
                    },
                }

            observed = self._optimizer_collective_check(observe)
            capture = self._optimizer_participation_capture
            rows = [None] * dist.get_world_size()
            dist.all_gather_object(rows, {"before": self._optimizer_participation_before, "after": observed})

            def merge():
                shapes, routes = {}, {}
                for name, parameter in capture.parameters.items():
                    shapes[name] = list(parameter.shape)
                    width = 1
                    group_name = getattr(parameter, "group_name", None)
                    if (
                        group_name is not None
                        and getattr(parameter, "allreduce", True) is False
                        and parameter.dim() >= 2
                    ):
                        import deepspeed.utils.groups as ds_groups

                        width = dist.get_world_size(group=ds_groups._get_expert_parallel_group(group_name))
                    routes[name] = [None] * width
                return merge_rank_participation(
                    [row["before"] for row in rows], [row["after"] for row in rows], shapes, routes
                )

            return self._optimizer_collective_check(merge)

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

        def _relink_optimizer_state(self) -> None:
            optimizer = getattr(self.engine, "optimizer", None)
            relink = getattr(optimizer, "_lazy_init_hp_params_optimizer_state", None)
            if callable(relink):
                optimizer._hp_optimizer_states_linked = False
                relink()

        def _snapshot_optimizer_state_parameters(self) -> None:
            if self._optimizer_state_output_dir is None:
                return

            from deepspeed.utils import safe_get_full_fp32_param
            from deepspeed.utils import safe_get_full_grad

            output_dir = Path(self._optimizer_state_output_dir)
            before_dir = output_dir / "before"
            self._optimizer_collective_check(self._relink_optimizer_state)

            def prepare_directory():
                shutil.rmtree(output_dir, ignore_errors=True)
                before_dir.mkdir(parents=True)

            self._optimizer_collective_check(prepare_directory, rank_zero=True)
            model = self.engine.module if hasattr(self.engine, "module") else self.engine
            accumulated_gradients = self._optimizer_collective_check(self._bf16_accumulated_gradients)
            participation, _ = self._optimizer_participation()
            self._optimizer_step_participation = participation
            for index, (name, param) in enumerate(sorted(model.named_parameters(), key=lambda item: item[0])):
                if not param.requires_grad:
                    continue
                full_parameter = safe_get_full_fp32_param(param)

                def validate_master():
                    if full_parameter is None:
                        raise RuntimeError(f"optimizer-state capture found no FP32 master for {name}")

                self._optimizer_collective_check(validate_master)
                full_parameter = self._globalize_expert_optimizer_tensor(param, full_parameter)
                gradient = (
                    accumulated_gradients.get(id(param))
                    if accumulated_gradients is not None
                    else safe_get_full_grad(param)
                )
                inactive = participation["parameters"][name]["state"] == "inactive"

                def validate_gradient():
                    if gradient is None and not inactive:
                        raise RuntimeError(f"optimizer-state capture found no reduced gradient for {name}")
                    if (
                        gradient is not None
                        and inactive
                        and (not bool(gradient.isfinite().all()) or bool(torch.count_nonzero(gradient)))
                    ):
                        raise ValueError(f"inactive parameter {name} has a nonzero or nonfinite gradient")

                self._optimizer_collective_check(validate_gradient)
                if gradient is not None:
                    gradient = self._globalize_expert_optimizer_tensor(param, gradient)
                exp_avg, exp_avg_sq = self._full_adam_moments(param)

                def validate_moments():
                    if (exp_avg is None) != (exp_avg_sq is None):
                        raise RuntimeError(f"optimizer-state capture found incomplete Adam moments for {name}")

                self._optimizer_collective_check(validate_moments)
                if exp_avg is not None:
                    exp_avg = self._globalize_expert_optimizer_tensor(param, exp_avg)
                    exp_avg_sq = self._globalize_expert_optimizer_tensor(param, exp_avg_sq)

                def save_before():
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

                self._optimizer_collective_check(save_before, rank_zero=True)

            def save_before_state():
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

            self._optimizer_collective_check(save_before_state, rank_zero=True)

        def _optimizer_state_artifact_metrics(self) -> dict[str, Any]:
            if self._optimizer_state_output_dir is None:
                return {}

            from deepspeed.utils import safe_get_full_fp32_param

            output_dir = Path(self._optimizer_state_output_dir)
            before_dir = output_dir / "before"
            self._optimizer_collective_check(self._relink_optimizer_state)
            model = self.engine.module if hasattr(self.engine, "module") else self.engine

            participation, inactive_weights = self._optimizer_participation()

            def validate_step_participation():
                if participation != self._optimizer_step_participation:
                    raise ValueError("optimizer participation changed during step")

            self._optimizer_collective_check(validate_step_participation)
            entries = []
            for index, (name, param) in enumerate(sorted(model.named_parameters(), key=lambda item: item[0])):
                if not param.requires_grad:
                    continue
                full_parameter = safe_get_full_fp32_param(param)

                def validate_master():
                    if full_parameter is None:
                        raise RuntimeError(f"optimizer-state capture found no post-step FP32 master for {name}")

                self._optimizer_collective_check(validate_master)
                exp_avg, exp_avg_sq = self._full_adam_moments(param)
                inactive = name in inactive_weights

                def validate_moments():
                    if (exp_avg is None) != (exp_avg_sq is None) or (
                        not inactive and (exp_avg is None or exp_avg_sq is None)
                    ):
                        raise RuntimeError(f"active optimizer parameter {name} has no post-step Adam moments")

                self._optimizer_collective_check(validate_moments)
                full_parameter = self._globalize_expert_optimizer_tensor(param, full_parameter)
                if exp_avg is not None:
                    exp_avg = self._globalize_expert_optimizer_tensor(param, exp_avg)
                if exp_avg_sq is not None:
                    exp_avg_sq = self._globalize_expert_optimizer_tensor(param, exp_avg_sq)

                def save_after():
                    before = torch.load(before_dir / f"{index:06d}.pt", map_location="cpu", weights_only=True)
                    if inactive:
                        inactive_weights[name]["master"] = {
                            "before": weight_identity(before["fp32_master"]),
                            "after": weight_identity(full_parameter),
                        }
                        for value in (before["exp_avg"], before["exp_avg_sq"], exp_avg, exp_avg_sq):
                            if value is not None and (
                                not bool(value.isfinite().all()) or bool(torch.count_nonzero(value))
                            ):
                                raise ValueError(
                                    f"inactive optimizer parameter {name} has nonzero or nonfinite moments"
                                )
                        (before_dir / f"{index:06d}.pt").unlink()
                        return
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

                self._optimizer_collective_check(save_after, rank_zero=True)

            def publish_manifest():
                validate_inactive_weights(participation, inactive_weights)
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
                            "participation": participation,
                            "inactive_weights": inactive_weights,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                return {"optimizer_state_manifest": str(manifest_path)}

            return self._optimizer_collective_check(publish_manifest, rank_zero=True) or {}

        def step(self, learning_rate: float | None = None) -> dict:
            try:
                self._snapshot_optimizer_state_parameters()
                result = super().step() if learning_rate is None else super().step(learning_rate=learning_rate)
                result.setdefault("metrics", {}).update(self._optimizer_state_artifact_metrics())
                return result
            finally:
                self._close_optimizer_participation()

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


# Participation

PARTICIPATION_SCHEMA = "trainable-parameter-participation-v1"


def validate_participation(record: dict) -> dict:
    if not isinstance(record, dict) or record.get("schema") != PARTICIPATION_SCHEMA:
        raise ValueError("gradient coverage requires a versioned participation inventory")
    parameters = record.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("participation inventory has no trainable parameters")
    for name, entry in parameters.items():
        if not isinstance(name, str) or not name or not isinstance(entry, dict):
            raise ValueError("participation inventory has an invalid parameter")
        shape = entry.get("shape")
        if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
            raise ValueError(f"participation shape is invalid for {name}")
        state = entry.get("state")
        if state not in {"active", "inactive"}:
            raise ValueError(f"unexpected missing gradient for {name}")
        if state == "inactive":
            evidence = entry.get("evidence")
            if not isinstance(evidence, dict):
                raise ValueError(f"inactive parameter {name} has no independent evidence")
            if not isinstance(evidence.get("module"), str) or not evidence["module"]:
                raise ValueError(f"inactive parameter {name} has invalid independent evidence")
            kind = evidence.get("kind")
            if kind == "text-decoder-bypass":
                valid = (
                    type(evidence.get("calls")) is int
                    and evidence["calls"] == 0
                    and evidence.get("outside_decoder_and_head") is True
                    and normalize(evidence["module"]) == "visual"
                )
            elif kind == "integer-selector":
                valid = (
                    type(evidence.get("calls")) is int
                    and evidence["calls"] > 0
                    and evidence.get("integer_outputs_only") is True
                    and evidence.get("module_class") == "Glm5NextTextIndexer"
                )
            else:
                valid = False
            digest = evidence.get("forward_sha256")
            if (
                not valid
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or not isinstance(evidence.get("module"), str)
                or not evidence["module"]
                or not normalize(name).startswith(normalize(evidence["module"]) + ".")
            ):
                raise ValueError(f"inactive parameter {name} has invalid independent evidence")
    return parameters


def canonical_participation(record: dict, *, backend: str) -> dict:
    if backend not in {"dss", "reference"}:
        raise ValueError(f"unknown participation backend: {backend}")
    parameters = validate_participation(record)
    normalized = {}
    for name, entry in parameters.items():
        key = normalize(name)
        if key in normalized:
            raise ValueError(f"participation inventory has duplicate canonical name {key}")
        normalized[key] = entry
    groups = canonical_dss_tensor_groups(normalized) if backend == "dss" else {name: None for name in normalized}
    covered = [name for group in groups.values() for name in group.originals] if backend == "dss" else list(groups)
    if sorted(covered) != sorted(normalized):
        raise ValueError("participation inventory has a canonical name collision")
    result = {}
    for name, group in groups.items():
        pieces = [normalized[original] for original in group.originals] if group else [normalized[name]]
        if len({entry["state"] for entry in pieces}) != 1:
            raise ValueError(f"participation disagreement within fused parameter {name}")
        shape = list(pieces[0]["shape"])
        if group and group.concatenate_dim is not None:
            dimension = group.concatenate_dim
            if any(
                len(entry["shape"]) != len(shape)
                or any(size != shape[index] for index, size in enumerate(entry["shape"]) if index != dimension)
                for entry in pieces
            ) or dimension >= len(shape):
                raise ValueError(f"participation fused shape mismatch for {name}")
            shape[dimension] = sum(entry["shape"][dimension] for entry in pieces)
        result[name] = {"shape": shape, "state": pieces[0]["state"]}
    return result


def compare_participation(target: dict, reference: dict) -> tuple[int, int]:
    actual = canonical_participation(target, backend="dss")
    expected = canonical_participation(reference, backend="reference")
    if set(actual) != set(expected):
        raise ValueError(
            "incomplete trainable inventory: "
            f"target-only={sorted(set(actual) - set(expected))[:4]}, "
            f"reference-only={sorted(set(expected) - set(actual))[:4]}"
        )
    for name in expected:
        if actual[name] != expected[name]:
            raise ValueError(f"gradient participation or shape disagreement for {name}")
    active = sum(entry["state"] == "active" for entry in expected.values())
    return active, len(expected) - active


def active_gradient_norms(record: dict, norms: dict) -> dict:
    parameters = validate_participation(record)
    if set(norms) - set(parameters):
        raise ValueError("gradient norms contain parameters outside the trainable inventory")
    active = {}
    for name, entry in parameters.items():
        value = norms.get(name)
        if value is not None and (type(value) not in {int, float} or not math.isfinite(value) or value < 0):
            raise ValueError(f"gradient norm for {name} must be finite and nonnegative")
        if entry["state"] == "active":
            if name not in norms or value is None:
                raise ValueError(f"active parameter {name} has no reduced gradient norm")
            active[name] = value
        elif name in norms and value != 0:
            raise ValueError(f"inactive parameter {name} has a nonzero gradient")
    return active


def weight_identity(tensor) -> dict:

    if not bool(tensor.isfinite().all()):
        raise ValueError("inactive weight witness contains nonfinite values")
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": tensor_digest(tensor)}


def validate_inactive_weights(record: dict, witnesses: dict) -> None:
    parameters = validate_participation(record)
    inactive = {name for name, entry in parameters.items() if entry["state"] == "inactive"}
    if not isinstance(witnesses, dict) or set(witnesses) != inactive:
        raise ValueError("inactive optimizer weights have incomplete coverage")
    for name, witness in witnesses.items():
        if not isinstance(witness, dict) or set(witness) != {"parameter", "master"}:
            raise ValueError(f"inactive optimizer weight witness is incomplete for {name}")
        for kind, values in witness.items():
            if not isinstance(values, dict) or set(values) != {"before", "after"}:
                raise ValueError(f"inactive {kind} weight witness is incomplete for {name}")
            for identity in values.values():
                digest = identity.get("sha256") if isinstance(identity, dict) else None
                if (
                    not isinstance(identity, dict)
                    or set(identity) != {"shape", "dtype", "sha256"}
                    or identity["shape"] != parameters[name]["shape"]
                    or not isinstance(identity["dtype"], str)
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ValueError(f"inactive {kind} weight identity is invalid for {name}")
            if values["before"] != values["after"]:
                raise ValueError(f"inactive {kind} weights changed for {name}")


def validate_optimizer_participation(summary: dict, *, backend: str) -> dict | None:
    record = summary.get("participation")
    if record is None:
        if "inactive_weights" in summary:
            raise ValueError("inactive optimizer weights require participation evidence")
        return None
    validate_inactive_weights(record, summary.get("inactive_weights"))
    canonical = canonical_participation(record, backend=backend)
    active = {name: entry for name, entry in canonical.items() if entry["state"] == "active"}
    norms = summary["parameters"]
    if set(norms) != set(active):
        raise ValueError("optimizer norms do not cover exactly the active trainable inventory")
    for name, entry in active.items():
        if norms[name]["shape"] != entry["shape"]:
            raise ValueError(f"optimizer participation shape mismatch for {name}")
    return canonical


def merge_rank_participation(before: list[dict], after: list[dict], shapes: dict, routes: dict) -> tuple[dict, dict]:
    """Union observed backward use across actual ranks, never infer it from zero storage."""
    if not before or len(before) != len(after):
        raise ValueError("participation capture requires every before/after rank")
    entries, witnesses = {}, {}
    for rank, (left, right) in enumerate(zip(before, after, strict=True)):
        initial = left.get("participation", {}).get("parameters")
        observed = right.get("participation", {}).get("parameters")
        if (
            left.get("participation", {}).get("schema") != PARTICIPATION_SCHEMA
            or right.get("participation", {}).get("schema") != PARTICIPATION_SCHEMA
            or not isinstance(initial, dict)
            or not isinstance(observed, dict)
            or set(initial) != set(shapes)
            or set(observed) != set(shapes)
        ):
            raise ValueError(f"rank {rank} participation inventory is incomplete")
        for name, shape in shapes.items():
            if initial[name]["shape"] != list(shape) or observed[name]["shape"] != list(shape):
                raise ValueError(f"rank {rank} participation shape drift for {name}")
            if initial[name]["state"] == "active":
                raise ValueError("participation must be armed before the first backward")
            state = observed[name]["state"]
            if state not in {"active", "inactive", "missing"}:
                raise ValueError(f"invalid rank participation for {name}")
            if state == "inactive":
                validate_participation({"schema": PARTICIPATION_SCHEMA, "parameters": {name: observed[name]}})
    for name, shape in shapes.items():
        observed = [row["participation"]["parameters"][name] for row in after]
        global_shape = list(shape)
        if len(routes[name]) > 1:
            if not global_shape:
                raise ValueError("expert participation requires a nonscalar parameter")
            global_shape[0] *= len(routes[name])
        if any(entry["state"] == "active" for entry in observed):
            entries[name] = {"shape": global_shape, "state": "active", "evidence": None}
            continue
        if any(entry["state"] != "inactive" for entry in observed):
            raise ValueError(f"unexpected missing gradient for {name} on every rank")
        if len(routes[name]) != 1:
            raise ValueError(f"inactive expert weights require global witnesses for {name}")
        identities = []
        for left, right in zip(before, after, strict=True):
            initial = left.get("candidate_weights", {}).get(name)
            final = right.get("candidate_weights", {}).get(name)
            if initial is None or final is None:
                raise ValueError(f"inactive parameter weight witness is incomplete for {name}")
            identities.append({"before": initial, "after": final})
        if any(value != identities[0] for value in identities):
            raise ValueError(f"inactive replicated parameter witnesses disagree for {name}")
        entries[name] = {**observed[0], "shape": global_shape}
        witnesses[name] = {"parameter": identities[0]}
    record = {"schema": PARTICIPATION_SCHEMA, "parameters": entries}
    validate_participation(record)
    return record, witnesses


class ParticipationCapture:
    """Observe actual autograd use; zero storage and Adam moments are not evidence."""

    def __init__(self, model, *, decoder=None, head=None):
        self.model = model
        self.parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.shapes = {name: list(parameter.shape) for name, parameter in self.parameters.items()}
        self.active = set()
        self.handles = []
        self.candidates = {}
        self.source_digests = {}
        self.forwards = []
        try:
            self._arm(model, decoder=decoder, head=head)
        except BaseException:
            self.close()
            raise

    def _arm(self, model, *, decoder, head):
        import torch

        reachable = (
            {id(parameter) for module in (decoder, head) for parameter in module.parameters()}
            if decoder is not None and head is not None
            else None
        )
        for name, parameter in self.parameters.items():
            self.handles.append(parameter.register_hook(self._backward_hook(name)))
        for name, module in model.named_modules():
            kind = None
            if type(module).__name__ == "Glm5NextTextIndexer":
                kind = "integer-selector"
            elif reachable is not None and normalize(name) == "visual":
                kind = "text-decoder-bypass"
            if kind is None:
                continue
            owned = {id(parameter) for parameter in module.parameters()}
            names = [parameter_name for parameter_name, parameter in self.parameters.items() if id(parameter) in owned]
            if kind == "text-decoder-bypass" and any(id(self.parameters[key]) in reachable for key in names):
                continue
            forward = inspect.unwrap(getattr(module.forward, "__func__", module.forward))
            source = inspect.getsourcefile(forward)
            if source is None:
                continue
            if source not in self.source_digests:
                self.source_digests[source] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
            self.forwards.append((module, forward))
            evidence = {
                "kind": kind,
                "module": name,
                "module_class": type(module).__name__,
                "forward_sha256": self.source_digests[source],
                "calls": 0,
                "integer_outputs_only": True,
                "outside_decoder_and_head": kind == "text-decoder-bypass",
            }

            def observe(_module, _inputs, output, *, evidence=evidence):
                evidence["calls"] += 1

                def integer_only(value):
                    if isinstance(value, torch.Tensor):
                        return not value.is_floating_point() and not value.is_complex() and not value.requires_grad
                    if isinstance(value, (tuple, list)):
                        return bool(value) and all(integer_only(item) for item in value)
                    return False

                evidence["integer_outputs_only"] &= integer_only(output)

            self.handles.append(module.register_forward_hook(observe))
            for parameter_name in names:
                self.candidates[parameter_name] = evidence

    def _backward_hook(self, name):
        def record(gradient):
            self.active.add(name)
            return gradient

        return record

    def result(self) -> dict:
        current = {name: parameter for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        if set(current) != set(self.parameters) or any(
            current[name] is not parameter or list(parameter.shape) != self.shapes[name]
            for name, parameter in self.parameters.items()
        ):
            raise ValueError("trainable participation inventory changed during capture")
        for source, digest in self.source_digests.items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
                raise ValueError("participation forward source drift")
        if any(
            inspect.unwrap(getattr(module.forward, "__func__", module.forward)) is not forward
            for module, forward in self.forwards
        ):
            raise ValueError("participation forward implementation changed during capture")
        entries = {}
        for name, parameter in self.parameters.items():
            evidence = self.candidates.get(name)
            justified = evidence is not None and (
                (evidence["kind"] == "text-decoder-bypass" and evidence["calls"] == 0)
                or (
                    evidence["kind"] == "integer-selector"
                    and evidence["calls"] > 0
                    and evidence["integer_outputs_only"]
                )
            )
            entries[name] = {
                "shape": list(parameter.shape),
                "state": "active" if name in self.active else "inactive" if justified else "missing",
                "evidence": dict(evidence) if justified and name not in self.active else None,
            }
        return {"schema": PARTICIPATION_SCHEMA, "parameters": entries}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.close()


# Cortex Optimizer

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


def _native_serialized_tracing_matches(selected, template):
    if (
        selected.__code__ != template.__code__
        or selected.__code__.co_filename != template.__code__.co_filename
        or selected.__code__.co_firstlineno != template.__code__.co_firstlineno
        or selected.__code__.co_qualname != template.__code__.co_qualname
        or selected.__builtins__ is not template.__builtins__
    ):
        return False
    codes, names = [template.__code__], set()
    while codes:
        code = codes.pop()
        names.update(
            instruction.argval
            for instruction in dis.get_instructions(code)
            if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME", "STORE_GLOBAL", "DELETE_GLOBAL"}
        )
        codes.extend(value for value in code.co_consts if isinstance(value, types.CodeType))
    metadata = {"__name__", "__package__", "__file__", "__builtins__"}
    dependencies = names - metadata
    # Cloudpickle retains referenced globals, not an arbitrary replacement module namespace.
    if not set(selected.__globals__) <= dependencies | metadata:
        return False
    for name in dependencies:
        if name in template.__globals__:
            if name not in selected.__globals__ or selected.__globals__[name] is not template.__globals__[name]:
                return False
        elif name in selected.__globals__:
            return False
    return all(
        selected.__globals__.get(name) == template.__globals__.get(name) for name in metadata - {"__builtins__"}
    )


def _native_callback_origin(callback):
    selected, seen = callback, set()
    while hasattr(selected, "__wrapped__"):
        if id(selected) in seen or not inspect.isfunction(selected):
            raise ValueError("native optimizer callback has an unsupported wrapper")
        seen.add(id(selected))
        wrapped = selected.__wrapped__
        if not inspect.isfunction(wrapped):
            raise ValueError("native optimizer callback has an unsupported wrapper")
        templates = [contextlib.contextmanager(wrapped)]
        tracing_template = None
        if selected.__module__ == "ray._private.worker" and selected.__name__ in {"is_initialized", "get", "put"}:
            ray = importlib.import_module("ray")
            if callback is not getattr(ray, selected.__name__, None):
                raise ValueError("native optimizer callback requires the current public Ray function")
            client_hook = importlib.import_module("ray._private.client_mode_hook")
            auto_init_hook = importlib.import_module("ray._private.auto_init_hook")
            templates.extend((client_hook.client_mode_hook(wrapped), auto_init_hook.wrap_auto_init(wrapped)))
        if selected.__globals__.get("__name__") == "ray.util.tracing.tracing_helper":
            tracing = importlib.import_module("ray.util.tracing.tracing_helper")
            factory_name = "async_span_wrapper" if inspect.iscoroutinefunction(wrapped) else "span_wrapper"
            factories = [
                code
                for code in tracing._inject_tracing_into_class.__code__.co_consts
                if isinstance(code, types.CodeType) and code.co_name == factory_name and not code.co_freevars
            ]
            if len(factories) == 1:
                # Calling the nested factory avoids the injector's mutation of method signatures.
                factory = types.FunctionType(factories[0], vars(tracing))
                tracing_template = factory(wrapped)
                templates.append(tracing_template)
        # Match executable code and every captured object, not only copied __wrapped__ metadata.
        if not any(
            (
                selected.__code__ is template.__code__
                and selected.__globals__ is template.__globals__
                or template is tracing_template
                and _native_serialized_tracing_matches(selected, template)
            )
            and selected.__defaults__ == template.__defaults__
            and selected.__kwdefaults__ == template.__kwdefaults__
            and len(selected.__closure__ or ()) == len(template.__closure__ or ())
            and all(
                actual.cell_contents is expected.cell_contents
                for actual, expected in zip(selected.__closure__ or (), template.__closure__ or (), strict=True)
            )
            for template in templates
        ):
            raise ValueError("native optimizer callback has an unsupported wrapper")
        selected = wrapped
    return selected


def verify_native_callback(callback, descriptor: dict) -> None:
    if not callable(callback) or not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise ValueError("native optimizer callback requires an exact source descriptor")
    selected = _native_callback_origin(callback)
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
        try:
            verify_native_callback(function, sources[role])
        except ValueError as error:
            raise ValueError(f"native selected source origin mismatch: {role}") from error
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

    if engine.global_steps != 0:
        raise ValueError("native participation must be armed before the first backward")
    model = engine.module
    decoder_getter = getattr(model, "get_decoder", None)
    head_getter = getattr(model, "get_output_embeddings", None)
    decoder = decoder_getter() if callable(decoder_getter) else None
    if decoder is None and hasattr(model, "model"):
        from arctic_platform.model.implementations.moe.vlm import get_language_model

        decoder = get_language_model(model)
    head = head_getter() if callable(head_getter) else getattr(model, "lm_head", None)
    capture_sources = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(inspect.getsourcefile(ParticipationCapture)), Path(inspect.getsourcefile(tensor_digest)))
    }
    participation = ParticipationCapture(model, decoder=decoder, head=head)

    def capture_evidence():
        for filename, digest in capture_sources.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                raise ValueError("native participation source drift")
        return {
            "participation": participation.result(),
            "candidate_weights": {
                name: weight_identity(participation.parameters[name].detach()) for name in participation.candidates
            },
            "participation_sources": capture_sources,
        }

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
            **capture_evidence(),
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
        "participation": participation,
        "restore": lambda: (
            vars(checkpointer).pop("save_resumable")
            if previous is missing
            else setattr(checkpointer, "save_resumable", previous)
        ),
    }
    try:
        checkpointer.save_resumable = save
    except BaseException:
        participation.close()
        del vars(worker)[_COLLECTOR_STATE]
        raise
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
        or not 0 < request["rpc_timeout_seconds"] <= 900
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
    selected_save = type(owner).save
    original_class = type(owner).__ray_actor_class__
    verify_native_callback(selected_save, training_source)
    verify_native_callback(original_class.save, training_source)
    if (
        not inspect.ismethod(owner.save)
        or owner.save.__self__ is not owner
        or owner.save.__func__ is not selected_save
        or _native_callback_origin(selected_save) is not _native_callback_origin(original_class.save)
        or owner._info.job_id != child_id
    ):
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
    state["participation"].close()
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


def _native_checkpoint_parameter_name(name: str) -> str:
    return ".".join(part for part in name.split(".") if part != "_checkpoint_wrapped_module")


def _native_checkpoint_parameter_names(parameters: dict, description: str) -> dict[str, str]:
    names = {}
    for name in parameters:
        if not isinstance(name, str):
            raise ValueError(f"{description}: parameter name is not a string")
        checkpoint_name = _native_checkpoint_parameter_name(name)
        if checkpoint_name in names:
            raise ValueError(f"{description} has duplicate checkpoint-wrapper parameter aliases: {name!r}")
        names[checkpoint_name] = name
    return names


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
        # PyTorch state_dict removes checkpoint wrappers; named_parameters retains them.
        module_names = _native_checkpoint_parameter_names(module, "native module")
        experts = [_load(path) for path in expert_files]
        for path, shard in zip(expert_files, experts, strict=True):
            if not shard or set(shard) != set(experts[0]):
                raise ValueError("inconsistent expert sidefile parameter coverage")
            for name, value in shard.items():
                reference = (
                    module.get(module_names.get(_native_checkpoint_parameter_name(name)))
                    if isinstance(name, str)
                    else None
                )
                if not isinstance(name, str):
                    reason = "parameter name is not a string"
                elif not isinstance(value, torch.Tensor):
                    reason = f"sidefile value is {type(value).__name__}, not a tensor"
                elif value.dtype != torch.bfloat16:
                    reason = f"sidefile dtype is {value.dtype}, expected {torch.bfloat16}"
                elif value.layout != torch.strided:
                    reason = f"sidefile layout is {value.layout}, expected {torch.strided}"
                elif value.ndim < 1:
                    reason = "sidefile tensor is scalar"
                elif value.numel() == 0:
                    reason = f"sidefile tensor is empty with shape {tuple(value.shape)}"
                elif not isinstance(reference, torch.Tensor):
                    reason = f"module value is {type(reference).__name__}, not a tensor"
                elif value.shape != reference.shape:
                    reason = f"sidefile shape {tuple(value.shape)} differs from module shape {tuple(reference.shape)}"
                elif reference.dtype != value.dtype:
                    reason = f"module dtype {reference.dtype} differs from sidefile dtype {value.dtype}"
                elif not bool(torch.isfinite(value).all()):
                    reason = "sidefile tensor contains nonfinite values"
                else:
                    continue
                raise ValueError(
                    f"invalid expert sidefile tensor or module shape: {path.name}, parameter {name!r}: {reason}"
                )
            _native_checkpoint_parameter_names(shard, path.name)
        if any(
            not torch.equal(value, module[module_names[_native_checkpoint_parameter_name(name)]])
            for name, value in experts[native["model_ep_rank"]].items()
        ):
            raise ValueError("rank-zero expert sidefile/module mismatch")
        _native_checkpoint_parameter_names(
            {name: None for group in shape_groups if isinstance(group, dict) for name in group},
            "native param_shapes",
        )
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
            module_name = module_names.get(_native_checkpoint_parameter_name(name)) if native is not None else name
            if isinstance(module, dict) and module_name in module and tuple(module[module_name].shape) != dimensions:
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
    participation_receipts: list[dict] | None = None,
    expected_participation_job_id: str | None = None,
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
    participation, inactive_weights = None, {}
    if participation_receipts is not None:

        if native is not None:
            raise ValueError("ordinary participation receipts cannot replace native collector evidence")
        merge_worker_receipts(
            participation_receipts,
            job_id=expected_participation_job_id,
            world_size=len(after.shards),
        )
        for receipt in participation_receipts:
            for filename, digest in receipt["after"]["participation_sources"].items():
                if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                    raise ValueError("participation capture source drift")
        shapes = {name: shape for group in after.shapes for name, shape in group.items()}
        participation, inactive_weights = merge_rank_participation(
            [receipt["before"] for receipt in participation_receipts],
            [receipt["after"] for receipt in participation_receipts],
            shapes,
            {name: [0] for name in shapes},
        )
    if native is not None:

        receipts = [
            [
                json.loads(path.read_text())
                for path in sorted((root / f"global_step{snapshot.step}" / "native_optimizer_layout").glob("*.json"))
            ]
            for root, snapshot in ((before_dir, before), (after_dir, after))
        ]
        # Older explicit collector fixtures remain readable, but cannot match a new participation-aware reference.
        if any("participation" in row for rows in receipts for row in rows):
            if any(len(rows) != native["world_size"] for rows in receipts):
                raise ValueError("native participation requires complete rank receipts")
            for left, right in zip(*receipts, strict=True):
                sources = left.get("participation_sources")
                if not sources or sources != right.get("participation_sources"):
                    raise ValueError("native participation source identity drift")
                for filename, digest in sources.items():
                    if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                        raise ValueError("native participation source drift")
            shapes = {name: shape for group in after.shapes for name, shape in group.items()}
            routes = {name: after.rank_routes[index] for index, group in enumerate(after.shapes) for name in group}
            participation, inactive_weights = merge_rank_participation(*receipts, shapes, routes)
    output_dir.mkdir(parents=True)
    try:
        entries = []
        for group_index, shapes in enumerate(after.shapes):
            for name, shape in shapes.items():
                master_before = _parameter(before, group_index, name, "fp32_master")
                master_after = _parameter(after, group_index, name, "fp32_master")
                if name in inactive_weights:

                    inactive_weights[name]["master"] = {
                        "before": weight_identity(master_before),
                        "after": weight_identity(master_after),
                    }
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
        if participation is not None:

            validate_inactive_weights(participation, inactive_weights)
            provenance.update(participation=participation, inactive_weights=inactive_weights)
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
            provenance.update(
                {
                    "native_layout": native,
                    "source_dtypes": {"master": "bfloat16", "exp_avg": "bfloat16", "exp_avg_sq": "bfloat16"},
                    "parameter_update_recipe": "FP32 difference of saved BF16 optimizer masters",
                    "expert_sidefiles_are_master_sources": False,
                    "frozen_expert_parameters": sorted(set(after.expert_shards[0]) - active_experts),
                    "runtime_provenance_is_caller_owned": True,
                }
            )
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


_DENSE_OWNER = ".correctness-dense-optimizer-owner.json"
_DENSE_RECEIPT = ".correctness-dense-optimizer-collection.json"


def validate_dense_checkpoint_allocation(admission, *, expected_dp_world_size):
    """Validate supplied same-allocation identities, not discovered or replacement nodes."""
    from arctic_platform.correctness.harness import inference as transfer

    nodes = transfer.validate_local_inference_allocation(admission)
    if type(expected_dp_world_size) is not int or sum(node["slots"] for node in nodes) != expected_dp_world_size:
        raise ValueError("dense checkpoint allocation must cover the complete DP world")
    return nodes


def _dense_root(request):
    root = Path(request["binding"]["root"])
    if root != require_data_fast_path(root, label="dense optimizer checkpoint") or root.is_symlink():
        raise ValueError("dense checkpoint requires a canonical FAST root")
    return root


def _dense_identity(path):
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise ValueError("dense checkpoint root must be an owned real directory")
    return [info.st_dev, info.st_ino, info.st_uid, info.st_mode]


def _dense_check_root(request):
    root = _dense_root(request)
    identity = _dense_identity(root)
    expected = request.get("root_identity")
    if expected is not None and identity != expected:
        raise ValueError("dense checkpoint root identity drift")
    marker = root / _DENSE_OWNER
    info = marker.lstat()
    marker_identity = [
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    ]
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1 or info.st_size > 8192:
        raise ValueError("dense checkpoint owner marker is not a bounded owned file")
    if request.get("marker_identity", marker_identity) != marker_identity:
        raise ValueError("dense checkpoint owner marker identity drift")
    with os.fdopen(os.open(marker, os.O_RDONLY | os.O_NOFOLLOW), "r") as stream:
        if (os.fstat(stream.fileno()).st_dev, os.fstat(stream.fileno()).st_ino) != (info.st_dev, info.st_ino):
            raise ValueError("dense checkpoint owner marker inode changed")
        if json.load(stream) != request["binding"]:
            raise ValueError("dense checkpoint owner binding drift")
        if os.fstat(stream.fileno()) != marker.lstat():
            raise ValueError("dense checkpoint owner marker changed")
    return root, identity


def _dense_node_admission(request, deadline):
    from arctic_platform.correctness.harness import inference as transfer
    from arctic_platform.correctness.harness.storage import parse_hostfile

    admission = request["allocation"]
    content = admission.get("hostfile_content")
    if not isinstance(content, str) or not content or len(content.encode()) > 65536:
        raise ValueError("dense checkpoint requires a bounded head hostfile snapshot")
    snapshot = content.encode()
    if hashlib.sha256(snapshot).hexdigest() != admission.get("hostfile_sha256"):
        raise ValueError("dense checkpoint head hostfile snapshot hash differs")
    with NamedTemporaryFile() as temporary:
        temporary.write(snapshot)
        temporary.flush()
        parsed = parse_hostfile(temporary.name)
    nodes = admission.get("nodes")
    if (
        not isinstance(nodes, list)
        or not nodes
        or any(not isinstance(node, dict) for node in nodes)
        or [(node.get("host"), node.get("slots")) for node in nodes] != [(row.host, row.slots) for row in parsed]
        or len({node.get("host") for node in nodes}) != len(nodes)
        or len({node.get("hostname") for node in nodes}) != len(nodes)
        or sum(node == request["node"] for node in nodes) != 1
    ):
        raise ValueError("dense checkpoint ordered head snapshot members differ from admitted nodes")
    transfer._validate_local_node(request["node"], deadline)
    try:
        transfer._LOCAL_HOSTFILE.lstat()
    except FileNotFoundError:
        if request["node"] == nodes[0]:
            raise ValueError("dense checkpoint head hostfile is required") from None
        identity = {
            "path": str(transfer._LOCAL_HOSTFILE),
            "absent": True,
            "hostfile_sha256": admission["hostfile_sha256"],
        }
        if request.get("hostfile_identity", identity) != identity:
            raise ValueError("dense checkpoint node hostfile drift")
        return identity
    identity = transfer._local_hostfile_identity()
    target = Path(identity["target_path"])
    if (
        transfer._regular_sha(target, deadline) != admission["hostfile_sha256"]
        or target.read_text() != admission["hostfile_content"]
        or transfer._local_hostfile_identity() != identity
        or request.get("hostfile_identity", identity) != identity
    ):
        raise ValueError("dense checkpoint node hostfile drift")
    return identity


def _dense_manifest(request, deadline):
    from arctic_platform.correctness.harness import inference as transfer

    root, identity = _dense_check_root(request)
    tag = root / request["binding"]["tag"]
    _dense_identity(tag)
    entries = []
    for index, path in enumerate(root.rglob("*")):
        if index >= 4096 or path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("dense checkpoint contains unsupported entries")
        if path.suffix != ".pt":
            continue
        if path.parent != tag or (path.name != "mp_rank_00_model_states.pt" and not _OPTIM_FILE.fullmatch(path.name)):
            raise ValueError("dense checkpoint contains an unsupported tensor file")
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1:
            raise ValueError("dense checkpoint requires exclusively owned regular tensor files")
        before = transfer._fingerprint(path)
        digest = transfer._regular_sha(path, deadline)
        if before != transfer._fingerprint(path) or before[2] < 1:
            raise ValueError("dense checkpoint file is empty or changed")
        entries.append({"path": path.name, "size_bytes": before[2], "sha256": digest})
    if _dense_check_root(request)[1] != identity:
        raise ValueError("dense checkpoint root changed during inventory")
    return {
        "binding": request["binding"],
        "node": request["node"],
        "files": sorted(entries, key=lambda row: row["path"]),
    }


def _dense_sources():
    from arctic_platform.correctness.harness import cortex as transport
    from arctic_platform.correctness.harness import inference as transfer
    from arctic_platform.correctness.harness import storage

    return {
        role: {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for role, path in (
            ("collector", __file__),
            ("transfer", transfer.__file__),
            ("storage", storage.__file__),
            ("cleanup_contract", transport.__file__),
        )
    }


def _dense_source_guard(request):
    if _dense_sources() != request["source_files"]:
        raise ValueError("dense checkpoint collection source drift")


def _dense_worker_argv(python, role, request):
    _dense_source_guard(request)
    root = Path(__file__).resolve().parents[3]
    checks = "".join(
        f"if hashlib.sha256(pathlib.Path({row['path']!r}).read_bytes()).hexdigest() != {row['sha256']!r}:\n"
        "    raise ValueError('dense checkpoint source drift before import')\n"
        for row in request["source_files"].values()
    )
    code = (
        f"import hashlib,pathlib,sys\nsys.path.insert(0,{str(root)!r})\n{checks}from"
        f" arctic_platform.correctness.harness import optimizer as m\nm._dense_worker({role!r},{request!r})"
    )
    return [python, "-B", "-c", code]


@contextlib.contextmanager
def _dense_argv(admission, node, role, request, deadline):
    from arctic_platform.correctness.harness import inference as transfer

    command = _dense_worker_argv(node["python"], role, request)
    if node == admission["nodes"][0]:
        yield command
    else:
        with transfer._local_ssh_argv(admission, node, {}, deadline) as argv:
            argv[-1] = shlex.join(_dense_worker_argv(node["python"], "supervise", {**request, "operation": role}))
            yield argv


@contextlib.contextmanager
def _dense_owned_processes():
    from arctic_platform.correctness.harness import inference as transfer

    children = []
    primary = None
    try:
        yield children
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            transfer._stop_local_children(children)
        except BaseException as exc:
            error = UnsafeCleanupError(
                "dense checkpoint owned process cleanup incomplete",
                owned_children={str(child.pid): {"process_group": child.pid} for child in children},
                shutdown_error=exc,
            )
            if primary is not None:
                error.preserve_workload(primary)
            raise error from exc


def _dense_call(admission, node, role, request, deadline):
    from arctic_platform.correctness.harness import inference as transfer

    with _dense_argv(admission, node, role, request, deadline) as argv, _dense_owned_processes() as children:
        child = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        children.append(child)
        output, error = child.communicate(timeout=transfer._local_remaining(deadline))
        transfer._local_remaining(deadline)
        if child.returncode:
            raise RuntimeError("dense checkpoint worker failed: " + error.decode(errors="replace")[-8000:])
        if len(output) > transfer._LOCAL_METADATA_LIMIT:
            raise ValueError("dense checkpoint worker receipt exceeds metadata bound")
        return json.loads(output, object_pairs_hook=transfer._unique_object)


def _dense_worker(role, request):
    from arctic_platform.correctness.harness import inference as transfer

    deadline = time.monotonic() + min(request["timeout_s"], request["deadline_epoch"] - time.time())
    transfer._local_remaining(deadline)
    _dense_source_guard(request)
    hostfile = _dense_node_admission(request, deadline)
    root = _dense_root(request)
    if role == "supervise":
        operation = request.get("operation")
        if operation not in {"prepare", "manifest", "send", "close"}:
            raise ValueError("unsupported dense checkpoint supervised operation")
        with _dense_owned_processes() as children:
            child = subprocess.Popen(_dense_worker_argv(sys.executable, operation, request), start_new_session=True)
            children.append(child)
            child.wait(timeout=transfer._local_remaining(deadline))
            transfer._local_remaining(deadline)
            if child.returncode:
                raise RuntimeError("dense checkpoint supervised worker failed")
        return
    if role == "prepare":
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir(mode=0o700)
        with os.fdopen(os.open(root / _DENSE_OWNER, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
            json.dump(request["binding"], stream, sort_keys=True, allow_nan=False)
        marker = (root / _DENSE_OWNER).lstat()
        result = {
            "binding": request["binding"],
            "node": request["node"],
            "root_identity": _dense_check_root(request)[1],
            "hostfile_identity": hostfile,
            "marker_identity": [
                marker.st_dev,
                marker.st_ino,
                marker.st_uid,
                marker.st_mode,
                marker.st_size,
                marker.st_mtime_ns,
                marker.st_ctime_ns,
            ],
        }
    elif role in {"manifest", "send"}:
        manifest = _dense_manifest(request, deadline)
        if role == "send":
            if manifest != request["manifest"]:
                raise ValueError("dense checkpoint changed before transfer")
            transfer._stream_local_export(root / request["binding"]["tag"], manifest, deadline)
            if _dense_manifest(request, deadline) != manifest:
                raise ValueError("dense checkpoint changed after transfer")
            if _dense_node_admission(request, deadline) != hostfile:
                raise ValueError("dense checkpoint sender node identity changed")
            _dense_source_guard(request)
            return
        result = manifest
    elif role == "close":
        if not os.path.lexists(root):
            result = {"owned_root_absent": True}
        else:
            _dense_check_root(request)
            for path in root.rglob("*"):
                if path.is_symlink() or not (path.is_file() or path.is_dir()):
                    raise ValueError("dense checkpoint cleanup refuses changed entries")
            if not shutil.rmtree.avoids_symlink_attacks:
                raise RuntimeError("dense checkpoint cleanup requires symlink-safe removal")
            shutil.rmtree(root)
            result = {"owned_root_absent": True}
    elif role == "receive":
        result = _dense_receive(request, deadline)
    else:
        raise ValueError("unknown dense checkpoint collection operation")
    if _dense_node_admission(request, deadline) != hostfile:
        raise ValueError("dense checkpoint node identity changed")
    _dense_source_guard(request)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


def _dense_validate_manifests(binding, nodes, manifests, expected_dp_world_size):
    if len(manifests) != len(nodes):
        raise ValueError("dense checkpoint requires every allocation node receipt")
    names, ranks = set(), set()
    for node, manifest in zip(nodes, manifests):
        if not isinstance(manifest, dict) or manifest.get("binding") != binding or manifest.get("node") != node:
            raise ValueError("dense checkpoint receipt allocation or child binding drift")
        files = manifest.get("files")
        if not isinstance(files, list):
            raise ValueError("dense checkpoint file inventory required")
        local_ranks = set()
        for row in files:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("path"), str)
                or row["path"] in names
                or type(row.get("size_bytes")) is not int
                or row["size_bytes"] < 1
                or not isinstance(row.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
            ):
                raise ValueError("dense checkpoint duplicate or invalid file")
            name = row["path"]
            match = _OPTIM_FILE.fullmatch(name)
            if match is not None:
                rank = int(match.group(1))
                if rank in ranks:
                    raise ValueError("dense checkpoint duplicate DP rank")
                ranks.add(rank)
                local_ranks.add(rank)
            elif name != "mp_rank_00_model_states.pt":
                raise ValueError("dense checkpoint unsupported filename")
            names.add(name)
        if len(local_ranks) != node["slots"]:
            raise ValueError("dense checkpoint node shard count differs from admitted slots")
    if ranks != set(range(expected_dp_world_size)) or "mp_rank_00_model_states.pt" not in names:
        raise ValueError("dense checkpoint requires the full DP rank set and model file")


def _dense_receive(request, deadline):
    from arctic_platform.correctness.harness import inference as transfer

    root, _ = _dense_check_root(request)
    manifest = json.loads(
        sys.stdin.buffer.readline(transfer._LOCAL_METADATA_LIMIT + 1), object_pairs_hook=transfer._unique_object
    )
    if manifest != request["manifest"]:
        raise ValueError("dense checkpoint stream manifest mismatch")
    tag = root / request["binding"]["tag"]
    if tag.exists():
        _dense_identity(tag)
    else:
        tag.mkdir(mode=0o700)
    for row in manifest["files"]:
        name = row["path"]
        if name != "mp_rank_00_model_states.pt" and _OPTIM_FILE.fullmatch(name) is None:
            raise ValueError("dense checkpoint stream filename mismatch")
        if type(row["size_bytes"]) is not int or row["size_bytes"] < 1:
            raise ValueError("dense checkpoint stream size mismatch")
        digest, remaining = hashlib.sha256(), row["size_bytes"]
        with transfer._open_exclusive(tag, name) as output:
            while remaining:
                transfer._local_remaining(deadline)
                block = sys.stdin.buffer.read(min(1024**2, remaining))
                if not block:
                    raise ValueError("truncated dense checkpoint stream")
                output.write(block)
                digest.update(block)
                remaining -= len(block)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != row["sha256"] or transfer._regular_sha(tag / name, deadline) != row["sha256"]:
            raise ValueError("dense checkpoint stream file hash mismatch")
    if sys.stdin.buffer.read(1):
        raise ValueError("extra bytes after dense checkpoint stream")
    _dense_check_root(request)
    return {"manifest_sha256": transfer.schema_sha(manifest), "files_verified": len(manifest["files"])}


def _dense_copy_peer(admission, node, request, head_request, deadline):
    from arctic_platform.correctness.harness import inference as transfer

    with TemporaryFile() as errors:
        with _dense_argv(admission, node, "send", request, deadline) as argv, _dense_owned_processes() as children:
            sender = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=errors,
                start_new_session=True,
            )
            children.append(sender)
            receiver = subprocess.Popen(
                _dense_worker_argv(sys.executable, "receive", {**head_request, "manifest": request["manifest"]}),
                stdin=sender.stdout,
                stdout=subprocess.PIPE,
                stderr=errors,
                start_new_session=True,
            )
            children.append(receiver)
            sender.stdout.close()
            output, _ = receiver.communicate(timeout=transfer._local_remaining(deadline))
            sender.wait(timeout=transfer._local_remaining(deadline))
            if sender.returncode or receiver.returncode:
                errors.seek(0)
                raise RuntimeError(
                    "dense checkpoint transfer failed: " + errors.read().decode(errors="replace")[-8000:]
                )
            expected = {
                "manifest_sha256": transfer.schema_sha(request["manifest"]),
                "files_verified": len(request["manifest"]["files"]),
            }
            if len(output) > transfer._LOCAL_METADATA_LIMIT or json.loads(output) != expected:
                raise ValueError("dense checkpoint persisted transfer receipt mismatch")


@contextlib.contextmanager
def _dense_snapshot_collection(root, admission, *, step, expected_dp_world_size, parent_job_id, training_child_id):
    from arctic_platform.correctness.harness import inference as transfer

    admission = json.loads(json.dumps(admission, allow_nan=False))
    nodes = validate_dense_checkpoint_allocation(admission, expected_dp_world_size=expected_dp_world_size)
    if (
        type(step) is not int
        or step not in (0, 1)
        or any(
            not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", value) is None
            for value in (parent_job_id, training_child_id)
        )
        or parent_job_id == training_child_id
    ):
        raise ValueError("dense checkpoint requires its exact initialized parent and training child")
    timeout = admission.get("timeout_s", 1800.0)
    reserve = admission.get("reserve_bytes", 0)
    if (
        type(timeout) not in (int, float)
        or not math.isfinite(timeout)
        or timeout <= 0
        or type(reserve) is not int
        or reserve < 0
    ):
        raise ValueError("dense checkpoint requires a finite timeout and nonnegative disk reserve")
    deadline = time.monotonic() + timeout
    host_identity = {}
    transfer._validate_local_allocation(admission, deadline, host_identity)
    binding = {
        "schema": "same-allocation-dense-optimizer-v1",
        "allocation_sha256": transfer.schema_sha(admission),
        "parent_job_id": parent_job_id,
        "training_child_id": training_child_id,
        "root": str(root),
        "tag": f"global_step{step}",
        "nonce": uuid.uuid4().hex,
    }
    owned = []
    primary = None
    proof = None
    try:
        for node in nodes:
            request = {
                "allocation": admission,
                "node": node,
                "binding": binding,
                "timeout_s": timeout,
                "deadline_epoch": time.time() + transfer._local_remaining(deadline),
                "source_files": _dense_sources(),
            }
            if owned and request["source_files"] != owned[0][1]["source_files"]:
                raise ValueError("dense checkpoint source changed before prepare")
            owned.append((node, request))
            prepared = _dense_call(admission, node, "prepare", request, deadline)
            if (
                not isinstance(prepared, dict)
                or set(prepared) != {"binding", "node", "root_identity", "hostfile_identity", "marker_identity"}
                or prepared["binding"] != binding
                or prepared["node"] != node
                or not isinstance(prepared["root_identity"], list)
                or len(prepared["root_identity"]) != 4
                or not isinstance(prepared["marker_identity"], list)
                or len(prepared["marker_identity"]) != 7
                or not isinstance(prepared["hostfile_identity"], dict)
                or not prepared["hostfile_identity"]
            ):
                raise ValueError("dense checkpoint prepare acknowledgement binding mismatch")
            request.update(prepared)

        def collect(result):
            nonlocal proof
            if result.get("job_id") != training_child_id:
                raise ValueError("dense checkpoint save returned a different training child")
            transfer._validate_local_allocation(admission, deadline, host_identity)
            _dense_source_guard(owned[0][1])
            manifests = [_dense_call(admission, node, "manifest", request, deadline) for node, request in owned]
            _dense_validate_manifests(binding, nodes, manifests, expected_dp_world_size)
            needed = sum(row["size_bytes"] for manifest in manifests[1:] for row in manifest["files"])
            if shutil.disk_usage(root).free < needed + reserve:
                raise ValueError("dense checkpoint collection has insufficient fresh disk budget")
            for (node, request), manifest in zip(owned[1:], manifests[1:]):
                _dense_copy_peer(admission, node, {**request, "manifest": manifest}, owned[0][1], deadline)
            model, shards = _checkpoint_files(root)
            expected = {row["path"]: row for manifest in manifests for row in manifest["files"]}
            for path in (model, *shards):
                row = expected[path.name]
                if path.stat().st_size != row["size_bytes"] or transfer._regular_sha(path, deadline) != row["sha256"]:
                    raise ValueError("dense checkpoint consolidated file drift")
            transfer._validate_local_allocation(admission, deadline, host_identity)
            _dense_source_guard(owned[0][1])
            proof = {
                "binding": binding,
                "node_manifests": manifests,
                "node_preparations": [
                    {key: request[key] for key in ("node", "root_identity", "marker_identity", "hostfile_identity")}
                    for _, request in owned
                ],
                "peer_cleanup": [],
                "source_files": owned[0][1]["source_files"],
            }
            return proof

        yield collect
    except BaseException as exc:
        primary = exc
        raise
    finally:
        failures = []
        for node, request in reversed(owned[1:]):
            try:
                cleanup_deadline = time.monotonic() + min(timeout, 120)
                cleanup_request = {
                    **request,
                    "timeout_s": min(timeout, 120),
                    "deadline_epoch": time.time() + min(timeout, 120),
                }
                if _dense_call(admission, node, "close", cleanup_request, cleanup_deadline) != {
                    "owned_root_absent": True
                }:
                    raise ValueError("dense checkpoint owned cleanup acknowledgement mismatch")
                if proof is not None:
                    proof["peer_cleanup"].append({"node": node, "owned_root_absent": True})
            except BaseException as exc:
                failures.append(exc)
        if failures:
            shutdown = BaseExceptionGroup("dense checkpoint owned peer cleanup incomplete", failures)
            error = UnsafeCleanupError(
                "dense checkpoint owned peer cleanup incomplete",
                owned_parent=parent_job_id,
                owned_children={"training": training_child_id, "peer_roots": [binding["root"]]},
                shutdown_error=shutdown,
            )
            if primary is not None:
                error.preserve_workload(primary)
            raise error from shutdown
        if primary is None and proof is not None:
            _dense_check_root(owned[0][1])
            _dense_source_guard(owned[0][1])
            with (root / _DENSE_RECEIPT).open("x") as stream:
                json.dump(proof, stream, sort_keys=True, allow_nan=False)


def _dense_verified_receipt(root, admission, *, step, parent_job_id, training_child_id, source_files):
    from arctic_platform.correctness.harness import inference as transfer

    proof = transfer.read_export_json(root / _DENSE_RECEIPT)
    binding = proof.get("binding")
    if (
        not isinstance(binding, dict)
        or set(binding)
        != {
            "schema",
            "allocation_sha256",
            "parent_job_id",
            "training_child_id",
            "root",
            "tag",
            "nonce",
        }
        or binding["schema"] != "same-allocation-dense-optimizer-v1"
        or binding["allocation_sha256"] != transfer.schema_sha(admission)
        or binding["parent_job_id"] != parent_job_id
        or binding["training_child_id"] != training_child_id
        or binding["root"] != str(root)
        or binding["tag"] != f"global_step{step}"
        or not isinstance(binding["nonce"], str)
        or re.fullmatch(r"[0-9a-f]{32}", binding["nonce"]) is None
        or proof.get("source_files") != source_files
    ):
        raise ValueError("dense optimizer retained receipt binding drift")
    _dense_source_guard(proof)
    nodes = admission["nodes"]
    _dense_validate_manifests(binding, nodes, proof["node_manifests"], sum(node["slots"] for node in nodes))
    preparations = proof.get("node_preparations")
    if (
        not isinstance(preparations, list)
        or len(preparations) != len(nodes)
        or any(row.get("node") != node for row, node in zip(preparations, nodes))
        or proof.get("peer_cleanup") != [{"node": node, "owned_root_absent": True} for node in reversed(nodes[1:])]
    ):
        raise ValueError("dense optimizer retained ownership or peer cleanup drift")
    _dense_check_root({**preparations[0], "binding": binding})
    expected = {row["path"]: row for manifest in proof["node_manifests"] for row in manifest["files"]}
    model, shards = _checkpoint_files(root)
    deadline = time.monotonic() + admission.get("timeout_s", 1800)
    for path in (model, *shards):
        row = expected[path.name]
        if path.stat().st_size != row["size_bytes"] or transfer._regular_sha(path, deadline) != row["sha256"]:
            raise ValueError("dense optimizer retained checkpoint byte drift")
    return proof


def _save_local_snapshot(
    client,
    root: Path,
    *,
    step: int,
    expected_dp_world_size: int,
    expected_native_layout: dict | None = None,
    native_layout_supplier=None,
    dense_checkpoint_allocation: dict | None = None,
    parent_job_id: str | None = None,
    training_child_id: str | None = None,
) -> Path:

    root = require_data_fast_path(root, label="local optimizer checkpoint")
    if root.exists() or root.is_symlink():
        raise FileExistsError("local optimizer checkpoint must be new")
    if dense_checkpoint_allocation is not None and (
        expected_native_layout is not None or native_layout_supplier is not None
    ):
        raise ValueError("dense collection cannot substitute for a native EP supplier")
    scope = (
        contextlib.nullcontext(None)
        if dense_checkpoint_allocation is None
        else _dense_snapshot_collection(
            root,
            dense_checkpoint_allocation,
            step=step,
            expected_dp_world_size=expected_dp_world_size,
            parent_job_id=parent_job_id,
            training_child_id=training_child_id,
        )
    )
    with scope as collect:
        return _save_local_snapshot_files(
            client,
            root,
            step=step,
            expected_dp_world_size=expected_dp_world_size,
            expected_native_layout=expected_native_layout,
            native_layout_supplier=native_layout_supplier,
            collect=collect,
        )


def _save_local_snapshot_files(
    client,
    root,
    *,
    step,
    expected_dp_world_size,
    expected_native_layout,
    native_layout_supplier,
    collect,
):
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
    if collect is not None:
        collect(result)
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


def _dense_training_child(transport, selector):
    from arctic_platform.correctness.harness.cortex import CorrectnessCortexTransport

    if not isinstance(transport, CorrectnessCortexTransport) or not isinstance(selector, str) or not selector:
        raise ValueError("dense optimizer requires its owned native training child binding")
    bindings = getattr(transport, "_native_bindings", None)
    children = getattr(transport, "_native_children", None)
    binding = bindings.get(selector) if type(bindings) is dict else None
    if (
        type(binding) is not tuple
        or len(binding) != 2
        or binding[0] != "training"
        or not isinstance(binding[1], str)
        or not binding[1]
        or type(children) is not dict
        or children.get("training") != binding[1]
    ):
        raise ValueError("dense optimizer requires its owned native training child binding")
    return binding[1]


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
    dense_checkpoint_allocation: dict | None = None,
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
    if dense_checkpoint_allocation is not None:
        if not local_checkpoint or native is not None or native_layout_supplier is not None:
            raise ValueError("dense collection requires an exclusive local dense capture route")
        validate_dense_checkpoint_allocation(
            dense_checkpoint_allocation, expected_dp_world_size=expected_dp_world_size
        )
        dense_checkpoint_allocation = json.loads(json.dumps(dense_checkpoint_allocation, allow_nan=False))
        dense_selector = job.client.jobs.training
        dense_child_id = _dense_training_child(transport, dense_selector)
        dense_sources = _dense_sources()

        def verify_dense_binding():
            if (
                job.client.transport is not transport
                or transport.job_id != job_id
                or job.client.jobs.training != dense_selector
                or _dense_training_child(transport, dense_selector) != dense_child_id
            ):
                raise ValueError("dense optimizer parent/training child changed")
            _dense_source_guard({"source_files": dense_sources})

    output_dir = require_data_fast_path(output_dir, label="managed optimizer capture output")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "states"
    if destination.exists():
        raise FileExistsError("managed optimizer capture output already exists")
    snapshots = TemporaryDirectory(prefix="snapshots-", dir=output_dir, delete=False)
    with snapshots as scratch, contextlib.ExitStack() as cleanup:
        cleanup.callback(snapshots.cleanup)
        scratch = Path(scratch)
        options = {} if native is None else {"expected_native_layout": native}
        save_options = options
        if dense_checkpoint_allocation is not None:
            save_options = {
                "dense_checkpoint_allocation": dense_checkpoint_allocation,
                "parent_job_id": job_id,
                "training_child_id": dense_child_id,
            }
        if native_layout_supplier is not None:

            def supply(root, *, step):
                nonlocal native, child_id
                verify_native_callback(native_layout_supplier, native_layout_supplier_source)
                try:
                    actual = native_layout_supplier(root, step=step)
                except BaseException as error:
                    # A failed supplier RPC can leave its owner reading these inputs.
                    cleanup.pop_all()
                    error.add_note(
                        f"Native optimizer supplier did not confirm completion for step {step}; "
                        f"checkpoint inputs retained at {scratch}; no optimizer states published"
                    )
                    raise
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
            if dense_checkpoint_allocation is not None:
                verify_dense_binding()
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
            if dense_checkpoint_allocation is not None:
                verify_dense_binding()
            result = fwd_bwd_step(session, job, body, learning_rate=learning_rate)
            if dense_checkpoint_allocation is not None:
                verify_dense_binding()
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
            from arctic_platform.correctness.harness.inference import export_checkpoint

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
        if dense_checkpoint_allocation is not None:
            dense_receipts = {
                label: _dense_verified_receipt(
                    root,
                    dense_checkpoint_allocation,
                    step=step,
                    parent_job_id=job_id,
                    training_child_id=dense_child_id,
                    source_files=dense_sources,
                )
                for label, root, step in (("before", before, 0), ("after", after, 1))
            }
        decoded = scratch / "decoded"
        participation_options = {}
        if result.participation is not None and "rank_receipts" in result.participation:
            actual_child = _dense_training_child(transport, job.client.jobs.training)
            if (
                job.client.transport is not transport
                or transport.job_id != job_id
                or result.participation.get("training_child_id") != actual_child
            ):
                raise ValueError("participation parent/training child changed during capture")
            participation_options = {
                "participation_receipts": result.participation["rank_receipts"],
                "expected_participation_job_id": actual_child,
            }
        manifest = Path(
            decode_optimizer_capture(
                before,
                after,
                decoded,
                expected_optimizer_config=optimizer_config,
                expected_learning_rate=learning_rate,
                **options,
                **participation_options,
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
        if dense_checkpoint_allocation is not None:
            verify_dense_binding()
            record = json.loads(manifest.read_text())
            record["dense_save_receipts"] = {
                label: _dense_verified_receipt(
                    root,
                    dense_checkpoint_allocation,
                    step=step,
                    parent_job_id=job_id,
                    training_child_id=dense_child_id,
                    source_files=dense_sources,
                )
                for label, root, step in (("before", before, 0), ("after", after, 1))
            }
            if record["dense_save_receipts"] != dense_receipts:
                raise ValueError("dense optimizer save receipts changed during decoding")
            manifest.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        relative_manifest = manifest.relative_to(decoded)
        decoded.rename(destination)
    return replace(result, optimizer_state_manifest=str(destination / relative_manifest))
