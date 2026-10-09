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

"""Correctness-only DeepSpeed worker overlay for optimizer artifact capture."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

_OPTIMIZER_CAPTURE_WORKER: Any | None = None


def _optimizer_capture_worker_class() -> Any:
    global _OPTIMIZER_CAPTURE_WORKER
    if _OPTIMIZER_CAPTURE_WORKER is not None:
        return _OPTIMIZER_CAPTURE_WORKER

    import ray

    from arctic_platform.common import deepspeed_worker as deepspeed_worker_module
    from arctic_platform.common.deepspeed_worker import _worker_debug_config

    base_worker = deepspeed_worker_module.DeepSpeedWorker.__ray_metadata__.modified_class

    @ray.remote
    class OptimizerCaptureDeepSpeedWorker(base_worker):
        """Add optimizer-state artifact capture for correctness harness jobs only."""

        def initialize(self, master_addr: str, job_config: dict) -> bool:
            debug_config = _worker_debug_config(job_config)
            optimizer_state_output_dir = debug_config.get("optimizer_state_output_dir")
            if optimizer_state_output_dir is not None:
                optimizer_state_output_dir = os.fspath(optimizer_state_output_dir)
                if not os.path.isabs(optimizer_state_output_dir):
                    raise ValueError("debug.optimizer_state_output_dir must be an absolute path")
            self._optimizer_state_output_dir = optimizer_state_output_dir
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
            for index, (name, param) in enumerate(sorted(model.named_parameters(), key=lambda item: item[0])):
                if not param.requires_grad:
                    continue
                full_parameter = safe_get_full_fp32_param(param)
                if full_parameter is None:
                    raise RuntimeError(f"optimizer-state capture found no FP32 master for {name}")
                full_parameter = self._globalize_expert_optimizer_tensor(param, full_parameter)
                gradient = safe_get_full_grad(param)
                if gradient is None:
                    raise RuntimeError(f"optimizer-state capture found no reduced gradient for {name}")
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
                            "gradient": gradient.detach().cpu(),
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
def optimizer_capture_worker() -> Any:
    """Install the correctness worker overlay for jobs created inside the context."""
    from arctic_platform.common import deepspeed_worker as deepspeed_worker_module

    capture_worker = _optimizer_capture_worker_class()
    original = deepspeed_worker_module.DeepSpeedWorker
    deepspeed_worker_module.DeepSpeedWorker = capture_worker
    try:
        yield
    finally:
        deepspeed_worker_module.DeepSpeedWorker = original
