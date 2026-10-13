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

"""Observe parameter hooks and ZeRO reduction inputs without CPU copies during backward."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import torch


def install_reduction_input_capture(model, optimizer):
    selected = {
        id(parameter): (name, parameter)
        for name, parameter in model.named_parameters()
        if name.endswith("layers.0.linear_attn.out_proj.weight")
        or (name.endswith(".norm.weight") and ".layers." not in name)
    }
    assert len(selected) == 2, [name for name, _ in selected.values()]
    captures = {name: {"local": [], "reduction_inputs": []} for name, _ in selected.values()}
    handles = []
    original = optimizer.get_gradient_for_reduction
    original_reduce = getattr(optimizer, "reduce_ipg_grads", None)
    bucket_events = []

    for name, parameter in selected.values():

        def record(gradient, name=name):
            captures[name]["local"].append(gradient.detach().clone())

        handles.append(parameter.register_hook(record))

    def reduction_input(parameter):
        result = original(parameter)
        if id(parameter) in selected:
            name, _ = selected[id(parameter)]
            accumulated = optimizer.get_param_gradient_attribute(parameter)
            communication_dtype = optimizer.get_param_comm_dtype(parameter)
            bucket = optimizer.ipg_buckets[communication_dtype]
            captures[name]["reduction_inputs"].append(
                {
                    "accumulated": None if accumulated is None else accumulated.detach().clone(),
                    "returned": None if result is None else result.detach().clone(),
                    "communication_dtype": str(communication_dtype),
                    "bucket_dtypes": [str(buffer.dtype) for buffer in bucket.buffer],
                    "caller": sys._getframe(1).f_code.co_name,
                    "bucket_elements": bucket.elements,
                    "gradient_ready": getattr(parameter, "ds_grad_is_ready", True),
                }
            )
        return result

    optimizer.get_gradient_for_reduction = reduction_input

    if original_reduce is not None:
        parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}

        def reduce_ipg_grads(comm_dtype=None):
            dtypes = [comm_dtype] if comm_dtype is not None else list(optimizer.ipg_buckets)
            for dtype in dtypes:
                bucket = optimizer.ipg_buckets[dtype]
                offset = 0
                parameters = []
                selected_offsets = {}
                for group_index, parameter_index, _ in bucket.params:
                    parameter = optimizer.bit16_groups[group_index][parameter_index]
                    name = parameter_names[id(parameter)]
                    parameters.append(name)
                    if id(parameter) in selected:
                        selected_offsets[name] = offset
                    offset += parameter.numel()
                if selected_offsets:
                    bucket_events.append(
                        {
                            "communication_dtype": str(dtype),
                            "elements": bucket.elements,
                            "index": bucket.index,
                            "parameter_names": parameters,
                            "selected_offsets": selected_offsets,
                        }
                    )
            return original_reduce(comm_dtype=comm_dtype)

        optimizer.reduce_ipg_grads = reduce_ipg_grads

    def remove():
        optimizer.get_gradient_for_reduction = original
        if original_reduce is not None:
            optimizer.reduce_ipg_grads = original_reduce
        for handle in handles:
            handle.remove()

    for values in captures.values():
        values["bucket_events"] = bucket_events
    return captures, remove


@contextlib.contextmanager
def reduction_capture_workers(directory: Path):
    import ray

    from arctic_platform.common import deepspeed_worker
    from arctic_platform.common import ray_server
    from arctic_platform.correctness.harness.optimizer_capture import _patched_deepspeed_worker_bindings

    base = deepspeed_worker.DeepSpeedWorker.__ray_metadata__.modified_class

    @ray.remote
    class ReductionBoundaryWorker(base):
        def initialize(self, master_addr, job_config):
            result = super().initialize(master_addr, job_config)
            self.capture_dir = directory / f"rank-{self.rank}"
            self.capture_dir.mkdir(parents=True, exist_ok=False)
            self.reduction_capture, self.remove_capture = install_reduction_input_capture(
                self.engine.module, self.engine.optimizer
            )
            return result

        def step(self, learning_rate=None):
            from deepspeed.utils import safe_get_full_grad

            parameters = dict(self.engine.module.named_parameters())
            snapshots = {}
            for name, captures in sorted(self.reduction_capture.items()):
                parameter = parameters[name]
                reduced = safe_get_full_grad(parameter)
                snapshots[name] = {
                    "local": [gradient.cpu() for gradient in captures["local"]],
                    "bucket_events": captures["bucket_events"],
                    "reduction_inputs": [
                        {
                            **entry,
                            "accumulated": None if entry["accumulated"] is None else entry["accumulated"].cpu(),
                            "returned": None if entry["returned"] is None else entry["returned"].cpu(),
                        }
                        for entry in captures["reduction_inputs"]
                    ],
                    "reduced": None if reduced is None else reduced.detach().cpu(),
                }
            torch.save(snapshots, self.capture_dir / "reduction-inputs.pt")
            print(
                "REDUCTION_INPUT_CAPTURE",
                self.rank,
                {name: [len(values["local"]), len(values["reduction_inputs"])] for name, values in snapshots.items()},
                flush=True,
            )
            self.remove_capture()
            return super().step(learning_rate=learning_rate)

    original_server = ray_server.ArcticRLRayServerState

    class ReductionBoundaryServer(original_server):
        async def initialize(self, job_config_dict):
            with _patched_deepspeed_worker_bindings(ReductionBoundaryWorker):
                return await super().initialize(job_config_dict)

    with _patched_deepspeed_worker_bindings(ReductionBoundaryWorker):
        ray_server.ArcticRLRayServerState = ReductionBoundaryServer
        try:
            yield
        finally:
            ray_server.ArcticRLRayServerState = original_server


def analyze_reduction_inputs(directory: Path):
    paths = sorted(directory.glob("rank-*/reduction-inputs.pt"))
    assert len(paths) == 8, paths
    ranks = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    reference_path = directory / "reference-capture" / "gradients.pt"
    references = torch.load(reference_path, map_location="cpu", weights_only=True) if reference_path.exists() else None
    reference_norms = (
        json.loads((reference_path.parent / "norms.json").read_text()) if references is not None else None
    )
    summary = []
    for name in sorted(ranks[0]):
        local_sums = [sum(gradient.float() for gradient in rank[name]["local"]) for rank in ranks]
        mean = sum(gradient / len(ranks) for gradient in local_sums)
        reduced = ranks[0][name]["reduced"].float()
        item = {
            "parameter": name,
            "local_mean_norm": float(mean.norm()),
            "bf16_rounded_mean_norm": float(mean.bfloat16().float().norm()),
            "reduced_norm": float(reduced.norm()),
            "reduced_vs_mean_l2": float((reduced - mean).norm()),
            "ranks": [],
        }
        if references is not None:
            suffix = "layers.0.linear_attn.out_proj.weight" if ".layers." in name else ".norm.weight"
            reference_names = [
                key
                for key in references
                if key.endswith(suffix) and (suffix != ".norm.weight" or ".layers." not in key)
            ]
            assert len(reference_names) == 1, reference_names
            reference_name = reference_names[0]
            tensors = {
                "reference": references[reference_name].float(),
                "local_mean": mean,
                "bf16_rounded_mean": mean.bfloat16().float(),
                "reduced": reduced,
            }
            measurements = {}
            reference_cuda = tensors["reference"].to("cuda")
            for label, tensor in tensors.items():
                cuda_tensor = tensor.to("cuda")
                measurements[label] = {
                    "cpu_fp32_norm": float(tensor.norm()),
                    "cuda_fp32_norm": float(cuda_tensor.norm().cpu()),
                    "cuda_fp64_norm": float(cuda_tensor.double().norm().cpu()),
                    "difference_from_reference_fp64_l2": float(
                        (cuda_tensor.double() - reference_cuda.double()).norm().cpu()
                    ),
                }
            item["matching_observations"] = {
                "reference_parameter": reference_name,
                "original_reference_norms": reference_norms[reference_name],
                "tensors": measurements,
            }
            print("MATCHING_GRADIENT_OBSERVATIONS", json.dumps(item["matching_observations"]), flush=True)
        for path, rank, local_sum in zip(paths, ranks, local_sums):
            inputs = rank[name]["reduction_inputs"]
            entries = []
            for entry in inputs:
                accumulated, returned = entry["accumulated"], entry["returned"]
                entries.append(
                    {
                        "communication_dtype": entry["communication_dtype"],
                        "bucket_dtypes": entry["bucket_dtypes"],
                        "accumulated_dtype": None if accumulated is None else str(accumulated.dtype),
                        "returned_dtype": None if returned is None else str(returned.dtype),
                        "caller": entry.get("caller"),
                        "bucket_elements": entry.get("bucket_elements"),
                        "gradient_ready": entry.get("gradient_ready"),
                        "accumulated_vs_local_l2": (
                            None if accumulated is None else float((accumulated.float() - local_sum).norm())
                        ),
                        "returned_vs_local_l2": (
                            None if returned is None else float((returned.float() - local_sum).norm())
                        ),
                    }
                )
            item["ranks"].append(
                {
                    "rank": path.parent.name,
                    "local_hook_count": len(rank[name]["local"]),
                    "bucket_events": rank[name].get("bucket_events", []),
                    "reduction_inputs": entries,
                }
            )
        summary.append(item)
        print("REDUCTION_INPUT_ANALYSIS", json.dumps(item), flush=True)
    (directory / "reduction-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("REDUCTION_INPUT_ANALYSIS_COMPLETE", directory, flush=True)
