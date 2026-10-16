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

"""Capture completed reduced gradients after the ordinary metric, without backward hooks."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import torch


def capture_after_metrics(original, model, full_gradient, directory: Path):
    result = original()
    gradients = {}
    for name, parameter in sorted(model.named_parameters()):
        if not (
            name.endswith("layers.0.linear_attn.out_proj.weight")
            or name.endswith(".norm.weight")
            and ".layers." not in name
        ):
            continue
        gradient = full_gradient(parameter)
        assert gradient is not None, name
        gradients[name] = gradient.detach().cpu()
    assert len(gradients) == 2, list(gradients)
    directory.mkdir(parents=True, exist_ok=False)
    torch.save(gradients, directory / "gradients.pt")
    (directory / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


@contextlib.contextmanager
def post_metric_capture_workers(directory: Path):
    import ray

    from arctic_platform.common import deepspeed_worker
    from arctic_platform.common import ray_server
    from arctic_platform.correctness.harness.optimizer_capture import _patched_deepspeed_worker_bindings

    base = deepspeed_worker.DeepSpeedWorker.__ray_metadata__.modified_class

    @ray.remote
    class PostMetricWorker(base):
        def _per_parameter_gradient_norm_metrics(self):
            from deepspeed.utils import safe_get_full_grad

            return capture_after_metrics(
                super()._per_parameter_gradient_norm_metrics,
                self.engine.module,
                safe_get_full_grad,
                directory / f"rank-{self.rank}",
            )

    original_server = ray_server.ArcticRLRayServerState

    class PostMetricServer(original_server):
        async def initialize(self, job_config_dict):
            with _patched_deepspeed_worker_bindings(PostMetricWorker):
                return await super().initialize(job_config_dict)

    with _patched_deepspeed_worker_bindings(PostMetricWorker):
        ray_server.ArcticRLRayServerState = PostMetricServer
        try:
            yield
        finally:
            ray_server.ArcticRLRayServerState = original_server


def analyze_post_metric_gradients(directory: Path):
    paths = sorted(directory.glob("rank-*/gradients.pt"))
    assert len(paths) == 8, paths
    ranks = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    references = torch.load(directory / "reference-capture" / "gradients.pt", map_location="cpu", weights_only=True)
    reference_norms = json.loads((directory / "reference-capture" / "norms.json").read_text())
    metrics = json.loads((paths[0].parent / "metrics.json").read_text())["gradient_norms_per_param"]
    summary = []
    for name, tensor in ranks[0].items():
        assert all(torch.equal(rank[name], tensor) for rank in ranks[1:]), name
        suffix = "layers.0.linear_attn.out_proj.weight" if ".layers." in name else ".norm.weight"
        names = [
            key for key in references if key.endswith(suffix) and (suffix != ".norm.weight" or ".layers." not in key)
        ]
        assert len(names) == 1, names
        reference_name = names[0]
        actual = tensor.float().to("cuda")
        reference = references[reference_name].float().to("cuda")
        item = {
            "parameter": name,
            "reference_parameter": reference_name,
            "original_ap_norm": metrics[name],
            "original_reference_norm": reference_norms[reference_name]["original_norm"],
            "ap_cuda_fp32_norm": float(actual.norm().cpu()),
            "reference_cuda_fp32_norm": float(reference.norm().cpu()),
            "ap_cuda_fp64_norm": float(actual.double().norm().cpu()),
            "reference_cuda_fp64_norm": float(reference.double().norm().cpu()),
            "full_tensor_delta_fp64_l2": float((actual.double() - reference.double()).norm().cpu()),
            "ranks_bitwise_equal": True,
        }
        summary.append(item)
        print("POST_METRIC_GRADIENT_OBSERVATIONS", json.dumps(item), flush=True)
    (directory / "post-metric-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("POST_METRIC_ANALYSIS_COMPLETE", directory, flush=True)
