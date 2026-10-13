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

"""Compare ordinary telemetry with local optimizer accumulation buffers after metrics."""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import torch

from arctic_platform.correctness.harness.names import normalize

CONFIG = "arctic_platform/correctness/configs/qwen3.6-35b-a3b/h200/train-sft-8gpus-64k.config"


def optimizer_accumulated_gradient(optimizer, parameter):
    groups = getattr(optimizer, "bf16_groups", ())
    gradients = getattr(optimizer, "fp32_groups_gradients", ())
    for group, buffers in zip(groups, gradients, strict=True):
        for candidate, gradient in zip(group, buffers, strict=True):
            if candidate is parameter:
                return gradient.view_as(parameter)
    return None


def observe_gradient_sources(original, model, full_gradient, destination: Path, *, optimizer=None):
    result = original()
    observations = []
    for name, parameter in sorted(model.named_parameters()):
        if normalize(name) not in {"layers.0.linear_attn.out_proj.weight", "norm.weight"}:
            continue
        ordinary = full_gradient(parameter)
        assert ordinary is not None, name
        accumulated = optimizer_accumulated_gradient(optimizer, parameter)
        local = parameter.grad
        observations.append(
            {
                "parameter": name,
                "parameter_dtype": str(parameter.dtype),
                "local_gradient_present": local is not None,
                "ordinary_is_local": ordinary is local,
                "ordinary_dtype": str(ordinary.dtype),
                "ordinary_norm": float(ordinary.detach().float().norm()),
                "local_norm": None if local is None else float(local.detach().float().norm()),
                "accumulated_dtype": None if accumulated is None else str(accumulated.dtype),
                "accumulated_norm": None if accumulated is None else float(accumulated.detach().float().norm()),
                "ordinary_accumulated_delta": (
                    None
                    if accumulated is None
                    else float((ordinary.detach().float() - accumulated.detach().float()).norm())
                ),
            }
        )
    assert len(observations) == 2, observations
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump({"metrics": result, "observations": observations}, stream, indent=2)
        stream.write("\n")
    return result


@contextlib.contextmanager
def gradient_source_workers(directory: Path):
    import ray

    from arctic_platform.common import deepspeed_worker
    from arctic_platform.common import ray_server
    from arctic_platform.correctness.harness.optimizer_capture import _patched_deepspeed_worker_bindings

    base = deepspeed_worker.DeepSpeedWorker.__ray_metadata__.modified_class

    @ray.remote
    class GradientSourceWorker(base):
        def _per_parameter_gradient_norm_metrics(self):
            from deepspeed.utils import safe_get_full_grad

            return observe_gradient_sources(
                super()._per_parameter_gradient_norm_metrics,
                self.engine.module,
                safe_get_full_grad,
                directory / f"gas{self.engine.gradient_accumulation_steps()}" / f"rank-{self.rank}.json",
                optimizer=self.engine.optimizer,
            )

    original_server = ray_server.ArcticRLRayServerState

    class GradientSourceServer(original_server):
        async def initialize(self, job_config_dict):
            with _patched_deepspeed_worker_bindings(GradientSourceWorker):
                return await super().initialize(job_config_dict)

    with _patched_deepspeed_worker_bindings(GradientSourceWorker):
        ray_server.ArcticRLRayServerState = GradientSourceServer
        try:
            yield
        finally:
            ray_server.ArcticRLRayServerState = original_server


def main():
    from arctic_platform.correctness import __main__ as cli

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.device_count() == 8
    assert all("H200" in torch.cuda.get_device_name(index) for index in range(8))
    args.out.mkdir(parents=True, exist_ok=False)
    with gradient_source_workers(args.out):
        status = cli.main(
            [
                "run",
                "--config",
                CONFIG,
                "--test",
                "single-step-grads",
                "--out",
                str(args.out / "report"),
            ]
        )
    for case in ("gas1", "gas4"):
        files = sorted((args.out / case).glob("rank-*.json"))
        assert len(files) == 8, (case, files)
        for path in files:
            print("DP_GRADIENT_SOURCES", case, path.name, path.read_text(), flush=True)
    for report in sorted((args.out / "report").glob("report-*.json")):
        print(
            "REPORT_JSON_BEGIN",
            report,
            report.read_text(),
            "REPORT_JSON_END",
            flush=True,
        )
    print("DP_GRADIENT_SOURCES_COMPLETE", status, args.out, flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
