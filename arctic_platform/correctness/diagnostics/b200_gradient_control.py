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

"""Compare synchronization hooks using the unchanged frozen regression runner."""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import torch

from .b200_gradient_boundary import CONFIG
from .b200_gradient_boundary import capture_workers
from .b200_post_metric_capture import analyze_post_metric_gradients
from .b200_post_metric_capture import post_metric_capture_workers
from .b200_reduction_boundary import analyze_reduction_inputs
from .b200_reduction_boundary import reduction_capture_workers
from .b200_reference_capture import capture_reference_subprocess


def install_norm_backward_sync(model):
    candidates = [
        module
        for name, module in model.named_modules()
        if name.endswith(".norm") and ".layers." not in name and type(module).__name__ == "Qwen3_5RMSNorm"
    ]
    assert len(candidates) == 1, len(candidates)

    def synchronize_backward(module, inputs, output):
        def synchronize(gradient):
            torch.cuda.synchronize(output.device)

        output.register_hook(synchronize)

    return candidates[0].register_forward_hook(synchronize_backward)


@contextlib.contextmanager
def synchronized_workers():
    import ray

    from arctic_platform.common import deepspeed_worker
    from arctic_platform.common import ray_server
    from arctic_platform.correctness.harness.optimizer_capture import _patched_deepspeed_worker_bindings

    base = deepspeed_worker.DeepSpeedWorker.__ray_metadata__.modified_class

    @ray.remote
    class SynchronizedWorker(base):
        def initialize(self, master_addr, job_config):
            result = super().initialize(master_addr, job_config)
            self.sync_handle = install_norm_backward_sync(self.engine.module)
            return result

    original = ray_server.ArcticRLRayServerState

    class SynchronizedServer(original):
        async def initialize(self, job_config_dict):
            with _patched_deepspeed_worker_bindings(SynchronizedWorker):
                return await super().initialize(job_config_dict)

    with _patched_deepspeed_worker_bindings(SynchronizedWorker):
        ray_server.ArcticRLRayServerState = SynchronizedServer
        try:
            yield
        finally:
            ray_server.ArcticRLRayServerState = original


def main():
    from arctic_platform.correctness import __main__ as cli

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("control", "norm-sync", "capture", "reduction-capture", "bucket-capture", "post-metric-capture"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", choices=("gas1", "gas4"), default="gas1")
    args = parser.parse_args()
    assert torch.cuda.device_count() == 8
    assert "B200" in torch.cuda.get_device_name()
    args.out.mkdir(parents=True, exist_ok=False)
    with contextlib.ExitStack() as stack:
        if args.mode == "norm-sync":
            stack.enter_context(synchronized_workers())
        elif args.mode == "capture":
            stack.enter_context(capture_workers(args.out))
        elif args.mode in ("reduction-capture", "bucket-capture"):
            stack.enter_context(reduction_capture_workers(args.out))
            if args.mode == "bucket-capture":
                stack.enter_context(capture_reference_subprocess(args.out / "reference-capture"))
        elif args.mode == "post-metric-capture":
            stack.enter_context(post_metric_capture_workers(args.out))
            stack.enter_context(capture_reference_subprocess(args.out / "reference-capture"))
        print("FROZEN_CONTROL_MODE", args.mode, flush=True)
        status = cli.main(
            [
                "run",
                "--config",
                str(CONFIG),
                "--test",
                "single-step-grads",
                "--case",
                args.case,
                "--out",
                str(args.out / "report"),
            ]
        )
    print("FROZEN_CONTROL_COMPLETE", args.mode, "status", status, flush=True)
    if args.mode in ("reduction-capture", "bucket-capture"):
        analyze_reduction_inputs(args.out)
    elif args.mode == "post-metric-capture":
        analyze_post_metric_gradients(args.out)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
