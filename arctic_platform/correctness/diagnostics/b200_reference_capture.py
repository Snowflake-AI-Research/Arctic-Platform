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

"""Save selected reference gradients only after the ordinary norm calculation."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def capturing_gradient_norms(original, directory: Path):
    def gradient_norms(gradients, device):
        result = original(gradients, device)
        selected = {
            name: tensor
            for name, tensor in gradients.items()
            if name.endswith("layers.0.linear_attn.out_proj.weight")
            or (name.endswith(".norm.weight") and ".layers." not in name)
        }
        assert len(selected) == 2, list(selected)
        directory.mkdir(parents=True, exist_ok=False)
        torch.save(selected, directory / "gradients.pt")
        measurements = {
            name: {
                "original_norm": result[name],
                "cpu_fp32_norm": float(tensor.float().cpu().norm()),
                "device_fp64_norm": float(tensor.to(device=device, dtype=torch.float64).norm().cpu()),
            }
            for name, tensor in selected.items()
        }
        (directory / "norms.json").write_text(json.dumps(measurements, indent=2) + "\n")
        print("REFERENCE_GRADIENT_CAPTURE", json.dumps(measurements), flush=True)
        return result

    return gradient_norms


@contextlib.contextmanager
def capture_reference_subprocess(directory: Path):
    from arctic_platform.correctness.harness import runner

    original = runner.subprocess

    def run(command, **kwargs):
        if command[1:3] == ["-m", "arctic_platform.correctness.reference"]:
            command = list(command)
            command[2] = __name__
            kwargs["env"] = {**kwargs["env"], "B200_REFERENCE_CAPTURE_DIR": str(directory)}
        return original.run(command, **kwargs)

    runner.subprocess = SimpleNamespace(run=run)
    try:
        yield
    finally:
        runner.subprocess = original


def main():
    import os

    from arctic_platform.correctness.reference import __main__ as cli
    from arctic_platform.correctness.reference import hf_single_gpu

    original = hf_single_gpu.gradient_norms
    hf_single_gpu.gradient_norms = capturing_gradient_norms(original, Path(os.environ["B200_REFERENCE_CAPTURE_DIR"]))
    try:
        cli.main()
    finally:
        hf_single_gpu.gradient_norms = original


if __name__ == "__main__":
    main()
