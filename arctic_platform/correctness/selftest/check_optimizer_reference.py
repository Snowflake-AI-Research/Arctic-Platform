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

from __future__ import annotations

import json
from pathlib import Path

import torch

from arctic_platform.correctness.reference.optimizer_step import run_adamw_step


def test_adamw_reference_writes_named_fp32_moments_and_updated_parameter(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.bfloat16))
    gradient = torch.tensor([[0.25, -0.5], [0.75, -1.0]], dtype=torch.float32)
    config = {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.1}

    initial = model.weight.detach().float().clone()
    expected = torch.nn.Parameter(initial.clone())
    expected.grad = gradient.clone()
    optimizer = torch.optim.AdamW(
        [{"params": [expected], "weight_decay": 0.1}],
        lr=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
        fused=False,
    )
    optimizer.step()

    artifact = run_adamw_step(
        model,
        {"weight": gradient},
        config,
        learning_rate=1e-2,
        gradient_clipping=None,
        optimizer_dtype="float32",
        output_dir=tmp_path,
    )

    manifest = json.loads(artifact.manifest_path.read_text())
    assert manifest["step"] == 1
    assert [entry["name"] for entry in manifest["parameters"]] == ["weight"]
    tensors = torch.load(tmp_path / manifest["parameters"][0]["file"], weights_only=True)
    expected_state = optimizer.state[expected]
    torch.testing.assert_close(tensors["parameter_update"], expected.detach() - initial)
    torch.testing.assert_close(tensors["exp_avg"], expected_state["exp_avg"])
    torch.testing.assert_close(tensors["exp_avg_sq"], expected_state["exp_avg_sq"])


def test_adamw_reference_excludes_norm_weight_from_weight_decay(tmp_path: Path) -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = torch.nn.LayerNorm(2, elementwise_affine=True, bias=False, dtype=torch.bfloat16)

    model = Model()
    gradient = torch.tensor([0.25, -0.5], dtype=torch.float32)

    run_adamw_step(
        model,
        {"norm.weight": gradient},
        {"name": "AdamW", "weight_decay": 1.0},
        learning_rate=1e-2,
        gradient_clipping=None,
        optimizer_dtype="float32",
        output_dir=tmp_path,
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    tensors = torch.load(tmp_path / manifest["parameters"][0]["file"], weights_only=True)
    initial = model.norm.weight.detach().float().clone()
    expected = torch.nn.Parameter(initial.clone())
    expected.grad = gradient.clone()
    optimizer = torch.optim.AdamW(
        [{"params": [expected], "weight_decay": 0.0}],
        lr=1e-2,
        foreach=False,
        fused=False,
    )
    optimizer.step()
    torch.testing.assert_close(tensors["parameter_update"], expected.detach() - initial)


def test_adamw_reference_treats_zero_clipping_as_disabled(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    gradient = torch.tensor([[0.25, -0.5]], dtype=torch.float32)

    artifact = run_adamw_step(
        model,
        {"weight": gradient},
        {"name": "AdamW", "weight_decay": 0.0},
        learning_rate=1e-2,
        gradient_clipping=0,
        optimizer_dtype="float32",
        output_dir=tmp_path,
    )

    manifest = json.loads(artifact.manifest_path.read_text())
    tensors = torch.load(tmp_path / manifest["parameters"][0]["file"], weights_only=True)
    assert torch.count_nonzero(tensors["parameter_update"]) > 0
