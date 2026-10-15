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
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.correctness.reference import optimizer_step
from arctic_platform.correctness.reference.optimizer_step import run_adamw_step


@pytest.fixture
def fast_output():
    with tempfile.TemporaryDirectory(prefix="optimizer-reference-selftest-", dir="/data-fast/tmp") as directory:
        yield Path(directory)


def test_adamw_reference_writes_named_fp32_moments_and_updated_parameter(fast_output: Path) -> None:
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
        output_dir=fast_output,
    )

    manifest = json.loads(artifact.manifest_path.read_text())
    assert manifest["step"] == 1
    assert [entry["name"] for entry in manifest["parameters"]] == ["weight"]
    tensors = torch.load(fast_output / manifest["parameters"][0]["file"], weights_only=True)
    expected_state = optimizer.state[expected]
    torch.testing.assert_close(tensors["parameter_update"], expected.detach() - initial)
    torch.testing.assert_close(tensors["exp_avg"], expected_state["exp_avg"])
    torch.testing.assert_close(tensors["exp_avg_sq"], expected_state["exp_avg_sq"])


def test_adamw_reference_excludes_norm_weight_from_weight_decay(fast_output: Path) -> None:
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
        output_dir=fast_output,
    )

    manifest = json.loads((fast_output / "manifest.json").read_text())
    tensors = torch.load(fast_output / manifest["parameters"][0]["file"], weights_only=True)
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


def test_adamw_reference_treats_zero_clipping_as_disabled(fast_output: Path) -> None:
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    gradient = torch.tensor([[0.25, -0.5]], dtype=torch.float32)

    artifact = run_adamw_step(
        model,
        {"weight": gradient},
        {"name": "AdamW", "weight_decay": 0.0},
        learning_rate=1e-2,
        gradient_clipping=0,
        optimizer_dtype="float32",
        output_dir=fast_output,
    )

    manifest = json.loads(artifact.manifest_path.read_text())
    tensors = torch.load(fast_output / manifest["parameters"][0]["file"], weights_only=True)
    assert torch.count_nonzero(tensors["parameter_update"]) > 0


@pytest.mark.parametrize("output", ("/code/optimizer", "/tmp/optimizer", "/data-fast", "/data-fast-sibling/optimizer"))
def test_adamw_reference_rejects_unsafe_storage_before_model_or_optimizer(output, monkeypatch, fast_output):
    before = list(fast_output.iterdir())
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        pytest.fail("model/optimizer/tensor callbacks must not run for invalid output")

    model = SimpleNamespace(named_parameters=forbidden)
    monkeypatch.setattr(optimizer_step, "master_dtype", forbidden)
    monkeypatch.setattr(optimizer_step, "build_adamw", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)
    with pytest.raises(ValueError, match="reference optimizer output.* /data-fast"):
        run_adamw_step(
            model,
            {},
            {"name": "AdamW"},
            learning_rate=0.01,
            gradient_clipping=None,
            optimizer_dtype="float32",
            output_dir=Path(output),
        )
    assert not calls
    assert list(fast_output.iterdir()) == before


def test_adamw_reference_rejects_symlink_escape_before_callbacks(monkeypatch, fast_output):
    link = fast_output / "escaped"
    link.symlink_to("/tmp", target_is_directory=True)
    before = list(fast_output.iterdir())
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        pytest.fail("model/optimizer/tensor callbacks must not run for escaped output")

    monkeypatch.setattr(optimizer_step, "master_dtype", forbidden)
    monkeypatch.setattr(optimizer_step, "build_adamw", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)
    with pytest.raises(ValueError, match="/data-fast"):
        run_adamw_step(
            SimpleNamespace(named_parameters=forbidden),
            {},
            {"name": "AdamW"},
            learning_rate=0.01,
            gradient_clipping=None,
            optimizer_dtype="float32",
            output_dir=link / "optimizer",
        )
    assert not calls
    assert list(fast_output.iterdir()) == before


def test_adamw_reference_returns_canonical_fast_manifest(fast_output):
    actual = fast_output / "actual"
    actual.mkdir()
    link = fast_output / "link"
    link.symlink_to(actual, target_is_directory=True)
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    artifact = run_adamw_step(
        model,
        {"weight": torch.tensor([[0.25, -0.5]], dtype=torch.float32)},
        {"name": "AdamW"},
        learning_rate=0.01,
        gradient_clipping=None,
        optimizer_dtype="float32",
        output_dir=link / "optimizer",
    )
    assert artifact.manifest_path == actual / "optimizer/manifest.json"
    assert artifact.manifest_path.is_file()
