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

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.correctness.checks.fwd_bwd_step import _compare_arm
from arctic_platform.correctness.checks.optimizer_state import compare_optimizer_artifacts
from arctic_platform.correctness.checks.optimizer_state import max_pairwise_optimizer_delta
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.runner import OPTIMIZER_LEARNING_RATE
from arctic_platform.correctness.harness.spec import TestTolerance as Tolerance

COMPARE_KWARGS = {"optimizer_config": {}, "learning_rate": OPTIMIZER_LEARNING_RATE}


def _write_artifact(root: Path, name: str, tensors: dict) -> Path:
    root.mkdir()
    torch.save(tensors, root / "000000.pt")
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"step": 1, "parameters": [{"name": name, "file": "000000.pt"}]}))
    return manifest


def _write_artifacts(root: Path, tensors_by_name: dict[str, dict]) -> Path:
    root.mkdir()
    parameters = []
    for index, (name, tensors) in enumerate(tensors_by_name.items()):
        filename = f"{index:06d}.pt"
        torch.save(tensors, root / filename)
        parameters.append({"name": name, "file": filename})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"step": 1, "parameters": parameters}))
    return manifest


def _states(value: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "parameter_update": value.clone(),
        "exp_avg": torch.zeros_like(value),
        "exp_avg_sq": torch.zeros_like(value),
    }


def test_compare_optimizer_artifacts_aligns_qwen_moe_tensor_layouts(tmp_path):
    w1 = torch.tensor([[[1.0, 2.0]]])
    w3 = torch.tensor([[[3.0, 4.0]]])
    w2 = torch.tensor([[[5.0], [6.0]]])
    router = torch.tensor([[7.0]])
    shared_gate = torch.tensor([[8.0, 9.0]])
    dss = _write_artifacts(
        tmp_path / "dss-moe",
        {
            "module.layers.0.mlp.experts.w1": _states(w1),
            "module.layers.0.mlp.experts.w2": _states(w2),
            "module.layers.0.mlp.experts.w3": _states(w3),
            "module.layers.0.mlp.router.gate.weight": _states(router),
            "module.layers.0.mlp.shared_expert.w1.weight": _states(shared_gate),
            "module.visual.unused.weight": _states(torch.zeros(2)),
        },
    )
    reference = _write_artifacts(
        tmp_path / "reference-moe",
        {
            "model.layers.0.mlp.experts.gate_up_proj": _states(torch.cat([w1, w3], dim=1)),
            "model.layers.0.mlp.experts.down_proj": _states(w2),
            "model.layers.0.mlp.gate.weight": _states(router),
            "model.layers.0.mlp.shared_expert.gate_proj.weight": _states(shared_gate),
        },
    )

    result = compare_optimizer_artifacts(dss, reference, **COMPARE_KWARGS)

    assert result.only_dss == []
    assert result.only_reference == []
    assert len(result.deltas) == 12
    assert result.worst_delta == 0.0


def test_compare_optimizer_artifacts_uses_full_tensor_residual_delta_norm(tmp_path):
    reference = _write_artifact(
        tmp_path / "reference",
        "model.layer.weight",
        {
            "parameter_update": torch.tensor([1.0, 2.0]),
            "exp_avg": torch.zeros(2),
            "exp_avg_sq": torch.zeros(2),
        },
    )
    dss = _write_artifact(
        tmp_path / "dss",
        "module.layer.weight",
        {
            "parameter_update": torch.tensor([4.0, 6.0]),
            "exp_avg": torch.zeros(2),
            "exp_avg_sq": torch.zeros(2),
        },
    )

    result = compare_optimizer_artifacts(dss, reference, **COMPARE_KWARGS)

    assert result.only_dss == []
    assert result.only_reference == []
    assert [(item.state, item.delta_norm) for item in result.deltas] == [
        ("parameter_update_residual", 5.0),
        ("exp_avg", 0.0),
        ("exp_avg_sq", 0.0),
    ]
    assert result.worst_delta == 5.0


def test_compare_optimizer_artifacts_rejects_shape_mismatch(tmp_path):
    reference = _write_artifact(
        tmp_path / "reference",
        "layer.weight",
        {
            "parameter_update": torch.zeros(2),
            "exp_avg": torch.zeros(2),
            "exp_avg_sq": torch.zeros(2),
        },
    )
    dss = _write_artifact(
        tmp_path / "dss",
        "layer.weight",
        {
            "parameter_update": torch.zeros(3),
            "exp_avg": torch.zeros(2),
            "exp_avg_sq": torch.zeros(2),
        },
    )

    with pytest.raises(ValueError, match="shape mismatch"):
        compare_optimizer_artifacts(dss, reference, **COMPARE_KWARGS)


def test_delta_update_arm_applies_frozen_gate_to_every_state(tmp_path):
    reference = _write_artifact(
        tmp_path / "reference",
        "layer.weight",
        {
            "parameter_update": torch.zeros(2),
            "exp_avg": torch.zeros(2),
            "exp_avg_sq": torch.zeros(2),
        },
    )
    dss_manifest = _write_artifact(
        tmp_path / "dss",
        "layer.weight",
        _first_step_adam_state(torch.tensor([0.02, 0.0]), backend="dss"),
    )
    spec = SimpleNamespace(
        tolerance_for=lambda test_id: 1e-3,
        test_tolerances={"single-step-optimizer": Tolerance(absolute=1e-3)},
    )
    ctx = SimpleNamespace(config_id="config", spec=spec, cfg=SimpleNamespace(training={"optimizer": {}}))
    arm = SimpleNamespace(name="gas1")
    reference_result = {"optimizer_state_manifest": str(reference), "loss": 2.0, "microbatches": 1}
    dss_result = SimpleNamespace(
        optimizer_state_manifest=str(dss_manifest),
        avg_loss=2.0,
        model_calls=1,
    )

    result = _compare_arm(ctx, arm, reference_result, dss_result)

    assert result.outcome is Outcome.FAIL
    assert result.metrics["optimizer_values_compared"] == 3
    assert result.metrics["optimizer_values_over_criterion"] == 1
    assert result.worst_name == "layer.weight::exp_avg"
    assert [mismatch.name for mismatch in result.mismatches] == ["layer.weight::exp_avg"]


def test_pairwise_optimizer_variation_is_not_limited_to_the_first_run(tmp_path: Path) -> None:
    manifests = []
    for index, value in enumerate((0.0, 1.0, -1.0)):
        manifests.append(
            _write_artifact(
                tmp_path / f"run-{index}",
                "layer.weight",
                {
                    "parameter_update": torch.tensor([value]),
                    "exp_avg": torch.zeros(1),
                    "exp_avg_sq": torch.zeros(1),
                },
            )
        )

    worst = max_pairwise_optimizer_delta(manifests, **COMPARE_KWARGS)

    assert worst is not None
    assert (worst.name, worst.state, worst.delta_norm) == ("layer.weight", "parameter_update_residual", 2.0)


def _first_step_adam_state(gradient: torch.Tensor, *, backend: str = "reference") -> dict[str, torch.Tensor]:
    beta1, beta2 = 0.9, 0.999
    epsilon = 1e-8
    learning_rate = OPTIMIZER_LEARNING_RATE
    if backend == "dss":
        beta1 = float(torch.tensor(beta1, dtype=torch.float32))
        beta2 = float(torch.tensor(beta2, dtype=torch.float32))
        epsilon = float(torch.tensor(epsilon, dtype=torch.float32))
        learning_rate = float(torch.tensor(learning_rate, dtype=torch.float32))
    elif backend != "reference":
        raise ValueError(f"unknown backend: {backend}")
    exp_avg = (1.0 - beta1) * gradient
    exp_avg_sq = (1.0 - beta2) * gradient.square()
    if backend == "dss":
        beta1_correction = float(torch.tensor(1.0 - beta1, dtype=torch.float32))
        beta2_correction = float(torch.tensor(1.0 - beta2, dtype=torch.float32))
        update = -(exp_avg / beta1_correction) / ((exp_avg_sq / beta2_correction).sqrt() + epsilon)
        update = update * learning_rate
    else:
        update = -learning_rate * gradient / (gradient.abs() + epsilon)
    return {"parameter_update": update, "exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}


def test_moment_implied_raw_update_difference_has_zero_residual(tmp_path: Path) -> None:
    reference = _write_artifact(
        tmp_path / "reference-sign",
        "layer.weight",
        _first_step_adam_state(torch.tensor([1e-6])),
    )
    dss = _write_artifact(
        tmp_path / "dss-sign",
        "layer.weight",
        _first_step_adam_state(torch.tensor([-1e-6]), backend="dss"),
    )

    result = compare_optimizer_artifacts(dss, reference, **COMPARE_KWARGS)

    by_state = {item.state: item.delta_norm for item in result.deltas}
    assert by_state["parameter_update_residual"] == pytest.approx(0.0, abs=1e-9)
    assert by_state["exp_avg"] == pytest.approx(2e-7)


def test_unexplained_update_corruption_remains_in_residual(tmp_path: Path) -> None:
    reference_states = _first_step_adam_state(torch.tensor([1.0]))
    dss_states = _first_step_adam_state(torch.tensor([1.0]), backend="dss")
    dss_states["parameter_update"] += 0.005
    reference = _write_artifact(tmp_path / "reference-corrupt", "layer.weight", reference_states)
    dss = _write_artifact(tmp_path / "dss-corrupt", "layer.weight", dss_states)

    result = compare_optimizer_artifacts(dss, reference, **COMPARE_KWARGS)

    by_state = {item.state: item.delta_norm for item in result.deltas}
    assert by_state["parameter_update_residual"] == pytest.approx(0.005)
    assert by_state["exp_avg"] < 1e-3
    assert by_state["exp_avg_sq"] < 1e-3
