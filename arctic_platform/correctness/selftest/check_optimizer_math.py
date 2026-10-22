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

"""Static correctness selftests: optimizer math."""


from __future__ import annotations

import copy
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.correctness.checks import fwd_bwd_step
from arctic_platform.correctness.checks.optimizer_state import COMPARISON_STATE_KEYS
from arctic_platform.correctness.checks.optimizer_state import OPTIMIZER_NORM_CRITERION
from arctic_platform.correctness.checks.optimizer_state import STATE_KEYS
from arctic_platform.correctness.checks.optimizer_state import compare_optimizer_artifacts
from arctic_platform.correctness.checks.optimizer_state import compare_optimizer_norms
from arctic_platform.correctness.checks.optimizer_state import max_pairwise_optimizer_norm_delta
from arctic_platform.correctness.checks.optimizer_state import summarize_optimizer_artifacts
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.runner import OPTIMIZER_LEARNING_RATE
from arctic_platform.correctness.reference import optimizer_step
from arctic_platform.correctness.reference.optimizer_step import run_adamw_step

_optimizer_norms_KWARGS = {"optimizer_config": {}, "learning_rate": OPTIMIZER_LEARNING_RATE}


@pytest.fixture
def _optimizer_norms_artifact_root(tmp_path):
    assert tmp_path.resolve().is_relative_to(
        Path("/data-fast")
    ), "run optimizer controls with --basetemp=/data-fast/..."
    return tmp_path


def _optimizer_norms_manifest(root, parameters, *, step=1):
    root.mkdir()
    records = []
    for index, (name, states) in enumerate(parameters.items()):
        filename = f"{index:06d}.pt"
        torch.save(states, root / filename)
        records.append({"name": name, "file": filename})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"step": step, "parameters": records}))
    return manifest


def _optimizer_norms_states(value):
    value = torch.as_tensor(value)
    return {
        "parameter_update": value.clone(),
        "exp_avg": torch.zeros_like(value),
        "exp_avg_sq": torch.zeros_like(value),
    }


def _optimizer_norms_summary(parameters=None):
    if parameters is None:
        parameters = {
            "layer.weight": {"shape": [2], "parameter_update_residual": 0.0, "exp_avg": 0.0, "exp_avg_sq": 0.0}
        }
    return {
        "criterion": OPTIMIZER_NORM_CRITERION,
        "step": 1,
        "state_keys": list(COMPARISON_STATE_KEYS),
        "parameter_count": len(parameters),
        "optimizer_values_count": 3 * len(parameters),
        "parameters": parameters,
    }


def _optimizer_norms_adam_states(gradient, backend):
    beta1, beta2, epsilon, lr = (0.9, 0.999, 1e-08, OPTIMIZER_LEARNING_RATE)
    if backend == "dss":
        beta1, beta2, epsilon, lr = [
            float(torch.tensor(value, dtype=torch.float32)) for value in (beta1, beta2, epsilon, lr)
        ]
        correction1, correction2 = [float(torch.tensor(1.0 - value, dtype=torch.float32)) for value in (beta1, beta2)]
    else:
        correction1, correction2 = (1.0 - beta1, 1.0 - beta2)
    first = (1.0 - beta1) * gradient
    second = (1.0 - beta2) * gradient.square()
    denominator = (second / correction2).sqrt() + epsilon
    update = (
        -(first / correction1 / denominator) * lr if backend == "dss" else -lr * (first / correction1) / denominator
    )
    return {"parameter_update": update, "exp_avg": first, "exp_avg_sq": second}


def test_optimizer_norms_summary_is_compact_serializable_float32_l2(_optimizer_norms_artifact_root):
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    manifest = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "ref", {"model.layer.weight": _optimizer_norms_states(value)}
    )
    result = summarize_optimizer_artifacts(manifest, **_optimizer_norms_KWARGS)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result == _optimizer_norms_summary(
        {
            "layer.weight": {
                "shape": [2, 2],
                "parameter_update_residual": float(torch.linalg.vector_norm(value.float())),
                "exp_avg": 0.0,
                "exp_avg_sq": 0.0,
            }
        }
    )


@pytest.mark.parametrize("backend", ["reference", "dss"])
def test_optimizer_norms_norms_keep_backend_specific_moment_update_residual(_optimizer_norms_artifact_root, backend):
    states = _optimizer_norms_adam_states(torch.tensor([0.2, -0.4, 2.0]), backend)
    manifest = _optimizer_norms_manifest(_optimizer_norms_artifact_root / backend, {"layer.weight": states})
    result = summarize_optimizer_artifacts(manifest, backend=backend, **_optimizer_norms_KWARGS)["parameters"][
        "layer.weight"
    ]
    assert result["parameter_update_residual"] < 1e-08
    assert result["exp_avg"] == float(torch.linalg.vector_norm(states["exp_avg"]))
    assert result["exp_avg_sq"] == float(torch.linalg.vector_norm(states["exp_avg_sq"]))


@pytest.mark.parametrize("backend", ["reference", "dss"])
def test_optimizer_norms_nonzero_initial_parameters_do_not_enter_update_norm(_optimizer_norms_artifact_root, backend):
    initial = torch.tensor([2.0, -3.0])
    states = _optimizer_norms_adam_states(torch.tensor([1.0, -1.0]), backend)
    after = initial + states["parameter_update"]
    states["parameter_update"] = after - initial
    manifest = _optimizer_norms_manifest(_optimizer_norms_artifact_root / backend, {"layer.weight": states})
    result = summarize_optimizer_artifacts(manifest, backend=backend, **_optimizer_norms_KWARGS)["parameters"][
        "layer.weight"
    ]
    assert result["parameter_update_residual"] < 5e-07
    assert result["parameter_update_residual"] < float(torch.linalg.vector_norm(initial))


def test_optimizer_norms_unexplained_update_is_not_removed_by_norm_reduction(_optimizer_norms_artifact_root):
    states = _optimizer_norms_adam_states(torch.tensor([1.0, -1.0]), "dss")
    states["parameter_update"] += torch.tensor([0.003, 0.004])
    manifest = _optimizer_norms_manifest(_optimizer_norms_artifact_root / "corrupt", {"module.layer.weight": states})
    result = summarize_optimizer_artifacts(manifest, backend="dss", **_optimizer_norms_KWARGS)
    assert result["parameters"]["layer.weight"]["parameter_update_residual"] == pytest.approx(0.005)


def test_optimizer_norms_moe_tensors_are_concatenated_before_residual_and_norm(_optimizer_norms_artifact_root):
    w1 = _optimizer_norms_adam_states(torch.tensor([[[0.2, 0.4]]]), "dss")
    w3 = _optimizer_norms_adam_states(torch.tensor([[[0.6, 0.8]]]), "dss")
    dss = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "moe",
        {"module.layers.0.mlp.experts.w1": w1, "module.layers.0.mlp.experts.w3": w3},
    )
    joined = {state: torch.cat([w1[state], w3[state]], dim=1) for state in STATE_KEYS}
    canonical = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "canonical", {"layers.0.mlp.experts.gate_up_proj": joined}
    )
    actual = summarize_optimizer_artifacts(dss, backend="dss", **_optimizer_norms_KWARGS)
    expected = summarize_optimizer_artifacts(canonical, backend="dss", **_optimizer_norms_KWARGS)
    assert actual == expected
    entry = actual["parameters"]["layers.0.mlp.experts.gate_up_proj"]
    assert entry["shape"] == [1, 2, 2]
    assert entry["exp_avg"] == float(torch.linalg.vector_norm(joined["exp_avg"]))
    assert entry["parameter_update_residual"] < 1e-08


def test_optimizer_norms_canonical_collision_cannot_hide_a_parameter(_optimizer_norms_artifact_root):
    manifest = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "collision",
        {
            "layers.0.mlp.experts.w1": _optimizer_norms_states(torch.zeros(1, 1, 1)),
            "layers.0.mlp.experts.w3": _optimizer_norms_states(torch.zeros(1, 1, 1)),
            "layers.0.mlp.experts.gate_up_proj": _optimizer_norms_states(torch.zeros(1, 2, 1)),
        },
    )
    with pytest.raises(ValueError, match="canonical parameter name collision"):
        summarize_optimizer_artifacts(manifest, backend="dss", **_optimizer_norms_KWARGS)


def test_optimizer_norms_normalization_collision_is_rejected(_optimizer_norms_artifact_root):
    manifest = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "aliases",
        {
            "model.layer.weight": _optimizer_norms_states(torch.zeros(2)),
            "module.layer.weight": _optimizer_norms_states(torch.zeros(2)),
        },
    )
    with pytest.raises(ValueError, match="duplicate normalized parameter"):
        summarize_optimizer_artifacts(manifest, **_optimizer_norms_KWARGS)


@pytest.mark.parametrize("step", [0, 2, 1.0, True, "1"])
def test_optimizer_norms_summary_requires_exact_step_one(_optimizer_norms_artifact_root, step):
    manifest = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "step", {"layer.weight": _optimizer_norms_states(torch.zeros(2))}, step=step
    )
    with pytest.raises(ValueError, match="step 1"):
        summarize_optimizer_artifacts(manifest, **_optimizer_norms_KWARGS)


def test_optimizer_norms_summary_rejects_empty_capture(_optimizer_norms_artifact_root):
    manifest = _optimizer_norms_manifest(_optimizer_norms_artifact_root / "empty", {})
    with pytest.raises(ValueError, match="no parameters"):
        summarize_optimizer_artifacts(manifest, **_optimizer_norms_KWARGS)


@pytest.mark.parametrize("state", STATE_KEYS)
def test_optimizer_norms_summary_requires_every_artifact_state(_optimizer_norms_artifact_root, state):
    states = _optimizer_norms_states(torch.zeros(2))
    states.pop(state)
    manifest = _optimizer_norms_manifest(_optimizer_norms_artifact_root / "missing", {"layer.weight": states})
    with pytest.raises(ValueError, match=state):
        summarize_optimizer_artifacts(manifest, **_optimizer_norms_KWARGS)


def test_optimizer_norms_summary_requires_identical_state_shapes(_optimizer_norms_artifact_root):
    states = _optimizer_norms_states(torch.zeros(2))
    states["exp_avg_sq"] = torch.zeros(1)
    manifest = _optimizer_norms_manifest(_optimizer_norms_artifact_root / "shape", {"layer.weight": states})
    with pytest.raises(ValueError, match="shape mismatch"):
        summarize_optimizer_artifacts(manifest, **_optimizer_norms_KWARGS)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e100])
def test_optimizer_norms_summary_rejects_nonfinite_or_float32_overflow(_optimizer_norms_artifact_root, value):
    manifest = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "invalid",
        {"layer.weight": _optimizer_norms_states(torch.tensor([value], dtype=torch.float64))},
    )
    with pytest.raises(ValueError, match="not finite"):
        summarize_optimizer_artifacts(manifest, **_optimizer_norms_KWARGS)


def test_optimizer_norms_summary_rejects_norm_reduction_overflow(_optimizer_norms_artifact_root):
    manifest = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "overflow",
        {"layer.weight": _optimizer_norms_states(torch.tensor([3e38, 3e38]))},
    )
    with pytest.raises(ValueError, match="optimizer norm .* not finite"):
        summarize_optimizer_artifacts(manifest, **_optimizer_norms_KWARGS)


def test_optimizer_norms_summary_rejects_invalid_second_moment(_optimizer_norms_artifact_root):
    states = _optimizer_norms_states(torch.zeros(2))
    states["exp_avg_sq"] = torch.tensor([-1.0, 0.0])
    manifest = _optimizer_norms_manifest(_optimizer_norms_artifact_root / "negative", {"layer.weight": states})
    with pytest.raises(ValueError, match="comparison tensor .* not finite"):
        summarize_optimizer_artifacts(manifest, **_optimizer_norms_KWARGS)


@pytest.mark.parametrize("dtype", [torch.int64, torch.complex64])
def test_optimizer_norms_summary_rejects_nonfloating_artifacts(_optimizer_norms_artifact_root, dtype):
    manifest = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "dtype",
        {"layer.weight": _optimizer_norms_states(torch.zeros(2, dtype=dtype))},
    )
    with pytest.raises(ValueError, match="floating tensor"):
        summarize_optimizer_artifacts(manifest, **_optimizer_norms_KWARGS)


def test_optimizer_norms_compare_uses_scalar_norm_difference_not_tensor_difference(_optimizer_norms_artifact_root):
    ref = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "ref", {"layer.weight": _optimizer_norms_states(torch.tensor([1.0, 2.0]))}
    )
    dss = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "dss", {"layer.weight": _optimizer_norms_states(torch.tensor([4.0, 6.0]))}
    )
    comparison = compare_optimizer_norms(
        summarize_optimizer_artifacts(dss, backend="dss", **_optimizer_norms_KWARGS),
        summarize_optimizer_artifacts(ref, **_optimizer_norms_KWARGS),
    )
    assert comparison.worst_delta == pytest.approx(math.sqrt(52) - math.sqrt(5))
    assert compare_optimizer_artifacts(dss, ref, **_optimizer_norms_KWARGS).worst_delta == 5.0


def test_optimizer_norms_equal_norm_different_direction_is_intentionally_undetected(_optimizer_norms_artifact_root):
    ref = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "ref", {"layer.weight": _optimizer_norms_states(torch.tensor([3.0, 4.0]))}
    )
    dss = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "dss", {"layer.weight": _optimizer_norms_states(torch.tensor([-3.0, -4.0]))}
    )
    comparison = compare_optimizer_norms(
        summarize_optimizer_artifacts(dss, backend="dss", **_optimizer_norms_KWARGS),
        summarize_optimizer_artifacts(ref, **_optimizer_norms_KWARGS),
    )
    assert comparison.worst_delta == 0.0
    assert compare_optimizer_artifacts(dss, ref, **_optimizer_norms_KWARGS).worst_delta == 10.0


@pytest.mark.parametrize("value", [0.0, 1e-30, 0.25])
def test_optimizer_norms_zero_and_near_zero_have_absolute_difference_without_exemption(value):
    reference = _optimizer_norms_summary()
    target = copy.deepcopy(reference)
    target["parameters"]["layer.weight"]["exp_avg"] = value
    comparison = compare_optimizer_norms(target, reference)
    assert comparison.worst_delta == value


def test_optimizer_norms_missing_zero_vision_parameter_is_still_a_coverage_failure():
    reference = _optimizer_norms_summary()
    target = copy.deepcopy(reference["parameters"])
    target["visual.unused.weight"] = copy.deepcopy(target["layer.weight"])
    comparison = compare_optimizer_norms(_optimizer_norms_summary(target), reference)
    assert comparison.only_dss == ["visual.unused.weight"]
    assert comparison.only_reference == []
    assert len(comparison.deltas) == 3


def test_optimizer_norms_comparison_reports_both_unmatched_parameter_sets():
    target = _optimizer_norms_summary({"target.weight": _optimizer_norms_summary()["parameters"]["layer.weight"]})
    reference = _optimizer_norms_summary({"ref.weight": _optimizer_norms_summary()["parameters"]["layer.weight"]})
    comparison = compare_optimizer_norms(target, reference)
    assert comparison.only_dss == ["target.weight"]
    assert comparison.only_reference == ["ref.weight"]
    assert comparison.deltas == []


def test_optimizer_norms_comparison_requires_matching_shapes():
    target = _optimizer_norms_summary()
    reference = copy.deepcopy(target)
    reference["parameters"]["layer.weight"]["shape"] = [1, 2]
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_optimizer_norms(target, reference)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("criterion", "full-tensor"),
        ("step", 2),
        ("step", True),
        ("state_keys", ["exp_avg"]),
        ("parameter_count", 2),
        ("parameter_count", True),
        ("optimizer_values_count", 2),
        ("parameters", {}),
    ],
)
@pytest.mark.parametrize("side", ["target", "reference"])
def test_optimizer_norms_comparison_validates_both_summary_envelopes(key, value, side):
    target, reference = (_optimizer_norms_summary(), _optimizer_norms_summary())
    (target if side == "target" else reference)[key] = value
    with pytest.raises(ValueError):
        compare_optimizer_norms(target, reference)


@pytest.mark.parametrize("state", COMPARISON_STATE_KEYS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, True, "0.0"])
def test_optimizer_norms_comparison_rejects_nonfinite_negative_or_nonnumeric_norm(state, value):
    target = _optimizer_norms_summary()
    target["parameters"]["layer.weight"][state] = value
    with pytest.raises(ValueError, match="finite and nonnegative"):
        compare_optimizer_norms(target, _optimizer_norms_summary())


@pytest.mark.parametrize("state", COMPARISON_STATE_KEYS)
def test_optimizer_norms_comparison_requires_complete_per_parameter_state_coverage(state):
    target = _optimizer_norms_summary()
    target["parameters"]["layer.weight"].pop(state)
    with pytest.raises(ValueError, match="state coverage"):
        compare_optimizer_norms(target, _optimizer_norms_summary())


@pytest.mark.parametrize("shape", [[True], [-1], [1.0], "2"])
def test_optimizer_norms_comparison_rejects_invalid_shape_metadata(shape):
    target = _optimizer_norms_summary()
    target["parameters"]["layer.weight"]["shape"] = shape
    with pytest.raises(ValueError, match="invalid shape"):
        compare_optimizer_norms(target, _optimizer_norms_summary())


@pytest.mark.parametrize("name", ["", "model.layer.weight", "module.layer.weight"])
def test_optimizer_norms_comparison_rejects_noncanonical_names(name):
    target = _optimizer_norms_summary({name: _optimizer_norms_summary()["parameters"]["layer.weight"]})
    with pytest.raises(ValueError, match="canonical parameter"):
        compare_optimizer_norms(target, _optimizer_norms_summary())


def test_optimizer_norms_pairwise_norm_calibration_checks_every_pair(_optimizer_norms_artifact_root):
    manifests = [
        _optimizer_norms_manifest(
            _optimizer_norms_artifact_root / f"run-{index}",
            {"layer.weight": _optimizer_norms_states(torch.tensor([value]))},
        )
        for index, value in enumerate((2.0, 0.0, 4.0))
    ]
    updates = []
    progress = SimpleNamespace(update=lambda: updates.append(1))
    worst = max_pairwise_optimizer_norm_delta(manifests, progress=progress, **_optimizer_norms_KWARGS)
    assert (worst.name, worst.state, worst.delta_norm) == ("layer.weight", "parameter_update_residual", 4.0)
    assert updates == [1]


def test_optimizer_norms_pairwise_norm_calibration_sign_change_has_no_magnitude_variation(
    _optimizer_norms_artifact_root,
):
    manifests = [
        _optimizer_norms_manifest(
            _optimizer_norms_artifact_root / f"run-{index}",
            {"layer.weight": _optimizer_norms_states(torch.tensor([value]))},
        )
        for index, value in enumerate((1.0, -1.0))
    ]
    assert max_pairwise_optimizer_norm_delta(manifests, **_optimizer_norms_KWARGS).delta_norm == 0.0


def test_optimizer_norms_pairwise_norm_calibration_empty_or_single_capture(_optimizer_norms_artifact_root):
    manifest = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "single", {"layer.weight": _optimizer_norms_states(torch.zeros(2))}
    )
    assert max_pairwise_optimizer_norm_delta([], **_optimizer_norms_KWARGS) is None
    assert max_pairwise_optimizer_norm_delta([manifest], **_optimizer_norms_KWARGS) is None


@pytest.mark.parametrize("kind", ["names", "shapes"])
def test_optimizer_norms_pairwise_norm_calibration_requires_identical_coverage(_optimizer_norms_artifact_root, kind):
    first = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "first", {"layer.weight": _optimizer_norms_states(torch.zeros(2))}
    )
    second = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "second",
        {
            "other.weight" if kind == "names" else "layer.weight": _optimizer_norms_states(
                torch.zeros(3 if kind == "shapes" else 2)
            )
        },
    )
    with pytest.raises(ValueError, match="parameter sets" if kind == "names" else "shape mismatch"):
        max_pairwise_optimizer_norm_delta([first, second], **_optimizer_norms_KWARGS)


def _optimizer_norms_context_and_results(manifest, reference_norms=None, gate=0.0):

    def forbid_frozen_gate(*args):
        raise AssertionError("must not read a frozen full-tensor gate")

    ctx = SimpleNamespace(
        config_id="config",
        cfg=SimpleNamespace(training={"optimizer": {}}),
        spec=SimpleNamespace(tolerance_for=forbid_frozen_gate),
    )
    reference = {
        "optimizer_norms": reference_norms or _optimizer_norms_summary(),
        "optimizer_norm_gate": {
            "criterion": OPTIMIZER_NORM_CRITERION,
            "absolute": gate,
            "status": "reviewed",
            "provenance": {"review": "explicit-control-review"},
        },
        "loss": 2.0,
        "microbatches": 1,
    }
    target = SimpleNamespace(optimizer_state_manifest=manifest, avg_loss=2.0, model_calls=1)
    return (ctx, SimpleNamespace(name="gas1"), reference, target)


def test_optimizer_norms_arm_consumes_only_stored_norms_and_explicit_reviewed_gate(_optimizer_norms_artifact_root):
    manifest = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "dss", {"layer.weight": _optimizer_norms_states(torch.tensor([0.01, 0.0]))}
    )
    ctx, arm, reference, target = _optimizer_norms_context_and_results(manifest, gate=0.02)
    reference["optimizer_state_manifest"] = "do-not-open-reference-tensors"
    result = fwd_bwd_step._compare_arm(ctx, arm, reference, target)
    assert result.outcome is Outcome.PASS
    assert result.metrics["stated_criterion_abs"] == 0.02
    assert result.metrics["max_optimizer_norm_delta"] == pytest.approx(0.01)
    assert OPTIMIZER_NORM_CRITERION in result.reason
    assert "explicit-control-review" in result.reason
    assert "Equal norms can conceal" in result.reason


@pytest.mark.parametrize("state", COMPARISON_STATE_KEYS)
def test_optimizer_norms_arm_gates_every_scalar_and_reports_actual_norms(monkeypatch, state):
    target_summary = _optimizer_norms_summary()
    reference_summary = _optimizer_norms_summary()
    target_summary["parameters"]["layer.weight"][state] = 0.125
    reference_summary["parameters"]["layer.weight"][state] = 0.1
    monkeypatch.setattr(fwd_bwd_step, "summarize_optimizer_artifacts", lambda *args, **kwargs: target_summary)
    ctx, arm, reference, target = _optimizer_norms_context_and_results("target", reference_summary, gate=0.01)
    result = fwd_bwd_step._compare_arm(ctx, arm, reference, target)
    assert result.outcome is Outcome.FAIL
    assert result.metrics["optimizer_values_compared"] == 3
    assert result.metrics["optimizer_values_over_criterion"] == 1
    assert result.worst_name == f"layer.weight::{state}"
    (mismatch,) = result.mismatches
    assert mismatch.target == 0.125
    assert mismatch.reference == 0.1
    assert mismatch.abs_diff == pytest.approx(0.025)


def test_optimizer_norms_arm_gate_boundary_is_inclusive(monkeypatch):
    summary = _optimizer_norms_summary()
    summary["parameters"]["layer.weight"]["exp_avg"] = 0.125
    monkeypatch.setattr(fwd_bwd_step, "summarize_optimizer_artifacts", lambda *args, **kwargs: summary)
    ctx, arm, reference, target = _optimizer_norms_context_and_results("target", gate=0.125)
    assert fwd_bwd_step._compare_arm(ctx, arm, reference, target).outcome is Outcome.PASS


def test_optimizer_norms_arm_coverage_failure_is_not_a_zero_norm_pass(monkeypatch):
    summary = _optimizer_norms_summary(
        {"visual.unused.weight": _optimizer_norms_summary()["parameters"]["layer.weight"]}
    )
    monkeypatch.setattr(fwd_bwd_step, "summarize_optimizer_artifacts", lambda *args, **kwargs: summary)
    ctx, arm, reference, target = _optimizer_norms_context_and_results("target", gate=100)
    result = fwd_bwd_step._compare_arm(ctx, arm, reference, target)
    assert result.outcome is Outcome.FAIL
    assert result.metrics["unmatched_dss"] == 1
    assert result.metrics["unmatched_reference"] == 1


def test_optimizer_norms_arm_missing_norms_has_no_manifest_fallback(monkeypatch):

    def forbid_tensor_read(*args, **kwargs):
        raise AssertionError("must reject missing compact reference before reading artifacts")

    monkeypatch.setattr(fwd_bwd_step, "summarize_optimizer_artifacts", forbid_tensor_read)
    ctx, arm, reference, target = _optimizer_norms_context_and_results("target")
    reference.pop("optimizer_norms")
    reference["optimizer_state_manifest"] = "legacy-manifest"
    with pytest.raises(ValueError, match="stored optimizer_norms"):
        fwd_bwd_step._compare_arm(ctx, arm, reference, target)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("criterion", "full-tensor"),
        ("status", "proposed"),
        ("absolute", -1.0),
        ("absolute", float("nan")),
        ("absolute", float("inf")),
        ("absolute", True),
        ("absolute", "0.01"),
        ("provenance", None),
        ("provenance", {}),
        ("provenance", ""),
        ("provenance", "   "),
    ],
)
def test_optimizer_norms_arm_rejects_invalid_or_unreviewed_gate_before_artifact_read(monkeypatch, key, value):

    def forbid_tensor_read(*args, **kwargs):
        raise AssertionError("must reject invalid gate before reading artifacts")

    monkeypatch.setattr(fwd_bwd_step, "summarize_optimizer_artifacts", forbid_tensor_read)
    ctx, arm, reference, target = _optimizer_norms_context_and_results("target")
    reference["optimizer_norm_gate"][key] = value
    with pytest.raises(ValueError, match="optimizer_norm_gate"):
        fwd_bwd_step._compare_arm(ctx, arm, reference, target)


def test_optimizer_norms_arm_requires_gate_and_target_capture():
    ctx, arm, reference, target = _optimizer_norms_context_and_results(None)
    with pytest.raises(ValueError, match="target optimizer-state manifest"):
        fwd_bwd_step._compare_arm(ctx, arm, reference, target)
    reference.pop("optimizer_norm_gate")
    with pytest.raises(ValueError, match="reviewed optimizer_norm_gate"):
        fwd_bwd_step._compare_arm(ctx, arm, reference, target)


@pytest.fixture
def _optimizer_reference_fast_output():
    with tempfile.TemporaryDirectory(prefix="optimizer-reference-selftest-", dir="/data-fast/tmp") as directory:
        yield Path(directory)


def test_optimizer_reference_adamw_reference_writes_named_fp32_moments_and_updated_parameter(
    _optimizer_reference_fast_output: Path,
) -> None:
    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.bfloat16))
    gradient = torch.tensor([[0.25, -0.5], [0.75, -1.0]], dtype=torch.float32)
    config = {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.1}
    initial = model.weight.detach().float().clone()
    expected = torch.nn.Parameter(initial.clone())
    expected.grad = gradient.clone()
    optimizer = torch.optim.AdamW(
        [{"params": [expected], "weight_decay": 0.1}],
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-08,
        foreach=False,
        fused=False,
    )
    optimizer.step()
    artifact = run_adamw_step(
        model,
        {"weight": gradient},
        config,
        learning_rate=0.01,
        gradient_clipping=None,
        optimizer_dtype="float32",
        output_dir=_optimizer_reference_fast_output,
    )
    manifest = json.loads(artifact.manifest_path.read_text())
    assert manifest["step"] == 1
    assert [entry["name"] for entry in manifest["parameters"]] == ["weight"]
    tensors = torch.load(_optimizer_reference_fast_output / manifest["parameters"][0]["file"], weights_only=True)
    expected_state = optimizer.state[expected]
    torch.testing.assert_close(tensors["parameter_update"], expected.detach() - initial)
    torch.testing.assert_close(tensors["exp_avg"], expected_state["exp_avg"])
    torch.testing.assert_close(tensors["exp_avg_sq"], expected_state["exp_avg_sq"])


def test_optimizer_reference_adamw_reference_excludes_norm_weight_from_weight_decay(
    _optimizer_reference_fast_output: Path,
) -> None:

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
        learning_rate=0.01,
        gradient_clipping=None,
        optimizer_dtype="float32",
        output_dir=_optimizer_reference_fast_output,
    )
    manifest = json.loads((_optimizer_reference_fast_output / "manifest.json").read_text())
    tensors = torch.load(_optimizer_reference_fast_output / manifest["parameters"][0]["file"], weights_only=True)
    initial = model.norm.weight.detach().float().clone()
    expected = torch.nn.Parameter(initial.clone())
    expected.grad = gradient.clone()
    optimizer = torch.optim.AdamW([{"params": [expected], "weight_decay": 0.0}], lr=0.01, foreach=False, fused=False)
    optimizer.step()
    torch.testing.assert_close(tensors["parameter_update"], expected.detach() - initial)


def test_optimizer_reference_adamw_reference_treats_zero_clipping_as_disabled(
    _optimizer_reference_fast_output: Path,
) -> None:
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    gradient = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    artifact = run_adamw_step(
        model,
        {"weight": gradient},
        {"name": "AdamW", "weight_decay": 0.0},
        learning_rate=0.01,
        gradient_clipping=0,
        optimizer_dtype="float32",
        output_dir=_optimizer_reference_fast_output,
    )
    manifest = json.loads(artifact.manifest_path.read_text())
    tensors = torch.load(_optimizer_reference_fast_output / manifest["parameters"][0]["file"], weights_only=True)
    assert torch.count_nonzero(tensors["parameter_update"]) > 0


@pytest.mark.parametrize("output", ("/code/optimizer", "/tmp/optimizer", "/data-fast", "/data-fast-sibling/optimizer"))
def test_optimizer_reference_adamw_reference_rejects_unsafe_storage_before_model_or_optimizer(
    output, monkeypatch, _optimizer_reference_fast_output
):
    before = list(_optimizer_reference_fast_output.iterdir())
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
    assert list(_optimizer_reference_fast_output.iterdir()) == before


def test_optimizer_reference_adamw_reference_rejects_symlink_escape_before_callbacks(
    monkeypatch, _optimizer_reference_fast_output
):
    link = _optimizer_reference_fast_output / "escaped"
    link.symlink_to("/tmp", target_is_directory=True)
    before = list(_optimizer_reference_fast_output.iterdir())
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
    assert list(_optimizer_reference_fast_output.iterdir()) == before


def test_optimizer_reference_adamw_reference_returns_canonical_fast_manifest(_optimizer_reference_fast_output):
    actual = _optimizer_reference_fast_output / "actual"
    actual.mkdir()
    link = _optimizer_reference_fast_output / "link"
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
