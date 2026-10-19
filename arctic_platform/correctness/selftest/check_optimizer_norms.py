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

"""Versioned scalar optimizer norms, strict coverage, and explicit reviewed gates."""

import copy
import json
import math
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

KWARGS = {"optimizer_config": {}, "learning_rate": OPTIMIZER_LEARNING_RATE}


@pytest.fixture
def artifact_root(tmp_path):
    assert tmp_path.resolve().is_relative_to(
        Path("/data-fast")
    ), "run optimizer controls with --basetemp=/data-fast/..."
    return tmp_path


def _manifest(root, parameters, *, step=1):
    root.mkdir()
    records = []
    for index, (name, states) in enumerate(parameters.items()):
        filename = f"{index:06d}.pt"
        torch.save(states, root / filename)
        records.append({"name": name, "file": filename})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"step": step, "parameters": records}))
    return manifest


def _states(value):
    value = torch.as_tensor(value)
    return {
        "parameter_update": value.clone(),
        "exp_avg": torch.zeros_like(value),
        "exp_avg_sq": torch.zeros_like(value),
    }


def _summary(parameters=None):
    if parameters is None:
        parameters = {
            "layer.weight": {
                "shape": [2],
                "parameter_update_residual": 0.0,
                "exp_avg": 0.0,
                "exp_avg_sq": 0.0,
            }
        }
    return {
        "criterion": OPTIMIZER_NORM_CRITERION,
        "step": 1,
        "state_keys": list(COMPARISON_STATE_KEYS),
        "parameter_count": len(parameters),
        "optimizer_values_count": 3 * len(parameters),
        "parameters": parameters,
    }


def _adam_states(gradient, backend):
    beta1, beta2, epsilon, lr = 0.9, 0.999, 1e-8, OPTIMIZER_LEARNING_RATE
    if backend == "dss":
        beta1, beta2, epsilon, lr = [
            float(torch.tensor(value, dtype=torch.float32)) for value in (beta1, beta2, epsilon, lr)
        ]
        correction1, correction2 = [float(torch.tensor(1.0 - value, dtype=torch.float32)) for value in (beta1, beta2)]
    else:
        correction1, correction2 = 1.0 - beta1, 1.0 - beta2
    first = (1.0 - beta1) * gradient
    second = (1.0 - beta2) * gradient.square()
    denominator = (second / correction2).sqrt() + epsilon
    update = (
        -(first / correction1 / denominator) * lr if backend == "dss" else -lr * (first / correction1) / denominator
    )
    return {"parameter_update": update, "exp_avg": first, "exp_avg_sq": second}


def test_summary_is_compact_serializable_float32_l2(artifact_root):
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    manifest = _manifest(artifact_root / "ref", {"model.layer.weight": _states(value)})

    result = summarize_optimizer_artifacts(manifest, **KWARGS)

    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result == _summary(
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
def test_norms_keep_backend_specific_moment_update_residual(artifact_root, backend):
    states = _adam_states(torch.tensor([0.2, -0.4, 2.0]), backend)
    manifest = _manifest(artifact_root / backend, {"layer.weight": states})

    result = summarize_optimizer_artifacts(manifest, backend=backend, **KWARGS)["parameters"]["layer.weight"]

    assert result["parameter_update_residual"] < 1e-8
    assert result["exp_avg"] == float(torch.linalg.vector_norm(states["exp_avg"]))
    assert result["exp_avg_sq"] == float(torch.linalg.vector_norm(states["exp_avg_sq"]))


@pytest.mark.parametrize("backend", ["reference", "dss"])
def test_nonzero_initial_parameters_do_not_enter_update_norm(artifact_root, backend):
    initial = torch.tensor([2.0, -3.0])
    states = _adam_states(torch.tensor([1.0, -1.0]), backend)
    after = initial + states["parameter_update"]
    states["parameter_update"] = after - initial
    manifest = _manifest(artifact_root / backend, {"layer.weight": states})

    result = summarize_optimizer_artifacts(manifest, backend=backend, **KWARGS)["parameters"]["layer.weight"]

    assert result["parameter_update_residual"] < 5e-7
    assert result["parameter_update_residual"] < float(torch.linalg.vector_norm(initial))


def test_unexplained_update_is_not_removed_by_norm_reduction(artifact_root):
    states = _adam_states(torch.tensor([1.0, -1.0]), "dss")
    states["parameter_update"] += torch.tensor([0.003, 0.004])
    manifest = _manifest(artifact_root / "corrupt", {"module.layer.weight": states})

    result = summarize_optimizer_artifacts(manifest, backend="dss", **KWARGS)

    assert result["parameters"]["layer.weight"]["parameter_update_residual"] == pytest.approx(0.005)


def test_moe_tensors_are_concatenated_before_residual_and_norm(artifact_root):
    w1 = _adam_states(torch.tensor([[[0.2, 0.4]]]), "dss")
    w3 = _adam_states(torch.tensor([[[0.6, 0.8]]]), "dss")
    dss = _manifest(
        artifact_root / "moe",
        {"module.layers.0.mlp.experts.w1": w1, "module.layers.0.mlp.experts.w3": w3},
    )
    joined = {state: torch.cat([w1[state], w3[state]], dim=1) for state in STATE_KEYS}
    canonical = _manifest(artifact_root / "canonical", {"layers.0.mlp.experts.gate_up_proj": joined})

    actual = summarize_optimizer_artifacts(dss, backend="dss", **KWARGS)
    expected = summarize_optimizer_artifacts(canonical, backend="dss", **KWARGS)

    assert actual == expected
    entry = actual["parameters"]["layers.0.mlp.experts.gate_up_proj"]
    assert entry["shape"] == [1, 2, 2]
    assert entry["exp_avg"] == float(torch.linalg.vector_norm(joined["exp_avg"]))
    assert entry["parameter_update_residual"] < 1e-8


def test_canonical_collision_cannot_hide_a_parameter(artifact_root):
    manifest = _manifest(
        artifact_root / "collision",
        {
            "layers.0.mlp.experts.w1": _states(torch.zeros(1, 1, 1)),
            "layers.0.mlp.experts.w3": _states(torch.zeros(1, 1, 1)),
            "layers.0.mlp.experts.gate_up_proj": _states(torch.zeros(1, 2, 1)),
        },
    )
    with pytest.raises(ValueError, match="canonical parameter name collision"):
        summarize_optimizer_artifacts(manifest, backend="dss", **KWARGS)


def test_normalization_collision_is_rejected(artifact_root):
    manifest = _manifest(
        artifact_root / "aliases",
        {"model.layer.weight": _states(torch.zeros(2)), "module.layer.weight": _states(torch.zeros(2))},
    )
    with pytest.raises(ValueError, match="duplicate normalized parameter"):
        summarize_optimizer_artifacts(manifest, **KWARGS)


@pytest.mark.parametrize("step", [0, 2, 1.0, True, "1"])
def test_summary_requires_exact_step_one(artifact_root, step):
    manifest = _manifest(artifact_root / "step", {"layer.weight": _states(torch.zeros(2))}, step=step)
    with pytest.raises(ValueError, match="step 1"):
        summarize_optimizer_artifacts(manifest, **KWARGS)


def test_summary_rejects_empty_capture(artifact_root):
    manifest = _manifest(artifact_root / "empty", {})
    with pytest.raises(ValueError, match="no parameters"):
        summarize_optimizer_artifacts(manifest, **KWARGS)


@pytest.mark.parametrize("state", STATE_KEYS)
def test_summary_requires_every_artifact_state(artifact_root, state):
    states = _states(torch.zeros(2))
    states.pop(state)
    manifest = _manifest(artifact_root / "missing", {"layer.weight": states})
    with pytest.raises(ValueError, match=state):
        summarize_optimizer_artifacts(manifest, **KWARGS)


def test_summary_requires_identical_state_shapes(artifact_root):
    states = _states(torch.zeros(2))
    states["exp_avg_sq"] = torch.zeros(1)
    manifest = _manifest(artifact_root / "shape", {"layer.weight": states})
    with pytest.raises(ValueError, match="shape mismatch"):
        summarize_optimizer_artifacts(manifest, **KWARGS)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e100])
def test_summary_rejects_nonfinite_or_float32_overflow(artifact_root, value):
    manifest = _manifest(
        artifact_root / "invalid", {"layer.weight": _states(torch.tensor([value], dtype=torch.float64))}
    )
    with pytest.raises(ValueError, match="not finite"):
        summarize_optimizer_artifacts(manifest, **KWARGS)


def test_summary_rejects_norm_reduction_overflow(artifact_root):
    manifest = _manifest(artifact_root / "overflow", {"layer.weight": _states(torch.tensor([3e38, 3e38]))})
    with pytest.raises(ValueError, match="optimizer norm .* not finite"):
        summarize_optimizer_artifacts(manifest, **KWARGS)


def test_summary_rejects_invalid_second_moment(artifact_root):
    states = _states(torch.zeros(2))
    states["exp_avg_sq"] = torch.tensor([-1.0, 0.0])
    manifest = _manifest(artifact_root / "negative", {"layer.weight": states})
    with pytest.raises(ValueError, match="comparison tensor .* not finite"):
        summarize_optimizer_artifacts(manifest, **KWARGS)


@pytest.mark.parametrize("dtype", [torch.int64, torch.complex64])
def test_summary_rejects_nonfloating_artifacts(artifact_root, dtype):
    manifest = _manifest(artifact_root / "dtype", {"layer.weight": _states(torch.zeros(2, dtype=dtype))})
    with pytest.raises(ValueError, match="floating tensor"):
        summarize_optimizer_artifacts(manifest, **KWARGS)


def test_compare_uses_scalar_norm_difference_not_tensor_difference(artifact_root):
    ref = _manifest(artifact_root / "ref", {"layer.weight": _states(torch.tensor([1.0, 2.0]))})
    dss = _manifest(artifact_root / "dss", {"layer.weight": _states(torch.tensor([4.0, 6.0]))})

    comparison = compare_optimizer_norms(
        summarize_optimizer_artifacts(dss, backend="dss", **KWARGS), summarize_optimizer_artifacts(ref, **KWARGS)
    )

    assert comparison.worst_delta == pytest.approx(math.sqrt(52) - math.sqrt(5))
    assert compare_optimizer_artifacts(dss, ref, **KWARGS).worst_delta == 5.0


def test_equal_norm_different_direction_is_intentionally_undetected(artifact_root):
    ref = _manifest(artifact_root / "ref", {"layer.weight": _states(torch.tensor([3.0, 4.0]))})
    dss = _manifest(artifact_root / "dss", {"layer.weight": _states(torch.tensor([-3.0, -4.0]))})

    comparison = compare_optimizer_norms(
        summarize_optimizer_artifacts(dss, backend="dss", **KWARGS), summarize_optimizer_artifacts(ref, **KWARGS)
    )

    assert comparison.worst_delta == 0.0
    assert compare_optimizer_artifacts(dss, ref, **KWARGS).worst_delta == 10.0


@pytest.mark.parametrize("value", [0.0, 1e-30, 0.25])
def test_zero_and_near_zero_have_absolute_difference_without_exemption(value):
    reference = _summary()
    target = copy.deepcopy(reference)
    target["parameters"]["layer.weight"]["exp_avg"] = value

    comparison = compare_optimizer_norms(target, reference)

    assert comparison.worst_delta == value


def test_missing_zero_vision_parameter_is_still_a_coverage_failure():
    reference = _summary()
    target = copy.deepcopy(reference["parameters"])
    target["visual.unused.weight"] = copy.deepcopy(target["layer.weight"])

    comparison = compare_optimizer_norms(_summary(target), reference)

    assert comparison.only_dss == ["visual.unused.weight"]
    assert comparison.only_reference == []
    assert len(comparison.deltas) == 3


def test_comparison_reports_both_unmatched_parameter_sets():
    target = _summary({"target.weight": _summary()["parameters"]["layer.weight"]})
    reference = _summary({"ref.weight": _summary()["parameters"]["layer.weight"]})
    comparison = compare_optimizer_norms(target, reference)
    assert comparison.only_dss == ["target.weight"]
    assert comparison.only_reference == ["ref.weight"]
    assert comparison.deltas == []


def test_comparison_requires_matching_shapes():
    target = _summary()
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
def test_comparison_validates_both_summary_envelopes(key, value, side):
    target, reference = _summary(), _summary()
    (target if side == "target" else reference)[key] = value
    with pytest.raises(ValueError):
        compare_optimizer_norms(target, reference)


@pytest.mark.parametrize("state", COMPARISON_STATE_KEYS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, True, "0.0"])
def test_comparison_rejects_nonfinite_negative_or_nonnumeric_norm(state, value):
    target = _summary()
    target["parameters"]["layer.weight"][state] = value
    with pytest.raises(ValueError, match="finite and nonnegative"):
        compare_optimizer_norms(target, _summary())


@pytest.mark.parametrize("state", COMPARISON_STATE_KEYS)
def test_comparison_requires_complete_per_parameter_state_coverage(state):
    target = _summary()
    target["parameters"]["layer.weight"].pop(state)
    with pytest.raises(ValueError, match="state coverage"):
        compare_optimizer_norms(target, _summary())


@pytest.mark.parametrize("shape", [[True], [-1], [1.0], "2"])
def test_comparison_rejects_invalid_shape_metadata(shape):
    target = _summary()
    target["parameters"]["layer.weight"]["shape"] = shape
    with pytest.raises(ValueError, match="invalid shape"):
        compare_optimizer_norms(target, _summary())


@pytest.mark.parametrize("name", ["", "model.layer.weight", "module.layer.weight"])
def test_comparison_rejects_noncanonical_names(name):
    target = _summary({name: _summary()["parameters"]["layer.weight"]})
    with pytest.raises(ValueError, match="canonical parameter"):
        compare_optimizer_norms(target, _summary())


def test_pairwise_norm_calibration_checks_every_pair(artifact_root):
    manifests = [
        _manifest(artifact_root / f"run-{index}", {"layer.weight": _states(torch.tensor([value]))})
        for index, value in enumerate((2.0, 0.0, 4.0))
    ]
    updates = []
    progress = SimpleNamespace(update=lambda: updates.append(1))

    worst = max_pairwise_optimizer_norm_delta(manifests, progress=progress, **KWARGS)

    assert (worst.name, worst.state, worst.delta_norm) == ("layer.weight", "parameter_update_residual", 4.0)
    assert updates == [1]


def test_pairwise_norm_calibration_sign_change_has_no_magnitude_variation(artifact_root):
    manifests = [
        _manifest(artifact_root / f"run-{index}", {"layer.weight": _states(torch.tensor([value]))})
        for index, value in enumerate((1.0, -1.0))
    ]
    assert max_pairwise_optimizer_norm_delta(manifests, **KWARGS).delta_norm == 0.0


def test_pairwise_norm_calibration_empty_or_single_capture(artifact_root):
    manifest = _manifest(artifact_root / "single", {"layer.weight": _states(torch.zeros(2))})
    assert max_pairwise_optimizer_norm_delta([], **KWARGS) is None
    assert max_pairwise_optimizer_norm_delta([manifest], **KWARGS) is None


@pytest.mark.parametrize("kind", ["names", "shapes"])
def test_pairwise_norm_calibration_requires_identical_coverage(artifact_root, kind):
    first = _manifest(artifact_root / "first", {"layer.weight": _states(torch.zeros(2))})
    second = _manifest(
        artifact_root / "second",
        {"other.weight" if kind == "names" else "layer.weight": _states(torch.zeros(3 if kind == "shapes" else 2))},
    )
    with pytest.raises(ValueError, match="parameter sets" if kind == "names" else "shape mismatch"):
        max_pairwise_optimizer_norm_delta([first, second], **KWARGS)


def _context_and_results(manifest, reference_norms=None, gate=0.0):
    def forbid_frozen_gate(*args):
        raise AssertionError("must not read a frozen full-tensor gate")

    ctx = SimpleNamespace(
        config_id="config",
        cfg=SimpleNamespace(training={"optimizer": {}}),
        spec=SimpleNamespace(tolerance_for=forbid_frozen_gate),
    )
    reference = {
        "optimizer_norms": reference_norms or _summary(),
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
    return ctx, SimpleNamespace(name="gas1"), reference, target


def test_arm_consumes_only_stored_norms_and_explicit_reviewed_gate(artifact_root):
    manifest = _manifest(artifact_root / "dss", {"layer.weight": _states(torch.tensor([0.01, 0.0]))})
    ctx, arm, reference, target = _context_and_results(manifest, gate=0.02)
    reference["optimizer_state_manifest"] = "do-not-open-reference-tensors"

    result = fwd_bwd_step._compare_arm(ctx, arm, reference, target)

    assert result.outcome is Outcome.PASS
    assert result.metrics["stated_criterion_abs"] == 0.02
    assert result.metrics["max_optimizer_norm_delta"] == pytest.approx(0.01)
    assert OPTIMIZER_NORM_CRITERION in result.reason
    assert "explicit-control-review" in result.reason
    assert "Equal norms can conceal" in result.reason


@pytest.mark.parametrize("state", COMPARISON_STATE_KEYS)
def test_arm_gates_every_scalar_and_reports_actual_norms(monkeypatch, state):
    target_summary = _summary()
    reference_summary = _summary()
    target_summary["parameters"]["layer.weight"][state] = 0.125
    reference_summary["parameters"]["layer.weight"][state] = 0.1
    monkeypatch.setattr(fwd_bwd_step, "summarize_optimizer_artifacts", lambda *args, **kwargs: target_summary)
    ctx, arm, reference, target = _context_and_results("target", reference_summary, gate=0.01)

    result = fwd_bwd_step._compare_arm(ctx, arm, reference, target)

    assert result.outcome is Outcome.FAIL
    assert result.metrics["optimizer_values_compared"] == 3
    assert result.metrics["optimizer_values_over_criterion"] == 1
    assert result.worst_name == f"layer.weight::{state}"
    (mismatch,) = result.mismatches
    assert mismatch.target == 0.125
    assert mismatch.reference == 0.1
    assert mismatch.abs_diff == pytest.approx(0.025)


def test_arm_gate_boundary_is_inclusive(monkeypatch):
    summary = _summary()
    summary["parameters"]["layer.weight"]["exp_avg"] = 0.125
    monkeypatch.setattr(fwd_bwd_step, "summarize_optimizer_artifacts", lambda *args, **kwargs: summary)
    ctx, arm, reference, target = _context_and_results("target", gate=0.125)
    assert fwd_bwd_step._compare_arm(ctx, arm, reference, target).outcome is Outcome.PASS


def test_arm_coverage_failure_is_not_a_zero_norm_pass(monkeypatch):
    summary = _summary({"visual.unused.weight": _summary()["parameters"]["layer.weight"]})
    monkeypatch.setattr(fwd_bwd_step, "summarize_optimizer_artifacts", lambda *args, **kwargs: summary)
    ctx, arm, reference, target = _context_and_results("target", gate=100)
    result = fwd_bwd_step._compare_arm(ctx, arm, reference, target)
    assert result.outcome is Outcome.FAIL
    assert result.metrics["unmatched_dss"] == 1
    assert result.metrics["unmatched_reference"] == 1


def test_arm_missing_norms_has_no_manifest_fallback(monkeypatch):
    def forbid_tensor_read(*args, **kwargs):
        raise AssertionError("must reject missing compact reference before reading artifacts")

    monkeypatch.setattr(fwd_bwd_step, "summarize_optimizer_artifacts", forbid_tensor_read)
    ctx, arm, reference, target = _context_and_results("target")
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
def test_arm_rejects_invalid_or_unreviewed_gate_before_artifact_read(monkeypatch, key, value):
    def forbid_tensor_read(*args, **kwargs):
        raise AssertionError("must reject invalid gate before reading artifacts")

    monkeypatch.setattr(fwd_bwd_step, "summarize_optimizer_artifacts", forbid_tensor_read)
    ctx, arm, reference, target = _context_and_results("target")
    reference["optimizer_norm_gate"][key] = value
    with pytest.raises(ValueError, match="optimizer_norm_gate"):
        fwd_bwd_step._compare_arm(ctx, arm, reference, target)


def test_arm_requires_gate_and_target_capture():
    ctx, arm, reference, target = _context_and_results(None)
    with pytest.raises(ValueError, match="target optimizer-state manifest"):
        fwd_bwd_step._compare_arm(ctx, arm, reference, target)
    reference.pop("optimizer_norm_gate")
    with pytest.raises(ValueError, match="reviewed optimizer_norm_gate"):
        fwd_bwd_step._compare_arm(ctx, arm, reference, target)
