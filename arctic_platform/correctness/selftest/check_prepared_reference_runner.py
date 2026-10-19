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

"""Regression consumes prepared references before allocating any target jobs."""

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.correctness.harness import reference_records
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness.arms import ArmDefinition
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.registry import TestResult as Result
from arctic_platform.correctness.harness.registry import TestSpecEntry as Entry
from arctic_platform.correctness.harness.spec import TestTolerance as Tolerance
from arctic_platform.correctness.selftest.check_applicable_tests import _spec

STEP_CHECKS = ["single-step-grads", "single-step-optimizer"]
WHOLE_CHECKS = ["checkpoint-resume-loss", "inference-checkpoint-loss"]


@pytest.fixture
def prepared_runner(tmp_path, monkeypatch):
    cfg = load_config(Path("arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config"))
    spec = _spec(STEP_CHECKS.copy())
    spec.config_id = cfg.config_id
    spec.model.cache_path = str(tmp_path / "reviewed-model")
    spec.test_tolerances["single-step-grads"] = Tolerance(absolute=1e-3)
    spec.test_settings["single-step-optimizer"] = {"reference_token_budget": 64}
    arms = [ArmDefinition("gas1", 4, 8), ArmDefinition("gas4", 16, 8)]
    events = []
    recipes = []
    published = []
    missing = set()
    record_overrides = {}
    contexts = []
    record = {
        "loss": 2.0,
        "microbatches": 1,
        "grad_norms": {"layer.weight": 1.0},
        "optimizer_norms": {
            "criterion": "optimizer-state-l2-norm-v1",
            "step": 1,
            "state_keys": ["parameter_update_residual", "exp_avg", "exp_avg_sq"],
            "parameter_count": 1,
            "optimizer_values_count": 3,
            "parameters": {
                "layer.weight": {
                    "shape": [2],
                    "parameter_update_residual": 0.0,
                    "exp_avg": 0.0,
                    "exp_avg_sq": 0.0,
                }
            },
        },
        "optimizer_norm_gate": {
            "criterion": "optimizer-state-l2-norm-v1",
            "absolute": 1e-3,
            "status": "reviewed",
            "provenance": "selftest prepared calibration",
        },
    }
    torch.save(
        {state: torch.zeros(2) for state in ("parameter_update", "exp_avg", "exp_avg_sq")},
        tmp_path / "target.pt",
    )
    manifest = tmp_path / "target-manifest.json"
    manifest.write_text(json.dumps({"step": 1, "parameters": [{"name": "module.layer.weight", "file": "target.pt"}]}))

    def model_identity(path):
        assert path == spec.model.cache_path
        return {"sha256": "mock-reviewed-model"}

    def workload_identity(ctx, check_id, recipe):
        assert ctx.cfg.config_id == cfg.config_id
        rows = recipe["synthetic_inputs"]["rows"]
        arm = next(arm for arm in arms if arm.global_batch_size == rows)
        recipes.append((check_id, arm.name, recipe))
        return {
            "check_id": check_id,
            "arm": arm.name,
            "active_tokens": recipe["synthetic_inputs"]["batch"]["active_tokens"],
        }

    def load_record(ctx, identity):
        contexts.append(ctx)
        key = (identity["check_id"], identity["arm"])
        events.append(("record", *key))
        if key in missing:
            raise reference_records.ReferencePrerequisiteError(f"prepared HF reference missing: {key[0]}/{key[1]}")
        ctx.reference_record_paths["/".join(key)] = str(tmp_path / "records" / f"{key[1]}-{key[0]}.json")
        return {
            **copy.deepcopy({**record, **record_overrides.get(key, {})}),
            "active_tokens": identity["active_tokens"],
        }

    def publish_record(ctx, identity, measurements):
        events.append(("publish", identity["check_id"], identity["arm"]))
        published.append((identity, measurements))

    @contextmanager
    def gateway(workdir, gpus, **kwargs):
        assert gpus == cfg.n_gpus == 4
        events.append(("gateway-enter",))
        yield "mock-target-session"
        events.append(("gateway-exit",))

    @contextmanager
    def running_job(session, payload):
        assert session == "mock-target-session"
        assert payload["model_name"] == spec.model.cache_path
        assert payload["training_config"]["debug"]["gradient_norms_per_param"]
        events.append(("job-enter",))
        yield "mock-target-job"
        events.append(("job-exit",))

    def target(session, job, body, *, learning_rate):
        assert session == "mock-target-session"
        assert job == "mock-target-job"
        assert learning_rate == runner.OPTIMIZER_LEARNING_RATE
        rows = body["batch"]["input_ids"].shape[0]
        arm = next(arm for arm in arms if arm.global_batch_size == rows)
        events.append(("target", arm.name))
        return SimpleNamespace(
            optimizer_state_manifest=str(manifest),
            avg_loss=2.0,
            grad_norms={"module.layer.weight": 1.0},
            model_calls=None,
        )

    def execute(*, selected=None, selected_arms=None, **kwargs):
        return runner.execute(
            cfg,
            spec,
            arms if selected_arms is None else selected_arms,
            vocab_size=128,
            select=STEP_CHECKS if selected is None else selected,
            workdir=tmp_path / "run",
            reference_records_dir=tmp_path / "records",
            **kwargs,
        )

    monkeypatch.setattr(reference_records, "model_identity", model_identity)
    monkeypatch.setattr(reference_records, "workload_identity", workload_identity)
    monkeypatch.setattr(reference_records, "load_record", load_record)
    monkeypatch.setattr(reference_records, "publish_record", publish_record)
    monkeypatch.setattr(runner, "run_reference", lambda *a, **k: pytest.fail("regression must not run HF"))
    monkeypatch.setattr(runner, "gateway", gateway)
    monkeypatch.setattr(runner, "running_job", running_job)
    monkeypatch.setattr(runner, "fwd_bwd_step", target)
    monkeypatch.setattr(runner, "uses_mixer_packing", lambda *a: False)
    return SimpleNamespace(
        cfg=cfg,
        spec=spec,
        arms=arms,
        events=events,
        recipes=recipes,
        published=published,
        missing=missing,
        record=record,
        record_overrides=record_overrides,
        contexts=contexts,
        target_manifest=manifest,
        execute=execute,
    )


@pytest.mark.parametrize("order", ["batched", "arm"])
def test_default_regression_uses_prepared_norms_for_both_arms(prepared_runner, order):
    control = prepared_runner
    results = control.execute(order=order)

    assert [(result.test_id, result.arm, result.outcome, result.gpus) for result in results] == [
        (check, arm.name, Outcome.PASS, 4) for arm in control.arms for check in STEP_CHECKS
    ]
    assert control.events[:4] == [("record", check, arm.name) for arm in control.arms for check in STEP_CHECKS]
    assert [event for event in control.events if event[0] == "target"] == [
        ("target", "gas1"),
        ("target", "gas4"),
    ]
    gateways = 1 if order == "batched" else 2
    assert control.events.count(("gateway-enter",)) == gateways
    assert control.events.count(("gateway-exit",)) == gateways
    assert control.events.count(("job-enter",)) == control.events.count(("job-exit",)) == 2
    assert not control.published
    assert len(control.recipes) == 4
    assert all(recipe["model"] == {"sha256": "mock-reviewed-model"} for _, _, recipe in control.recipes)
    assert all(
        recipe["optimizer"]["learning_rate"] == runner.OPTIMIZER_LEARNING_RATE for _, _, recipe in control.recipes
    )
    for result in results:
        if result.test_id == "single-step-optimizer":
            assert result.metrics["optimizer_values_compared"] == 3
            assert result.metrics["optimizer_values_over_criterion"] == 0
            assert result.metrics["stated_criterion_abs"] == 1e-3


def test_each_comparator_uses_its_own_prepared_reference(prepared_runner):
    control = prepared_runner
    misleading_optimizer_norms = copy.deepcopy(control.record["optimizer_norms"])
    misleading_optimizer_norms["parameters"]["layer.weight"]["exp_avg"] = 9.0
    for arm in control.arms:
        control.record_overrides[("single-step-grads", arm.name)] = {
            "optimizer_norms": misleading_optimizer_norms,
        }
        control.record_overrides[("single-step-optimizer", arm.name)] = {
            "grad_norms": {"layer.weight": 9.0},
            "loss": 3.0,
        }

    results = control.execute()

    assert len(results) == 4
    assert all(result.outcome is Outcome.PASS for result in results)
    for result in results:
        if result.test_id == "single-step-grads":
            assert result.metrics["reference_loss"] == 2.0
            assert result.metrics["max_abs_diff"] == 0.0
        else:
            assert result.test_id == "single-step-optimizer"
            assert result.metrics["reference_loss"] == 3.0
            assert result.metrics["max_optimizer_norm_delta"] == 0.0
    assert not control.published


@pytest.mark.parametrize("order", ["batched", "arm"])
def test_missing_gas4_reference_rejects_before_any_target(prepared_runner, order):
    control = prepared_runner
    control.missing.add(("single-step-optimizer", "gas4"))

    with pytest.raises(reference_records.ReferencePrerequisiteError, match="single-step-optimizer/gas4"):
        control.execute(order=order)

    assert control.events == [("record", check, arm.name) for arm in control.arms for check in STEP_CHECKS]
    assert not control.published


def test_prepare_mode_measures_hf_once_and_never_starts_target(prepared_runner, monkeypatch):
    control = prepared_runner
    control.missing.add(("single-step-grads", "gas1"))

    def reference(model, batch, output, **kwargs):
        assert model == control.spec.model.cache_path
        assert batch.name == "batch-gas1.pt"
        assert kwargs["optimizer_config"] is None
        control.events.append(("hf", "gas1"))
        return {
            **{key: value for key, value in control.record.items() if not key.startswith("optimizer_")},
            "active_tokens": control.recipes[-1][2]["synthetic_inputs"]["batch"]["active_tokens"],
        }

    monkeypatch.setattr(runner, "run_reference", reference)
    results = control.execute(
        selected=["single-step-grads"],
        selected_arms=control.arms[:1],
        prepare_references=True,
    )

    assert results == []
    assert control.events == [
        ("record", "single-step-grads", "gas1"),
        ("hf", "gas1"),
        ("publish", "single-step-grads", "gas1"),
    ]
    assert len(control.published) == 1
    identity, measurements = control.published[0]
    assert identity["check_id"] == "single-step-grads"
    assert measurements["grad_norms"] == {"layer.weight": 1.0}


def test_combined_prepare_preserves_existing_gradient_reference(prepared_runner, monkeypatch):
    control = prepared_runner
    control.missing.add(("single-step-optimizer", "gas1"))
    control.record_overrides[("single-step-grads", "gas1")] = {
        "grad_norms": {"layer.weight": 9.0},
        "loss": 9.0,
    }
    captures = []

    def reference(model, batch, output, **kwargs):
        assert model == control.spec.model.cache_path
        assert batch.name == "batch-gas1.pt"
        assert kwargs["optimizer_config"] == control.cfg.training["optimizer"]
        captures.append(kwargs["deterministic"])
        return {
            "loss": 3.0,
            "microbatches": 1,
            "active_tokens": control.recipes[-1][2]["synthetic_inputs"]["batch"]["active_tokens"],
            "grad_norms": {"layer.weight": 3.0},
            "optimizer_state_manifest": str(control.target_manifest),
        }

    monkeypatch.setattr(runner, "run_reference", reference)
    results = control.execute(
        selected_arms=control.arms[:1],
        prepare_references=True,
        reference_calibration_runs=2,
        reference_calibration_workers=1,
    )

    assert results == []
    assert captures == [True, False, False]
    assert control.events == [
        ("record", "single-step-grads", "gas1"),
        ("record", "single-step-optimizer", "gas1"),
        ("publish", "single-step-optimizer", "gas1"),
    ]
    assert len(control.published) == 1
    identity, measurements = control.published[0]
    assert identity["check_id"] == "single-step-optimizer"
    assert "optimizer_state_manifest" not in measurements
    assert measurements["optimizer_norms"]["criterion"] == "optimizer-state-l2-norm-v1"
    assert measurements["optimizer_norm_gate"]["status"] == "reviewed"
    records = control.contexts[0]._reference_by_check["gas1"]
    assert records["single-step-grads"]["grad_norms"] == {"layer.weight": 9.0}
    assert records["single-step-grads"]["loss"] == 9.0
    assert records["single-step-optimizer"] == measurements
    assert records["single-step-optimizer"]["grad_norms"] == {"layer.weight": 3.0}


@pytest.mark.parametrize("include_per_arm", [False, True])
def test_normal_whole_run_checkpoint_and_inference_still_execute(prepared_runner, monkeypatch, include_per_arm):
    control = prepared_runner
    registered = runner.registered_tests()

    def whole_check(check_id):
        def run(ctx):
            control.events.append(("whole", check_id))
            assert ctx.prepare_references is False
            assert ctx.cfg.model_name == control.cfg.model_name
            assert [arm.name for arm in ctx.arms] == ["gas1", "gas4"]
            if include_per_arm:
                assert set(ctx._reference) == set(ctx._target) == {"gas1", "gas4"}
            else:
                assert ctx._reference == ctx._target == {}
            return [Result(check_id, ctx.config_id, Outcome.PASS, summary="whole-run routing control")]

        return run

    for check in WHOLE_CHECKS:
        registered[check] = Entry(check, "whole-run control", "routing only", whole_check(check))
    monkeypatch.setattr(runner, "registered_tests", lambda: registered)
    selected = (STEP_CHECKS if include_per_arm else []) + WHOLE_CHECKS
    control.spec.applicable_tests = selected

    results = control.execute(selected=selected)

    assert [event for event in control.events if event[0] == "whole"] == [("whole", check) for check in WHOLE_CHECKS]
    assert [(result.test_id, result.outcome) for result in results[-2:]] == [
        (check, Outcome.PASS) for check in WHOLE_CHECKS
    ]
    if include_per_arm:
        assert control.events.index(("whole", WHOLE_CHECKS[0])) > control.events.index(("target", "gas4"))
    else:
        assert control.events == [("whole", check) for check in WHOLE_CHECKS]
