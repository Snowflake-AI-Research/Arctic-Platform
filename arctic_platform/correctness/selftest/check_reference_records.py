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

"""Static correctness selftests: reference records."""

from __future__ import annotations

import builtins
import contextlib
import copy
import hashlib
import json
import os
import shlex
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.correctness.checks import checkpoint_resume
from arctic_platform.correctness.harness import reference_records
from arctic_platform.correctness.harness import reference_records as records
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness.batches import SEED
from arctic_platform.correctness.harness.batches import build_batch
from arctic_platform.correctness.harness.config import ArmDefinition
from arctic_platform.correctness.harness.config import ArmSpec
from arctic_platform.correctness.harness.config import TestTolerance as Tolerance
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.registry import TestResult as Result
from arctic_platform.correctness.harness.registry import TestSpecEntry as Entry
from arctic_platform.correctness.reference.hf_single_gpu import fixed_row_groups
from arctic_platform.correctness.reference.hf_single_gpu import mixer_boundaries
from arctic_platform.correctness.reference.hf_single_gpu import mixer_call_inputs
from arctic_platform.correctness.reference.hf_single_gpu import reference_row_groups

from .check_catalog_configuration import _applicable_tests_spec

# Checkpoint calibration.


@pytest.mark.parametrize("slots,sp_size", [(8, 1), (16, 1), (8, 8), (16, 8)])
@pytest.mark.parametrize("provider", ["huggingface", "prime_rl"])
@pytest.mark.parametrize("reviewed_arms", [False, True])
def test_checkpoint_calibration_calibration_preserves_config_accumulation_and_identical_work(
    monkeypatch, tmp_path, slots, sp_size, provider, reviewed_arms
):
    training = {
        "n_gpus": slots,
        "sp_size": sp_size,
        "model_provider": provider,
        "mb_spec": {"max_tokens_per_mb": 64},
        "ds_config": {"gradient_accumulation_steps": 1},
    }
    original = deepcopy(training)
    definitions = [ArmDefinition(name, slots * gas, 64) for name, gas in (("gas1", 1), ("gas4", 4))]
    arms = [
        (
            ArmSpec(arm.name, arm.global_batch_size, arm.max_seq_len, arm.total_tokens, 1, 1, 0.0)
            if reviewed_arms
            else arm
        )
        for arm in definitions
    ]
    transports = []

    class Transport:

        def __init__(self, workdir, config, model, seed, **kwargs):
            assert config == original
            assert model == "/models/reduced" and seed == SEED
            self.workdir, self.options, self.jobs = (workdir, kwargs, [])
            transports.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        @contextlib.contextmanager
        def job(self):
            calls = []
            self.jobs.append(calls)

            def step(body, *, learning_rate):
                calls.append((body, learning_rate))
                return float(len(calls))

            yield SimpleNamespace(fwd_bwd_step=step)

    monkeypatch.setattr(checkpoint_resume, "GatewayTransport", Transport)
    raw, where = checkpoint_resume.measure_undeterministic_loss_range(
        training,
        "/models/reduced",
        slots=slots,
        vocab_size=128,
        arms=arms,
        workdir=tmp_path,
        attn_implementation="flash_attention_3",
    )
    assert len(transports) == 2
    assert raw == 0.0 and where == "gas1 loss at iteration 1"
    for arm, transport in zip(definitions, transports, strict=True):
        assert transport.workdir == tmp_path / arm.name
        assert transport.options == {"slots": slots, "attn_implementation": "flash_attention_3", "determinism": "off"}
        assert len(transport.jobs) == 3
        for job in transport.jobs:
            assert len(job) == 10
            for iteration, (body, rate) in enumerate(job, start=1):
                batch = build_batch(arm.name, arm.global_batch_size, arm.max_seq_len, 128, seed=SEED + iteration)
                assert isinstance(body["batch"], dict)
                wire = body["batch"]
                for field in ("input_ids", "position_ids"):
                    assert torch.equal(wire[field], getattr(batch, field))
                labels = batch.shifted_labels() if provider == "prime_rl" else batch.labels
                assert torch.equal(wire["labels"], labels)
                assert body["processing"] == {"loss_fn": "sft"}
                assert rate == checkpoint_resume.LEARNING_RATE
        for run in transport.jobs[1:]:
            assert all((body is prior for (body, _), (prior, _) in zip(run, transport.jobs[0], strict=True)))
    assert training == original


def test_checkpoint_calibration_qwen38_calibration_requires_four_gas4_microbatches():
    for slots in (8, 16):
        assert ArmDefinition("gas1", slots, 8192).microbatches(10240, slots) == 1
        assert ArmDefinition("gas4", slots * 4, 8192).microbatches(10240, slots) == 4


# Prepared reference runner.


_prepared_reference_runner_STEP_CHECKS = ["single-step-grads", "single-step-optimizer"]

_prepared_reference_runner_WHOLE_CHECKS = ["checkpoint-resume-loss", "inference-checkpoint-loss"]


@pytest.fixture
def _prepared_reference_runner_prepared_runner(tmp_path, monkeypatch):
    cfg = load_config(Path("arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config"))
    spec = _applicable_tests_spec(_prepared_reference_runner_STEP_CHECKS.copy())
    spec.config_id = cfg.config_id
    spec.model.cache_path = str(tmp_path / "reviewed-model")
    spec.test_tolerances["single-step-grads"] = Tolerance(absolute=0.001)
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
                "layer.weight": {"shape": [2], "parameter_update_residual": 0.0, "exp_avg": 0.0, "exp_avg_sq": 0.0}
            },
        },
        "optimizer_norm_gate": {
            "criterion": "optimizer-state-l2-norm-v1",
            "absolute": 0.001,
            "status": "reviewed",
            "provenance": "selftest prepared calibration",
        },
    }
    torch.save(
        {state: torch.zeros(2) for state in ("parameter_update", "exp_avg", "exp_avg_sq")}, tmp_path / "target.pt"
    )
    manifest = tmp_path / "target-manifest.json"
    manifest.write_text(json.dumps({"step": 1, "parameters": [{"name": "module.layer.weight", "file": "target.pt"}]}))

    def model_identity(path):
        assert path == spec.model.cache_path
        return {"sha256": "mock-reviewed-model"}

    def workload_identity(ctx, check_id, recipe):
        assert ctx.cfg.config_id == cfg.config_id
        rows = recipe["synthetic_inputs"]["rows"]
        arm = next((arm for arm in arms if arm.global_batch_size == rows))
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
        arm = next((arm for arm in arms if arm.global_batch_size == rows))
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
            select=_prepared_reference_runner_STEP_CHECKS if selected is None else selected,
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
def test_prepared_reference_runner_default_regression_uses_prepared_norms_for_both_arms(
    _prepared_reference_runner_prepared_runner, order
):
    control = _prepared_reference_runner_prepared_runner
    results = control.execute(order=order)
    assert [(result.test_id, result.arm, result.outcome, result.gpus) for result in results] == [
        (check, arm.name, Outcome.PASS, 4) for arm in control.arms for check in _prepared_reference_runner_STEP_CHECKS
    ]
    assert control.events[:4] == [
        ("record", check, arm.name) for arm in control.arms for check in _prepared_reference_runner_STEP_CHECKS
    ]
    assert [event for event in control.events if event[0] == "target"] == [("target", "gas1"), ("target", "gas4")]
    gateways = 1 if order == "batched" else 2
    assert control.events.count(("gateway-enter",)) == gateways
    assert control.events.count(("gateway-exit",)) == gateways
    assert control.events.count(("job-enter",)) == control.events.count(("job-exit",)) == 2
    assert not control.published
    assert len(control.recipes) == 4
    assert all((recipe["model"] == {"sha256": "mock-reviewed-model"} for _, _, recipe in control.recipes))
    assert all(
        (recipe["optimizer"]["learning_rate"] == runner.OPTIMIZER_LEARNING_RATE for _, _, recipe in control.recipes)
    )
    for result in results:
        if result.test_id == "single-step-optimizer":
            assert result.metrics["optimizer_values_compared"] == 3
            assert result.metrics["optimizer_values_over_criterion"] == 0
            assert result.metrics["stated_criterion_abs"] == 0.001


def test_prepared_reference_runner_each_comparator_uses_its_own_prepared_reference(
    _prepared_reference_runner_prepared_runner,
):
    control = _prepared_reference_runner_prepared_runner
    misleading_optimizer_norms = copy.deepcopy(control.record["optimizer_norms"])
    misleading_optimizer_norms["parameters"]["layer.weight"]["exp_avg"] = 9.0
    for arm in control.arms:
        control.record_overrides["single-step-grads", arm.name] = {"optimizer_norms": misleading_optimizer_norms}
        control.record_overrides["single-step-optimizer", arm.name] = {
            "grad_norms": {"layer.weight": 9.0},
            "loss": 3.0,
        }
    results = control.execute()
    assert len(results) == 4
    assert all((result.outcome is Outcome.PASS for result in results))
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
def test_prepared_reference_runner_missing_gas4_reference_rejects_before_any_target(
    _prepared_reference_runner_prepared_runner, order
):
    control = _prepared_reference_runner_prepared_runner
    control.missing.add(("single-step-optimizer", "gas4"))
    with pytest.raises(reference_records.ReferencePrerequisiteError, match="single-step-optimizer/gas4"):
        control.execute(order=order)
    assert control.events == [
        ("record", check, arm.name) for arm in control.arms for check in _prepared_reference_runner_STEP_CHECKS
    ]
    assert not control.published


def test_prepared_reference_runner_prepare_mode_measures_hf_once_and_never_starts_target(
    _prepared_reference_runner_prepared_runner, monkeypatch
):
    control = _prepared_reference_runner_prepared_runner
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
    results = control.execute(selected=["single-step-grads"], selected_arms=control.arms[:1], prepare_references=True)
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


def test_prepared_reference_runner_combined_prepare_preserves_existing_gradient_reference(
    _prepared_reference_runner_prepared_runner, monkeypatch
):
    control = _prepared_reference_runner_prepared_runner
    control.missing.add(("single-step-optimizer", "gas1"))
    control.record_overrides["single-step-grads", "gas1"] = {"grad_norms": {"layer.weight": 9.0}, "loss": 9.0}
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
def test_prepared_reference_runner_normal_whole_run_checkpoint_and_inference_still_execute(
    _prepared_reference_runner_prepared_runner, monkeypatch, include_per_arm
):
    control = _prepared_reference_runner_prepared_runner
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

    for check in _prepared_reference_runner_WHOLE_CHECKS:
        registered[check] = Entry(check, "whole-run control", "routing only", whole_check(check))
    monkeypatch.setattr(runner, "registered_tests", lambda: registered)
    selected = (
        _prepared_reference_runner_STEP_CHECKS if include_per_arm else []
    ) + _prepared_reference_runner_WHOLE_CHECKS
    control.spec.applicable_tests = selected
    results = control.execute(selected=selected)
    assert [event for event in control.events if event[0] == "whole"] == [
        ("whole", check) for check in _prepared_reference_runner_WHOLE_CHECKS
    ]
    assert [(result.test_id, result.outcome) for result in results[-2:]] == [
        (check, Outcome.PASS) for check in _prepared_reference_runner_WHOLE_CHECKS
    ]
    if include_per_arm:
        assert control.events.index(("whole", _prepared_reference_runner_WHOLE_CHECKS[0])) > control.events.index(
            ("target", "gas4")
        )
    else:
        assert control.events == [("whole", check) for check in _prepared_reference_runner_WHOLE_CHECKS]


# Reference grouping.


def test_reference_grouping_mixer_packing_preserves_one_row_reference_default() -> None:
    groups = reference_row_groups([8192] * 4, 8192, 65536, mixer_packing=True)
    assert groups == [[0], [1], [2], [3]]


def test_reference_grouping_mixer_packing_can_group_rows_for_diagnostics() -> None:
    groups = reference_row_groups([8192] * 10, 8192, 65536, mixer_packing=True, mixer_packing_group_rows=4)
    assert groups == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_reference_grouping_fixed_row_groups_rejects_non_positive_group_size() -> None:
    with pytest.raises(ValueError, match="group_rows"):
        fixed_row_groups([1, 2, 3], 0)


def test_reference_grouping_mixer_boundaries_describe_grouped_row_edges() -> None:
    boundaries = mixer_boundaries(row_length=4, rows=3, device="cpu")
    assert torch.equal(boundaries["cu_seq_lens_q"], torch.tensor([0, 4, 8, 12], dtype=torch.int32))
    assert torch.equal(
        boundaries["seq_idx"], torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]], dtype=torch.int32)
    )


def test_reference_grouping_mixer_boundaries_can_flatten_grouped_rows_for_varlen_decoder() -> None:
    boundaries = mixer_boundaries(row_length=4, rows=3, device="cpu", flattened=True)
    assert torch.equal(boundaries["cu_seq_lens_q"], torch.tensor([0, 4, 8, 12], dtype=torch.int32))
    assert torch.equal(boundaries["seq_idx"], torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]], dtype=torch.int32))


def test_reference_grouping_mixer_call_inputs_flatten_only_multirow_varlen_calls() -> None:
    input_ids = torch.arange(12).reshape(3, 4)
    labels = input_ids + 100
    packed_ids, packed_labels, mixer_kwargs = mixer_call_inputs(input_ids, labels, "cpu")
    assert torch.equal(packed_ids, torch.arange(12).reshape(1, 12))
    assert torch.equal(packed_labels, torch.arange(100, 112).reshape(1, 12))
    assert torch.equal(mixer_kwargs["seq_idx"], torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]]))
    one_ids, one_labels, one_kwargs = mixer_call_inputs(input_ids[:1], labels[:1], "cpu")
    assert one_ids.shape == (1, 4)
    assert one_labels.shape == (1, 4)
    assert one_kwargs["seq_idx"].shape == (1, 4)


# Reference records.


_reference_records_GRADS = "single-step-grads"

_reference_records_OPTIMIZER = "single-step-optimizer"

_reference_records_E2E = "e2e-sft-train-validate"

_reference_records_PARAMETER = "layers.0.self_attn.q_proj.lora_A.default.weight"


@dataclass
class _reference_records_Tolerance:
    absolute: float = 0.001
    calibration_runs: int = 16


@pytest.fixture
def _reference_records_ctx(tmp_path):
    return SimpleNamespace(
        config_id="unit-lora",
        cfg=SimpleNamespace(gpu_type="H200"),
        spec=SimpleNamespace(
            config_checksum="c" * 64,
            config_path="unit-lora.config",
            test_tolerances={_reference_records_GRADS: _reference_records_Tolerance()},
        ),
        reference_records_dir=tmp_path / "references",
        reference_record_paths={},
    )


@pytest.fixture
def _reference_records_identity(_reference_records_ctx, monkeypatch):
    monkeypatch.setattr(records.importlib.metadata, "version", lambda package: "unit-version")
    monkeypatch.setattr(records, "file_identity", lambda path: {"sha256": "a" * 64, "size": 1})
    return records.workload_identity(
        _reference_records_ctx,
        _reference_records_GRADS,
        {"arm": "gas1", "model": {"sha256": "b" * 64}, "batch": {"sha256": "d" * 64}},
    )


def _reference_records_identity_for(identity, check_id):
    result = copy.deepcopy(identity)
    result["check_id"] = check_id
    result["criterion"] = records.CRITERIA[check_id]
    if check_id != _reference_records_GRADS:
        result.pop("gate", None)
    return result


def _reference_records_measurements(check_id=_reference_records_GRADS):
    if check_id == _reference_records_E2E:
        return {
            "train_losses": [1.25 - step / 1000 for step in range(100)],
            "train_gradient_norms": [0.5] * 100,
            "optimizer_steps": 100,
            "validation_cross_entropy": 0.75,
            "scored_answer_tokens": 32,
        }
    measurements = {
        "loss": 1.25,
        "microbatches": 4,
        "active_tokens": 16,
        "grad_norms": {_reference_records_PARAMETER: 0.5},
    }
    if check_id == _reference_records_OPTIMIZER:
        measurements.update(
            optimizer_norms={
                "criterion": records.CRITERIA[_reference_records_OPTIMIZER],
                "step": 1,
                "state_keys": ["parameter_update_residual", "exp_avg", "exp_avg_sq"],
                "parameter_count": 1,
                "optimizer_values_count": 3,
                "parameters": {
                    _reference_records_PARAMETER: {
                        "shape": [2, 3],
                        "parameter_update_residual": 0.1,
                        "exp_avg": 0.05,
                        "exp_avg_sq": 0.001,
                    }
                },
            },
            optimizer_norm_gate={
                "criterion": records.CRITERIA[_reference_records_OPTIMIZER],
                "absolute": 0.001,
                "status": "reviewed",
                "provenance": {"review": "unit-onboarding", "runs": 16},
            },
        )
    return measurements


def _reference_records_no_hf(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"transformers", "peft"} or name.startswith("arctic_platform.correctness.reference"):
            raise AssertionError(f"prepared record attempted an HF import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


@pytest.mark.parametrize("check_id", [_reference_records_GRADS, _reference_records_OPTIMIZER, _reference_records_E2E])
def test_reference_records_publish_load_compact_measurements_without_hf(
    _reference_records_ctx, _reference_records_identity, monkeypatch, check_id
):
    _reference_records_identity = _reference_records_identity_for(_reference_records_identity, check_id)
    measurements = _reference_records_measurements(check_id)
    _reference_records_no_hf(monkeypatch)
    path = records.publish_record(_reference_records_ctx, _reference_records_identity, measurements)
    loaded = records.load_record(_reference_records_ctx, _reference_records_identity)
    assert loaded == measurements
    assert "measurements" not in loaded
    assert path.parent == _reference_records_ctx.reference_records_dir / _reference_records_ctx.config_id / check_id
    assert path.suffix == ".json"
    assert _reference_records_ctx.reference_record_paths == {path.stem: str(path)}
    payload = path.read_bytes()
    assert payload.endswith(b"\n")
    assert b"\n " not in payload
    assert len(payload) < 8192
    assert set(json.loads(payload)) == {"schema", "identity", "measurements", "payload_sha256"}
    assert not list(_reference_records_ctx.reference_records_dir.rglob("*.pt"))
    assert not list(_reference_records_ctx.reference_records_dir.rglob(".prepare-*"))


def test_reference_records_missing_record_has_actionable_onboarding_command_without_hf(
    _reference_records_ctx, _reference_records_identity, monkeypatch
):
    _reference_records_no_hf(monkeypatch)
    with pytest.raises(records.ReferencePrerequisiteError) as error:
        records.load_record(_reference_records_ctx, _reference_records_identity)
    assert "prepared HF reference missing" in str(error.value)
    assert (
        "python -m arctic_platform.correctness onboard --config unit-lora.config --prepare-references --test-id"
        " single-step-grads"
        in str(error.value)
    )
    assert str(_reference_records_ctx.reference_records_dir) in str(error.value)
    assert not _reference_records_ctx.reference_records_dir.exists()
    assert _reference_records_ctx.reference_record_paths == {}


def test_reference_records_missing_combined_reference_command_roundtrips_selection_and_workload(
    _reference_records_ctx, _reference_records_identity, monkeypatch
):
    _reference_records_no_hf(monkeypatch)
    _reference_records_ctx.spec.config_path = "configs/unit lora.config"
    _reference_records_ctx.reference_records_dir /= "compact records"
    _reference_records_ctx.local_gateway_full_model = True
    _reference_records_identity["recipe"].update(optimizer={"lr": 0.0001}, token_budget=64)
    with pytest.raises(records.ReferencePrerequisiteError) as error:
        records.load_record(_reference_records_ctx, _reference_records_identity)
    command = shlex.split(str(error.value).split("; run ", 1)[1])
    assert command == [
        "python",
        "-m",
        "arctic_platform.correctness",
        "onboard",
        "--config",
        _reference_records_ctx.spec.config_path,
        "--prepare-references",
        "--reference-tests",
        _reference_records_GRADS,
        _reference_records_OPTIMIZER,
        "--reference-records-dir",
        str(_reference_records_ctx.reference_records_dir),
        "--local-gateway-full-model",
        "--reference-token-budget",
        "64",
    ]
    assert not _reference_records_ctx.reference_records_dir.exists()
    assert not _reference_records_ctx.reference_record_paths


def test_reference_records_repeat_publication_is_byte_identical_and_does_not_replace_file(
    _reference_records_ctx, _reference_records_identity
):
    measurements = _reference_records_measurements()
    path = records.publish_record(_reference_records_ctx, _reference_records_identity, measurements)
    before = path.stat()
    payload = path.read_bytes()
    repeated = records.publish_record(
        _reference_records_ctx, _reference_records_identity, dict(reversed(list(measurements.items())))
    )
    assert repeated == path
    assert path.read_bytes() == payload
    assert path.stat().st_ino == before.st_ino
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert list(path.parent.glob("*.json")) == [path]


def test_reference_records_conflicting_publication_preserves_original_record(
    _reference_records_ctx, _reference_records_identity
):
    original = _reference_records_measurements()
    path = records.publish_record(_reference_records_ctx, _reference_records_identity, original)
    payload = path.read_bytes()
    changed = {**original, "loss": 2.0}
    with pytest.raises(records.ReferencePrerequisiteError, match="conflicting immutable"):
        records.publish_record(_reference_records_ctx, _reference_records_identity, changed)
    assert path.read_bytes() == payload
    assert records.load_record(_reference_records_ctx, _reference_records_identity) == original
    assert not list(path.parent.glob(".prepare-*"))


@pytest.mark.parametrize("corruption", ["measurement", "payload_hash", "schema", "identity"])
def test_reference_records_corrupt_or_incompatible_record_requires_onboarding(
    _reference_records_ctx, _reference_records_identity, corruption
):
    path = records.publish_record(
        _reference_records_ctx, _reference_records_identity, _reference_records_measurements()
    )
    record = json.loads(path.read_text())
    if corruption == "measurement":
        record["measurements"]["loss"] += 1
    elif corruption == "payload_hash":
        record["payload_sha256"] = "0" * 64
    elif corruption == "schema":
        record["schema"] = records.SCHEMA_VERSION + 1
    else:
        record["identity"]["config_checksum"] = "e" * 64
    path.write_text(json.dumps(record))
    _reference_records_ctx.reference_record_paths.clear()
    with pytest.raises(records.ReferencePrerequisiteError, match="corrupt or incompatible.*run"):
        records.load_record(_reference_records_ctx, _reference_records_identity)
    assert _reference_records_ctx.reference_record_paths == {}


@pytest.mark.parametrize("payload", ["{", "[]", "null"])
def test_reference_records_malformed_record_is_an_actionable_prerequisite_error(
    _reference_records_ctx, _reference_records_identity, payload
):
    path = records.publish_record(
        _reference_records_ctx, _reference_records_identity, _reference_records_measurements()
    )
    path.write_text(payload)
    with pytest.raises(records.ReferencePrerequisiteError, match="run.*--prepare-references"):
        records.load_record(_reference_records_ctx, _reference_records_identity)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_reference_records_nonfinite_record_payload_is_an_actionable_prerequisite_error(
    _reference_records_ctx, _reference_records_identity, nonfinite
):
    path = records.publish_record(
        _reference_records_ctx, _reference_records_identity, _reference_records_measurements()
    )
    record = json.loads(path.read_text())
    record["measurements"]["loss"] = nonfinite
    path.write_text(json.dumps(record))
    _reference_records_ctx.reference_record_paths.clear()
    with pytest.raises(records.ReferencePrerequisiteError, match="run.*--prepare-references"):
        records.load_record(_reference_records_ctx, _reference_records_identity)
    assert not _reference_records_ctx.reference_record_paths


@pytest.mark.parametrize("measurements", [None, [], 7, "measurements"])
def test_reference_records_nonobject_measurements_with_matching_hash_require_onboarding(
    _reference_records_ctx, _reference_records_identity, measurements
):
    path = records.publish_record(
        _reference_records_ctx, _reference_records_identity, _reference_records_measurements()
    )
    record = json.loads(path.read_text())
    record["measurements"] = measurements
    record["payload_sha256"] = records._digest(measurements)
    path.write_text(json.dumps(record))
    _reference_records_ctx.reference_record_paths.clear()
    with pytest.raises(records.ReferencePrerequisiteError, match="object.*run.*--prepare-references"):
        records.load_record(_reference_records_ctx, _reference_records_identity)
    assert not _reference_records_ctx.reference_record_paths


def test_reference_records_changed_workload_identity_does_not_reuse_old_record(
    _reference_records_ctx, _reference_records_identity
):
    path = records.publish_record(
        _reference_records_ctx, _reference_records_identity, _reference_records_measurements()
    )
    changed = copy.deepcopy(_reference_records_identity)
    changed["recipe"]["arm"] = "gas4"
    _reference_records_ctx.reference_record_paths.clear()
    with pytest.raises(records.ReferencePrerequisiteError, match="missing.*--prepare-references"):
        records.load_record(_reference_records_ctx, changed)
    assert list(_reference_records_ctx.reference_records_dir.rglob("*.json")) == [path]
    assert not _reference_records_ctx.reference_record_paths


@pytest.mark.parametrize("check_id", [_reference_records_GRADS, _reference_records_OPTIMIZER, _reference_records_E2E])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_reference_records_nonfinite_nested_measurements_are_not_published(
    _reference_records_ctx, _reference_records_identity, check_id, nonfinite
):
    _reference_records_identity = _reference_records_identity_for(_reference_records_identity, check_id)
    measurements = _reference_records_measurements(check_id)
    measurements["diagnostic"] = {"nested": [nonfinite]}
    with pytest.raises(records.ReferencePrerequisiteError, match="nonfinite.*diagnostic.nested"):
        records.publish_record(_reference_records_ctx, _reference_records_identity, measurements)
    assert not _reference_records_ctx.reference_records_dir.exists()


@pytest.mark.parametrize("field", ["loss", "microbatches", "active_tokens"])
def test_reference_records_incomplete_single_step_measurements_are_rejected(
    _reference_records_ctx, _reference_records_identity, field
):
    measurements = _reference_records_measurements()
    del measurements[field]
    with pytest.raises(records.ReferencePrerequisiteError, match=f"missing {field}"):
        records.publish_record(_reference_records_ctx, _reference_records_identity, measurements)


@pytest.mark.parametrize(
    "norms", [{}, None, {_reference_records_PARAMETER: -0.1}, {_reference_records_PARAMETER: "0.1"}, {7: 0.1}]
)
def test_reference_records_gradient_norms_require_named_nonnegative_scalars(
    _reference_records_ctx, _reference_records_identity, norms
):
    measurements = {**_reference_records_measurements(), "grad_norms": norms}
    with pytest.raises(records.ReferencePrerequisiteError, match="gradient norms"):
        records.publish_record(_reference_records_ctx, _reference_records_identity, measurements)


@pytest.mark.parametrize("field", ["train_losses", "train_gradient_norms"])
@pytest.mark.parametrize("length", [0, 99, 101])
def test_reference_records_e2e_requires_exactly_100_losses_and_norms(
    _reference_records_ctx, _reference_records_identity, field, length
):
    _reference_records_identity = _reference_records_identity_for(_reference_records_identity, _reference_records_E2E)
    measurements = _reference_records_measurements(_reference_records_E2E)
    measurements[field] = [0.5] * length
    with pytest.raises(records.ReferencePrerequisiteError, match=f"all 100 scalar {field}"):
        records.publish_record(_reference_records_ctx, _reference_records_identity, measurements)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("train_losses", [1.0] * 99 + ["1.0"]),
        ("train_gradient_norms", [0.5] * 99 + [None]),
        ("optimizer_steps", 99),
        ("validation_cross_entropy", None),
        ("scored_answer_tokens", 0),
        ("scored_answer_tokens", 1.5),
    ],
)
def test_reference_records_incomplete_e2e_scoring_or_steps_are_rejected(
    _reference_records_ctx, _reference_records_identity, field, invalid
):
    _reference_records_identity = _reference_records_identity_for(_reference_records_identity, _reference_records_E2E)
    measurements = {**_reference_records_measurements(_reference_records_E2E), field: invalid}
    with pytest.raises(records.ReferencePrerequisiteError):
        records.publish_record(_reference_records_ctx, _reference_records_identity, measurements)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("criterion", "old-criterion"),
        ("status", "unreviewed"),
        ("provenance", {}),
        ("absolute", -0.001),
        ("absolute", float("inf")),
    ],
)
def test_reference_records_optimizer_gate_requires_reviewed_finite_norm_criterion(
    _reference_records_ctx, _reference_records_identity, field, invalid
):
    _reference_records_identity = _reference_records_identity_for(
        _reference_records_identity, _reference_records_OPTIMIZER
    )
    measurements = _reference_records_measurements(_reference_records_OPTIMIZER)
    measurements["optimizer_norm_gate"][field] = invalid
    with pytest.raises(records.ReferencePrerequisiteError):
        records.publish_record(_reference_records_ctx, _reference_records_identity, measurements)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("absolute", True),
        ("absolute", False),
        ("provenance", " "),
        ("provenance", True),
        ("provenance", 7),
        ("provenance", ["unit-review"]),
    ],
)
def test_reference_records_optimizer_gate_rejects_bool_tolerance_and_invalid_provenance(
    _reference_records_ctx, _reference_records_identity, field, invalid
):
    _reference_records_identity = _reference_records_identity_for(
        _reference_records_identity, _reference_records_OPTIMIZER
    )
    measurements = _reference_records_measurements(_reference_records_OPTIMIZER)
    measurements["optimizer_norm_gate"][field] = invalid
    with pytest.raises(records.ReferencePrerequisiteError, match="gate|provenance"):
        records.publish_record(_reference_records_ctx, _reference_records_identity, measurements)
    assert not _reference_records_ctx.reference_records_dir.exists()
    assert not _reference_records_ctx.reference_record_paths


def test_reference_records_optimizer_requires_parameter_norms(_reference_records_ctx, _reference_records_identity):
    _reference_records_identity = _reference_records_identity_for(
        _reference_records_identity, _reference_records_OPTIMIZER
    )
    measurements = _reference_records_measurements(_reference_records_OPTIMIZER)
    measurements["optimizer_norms"]["parameters"] = {}
    with pytest.raises(records.ReferencePrerequisiteError, match="no parameter norms"):
        records.publish_record(_reference_records_ctx, _reference_records_identity, measurements)


def test_reference_records_optimizer_gate_does_not_accept_legacy_reviewed_boolean(
    _reference_records_ctx, _reference_records_identity
):
    _reference_records_identity = _reference_records_identity_for(
        _reference_records_identity, _reference_records_OPTIMIZER
    )
    measurements = _reference_records_measurements(_reference_records_OPTIMIZER)
    del measurements["optimizer_norm_gate"]["status"]
    measurements["optimizer_norm_gate"]["reviewed"] = True
    with pytest.raises(records.ReferencePrerequisiteError, match="review provenance"):
        records.publish_record(_reference_records_ctx, _reference_records_identity, measurements)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("step", 2),
        ("state_keys", ["parameter_update", "exp_avg", "exp_avg_sq"]),
        ("parameter_count", 2),
        ("optimizer_values_count", 2),
        ("parameters", {_reference_records_PARAMETER: {"shape": [2, 3], "exp_avg": 0.05}}),
        (
            "parameters",
            {
                _reference_records_PARAMETER: {
                    "shape": [2, 3],
                    "exp_avg": 0.05,
                    "exp_avg_sq": 0.001,
                    "parameter_update_residual": -0.1,
                }
            },
        ),
    ],
)
def test_reference_records_optimizer_summary_is_validated_before_publication(
    _reference_records_ctx, _reference_records_identity, field, invalid
):
    _reference_records_identity = _reference_records_identity_for(
        _reference_records_identity, _reference_records_OPTIMIZER
    )
    measurements = _reference_records_measurements(_reference_records_OPTIMIZER)
    measurements["optimizer_norms"][field] = invalid
    with pytest.raises(ValueError, match="optimizer norm"):
        records.publish_record(_reference_records_ctx, _reference_records_identity, measurements)
    assert not _reference_records_ctx.reference_records_dir.exists()


def test_reference_records_load_revalidates_measurements_even_when_payload_hash_matches(
    _reference_records_ctx, _reference_records_identity
):
    path = records.publish_record(
        _reference_records_ctx, _reference_records_identity, _reference_records_measurements()
    )
    record = json.loads(path.read_text())
    record["measurements"]["grad_norms"] = {}
    record["payload_sha256"] = records._digest(record["measurements"])
    path.write_text(json.dumps(record))
    with pytest.raises(records.ReferencePrerequisiteError, match="gradient norms"):
        records.load_record(_reference_records_ctx, _reference_records_identity)


def test_reference_records_concurrent_identical_writers_publish_one_complete_record(
    _reference_records_ctx, _reference_records_identity
):
    barrier = Barrier(4)
    measurements = _reference_records_measurements()

    def publish():
        barrier.wait(timeout=10)
        return records.publish_record(_reference_records_ctx, _reference_records_identity, measurements)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(publish) for _ in range(4)]
        paths = [future.result(timeout=20) for future in futures]
    assert len(set(paths)) == 1
    assert records.load_record(_reference_records_ctx, _reference_records_identity) == measurements
    assert list(paths[0].parent.iterdir()) == [paths[0]]


def test_reference_records_concurrent_conflicting_writers_do_not_overwrite_winner(
    _reference_records_ctx, _reference_records_identity
):
    barrier = Barrier(2)

    def publish(loss):
        measurements = {**_reference_records_measurements(), "loss": loss}
        barrier.wait(timeout=10)
        try:
            return (
                "published",
                loss,
                records.publish_record(_reference_records_ctx, _reference_records_identity, measurements),
            )
        except records.ReferencePrerequisiteError as error:
            return ("conflict", loss, str(error))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(publish, loss) for loss in (1.25, 2.5)]
        results = [future.result(timeout=20) for future in futures]
    winner = next((result for result in results if result[0] == "published"))
    conflict = next((result for result in results if result[0] == "conflict"))
    assert "conflicting immutable" in conflict[2]
    assert records.load_record(_reference_records_ctx, _reference_records_identity)["loss"] == winner[1]
    assert list(winner[2].parent.iterdir()) == [winner[2]]


def test_reference_records_publication_is_visible_only_after_complete_bytes_are_synced(
    _reference_records_ctx, _reference_records_identity, monkeypatch
):
    measurements = _reference_records_measurements()
    original_fsync = records.os.fsync
    original_link = records.os.link
    synced = []
    observed = []

    def fsync(fd):
        original_fsync(fd)
        synced.append(fd)

    def link(source, destination):
        assert synced
        assert not destination.exists()
        with open(source, "rb") as stream:
            pending = json.load(stream)
        assert pending["measurements"] == measurements
        assert pending["payload_sha256"] == records._digest(measurements)
        original_link(source, destination)
        observed.append(records.load_record(_reference_records_ctx, _reference_records_identity))

    monkeypatch.setattr(records.os, "fsync", fsync)
    monkeypatch.setattr(records.os, "link", link)
    path = records.publish_record(_reference_records_ctx, _reference_records_identity, measurements)
    assert observed == [measurements]
    assert not list(path.parent.glob(".prepare-*"))


@pytest.mark.parametrize("failure", ["fsync", "link"])
def test_reference_records_failed_publication_removes_temporary_files(
    _reference_records_ctx, _reference_records_identity, monkeypatch, failure
):

    def fail(*args, **kwargs):
        raise OSError("unit publication failure")

    monkeypatch.setattr(records.os, failure, fail)
    with pytest.raises(OSError, match="unit publication failure"):
        records.publish_record(_reference_records_ctx, _reference_records_identity, _reference_records_measurements())
    assert not list(_reference_records_ctx.reference_records_dir.rglob("*.json"))
    assert not list(_reference_records_ctx.reference_records_dir.rglob(".prepare-*"))
    assert not _reference_records_ctx.reference_record_paths


def test_reference_records_byte_cap_is_checked_before_publication_creates_files(
    _reference_records_ctx, _reference_records_identity, monkeypatch
):
    monkeypatch.setattr(records, "MAX_RECORD_BYTES", 1)
    with pytest.raises(records.ReferencePrerequisiteError, match="size limit"):
        records.publish_record(_reference_records_ctx, _reference_records_identity, _reference_records_measurements())
    assert not _reference_records_ctx.reference_records_dir.exists()
    assert not _reference_records_ctx.reference_record_paths


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("config_id", "../outside"),
        ("config_id", "."),
        ("config_id", ".."),
        ("config_id", "/outside"),
        ("config_id", "nested/config"),
        ("config_id", "nested\\config"),
        ("config_id", ""),
        ("config_id", None),
        ("check_id", "../outside"),
        ("check_id", "/outside"),
        ("check_id", "checkpoint-resume-loss"),
    ],
)
@pytest.mark.parametrize("operation", ["publish", "load"])
def test_reference_records_invalid_record_path_identity_is_rejected_before_filesystem_writes(
    _reference_records_ctx, _reference_records_identity, field, invalid, operation
):
    changed = {**_reference_records_identity, field: invalid}
    with pytest.raises(records.ReferencePrerequisiteError, match="invalid reference record"):
        if operation == "publish":
            records.publish_record(_reference_records_ctx, changed, _reference_records_measurements())
        else:
            records.load_record(_reference_records_ctx, changed)
    assert not _reference_records_ctx.reference_records_dir.exists()
    assert not _reference_records_ctx.reference_record_paths


@pytest.fixture
def _reference_records_checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / ".git").mkdir()
    monkeypatch.setattr(
        records, "__file__", str(root / "arctic_platform" / "correctness" / "harness" / "reference_records.py")
    )
    return root


def test_reference_records_git_records_directory_accepts_checkout_path_without_creating_it(
    _reference_records_checkout,
):
    destination = _reference_records_checkout / "arctic_platform" / "correctness" / "reference_data"
    assert records.require_git_records_dir(destination) == destination
    assert not destination.exists()


@pytest.mark.parametrize("relative", ["../outside", "../checkout-sibling", ".git", ".git/objects"])
def test_reference_records_git_records_directory_rejects_outside_checkout_and_git_metadata(
    _reference_records_checkout, relative
):
    with pytest.raises(records.ReferencePrerequisiteError, match="Git checkout"):
        records.require_git_records_dir(_reference_records_checkout / relative)


@pytest.mark.parametrize("target", ["outside", "git"])
def test_reference_records_git_records_directory_rejects_symlink_escape(_reference_records_checkout, target):
    destination = (
        _reference_records_checkout.parent / "outside" if target == "outside" else _reference_records_checkout / ".git"
    )
    destination.mkdir(exist_ok=True)
    alias = _reference_records_checkout / "records"
    alias.symlink_to(destination, target_is_directory=True)
    with pytest.raises(records.ReferencePrerequisiteError, match="Git checkout"):
        records.require_git_records_dir(alias / "new-references")
    assert not (destination / "new-references").exists()


def test_reference_records_exact_byte_cap_loads_but_one_byte_over_cap_fails(
    _reference_records_ctx, _reference_records_identity, monkeypatch
):
    measurements = _reference_records_measurements()
    path = records.publish_record(_reference_records_ctx, _reference_records_identity, measurements)
    size = path.stat().st_size
    monkeypatch.setattr(records, "MAX_RECORD_BYTES", size)
    assert records.publish_record(_reference_records_ctx, _reference_records_identity, measurements) == path
    assert records.load_record(_reference_records_ctx, _reference_records_identity) == measurements
    monkeypatch.setattr(records, "MAX_RECORD_BYTES", size - 1)
    _reference_records_ctx.reference_record_paths.clear()
    with pytest.raises(records.ReferencePrerequisiteError, match="oversized"):
        records.load_record(_reference_records_ctx, _reference_records_identity)
    assert not _reference_records_ctx.reference_record_paths


def test_reference_records_workload_identity_binds_inputs_config_and_gate_not_local_record_root(
    _reference_records_ctx, _reference_records_identity
):
    recipe = copy.deepcopy(_reference_records_identity["recipe"])
    same = records.workload_identity(_reference_records_ctx, _reference_records_GRADS, recipe)
    relocated = copy.deepcopy(_reference_records_ctx)
    relocated.reference_records_dir = _reference_records_ctx.reference_records_dir / "another-machine"
    relocated.spec.config_path = "/another-machine/unit-lora.config"
    assert records.workload_identity(relocated, _reference_records_GRADS, recipe) == same
    changed = copy.deepcopy(recipe)
    changed["batch"]["sha256"] = "e" * 64
    assert records.workload_identity(_reference_records_ctx, _reference_records_GRADS, changed) != same
    _reference_records_ctx.spec.config_checksum = "f" * 64
    assert records.workload_identity(_reference_records_ctx, _reference_records_GRADS, recipe) != same
    _reference_records_ctx.spec.config_checksum = _reference_records_identity["config_checksum"]
    _reference_records_ctx.spec.test_tolerances[_reference_records_GRADS].absolute = 0.002
    assert records.workload_identity(_reference_records_ctx, _reference_records_GRADS, recipe) != same


def test_reference_records_workload_identity_supports_missing_optional_dependencies(
    _reference_records_ctx, _reference_records_identity, monkeypatch
):
    import torch

    def absent(package):
        raise records.importlib.metadata.PackageNotFoundError(package)

    monkeypatch.setattr(records.importlib.metadata, "version", absent)
    result = records.workload_identity(
        _reference_records_ctx, _reference_records_GRADS, _reference_records_identity["recipe"]
    )
    assert result["dependencies"]
    assert result["dependencies"]["torch"] == str(torch.__version__)
    assert all((value is None for package, value in result["dependencies"].items() if package != "torch"))


def test_reference_records_workload_identity_uses_loaded_torch_build_not_distribution_metadata(
    _reference_records_ctx, _reference_records_identity, monkeypatch
):
    import torch

    monkeypatch.setattr(torch, "__version__", "2.13.0+cu130")
    monkeypatch.setattr(
        records.importlib.metadata, "version", lambda package: "2.13.0" if package == "torch" else "unit-version"
    )
    without_suffix = records.workload_identity(
        _reference_records_ctx, _reference_records_E2E, _reference_records_identity["recipe"]
    )
    monkeypatch.setattr(
        records.importlib.metadata, "version", lambda package: "2.13.0+cu130" if package == "torch" else "unit-version"
    )
    with_suffix = records.workload_identity(
        _reference_records_ctx, _reference_records_E2E, _reference_records_identity["recipe"]
    )
    assert without_suffix == with_suffix
    assert with_suffix["dependencies"]["torch"] == "2.13.0+cu130"
    monkeypatch.setattr(torch, "__version__", "2.13.0+cu128")
    assert (
        records.workload_identity(
            _reference_records_ctx, _reference_records_E2E, _reference_records_identity["recipe"]
        )
        != with_suffix
    )


def test_reference_records_metadata_only_torch_drift_reuses_record_without_hf(
    _reference_records_ctx, _reference_records_identity, monkeypatch
):
    import torch

    monkeypatch.setattr(torch, "__version__", "2.13.0+cu130")
    monkeypatch.setattr(
        records.importlib.metadata, "version", lambda package: "2.13.0+cu130" if package == "torch" else "unit-version"
    )
    prepared = records.workload_identity(
        _reference_records_ctx, _reference_records_E2E, _reference_records_identity["recipe"]
    )
    measurements = _reference_records_measurements(_reference_records_E2E)
    records.publish_record(_reference_records_ctx, prepared, measurements)
    monkeypatch.setattr(
        records.importlib.metadata, "version", lambda package: "2.13.0" if package == "torch" else "unit-version"
    )
    _reference_records_no_hf(monkeypatch)
    target = records.workload_identity(
        _reference_records_ctx, _reference_records_E2E, _reference_records_identity["recipe"]
    )
    assert records.load_record(_reference_records_ctx, target) == measurements


def test_reference_records_workload_identity_rejects_non_reference_check(_reference_records_ctx):
    with pytest.raises(ValueError, match="not an HF reference check"):
        records.workload_identity(_reference_records_ctx, "checkpoint-resume-loss", {})


def test_reference_records_tensor_semantic_hash_is_independent_of_torch_save_archive(tmp_path, monkeypatch):
    import torch

    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    first, second = (tmp_path / "first.pt", tmp_path / "second.pt")
    torch.save(tensor, first)
    torch.save(tensor.clone(), second)
    assert hashlib.sha256(first.read_bytes()).digest() != hashlib.sha256(second.read_bytes()).digest()
    first_tensor = torch.load(first, map_location="cpu", weights_only=True)
    second_tensor = torch.load(second, map_location="cpu", weights_only=True)

    def forbidden_save(*args, **kwargs):
        raise AssertionError("semantic tensor hashing must not serialize a torch archive")

    monkeypatch.setattr(torch, "save", forbidden_save)
    assert records.tensor_identity(first_tensor) == records.tensor_identity(second_tensor)


def test_reference_records_tensor_semantic_hash_does_not_consume_or_depend_on_global_rng():
    import torch

    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        before = torch.random.get_rng_state().clone()
        identity = records.tensor_identity(tensor)
        assert torch.equal(torch.random.get_rng_state(), before)
        torch.rand(32)
        advanced = torch.random.get_rng_state().clone()
        assert records.tensor_identity(tensor) == identity
        assert torch.equal(torch.random.get_rng_state(), advanced)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "int64"])
def test_reference_records_tensor_semantic_hash_normalizes_storage_layout_and_detaches(dtype):
    import torch

    tensor = torch.arange(6, dtype=getattr(torch, dtype)).reshape(2, 3)
    noncontiguous = tensor.t().contiguous().t()
    assert not noncontiguous.is_contiguous()
    assert records.tensor_identity(noncontiguous) == records.tensor_identity(tensor)
    if tensor.is_floating_point():
        tensor.requires_grad_()
        assert records.tensor_identity(tensor)["dtype"] == f"torch.{dtype}"
        assert tensor.grad is None


def test_reference_records_tensor_semantic_hash_binds_shape_dtype_and_values():
    import torch

    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    identity = records.tensor_identity(tensor)
    assert identity["shape"] == [2, 3]
    assert identity["dtype"] == "torch.float32"
    assert len(identity["sha256"]) == 64
    assert records.tensor_identity(tensor.reshape(6))["sha256"] != identity["sha256"]
    assert records.tensor_identity(tensor.view(torch.int32))["sha256"] != identity["sha256"]
    changed = tensor.clone()
    changed[0, 0] = 0.5
    assert records.tensor_identity(changed)["sha256"] != identity["sha256"]


@pytest.mark.parametrize("field", ["input_ids", "position_ids", "labels"])
def test_reference_records_batch_identity_binds_all_token_semantics(field):
    import torch

    batch = SimpleNamespace(
        name="train-step-001",
        active_tokens=2,
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.int64),
        position_ids=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        labels=torch.tensor([[-100, 2, 3]], dtype=torch.int64),
    )
    identity = records.batch_identity(batch)
    assert identity["name"] == batch.name
    assert identity["active_tokens"] == 2
    assert set(identity["tensors"]) == {"input_ids", "position_ids", "labels"}
    changed = copy.deepcopy(batch)
    getattr(changed, field)[0, 1] += 1
    assert records.batch_identity(changed)["tensors"][field] != identity["tensors"][field]


def _reference_records_model_assets(root):
    root.mkdir()
    (root / "config.json").write_text('{"num_hidden_layers":4}\n')
    (root / "model.safetensors").write_bytes(b"unit model weights A")
    return root


def test_reference_records_model_identity_detects_same_size_content_drift_with_restored_mtime(tmp_path):
    root = _reference_records_model_assets(tmp_path / "model")
    before = records.model_identity(root)
    weights = root / "model.safetensors"
    old_stat = weights.stat()
    time.sleep(0.02)
    weights.write_bytes(b"unit model weights B")
    os.utime(weights, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    after = records.model_identity(root)
    assert weights.stat().st_size == old_stat.st_size
    assert weights.stat().st_mtime_ns == old_stat.st_mtime_ns
    assert before["assets"]["model.safetensors"]["sha256"] != after["assets"]["model.safetensors"]["sha256"]
    assert before["sha256"] != after["sha256"]


def test_reference_records_model_identity_is_independent_of_asset_root(tmp_path):
    first = _reference_records_model_assets(tmp_path / "first")
    second = _reference_records_model_assets(tmp_path / "second")
    assert records.model_identity(first) == records.model_identity(second)


@pytest.mark.parametrize("missing", ["directory", "config.json", "model.safetensors"])
def test_reference_records_model_identity_requires_complete_external_assets(tmp_path, missing):
    root = tmp_path / "model"
    if missing != "directory":
        _reference_records_model_assets(root)
        (root / missing).unlink()
    with pytest.raises(records.ReferencePrerequisiteError, match="assets unavailable|complete model"):
        records.model_identity(root)
