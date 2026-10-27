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
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier
from threading import Event
from types import FunctionType
from types import MethodType
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
from arctic_platform.correctness.reference import hf_single_gpu
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


@pytest.mark.parametrize("override", [None, "eager"])
def test_reference_attention_runner_uses_same_glm_backend_in_measurement_and_identity(
    _prepared_reference_runner_prepared_runner, monkeypatch, override
):
    control = _prepared_reference_runner_prepared_runner
    control.cfg.sub_job["model_name"] = "zai-org/GLM-5.3-Flash-BF16"
    control.cfg.training["attn_implementation"] = "sparse_mla"
    control.missing.add(("single-step-grads", "gas1"))
    expected = override or "sdpa"
    calls = []

    def reference(model, batch, output, **kwargs):
        assert kwargs["attn"] == expected
        assert control.recipes[-1][2]["attention"] == expected
        assert control.contexts[-1].attn_implementation == "sparse_mla"
        calls.append(kwargs)
        return {
            **{key: value for key, value in control.record.items() if not key.startswith("optimizer_")},
            "active_tokens": control.recipes[-1][2]["synthetic_inputs"]["batch"]["active_tokens"],
        }

    monkeypatch.setattr(runner, "run_reference", reference)
    assert (
        control.execute(
            selected=["single-step-grads"],
            selected_arms=control.arms[:1],
            prepare_references=True,
            reference_attn=override,
        )
        == []
    )
    assert len(calls) == len(control.published) == 1
    assert all(event[0] not in {"gateway-enter", "target"} for event in control.events)


@pytest.mark.parametrize(
    ("relative", "target", "override", "expected"),
    [
        ("glm5.3-flash-bf16/h200/train-sft-16gpus-sp16-256k.config", "sparse_mla", None, "sdpa"),
        ("glm5.3-flash-bf16/h200/train-sft-16gpus-sp16-256k.config", "sparse_mla", "eager", "eager"),
        ("qwen3-8b/h200/train-sft-full-4gpus-64k.config", "eager", None, "eager"),
    ],
)
def test_reference_attention_e2e_command_and_options_preserve_target_and_explicit_overrides(
    tmp_path, monkeypatch, relative, target, override, expected
):
    from arctic_platform.correctness.checks import e2e_sft_train_validate

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / relative)
    spec = _applicable_tests_spec([])
    ctx = runner.RunContext(
        config_id=cfg.config_id,
        cfg=cfg,
        spec=spec,
        arms=[],
        attn_implementation=target,
        workdir=tmp_path,
        batch_paths={},
        vocab_size=16,
        reference_attn_implementation=override,
    )
    monkeypatch.setattr(e2e_sft_train_validate, "uses_mixer_packing", lambda *args: False)
    options = e2e_sft_train_validate.reference_options(ctx, spec.model.cache_path)
    command = e2e_sft_train_validate.reference_command(ctx, None, tmp_path / "replay", tmp_path / "output", None)
    assert options["attn"] == command[command.index("--attn") + 1] == expected
    assert ctx.attn_implementation == target


@pytest.mark.parametrize("arm_name", ["gas1", "gas4"])
def test_prepared_reference_runner_combined_prepare_reuses_optimizer_calibration(
    _prepared_reference_runner_prepared_runner, monkeypatch, arm_name
):
    control = _prepared_reference_runner_prepared_runner
    arm = next(arm for arm in control.arms if arm.name == arm_name)
    control.missing.add(("single-step-grads", arm_name))
    retained = {
        "loss": 9.0,
        "optimizer_norm_gate": {
            **copy.deepcopy(control.record["optimizer_norm_gate"]),
            "absolute": 0.017,
            "provenance": "retained reviewed optimizer calibration",
        },
    }
    control.record_overrides["single-step-optimizer", arm_name] = retained
    captures = []

    def reference(model, batch, output, **kwargs):
        assert model == control.spec.model.cache_path
        assert batch.name == f"batch-{arm_name}.pt"
        assert kwargs["optimizer_config"] == control.cfg.training["optimizer"]
        assert kwargs["deterministic"] is True
        captures.append(output)
        return {
            "loss": 3.0,
            "microbatches": 1,
            "active_tokens": control.recipes[-1][2]["synthetic_inputs"]["batch"]["active_tokens"],
            "grad_norms": {"layer.weight": 3.0},
            "optimizer_state_manifest": str(control.target_manifest),
        }

    def unexpected_calibration(*args, **kwargs):
        pytest.fail("a retained optimizer record must not be recalibrated")

    from arctic_platform.correctness.checks import optimizer_state

    monkeypatch.setattr(runner, "run_reference", reference)
    monkeypatch.setattr(optimizer_state, "summarize_optimizer_artifacts", unexpected_calibration)
    monkeypatch.setattr(optimizer_state, "max_pairwise_optimizer_norm_delta", unexpected_calibration)
    assert control.execute(selected_arms=[arm], prepare_references=True) == []
    assert len(captures) == 1
    assert control.events == [
        ("record", "single-step-grads", arm_name),
        ("record", "single-step-optimizer", arm_name),
        ("publish", "single-step-grads", arm_name),
    ]
    identity, measurements = control.published[0]
    assert identity["check_id"] == "single-step-grads"
    assert measurements["loss"] == 3.0
    assert measurements["grad_norms"] == {"layer.weight": 3.0}
    assert "optimizer_state_manifest" not in measurements
    assert "optimizer_norm_gate" not in measurements
    records = control.contexts[0]._reference_by_check[arm_name]
    assert records["single-step-grads"] == measurements
    assert records["single-step-optimizer"]["loss"] == retained["loss"]
    assert records["single-step-optimizer"]["optimizer_norm_gate"] == retained["optimizer_norm_gate"]


@pytest.mark.parametrize("workers", [1, 2])
def test_optimizer_calibration_progress_completes_each_arm_and_retains_all_reference_runs(
    _prepared_reference_runner_prepared_runner, monkeypatch, capsys, workers
):
    control = _prepared_reference_runner_prepared_runner
    control.missing.update(("single-step-optimizer", arm.name) for arm in control.arms)
    calls = []

    def reference(model, batch, output, **kwargs):
        calls.append((batch.name, kwargs["deterministic"]))
        return {
            "loss": 3.0,
            "grad_norms": {"layer.weight": 3.0},
            "optimizer_state_manifest": str(control.target_manifest),
        }

    monkeypatch.setattr(runner, "run_reference", reference)
    assert control.execute(prepare_references=True, reference_calibration_workers=workers) == []
    output = capsys.readouterr().out
    for arm in control.arms:
        assert f"Optimizer calibration {arm.name}" in output
        for completed in range(1, 9):
            assert f"{completed}/8" in output
        assert output.index(f"Optimizer calibration {arm.name}") < output.index(
            f"*** Optimizer Norm Summary: {arm.name} ***"
        )
        assert calls.count((f"batch-{arm.name}.pt", True)) == 1
        assert calls.count((f"batch-{arm.name}.pt", False)) == 8
    assert len(control.published) == 2
    assert all(
        measurements["optimizer_norm_gate"]["provenance"]["calibration_runs"] == 8
        for identity, measurements in control.published
    )
    assert all(event[0] not in {"gateway-enter", "target"} for event in control.events)


@pytest.mark.parametrize("available", [8, 16])
def test_optimizer_calibration_defaults_to_one_eight_gpu_wave_per_arm(
    _prepared_reference_runner_prepared_runner, monkeypatch, available
):
    from threading import Barrier
    from threading import Lock

    control = _prepared_reference_runner_prepared_runner
    control.missing.update(("single-step-optimizer", arm.name) for arm in control.arms)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: available)
    barrier = Barrier(8, timeout=10)
    lock = Lock()
    active = set()
    completed = {}
    baseline = []

    def reference(model, batch, output, **kwargs):
        arm = batch.stem.removeprefix("batch-")
        if kwargs["deterministic"]:
            assert kwargs["cuda_device"] is None
            baseline.append(arm)
        else:
            index = int(output.stem.rsplit("-", 1)[-1])
            device = kwargs["cuda_device"]
            with lock:
                assert device == index % 8
                assert device not in active
                assert 0 <= index < 8
                active.add(device)
            try:
                barrier.wait()
            finally:
                with lock:
                    active.remove(device)
                    completed.setdefault(arm, set()).add(index)
        return {
            "loss": 3.0,
            "grad_norms": {"layer.weight": 3.0},
            "optimizer_state_manifest": str(control.target_manifest),
        }

    monkeypatch.setattr(runner, "run_reference", reference)
    assert control.execute(prepare_references=True) == []
    assert not active
    assert baseline == [arm.name for arm in control.arms]
    assert completed == {arm.name: set(range(8)) for arm in control.arms}
    assert len(control.published) == len(control.arms)
    assert all(
        measurements["optimizer_norm_gate"]["provenance"]["calibration_runs"] == 8
        for identity, measurements in control.published
    )
    assert all(event[0] not in {"gateway-enter", "target"} for event in control.events)


def test_optimizer_calibration_failed_run_leaves_partial_progress_without_publication(
    _prepared_reference_runner_prepared_runner, monkeypatch, capsys
):
    control = _prepared_reference_runner_prepared_runner
    control.missing.add(("single-step-optimizer", "gas1"))
    count = 0

    def reference(model, batch, output, **kwargs):
        nonlocal count
        if not kwargs["deterministic"]:
            count += 1
            if count == 2:
                raise RuntimeError("synthetic calibration failure")
        return {
            "loss": 3.0,
            "grad_norms": {"layer.weight": 3.0},
            "optimizer_state_manifest": str(control.target_manifest),
        }

    monkeypatch.setattr(runner, "run_reference", reference)
    with pytest.raises(RuntimeError, match="synthetic calibration failure"):
        control.execute(
            selected_arms=control.arms[:1],
            prepare_references=True,
            reference_calibration_workers=1,
        )
    output = capsys.readouterr().out
    assert "Optimizer calibration gas1" in output and "1/8" in output
    assert "8/8" not in output and "Optimizer Norm Summary" not in output
    assert control.published == []
    assert all(event[0] not in {"gateway-enter", "target"} for event in control.events)


def test_prepared_reference_runner_combined_prepare_preserves_existing_gradient_reference(
    _prepared_reference_runner_prepared_runner, monkeypatch, capsys
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
    output = capsys.readouterr().out
    assert "\n*** Optimizer Norm Summary: gas1 ***\n" in output
    assert "Reading saved optimizer states from 2 completed HF runs." in output
    assert "Optimizer norm summary complete: arm=gas1, elapsed=" in output
    assert "maximum_pairwise_norm_delta=0.000000000e+00\n\n" in output


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


@pytest.fixture
def _historical_gradient_record(_reference_records_ctx, monkeypatch):
    historical = {
        "schema": records.SCHEMA_VERSION,
        "check_id": _reference_records_GRADS,
        "criterion": records.CRITERIA[_reference_records_GRADS],
        "config_id": _reference_records_ctx.config_id,
        "config_checksum": _reference_records_ctx.spec.config_checksum,
        "parameter_name_map": {},
        "gpu_family": "H200",
        "reference_sources": {
            "reference/hf_single_gpu.py": {"sha256": "a" * 64, "size": 10},
            "harness/names.py": copy.deepcopy(records._REVIEWED_GRADIENT_NAMES_TRANSITION["historical"]),
        },
        "dependencies": {"torch": "unit-torch", "transformers": "unit-transformers"},
        "python": "3.12.15",
        "recipe": {
            "arm": "gas1",
            "model": {"sha256": "b" * 64},
            "batch": {"sha256": "d" * 64},
            "token_budget": 10240,
        },
        "gate": {"absolute": 0.011, "calibration_runs": 16},
    }
    current = copy.deepcopy(historical)
    current["reference_sources"]["harness/names.py"] = copy.deepcopy(
        records._REVIEWED_GRADIENT_NAMES_TRANSITION["current"]
    )
    measurements = {**_reference_records_measurements(), "skipped_no_grad": ["visual.patch_embed.weight"]}
    path = records.publish_record(_reference_records_ctx, historical, measurements)
    raw = path.read_bytes()
    binding = {
        "record_path": str(path),
        "record_sha256": hashlib.sha256(raw).hexdigest(),
        "review": "Exact DSS-only shared-expert alias transition reviewed; HF source unchanged.",
    }
    _reference_records_ctx.reference_record_paths.clear()
    _reference_records_no_hf(monkeypatch)
    return SimpleNamespace(
        ctx=_reference_records_ctx,
        historical=historical,
        current=current,
        measurements=measurements,
        path=path,
        raw=raw,
        binding=binding,
    )


def _historical_gradient_rewrite(fixture, record):
    raw = records._json(record)
    fixture.path.write_bytes(raw)
    fixture.binding["record_sha256"] = hashlib.sha256(raw).hexdigest()


def test_historical_gradient_record_explicit_binding_preserves_original_and_both_identities(
    _historical_gradient_record,
):
    fixture = _historical_gradient_record
    before = fixture.path.stat()
    with pytest.raises(records.ReferencePrerequisiteError, match="missing"):
        records.load_record(fixture.ctx, fixture.current)
    assert not hasattr(fixture.ctx, "reference_record_provenance")
    loaded = records.load_record(fixture.ctx, fixture.current, historical_binding=fixture.binding)
    assert loaded == fixture.measurements
    assert fixture.path.read_bytes() == fixture.raw
    assert fixture.path.stat().st_ino == before.st_ino
    assert fixture.path.stat().st_mtime_ns == before.st_mtime_ns
    assert list(fixture.ctx.reference_records_dir.rglob("*.json")) == [fixture.path]
    key = records._digest(fixture.current)
    assert fixture.ctx.reference_record_paths == {key: str(fixture.path)}
    assert fixture.ctx.reference_record_provenance == {
        key: {
            "historical_identity": fixture.historical,
            "current_identity": fixture.current,
            **fixture.binding,
        }
    }
    fixture.current["gate"]["absolute"] = 99.0
    fixture.binding["review"] = "mutated caller"
    provenance = fixture.ctx.reference_record_provenance[key]
    assert provenance["current_identity"]["gate"]["absolute"] == 0.011
    assert provenance["historical_identity"]["gate"]["absolute"] == 0.011
    assert provenance["review"] != "mutated caller"


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "check_id",
        "criterion",
        "config_id",
        "config_checksum",
        "parameter_name_map",
        "gpu_family",
        "reference_sources",
        "dependencies",
        "python",
        "recipe",
        "gate",
        "extra_field",
    ],
)
def test_historical_gradient_record_rejects_every_other_workload_field(_historical_gradient_record, field):
    fixture = _historical_gradient_record
    if field == "schema":
        fixture.current[field] = True
    elif field == "check_id":
        fixture.current[field] = _reference_records_OPTIMIZER
    elif field == "parameter_name_map":
        fixture.current[field] = {"wrong": "alias"}
    elif field == "reference_sources":
        fixture.current[field]["reference/hf_single_gpu.py"]["sha256"] = "e" * 64
    elif field == "dependencies":
        fixture.current[field]["torch"] = "different-torch"
    elif field == "recipe":
        fixture.current[field]["batch"]["sha256"] = "e" * 64
    elif field == "gate":
        fixture.current[field]["absolute"] = 0.025
    else:
        fixture.current[field] = "different"
    with pytest.raises(records.ReferencePrerequisiteError, match="another workload field|gradients only"):
        records.load_record(fixture.ctx, fixture.current, historical_binding=fixture.binding)
    assert not fixture.ctx.reference_record_paths
    assert not hasattr(fixture.ctx, "reference_record_provenance")


@pytest.mark.parametrize(
    "invalid",
    [
        {},
        {"extra": "waiver"},
        {"record_sha256": "0" * 64},
        {"record_sha256": "A" * 64},
        {"record_sha256": "abc"},
        {"review": ""},
        {"review": " "},
        {"review": {"override": True}},
        {"review": True},
        {"record_path": ""},
        {"record_path": 7},
    ],
)
def test_historical_gradient_record_requires_exact_binding_and_raw_pin(_historical_gradient_record, invalid):
    fixture = _historical_gradient_record
    binding = {**fixture.binding, **invalid} if invalid else {}
    with pytest.raises(records.ReferencePrerequisiteError):
        records.load_record(fixture.ctx, fixture.current, historical_binding=binding)
    assert not fixture.ctx.reference_record_paths
    assert not hasattr(fixture.ctx, "reference_record_provenance")


@pytest.mark.parametrize("side", ["historical", "current"])
def test_historical_gradient_record_rejects_unreviewed_names_transition(_historical_gradient_record, side):
    fixture = _historical_gradient_record
    if side == "current":
        fixture.current["reference_sources"]["harness/names.py"]["sha256"] = "0" * 64
    else:
        record = json.loads(fixture.raw)
        record["identity"]["reference_sources"]["harness/names.py"]["sha256"] = "0" * 64
        renamed = fixture.path.with_name(f"{records._digest(record['identity'])}.json")
        fixture.path.rename(renamed)
        fixture.path = renamed
        fixture.binding["record_path"] = str(renamed)
        _historical_gradient_rewrite(fixture, record)
    with pytest.raises(records.ReferencePrerequisiteError, match="transition is not reviewed"):
        records.load_record(fixture.ctx, fixture.current, historical_binding=fixture.binding)


def test_historical_gradient_record_revalidates_actual_current_names_source(_historical_gradient_record, monkeypatch):
    fixture = _historical_gradient_record
    monkeypatch.setattr(records, "file_identity", lambda path: {"sha256": "0" * 64, "size": 10294})
    with pytest.raises(records.ReferencePrerequisiteError, match="current names.py source mismatch"):
        records.load_record(fixture.ctx, fixture.current, historical_binding=fixture.binding)
    assert not fixture.ctx.reference_record_paths
    assert not hasattr(fixture.ctx, "reference_record_provenance")


@pytest.mark.parametrize("corruption", ["schema", "filename", "payload", "measurements"])
def test_historical_gradient_record_validates_original_record(_historical_gradient_record, corruption):
    fixture = _historical_gradient_record
    record = json.loads(fixture.raw)
    if corruption == "schema":
        record["schema"] = True
    elif corruption == "filename":
        renamed = fixture.path.with_name(f"{records._digest(fixture.current)}.json")
        fixture.path.rename(renamed)
        fixture.path = renamed
        fixture.binding["record_path"] = str(renamed)
    elif corruption == "payload":
        record["measurements"]["loss"] += 1
    else:
        record["measurements"]["grad_norms"] = {}
        record["payload_sha256"] = records._digest(record["measurements"])
    _historical_gradient_rewrite(fixture, record)
    with pytest.raises(records.ReferencePrerequisiteError, match="schema|filename|payload|gradient norms"):
        records.load_record(fixture.ctx, fixture.current, historical_binding=fixture.binding)
    assert not fixture.ctx.reference_record_paths
    assert not hasattr(fixture.ctx, "reference_record_provenance")


@pytest.mark.parametrize("bad_json", ["duplicate", "nested_duplicate", "nan", "overflow", "malformed"])
def test_historical_gradient_record_strict_json(_historical_gradient_record, bad_json):
    fixture = _historical_gradient_record
    text = fixture.raw.decode()
    if bad_json == "duplicate":
        text = '{"schema":1,' + text[1:]
    elif bad_json == "nested_duplicate":
        text = text.replace('"loss":1.25', '"loss":1.25,"loss":1.25')
    elif bad_json == "nan":
        text = text.replace('"loss":1.25', '"loss":NaN')
    elif bad_json == "overflow":
        text = text.replace('"loss":1.25', '"loss":1e999')
    else:
        text = "{"
    raw = text.encode()
    fixture.path.write_bytes(raw)
    fixture.binding["record_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(records.ReferencePrerequisiteError):
        records.load_record(fixture.ctx, fixture.current, historical_binding=fixture.binding)
    assert not fixture.ctx.reference_record_paths
    assert not hasattr(fixture.ctx, "reference_record_provenance")


def test_historical_gradient_record_never_falls_back_to_current_record(_historical_gradient_record):
    fixture = _historical_gradient_record
    current_path = records.publish_record(fixture.ctx, fixture.current, fixture.measurements)
    fixture.ctx.reference_record_paths.clear()
    fixture.binding["record_sha256"] = "0" * 64
    with pytest.raises(records.ReferencePrerequisiteError, match="raw SHA256 mismatch"):
        records.load_record(fixture.ctx, fixture.current, historical_binding=fixture.binding)
    assert current_path.exists()
    assert not fixture.ctx.reference_record_paths
    assert not hasattr(fixture.ctx, "reference_record_provenance")


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


@pytest.mark.parametrize("returncode", [0, 17])
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("durable", [False, True])
def test_reference_observability_subprocess_retains_complete_logs_and_returncode(
    tmp_path, monkeypatch, capsys, returncode, fallback, durable
):
    from arctic_platform.correctness.harness import storage

    output = Path("/tmp") / tmp_path.name / "reference.json" if fallback else tmp_path / "reference.json"
    log_dir = tmp_path / "fallback" if fallback else tmp_path
    stdout = "reference first output\n" + "s" * 6000 + "\nreference last output\n"
    stderr = "Traceback (most recent call last):\n" + "e" * 6000 + "\nRuntimeError: child failure\n"
    payload = {"loss": 2.0, "microbatches": 3, "grad_norms": {"layer.weight": 4.0}}
    commands = []
    fallbacks = []
    read_text = Path.read_text

    def read_output(path, *args, **kwargs):
        return json.dumps(payload) if path == output else read_text(path, *args, **kwargs)

    def child(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    def fast_fallback(prefix):
        fallbacks.append(prefix)
        return log_dir

    monkeypatch.setattr(runner.subprocess, "run", child)
    monkeypatch.setattr(runner, "correctness_workdir", fast_fallback)
    if durable:
        monkeypatch.setenv(storage.CORRECTNESS_LOGDIR_ENV, str(tmp_path / "work" / "logs"))
        monkeypatch.setattr(storage, "correctness_logdir", lambda prefix: log_dir)
    else:
        monkeypatch.delenv(storage.CORRECTNESS_LOGDIR_ENV, raising=False)
    monkeypatch.setattr(Path, "read_text", read_output)
    options = {"token_budget": 64, "ce_chunk": 8, "attn": "sdpa", "fp32_lm_head": False, "cuda_device": 0}
    if returncode:
        with pytest.raises(RuntimeError, match=r"reference failed \(17\)") as failure:
            runner.run_reference("read-only-model", tmp_path / "batch.pt", output, **options)
        assert str(log_dir / f"{output.stem}.stdout.log") in str(failure.value)
        assert str(log_dir / f"{output.stem}.stderr.log") in str(failure.value)
        assert stdout[-4000:] in str(failure.value)
        assert stderr[-4000:] in str(failure.value)
    else:
        assert runner.run_reference("read-only-model", tmp_path / "batch.pt", output, **options) == payload
    assert len(commands) == 1
    command, subprocess_options = commands[0]
    assert command[command.index("--out") + 1] == str(output)
    assert command[command.index("--seed") + 1] == str(SEED)
    assert "--deterministic" in command
    assert subprocess_options["capture_output"] is True
    assert subprocess_options["text"] is True
    assert subprocess_options["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert fallbacks == (["hf-reference-logs-"] if fallback and not durable else [])
    assert (log_dir / f"{output.stem}.stdout.log").read_text() == stdout
    assert (log_dir / f"{output.stem}.stderr.log").read_text() == stderr
    assert (log_dir / f"{output.stem}.returncode").read_text() == f"{returncode}\n"
    assert str(log_dir.resolve()).startswith("/data-fast/")
    assert not output.exists()
    message = capsys.readouterr().out
    assert f"run={output.stem}" in message
    assert f"returncode={returncode}" in message
    for suffix in ("stdout.log", "stderr.log", "returncode"):
        assert str(log_dir / f"{output.stem}.{suffix}") in message


def test_reference_observability_keeps_successful_process_logs_when_result_json_is_invalid(tmp_path, monkeypatch):
    output = tmp_path / "reference.json"
    calls = []

    def child(command, **kwargs):
        calls.append(command)
        output.write_text("{not valid JSON}")
        return SimpleNamespace(returncode=0, stdout="child completed\n", stderr="child warning\n")

    monkeypatch.setattr(runner.subprocess, "run", child)
    with pytest.raises(json.JSONDecodeError):
        runner.run_reference(
            "read-only-model",
            tmp_path / "batch.pt",
            output,
            token_budget=64,
            ce_chunk=8,
            attn="sdpa",
            fp32_lm_head=False,
        )
    assert len(calls) == 1
    assert (tmp_path / "reference.stdout.log").read_text() == "child completed\n"
    assert (tmp_path / "reference.stderr.log").read_text() == "child warning\n"
    assert (tmp_path / "reference.returncode").read_text() == "0\n"
    assert output.read_text() == "{not valid JSON}"


def test_reference_observability_calibration_consumes_completed_runs_without_changing_records(
    _prepared_reference_runner_prepared_runner, monkeypatch, capsys
):
    from arctic_platform.correctness.checks import optimizer_state

    control = _prepared_reference_runner_prepared_runner
    control.missing.add(("single-step-optimizer", "gas1"))
    control.record_overrides["single-step-grads", "gas1"] = {"grad_norms": {"layer.weight": 9.0}, "loss": 9.0}
    original_cfg = deepcopy(control.cfg.training)
    original_gate = deepcopy(control.spec.test_tolerances["single-step-grads"])
    release_slow = Event()
    calls = []
    consumed = []
    reductions = []

    class CompletionTrackingPool(ThreadPoolExecutor):
        def submit(self, function, *args, **kwargs):
            future = super().submit(function, *args, **kwargs)
            original_result = future.result

            def result(timeout=None):
                value = original_result(timeout)
                consumed.append(kwargs["output_name"])
                if kwargs["output_name"] == "gas1-calibration-1":
                    release_slow.set()
                return value

            future.result = result
            return future

    def reference(model, batch, output, **kwargs):
        assert model == control.spec.model.cache_path
        assert batch.name == "batch-gas1.pt"
        assert kwargs["optimizer_config"] == original_cfg["optimizer"]
        assert kwargs["learning_rate"] == runner.OPTIMIZER_LEARNING_RATE
        calls.append((output.stem, kwargs["deterministic"], kwargs["cuda_device"]))
        if output.stem == "ref-gas1-calibration-0":
            assert release_slow.wait(5), "completed calibration run was blocked behind a pending earlier run"
        return {
            "loss": 3.0,
            "microbatches": 1,
            "active_tokens": control.recipes[-1][2]["synthetic_inputs"]["batch"]["active_tokens"],
            "grad_norms": {"layer.weight": 3.0},
            "optimizer_state_manifest": str(
                control.target_manifest if kwargs["deterministic"] else output.with_suffix(".manifest.json")
            ),
        }

    def norm_delta(manifests, **kwargs):
        reductions.append((list(manifests), kwargs))
        return SimpleNamespace(delta_norm=0.0021, name="layer.weight", state="exp_avg")

    monkeypatch.setattr(runner, "ThreadPoolExecutor", CompletionTrackingPool)
    monkeypatch.setattr(runner, "run_reference", reference)
    monkeypatch.setattr(optimizer_state, "max_pairwise_optimizer_norm_delta", norm_delta)
    try:
        assert (
            control.execute(
                selected_arms=control.arms[:1],
                prepare_references=True,
                reference_calibration_runs=3,
                reference_calibration_workers=2,
            )
            == []
        )
    finally:
        release_slow.set()

    assert calls[0] == ("ref-gas1", True, None)
    assert sorted(calls[1:]) == [
        ("ref-gas1-calibration-0", False, 0),
        ("ref-gas1-calibration-1", False, 1),
        ("ref-gas1-calibration-2", False, 0),
    ]
    assert consumed == ["gas1-calibration-1", "gas1-calibration-0", "gas1-calibration-2"]
    assert reductions == [
        (
            [control.contexts[0].workdir / f"ref-gas1-calibration-{index}.manifest.json" for index in range(3)],
            {"optimizer_config": original_cfg["optimizer"], "learning_rate": runner.OPTIMIZER_LEARNING_RATE},
        )
    ]
    assert control.cfg.training == original_cfg
    assert control.spec.test_tolerances["single-step-grads"] == original_gate
    assert [event for event in control.events if event[0] == "publish"] == [
        ("publish", "single-step-optimizer", "gas1")
    ]
    assert not any(event[0] in {"gateway-enter", "job-enter", "target"} for event in control.events)
    identity, measurements = control.published[0]
    assert identity["check_id"] == "single-step-optimizer"
    assert measurements["loss"] == 3.0
    assert measurements["grad_norms"] == {"layer.weight": 3.0}
    assert "optimizer_state_manifest" not in measurements
    assert measurements["optimizer_norm_gate"] == {
        "criterion": "optimizer-state-l2-norm-v1",
        "absolute": 0.009,
        "status": "reviewed",
        "provenance": {
            "policy": "onboarding-norm-repeatability-4x-upward-1e-3-v1",
            "calibration_runs": 3,
            "maximum_pairwise_norm_delta": 0.0021,
            "worst_quantity": "layer.weight::exp_avg",
            "multiplier": 4,
            "minimum_gate": 0.001,
        },
    }
    cached = control.contexts[0]._reference_by_check["gas1"]
    assert cached["single-step-grads"]["loss"] == 9.0
    assert cached["single-step-grads"]["grad_norms"] == {"layer.weight": 9.0}
    assert cached["single-step-optimizer"] == measurements
    lines = capsys.readouterr().out.splitlines()
    for run, device in [(0, 0), (1, 1), (2, 0)]:
        matching = [line for line in lines if "gas1" in line and f"gas1-calibration-{run}" in line]
        assert any("start" in line and f"device={device}" in line for line in matching)
        assert any("done" in line and f"device={device}" in line for line in matching)
        assert not any("failed" in line for line in matching)


def test_reference_observability_calibration_error_keeps_existing_record_and_does_not_publish(
    _prepared_reference_runner_prepared_runner, monkeypatch, capsys
):
    control = _prepared_reference_runner_prepared_runner
    control.missing.add(("single-step-optimizer", "gas1"))
    control.record_overrides["single-step-grads", "gas1"] = {"grad_norms": {"layer.weight": 9.0}, "loss": 9.0}
    calls = []

    def reference(model, batch, output, **kwargs):
        calls.append((output.stem, kwargs["deterministic"], kwargs["cuda_device"]))
        if not kwargs["deterministic"]:
            raise RuntimeError("synthetic calibration childreturncode=17")
        return {
            "loss": 3.0,
            "microbatches": 1,
            "active_tokens": control.recipes[-1][2]["synthetic_inputs"]["batch"]["active_tokens"],
            "grad_norms": {"layer.weight": 3.0},
            "optimizer_state_manifest": str(control.target_manifest),
        }

    monkeypatch.setattr(runner, "run_reference", reference)
    with pytest.raises(RuntimeError, match="synthetic calibration childreturncode=17"):
        control.execute(
            selected_arms=control.arms[:1],
            prepare_references=True,
            reference_calibration_runs=2,
            reference_calibration_workers=1,
        )
    assert calls == [("ref-gas1", True, None), ("ref-gas1-calibration-0", False, 0)]
    assert not control.published
    assert not any(event[0] in {"gateway-enter", "job-enter", "target"} for event in control.events)
    assert control.record_overrides["single-step-grads", "gas1"] == {
        "grad_norms": {"layer.weight": 9.0},
        "loss": 9.0,
    }
    lines = capsys.readouterr().out.splitlines()
    matching = [line for line in lines if "gas1-calibration-0" in line]
    assert any("start" in line and "device=0" in line for line in matching)
    assert any("failed" in line and "device=0" in line for line in matching)
    assert not any("done" in line for line in matching)


# Actual HF execution provenance.


class _ReferenceExecutionModel:
    def __init__(self, config):
        self.config = config

    def forward(self, input_ids):
        return input_ids


class _ReferenceExecutionDecoder(_ReferenceExecutionModel):
    def forward(self, input_ids):
        return input_ids + 1


@pytest.fixture
def _reference_execution_engine():
    def config(layers, attention):
        values = {"model_type": "synthetic", "num_hidden_layers": layers, "use_cache": False}
        return SimpleNamespace(values=values, to_dict=lambda: values, _attn_implementation=attention)

    return hf_single_gpu.ReferenceEngine(
        model=_ReferenceExecutionModel(config(4, "sdpa")),
        decoder=_ReferenceExecutionDecoder(config(4, "eager")),
        head=None,
        hidden_size=1,
        head_name=None,
        device="cpu",
    )


def test_reference_execution_witness_records_actual_configurations_and_selected_forward(_reference_execution_engine):
    engine = _reference_execution_engine
    witness = hf_single_gpu.execution_witness(engine, requested_attention="flash_attention_3")
    assert witness["schema"] == "hf-reference-execution-v1"
    assert witness["requested_attention"] == "flash_attention_3"
    assert witness["effective_configurations"] == {
        "model": {
            "config": {"model_type": "synthetic", "num_hidden_layers": 4, "use_cache": False},
            "attention_implementation": "sdpa",
        },
        "decoder": {
            "config": {"model_type": "synthetic", "num_hidden_layers": 4, "use_cache": False},
            "attention_implementation": "eager",
        },
    }
    assert (
        witness["effective_configurations_sha256"]
        == hashlib.sha256(
            json.dumps(witness["effective_configurations"], sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
    )
    for name, cls in (("model", _ReferenceExecutionModel), ("decoder", _ReferenceExecutionDecoder)):
        implementation = witness["implementations"][name]
        assert implementation["class"]["module"] == cls.__module__
        assert implementation["class"]["qualname"] == cls.__qualname__
        assert implementation["forward"]["qualname"] == cls.forward.__qualname__
        assert implementation["class"]["source"] == records.file_identity(__file__)
        assert implementation["forward"]["source"] == records.file_identity(__file__)
        assert len(implementation["forward"]["python_code_sha256"]) == 64
    assert json.loads(json.dumps(witness, allow_nan=False)) == witness
    engine.model.config.values["num_hidden_layers"] = 99
    assert witness["effective_configurations"]["model"]["config"]["num_hidden_layers"] == 4


def test_reference_execution_witness_hashes_loaded_forward_separately_from_file(_reference_execution_engine):
    engine = _reference_execution_engine
    original = hf_single_gpu.execution_witness(engine, requested_attention="sdpa")
    engine.decoder.forward = MethodType(_ReferenceExecutionModel.forward, engine.decoder)
    changed = hf_single_gpu.execution_witness(engine, requested_attention="sdpa")
    assert changed["implementations"]["decoder"]["class"] == original["implementations"]["decoder"]["class"]
    assert (
        changed["implementations"]["decoder"]["forward"]["source"]
        == original["implementations"]["decoder"]["forward"]["source"]
    )
    assert (
        changed["implementations"]["decoder"]["forward"]["python_code_sha256"]
        != original["implementations"]["decoder"]["forward"]["python_code_sha256"]
    )


def test_reference_execution_witness_code_digest_is_checkout_independent(_reference_execution_engine, tmp_path):
    engine = _reference_execution_engine
    original = hf_single_gpu.execution_witness(engine, requested_attention="sdpa")
    source = tmp_path / "relocated-reference.py"
    source.write_bytes(Path(__file__).read_bytes())
    forward = _ReferenceExecutionDecoder.forward
    relocated = FunctionType(forward.__code__.replace(co_filename=str(source)), forward.__globals__, forward.__name__)
    engine.decoder.forward = MethodType(relocated, engine.decoder)
    changed = hf_single_gpu.execution_witness(engine, requested_attention="sdpa")
    assert (
        changed["implementations"]["decoder"]["forward"]["source"]
        == original["implementations"]["decoder"]["forward"]["source"]
    )
    assert (
        changed["implementations"]["decoder"]["forward"]["python_code_sha256"]
        == original["implementations"]["decoder"]["forward"]["python_code_sha256"]
    )


def test_reference_execution_witness_preserves_upstream_config_and_effective_attention(_reference_execution_engine):
    from transformers import PretrainedConfig

    config = PretrainedConfig(dtype=torch.bfloat16, use_cache=False)
    config._attn_implementation = "eager"
    _reference_execution_engine.model.config = config
    _reference_execution_engine.decoder.config = config
    witness = hf_single_gpu.execution_witness(_reference_execution_engine, requested_attention="sdpa")
    for value in witness["effective_configurations"].values():
        assert value["config"] == json.loads(config.to_json_string(use_diff=False))
        assert value["attention_implementation"] == "eager"
        assert json.loads(json.dumps(value, allow_nan=False)) == value


@pytest.mark.parametrize("defect", ["native-forward", "nonfinite-config"])
def test_reference_execution_witness_rejects_unrecordable_runtime(_reference_execution_engine, defect):
    engine = _reference_execution_engine
    if defect == "native-forward":
        engine.decoder.forward = len
        message = "inspectable Python forward"
    else:
        engine.model.config.values["invalid"] = float("nan")
        message = "Out of range float"
    with pytest.raises(ValueError, match=message):
        hf_single_gpu.execution_witness(engine, requested_attention="sdpa")


@pytest.mark.parametrize("change", [None, "config", "forward", "source"])
def test_reference_execution_witness_survives_run_and_rejects_execution_changes(
    _reference_execution_engine, monkeypatch, tmp_path, change
):
    from arctic_platform.correctness.harness.participation import PARTICIPATION_SCHEMA

    engine = _reference_execution_engine
    if change == "source":
        source = tmp_path / "changing-forward.py"
        source.write_bytes(Path(__file__).read_bytes())
        forward = _ReferenceExecutionDecoder.forward
        relocated = FunctionType(
            forward.__code__.replace(co_filename=str(source)), forward.__globals__, forward.__name__
        )
        engine.decoder.forward = MethodType(relocated, engine.decoder)
    monkeypatch.setattr(hf_single_gpu, "build_engine", lambda *args, **kwargs: engine)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def forward(*args, **kwargs):
        if change == "config":
            engine.model.config.values["num_hidden_layers"] = 8
        elif change == "forward":
            engine.decoder.forward = MethodType(_ReferenceExecutionModel.forward, engine.decoder)
        elif change == "source":
            source.write_bytes(source.read_bytes() + b"\n# changed after reference start\n")
        return hf_single_gpu.ForwardBackwardResult(
            loss=1.25,
            gradients={},
            active_tokens=8,
            microbatches=1,
            skipped_no_grad=["indexer.weight"],
            participation={
                "schema": PARTICIPATION_SCHEMA,
                "parameters": {
                    "indexer.weight": {
                        "shape": [1],
                        "state": "inactive",
                        "evidence": {
                            "kind": "integer-selector",
                            "module": "indexer",
                            "module_class": "Glm5NextTextIndexer",
                            "calls": 1,
                            "integer_outputs_only": True,
                            "forward_sha256": "0" * 64,
                        },
                    }
                },
            },
        )

    monkeypatch.setattr(hf_single_gpu, "forward_backward", forward)
    if change is not None:
        with pytest.raises(ValueError, match="changed during execution"):
            hf_single_gpu.run("unused", None, device="cpu", attn_implementation="sdpa")
        return
    result = hf_single_gpu.run("unused", None, device="cpu", attn_implementation="sdpa")
    assert result.reference_execution == hf_single_gpu.execution_witness(engine, requested_attention="sdpa")
    assert result.skipped_no_grad == ["indexer.weight"]


def test_reference_execution_witness_cli_and_compact_record_preserve_provenance(
    _reference_execution_engine, _reference_records_ctx, monkeypatch, tmp_path
):
    from arctic_platform.correctness.harness import batches
    from arctic_platform.correctness.reference import __main__ as reference_cli

    witness = hf_single_gpu.execution_witness(_reference_execution_engine, requested_attention="sdpa")
    result = hf_single_gpu.ReferenceResult(
        **_reference_records_measurements(),
        peak_gib=0.0,
        skipped_no_grad=["indexer.weight"],
        reference_execution=witness,
    )
    output = tmp_path / "reference.json"
    monkeypatch.setattr(
        sys, "argv", ["reference", "--model", "unused", "--batch", "unused", "--out", str(output), "--seed", "42"]
    )
    monkeypatch.setattr(batches, "load", lambda *args: None)
    monkeypatch.setattr(hf_single_gpu, "run", lambda *args, **kwargs: result)
    reference_cli.main()
    measurements = json.loads(output.read_text())
    assert measurements["reference_execution"] == witness
    assert measurements["skipped_no_grad"] == ["indexer.weight"]
    identity = {"check_id": _reference_records_GRADS, "config_id": _reference_records_ctx.config_id}
    path = records.publish_record(_reference_records_ctx, identity, measurements)
    original = path.read_bytes()
    _reference_records_no_hf(monkeypatch)
    assert records.load_record(_reference_records_ctx, identity) == measurements
    assert path.read_bytes() == original
