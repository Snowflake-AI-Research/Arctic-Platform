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

"""Runner sequencing controls; saved tensor bytes have separate SAVE0 helper controls."""

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.common.peft import normalize_lora_peft_config
from arctic_platform.correctness.harness import cortex_fixture
from arctic_platform.correctness.harness import cortex_optimizer_capture
from arctic_platform.correctness.harness import deterministic_lora
from arctic_platform.correctness.harness import dss_driver
from arctic_platform.correctness.harness import hosted
from arctic_platform.correctness.harness import lora_init
from arctic_platform.correctness.harness import lora_save0
from arctic_platform.correctness.harness import reference_records
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.selftest.check_prepared_reference_runner import STEP_CHECKS
from arctic_platform.correctness.selftest.check_prepared_reference_runner import prepared_runner  # noqa: F401


@pytest.fixture
def save0_runner(prepared_runner, tmp_path, monkeypatch):  # noqa: F811
    control = prepared_runner
    lora_cfg = load_config(Path("arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-lora-4gpus-64k.config"))
    # Existing prepared-record closures capture the original config object.
    control.cfg.__dict__.update(lora_cfg.__dict__)
    control.spec.config_id = control.cfg.config_id
    control.spec.model.source_checkpoint = control.cfg.model_name
    control.spec.model.num_hidden_layers = 4
    control.backend = CortexConfig(base_url="http://127.0.0.1:8003", database="dev", schema="dev", host=None, pat=None)
    control.payloads = []
    control.proofs = []
    control.failure = None
    control.adapter_path = tmp_path / "adapter"
    control.metadata = {
        "model_name": control.cfg.model_name,
        "peft_config": normalize_lora_peft_config(control.cfg.training["peft_config"]),
    }

    def set_manifest(pairs):
        rank = control.metadata["peft_config"]["r"]
        specs = []
        for index in range(pairs):
            prefix = f"base_model.model.model.layers.{index}.self_attn.q_proj"
            specs.extend(
                [
                    lora_init.LoraTensorSpec(f"{prefix}.lora_A.weight", (rank, 3 + index), "float32"),
                    lora_init.LoraTensorSpec(f"{prefix}.lora_B.weight", (5 + index, rank), "float32"),
                ]
            )
        control.metadata["initialization"] = lora_init.build_manifest(specs, runner.SEED)

    set_manifest(1)
    control.set_manifest = set_manifest

    def materialize(cfg, model, workdir):
        assert cfg.config_id == control.cfg.config_id
        assert model == control.spec.model.cache_path
        assert workdir == tmp_path / "run" / "reference-adapter-init"
        return control.adapter_path

    def identity(path):
        assert path == control.adapter_path
        return copy.deepcopy(control.metadata)

    def fixture(backend, spec):
        assert backend is control.backend and spec is control.spec
        return {"model_name": control.spec.model.cache_path}

    @contextmanager
    def gateway(workdir, gpus, *, backend):
        assert workdir == tmp_path / "run"
        assert gpus == 4 and backend is control.backend
        control.events.append(("gateway-enter",))
        try:
            yield SimpleNamespace(backend=backend)
        finally:
            control.events.append(("gateway-exit",))

    @contextmanager
    def running_job(session, payload):
        assert session.backend is control.backend
        assert payload["model_name"] == control.spec.model.cache_path
        assert payload["training_config"]["peft_config"]["deterministic_init"] == {
            "recipe_version": control.metadata["initialization"]["recipe_version"],
            "global_seed": runner.SEED,
            "dtype": "float32",
        }
        control.payloads.append(copy.deepcopy(payload))
        control.events.append(("initialized",))
        try:
            yield SimpleNamespace(client=object())
        finally:
            control.events.append(("closed",))

    root = tmp_path / "saved"
    response = {
        "checkpoint_path": str(root),
        "checkpoint_tag": "global_step0",
        "version": 0,
        "stage_path": None,
    }

    def save(client, kind):
        assert client is not None and kind == "weights-only"
        control.events.append(("save0",))
        if control.failure == "save":
            raise RuntimeError("public SAVE0 failed")
        return root, response

    def verify(saved_root, saved_response, metadata):
        assert saved_root == root and saved_response == response
        assert metadata == control.metadata
        control.events.append(("proof",))
        if control.failure == "proof":
            raise ValueError("saved tensor identity differs")
        proof = {
            "identity_sha256": metadata["initialization"]["identity_sha256"],
            "verified_tensors": [
                {key: entry[key] for key in ("name", "dtype", "sha256")}
                for entry in metadata["initialization"]["tensors"]
            ],
            **response,
            "proof_scope": "exported-adapter-pack-not-all-rank-equality",
        }
        control.proofs.append(copy.deepcopy(proof))
        return proof

    def target(session, job, body, *, learning_rate, **kwargs):
        assert session.backend is control.backend and job.client is not None
        rows = body["batch"]["input_ids"].shape[0]
        arm = next(arm for arm in control.arms if arm.global_batch_size == rows)
        proof_path = tmp_path / "run" / "lora-save0" / f"{arm.name}.json"
        assert json.loads(proof_path.read_text()) == control.proofs[-1]
        assert control.events[-1] == ("proof",)
        if kwargs:
            assert learning_rate == runner.OPTIMIZER_LEARNING_RATE
            assert kwargs["local_checkpoint"] is True
            assert kwargs["expected_dp_world_size"] == 4
            assert kwargs["output_dir"] == tmp_path / "run" / "optimizer" / "dss" / arm.name
        else:
            assert learning_rate == 0.0
        control.events.append(("target", arm.name))
        return SimpleNamespace(
            optimizer_state_manifest=str(control.target_manifest),
            avg_loss=2.0,
            grad_norms={"module.layer.weight": 1.0},
            model_calls=None,
        )

    def execute(*, selected=STEP_CHECKS, selected_arms=None, **overrides):
        options = {
            "vocab_size": 128,
            "select": selected,
            "workdir": tmp_path / "run",
            "reference_records_dir": tmp_path / "records",
            "client_backend": control.backend,
            "local_cortex_fixture": True,
            "local_gateway_lora_save0": True,
        }
        options.update(overrides)
        return runner.execute(
            control.cfg,
            control.spec,
            control.arms if selected_arms is None else selected_arms,
            **options,
        )

    monkeypatch.setattr(deterministic_lora, "materialize_adapter", materialize)
    monkeypatch.setattr(deterministic_lora, "adapter_identity", identity)
    monkeypatch.setattr(cortex_fixture, "load_local_fixture", fixture)
    monkeypatch.setattr(runner, "gateway", gateway)
    monkeypatch.setattr(dss_driver, "running_job", running_job)
    monkeypatch.setattr(hosted, "_save_local_checkpoint", save)
    monkeypatch.setattr(lora_save0, "verify_saved_initialization", verify)
    monkeypatch.setattr(runner, "running_job", lambda *a, **k: pytest.fail("must enter verified SAVE0 wrapper"))
    monkeypatch.setattr(runner, "fwd_bwd_step", target)
    monkeypatch.setattr(cortex_optimizer_capture, "capture_optimizer_step", target)
    control.execute = execute
    return control


@pytest.mark.parametrize("order", ["batched", "arm"])
@pytest.mark.parametrize("selected", [STEP_CHECKS[:1], STEP_CHECKS[1:], STEP_CHECKS])
@pytest.mark.parametrize("pairs", [1, 2])
def test_cached_records_then_save0_proof_before_each_measurement(save0_runner, order, selected, pairs, tmp_path):
    control = save0_runner
    control.set_manifest(pairs)
    before = copy.deepcopy((control.cfg.sub_job, control.metadata, control.record))

    results = control.execute(order=order, selected=selected)

    assert [(result.test_id, result.arm, result.outcome, result.gpus) for result in results] == [
        (check, arm.name, Outcome.PASS, 4) for arm in control.arms for check in selected
    ]
    records = [("record", check, arm.name) for arm in control.arms for check in selected]
    assert control.events[: len(records)] == records
    assert [event for event in control.events if event[0] not in {"record", "gateway-enter", "gateway-exit"}] == [
        event
        for arm in control.arms
        for event in [("initialized",), ("save0",), ("proof",), ("target", arm.name), ("closed",)]
    ]
    assert (
        control.events.count(("gateway-enter",))
        == control.events.count(("gateway-exit",))
        == (1 if order == "batched" else 2)
    )
    assert (control.cfg.sub_job, control.metadata, control.record) == before
    assert not control.published
    assert len(control.payloads) == len(control.arms)
    for _, _, recipe in control.recipes:
        assert recipe["initialization"] == control.metadata
        assert "deterministic_init" not in recipe["training"]["peft_config"]
    for arm in control.arms:
        proof = json.loads((tmp_path / "run" / "lora-save0" / f"{arm.name}.json").read_text())
        assert proof["identity_sha256"] == control.metadata["initialization"]["identity_sha256"]
        assert len(proof["verified_tensors"]) == len(control.metadata["initialization"]["tensors"]) == 2 * pairs
    for result in results:
        if result.test_id == "single-step-optimizer":
            assert result.metrics["optimizer_values_compared"] == 3
            assert result.metrics["stated_criterion_abs"] == 1e-3


def test_default_lora_cortex_rejection_is_unchanged(save0_runner):
    control = save0_runner
    results = control.execute(local_gateway_lora_save0=False)
    assert len(results) == 2
    assert all(result.outcome is Outcome.FAIL for result in results)
    assert all("product-initialized LoRA adapter" in result.reason for result in results)
    assert not control.events and not control.payloads and not control.published


def test_opt_in_does_not_bypass_fixture_capability_rejection(save0_runner, monkeypatch):
    control = save0_runner
    monkeypatch.setattr(cortex_fixture, "load_local_fixture", lambda *a: None)
    results = control.execute()
    assert len(results) == 2
    assert all(result.outcome is Outcome.FAIL and "published" in result.reason for result in results)
    assert not control.events and not control.payloads and not control.published


@pytest.mark.parametrize("order", ["batched", "arm"])
def test_missing_cached_record_stops_before_allocation_or_save0(save0_runner, order):
    control = save0_runner
    control.missing.add(("single-step-optimizer", "gas4"))
    with pytest.raises(reference_records.ReferencePrerequisiteError, match="single-step-optimizer/gas4"):
        control.execute(order=order)
    assert control.events == [("record", check, arm.name) for arm in control.arms for check in STEP_CHECKS]
    assert not control.payloads and not control.proofs and not control.published


@pytest.mark.parametrize("failure", ["save", "proof"])
def test_save0_failure_closes_each_job_and_never_measures_or_passes(save0_runner, failure, tmp_path):
    control = save0_runner
    control.failure = failure
    results = control.execute()
    assert len(results) == 4
    assert all(result.outcome is Outcome.FAIL and "execution error:" in result.summary for result in results)
    assert control.events.count(("initialized",)) == control.events.count(("closed",)) == 2
    assert not any(event[0] == "target" for event in control.events)
    assert not control.proofs and not control.published
    assert not (tmp_path / "run" / "lora-save0").exists()


@pytest.mark.parametrize(
    "failure",
    [
        "width",
        "config",
        "spec",
        "model",
        "source",
        "layers",
        "sp",
        "ep",
        "rl",
        "peft",
        "onprem",
        "remote",
        "auth",
        "fullmodel",
        "prepare",
        "records",
        "selection",
        "empty",
        "e2e",
        "publication",
        "fixture",
    ],
)
def test_opt_in_refuses_outside_reviewed_scope_before_any_work(save0_runner, failure):
    control = save0_runner
    overrides = {}
    if failure == "width":
        control.cfg.native = control.cfg.native.model_copy(update={"training_gpus": 8})
        control.cfg.training["n_gpus"] = 8
    elif failure == "config":
        control.cfg.config_id = "another-config"
    elif failure == "spec":
        control.spec.config_id = "another-config"
    elif failure == "model":
        control.cfg.sub_job["model_name"] = "another-model"
    elif failure == "source":
        control.spec.model.source_checkpoint = "another-model"
    elif failure == "layers":
        control.spec.model.num_hidden_layers = 32
    elif failure in {"sp", "ep"}:
        control.cfg.training[f"{failure}_size"] = 2
    elif failure == "rl":
        control.cfg.native = control.cfg.native.model_copy(update={"sampling_gpus": 4})
    elif failure == "peft":
        control.cfg.training.pop("peft_config")
    elif failure == "onprem":
        overrides["client_backend"] = OnPremConfig()
    elif failure in {"remote", "auth"}:
        overrides["client_backend"] = CortexConfig(
            base_url="https://remote.invalid" if failure == "remote" else "http://127.0.0.1:8003",
            database="dev",
            schema="dev",
            host=None,
            pat="not-a-real-token" if failure == "auth" else None,
        )
    elif failure == "fullmodel":
        overrides["local_gateway_full_model"] = True
    elif failure == "prepare":
        overrides["prepare_references"] = True
    elif failure == "records":
        overrides["reference_records_dir"] = None
    elif failure in {"selection", "empty", "e2e"}:
        overrides["selected"] = {
            "selection": None,
            "empty": [],
            "e2e": ["e2e-sft-train-validate"],
        }[failure]
    elif failure == "publication":
        overrides["cortex_fixture"] = Path("unread-publication.json")
    elif failure == "fixture":
        overrides["local_cortex_fixture"] = False
    with pytest.raises(ValueError):
        control.execute(**overrides)
    assert not control.events and not control.payloads and not control.published


@pytest.mark.parametrize("failure", ["missing-adapter", "topology", "zero3", "sp-alias", "ep-alias"])
def test_metadata_or_binding_failure_stops_before_records_and_gateway(save0_runner, monkeypatch, failure):
    control = save0_runner
    if failure == "missing-adapter":
        monkeypatch.setattr(deterministic_lora, "materialize_adapter", lambda *a: None)
    elif failure == "topology":
        control.metadata["peft_config"]["r"] += 1
    elif failure == "zero3":
        control.cfg.training["ds_config"]["zero_optimization"]["stage"] = 3
    else:
        key = "sequence_parallel_size" if failure == "sp-alias" else "expert_parallel_size"
        control.cfg.training[key] = 2
    with pytest.raises(ValueError):
        control.execute()
    assert not control.events and not control.payloads and not control.published


def test_default_onprem_binding_still_fails_closed(save0_runner):
    control = save0_runner
    with pytest.raises(deterministic_lora.LoraInitializationPrerequisiteError, match="no supported public"):
        control.execute(client_backend=OnPremConfig(), local_cortex_fixture=False, local_gateway_lora_save0=False)
    assert not control.events and not control.payloads and not control.published
