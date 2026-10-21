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

"""Static correctness selftests: optimizer capture."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import hashlib
import json
import math
import os
import shutil
import sys
import types
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import aiohttp
import pytest
import requests
import torch
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.utils.tensor_fragment import fragment_address
from safetensors.torch import save_file

from arctic_platform import wire
from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.client.transports.cortex import _DEFAULT
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.client.transports.cortex import _SendPolicy
from arctic_platform.correctness.checks.optimizer_state import compare_optimizer_artifacts
from arctic_platform.correctness.checks.optimizer_state import summarize_optimizer_artifacts
from arctic_platform.correctness.harness import cortex as cortex_fixture
from arctic_platform.correctness.harness import cortex_optimizer as capture
from arctic_platform.correctness.harness import cortex_optimizer as decoder
from arctic_platform.correctness.harness import execution as cli
from arctic_platform.correctness.harness import optimizer_capture
from arctic_platform.correctness.harness import reference_records
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness.batches import Batch
from arctic_platform.correctness.harness.config import ArmDefinition
from arctic_platform.correctness.harness.config import TestSpec as Spec
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.cortex import CorrectnessClient
from arctic_platform.correctness.harness.dss_driver import ArcticJob
from arctic_platform.correctness.harness.dss_driver import GatewaySession
from arctic_platform.correctness.harness.dss_driver import StepResult
from arctic_platform.correctness.harness.dss_driver import _gateway_runtime
from arctic_platform.correctness.harness.dss_driver import _globalize_expert_norms
from arctic_platform.correctness.harness.dss_driver import _labels_for_provider
from arctic_platform.correctness.harness.dss_driver import _training_client_config
from arctic_platform.correctness.harness.dss_driver import available_gpus
from arctic_platform.correctness.harness.dss_driver import build_payload
from arctic_platform.correctness.harness.dss_driver import copy_checkpoint_peft_adapter
from arctic_platform.correctness.harness.dss_driver import pack
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.transport import pack_microbatches

# Cortex optimizer capture.


_cortex_optimizer_capture_ROOT = Path(__file__).resolve().parents[1]

_cortex_optimizer_capture_CONFIG = (
    _cortex_optimizer_capture_ROOT / "configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config"
)

_cortex_optimizer_capture_SPEC = _cortex_optimizer_capture_ROOT / "specs/qwen3-8b-h200-train-sft-full-4gpus-64k.json"

_cortex_optimizer_capture_OPTIMIZER = {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.0}


@pytest.fixture(autouse=False)
def _cortex_optimizer_capture_no_network(monkeypatch):
    monkeypatch.setattr(
        requests.Session, "request", lambda *args, **kwargs: pytest.fail("mock tests must not access network")
    )


def _cortex_optimizer_capture_manifest(root, *, step=1):
    root.mkdir(parents=True)
    filename = "000000.pt"
    torch.save({key: torch.zeros(2) for key in ("parameter_update", "exp_avg", "exp_avg_sq")}, root / filename)
    path = root / "manifest.json"
    path.write_text(json.dumps({"step": step, "parameters": [{"name": "weight", "file": filename}]}))
    return path


def _cortex_optimizer_capture_prepared_references(monkeypatch, spec, cfg, tmp_path):
    references = []
    model = spec.test_settings["single-step-optimizer"]["model_cache_path"]
    compact = {
        "loss": 2.0,
        "grad_norms": {"weight": 1.0},
        "microbatches": 1,
        "optimizer_norms": summarize_optimizer_artifacts(
            _cortex_optimizer_capture_manifest(tmp_path / "prepared-reference"),
            optimizer_config=cfg.training["optimizer"],
            learning_rate=runner.OPTIMIZER_LEARNING_RATE,
        ),
        "optimizer_norm_gate": {
            "criterion": "optimizer-state-l2-norm-v1",
            "absolute": 0.001,
            "status": "reviewed",
            "provenance": "selftest reviewed prepared reference",
        },
    }

    def model_identity(path):
        assert path == model == spec.model.cache_path
        return {"sha256": "mock-reviewed-model"}

    def workload_identity(ctx, test_id, recipe):
        assert recipe["model"] == {"sha256": "mock-reviewed-model"}
        assert recipe["token_budget"] == 10240
        assert recipe["optimizer"]["learning_rate"] == runner.OPTIMIZER_LEARNING_RATE == 0.01
        assert recipe["optimizer"]["config"] == cfg.training["optimizer"]
        return {
            "check_id": test_id,
            "arm": f"gas{recipe['synthetic_inputs']['rows'] // cfg.n_gpus}",
            "active_tokens": recipe["synthetic_inputs"]["batch"]["active_tokens"],
        }

    def load_record(ctx, identity):
        key = (identity["check_id"], identity["arm"])
        references.append(key)
        ctx.reference_record_paths["/".join(key)] = str(tmp_path / f"{key[0]}-{key[1]}.json")
        return {**compact, "active_tokens": identity["active_tokens"]}

    monkeypatch.setattr(reference_records, "model_identity", model_identity)
    monkeypatch.setattr(reference_records, "workload_identity", workload_identity)
    monkeypatch.setattr(reference_records, "load_record", load_record)
    monkeypatch.setattr(runner, "run_reference", lambda *a, **k: pytest.fail("regression must not run HF"))
    return references


@pytest.fixture
def _cortex_optimizer_capture_artifact_stubs(monkeypatch):
    export = ModuleType("arctic_platform.correctness.harness.inference_export")
    decode = ModuleType("arctic_platform.correctness.harness.cortex_optimizer")
    monkeypatch.setitem(sys.modules, export.__name__, export)
    monkeypatch.setattr(
        capture, "decode_optimizer_capture", lambda *args, **kwargs: decode.decode_optimizer_capture(*args, **kwargs)
    )
    return (export, decode)


@pytest.fixture
def _cortex_optimizer_capture_lifecycle(tmp_path, _cortex_optimizer_capture_artifact_stubs):
    events = []
    snapshots = []
    state = SimpleNamespace(failure=None, step=0, saves=0, manifest_step=1, duplicate=False)
    backend = CortexConfig(base_url="http://unused")
    transport = SimpleNamespace(job_id="parent-job", list_checkpoints=lambda: [])

    def event(name):
        events.append(name)
        if state.failure == name:
            raise RuntimeError(f"injected {name} failure")

    class Client:
        config = SimpleNamespace(backend=backend)
        jobs = SimpleNamespace(training="opaque-child-id")

        def save_checkpoint(self, **kwargs):
            assert kwargs == {"checkpoint_type": "resumable"}
            event(f"save-{state.step}")
            state.saves += 1
            cp = "cp_bef0" if state.duplicate or state.step == 0 else "cp_af71"
            return {"checkpoint_id": "global_step", "stage_path": f"@stage/checkpoints/{cp}/"}

        def fwd_bwd(self, body):
            assert body == {"same": "batch"}
            event("fwd_bwd")
            return {"avg_loss": 2.0, "metrics": {"sft_model_calls": 3, "sft_packed_rows": 4}}

        def step(self, learning_rate):
            assert learning_rate == runner.OPTIMIZER_LEARNING_RATE == 0.01
            event("step")
            state.step += 1
            return {"gradient_norms_per_param": {"weight": 1.0}}

    client = Client()
    client.transport = transport
    job = ArcticJob(client, "training")
    session = GatewaySession(tmp_path, 4, backend)

    def export(client_arg, job_id, cp_id, parent, *, expected_dp_world_size):
        assert client_arg is client and job_id == "parent-job"
        assert expected_dp_world_size == 4
        root = parent / "snapshot"
        root.mkdir(parents=True)
        (root / "partial").write_bytes(b"artifact")
        snapshots.append(root)
        event(f"export-{cp_id}")
        return root

    def decode(before, after, output, **kwargs):
        assert snapshots == [before, after]
        assert kwargs == {
            "expected_optimizer_config": _cortex_optimizer_capture_OPTIMIZER,
            "expected_learning_rate": 0.01,
        }
        path = _cortex_optimizer_capture_manifest(output, step=state.manifest_step)
        event("decode")
        return path

    _cortex_optimizer_capture_artifact_stubs[0].export_checkpoint = export
    _cortex_optimizer_capture_artifact_stubs[1].decode_optimizer_capture = decode
    return SimpleNamespace(
        events=events, state=state, session=session, job=job, snapshots=snapshots, output=tmp_path / "output"
    )


def _cortex_optimizer_capture_run(lifecycle):
    return capture.capture_optimizer_step(
        lifecycle.session,
        lifecycle.job,
        {"same": "batch"},
        learning_rate=0.01,
        output_dir=lifecycle.output,
        expected_dp_world_size=4,
        optimizer_config=_cortex_optimizer_capture_OPTIMIZER,
    )


@pytest.mark.usefixtures("_cortex_optimizer_capture_no_network")
@pytest.mark.parametrize("unsafe", ["/tmp", "/code", "/data-fast-shadow", "/data-fast", "symlink"])
def test_cortex_optimizer_capture_capture_requires_data_fast_before_mkdir_or_checkpoint(
    _cortex_optimizer_capture_lifecycle, tmp_path, monkeypatch, unsafe
):
    destination = Path(unsafe)
    if unsafe == "symlink":
        destination = tmp_path / "escape"
        destination.symlink_to("/tmp", target_is_directory=True)
        destination /= "output"
    _cortex_optimizer_capture_lifecycle.output = destination
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: pytest.fail("unsafe output reached mkdir"))
    with pytest.raises(ValueError, match="/data-fast"):
        _cortex_optimizer_capture_run(_cortex_optimizer_capture_lifecycle)
    assert _cortex_optimizer_capture_lifecycle.events == []
    assert _cortex_optimizer_capture_lifecycle.state.saves == _cortex_optimizer_capture_lifecycle.state.step == 0


@pytest.mark.usefixtures("_cortex_optimizer_capture_no_network")
def test_cortex_optimizer_capture_same_job_snapshots_bracket_exactly_one_frozen_lr_step(
    _cortex_optimizer_capture_lifecycle,
):
    result = _cortex_optimizer_capture_run(_cortex_optimizer_capture_lifecycle)
    assert _cortex_optimizer_capture_lifecycle.events == [
        "save-0",
        "fwd_bwd",
        "step",
        "save-1",
        "export-cp_bef0",
        "export-cp_af71",
        "decode",
    ]
    assert _cortex_optimizer_capture_lifecycle.state.step == 1
    assert _cortex_optimizer_capture_lifecycle.state.saves == 2
    assert result.avg_loss == 2.0 and result.grad_norms == {"weight": 1.0}
    assert result.model_calls == 3 and result.packed_rows == 4
    assert (
        Path(result.optimizer_state_manifest)
        == _cortex_optimizer_capture_lifecycle.output / "states" / "manifest.json"
    )
    assert Path(result.optimizer_state_manifest).is_file()
    assert not list(_cortex_optimizer_capture_lifecycle.output.glob("snapshots-*"))
    assert all((not snapshot.exists() for snapshot in _cortex_optimizer_capture_lifecycle.snapshots))
    assert _cortex_optimizer_capture_lifecycle.job.client.transport.job_id == "parent-job"


@pytest.mark.usefixtures("_cortex_optimizer_capture_no_network")
@pytest.mark.parametrize(
    "failure", ["save-0", "fwd_bwd", "step", "save-1", "export-cp_bef0", "export-cp_af71", "decode"]
)
def test_cortex_optimizer_capture_failures_remove_downloads_and_never_publish_partial_manifest(
    _cortex_optimizer_capture_lifecycle, failure
):
    _cortex_optimizer_capture_lifecycle.state.failure = failure
    with pytest.raises(RuntimeError, match="injected"):
        _cortex_optimizer_capture_run(_cortex_optimizer_capture_lifecycle)
    assert not list(_cortex_optimizer_capture_lifecycle.output.iterdir())
    assert all((not snapshot.exists() for snapshot in _cortex_optimizer_capture_lifecycle.snapshots))


@pytest.mark.usefixtures("_cortex_optimizer_capture_no_network")
@pytest.mark.parametrize("step", [0, 2, True, "1", None])
def test_cortex_optimizer_capture_decoder_cannot_relabel_non_initial_update_as_step_one(
    _cortex_optimizer_capture_lifecycle, step
):
    _cortex_optimizer_capture_lifecycle.state.manifest_step = step
    with pytest.raises(ValueError, match="exactly step 1"):
        _cortex_optimizer_capture_run(_cortex_optimizer_capture_lifecycle)
    assert not list(_cortex_optimizer_capture_lifecycle.output.iterdir())


@pytest.mark.usefixtures("_cortex_optimizer_capture_no_network")
def test_cortex_optimizer_capture_duplicate_durable_checkpoints_rejected_before_export(
    _cortex_optimizer_capture_lifecycle,
):
    _cortex_optimizer_capture_lifecycle.state.duplicate = True
    with pytest.raises(ValueError, match="distinct durable"):
        _cortex_optimizer_capture_run(_cortex_optimizer_capture_lifecycle)
    assert _cortex_optimizer_capture_lifecycle.events == ["save-0", "fwd_bwd", "step", "save-1"]
    assert not list(_cortex_optimizer_capture_lifecycle.output.iterdir())


@pytest.mark.usefixtures("_cortex_optimizer_capture_no_network")
def test_cortex_optimizer_capture_existing_output_never_overwritten_or_stepped(_cortex_optimizer_capture_lifecycle):
    existing = _cortex_optimizer_capture_manifest(_cortex_optimizer_capture_lifecycle.output / "states")
    original = existing.read_bytes()
    with pytest.raises(FileExistsError):
        _cortex_optimizer_capture_run(_cortex_optimizer_capture_lifecycle)
    assert existing.read_bytes() == original
    assert _cortex_optimizer_capture_lifecycle.events == []


@pytest.mark.usefixtures("_cortex_optimizer_capture_no_network")
def test_cortex_optimizer_capture_decoder_must_return_manifest_inside_its_owned_output(
    _cortex_optimizer_capture_lifecycle, _cortex_optimizer_capture_artifact_stubs
):
    outside = _cortex_optimizer_capture_manifest(_cortex_optimizer_capture_lifecycle.output / "unrelated")
    _cortex_optimizer_capture_artifact_stubs[1].decode_optimizer_capture = lambda *args, **kwargs: outside
    with pytest.raises(ValueError, match="must return its output manifest"):
        _cortex_optimizer_capture_run(_cortex_optimizer_capture_lifecycle)
    assert outside.is_file()
    assert not list(_cortex_optimizer_capture_lifecycle.output.glob("snapshots-*"))
    assert not (_cortex_optimizer_capture_lifecycle.output / "states").exists()


@pytest.mark.usefixtures("_cortex_optimizer_capture_no_network")
def test_cortex_optimizer_capture_uninitialized_parent_job_rejected_before_save(_cortex_optimizer_capture_lifecycle):
    _cortex_optimizer_capture_lifecycle.job.client.transport.job_id = None
    with pytest.raises(ValueError, match="initialized parent job"):
        _cortex_optimizer_capture_run(_cortex_optimizer_capture_lifecycle)
    assert _cortex_optimizer_capture_lifecycle.events == []


@pytest.mark.usefixtures("_cortex_optimizer_capture_no_network")
@pytest.mark.parametrize("degree", [0, -1, True, 1.0])
def test_cortex_optimizer_capture_invalid_dp_degree_rejected_before_save(_cortex_optimizer_capture_lifecycle, degree):
    with pytest.raises(ValueError, match="positive integer"):
        capture.capture_optimizer_step(
            _cortex_optimizer_capture_lifecycle.session,
            _cortex_optimizer_capture_lifecycle.job,
            {},
            learning_rate=0.01,
            output_dir=_cortex_optimizer_capture_lifecycle.output,
            expected_dp_world_size=degree,
            optimizer_config=_cortex_optimizer_capture_OPTIMIZER,
        )
    assert _cortex_optimizer_capture_lifecycle.events == []


@pytest.mark.usefixtures("_cortex_optimizer_capture_no_network")
def test_cortex_optimizer_capture_onprem_cannot_use_managed_capture(_cortex_optimizer_capture_lifecycle):
    _cortex_optimizer_capture_lifecycle.session = GatewaySession(
        _cortex_optimizer_capture_lifecycle.output, 4, OnPremConfig()
    )
    with pytest.raises(ValueError, match="requires CortexConfig"):
        _cortex_optimizer_capture_run(_cortex_optimizer_capture_lifecycle)
    assert _cortex_optimizer_capture_lifecycle.events == []


@pytest.mark.usefixtures("_cortex_optimizer_capture_no_network")
@pytest.mark.parametrize("order", ["batched", "interleaved"])
@pytest.mark.parametrize("failure", [None, "save-0", "fwd_bwd", "step", "save-1", "export-0", "export-1", "decode"])
def test_cortex_optimizer_capture_runner_public_cortex_capture_keeps_fixture_budget_gradient_and_cleanup(
    tmp_path, monkeypatch, _cortex_optimizer_capture_artifact_stubs, capsys, order, failure
):
    cfg = load_config(_cortex_optimizer_capture_CONFIG)
    spec = Spec.read(_cortex_optimizer_capture_SPEC)
    original = (_cortex_optimizer_capture_CONFIG.read_bytes(), _cortex_optimizer_capture_SPEC.read_bytes())
    creates, cancels, calls, snapshots = ([], [], [], [])
    references = _cortex_optimizer_capture_prepared_references(monkeypatch, spec, cfg, tmp_path)
    transports, closed = ([], [])
    replies = {}
    counts = {}
    exports, decodes = _cortex_optimizer_capture_artifact_stubs
    cfg.training["debug"] = {"optimizer_state_output_dir": "/local-only-ray-hook"}
    original_close = CortexTransport._close_session

    async def close(transport):
        if transport._session is not None:
            closed.append((transport, transport._session))
        await original_close(transport)

    monkeypatch.setattr(CortexTransport, "_close_session", close)

    def record(job_id, action):
        calls.append((job_id, action))
        if action == failure:
            raise RuntimeError("injected capture failure")

    async def send(transport, method, url, **kwargs):
        if transport._session is None:
            transport._session = aiohttp.ClientSession(headers=transport._auth_headers())
        if method == "POST" and url == transport._prefix:
            transports.append(transport)
            creates.append(kwargs["json"])
            parent = f"job-{len(creates)}"
            counts[parent] = 0
            return {"job_id": parent}
        if method == "GET" and url == f"{transport._prefix}/{transport.job_id}":
            state = "cancelled" if transport.job_id in cancels else "running"
            return {
                "job_id": transport.job_id,
                "status": state,
                "sub_jobs": [
                    {"job_type": "training", "sub_job_id": f"{transport.job_id}:training:0", "status": state}
                ],
            }
        if method == "GET" and url.endswith("/checkpoints"):
            return {"checkpoints": []}
        if method == "GET":
            return {"status": "REQUEST_STATE_DONE", "result": replies[url.rsplit("/", 1)[-1]]}
        if url.endswith(":cancel"):
            cancels.append(transport.job_id)
            return {}
        if url.endswith("/save"):
            assert kwargs["json"] == {"checkpoint_type": "resumable", "export_hf": False}
            step = counts[transport.job_id]
            record(transport.job_id, f"save-{step}")
            result = {"stage_path": f"@stage/checkpoints/cp_{transport.job_id.rsplit('-', 1)[-1]}-{step}/"}
        elif url.endswith("/forward-backward"):
            body = wire.loads(wire.decode_byte_chunks([kwargs["data"]], kind="request"))
            assert "processing" not in body
            record(transport.job_id, "fwd_bwd")
            result = {"avg_loss": 2.0}
        elif url.endswith("/step"):
            assert kwargs["json"]["learning_rate"] == 0.01
            counts[transport.job_id] += 1
            record(transport.job_id, "step")
            result = {"gradient_norms_per_param": {"weight": 1.0}}
        else:
            pytest.fail(f"unexpected transport call: {method} {url}")
        request_id = f"request-{len(replies)}"
        replies[request_id] = result
        return {"request_id": request_id}

    def export(client, job_id, cp_id, output_dir, *, expected_dp_world_size):
        assert client.transport.job_id == job_id
        assert expected_dp_world_size == cfg.dp_size
        assert cp_id.startswith(f"cp_{job_id.rsplit('-', 1)[-1]}-")
        root = output_dir / "snapshot"
        root.mkdir(parents=True)
        snapshots.append(root)
        record(job_id, f"export-{cp_id.rsplit('-', 1)[-1]}")
        return root

    def decode(before, after, output_dir, **kwargs):
        assert before != after
        assert kwargs == {"expected_optimizer_config": cfg.training["optimizer"], "expected_learning_rate": 0.01}
        manifest = _cortex_optimizer_capture_manifest(output_dir)
        if failure == "decode":
            raise RuntimeError("injected capture failure")
        return manifest

    exports.export_checkpoint = export
    decodes.decode_optimizer_capture = decode
    monkeypatch.setattr(cortex_fixture, "load_fixture", lambda *args: {"model_name": "published-reviewed"})
    monkeypatch.setattr(runner, "uses_mixer_packing", lambda *args: False)
    monkeypatch.setattr(CortexTransport, "_send", send)

    def execute():
        return runner.execute(
            cfg,
            spec,
            [ArmDefinition("gas1", 4, 32), ArmDefinition("gas4", 16, 32)],
            vocab_size=128,
            select=["single-step-grads", "single-step-optimizer"],
            order=order,
            client_backend=CortexConfig(base_url="http://unused"),
            cortex_fixture=tmp_path / "receipt.json",
            workdir=tmp_path,
            fail_fast=failure is not None,
        )

    if failure is not None:
        results = execute()
        assert len(results) == 2
        assert {result.test_id for result in results} == {"single-step-grads", "single-step-optimizer"}
        assert all(
            (
                result.outcome is Outcome.FAIL
                and result.arm == "gas1"
                and (result.gpus == cfg.n_gpus)
                and (result.reason == "injected capture failure")
                and (result.summary == "execution error: RuntimeError: injected capture failure")
                for result in results
            )
        )
        actions = ["save-0", "fwd_bwd", "step", "save-1", "export-0", "export-1"]
        expected = actions if failure == "decode" else actions[: actions.index(failure) + 1]
        assert calls == [("job-1", action) for action in expected]
        traceback = capsys.readouterr().err
        assert "Traceback (most recent call last):" in traceback
        assert "capture_optimizer_step" in traceback
        assert "RuntimeError: injected capture failure" in traceback
        assert cancels == ["job-1"]
        assert [transport for transport, _ in closed] == transports
        assert all((session is not None and session.closed for _, session in closed))
        assert all((transport.job_id is None and transport._session is None for transport in transports))
        assert all((not snapshot.exists() for snapshot in snapshots))
        assert not list((tmp_path / "optimizer" / "dss" / "gas1").iterdir())
        assert references == [
            (check, arm) for arm in ("gas1", "gas4") for check in ("single-step-grads", "single-step-optimizer")
        ]
        assert (
            _cortex_optimizer_capture_CONFIG.read_bytes() == original[0]
            and _cortex_optimizer_capture_SPEC.read_bytes() == original[1]
        )
        return
    results = execute()
    assert len(results) == 4 and all((result.outcome is Outcome.PASS for result in results))
    assert references == [
        (check, arm) for arm in ("gas1", "gas4") for check in ("single-step-grads", "single-step-optimizer")
    ]
    assert len(creates) == len(cancels) == 2
    assert all(
        (
            result.metrics["stated_criterion_abs"] == 0.001
            for result in results
            if result.test_id == "single-step-optimizer"
        )
    )
    assert cancels == ["job-1", "job-2"]
    assert [transport for transport, _ in closed] == transports
    assert all((session is not None and session.closed for _, session in closed))
    assert all((transport.job_id is None and transport._session is None for transport in transports))
    assert counts == {"job-1": 1, "job-2": 1}
    for job_id in counts:
        assert [action for parent, action in calls if parent == job_id] == [
            "save-0",
            "fwd_bwd",
            "step",
            "save-1",
            "export-0",
            "export-1",
        ]
    for create in creates:
        assert create["hardware"] == "H200"
        subjob = create["sub_job_configs"][0]
        assert subjob["model_name"] == "published-reviewed"
        assert subjob["training_config"]["debug"]["gradient_norms_per_param"] is True
        assert "optimizer_state_output_dir" not in subjob["training_config"]["debug"]
    assert all((not snapshot.exists() for snapshot in snapshots))
    assert (_cortex_optimizer_capture_CONFIG.read_bytes(), _cortex_optimizer_capture_SPEC.read_bytes()) == original


@pytest.mark.usefixtures("_cortex_optimizer_capture_no_network")
@pytest.mark.parametrize("order", ["batched", "interleaved"])
def test_cortex_optimizer_capture_runner_onprem_preserves_local_optimizer_capture(tmp_path, monkeypatch, order):
    cfg = load_config(_cortex_optimizer_capture_CONFIG)
    spec = Spec.read(_cortex_optimizer_capture_SPEC)
    observed = []
    backend = OnPremConfig()
    references = _cortex_optimizer_capture_prepared_references(monkeypatch, spec, cfg, tmp_path)

    @contextmanager
    def gateway(path, slots, **kwargs):
        assert kwargs["backend"] is backend
        yield "onprem-session"

    @contextmanager
    def running_job(session, payload):
        observed.append(payload)
        yield payload

    def step(session, payload, body, *, learning_rate):
        assert learning_rate == 0.01
        path = Path(payload["training_config"]["debug"]["optimizer_state_output_dir"]) / "step1"
        return StepResult(2.0, {}, optimizer_state_manifest=str(_cortex_optimizer_capture_manifest(path)))

    monkeypatch.setattr(runner, "uses_mixer_packing", lambda *args: False)
    monkeypatch.setattr(runner, "gateway", gateway)
    monkeypatch.setattr(runner, "running_job", running_job)
    monkeypatch.setattr(runner, "fwd_bwd_step", step)
    monkeypatch.setattr(capture, "capture_optimizer_step", lambda *args, **kwargs: pytest.fail("no managed capture"))
    results = runner.execute(
        cfg,
        spec,
        [ArmDefinition("gas1", 4, 32)],
        vocab_size=128,
        select=["single-step-optimizer"],
        client_backend=backend,
        workdir=tmp_path,
        order=order,
    )
    assert len(results) == 1 and results[0].outcome is Outcome.PASS
    assert references == [("single-step-optimizer", "gas1")]
    assert results[0].metrics["stated_criterion_abs"] == 0.001
    assert observed[0]["model_name"] == spec.model.cache_path
    assert "hardware" not in observed[0]


# Cortex optimizer decode.


_cortex_optimizer_decode_OPTIMIZER = {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.0}

_cortex_optimizer_decode_LR = 0.01


def _cortex_optimizer_decode_pair(tmp_path, *, world_size=2, before_step=0, prefix="bf16_", lazy_baseline=True):
    shapes = [
        OrderedDict([("layer.weight", torch.Size([2, 3])), ("layer.bias", torch.Size([3]))]),
        OrderedDict([("layer.norm.weight", torch.Size([5]))]),
    ]
    config = {
        "zero_optimization": {"stage": 2},
        "bf16": {"enabled": True},
        "optimizer": {"type": "AdamW", "params": {**_cortex_optimizer_decode_OPTIMIZER, "lr": 2e-05}},
    }
    baseline = {}
    expected = {}
    for group_index, group in enumerate(shapes):
        for name, shape in group.items():
            before = torch.full(shape, 1.0 + group_index, dtype=torch.float32)
            after = before + 0.0001
            baseline[name] = before
            expected[name] = {
                "parameter_update": after - before,
                "exp_avg": torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape) + 1,
                "exp_avg_sq": torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape) + 2,
            }
    before_root, after_root = (tmp_path / "before", tmp_path / "after")
    for root, step, post in ((before_root, before_step, False), (after_root, before_step + 1, True)):
        tag = root / f"global_step{step}"
        tag.mkdir(parents=True)
        (root / "metadata.json").write_text("{}")
        (root / "export_receipt.json").write_text("{}")
        module = {name: (value + (0.0001 if post else 0)).to(torch.bfloat16) for name, value in baseline.items()}
        torch.save(
            {
                "module": module,
                "param_shapes": shapes,
                "global_steps": step,
                "dp_world_size": world_size,
                "mp_world_size": 1,
                "ds_config": config,
                "shared_params": {},
            },
            tag / "mp_rank_00_model_states.pt",
        )
        for rank in range(world_size):
            masters, mappings, paddings, states, groups = ([], [], [], {}, [])
            for group_index, group in enumerate(shapes):
                total = sum((shape.numel() for shape in group.values()))
                length = math.ceil(total / (2 * world_size)) * 2
                start, end = (rank * length, (rank + 1) * length)
                valid = max(0, min(length, total - start))
                values = [baseline[name].flatten() + (0.0001 if post else 0) for name in group]
                vector = torch.cat(values)
                masters.append(vector[start : start + valid].clone())
                paddings.append(length - valid)
                mapping = OrderedDict()
                offset = 0
                for name, shape in group.items():
                    fragment_start, fragment_end = (max(offset, start), min(offset + shape.numel(), end))
                    if fragment_start < fragment_end:
                        mapping[name] = fragment_address(
                            numel=fragment_end - fragment_start, start=fragment_start - start
                        )
                    offset += shape.numel()
                mappings.append(mapping)
                param_id = 17 + group_index * 13
                groups.append(
                    {
                        "params": [param_id],
                        "betas": (0.9, 0.999),
                        "lr": _cortex_optimizer_decode_LR if post else 0.001,
                        "eps": 1e-08,
                        "weight_decay": 0.0,
                        "bias_correction": True,
                        "step": 0,
                    }
                )
                if post or before_step or (not lazy_baseline):
                    state = {"step": torch.tensor(float(step))}
                    for key in ("exp_avg", "exp_avg_sq"):
                        full = torch.cat([expected[name][key].flatten() for name in group])
                        partition = torch.zeros(length, dtype=torch.float32)
                        partition[:valid].copy_(full[start : start + valid] if post else torch.zeros(valid))
                        state[key] = partition
                    states[param_id] = state
            torch.save(
                {
                    "ds_config": config,
                    "ds_version": "synthetic",
                    "optimizer_state_dict": {
                        "zero_stage": ZeroStageEnum.gradients,
                        "base_optimizer_state": {"state": states, "param_groups": groups},
                        "single_partition_of_fp32_groups": masters,
                        "param_slice_mappings": mappings,
                        "partition_count": [world_size] * len(shapes),
                        "group_paddings": paddings,
                        "param_alignment_paddings": [[0] * len(group) for group in shapes],
                        "loss_scaler": LossScaler(1.0),
                        "dynamic_loss_scale": False,
                        "overflow": False,
                        "clip_grad": 1.0,
                        "ds_version": "synthetic",
                    },
                },
                tag / f"{prefix}zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",
            )
    return SimpleNamespace(before=before_root, after=after_root, expected=expected, baseline=baseline)


def _cortex_optimizer_decode_model(root):
    return next(root.rglob("*_model_states.pt"))


def _cortex_optimizer_decode_optim(root, rank=0):
    return next(root.rglob(f"*zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"))


def _cortex_optimizer_decode_rewrite(path, edit):
    state = decoder._load(path)
    edit(state)
    replacement = path.with_suffix(".replacement")
    torch.save(state, replacement)
    replacement.replace(path)


@pytest.mark.parametrize("unsafe", ["/tmp", "/code", "/data-fast-shadow", "/data-fast", "symlink"])
def test_cortex_optimizer_decode_decode_requires_data_fast_before_checkpoint_load(tmp_path, monkeypatch, unsafe):
    destination = Path(unsafe)
    if unsafe == "symlink":
        destination = tmp_path / "escape"
        destination.symlink_to("/tmp", target_is_directory=True)
        destination /= "output"
    monkeypatch.setattr(
        decoder, "_snapshot", lambda *args, **kwargs: pytest.fail("unsafe output reached checkpoint load")
    )
    with pytest.raises(ValueError, match="/data-fast"):
        decoder.decode_optimizer_capture(tmp_path / "before", tmp_path / "after", destination)
    assert not (tmp_path / "before").exists()
    assert not (tmp_path / "after").exists()


@pytest.mark.parametrize("prefix", ["", "bf16_", "fp16_"])
@pytest.mark.parametrize("world_size", [1, 2, 4, 12])
def test_cortex_optimizer_decode_decode_named_fp32_update_moments_and_padding(tmp_path, prefix, world_size):
    pair = _cortex_optimizer_decode_pair(tmp_path, world_size=world_size, prefix=prefix)
    manifest = decoder.decode_optimizer_capture(
        pair.before,
        pair.after,
        tmp_path / "out",
        expected_optimizer_config=_cortex_optimizer_decode_OPTIMIZER,
        expected_learning_rate=_cortex_optimizer_decode_LR,
    )
    metadata = json.loads(manifest.read_text())
    assert metadata["step"] == 1
    assert metadata["before"]["engine_global_step"] == 0
    assert metadata["after"]["engine_global_step"] == 1
    assert [item["name"] for item in metadata["parameters"]] == list(pair.expected)
    for entry in metadata["parameters"]:
        tensor = torch.load(manifest.parent / entry["file"], weights_only=True)
        assert set(tensor) == {"parameter_update", "exp_avg", "exp_avg_sq"}
        for key, expected in pair.expected[entry["name"]].items():
            assert tensor[key].dtype == torch.float32
            assert tuple(entry["shape"]) == tuple(expected.shape)
            assert torch.equal(tensor[key], expected)
        assert bool(torch.count_nonzero(tensor["parameter_update"]))
    before_module = decoder._load(_cortex_optimizer_decode_model(pair.before))["module"]
    after_module = decoder._load(_cortex_optimizer_decode_model(pair.after))["module"]
    assert all((torch.equal(before_module[name], after_module[name]) for name in before_module))
    comparison = compare_optimizer_artifacts(
        manifest,
        manifest,
        optimizer_config=_cortex_optimizer_decode_OPTIMIZER,
        learning_rate=_cortex_optimizer_decode_LR,
    )
    assert len(comparison.deltas) == 3 * len(pair.expected)
    assert all((item.delta_norm == 0 for item in comparison.deltas if item.state in ("exp_avg", "exp_avg_sq")))
    assert not comparison.only_dss and (not comparison.only_reference)


def test_cortex_optimizer_decode_decode_uses_actual_state_step_and_group_lr_not_legacy_or_config_defaults(tmp_path):
    pair = _cortex_optimizer_decode_pair(tmp_path)
    state = decoder._load(_cortex_optimizer_decode_optim(pair.after))["optimizer_state_dict"]
    assert state["base_optimizer_state"]["param_groups"][0]["step"] == 0
    assert state["base_optimizer_state"]["state"][17]["step"] == 1
    assert (
        decoder._load(_cortex_optimizer_decode_optim(pair.after))["ds_config"]["optimizer"]["params"]["lr"]
        != _cortex_optimizer_decode_LR
    )
    assert decoder.decode_optimizer_capture(
        pair.before,
        pair.after,
        tmp_path / "out",
        expected_optimizer_config=_cortex_optimizer_decode_OPTIMIZER,
        expected_learning_rate=_cortex_optimizer_decode_LR,
    ).exists()


def test_cortex_optimizer_decode_decode_preserves_nonzero_actual_steps_without_relabeling_as_step_one(tmp_path):
    pair = _cortex_optimizer_decode_pair(tmp_path, before_step=10)
    manifest = decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")
    assert json.loads(manifest.read_text())["step"] == 11
    with pytest.raises(ValueError, match="expected=1"):
        compare_optimizer_artifacts(
            manifest,
            manifest,
            optimizer_config=_cortex_optimizer_decode_OPTIMIZER,
            learning_rate=_cortex_optimizer_decode_LR,
        )


def test_cortex_optimizer_decode_decode_accepts_initialized_step_zero_and_tag_directories(tmp_path):
    pair = _cortex_optimizer_decode_pair(tmp_path, lazy_baseline=False)
    assert decoder.decode_optimizer_capture(
        _cortex_optimizer_decode_model(pair.before).parent,
        _cortex_optimizer_decode_model(pair.after).parent,
        tmp_path / "out",
    ).exists()


def test_cortex_optimizer_decode_decode_mmap_restricted_loads_and_does_not_mutate_source_tensors(
    tmp_path, monkeypatch
):
    pair = _cortex_optimizer_decode_pair(tmp_path)
    before_bytes = {path: path.read_bytes() for root in (pair.before, pair.after) for path in root.rglob("*.pt")}
    original_load = torch.load
    calls = []

    def recording_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")
    assert calls
    assert all(
        (call["weights_only"] is True and call["mmap"] is True and (call["map_location"] == "cpu") for call in calls)
    )
    assert all((path.read_bytes() == data for path, data in before_bytes.items()))


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("zero_stage", 1, "ZeRO2"),
        ("zero_stage", 3, "ZeRO2"),
        ("zero_stage", None, "ZeRO stage"),
        ("partition_count", [2], "groups"),
        ("partition_count", [1, 2], "partition_count"),
        ("partition_count", [3, 3], "partition_count"),
        ("group_paddings", [1, 0], "partition padding"),
        ("group_paddings", [0], "groups"),
        ("group_paddings", [-1, 0], "padding"),
        ("param_alignment_paddings", [[0, 1], [0]], "alignment padding"),
        ("param_alignment_paddings", [[0], [0]], "alignment padding"),
        ("overflow", True, "overflow"),
        ("base_optimizer_state", [], "nonelastic"),
        ("single_partition_of_fp32_groups", [], "groups"),
        ("param_slice_mappings", [], "groups"),
    ],
)
def test_cortex_optimizer_decode_decode_rejects_invalid_optimizer_schema(tmp_path, field, value, error):
    pair = _cortex_optimizer_decode_pair(tmp_path)
    _cortex_optimizer_decode_rewrite(
        _cortex_optimizer_decode_optim(pair.after),
        lambda state: state["optimizer_state_dict"].__setitem__(field, value),
    )
    with pytest.raises(ValueError, match=error):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("key", ["fp32_master", "exp_avg", "exp_avg_sq"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_cortex_optimizer_decode_decode_rejects_non_fp32_tensors(tmp_path, side, key, dtype):
    pair = _cortex_optimizer_decode_pair(tmp_path, lazy_baseline=False)

    def edit(state):
        opt = state["optimizer_state_dict"]
        if key == "fp32_master":
            opt["single_partition_of_fp32_groups"][0] = opt["single_partition_of_fp32_groups"][0].to(dtype)
        else:
            opt["base_optimizer_state"]["state"][17][key] = opt["base_optimizer_state"]["state"][17][key].to(dtype)

    _cortex_optimizer_decode_rewrite(_cortex_optimizer_decode_optim(getattr(pair, side)), edit)
    with pytest.raises(ValueError, match="FP32"):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")


@pytest.mark.parametrize("key", ["exp_avg", "exp_avg_sq"])
@pytest.mark.parametrize("mutation", ["missing", "short", "nonvector"])
def test_cortex_optimizer_decode_decode_rejects_missing_or_malformed_moments(tmp_path, key, mutation):
    pair = _cortex_optimizer_decode_pair(tmp_path)

    def edit(state):
        moments = state["optimizer_state_dict"]["base_optimizer_state"]["state"][17]
        if mutation == "missing":
            del moments[key]
        elif mutation == "short":
            moments[key] = moments[key][:-1]
        else:
            moments[key] = moments[key].reshape(2, 3)

    _cortex_optimizer_decode_rewrite(_cortex_optimizer_decode_optim(pair.after), edit)
    with pytest.raises(ValueError, match="FP32"):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")


@pytest.mark.parametrize("step", [-1, 0, 2, 1.5, float("nan"), True])
def test_cortex_optimizer_decode_decode_rejects_inconsistent_post_optimizer_step(tmp_path, step):
    pair = _cortex_optimizer_decode_pair(tmp_path)
    _cortex_optimizer_decode_rewrite(
        _cortex_optimizer_decode_optim(pair.after, 1),
        lambda state: state["optimizer_state_dict"]["base_optimizer_state"]["state"][17].__setitem__("step", step),
    )
    with pytest.raises(ValueError, match="step"):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")


def test_cortex_optimizer_decode_decode_requires_exactly_one_engine_step(tmp_path):
    pair = _cortex_optimizer_decode_pair(tmp_path)
    for path in pair.after.rglob("*.pt"):
        if path == _cortex_optimizer_decode_model(pair.after):
            _cortex_optimizer_decode_rewrite(path, lambda state: state.__setitem__("global_steps", 3))
        else:
            _cortex_optimizer_decode_rewrite(
                path,
                lambda state: [
                    item.__setitem__("step", 3)
                    for item in state["optimizer_state_dict"]["base_optimizer_state"]["state"].values()
                ],
            )
    _cortex_optimizer_decode_model(pair.after).parent.rename(pair.after / "global_step3")
    with pytest.raises(ValueError, match="exactly one step"):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")


@pytest.mark.parametrize("mutation", ["remove", "duplicate", "foreign", "symlink", "second_tag", "second_model"])
def test_cortex_optimizer_decode_decode_rejects_ambiguous_or_incomplete_checkpoint_files(tmp_path, mutation):
    pair = _cortex_optimizer_decode_pair(tmp_path)
    path = _cortex_optimizer_decode_optim(pair.after, 1)
    if mutation == "remove":
        path.unlink()
    elif mutation == "duplicate":
        (path.parent / path.name.replace("bf16_", "")).write_bytes(path.read_bytes())
    elif mutation == "foreign":
        (path.parent / "expert_0_model_states.pt").write_bytes(path.read_bytes())
    elif mutation == "symlink":
        original = tmp_path / "saved.pt"
        path.rename(original)
        path.symlink_to(original)
    elif mutation == "second_tag":
        another = pair.after / "global_step2"
        another.mkdir()
        (another / path.name).write_bytes(path.read_bytes())
    else:
        model = _cortex_optimizer_decode_model(pair.after)
        (model.parent / "mp_rank_01_model_states.pt").write_bytes(model.read_bytes())
    with pytest.raises(ValueError):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")


@pytest.mark.parametrize("mutation", ["shape", "order", "duplicate", "empty", "module_shape"])
def test_cortex_optimizer_decode_decode_rejects_shape_or_order_drift(tmp_path, mutation):
    pair = _cortex_optimizer_decode_pair(tmp_path)

    def edit(state):
        shapes = state["param_shapes"]
        if mutation == "shape":
            shapes[0]["layer.weight"] = torch.Size([3, 2])
            state["module"]["layer.weight"] = state["module"]["layer.weight"].reshape(3, 2)
        elif mutation == "order":
            shapes[0] = OrderedDict(reversed(list(shapes[0].items())))
        elif mutation == "duplicate":
            shapes[1]["layer.weight"] = torch.Size([2, 3])
        elif mutation == "empty":
            shapes[0] = {}
        else:
            state["module"]["layer.weight"] = torch.empty(2, 2)

    _cortex_optimizer_decode_rewrite(_cortex_optimizer_decode_model(pair.after), edit)
    with pytest.raises(ValueError, match="shape|order|parameter|mapping"):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")


@pytest.mark.parametrize("mutation", ["missing", "extra", "offset", "length"])
def test_cortex_optimizer_decode_decode_rejects_bad_named_fragment_mappings(tmp_path, mutation):
    pair = _cortex_optimizer_decode_pair(tmp_path)

    def edit(state):
        mapping = state["optimizer_state_dict"]["param_slice_mappings"][0]
        if mutation == "missing":
            mapping.pop("layer.weight")
        elif mutation == "extra":
            mapping["unknown"] = fragment_address(start=0, numel=1)
        elif mutation == "offset":
            mapping["layer.weight"].start += 1
        else:
            mapping["layer.weight"].numel -= 1

    _cortex_optimizer_decode_rewrite(_cortex_optimizer_decode_optim(pair.after), edit)
    with pytest.raises(ValueError, match="mapping"):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("ep_config", "expert parallel"),
        ("ep_group", "EP"),
        ("tp", "parallel"),
        ("amsgrad", "AMSGrad"),
        ("missing_group", "groups"),
        ("extra_state", "state groups"),
        ("missing_post_state", "step"),
        ("rank_lr", "rank config"),
        ("rank_betas", "rank config"),
        ("lr", "learning rate"),
        ("betas", "config"),
        ("bias_correction", "bias_correction"),
    ],
)
def test_cortex_optimizer_decode_decode_rejects_unsupported_or_inconsistent_group_configuration(
    tmp_path, mutation, error
):
    pair = _cortex_optimizer_decode_pair(tmp_path)

    def edit(state):
        config = state["ds_config"]
        opt = state["optimizer_state_dict"]
        base = opt["base_optimizer_state"]
        group = base["param_groups"][0]
        if mutation == "ep_config":
            config["expert_parallel"] = {"enabled": True, "autoep_size": 2}
        elif mutation == "ep_group":
            group["moe"] = True
        elif mutation == "tp":
            config["tensor_parallel"] = {"tp_size": 2}
        elif mutation == "amsgrad":
            group["amsgrad"] = True
        elif mutation == "missing_group":
            base["param_groups"].pop()
        elif mutation == "extra_state":
            base["state"][99] = base["state"][17]
        elif mutation == "missing_post_state":
            base["state"].pop(17)
        elif mutation in ("lr", "rank_lr"):
            group["lr"] = 0.002
        elif mutation in ("betas", "rank_betas"):
            group["betas"] = (0.8, 0.999)
        else:
            group["bias_correction"] = False

    _cortex_optimizer_decode_rewrite(
        _cortex_optimizer_decode_optim(pair.after, int(mutation.startswith("rank_"))), edit
    )
    kwargs = (
        {}
        if mutation.startswith("rank_")
        else {
            "expected_optimizer_config": _cortex_optimizer_decode_OPTIMIZER,
            "expected_learning_rate": _cortex_optimizer_decode_LR,
        }
    )
    with pytest.raises(ValueError, match=error):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out", **kwargs)


@pytest.mark.parametrize("key", ["fp32_master", "exp_avg", "exp_avg_sq"])
def test_cortex_optimizer_decode_decode_nonfinite_parameter_rolls_back_only_new_output(tmp_path, key):
    pair = _cortex_optimizer_decode_pair(tmp_path)

    def edit(state):
        opt = state["optimizer_state_dict"]
        vector = (
            opt["single_partition_of_fp32_groups"][1]
            if key == "fp32_master"
            else opt["base_optimizer_state"]["state"][30][key]
        )
        vector[0] = float("nan")

    _cortex_optimizer_decode_rewrite(_cortex_optimizer_decode_optim(pair.after), edit)
    with pytest.raises(ValueError, match="nonfinite"):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert _cortex_optimizer_decode_model(pair.before).exists() and _cortex_optimizer_decode_model(pair.after).exists()


def test_cortex_optimizer_decode_decode_preserves_preexisting_output(tmp_path):
    pair = _cortex_optimizer_decode_pair(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "sentinel"
    sentinel.write_text("owned elsewhere")
    with pytest.raises(ValueError, match="already exist"):
        decoder.decode_optimizer_capture(pair.before, pair.after, out)
    assert sentinel.read_text() == "owned elsewhere"


def test_cortex_optimizer_decode_decode_rejects_output_nested_in_input(tmp_path):
    pair = _cortex_optimizer_decode_pair(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        decoder.decode_optimizer_capture(pair.before, pair.after, pair.before / "out")


def test_cortex_optimizer_decode_decode_does_not_allow_arbitrary_pickled_checkpoint_classes(tmp_path):
    pair = _cortex_optimizer_decode_pair(tmp_path)
    _cortex_optimizer_decode_rewrite(
        _cortex_optimizer_decode_optim(pair.after),
        lambda state: state.__setitem__("unknown", SimpleNamespace(value=1)),
    )
    with pytest.raises(ValueError, match="unsupported checkpoint globals"):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("source", ["model", "rank0", "rank1"])
@pytest.mark.parametrize("optimizer_type", ["Adam", "SGD", None, "FusedAdam"])
def test_cortex_optimizer_decode_decode_rejects_non_adamw_or_missing_model_and_shard_optimizer_types(
    tmp_path, side, source, optimizer_type
):
    pair = _cortex_optimizer_decode_pair(tmp_path)
    root = getattr(pair, side)
    path = (
        _cortex_optimizer_decode_model(root)
        if source == "model"
        else _cortex_optimizer_decode_optim(root, int(source[-1]))
    )
    _cortex_optimizer_decode_rewrite(
        path, lambda state: state["ds_config"]["optimizer"].__setitem__("type", optimizer_type)
    )
    with pytest.raises(ValueError, match="optimizer type AdamW"):
        decoder.decode_optimizer_capture(
            pair.before, pair.after, tmp_path / "out", expected_optimizer_config=_cortex_optimizer_decode_OPTIMIZER
        )
    assert not (tmp_path / "out").exists()


def test_cortex_optimizer_decode_decode_rejects_unmatched_expected_optimizer_type(tmp_path):
    pair = _cortex_optimizer_decode_pair(tmp_path)
    with pytest.raises(ValueError, match="expected optimizer type"):
        decoder.decode_optimizer_capture(
            pair.before,
            pair.after,
            tmp_path / "out",
            expected_optimizer_config={**_cortex_optimizer_decode_OPTIMIZER, "name": "Adam"},
        )


# Dss driver.


def test_dss_driver_globalize_expert_norms_replaces_rank_local_value() -> None:
    norms = _globalize_expert_norms(
        {"layers.0.mlp.experts.w1": 3.0, "layers.0.input_layernorm.weight": 7.0},
        {"layers.0.mlp.experts.w1": [3.0, 4.0]},
    )
    assert norms == {"layers.0.mlp.experts.w1": 5.0, "layers.0.input_layernorm.weight": 7.0}


def test_dss_driver_fwd_bwd_step_reads_rank_owned_norms_from_merged_list() -> None:
    from arctic_platform.correctness.harness.dss_driver import fwd_bwd_step

    class Client:

        def fwd_bwd(self, body):
            return {"avg_loss": 1.25, "metrics": {}}

        def step(self, learning_rate):
            return {
                "metrics": {
                    "gradient_norms_per_param": [{}, {"layers.0.input_layernorm.weight": 2.0}],
                    "gradient_norms_per_expert": [],
                }
            }

    class Job:
        client = Client()

    result = fwd_bwd_step(None, Job(), {}, learning_rate=0.001)
    assert result.avg_loss == 1.25
    assert result.grad_norms == {"layers.0.input_layernorm.weight": 2.0}


def test_dss_driver_fwd_bwd_step_unwraps_rank_owned_optimizer_manifest() -> None:
    from arctic_platform.correctness.harness.dss_driver import fwd_bwd_step

    class Client:

        def fwd_bwd(self, body):
            return {"avg_loss": 1.25, "metrics": {}}

        def step(self, learning_rate):
            return {"metrics": {"optimizer_state_manifest": ["/tmp/manifest.json"]}}

    class Job:
        client = Client()

    result = fwd_bwd_step(None, Job(), {}, learning_rate=0.001)
    assert result.optimizer_state_manifest == "/tmp/manifest.json"


def test_dss_driver_optimizer_capture_worker_patches_imported_server_worker_bindings(monkeypatch) -> None:
    import arctic_platform.common as common_package
    from arctic_platform.correctness.harness.optimizer_capture import _patched_deepspeed_worker_bindings

    worker_module = types.ModuleType("arctic_platform.common.deepspeed_worker")
    original_worker = object()
    stale_worker = object()
    capture_worker = object()
    common_server = types.ModuleType("arctic_platform.common.ray_server")
    rl_server = types.ModuleType("arctic_platform.rl.ray_server")
    worker_module.DeepSpeedWorker = original_worker
    common_server.DeepSpeedWorker = stale_worker
    rl_server.DeepSpeedWorker = stale_worker
    monkeypatch.setitem(sys.modules, "arctic_platform.common.deepspeed_worker", worker_module)
    monkeypatch.setattr(common_package, "deepspeed_worker", worker_module, raising=False)
    monkeypatch.setitem(sys.modules, "arctic_platform.common.ray_server", common_server)
    monkeypatch.setitem(sys.modules, "arctic_platform.rl.ray_server", rl_server)
    with _patched_deepspeed_worker_bindings(capture_worker):
        assert worker_module.DeepSpeedWorker is capture_worker
        assert common_server.DeepSpeedWorker is capture_worker
        assert rl_server.DeepSpeedWorker is capture_worker
    assert worker_module.DeepSpeedWorker is original_worker
    assert common_server.DeepSpeedWorker is stale_worker
    assert rl_server.DeepSpeedWorker is stale_worker


def test_dss_driver_optimizer_capture_server_patches_workers_inside_actor(monkeypatch) -> None:
    import arctic_platform.common as common_package
    from arctic_platform.correctness.harness import optimizer_capture

    worker_module = types.ModuleType("arctic_platform.common.deepspeed_worker")
    server_module = types.ModuleType("arctic_platform.common.ray_server")
    original_worker, capture_worker = (object(), object())
    worker_module.DeepSpeedWorker = original_worker
    server_module.DeepSpeedWorker = original_worker
    events = []

    class ServerState:

        async def initialize(self, payload):
            events.append((worker_module.DeepSpeedWorker, server_module.DeepSpeedWorker))
            if payload.get("fail"):
                raise RuntimeError("initialize failed")
            return {"job_id": 7}

    server_module.ArcticRLRayServerState = ServerState
    server_module.ArcticRLRayServer = type("RequestServer", (), {})
    monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)
    monkeypatch.setitem(sys.modules, server_module.__name__, server_module)
    monkeypatch.setattr(common_package, "deepspeed_worker", worker_module, raising=False)
    monkeypatch.setattr(common_package, "ray_server", server_module, raising=False)
    monkeypatch.setattr(optimizer_capture, "_optimizer_capture_worker_class", lambda: capture_worker)
    with optimizer_capture.optimizer_capture_worker():
        capture_state = server_module.ArcticRLRayServerState
    assert server_module.ArcticRLRayServerState is ServerState
    actor = capture_state()
    assert asyncio.run(actor.initialize({})) == {"job_id": 7}
    with pytest.raises(RuntimeError, match="initialize failed"):
        asyncio.run(actor.initialize({"fail": True}))
    assert events == [(capture_worker, capture_worker)] * 2
    assert worker_module.DeepSpeedWorker is original_worker
    assert server_module.DeepSpeedWorker is original_worker


def test_dss_driver_prime_rl_receives_prediction_aligned_labels() -> None:
    batch = Batch(
        name="one",
        input_ids=torch.tensor([[4, 5, 6]]),
        position_ids=torch.tensor([[0, 1, 2]]),
        labels=torch.tensor([[4, 5, 6]]),
    )
    assert _labels_for_provider(batch, "prime_rl").tolist() == [[5, 6, -100]]
    assert _labels_for_provider(batch, "huggingface").tolist() == [[4, 5, 6]]


def test_dss_driver_prime_rl_uses_fused_model_loss_with_aligned_labels() -> None:
    batch = Batch(
        name="one",
        input_ids=torch.tensor([[4, 5, 6]]),
        position_ids=torch.tensor([[0, 1, 2]]),
        labels=torch.tensor([[4, 5, 6]]),
    )
    from arctic_platform.correctness.harness.dss_driver import pack

    prime = pack(batch, model_provider="prime_rl")
    assert prime["processing"]["loss_fn"] == "sft"
    assert prime["meta"]["labels_are_shifted"] is True
    assert prime["batch"]["labels"].tolist() == [[5, 6, -100]]
    assert pack(batch, model_provider="huggingface")["processing"]["loss_fn"] == "sft"


def test_dss_driver_pack_microbatches_emits_exact_gas_list() -> None:
    batch = Batch(
        name="gas4",
        input_ids=torch.arange(15).view(5, 3),
        position_ids=torch.arange(3).unsqueeze(0).expand(5, -1).contiguous(),
        labels=torch.arange(15).view(5, 3),
    )
    body = pack_microbatches(batch, 4, model_provider="huggingface")
    assert body["meta"] == {}
    assert [micro["input_ids"].shape[0] for micro in body["batch"]] == [2, 1, 1, 1]
    assert body["batch"][0]["input_ids"].tolist() == [[0, 1, 2], [3, 4, 5]]
    assert body["batch"][-1]["labels"].tolist() == [[12, 13, 14]]


def test_dss_driver_pack_microbatches_preserves_prime_rl_shifted_labels() -> None:
    batch = Batch(
        name="prime",
        input_ids=torch.tensor([[4, 5, 6], [7, 8, 9]]),
        position_ids=torch.arange(3).unsqueeze(0).expand(2, -1).contiguous(),
        labels=torch.tensor([[4, 5, 6], [7, 8, 9]]),
    )
    body = pack_microbatches(batch, 2, model_provider="prime_rl")
    assert body["meta"]["labels_are_shifted"] is True
    assert body["batch"][0]["labels"].tolist() == [[5, 6, -100]]
    assert body["batch"][1]["labels"].tolist() == [[8, 9, -100]]


def test_dss_driver_pack_microbatches_can_request_memory_sft_ce_without_mutating_config() -> None:
    batch = Batch(
        name="gas4",
        input_ids=torch.arange(12).view(4, 3),
        position_ids=torch.arange(3).unsqueeze(0).expand(4, -1).contiguous(),
        labels=torch.arange(12).view(4, 3),
    )
    processing_config = {"logits_optimization": "memory", "logits_optimization_peak_mem_size_in_gib": 4}
    body = pack_microbatches(batch, 4, loss_fn="sft_ce", processing_config=processing_config)
    processing_config["logits_optimization"] = "compute"
    assert body["processing"] == {
        "loss_fn": "sft_ce",
        "config": {"logits_optimization": "memory", "logits_optimization_peak_mem_size_in_gib": 4},
    }


def test_dss_driver_normal_packing_uses_plain_supervised_loss() -> None:
    from arctic_platform.correctness.harness.batches import build_batch

    body = pack(build_batch("gas4", 16, 8, 128))
    assert isinstance(body["batch"], dict)
    assert body["processing"] == {"loss_fn": "sft"}


def test_dss_driver_reference_runner_passes_the_harness_seed(tmp_path, monkeypatch) -> None:
    from arctic_platform.correctness.harness import runner

    output = tmp_path / "reference.json"
    captured = {}

    def run_subprocess(command, **kwargs):
        captured["command"] = command
        output.write_text('{"loss": 0.0}')
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(runner.subprocess, "run", run_subprocess)
    result = runner.run_reference(
        "model",
        tmp_path / "batch.pt",
        output,
        token_budget=1,
        ce_chunk=1,
        attn="sdpa",
        fp32_lm_head=False,
        seed=1234,
        peft_adapter_path=tmp_path / "adapter",
    )
    seed_index = captured["command"].index("--seed")
    adapter_index = captured["command"].index("--peft-adapter")
    assert captured["command"][seed_index + 1] == "1234"
    assert captured["command"][adapter_index + 1] == str(tmp_path / "adapter")
    assert result == {"loss": 0.0}


@pytest.mark.parametrize("unsafe", ["/tmp", "/code", "/data-fast-shadow", "/data-fast", "symlink"])
@pytest.mark.parametrize("boundary", ["execute", "execute_rl", "adapter", "reference_optimizer"])
def test_dss_driver_runner_requires_data_fast_before_operational_side_effects(tmp_path, monkeypatch, unsafe, boundary):
    from arctic_platform.correctness.harness import runner

    destination = Path(unsafe)
    if unsafe == "symlink":
        destination = tmp_path / "escape"
        destination.symlink_to("/tmp", target_is_directory=True)
        destination /= "output"
    cfg = SimpleNamespace(
        training={"peft_config": {"peft_type": "LORA"}},
        n_gpus=1,
        at_gpu_width=lambda *args: pytest.fail("unsafe workdir reached config preparation"),
    )
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("unsafe output reached subprocess")
    )
    with pytest.raises(ValueError, match="/data-fast"):
        if boundary == "execute":
            runner.execute(cfg, None, [], vocab_size=128, workdir=destination)
        elif boundary == "execute_rl":
            runner.execute_rl(cfg, None, vocab_size=128, workdir=destination)
        elif boundary == "adapter":
            runner.materialize_dss_peft_adapter(cfg, "read-only-model", destination, attn_implementation="eager")
        else:
            runner.run_reference(
                "read-only-model",
                tmp_path / "batch.pt",
                tmp_path / "reference.json",
                token_budget=1,
                ce_chunk=1,
                attn="sdpa",
                fp32_lm_head=False,
                optimizer_output_dir=destination,
            )
    assert not (tmp_path / "reference.json").exists()


def test_dss_driver_reference_allows_small_repo_report_with_data_fast_optimizer_output(tmp_path, monkeypatch):
    from arctic_platform.correctness.harness import runner

    output = Path(__file__).resolve().parents[3] / ".codex-run-state" / f"reference-report-{tmp_path.name}.json"
    optimizer_output = tmp_path / "optimizer"
    commands = []

    def run_subprocess(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    original_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == output:
            return '{"loss": 0.0}'
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", run_subprocess)
    monkeypatch.setattr(Path, "read_text", read_text)
    assert not output.exists()
    result = runner.run_reference(
        "read-only-model",
        tmp_path / "batch.pt",
        output,
        token_budget=1,
        ce_chunk=1,
        attn="sdpa",
        fp32_lm_head=False,
        optimizer_config={"name": "AdamW"},
        learning_rate=0.01,
        optimizer_output_dir=optimizer_output,
    )
    assert result == {"loss": 0.0}
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--out") + 1] == str(output)
    assert command[command.index("--optimizer-output-dir") + 1] == str(optimizer_output)
    assert not output.exists()


def test_dss_driver_copy_checkpoint_peft_adapter_accepts_a_checkpoint_adapter_pack(tmp_path) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    source = checkpoint_root / "global_step0" / "default"
    source.mkdir(parents=True)
    (source / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    save_file({"base_model.model.q_proj.lora_A.weight": torch.ones(2, 2)}, source / "adapter_model.safetensors")
    destination = tmp_path / "adapter"
    assert copy_checkpoint_peft_adapter(checkpoint_root, destination) == destination
    assert (destination / "adapter_model.safetensors").read_bytes() == (
        source / "adapter_model.safetensors"
    ).read_bytes()


def test_dss_driver_copy_checkpoint_peft_adapter_accepts_a_weights_only_hf_pack(tmp_path) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    source = checkpoint_root / "hf"
    source.mkdir(parents=True)
    (source / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    save_file({"base_model.model.q_proj.lora_A.weight": torch.ones(2, 2)}, source / "adapter_model.safetensors")
    destination = tmp_path / "adapter"
    assert copy_checkpoint_peft_adapter(checkpoint_root, destination) == destination
    assert (destination / "adapter_model.safetensors").read_bytes() == (
        source / "adapter_model.safetensors"
    ).read_bytes()


def test_dss_driver_copy_checkpoint_peft_adapter_rejects_empty_tensors_before_copy(tmp_path) -> None:
    source = tmp_path / "checkpoint" / "hf"
    source.mkdir(parents=True)
    (source / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    save_file({}, source / "adapter_model.safetensors")
    destination = tmp_path / "adapter"
    destination.mkdir()
    (destination / "sentinel").write_text("preserve")
    with pytest.raises(RuntimeError, match="no adapter tensors"):
        copy_checkpoint_peft_adapter(source.parent, destination)
    assert (destination / "sentinel").read_text() == "preserve"


def test_dss_driver_materialize_peft_adapter_accepts_adapter_only_export(tmp_path, monkeypatch) -> None:
    from arctic_platform.correctness.harness import runner

    cfg = SimpleNamespace(training={"peft_config": {"peft_type": "LORA"}}, n_gpus=1)
    cfg.at_gpu_width = lambda width: cfg
    source = tmp_path / "checkpoint" / "hf"
    source.mkdir(parents=True)
    (source / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    save_file({"base_model.model.q_proj.lora_A.weight": torch.ones(2, 2)}, source / "adapter_model.safetensors")
    calls = []

    def save_checkpoint(**kwargs):
        calls.append(kwargs)
        return {"path": str(source.parent), "hf_path": str(source)}

    monkeypatch.setattr(runner, "build_payload", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "gateway", lambda *args: contextlib.nullcontext(object()))
    job = SimpleNamespace(client=SimpleNamespace(save_checkpoint=save_checkpoint))
    monkeypatch.setattr(runner, "running_job", lambda *args: contextlib.nullcontext(job))
    exported = runner.materialize_dss_peft_adapter(cfg, "unused", tmp_path, attn_implementation="eager")
    assert exported == tmp_path / "adapter"
    assert (exported / "adapter_model.safetensors").read_bytes() == (source / "adapter_model.safetensors").read_bytes()
    assert calls == [{"path": str(source.parent), "export_hf": True}]


def test_dss_driver_full_model_weights_only_export_still_requires_model_config(tmp_path) -> None:
    from arctic_platform.correctness.harness.dss_driver import save_weights_only_checkpoint

    client = SimpleNamespace(save_checkpoint=lambda **kwargs: {"hf_path": str(tmp_path), "path": str(tmp_path)})
    with pytest.raises(RuntimeError, match="no model config"):
        save_weights_only_checkpoint(None, SimpleNamespace(client=client), tmp_path)


def test_dss_driver_copy_checkpoint_peft_adapter_rejects_a_full_model_pack(tmp_path) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    source = checkpoint_root / "global_step0" / "default"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"full-model")
    with pytest.raises(RuntimeError, match="no PEFT pack"):
        copy_checkpoint_peft_adapter(checkpoint_root, tmp_path / "adapter")


def test_dss_driver_payload_routes_optimizer_capture_to_debug_for_harness(tmp_path) -> None:
    destination = tmp_path / "optimizer"
    payload = build_payload(
        {"n_gpus": 1, "mb_spec": {"max_tokens_per_mb": 1024}}, "model", 42, optimizer_state_output_dir=destination
    )
    client_config = _training_client_config(type("Session", (), {"workdir": tmp_path})(), payload)
    worker_config = client_config.to_onprem("training")["ds_worker_config"]
    assert worker_config["debug"]["optimizer_state_output_dir"] == str(destination.resolve())


@pytest.mark.parametrize(
    "job_config",
    [
        {},
        {"training_config": None, "ds_worker_config": None},
        {"training_config": {"debug": None}, "ds_worker_config": {"debug": {}}},
        {"debug": {"optimizer_state_output_dir": "/ignored"}},
    ],
)
def test_dss_driver_optimizer_capture_debug_config_defaults(job_config) -> None:
    from arctic_platform.correctness.harness.optimizer_capture import _optimizer_capture_debug_config

    assert _optimizer_capture_debug_config(job_config) == {}


def test_dss_driver_optimizer_capture_debug_config_merges_supported_sections_without_mutation() -> None:
    from arctic_platform.correctness.harness.optimizer_capture import _optimizer_capture_debug_config

    training_debug = {"optimizer_state_output_dir": "/training", "model_state_output_dir": "/model"}
    worker_debug = {"optimizer_state_output_dir": "/worker"}
    debug = _optimizer_capture_debug_config(
        {"training_config": {"debug": training_debug}, "ds_worker_config": {"debug": worker_debug}}
    )
    assert debug == {"optimizer_state_output_dir": "/worker", "model_state_output_dir": "/model"}
    debug["optimizer_state_output_dir"] = "/changed"
    assert training_debug == {"optimizer_state_output_dir": "/training", "model_state_output_dir": "/model"}
    assert worker_debug == {"optimizer_state_output_dir": "/worker"}


@pytest.mark.parametrize("section", ["training_config", "ds_worker_config"])
@pytest.mark.parametrize("nested_debug", [False, True], ids=["config", "debug"])
@pytest.mark.parametrize("value", [[], "", False])
def test_dss_driver_optimizer_capture_debug_config_rejects_malformed_sections(section, nested_debug, value) -> None:
    from arctic_platform.correctness.harness.optimizer_capture import _optimizer_capture_debug_config

    config = {section: {"debug": value} if nested_debug else value}
    field = f"{section}.debug" if nested_debug else section
    with pytest.raises(TypeError, match=f"{field} must be a dict"):
        _optimizer_capture_debug_config(config)


@pytest.fixture
def _dss_driver_optimizer_capture_initializer(monkeypatch):
    import arctic_platform.common as common_package
    from arctic_platform.correctness.harness import optimizer_capture

    class Worker:

        def initialize(self, master_addr, job_config):
            self.production_initialization = (master_addr, job_config)
            return True

    worker_module = types.ModuleType("arctic_platform.common.deepspeed_worker")
    worker_module.DeepSpeedWorker = SimpleNamespace(__ray_metadata__=SimpleNamespace(modified_class=Worker))
    ray_module = types.ModuleType("ray")
    ray_module.remote = lambda worker: worker
    monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)
    monkeypatch.setitem(sys.modules, ray_module.__name__, ray_module)
    monkeypatch.setattr(common_package, "deepspeed_worker", worker_module, raising=False)
    monkeypatch.setattr(optimizer_capture, "_OPTIMIZER_CAPTURE_WORKER", None)
    return optimizer_capture._optimizer_capture_worker_class()()


@pytest.mark.parametrize(
    "fields",
    [
        (),
        ("optimizer_state_output_dir",),
        ("model_state_output_dir",),
        ("optimizer_state_output_dir", "model_state_output_dir"),
    ],
    ids=["disabled", "optimizer", "model", "both"],
)
def test_dss_driver_optimizer_capture_initialize_uses_current_client_job_config(
    _dss_driver_optimizer_capture_initializer, tmp_path, fields
):
    payload = build_payload({"n_gpus": 1}, "model", 42)
    payload["training_config"]["debug"].update({field: str(tmp_path / field) for field in fields})
    client_config = _training_client_config(SimpleNamespace(workdir=tmp_path), payload)
    job_config = client_config.to_onprem("training")
    worker = _dss_driver_optimizer_capture_initializer
    assert worker.initialize("master", job_config) is True
    assert worker._optimizer_state_output_dir == (
        str(tmp_path / "optimizer_state_output_dir") if "optimizer_state_output_dir" in fields else None
    )
    assert worker._model_state_output_dir == (
        str(tmp_path / "model_state_output_dir") if "model_state_output_dir" in fields else None
    )
    assert worker.production_initialization[0] == "master"
    assert worker.production_initialization[1] is job_config


def test_dss_driver_optimizer_capture_initialize_accepts_path_objects(
    _dss_driver_optimizer_capture_initializer, tmp_path
):
    worker = _dss_driver_optimizer_capture_initializer
    job_config = {
        "training_config": {
            "debug": {
                "optimizer_state_output_dir": tmp_path / "optimizer",
                "model_state_output_dir": tmp_path / "model",
            }
        }
    }
    assert worker.initialize("master", job_config) is True
    assert worker._optimizer_state_output_dir == str(tmp_path / "optimizer")
    assert worker._model_state_output_dir == str(tmp_path / "model")


def test_dss_driver_optimizer_capture_initialize_worker_settings_can_disable_training_capture(
    _dss_driver_optimizer_capture_initializer,
):
    worker = _dss_driver_optimizer_capture_initializer
    job_config = {
        "training_config": {"debug": {"optimizer_state_output_dir": "/training", "model_state_output_dir": "/model"}},
        "ds_worker_config": {"debug": {"optimizer_state_output_dir": None, "model_state_output_dir": None}},
    }
    assert worker.initialize("master", job_config) is True
    assert worker._optimizer_state_output_dir is None
    assert worker._model_state_output_dir is None


@pytest.mark.parametrize("field", ["optimizer_state_output_dir", "model_state_output_dir"])
@pytest.mark.parametrize("value", ["", "relative/path"])
def test_dss_driver_optimizer_capture_initialize_rejects_relative_paths(
    _dss_driver_optimizer_capture_initializer, field, value
):
    worker = _dss_driver_optimizer_capture_initializer
    with pytest.raises(ValueError, match=f"debug.{field} must be an absolute path"):
        worker.initialize("master", {"ds_worker_config": {"debug": {field: value}}})
    assert not hasattr(worker, "production_initialization")


@pytest.mark.parametrize("field", ["optimizer_state_output_dir", "model_state_output_dir"])
def test_dss_driver_optimizer_capture_initialize_rejects_non_path_values(
    _dss_driver_optimizer_capture_initializer, field
):
    worker = _dss_driver_optimizer_capture_initializer
    with pytest.raises(TypeError):
        worker.initialize("master", {"ds_worker_config": {"debug": {field: 123}}})
    assert not hasattr(worker, "production_initialization")


def test_dss_driver_optimizer_capture_worker_context_restores_production_worker() -> None:
    pytest.importorskip("ray")
    from arctic_platform.common import deepspeed_worker as deepspeed_worker_module
    from arctic_platform.correctness.harness.optimizer_capture import _optimizer_capture_worker_class
    from arctic_platform.correctness.harness.optimizer_capture import optimizer_capture_worker

    original = deepspeed_worker_module.DeepSpeedWorker
    capture_worker = _optimizer_capture_worker_class()
    with optimizer_capture_worker():
        assert deepspeed_worker_module.DeepSpeedWorker is capture_worker
    assert deepspeed_worker_module.DeepSpeedWorker is original


def test_dss_driver_payload_can_skip_unused_gradient_norm_collection() -> None:
    payload = build_payload({}, "model", 42, gradient_norms_per_param=False)
    assert payload["training_config"]["debug"]["gradient_norms_per_param"] is False


def test_dss_driver_payload_preserves_configured_deepspeed_accumulation() -> None:
    payload = build_payload(
        {
            "n_gpus": 4,
            "ds_config": {
                "train_batch_size": 4,
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
            },
        },
        "model",
        42,
    )
    ds_config = payload["training_config"]["ds_config"]
    assert ds_config["gradient_accumulation_steps"] == 1
    assert ds_config["train_batch_size"] == 4


def test_dss_driver_payload_caps_large_microbatch_budget_at_64k() -> None:
    payload = build_payload({"mb_spec": {"max_tokens_per_mb": 262144}}, "model", 42)
    assert payload["training_config"]["mb_spec"]["max_tokens_per_mb"] == 65536


def test_dss_driver_payload_applies_spec_lm_head_chunk_without_overriding_config() -> None:
    injected = build_payload({}, "model", 42, lm_head_token_chunk_size=2048)
    configured = build_payload({"fused_lm_head_token_chunk_size": 4096}, "model", 42, lm_head_token_chunk_size=2048)
    assert injected["training_config"]["fused_lm_head_token_chunk_size"] == 2048
    assert configured["training_config"]["fused_lm_head_token_chunk_size"] == 4096


def test_dss_driver_payload_preserves_smaller_microbatch_budget() -> None:
    payload = build_payload({"mb_spec": {"max_tokens_per_mb": 10240}}, "model", 42)
    assert payload["training_config"]["mb_spec"]["max_tokens_per_mb"] == 10240


def test_dss_driver_payload_does_not_choose_deepep_chunks() -> None:
    payload = build_payload({"ep_size": 4, "mb_spec": {"max_tokens_per_mb": 10240}}, "model", 42)
    assert "deepep_token_chunk_size" not in payload["training_config"]


def test_dss_driver_pool_is_the_local_devices_when_the_harness_owns_the_gateway(monkeypatch) -> None:
    """With no external gateway the harness starts one on this node, so its devices are the whole pool."""
    monkeypatch.delenv("DSS_GATEWAY_URL", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    assert available_gpus() == 8


def test_dss_driver_pool_is_the_hostfile_slots_when_an_external_gateway_owns_placement(monkeypatch, tmp_path) -> None:
    """A gateway the harness did not start places jobs across its hostfile, which this node cannot see."""
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("10.0.0.1 slots=8\n10.0.0.2 slots=8\n")
    monkeypatch.setenv("DSS_GATEWAY_URL", "http://gateway:8000")
    monkeypatch.setenv("DSS_GATEWAY_HOSTFILE", str(hostfile))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    assert available_gpus() == 16


def test_dss_driver_pool_falls_back_to_local_devices_when_the_named_hostfile_is_absent(monkeypatch, tmp_path) -> None:
    """An unreadable hostfile leaves the pool unknown, so the guard keeps the number it can verify."""
    monkeypatch.setenv("DSS_GATEWAY_URL", "http://gateway:8000")
    monkeypatch.setenv("DSS_GATEWAY_HOSTFILE", str(tmp_path / "missing"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    assert available_gpus() == 8


def test_dss_driver_payload_pins_determinism_over_a_config_that_asks_otherwise() -> None:
    payload = build_payload(
        {
            "debug": {"full_determinism": False, "full_determinism_must_comply": True},
            "mb_spec": {"max_tokens_per_mb": 1024},
        },
        "model",
        42,
    )
    debug = payload["training_config"]["debug"]
    assert debug["full_determinism"] is True
    assert debug["full_determinism_must_comply"] is False


def test_dss_driver_payload_determinism_off_clears_a_config_that_asks_for_the_pin() -> None:
    payload = build_payload(
        {
            "debug": {"full_determinism": True, "full_determinism_must_comply": True},
            "mb_spec": {"max_tokens_per_mb": 1024},
        },
        "model",
        42,
        determinism="off",
    )
    debug = payload["training_config"]["debug"]
    assert debug["full_determinism"] is False
    assert debug["full_determinism_must_comply"] is False


@pytest.mark.parametrize("auth_mode", [None, "disabled", "token", "legacy-mode"])
@pytest.mark.parametrize("slots", [8, 16])
def test_dss_driver_gateway_runtime_cleans_before_and_after_standard_port(monkeypatch, auth_mode, slots) -> None:
    import torch

    from arctic_platform.correctness.harness import dss_driver

    events = []
    monkeypatch.setenv("MASTER_PORT", "12345")
    if auth_mode is None:
        monkeypatch.delenv("RAY_AUTH_MODE", raising=False)
    else:
        monkeypatch.setenv("RAY_AUTH_MODE", auth_mode)
    monkeypatch.setattr(dss_driver, "_stop_and_verify_gateway_runtime", lambda: events.append("clean"))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)

    class RayCluster:
        _HOSTFILE = "/job/hostfile"

        @staticmethod
        def init_ray_cluster(*, auto_attach):
            events.append(
                (
                    "start",
                    auto_attach,
                    RayCluster._HOSTFILE,
                    os.environ["MASTER_PORT"],
                    os.environ.get("RAY_AUTH_MODE"),
                )
            )

        @staticmethod
        def _shutdown():
            events.append(("shutdown", RayCluster._HOSTFILE))

    def load_ray_cluster():
        events.append(("load", os.environ["MASTER_PORT"], os.environ.get("RAY_AUTH_MODE")))
        return RayCluster

    monkeypatch.setattr(dss_driver, "_load_gateway_ray_cluster", load_ray_cluster)
    selected_hostfile = os.devnull if slots == 8 else "/job/hostfile"
    with _gateway_runtime(slots):
        assert RayCluster._HOSTFILE == selected_hostfile
        events.append(("body", os.environ["MASTER_PORT"], os.environ.get("RAY_AUTH_MODE")))
    assert events == [
        ("load", "29500", auth_mode),
        "clean",
        ("start", False, selected_hostfile, "29500", auth_mode),
        ("body", "29500", auth_mode),
        ("shutdown", selected_hostfile),
        "clean",
    ]
    assert os.environ["MASTER_PORT"] == "12345"
    assert os.environ.get("RAY_AUTH_MODE") == auth_mode
    assert RayCluster._HOSTFILE == "/job/hostfile"


@pytest.mark.parametrize("stage", ["start", "body"])
def test_dss_driver_gateway_runtime_restores_hostfile_after_failure(monkeypatch, stage) -> None:
    import torch

    from arctic_platform.correctness.harness import dss_driver

    failure = RuntimeError("runtime failed")
    shutdown_hostfiles = []
    ray = SimpleNamespace(_HOSTFILE="/job/hostfile")

    def start(*, auto_attach):
        assert auto_attach is False
        assert ray._HOSTFILE == os.devnull
        if stage == "start":
            raise failure

    ray.init_ray_cluster = start
    ray._shutdown = lambda: shutdown_hostfiles.append(ray._HOSTFILE)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(dss_driver, "_load_gateway_ray_cluster", lambda: ray)
    monkeypatch.setattr(dss_driver, "_stop_and_verify_gateway_runtime", lambda: None)
    monkeypatch.delenv("MASTER_PORT", raising=False)
    with pytest.raises(RuntimeError) as caught:
        with _gateway_runtime(8):
            raise failure
    assert caught.value is failure
    assert shutdown_hostfiles == [os.devnull]
    assert ray._HOSTFILE == "/job/hostfile"
    assert "MASTER_PORT" not in os.environ


def test_dss_driver_allocation_runtime_command_uses_hostfile_ds_ssh(monkeypatch, tmp_path) -> None:
    from arctic_platform.correctness.harness import dss_driver

    hostfile = tmp_path / "hostfile"
    hostfile.write_text("10.0.0.1 slots=8\n10.0.0.2 slots=8\n")
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setenv("DSS_GATEWAY_HOSTFILE", str(hostfile))
    monkeypatch.setattr(dss_driver.subprocess, "run", run)
    dss_driver._run_on_allocation("ray stop --force")
    assert captured["command"][1:3] == ["-f", str(hostfile)]
    assert captured["command"][0].endswith("/ds_ssh")
    assert captured["command"][3] == "ray stop --force"
    assert captured["kwargs"] == {"capture_output": True, "text": True, "timeout": 180}


def test_dss_driver_runtime_cleanup_stops_ray_then_checks_processes_and_port(monkeypatch) -> None:
    from arctic_platform.correctness.harness import dss_driver

    scripts = []
    monkeypatch.setattr(dss_driver, "_run_on_allocation", lambda script: scripts.append(script))
    dss_driver._stop_and_verify_gateway_runtime()
    assert len(scripts) == 2
    assert scripts[0].endswith("/ray stop --force")
    assert "pgrep -af" in scripts[1]
    assert "29500" in scripts[1]


def test_dss_driver_gateway_runtime_restores_port_when_preflight_fails(monkeypatch) -> None:
    from arctic_platform.correctness.harness import dss_driver

    calls = 0

    def cleanup():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("port is still bound")

    monkeypatch.setenv("MASTER_PORT", "12345")
    ray = SimpleNamespace(_HOSTFILE="/job/hostfile", init_ray_cluster=lambda **kwargs: None, _shutdown=lambda: None)
    monkeypatch.setattr(
        dss_driver,
        "_load_gateway_ray_cluster",
        lambda: ray,
    )
    monkeypatch.setattr(dss_driver, "_stop_and_verify_gateway_runtime", cleanup)
    with pytest.raises(RuntimeError, match="port is still bound"):
        with _gateway_runtime(1):
            pytest.fail("preflight must fail before launch")
    assert os.environ["MASTER_PORT"] == "12345"
    assert ray._HOSTFILE == "/job/hostfile"


def test_dss_driver_running_job_keeps_cleanup_outside_client_shutdown(monkeypatch, tmp_path) -> None:
    from arctic_platform.correctness.harness import dss_driver

    events = []

    @contextlib.contextmanager
    def runtime(slots):
        events.append(("runtime-enter", slots))
        try:
            yield
        finally:
            events.append("runtime-exit")

    class Client:

        def __init__(self, config):
            events.append("client-start")
            self.jobs = type("Jobs", (), {"training": 7})()

        def shutdown(self):
            events.append("client-shutdown")

    monkeypatch.setattr(dss_driver, "_gateway_runtime", runtime)
    monkeypatch.setattr(dss_driver, "ArcticClient", Client)
    payload = {
        "model_name": "model",
        "job_type": "training",
        "training_config": {"n_gpus": 1, "mb_spec": {"max_tokens_per_mb": 1}},
    }
    with dss_driver.running_job(dss_driver.GatewaySession(tmp_path, 1), payload) as job:
        events.append(("job", job.job_id))
    assert events == [("runtime-enter", 1), "client-start", ("job", 7), "client-shutdown", "runtime-exit"]


@pytest.mark.parametrize("stage", ["shutdown", "verify"])
def test_dss_driver_gateway_runtime_cleanup_failure_is_unsafe_and_restores_port(monkeypatch, stage):
    from arctic_platform.correctness.harness import dss_driver

    failure = RuntimeError("owned runtime not clean")
    calls = []

    def cleanup():
        calls.append("verify")
        if stage == "verify" and len(calls) > 1:
            raise failure

    def shutdown():
        if stage == "shutdown":
            raise failure

    ray = SimpleNamespace(_HOSTFILE="/job/hostfile", init_ray_cluster=lambda **kwargs: None, _shutdown=shutdown)
    monkeypatch.setattr(dss_driver, "_load_gateway_ray_cluster", lambda: ray)
    monkeypatch.setattr(dss_driver, "_stop_and_verify_gateway_runtime", cleanup)
    monkeypatch.setenv("MASTER_PORT", "12345")
    with pytest.raises(dss_driver.UnsafeCleanupError) as caught:
        with dss_driver._gateway_runtime(1):
            pass
    assert caught.value.__cause__ is failure
    assert os.environ["MASTER_PORT"] == "12345"
    assert calls == ["verify", "verify"]
    assert ray._HOSTFILE == "/job/hostfile"


def test_dss_driver_running_job_shutdown_failure_is_unsafe(monkeypatch, tmp_path):
    from arctic_platform.correctness.harness import dss_driver

    failure = RuntimeError("cancel failed")

    class Client:

        def __init__(self, config):
            pass

        def shutdown(self):
            raise failure

    monkeypatch.setattr(dss_driver, "_gateway_runtime", lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(dss_driver, "ArcticClient", Client)
    payload = {
        "model_name": "model",
        "job_type": "training",
        "training_config": {"n_gpus": 1, "mb_spec": {"max_tokens_per_mb": 1}},
    }
    with pytest.raises(dss_driver.UnsafeCleanupError) as caught:
        with dss_driver.running_job(dss_driver.GatewaySession(tmp_path, 1), payload):
            pass
    assert caught.value.__cause__ is failure


# Optimizer capture.


@pytest.fixture
def _optimizer_capture_capture_worker(monkeypatch, tmp_path):
    import arctic_platform.common as common_package

    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    model.weight.grad = torch.full_like(model.weight, 2.0)
    accumulated = {id(model.weight): torch.full(model.weight.shape, 6.25, dtype=torch.float32)}
    events = []

    class Worker:

        def _bf16_accumulated_gradients(self):
            events.append("accumulated")
            return accumulated

    worker_module = types.ModuleType("arctic_platform.common.deepspeed_worker")
    worker_module.DeepSpeedWorker = SimpleNamespace(__ray_metadata__=SimpleNamespace(modified_class=Worker))
    worker_module._worker_debug_config = lambda config: config.get("debug", {})
    ray_module = types.ModuleType("ray")
    ray_module.remote = lambda worker: worker
    deepspeed_utils = types.ModuleType("deepspeed.utils")
    deepspeed_utils.safe_get_full_fp32_param = lambda parameter: parameter.detach().float()

    def full_gradient(parameter):
        events.append("fallback")
        return parameter.grad

    deepspeed_utils.safe_get_full_grad = full_gradient
    monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)
    monkeypatch.setitem(sys.modules, ray_module.__name__, ray_module)
    monkeypatch.setitem(sys.modules, deepspeed_utils.__name__, deepspeed_utils)
    monkeypatch.setattr(common_package, "deepspeed_worker", worker_module, raising=False)
    monkeypatch.setattr(optimizer_capture, "_OPTIMIZER_CAPTURE_WORKER", None)
    monkeypatch.setattr(optimizer_capture.dist, "barrier", lambda: None)
    actor = optimizer_capture._optimizer_capture_worker_class()()
    actor._optimizer_state_output_dir = str(tmp_path / "capture")
    actor.rank = 0
    actor.engine = SimpleNamespace(
        module=model, optimizer=SimpleNamespace(state={}), global_steps=0, get_lr=lambda: [0.0001]
    )
    actor._full_adam_moments = lambda parameter: (None, None)
    return SimpleNamespace(
        actor=actor,
        model=model,
        accumulated=accumulated,
        events=events,
        artifact=tmp_path / "capture/before/000000.pt",
    )


def test_optimizer_capture_bf16_capture_uses_reduced_accumulation_not_rank_local_parameter_grad(
    _optimizer_capture_capture_worker,
):
    state = _optimizer_capture_capture_worker
    state.actor._snapshot_optimizer_state_parameters()
    saved = torch.load(state.artifact, weights_only=True)
    assert saved["gradient"].dtype == torch.float32
    assert torch.equal(saved["gradient"], state.accumulated[id(state.model.weight)])
    assert state.events == ["accumulated"]
    assert torch.equal(state.model.weight.grad, torch.full_like(state.model.weight, 2.0))


def test_optimizer_capture_missing_bf16_accumulation_does_not_fall_back_to_local_gradient(
    _optimizer_capture_capture_worker,
):
    state = _optimizer_capture_capture_worker
    state.accumulated.clear()
    with pytest.raises(RuntimeError, match="no reduced gradient"):
        state.actor._snapshot_optimizer_state_parameters()
    assert state.events == ["accumulated"]
    assert not state.artifact.exists()


def test_optimizer_capture_non_bf16_capture_keeps_deepspeed_full_gradient_fallback(_optimizer_capture_capture_worker):
    state = _optimizer_capture_capture_worker
    state.actor._bf16_accumulated_gradients = lambda: None
    state.actor._snapshot_optimizer_state_parameters()
    saved = torch.load(state.artifact, weights_only=True)
    assert torch.equal(saved["gradient"], state.model.weight.grad)
    assert state.events == ["fallback"]


def test_optimizer_capture_disabled_optimizer_capture_reads_no_gradients(_optimizer_capture_capture_worker):
    state = _optimizer_capture_capture_worker
    state.actor._optimizer_state_output_dir = None
    state.actor._snapshot_optimizer_state_parameters()
    assert state.events == []
    assert not state.artifact.exists()


def test_optimizer_capture_inactive_bf16_parameter_records_absent_gradient_without_local_fallback(
    _optimizer_capture_capture_worker,
):
    state = _optimizer_capture_capture_worker
    parameter = state.model.weight
    parameter.grad = None
    state.accumulated.clear()
    state.actor.engine.optimizer.bf16_groups = [[parameter]]
    state.actor.engine.optimizer.fp32_groups_has_gradients = [[False]]
    state.actor._snapshot_optimizer_state_parameters()
    saved = torch.load(state.artifact, weights_only=True)
    assert saved["gradient"] is None
    assert torch.equal(saved["fp32_master"], parameter.detach().float())
    assert state.events == ["accumulated"]


def test_optimizer_capture_inactive_gradient_does_not_hide_post_step_optimizer_state(
    _optimizer_capture_capture_worker,
):
    state = _optimizer_capture_capture_worker
    parameter = state.model.weight
    parameter.grad = None
    state.accumulated.clear()
    state.actor.engine.optimizer.bf16_groups = [[parameter]]
    state.actor.engine.optimizer.fp32_groups_has_gradients = [[False]]
    state.actor._snapshot_optimizer_state_parameters()
    moments = (torch.full_like(parameter, 0.5, dtype=torch.float32), torch.ones_like(parameter, dtype=torch.float32))
    state.actor._full_adam_moments = lambda parameter: moments
    state.actor.engine.global_steps = 1
    metrics = state.actor._optimizer_state_artifact_metrics()
    manifest = Path(metrics["optimizer_state_manifest"])
    saved = torch.load(manifest.parent / "000000.pt", weights_only=True)
    assert saved["gradient"] is None
    assert torch.equal(saved["exp_avg"], moments[0])
    assert torch.equal(saved["exp_avg_sq"], moments[1])


def test_optimizer_capture_missing_participating_bf16_buffer_with_no_local_grad_still_errors(
    _optimizer_capture_capture_worker,
):
    state = _optimizer_capture_capture_worker
    parameter = state.model.weight
    parameter.grad = None
    state.accumulated.clear()
    state.actor.engine.optimizer.bf16_groups = [[parameter]]
    state.actor.engine.optimizer.fp32_groups_has_gradients = [[True]]
    with pytest.raises(RuntimeError, match="no reduced gradient"):
        state.actor._snapshot_optimizer_state_parameters()
    assert state.events == ["accumulated"]
    assert not state.artifact.exists()


@pytest.fixture
def _optimizer_capture_capture_server(monkeypatch):
    import arctic_platform.common as common_package

    files = {"head": {}, "peer": {}}
    events = []
    manifest = "/data-fast/tmp/capture/manifest.json"
    files["peer"] = {manifest: b'{"parameters": []}', "/data-fast/tmp/capture/000000.pt": b"tensor"}
    response = {"job_id": 7, "metrics": {"optimizer_state_manifest": [manifest]}, "batch": {}}
    failures = {}
    nodes = [
        {"Alive": True, "NodeManagerAddress": "head"},
        {"Alive": True, "NodeManagerAddress": "peer"},
        {"Alive": False, "NodeManagerAddress": "old"},
    ]

    class Worker:

        def __init__(self, host):
            self.host = host
            self.routable_ip = SimpleNamespace(remote=lambda: self.host)
            self.export_node_checkpoint = SimpleNamespace(remote=self.export)
            self.require_checkpoint_files = SimpleNamespace(remote=self.require)

        def export(self, root, peers):
            events.append(("export", self.host, root, peers))
            if failures.get("export"):
                raise RuntimeError("artifact transfer failed")
            selected = {path: content for path, content in files[self.host].items() if Path(path).is_relative_to(root)}
            for peer in peers:
                files[peer].update(selected)
            return [str(Path(path).relative_to(root)) for path in selected]

        def require(self, root, relative_paths):
            events.append(("require", self.host))
            if failures.get("require"):
                raise FileNotFoundError("artifact unreadable after publication")
            for relative_path in relative_paths:
                if str(Path(root) / relative_path) not in files[self.host]:
                    raise FileNotFoundError(relative_path)

    class ServerState:

        async def initialize(self, payload):
            return {"job_id": 7}

    class Server:

        async def step(self, job_id, body=None):
            events.append(("step", job_id, body))
            if failures.get("step"):
                raise RuntimeError("optimizer step failed")
            return response

    worker_module = types.ModuleType("arctic_platform.common.deepspeed_worker")
    worker_module.DeepSpeedWorker = object()
    server_module = types.ModuleType("arctic_platform.common.ray_server")
    server_module.DeepSpeedWorker = worker_module.DeepSpeedWorker
    server_module.ArcticRLRayServerState = ServerState
    server_module.ArcticRLRayServer = Server
    compatibility_module = types.ModuleType("arctic_platform.rl.ray_server")
    compatibility_module.DeepSpeedWorker = worker_module.DeepSpeedWorker
    compatibility_module.ArcticRLRayServerState = ServerState
    compatibility_module.ArcticRLRayServer = Server
    ray_module = types.ModuleType("ray")
    ray_module.get = lambda value: value
    ray_module.nodes = lambda: nodes
    for module in (worker_module, server_module, compatibility_module, ray_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(common_package, "deepspeed_worker", worker_module, raising=False)
    monkeypatch.setattr(common_package, "ray_server", server_module, raising=False)
    production_worker = worker_module.DeepSpeedWorker
    capture_worker = object()
    monkeypatch.setattr(optimizer_capture, "_optimizer_capture_worker_class", lambda: capture_worker)
    with optimizer_capture.optimizer_capture_worker():
        actor = server_module.ArcticRLRayServer()
    actor.training_workers = [Worker("peer"), Worker("peer"), Worker("head"), Worker("head")]
    return SimpleNamespace(
        actor=actor,
        files=files,
        events=events,
        manifest=manifest,
        response=response,
        failures=failures,
        nodes=nodes,
        server_module=server_module,
        compatibility_module=compatibility_module,
        worker_module=worker_module,
        state_class=ServerState,
        server_class=Server,
        production_worker=production_worker,
        capture_worker=capture_worker,
    )


def test_optimizer_capture_publication_fixture_matches_product_request_server():
    source = Path(__file__).resolve().parents[2] / "common/ray_server.py"
    owners = [
        node.name
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef)
        and any((isinstance(member, ast.AsyncFunctionDef) and member.name == "step" for member in node.body))
    ]
    assert owners == ["ArcticRLRayServer"]


def test_optimizer_capture_capture_context_restores_both_server_classes_after_exception(
    _optimizer_capture_capture_server,
):
    state = _optimizer_capture_capture_server
    with pytest.raises(RuntimeError, match="context failed"):
        with optimizer_capture.optimizer_capture_worker():
            assert state.server_module.ArcticRLRayServer is not state.server_class
            assert not hasattr(state.server_module.ArcticRLRayServerState, "step")
            assert state.worker_module.DeepSpeedWorker is state.capture_worker
            raise RuntimeError("context failed")
    assert state.server_module.ArcticRLRayServer is state.server_class
    assert state.server_module.ArcticRLRayServerState is state.state_class
    assert state.worker_module.DeepSpeedWorker is state.production_worker
    assert state.server_module.DeepSpeedWorker is state.production_worker


def test_optimizer_capture_nested_capture_context_restores_outer_bindings(_optimizer_capture_capture_server):
    state = _optimizer_capture_capture_server
    with optimizer_capture.optimizer_capture_worker():
        outer_server = state.server_module.ArcticRLRayServer
        outer_state = state.server_module.ArcticRLRayServerState
        with optimizer_capture.optimizer_capture_worker():
            assert state.server_module.ArcticRLRayServer is not outer_server
            assert state.server_module.ArcticRLRayServerState is not outer_state
        assert state.server_module.ArcticRLRayServer is outer_server
        assert state.server_module.ArcticRLRayServerState is outer_state
        assert state.worker_module.DeepSpeedWorker is state.capture_worker
    assert state.server_module.ArcticRLRayServer is state.server_class
    assert state.server_module.ArcticRLRayServerState is state.state_class
    assert state.worker_module.DeepSpeedWorker is state.production_worker


def test_optimizer_capture_peer_rank_zero_capture_is_published_before_step_response(_optimizer_capture_capture_server):
    state = _optimizer_capture_capture_server
    assert state.manifest not in state.files["head"]
    result = asyncio.run(state.actor.step(7, {"learning_rate": 0.001}))
    assert result is state.response
    assert state.files["head"] == state.files["peer"]
    assert state.events[0] == ("step", 7, {"learning_rate": 0.001})
    assert state.events[1] == ("export", "peer", "/data-fast/tmp/capture", ["head"])
    assert sorted((event[1] for event in state.events if event[0] == "require")) == ["head", "peer"]
    assert sum((event[0] == "step" for event in state.events)) == 1


def test_optimizer_capture_local_capture_uses_no_peer_transfer(_optimizer_capture_capture_server):
    state = _optimizer_capture_capture_server
    state.nodes[:] = [state.nodes[1]]
    state.actor.training_workers = state.actor.training_workers[:2]
    result = asyncio.run(state.actor.step(7))
    assert result is state.response
    assert ("export", "peer", "/data-fast/tmp/capture", []) in state.events


def test_optimizer_capture_step_without_optimizer_capture_leaves_product_response_unchanged(
    _optimizer_capture_capture_server,
):
    state = _optimizer_capture_capture_server
    state.response["metrics"] = {"grad_norm": 1.0}
    assert asyncio.run(state.actor.step(7)) is state.response
    assert state.events == [("step", 7, None)]


@pytest.mark.parametrize("failure", ["step", "export", "require"])
def test_optimizer_capture_capture_failures_are_not_successful_step_responses(
    _optimizer_capture_capture_server, failure
):
    state = _optimizer_capture_capture_server
    state.failures[failure] = True
    with pytest.raises((RuntimeError, FileNotFoundError)):
        asyncio.run(state.actor.step(7))
    assert sum((event[0] == "step" for event in state.events)) == 1


def test_optimizer_capture_workers_outside_live_allocation_are_rejected_before_export(
    _optimizer_capture_capture_server,
):
    state = _optimizer_capture_capture_server
    state.nodes.pop(1)
    with pytest.raises(RuntimeError, match="outside"):
        asyncio.run(state.actor.step(7))
    assert not any((event[0] == "export" for event in state.events))


def test_optimizer_capture_manifest_is_required_in_exported_files(_optimizer_capture_capture_server):
    state = _optimizer_capture_capture_server
    state.files["peer"].pop(state.manifest)
    with pytest.raises(FileNotFoundError, match="manifest"):
        asyncio.run(state.actor.step(7))


@pytest.mark.parametrize("as_string", [False, True])
def test_optimizer_capture_manifest_scalar_and_duplicate_metrics_publish_once(
    _optimizer_capture_capture_server, as_string
):
    state = _optimizer_capture_capture_server
    state.response["metrics"]["optimizer_state_manifest"] = (
        state.manifest if as_string else [state.manifest, state.manifest]
    )
    assert asyncio.run(state.actor.step(7)) is state.response
    assert sum((event[0] == "export" for event in state.events)) == 1


def test_optimizer_capture_empty_capture_worker_list_is_rejected(_optimizer_capture_capture_server):
    state = _optimizer_capture_capture_server
    state.actor.training_workers = []
    with pytest.raises(RuntimeError, match="no training workers"):
        asyncio.run(state.actor.step(7))
    assert not any((event[0] == "export" for event in state.events))


def test_optimizer_capture_relative_manifest_is_rejected_before_publication(_optimizer_capture_capture_server):
    state = _optimizer_capture_capture_server
    state.response["metrics"]["optimizer_state_manifest"] = ["manifest.json"]
    with pytest.raises(ValueError, match="absolute"):
        asyncio.run(state.actor.step(7))
    assert not any((event[0] == "export" for event in state.events))


def test_optimizer_capture_head_without_training_workers_still_receives_capture(_optimizer_capture_capture_server):
    state = _optimizer_capture_capture_server
    state.actor.training_workers = state.actor.training_workers[:2]
    asyncio.run(state.actor.step(7))
    assert state.files["head"] == state.files["peer"]
    assert ("export", "peer", "/data-fast/tmp/capture", ["head"]) in state.events


@pytest.mark.parametrize("fail_context", [False, True])
def test_optimizer_capture_preloaded_compatibility_server_publishes_and_restores(
    _optimizer_capture_capture_server, monkeypatch, fail_context
):
    state = _optimizer_capture_capture_server
    compatibility = types.ModuleType("arctic_platform.rl.ray_server")
    compatibility.DeepSpeedWorker = state.production_worker
    compatibility.ArcticRLRayServer = state.server_class
    compatibility.ArcticRLRayServerState = state.state_class
    monkeypatch.setitem(sys.modules, compatibility.__name__, compatibility)
    expected = pytest.raises(RuntimeError, match="context failed") if fail_context else contextlib.nullcontext()
    with expected:
        with optimizer_capture.optimizer_capture_worker():
            assert compatibility.ArcticRLRayServer is state.server_module.ArcticRLRayServer
            assert compatibility.ArcticRLRayServerState is state.server_module.ArcticRLRayServerState
            server = compatibility.ArcticRLRayServer()
            server.training_workers = state.actor.training_workers
            assert asyncio.run(server.step(7)) is state.response
            assert state.files["head"] == state.files["peer"]
            if fail_context:
                raise RuntimeError("context failed")
    assert compatibility.ArcticRLRayServer is state.server_class
    assert compatibility.ArcticRLRayServerState is state.state_class
    assert compatibility.DeepSpeedWorker is state.production_worker


def test_optimizer_capture_lazy_compatibility_import_restores_production_classes(
    _optimizer_capture_capture_server, monkeypatch
):
    state = _optimizer_capture_capture_server
    monkeypatch.delitem(sys.modules, "arctic_platform.rl.ray_server")
    with optimizer_capture.optimizer_capture_worker():
        compatibility = sys.modules["arctic_platform.rl.ray_server"]
        assert compatibility.ArcticRLRayServer is state.server_module.ArcticRLRayServer
        assert compatibility.ArcticRLRayServerState is state.server_module.ArcticRLRayServerState
    assert compatibility.ArcticRLRayServer is state.server_class
    assert compatibility.ArcticRLRayServerState is state.state_class
    assert compatibility.DeepSpeedWorker is state.production_worker
    monkeypatch.setitem(sys.modules, compatibility.__name__, state.compatibility_module)


def test_optimizer_capture_nested_context_restores_compatibility_outer_bindings(_optimizer_capture_capture_server):
    state = _optimizer_capture_capture_server
    compatibility = state.compatibility_module
    with optimizer_capture.optimizer_capture_worker():
        outer_server = compatibility.ArcticRLRayServer
        outer_state = compatibility.ArcticRLRayServerState
        with optimizer_capture.optimizer_capture_worker():
            assert compatibility.ArcticRLRayServer is state.server_module.ArcticRLRayServer
            assert compatibility.ArcticRLRayServerState is state.server_module.ArcticRLRayServerState
            assert compatibility.ArcticRLRayServer is not outer_server
        assert compatibility.ArcticRLRayServer is outer_server
        assert compatibility.ArcticRLRayServerState is outer_state
    assert compatibility.ArcticRLRayServer is state.server_class
    assert compatibility.ArcticRLRayServerState is state.state_class


# Local cortex optimizer.


@pytest.fixture(autouse=False)
def _local_cortex_optimizer_no_network(monkeypatch):

    def forbidden(*args, **kwargs):
        pytest.fail("no network allowed")

    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(aiohttp.ClientSession, "request", forbidden)
    closes = []
    original_close = CortexTransport._close_session

    async def close(transport):
        assert asyncio.get_running_loop() is transport._loop
        session = transport._session
        await original_close(transport)
        assert transport._session is None
        assert session is None or session.closed
        closes.append(transport)

    monkeypatch.setattr(CortexTransport, "_close_session", close)
    yield closes
    for transport in set(closes):
        transport._loop.call_soon_threadsafe(transport._loop.stop)


@pytest.fixture
def _local_cortex_optimizer_local_capture(tmp_path, monkeypatch, _local_cortex_optimizer_no_network):
    pair = _cortex_optimizer_decode_pair(tmp_path / "source", world_size=4)
    backend = CortexConfig(base_url="http://127.0.0.1:8003")
    session = GatewaySession(tmp_path, 4, backend)
    events, replies, paths = ([], {}, [])
    state = SimpleNamespace(step=0, failure=None, receipt=None, cancelled=False)

    async def send(transport, method, url, *, policy=_DEFAULT, **kwargs):
        assert asyncio.get_running_loop() is transport._loop
        assert isinstance(policy, _SendPolicy)
        assert kwargs["allow_redirects"] is False
        assert transport._session.trust_env is False
        assert "Authorization" not in transport._session.headers
        if url.endswith(":cancel") or (state.cancelled and method == "GET"):
            assert policy.max_retries == 0 and policy.timeout > 0
        if method == "POST" and url == transport._prefix:
            return {"job_id": "local-parent"}
        if method == "GET" and url == f"{transport._prefix}/{transport.job_id}":
            return {
                "job_id": "local-parent",
                "status": "cancelled" if state.cancelled else "RUNNING",
                "sub_jobs": [
                    {
                        "job_type": "training",
                        "sub_job_id": "local-parent:training:0",
                        "status": "cancelled" if state.cancelled else "running",
                    }
                ],
            }
        if method == "GET" and "/requests/" in url:
            return {"status": "REQUEST_STATE_DONE", "result": replies[url.rsplit("/", 1)[-1]]}
        if url.endswith(":cancel"):
            state.cancelled = True
            return {}
        if url.endswith("/save"):
            body = kwargs["json"]
            assert body == {"path": body["path"], "checkpoint_type": "resumable", "export_hf": False}
            path = Path(body["path"])
            assert path.is_absolute() and path.is_relative_to(tmp_path / "output")
            assert not path.exists()
            paths.append(path)
            events.append(f"save-{state.step}")
            shutil.copytree(pair.before if state.step == 0 else pair.after, path)
            result = {
                "job_id": "actual-training-child",
                "checkpoint_path": str(path),
                "checkpoint_tag": f"global_step{state.step}",
                "version": state.step,
                "stage_path": "",
            }
            if state.receipt is not None:
                result = state.receipt(result, path)
            if state.failure == events[-1]:
                raise RuntimeError("injected save failure")
        elif url.endswith("/forward-backward"):
            body = wire.loads(wire.decode_byte_chunks([kwargs["data"]], kind="request"))
            assert set(body) == {"input_ids"}
            torch.testing.assert_close(body["input_ids"], torch.tensor([[1, 2]]), rtol=0, atol=0)
            events.append("fwd_bwd")
            result = {"avg_loss": 2.0}
        elif url.endswith("/step"):
            assert kwargs["json"] == {"learning_rate": 0.01}
            events.append("step")
            state.step += 1
            if state.failure == "step":
                raise RuntimeError("injected step failure")
            result = {"gradient_norms_per_param": {"weight": 1.0}}
        else:
            pytest.fail(f"local capture attempted unsupported catalog/export route: {method} {url}")
        request_id = f"request-{len(replies)}"
        replies[request_id] = result
        return {"request_id": request_id}

    monkeypatch.setattr(CortexTransport, "_send", send)
    client_config = _training_client_config(
        session, build_payload(load_config(_cortex_optimizer_capture_CONFIG).training, "local-fixture", 42)
    )
    client = CorrectnessClient(client_config)
    yield SimpleNamespace(
        session=session,
        job=ArcticJob(client, "training"),
        state=state,
        events=events,
        paths=paths,
        output=tmp_path / "output",
        pair=pair,
    )
    client.shutdown()
    assert state.cancelled
    assert len(_local_cortex_optimizer_no_network) == 1 and _local_cortex_optimizer_no_network[0].job_id is None


def _local_cortex_optimizer_capture(case):
    return capture.capture_optimizer_step(
        case.session,
        case.job,
        {"input_ids": torch.tensor([[1, 2]])},
        learning_rate=0.01,
        output_dir=case.output,
        expected_dp_world_size=4,
        optimizer_config=_cortex_optimizer_capture_OPTIMIZER,
        local_checkpoint=True,
    )


@pytest.mark.usefixtures("_local_cortex_optimizer_no_network")
def test_local_cortex_optimizer_local_save_wire_and_real_dense_decoder_bracket_one_step(
    _local_cortex_optimizer_local_capture,
):
    result = _local_cortex_optimizer_capture(_local_cortex_optimizer_local_capture)
    assert _local_cortex_optimizer_local_capture.events == ["save-0", "fwd_bwd", "step", "save-1"]
    assert result.avg_loss == 2.0 and result.grad_norms == {"weight": 1.0}
    manifest = Path(result.optimizer_state_manifest)
    assert manifest == _local_cortex_optimizer_local_capture.output / "states" / "manifest.json"
    payload = json.loads(manifest.read_text())
    assert payload["step"] == 1
    for entry in payload["parameters"]:
        values = torch.load(manifest.parent / entry["file"], weights_only=True)
        for key in ("parameter_update", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                values[key], _local_cortex_optimizer_local_capture.pair.expected[entry["name"]][key], rtol=0, atol=0
            )
    assert all((not path.exists() for path in _local_cortex_optimizer_local_capture.paths))
    assert not list(_local_cortex_optimizer_local_capture.output.glob("snapshots-*"))


@pytest.mark.usefixtures("_local_cortex_optimizer_no_network")
@pytest.mark.parametrize("failure", ["save-0", "step", "save-1"])
def test_local_cortex_optimizer_local_failure_cleans_only_owned_snapshots(
    _local_cortex_optimizer_local_capture, failure
):
    _local_cortex_optimizer_local_capture.state.failure = failure
    with pytest.raises(RuntimeError, match="injected"):
        _local_cortex_optimizer_capture(_local_cortex_optimizer_local_capture)
    assert not list(_local_cortex_optimizer_local_capture.output.iterdir())
    assert (
        _local_cortex_optimizer_local_capture.pair.before.is_dir()
        and _local_cortex_optimizer_local_capture.pair.after.is_dir()
    )


@pytest.mark.usefixtures("_local_cortex_optimizer_no_network")
def test_local_cortex_optimizer_invalid_after_save_never_publishes_partial_artifact(
    _local_cortex_optimizer_local_capture,
):

    def receipt(result, root):
        if _local_cortex_optimizer_local_capture.state.step:
            result["version"] = 2
        return result

    _local_cortex_optimizer_local_capture.state.receipt = receipt
    with pytest.raises(ValueError, match="receipt"):
        _local_cortex_optimizer_capture(_local_cortex_optimizer_local_capture)
    assert _local_cortex_optimizer_local_capture.events == ["save-0", "fwd_bwd", "step", "save-1"]
    assert not list(_local_cortex_optimizer_local_capture.output.iterdir())


@pytest.mark.usefixtures("_local_cortex_optimizer_no_network")
def test_local_cortex_optimizer_local_decoder_failure_preserves_unrelated_data_and_cleans_owned_snapshots(
    _local_cortex_optimizer_local_capture, monkeypatch
):
    from arctic_platform.correctness.harness import cortex_optimizer as decoder

    def fail(before, after, output, **kwargs):
        output.mkdir()
        (output / "partial.pt").write_bytes(b"not publishable")
        raise ValueError("injected invalid optimizer schema")

    monkeypatch.setattr(decoder, "decode_optimizer_capture", fail)
    with pytest.raises(ValueError, match="schema"):
        _local_cortex_optimizer_capture(_local_cortex_optimizer_local_capture)
    assert not list(_local_cortex_optimizer_local_capture.output.iterdir())
    assert (
        _local_cortex_optimizer_local_capture.pair.before.is_dir()
        and _local_cortex_optimizer_local_capture.pair.after.is_dir()
    )


@pytest.mark.usefixtures("_local_cortex_optimizer_no_network")
def test_local_cortex_optimizer_existing_local_artifact_never_overwritten_or_stepped(
    _local_cortex_optimizer_local_capture,
):
    path = _cortex_optimizer_capture_manifest(_local_cortex_optimizer_local_capture.output / "states")
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        _local_cortex_optimizer_capture(_local_cortex_optimizer_local_capture)
    assert path.read_bytes() == original
    assert _local_cortex_optimizer_local_capture.events == []


@pytest.mark.usefixtures("_local_cortex_optimizer_no_network")
@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_path", "/data-fast/unrelated"),
        ("checkpoint_path", "../elsewhere"),
        ("checkpoint_path", None),
        ("checkpoint_tag", "../global_step0"),
        ("checkpoint_tag", "global_step1"),
        ("version", True),
        ("version", 1),
        ("stage_path", "@stage/remote"),
    ],
)
def test_local_cortex_optimizer_local_receipt_mismatch_rejected_before_step(
    _local_cortex_optimizer_local_capture, field, value
):
    _local_cortex_optimizer_local_capture.state.receipt = lambda result, path: {**result, field: value}
    with pytest.raises(ValueError):
        _local_cortex_optimizer_capture(_local_cortex_optimizer_local_capture)
    assert _local_cortex_optimizer_local_capture.events == ["save-0"]
    assert not list(_local_cortex_optimizer_local_capture.output.iterdir())
    assert _local_cortex_optimizer_local_capture.pair.before.is_dir()


@pytest.mark.usefixtures("_local_cortex_optimizer_no_network")
@pytest.mark.parametrize("bad", ["missing-rank", "extra-rank", "symlink", "empty", "wrong-tag"])
def test_local_cortex_optimizer_incomplete_local_checkpoint_never_reaches_step(
    _local_cortex_optimizer_local_capture, bad
):

    def damage(result, root):
        tag = root / "global_step0"
        shard = next(tag.glob("*rank_0_*optim_states.pt"))
        if bad == "missing-rank":
            shard.unlink()
        elif bad == "extra-rank":
            shutil.copyfile(shard, tag / "bf16_zero_pp_rank_4_mp_rank_00_optim_states.pt")
        elif bad == "symlink":
            shard.unlink()
            shard.symlink_to(
                next((_local_cortex_optimizer_local_capture.pair.before / "global_step0").glob("*optim_states.pt"))
            )
        elif bad == "empty":
            shard.write_bytes(b"")
        else:
            tag.rename(root / "global_step99")
        return result

    _local_cortex_optimizer_local_capture.state.receipt = damage
    with pytest.raises(ValueError):
        _local_cortex_optimizer_capture(_local_cortex_optimizer_local_capture)
    assert _local_cortex_optimizer_local_capture.events == ["save-0"]
    assert not list(_local_cortex_optimizer_local_capture.output.iterdir())


@pytest.mark.usefixtures("_local_cortex_optimizer_no_network")
@pytest.mark.parametrize(
    "url,credentials",
    [
        ("http://remote.example:8003", {}),
        ("https://127.0.0.1:8003", {}),
        ("http://127.0.0.1:8003/proxy", {}),
        ("http://127.0.0.1:8003?target=remote", {}),
        ("http://user:secret@127.0.0.1:8003", {}),
        ("http://127.0.0.1:8003", {"pat": "not-a-real-token"}),
        ("http://127.0.0.1:8003", {"host": "remote.example"}),
    ],
)
def test_local_cortex_optimizer_local_checkpoint_flag_rejects_nonloopback_or_auth_before_writes(
    _local_cortex_optimizer_local_capture, monkeypatch, url, credentials
):
    _local_cortex_optimizer_local_capture.session = GatewaySession(
        _local_cortex_optimizer_local_capture.output, 4, CortexConfig(base_url=url, **credentials)
    )
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: pytest.fail("unsafe backend reached mkdir"))
    with pytest.raises(ValueError, match="loopback"):
        _local_cortex_optimizer_capture(_local_cortex_optimizer_local_capture)
    assert _local_cortex_optimizer_local_capture.events == []


@pytest.mark.usefixtures("_local_cortex_optimizer_no_network")
@pytest.mark.parametrize("order", ["batched", "interleaved"])
def test_local_cortex_optimizer_local_fixture_optimizer_runner_uses_same_model_and_frozen_budget(
    tmp_path, monkeypatch, _local_cortex_optimizer_no_network, order
):
    cfg, spec = (load_config(_cortex_optimizer_capture_CONFIG), Spec.read(_cortex_optimizer_capture_SPEC))
    original = (_cortex_optimizer_capture_CONFIG.read_bytes(), _cortex_optimizer_capture_SPEC.read_bytes())
    captures, creates = ([], [])
    cancelled = set()
    references = _cortex_optimizer_capture_prepared_references(monkeypatch, spec, cfg, tmp_path)
    model = str(tmp_path / "reviewed-fixture")
    monkeypatch.setattr(
        cortex_fixture, "load_local_fixture", lambda *args: {"model_name": model, "distribution": "local-gateway"}
    )
    monkeypatch.setattr(runner, "uses_mixer_packing", lambda *args: False)

    def local_step(session, job, body, **kwargs):
        assert kwargs["local_checkpoint"] is True
        assert kwargs["expected_dp_world_size"] == 4
        assert kwargs["optimizer_config"] == cfg.training["optimizer"]
        captures.append(kwargs)
        return StepResult(
            2.0, {}, optimizer_state_manifest=str(_cortex_optimizer_capture_manifest(kwargs["output_dir"] / "states"))
        )

    async def send(transport, method, url, *, policy=_DEFAULT, **kwargs):
        assert asyncio.get_running_loop() is transport._loop
        assert isinstance(policy, _SendPolicy)
        assert kwargs["allow_redirects"] is False
        assert transport._session.trust_env is False
        assert "Authorization" not in transport._session.headers
        if url.endswith(":cancel") or (method == "GET" and transport.job_id in cancelled):
            assert policy.max_retries == 0 and policy.timeout > 0
        if method == "POST" and url == transport._prefix:
            creates.append(kwargs["json"])
            return {"job_id": f"parent-{len(creates)}"}
        if method == "GET":
            return {
                "job_id": transport.job_id,
                "status": "cancelled" if transport.job_id in cancelled else "RUNNING",
                "sub_jobs": [
                    {
                        "job_type": "training",
                        "sub_job_id": f"{transport.job_id}:training:0",
                        "status": "cancelled" if transport.job_id in cancelled else "running",
                    }
                ],
            }
        if url.endswith(":cancel"):
            cancelled.add(transport.job_id)
            return {}
        pytest.fail("runner should call local capture, no catalog/export or native worker hooks")

    monkeypatch.setattr(capture, "capture_optimizer_step", local_step)
    monkeypatch.setattr(CortexTransport, "_send", send)
    results = runner.execute(
        cfg,
        spec,
        [ArmDefinition("gas1", 4, 32), ArmDefinition("gas4", 16, 32)],
        vocab_size=128,
        select=["single-step-optimizer"],
        order=order,
        client_backend=CortexConfig(base_url="http://127.0.0.1:8003"),
        local_cortex_fixture=True,
        workdir=tmp_path,
    )
    assert [result.arm for result in results] == ["gas1", "gas4"]
    assert all((result.outcome is Outcome.PASS and result.gpus == 4 for result in results))
    assert all((result.metrics["stated_criterion_abs"] == 0.001 for result in results))
    assert references == [("single-step-optimizer", "gas1"), ("single-step-optimizer", "gas4")]
    assert len(references) == len(captures) == len(creates) == 2
    assert cancelled == {f"parent-{index}" for index in (1, 2)}
    assert len(_local_cortex_optimizer_no_network) == 2 and all(
        (transport.job_id is None for transport in _local_cortex_optimizer_no_network)
    )
    for create in creates:
        child = create["sub_job_configs"][0]
        assert child["model_name"] == model and child["training_config"]["n_gpus"] == 4
        assert "optimizer_state_output_dir" not in child["training_config"]["debug"]
    assert (_cortex_optimizer_capture_CONFIG.read_bytes(), _cortex_optimizer_capture_SPEC.read_bytes()) == original


@pytest.mark.usefixtures("_local_cortex_optimizer_no_network")
@pytest.mark.parametrize("tests", [["single-step-optimizer"], ["single-step-grads", "single-step-optimizer"]])
def test_local_cortex_optimizer_cli_local_fixture_accepts_optimizer_selection_before_dispatch(monkeypatch, tests):
    args = SimpleNamespace(
        local_cortex_fixture=True,
        cortex_fixture=None,
        config=str(_cortex_optimizer_capture_CONFIG),
        all_configs=False,
        test=tests,
    )
    monkeypatch.setattr(cli, "_client_backend", lambda args: CortexConfig(base_url="http://127.0.0.1:8003"))

    def stop(*args, **kwargs):
        raise RuntimeError("eligible dispatch")

    monkeypatch.setattr(cli, "_resolve_configs", stop)
    with pytest.raises(RuntimeError, match="eligible dispatch"):
        cli.cmd_run(args)


@pytest.mark.usefixtures("_local_cortex_optimizer_no_network")
@pytest.mark.parametrize(
    "selected", [None, ["e2e-sft-train-validate"], ["single-step-optimizer", "inference-checkpoint-loss"]]
)
def test_local_cortex_optimizer_cli_local_fixture_does_not_implicitly_enable_other_checks(monkeypatch, selected):
    args = SimpleNamespace(
        local_cortex_fixture=True,
        cortex_fixture=None,
        config=str(_cortex_optimizer_capture_CONFIG),
        all_configs=False,
        test=selected,
    )
    monkeypatch.setattr(cli, "_client_backend", lambda args: CortexConfig(base_url="http://127.0.0.1:8003"))
    monkeypatch.setattr(cli, "_resolve_configs", lambda *args, **kwargs: pytest.fail("invalid selection dispatched"))
    with pytest.raises(SystemExit, match="single-step"):
        cli.cmd_run(args)


@pytest.mark.usefixtures("_local_cortex_optimizer_no_network")
def test_local_cortex_optimizer_runner_rejects_different_frozen_optimizer_reference_before_batch_or_job(
    tmp_path, monkeypatch
):
    cfg, spec = (load_config(_cortex_optimizer_capture_CONFIG), Spec.read(_cortex_optimizer_capture_SPEC))
    spec.test_settings["single-step-optimizer"]["model_cache_path"] = str(tmp_path / "different-model")
    monkeypatch.setattr(cortex_fixture, "load_local_fixture", lambda *args: {"model_name": spec.model.cache_path})
    monkeypatch.setattr(
        runner, "build_batch", lambda *args, **kwargs: pytest.fail("mismatch reached batch allocation")
    )
    with pytest.raises(ValueError, match="frozen optimizer reference"):
        runner.execute(
            cfg,
            spec,
            [],
            vocab_size=128,
            select=["single-step-optimizer"],
            client_backend=CortexConfig(base_url="http://127.0.0.1:8003"),
            local_cortex_fixture=True,
            workdir=tmp_path,
        )


@pytest.mark.usefixtures("_local_cortex_optimizer_no_network")
def test_local_cortex_optimizer_cli_local_fixture_never_expands_declared_four_gpu_width(tmp_path, monkeypatch):
    cfg, spec = (load_config(_cortex_optimizer_capture_CONFIG), Spec.read(_cortex_optimizer_capture_SPEC))
    args = SimpleNamespace(
        local_cortex_fixture=True,
        cortex_fixture=None,
        config=str(_cortex_optimizer_capture_CONFIG),
        all_configs=False,
        test=["single-step-optimizer"],
        gpu_pool=8,
        attn=None,
        case=None,
        order="batched",
        fail_fast=False,
        out=str(tmp_path / "report"),
    )
    backend = CortexConfig(base_url="http://127.0.0.1:8003")
    widths = []
    monkeypatch.setattr(cli, "_client_backend", lambda args: backend)
    monkeypatch.setattr(cli, "_resolve_configs", lambda *args, **kwargs: [_cortex_optimizer_capture_CONFIG])
    monkeypatch.setattr(cli, "_frozen_spec", lambda *args, **kwargs: (spec, [ArmDefinition("gas1", 4, 32)]))
    monkeypatch.setattr(cli, "_run_vocab_size", lambda *args, **kwargs: 128)
    monkeypatch.setattr(
        cortex_fixture,
        "load_local_fixture",
        lambda *args: {
            "model_name": spec.model.cache_path,
            "fixture_sha256": "local-byte-binding",
            "distribution": "local-gateway",
        },
    )

    def execute(run_cfg, run_spec, arms, **kwargs):
        assert kwargs["local_cortex_fixture"] is True and kwargs["client_backend"] is backend
        assert run_spec is spec
        assert arms[0].global_batch_size == 4
        widths.append(run_cfg.n_gpus)
        return []

    monkeypatch.setattr(cli, "execute", execute)
    monkeypatch.setattr(cli, "_finish", lambda *args, **kwargs: 0)
    assert cli.cmd_run(args) == 0
    assert widths == [cfg.n_gpus] == [4]


def _native_ep_bf16_pair(tmp_path, *, world=8, ep=8, train_experts=True):
    config = {
        "zero_optimization": {"stage": 2},
        "bf16": {
            "enabled": True,
            "bf16_master_weights_and_grads": True,
            "bf16_optimizer_states": True,
        },
        "optimizer": {"type": "AdamW"},
    }
    # Deliberately reversed EP ownership; the decoder must use the supplied actual order.
    routes = [[ep - 1 - rank + replica * ep for replica in range(world // ep)] for rank in range(ep)]
    expert_name = "model.layers.0.mlp.experts.w2"
    shapes = [OrderedDict([("dense.weight", torch.Size([3]))])]
    if train_experts:
        shapes.append(OrderedDict([(expert_name, torch.Size([1, 5]))]))
    sources = {}
    for role in sorted(decoder._NATIVE_SOURCE_ROLES):
        path = tmp_path / f"{role}.py"
        path.write_text(f"# synthetic producer descriptor: {role}\n")
        sources[role] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    layout = {
        "schema": "native-ep-zero2-layout-v1",
        "world_size": world,
        "ep_size": ep,
        "sp_size": ep,
        "model_mp_world_size": ep,
        "model_ep_rank": ep - 1,
        "optimizer_mp_ranks": [rank % ep for rank in range(world)],
        "expert_data_parallel_groups": {"native-experts": routes} if train_experts else {},
        "ds_config": config,
        "source_files": sources,
    }
    expected = {}
    roots = [tmp_path / "before", tmp_path / "after"]
    for root, step in zip(roots, (0, 1)):
        tag = root / f"global_step{step}"
        tag.mkdir(parents=True)
        dense = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16) + step * 0.25
        expert_values = [torch.full((1, 5), 4.0 + rank + step * 0.25, dtype=torch.bfloat16) for rank in range(ep)]
        module = {"dense.weight": dense, expert_name: expert_values[layout["model_ep_rank"]]}
        torch.save(
            {
                "module": module,
                "param_shapes": shapes,
                "global_steps": step,
                "dp_world_size": world,
                "mp_world_size": ep,
                "ds_config": config,
            },
            tag / "mp_rank_00_model_states.pt",
        )
        for rank, value in enumerate(expert_values):
            torch.save({expert_name: value}, tag / f"experts_ep_rank_{rank:03d}.pt")
        for rank in range(world):
            masters, mappings, paddings, states, groups, counts = [], [], [], {}, [], []
            for index, shape_group in enumerate(shapes):
                route = list(range(world)) if index == 0 else next(route for route in routes if rank in route)
                partition_rank = route.index(rank)
                ep_rank = next(position for position, route in enumerate(routes) if rank in route)
                name, shape = next(iter(shape_group.items()))
                values = dense if index == 0 else expert_values[ep_rank]
                length = math.ceil(values.numel() / (2 * len(route))) * 2
                start = partition_rank * length
                valid = max(0, min(length, values.numel() - start))
                masters.append(values.flatten()[start : start + valid].clone())
                paddings.append(length - valid)
                mappings.append({name: fragment_address(start=0, numel=valid)} if valid else {})
                counts.append(len(route))
                group = {
                    "params": [index],
                    "lr": 0.01,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                    "bias_correction": True,
                    "moe": index != 0,
                }
                if index:
                    group["name"] = "native-experts"
                groups.append(group)
                if step:
                    vector = torch.zeros(length, dtype=torch.bfloat16)
                    vector[:valid] = 1 if index == 0 else ep_rank + 2
                    states[index] = {"step": step, "exp_avg": vector, "exp_avg_sq": vector * 2}
            torch.save(
                {
                    "ds_config": config,
                    "optimizer_state_dict": {
                        "zero_stage": ZeroStageEnum.gradients,
                        "base_optimizer_state": {"state": states, "param_groups": groups},
                        "single_partition_of_fp32_groups": masters,
                        "param_slice_mappings": mappings,
                        "partition_count": counts,
                        "group_paddings": paddings,
                    },
                },
                tag / f"bf16_zero_pp_rank_{rank}_mp_rank_{layout['optimizer_mp_ranks'][rank]:02d}_optim_states.pt",
            )
    expected["dense.weight"] = torch.full((3,), 0.25)
    if train_experts:
        expected[expert_name] = torch.full((ep, 5), 0.25)
    return SimpleNamespace(before=roots[0], after=roots[1], layout=layout, expected=expected, expert_name=expert_name)


def _maintained_native_save_rows(pair, root, step):
    rows = []
    world = pair.layout["world_size"]
    routes = pair.layout["expert_data_parallel_groups"]["native-experts"]
    source = pair.before if step == 0 else pair.after
    for rank in range(world):
        ep_rank = next(index for index, route in enumerate(routes) if rank in route)
        edp_rank = routes[ep_rank].index(rank)
        optimizer = f"bf16_zero_pp_rank_{rank}_mp_rank_{pair.layout['optimizer_mp_ranks'][rank]:02d}_optim_states.pt"
        names = [optimizer]
        if rank == 0:
            names.append("mp_rank_00_model_states.pt")
        if edp_rank == 0:
            names.append(f"experts_ep_rank_{ep_rank:03d}.pt")
        files = [source / f"global_step{step}" / name for name in names]
        rows.append(
            {
                "schema": "native-ep-zero2-worker-save-v1",
                "parent_job_id": "actual-parent",
                "training_child_id": "actual-child",
                "step": step,
                "tag": f"global_step{step}",
                "checkpoint_root": str(root),
                "nonce": f"{rank:032x}",
                "written_files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
                "written_file_sizes": {path.name: path.stat().st_size for path in files},
                "layout": {
                    "checkpoint_rank": rank,
                    "global_rank": rank,
                    "dp_order": list(range(world)),
                    "world_size": world,
                    "ep_size": pair.layout["ep_size"],
                    "sp_size": pair.layout["sp_size"],
                    "model_mp_world_size": pair.layout["model_mp_world_size"],
                    "ep_rank": ep_rank,
                    "edp_rank": edp_rank,
                    "optimizer_mp_rank": pair.layout["optimizer_mp_ranks"][rank],
                    "optimizer_filename": optimizer,
                    "source_files": pair.layout["source_files"],
                    "ds_config": pair.layout["ds_config"],
                    "groups": [
                        {"name": None, "moe": False, "ranks": list(range(world)), "ep_rank": None},
                        {"name": "native-experts", "moe": True, "ranks": routes[ep_rank], "ep_rank": ep_rank},
                    ],
                },
            }
        )
    return rows


@pytest.fixture
def _maintained_native_capture(tmp_path, monkeypatch):
    (tmp_path / "source").mkdir()
    pair = _native_ep_bf16_pair(tmp_path / "source")
    events, saved = [], []
    state = SimpleNamespace(step=0, failure=None, damage=None)
    backend = CortexConfig(base_url="http://127.0.0.1:8003")

    class Client:
        def save_checkpoint(self, *, path, checkpoint_type, export_hf):
            assert checkpoint_type == "resumable" and export_hf is False
            root = Path(path)
            assert not root.exists()
            tag = f"global_step{state.step}"
            rows = _maintained_native_save_rows(pair, root, state.step)
            (root / tag).mkdir(parents=True)
            source = pair.before if state.step == 0 else pair.after
            # Head SAVE contains only rank0 files; remaining actual fixture bytes require consolidation.
            for name in rows[0]["written_files"]:
                shutil.copyfile(source / tag / name, root / tag / name)
            events.append(f"save-{state.step}")
            saved.append(root)
            if state.failure == events[-1]:
                raise RuntimeError("injected completed-save failure")
            return {
                "checkpoint_path": path,
                "checkpoint_tag": tag,
                "version": 2 if state.failure == "save-metadata" else state.step,
                "stage_path": "",
            }

        def fwd_bwd(self, body):
            torch.testing.assert_close(body["input_ids"], torch.tensor([[1, 2]]), rtol=0, atol=0)
            events.append("fwd_bwd")
            if state.failure == "fwd_bwd":
                raise RuntimeError("injected target failure")
            return {"avg_loss": 2.0, "metrics": {"sft_model_calls": 1, "sft_packed_rows": 1}}

        def step(self, learning_rate):
            assert learning_rate == 0.01
            events.append("step")
            if state.failure == "step":
                raise RuntimeError("injected optimizer failure")
            state.step += 1
            return {"gradient_norms_per_param": {"dense.weight": 1.0}}

        def shutdown(self):
            events.append("release")

    client = Client()
    client.config = SimpleNamespace(backend=backend)
    client.transport = SimpleNamespace(job_id="actual-parent")
    client.jobs = SimpleNamespace(training="actual-parent:train:0")
    descriptor = {"path": __file__, "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def supplier(root, *, step):
        assert saved[-1] == root and state.step == step
        events.append(f"supply-{step}")
        source = pair.before if step == 0 else pair.after
        rows = _maintained_native_save_rows(pair, root, step)
        if state.damage is not None:
            state.damage(rows, root, step)

        def read_worker(rank, filename, *, offset, length):
            row = rows[rank]
            assert filename in row["written_files"]
            events.append(f"read-{step}-{rank}")
            data = (source / f"global_step{step}" / filename).read_bytes()
            return {
                "offset": offset,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": data[offset : offset + length],
            }

        decoder.consolidate_native_optimizer_checkpoints(
            root,
            rows,
            step=step,
            read_worker=read_worker,
            read_worker_source=descriptor,
            reserve_bytes=0,
        )
        layout = decoder.assemble_native_optimizer_layout(
            root,
            step=step,
            parent_job_id="actual-parent",
            training_child_id="actual-child",
            expected_world_size=8,
            source_files=pair.layout["source_files"],
        )
        if state.failure == "source-after-supply":
            path = Path(pair.layout["source_files"]["training"]["path"])
            path.write_text(path.read_text() + "# changed during supplier\n")
        return layout

    session = GatewaySession(tmp_path, 8, backend)
    job = ArcticJob(client, "training")
    case = SimpleNamespace(
        pair=pair,
        events=events,
        saved=saved,
        state=state,
        client=client,
        session=session,
        job=job,
        supplier=supplier,
        descriptor=descriptor,
        output=tmp_path / "output",
    )
    return case


def _maintained_native_run(case, **options):
    return capture.capture_optimizer_step(
        case.session,
        case.job,
        {"input_ids": torch.tensor([[1, 2]])},
        learning_rate=0.01,
        output_dir=case.output,
        expected_dp_world_size=8,
        optimizer_config=_cortex_optimizer_capture_OPTIMIZER,
        local_checkpoint=True,
        native_layout_supplier=case.supplier,
        native_layout_supplier_source=case.descriptor,
        **options,
    )


def _maintained_native_unpublished(case):
    assert not (case.output / "states").exists()
    assert not list(case.output.glob("snapshots-*"))
    assert all(not root.exists() for root in case.saved)
    assert case.pair.before.is_dir() and case.pair.after.is_dir()


def test_maintained_native_optimizer_completed_save_supplier_retains_pair_and_actual_child(
    _maintained_native_capture,
):
    case = _maintained_native_capture
    result = _maintained_native_run(case)
    manifest = Path(result.optimizer_state_manifest)
    record = json.loads(manifest.read_text())
    assert manifest == case.output / "states" / "manifest.json"
    assert result.avg_loss == 2 and result.model_calls == result.packed_rows == 1
    assert [event for event in case.events if not event.startswith("read-")] == [
        "save-0",
        "supply-0",
        "fwd_bwd",
        "step",
        "save-1",
        "supply-1",
    ]
    assert any(event.startswith("read-0-") for event in case.events)
    assert any(event.startswith("read-1-") for event in case.events)
    for label, step in (("before", 0), ("after", 1)):
        rows = record["native_save_receipts"][label]
        assert len(rows) == 8
        assert [row["layout"]["checkpoint_rank"] for row in rows] == list(range(8))
        assert all(row["training_child_id"] == "actual-child" and row["step"] == step for row in rows)
        assert len({row["nonce"] for row in rows}) == 8
    for entry in record["parameters"]:
        value = torch.load(manifest.parent / entry["file"], weights_only=True)
        torch.testing.assert_close(value["parameter_update"], case.pair.expected[entry["name"]], rtol=0, atol=0)
    assert not list(case.output.glob("snapshots-*"))
    assert all(not root.exists() for root in case.saved)


@pytest.mark.parametrize("failure", ["save-0", "save-metadata", "fwd_bwd", "step", "save-1", "source-after-supply"])
def test_maintained_native_optimizer_failed_stage_never_publishes_or_uses_unadmitted_baseline(
    _maintained_native_capture, failure
):
    case = _maintained_native_capture
    case.state.failure = failure
    with pytest.raises((RuntimeError, ValueError)):
        _maintained_native_run(case)
    if failure in ("save-0", "save-metadata"):
        assert case.events == ["save-0"]
    if failure in ("save-0", "save-metadata", "source-after-supply"):
        assert "fwd_bwd" not in case.events
    _maintained_native_unpublished(case)


@pytest.mark.parametrize("descriptor", ["missing", "wrong-origin", "bad-hash", "orphan"])
def test_maintained_native_optimizer_requires_selected_supplier_source_before_save(
    _maintained_native_capture, descriptor
):
    case = _maintained_native_capture
    if descriptor == "missing":
        case.descriptor = None
    elif descriptor == "wrong-origin":
        case.descriptor = case.pair.layout["source_files"]["training"]
    elif descriptor == "bad-hash":
        case.descriptor = {**case.descriptor, "sha256": "0" * 64}
    else:
        case.supplier = None
    with pytest.raises(ValueError, match="callback|source"):
        _maintained_native_run(case)
    assert case.events == []
    _maintained_native_unpublished(case)


@pytest.mark.parametrize("damage", ["parent", "child", "nonce", "missing-rank", "rank", "world", "write-hash"])
def test_maintained_native_optimizer_rejects_completed_save_receipt_binding_damage(_maintained_native_capture, damage):
    case = _maintained_native_capture

    def alter(rows, root, step):
        if damage in ("parent", "child"):
            rows[0][f"{damage}_job_id" if damage == "parent" else "training_child_id"] = "wrong"
        elif damage == "nonce":
            rows[1]["nonce"] = rows[0]["nonce"]
        elif damage == "missing-rank":
            rows.pop()
        elif damage == "rank":
            rows[1]["layout"]["checkpoint_rank"] = 0
        elif damage == "world":
            rows[0]["layout"]["world_size"] = 16
        else:
            name = next(iter(rows[0]["written_files"]))
            rows[0]["written_files"][name] = "0" * 64

    case.state.damage = alter
    with pytest.raises(ValueError):
        _maintained_native_run(case)
    assert "fwd_bwd" not in case.events
    _maintained_native_unpublished(case)


def test_maintained_native_optimizer_rejects_nonce_change_between_completed_saves(_maintained_native_capture):
    case = _maintained_native_capture

    def alter(rows, root, step):
        if step == 1:
            rows[0]["nonce"] = "f" * 32

    case.state.damage = alter
    with pytest.raises(ValueError, match="between saves"):
        _maintained_native_run(case)
    assert "step" in case.events
    _maintained_native_unpublished(case)


@pytest.mark.parametrize("damage", ["parent", "selector", "source", "receipt", "tensor"])
def test_maintained_native_optimizer_rechecks_owner_sources_receipts_and_inputs_before_publication(
    _maintained_native_capture, monkeypatch, damage
):
    case = _maintained_native_capture
    original = decoder.decode_optimizer_capture

    def decode(before, after, output, **kwargs):
        manifest = original(before, after, output, **kwargs)
        if damage == "parent":
            case.client.transport.job_id = "wrong"
        elif damage == "selector":
            case.client.jobs.training = "wrong"
        elif damage == "source":
            path = Path(case.pair.layout["source_files"]["training"]["path"])
            path.write_text("# changed after decoder\n")
        elif damage == "receipt":
            path = after / "global_step1" / "native_optimizer_layout" / "rank00000.json"
            record = json.loads(path.read_text())
            record["nonce"] = "f" * 32
            path.write_text(json.dumps(record))
        else:
            with next((after / "global_step1").glob("*optim_states.pt")).open("ab") as stream:
                stream.write(b"changed after decode")
        return manifest

    monkeypatch.setattr(decoder, "decode_optimizer_capture", decode)
    with pytest.raises(ValueError):
        _maintained_native_run(case)
    _maintained_native_unpublished(case)


def test_maintained_native_optimizer_baseline_decoder_rejects_before_target(_maintained_native_capture):
    case = _maintained_native_capture
    path = next((case.pair.before / "global_step0").glob("*optim_states.pt"))
    record = torch.load(path, weights_only=False)
    record["optimizer_state_dict"]["group_paddings"][0] += 1
    torch.save(record, path)
    with pytest.raises(ValueError):
        _maintained_native_run(case)
    assert "fwd_bwd" not in case.events
    _maintained_native_unpublished(case)


@pytest.mark.parametrize("phase", ["baseline", "target", "mutable-descriptor"])
def test_maintained_native_optimizer_supplier_authority_drift_blocks_next_operation(
    _maintained_native_capture, monkeypatch, phase
):
    case = _maintained_native_capture
    path = case.output.parent / "selected_supplier.py"
    text = "def supplier(root, *, step):\n    return delegate(root, step=step)\n"
    path.write_text(text)
    namespace = {"delegate": case.supplier}
    exec(compile(text, str(path), "exec"), namespace)
    case.supplier = namespace["supplier"]
    case.descriptor = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def drift():
        path.write_text(text + "# supplier source changed\n")
        if phase == "mutable-descriptor":
            case.descriptor["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    if phase == "baseline":
        original = decoder._snapshot

        def snapshot(*args, **kwargs):
            result = original(*args, **kwargs)
            drift()
            return result

        monkeypatch.setattr(decoder, "_snapshot", snapshot)
    else:
        original = case.client.step

        def step(learning_rate):
            result = original(learning_rate)
            drift()
            return result

        monkeypatch.setattr(case.client, "step", step)
    with pytest.raises(ValueError, match="source drift"):
        _maintained_native_run(case)
    assert "save-1" not in case.events
    if phase == "baseline":
        assert "fwd_bwd" not in case.events
    else:
        assert "step" in case.events
    _maintained_native_unpublished(case)


def _maintained_native_scope(case, monkeypatch, *, split):
    descriptor = case.descriptor
    helper_source = {
        "path": decoder.__file__,
        "sha256": hashlib.sha256(Path(decoder.__file__).read_bytes()).hexdigest(),
    }
    workers = []

    class Parent:
        pass

    class Child:
        pass

    class Manager:
        def get_job(self, child_id):
            assert child_id == "actual-child"
            return info

    class RemoteCall:
        def __init__(self, owner):
            self.owner = owner

        def remote(self, function, *args, **kwargs):
            try:
                return SimpleNamespace(value=function(self.owner, *args, **kwargs), error=None)
            except BaseException as exc:
                return SimpleNamespace(value=None, error=exc)

    def get(reference, *, timeout):
        assert timeout == 30
        if reference.error is not None:
            raise reference.error
        return reference.value

    def is_initialized():
        return True

    for rank in range(8):
        worker = SimpleNamespace(rank=rank, world_size=8, job_id="actual-child")
        worker.__ray_call__ = RemoteCall(worker)
        worker._actor_id = SimpleNamespace(hex=lambda rank=rank: f"worker-{rank}")
        workers.append(worker)
    actor = SimpleNamespace(_actor_id=SimpleNamespace(hex=lambda: "training-actor"))
    actor.__ray_call__ = RemoteCall(actor)
    info = SimpleNamespace(
        actor=actor,
        job_id="actual-child",
        zone_id="actual-zone",
        layout=[
            SimpleNamespace(host="head", gpu_ids=list(range(4))),
            SimpleNamespace(host="peer", gpu_ids=list(range(4))),
        ],
    )
    snow = ModuleType("dss.ray_dss.gateway_snowapi")
    snow.__file__ = __file__
    snow.ParentRecord, snow.ChildRecord = Parent, Child
    child = Child()
    child.role, child.job_id = "training", "actual-child"
    parent = Parent()
    parent.parent_id, parent.state, parent.cancel_requested = "actual-parent", "running", False
    parent.children = {"training": child}
    snow._parents = {"actual-parent": parent}
    gateway = ModuleType("dss.ray_dss.gateway_server")
    gateway.__file__ = __file__
    gateway._jobs, gateway._zones = {"actual-child": "actual-zone"}, {"actual-zone": object()}
    manager_module = ModuleType("dss.ray_dss.zones.manager")
    manager_module.__file__, manager_module.ZoneManager = __file__, Manager
    manager = Manager()
    manager.zone_type, manager.zone_id = SimpleNamespace(value="training"), "actual-zone"
    ray = ModuleType("ray")
    ray.get, ray.is_initialized = get, is_initialized
    modules = {module.__name__: module for module in (snow, gateway, manager_module, ray)}
    original_import = decoder.importlib.import_module
    monkeypatch.setattr(
        decoder.importlib,
        "import_module",
        lambda name, *args, **kwargs: modules[name] if name in modules else original_import(name, *args, **kwargs),
    )
    monkeypatch.setattr(
        decoder,
        "_native_scope_actor_workers",
        lambda owner, child_id, source: {
            "workers": workers,
            "rank_hosts": ["head"] * 4 + ["peer"] * 4,
            "actor_ids": [worker._actor_id.hex() for worker in workers],
        },
    )

    def arm(worker, *, nonce, rank, world_size, lease, source_files, helper_source):
        assert worker.rank == rank and world_size == 8
        checkpointer = SimpleNamespace()
        hook = object()
        checkpointer.save_resumable = hook
        vars(worker)[decoder._COLLECTOR_STATE] = {
            "nonce": nonce,
            "checkpointer": checkpointer,
            "hook": hook,
            "restore": lambda: vars(checkpointer).pop("save_resumable"),
            "roots": {},
            "checkpoint_lease": lease,
        }
        return nonce

    def receipt(worker, nonce, root, step, helper_source):
        assert Path(root) in case.saved and step == case.state.step
        row = _maintained_native_save_rows(case.pair, Path(root), step)[worker.rank]
        assert vars(worker)[decoder._COLLECTOR_STATE]["nonce"] == nonce
        row["nonce"] = nonce
        return row

    def read(worker, nonce, root, *, step, filename, offset, length):
        assert vars(worker)[decoder._COLLECTOR_STATE]["nonce"] == nonce
        source = case.pair.before if step == 0 else case.pair.after
        data = (source / f"global_step{step}" / filename).read_bytes()
        return {
            "offset": offset,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": data[offset : offset + length],
        }

    monkeypatch.setattr(decoder, "_native_scope_arm_worker", arm)
    monkeypatch.setattr(decoder, "_native_scope_save_receipt", receipt)
    monkeypatch.setattr(decoder, "read_native_optimizer_checkpoint", read)
    request = {
        "schema": "native-optimizer-scope-request-v1",
        "source_files": case.pair.layout["source_files"],
        "route_sources": {
            role: dict(descriptor) for role in ("gateway_snowapi", "gateway", "manager", "ray_get", "ray_actor")
        },
        "helper_source": helper_source,
        "allocation_hosts": ["head", "peer"],
        "checkpoint_parent": str(case.output.parent),
        "reserve_bytes": 0,
        "rpc_timeout_seconds": 30,
    }
    if split:
        request["schema"] = "native-optimizer-owner-split-request-v1"
        request["expected_world_size"] = 8
        request["route_sources"]["ray_is_initialized"] = dict(descriptor)
        request["verify_parent_binding_source"] = dict(descriptor)
        binding = decoder.resolve_native_optimizer_parent(
            request, "actual-parent", gateway_snowapi=snow, gateway=gateway
        )

        def verify_parent_binding(received):
            assert received == binding
            return decoder.resolve_native_optimizer_parent(
                request, "actual-parent", gateway_snowapi=snow, gateway=gateway
            )

        scope = decoder.open_native_optimizer_zone_scope(
            request, binding, manager=manager, ray=ray, verify_parent_binding=verify_parent_binding
        )
    else:
        factory = decoder.make_native_optimizer_scope(
            request, gateway_snowapi=snow, gateway=gateway, manager=manager, ray=ray
        )
        scope = factory(case.session, case.job)
    return SimpleNamespace(scope=scope, workers=workers, child=child, helper_source=helper_source)


@pytest.mark.parametrize("split", [False, True], ids=["colocated", "owner-split"])
@pytest.mark.parametrize("failure", [None, "target", "source-after", "route-after"])
def test_maintained_native_optimizer_scopes_integrate_capture_and_restore_before_release(
    _maintained_native_capture, monkeypatch, split, failure
):
    case = _maintained_native_capture
    fixture = _maintained_native_scope(case, monkeypatch, split=split)
    if failure == "target":
        case.state.failure = "fwd_bwd"

    def run():
        with fixture.scope as supplier:
            case.supplier, case.descriptor = supplier, fixture.helper_source
            if not split:
                with pytest.raises(RuntimeError, match="child release is blocked"):
                    case.client.shutdown()
            result = _maintained_native_run(case)
            assert Path(result.optimizer_state_manifest).is_file()
            if failure == "source-after":
                Path(case.pair.layout["source_files"]["training"]["path"]).write_text("# scope drift\n")
            elif failure == "route-after":
                fixture.child.job_id = "wrong"

    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            run()
    else:
        run()
    assert all(decoder._COLLECTOR_STATE not in vars(worker) for worker in fixture.workers)
    assert all(decoder._CLEANUP_STATE not in vars(worker) for worker in fixture.workers)
    assert decoder._SCOPE_STATE not in vars(case.client)
    assert all(not root.exists() for root in case.saved)
    if split:
        assert fixture.scope.cleanup_complete is True
    case.client.shutdown()
    assert case.events[-1] == "release"
    if failure == "target":
        _maintained_native_unpublished(case)


def _native_ep_bf16_decode(pair, output):
    return decoder.decode_optimizer_capture(
        pair.before,
        pair.after,
        output,
        expected_optimizer_config=_cortex_optimizer_capture_OPTIMIZER,
        expected_learning_rate=0.01,
        expected_native_layout=pair.layout,
    )


@pytest.mark.parametrize("world,ep", [(8, 8), (16, 8), (16, 16)])
def test_native_ep_bf16_optimizer_reconstructs_actual_ordered_groups_without_sidefile_substitution(
    tmp_path, world, ep
):
    pair = _native_ep_bf16_pair(tmp_path, world=world, ep=ep)
    manifest = json.loads(_native_ep_bf16_decode(pair, tmp_path / "out").read_text())
    assert manifest["source_dtypes"] == dict.fromkeys(("master", "exp_avg", "exp_avg_sq"), "bfloat16")
    assert manifest["expert_sidefiles_are_master_sources"] is False
    assert manifest["runtime_provenance_is_caller_owned"] is True
    assert manifest["step"] == 1
    for entry in manifest["parameters"]:
        tensors = torch.load(tmp_path / "out" / entry["file"], weights_only=True)
        assert torch.equal(tensors["parameter_update"], pair.expected[entry["name"]])
        assert list(tensors["parameter_update"].shape) == entry["shape"]
        assert tensors["parameter_update"].dtype == torch.float32
        assert tensors["exp_avg"].dtype == tensors["exp_avg_sq"].dtype == torch.bfloat16
        if entry["name"] == pair.expert_name:
            assert torch.equal(tensors["exp_avg"][:, 0], torch.arange(2, ep + 2, dtype=torch.bfloat16))


def test_native_ep_bf16_optimizer_frozen_experts_do_not_become_optimizer_records(tmp_path):
    pair = _native_ep_bf16_pair(tmp_path, train_experts=False)
    manifest = json.loads(_native_ep_bf16_decode(pair, tmp_path / "out").read_text())
    assert [entry["name"] for entry in manifest["parameters"]] == ["dense.weight"]
    assert manifest["frozen_expert_parameters"] == [pair.expert_name]


@pytest.mark.parametrize("field", ["master", "exp_avg", "exp_avg_sq"])
def test_native_ep_bf16_optimizer_rejects_false_fp32_state_recipe(tmp_path, field):
    pair = _native_ep_bf16_pair(tmp_path)
    path = next(pair.after.rglob("*zero_pp_rank_0_*optim_states.pt"))

    def edit(state):
        optimizer = state["optimizer_state_dict"]
        if field == "master":
            optimizer["single_partition_of_fp32_groups"][1] = optimizer["single_partition_of_fp32_groups"][1].float()
        else:
            optimizer["base_optimizer_state"]["state"][1][field] = optimizer["base_optimizer_state"]["state"][1][
                field
            ].float()

    _cortex_optimizer_decode_rewrite(path, edit)
    with pytest.raises(ValueError, match="BF16 vector"):
        _native_ep_bf16_decode(pair, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", ["optimizer", "expert"])
def test_native_ep_bf16_optimizer_rejects_incomplete_native_inventory(tmp_path, kind):
    pair = _native_ep_bf16_pair(tmp_path)
    pattern = "*zero_pp_rank_7_*optim_states.pt" if kind == "optimizer" else "experts_ep_rank_007.pt"
    path = next(pair.after.rglob(pattern))
    path.rename(tmp_path / "retained-missing.pt")
    with pytest.raises(ValueError, match="every optimizer and expert rank"):
        _native_ep_bf16_decode(pair, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_native_ep_bf16_optimizer_wrong_ep_order_cannot_relabel_valid_weights(tmp_path):
    pair = _native_ep_bf16_pair(tmp_path)
    routes = pair.layout["expert_data_parallel_groups"]["native-experts"]
    routes[0], routes[1] = routes[1], routes[0]
    with pytest.raises(ValueError, match="master/sidefile ownership"):
        _native_ep_bf16_decode(pair, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_native_ep_bf16_optimizer_rejects_wrong_partition_counter_and_mapping(tmp_path):
    pair = _native_ep_bf16_pair(tmp_path)
    path = next(pair.after.rglob("*zero_pp_rank_0_*optim_states.pt"))
    _cortex_optimizer_decode_rewrite(
        path, lambda state: state["optimizer_state_dict"]["partition_count"].__setitem__(1, 8)
    )
    with pytest.raises(ValueError, match="partition_count"):
        _native_ep_bf16_decode(pair, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_native_ep_bf16_optimizer_rejects_inconsistent_adam_step(tmp_path):
    pair = _native_ep_bf16_pair(tmp_path)
    path = next(pair.after.rglob("*zero_pp_rank_0_*optim_states.pt"))
    _cortex_optimizer_decode_rewrite(
        path, lambda state: state["optimizer_state_dict"]["base_optimizer_state"]["state"][1].update(step=2)
    )
    with pytest.raises(ValueError, match="engine/Adam optimizer step"):
        _native_ep_bf16_decode(pair, tmp_path / "out")


@pytest.mark.parametrize("mutation", ["source-before", "source-after", "input-after"])
def test_native_ep_bf16_optimizer_source_and_input_drift_never_publish(tmp_path, monkeypatch, mutation):
    pair = _native_ep_bf16_pair(tmp_path)
    source = Path(pair.layout["source_files"]["zero"]["path"])
    sentinel = tmp_path / "unrelated"
    sentinel.write_text("keep")
    save = torch.save

    def changed_save(value, path, *args, **kwargs):
        save(value, path, *args, **kwargs)
        if Path(path).parent == tmp_path / "out":
            if mutation == "source-after":
                source.write_text("# changed source\n")
            elif mutation == "input-after":
                (pair.after / "added-metadata.json").write_text("{}")

    if mutation == "source-before":
        source.write_text("# changed source\n")
    monkeypatch.setattr(torch, "save", changed_save)
    with pytest.raises(ValueError, match="source drift|input drift"):
        _native_ep_bf16_decode(pair, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("failure", [None, "baseline", "child", "child-after-decode", "source-after-decode"])
def test_native_ep_bf16_optimizer_local_lifecycle_validates_before_step_and_retains_child(
    tmp_path, monkeypatch, failure
):
    pair = _native_ep_bf16_pair(tmp_path)
    backend = CortexConfig(base_url="http://127.0.0.1:8003")
    events = []
    client = SimpleNamespace(
        config=SimpleNamespace(backend=backend),
        jobs=SimpleNamespace(training="same-child"),
        transport=SimpleNamespace(job_id="same-parent"),
    )

    def save_checkpoint(**kwargs):
        step = 0 if not events else 1
        source = pair.before if step == 0 else pair.after
        destination = Path(kwargs["path"])
        assert kwargs == {"path": str(destination), "checkpoint_type": "resumable", "export_hf": False}
        shutil.copytree(source, destination)
        if failure == "baseline" and step == 0:
            model = next(destination.rglob("*_model_states.pt"))
            _cortex_optimizer_decode_rewrite(model, lambda state: state.update(ds_config={}))
        events.append(f"save{step}")
        return {"checkpoint_path": str(destination), "checkpoint_tag": f"global_step{step}", "version": step}

    def step(*args, **kwargs):
        events.append("step")
        if failure == "child":
            client.jobs.training = "other-child"
        return StepResult(2.0, {"dense.weight": 1.0}, model_calls=1, packed_rows=8)

    client.save_checkpoint = save_checkpoint
    monkeypatch.setattr(capture, "fwd_bwd_step", step)
    decode = capture.decode_optimizer_capture

    def changed_decode(*args, **kwargs):
        result = decode(*args, **kwargs)
        if failure == "child-after-decode":
            client.jobs.training = "other-child"
        elif failure == "source-after-decode":
            Path(pair.layout["source_files"]["zero"]["path"]).write_text("# changed source\n")
        return result

    monkeypatch.setattr(capture, "decode_optimizer_capture", changed_decode)
    output = tmp_path / "capture"

    def run():
        return capture.capture_optimizer_step(
            GatewaySession(tmp_path, 8, backend),
            ArcticJob(client, "training"),
            {"original": "batch"},
            learning_rate=0.01,
            output_dir=output,
            expected_dp_world_size=8,
            optimizer_config=_cortex_optimizer_capture_OPTIMIZER,
            local_checkpoint=True,
            expected_native_layout=pair.layout,
        )

    if failure:
        with pytest.raises(ValueError, match="selected ds_config|training child changed|source drift"):
            run()
        assert not (output / "states").exists()
        if failure == "baseline":
            assert events == ["save0"]
    else:
        result = run()
        assert events == ["save0", "step", "save1"]
        assert result.model_calls == 1 and result.packed_rows == 8
        assert Path(result.optimizer_state_manifest).is_file()
    assert not list(output.glob("snapshots-*"))


@pytest.mark.parametrize("mutation", ["duplicate-route", "missing-route", "mp-filename", "slice-mapping"])
def test_native_ep_bf16_optimizer_new_layout_negatives_cannot_guess_ownership(tmp_path, mutation):
    pair = _native_ep_bf16_pair(tmp_path)
    routes = pair.layout["expert_data_parallel_groups"]["native-experts"]
    if mutation == "duplicate-route":
        routes[0] = list(routes[1])
    elif mutation == "missing-route":
        pair.layout["expert_data_parallel_groups"] = {}
    elif mutation == "mp-filename":
        pair.layout["optimizer_mp_ranks"][0] = 1
    else:
        path = next(pair.after.rglob("*zero_pp_rank_0_*optim_states.pt"))
        _cortex_optimizer_decode_rewrite(
            path,
            lambda state: state["optimizer_state_dict"]["param_slice_mappings"][1].update(
                {pair.expert_name: fragment_address(start=1, numel=5)}
            ),
        )
    with pytest.raises(ValueError, match="cover every rank|actual ordered DP route|MP filename|slice mapping"):
        _native_ep_bf16_decode(pair, tmp_path / "out")
    assert not (tmp_path / "out").exists()
