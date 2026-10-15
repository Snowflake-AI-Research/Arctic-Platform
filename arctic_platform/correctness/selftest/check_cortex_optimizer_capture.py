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

"""CPU lifecycle and public-wire tests for managed one-step optimizer capture."""

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest
import requests
import torch

from arctic_platform import wire
from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.correctness.harness import cortex_fixture
from arctic_platform.correctness.harness import cortex_optimizer_capture as capture
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness.arms import ArmDefinition
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.dss_driver import ArcticJob
from arctic_platform.correctness.harness.dss_driver import GatewaySession
from arctic_platform.correctness.harness.dss_driver import StepResult
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.spec import TestSpec as Spec

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config"
SPEC = ROOT / "specs/qwen3-8b-h200-train-sft-full-4gpus-64k.json"
OPTIMIZER = {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(
        requests.Session, "request", lambda *args, **kwargs: pytest.fail("mock tests must not access network")
    )


def _manifest(root, *, step=1):
    root.mkdir(parents=True)
    filename = "000000.pt"
    torch.save({key: torch.zeros(2) for key in ("parameter_update", "exp_avg", "exp_avg_sq")}, root / filename)
    path = root / "manifest.json"
    path.write_text(json.dumps({"step": step, "parameters": [{"name": "weight", "file": filename}]}))
    return path


@pytest.fixture
def artifact_stubs(monkeypatch):
    export = ModuleType("arctic_platform.correctness.harness.cortex_checkpoint_export")
    decode = ModuleType("arctic_platform.correctness.harness.cortex_optimizer_decode")
    monkeypatch.setitem(sys.modules, export.__name__, export)
    monkeypatch.setitem(sys.modules, decode.__name__, decode)
    return export, decode


@pytest.fixture
def lifecycle(tmp_path, artifact_stubs):
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
        assert kwargs == {"expected_optimizer_config": OPTIMIZER, "expected_learning_rate": 0.01}
        path = _manifest(output, step=state.manifest_step)
        event("decode")
        return path

    artifact_stubs[0].export_checkpoint = export
    artifact_stubs[1].decode_optimizer_capture = decode
    return SimpleNamespace(
        events=events, state=state, session=session, job=job, snapshots=snapshots, output=tmp_path / "output"
    )


def _run(lifecycle):
    return capture.capture_optimizer_step(
        lifecycle.session,
        lifecycle.job,
        {"same": "batch"},
        learning_rate=0.01,
        output_dir=lifecycle.output,
        expected_dp_world_size=4,
        optimizer_config=OPTIMIZER,
    )


@pytest.mark.parametrize("unsafe", ["/tmp", "/code", "/data-fast-shadow", "/data-fast", "symlink"])
def test_capture_requires_data_fast_before_mkdir_or_checkpoint(lifecycle, tmp_path, monkeypatch, unsafe):
    destination = Path(unsafe)
    if unsafe == "symlink":
        destination = tmp_path / "escape"
        destination.symlink_to("/tmp", target_is_directory=True)
        destination /= "output"
    lifecycle.output = destination
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: pytest.fail("unsafe output reached mkdir"))
    with pytest.raises(ValueError, match="/data-fast"):
        _run(lifecycle)
    assert lifecycle.events == []
    assert lifecycle.state.saves == lifecycle.state.step == 0


def test_same_job_snapshots_bracket_exactly_one_frozen_lr_step(lifecycle):
    result = _run(lifecycle)
    assert lifecycle.events == ["save-0", "fwd_bwd", "step", "save-1", "export-cp_bef0", "export-cp_af71", "decode"]
    assert lifecycle.state.step == 1
    assert lifecycle.state.saves == 2
    assert result.avg_loss == 2.0 and result.grad_norms == {"weight": 1.0}
    assert result.model_calls == 3 and result.packed_rows == 4
    assert Path(result.optimizer_state_manifest) == lifecycle.output / "states" / "manifest.json"
    assert Path(result.optimizer_state_manifest).is_file()
    assert not list(lifecycle.output.glob("snapshots-*"))
    assert all(not snapshot.exists() for snapshot in lifecycle.snapshots)
    assert lifecycle.job.client.transport.job_id == "parent-job"


@pytest.mark.parametrize(
    "failure", ["save-0", "fwd_bwd", "step", "save-1", "export-cp_bef0", "export-cp_af71", "decode"]
)
def test_failures_remove_downloads_and_never_publish_partial_manifest(lifecycle, failure):
    lifecycle.state.failure = failure
    with pytest.raises(RuntimeError, match="injected"):
        _run(lifecycle)
    assert not list(lifecycle.output.iterdir())
    assert all(not snapshot.exists() for snapshot in lifecycle.snapshots)


@pytest.mark.parametrize("step", [0, 2, True, "1", None])
def test_decoder_cannot_relabel_non_initial_update_as_step_one(lifecycle, step):
    lifecycle.state.manifest_step = step
    with pytest.raises(ValueError, match="exactly step 1"):
        _run(lifecycle)
    assert not list(lifecycle.output.iterdir())


def test_duplicate_durable_checkpoints_rejected_before_export(lifecycle):
    lifecycle.state.duplicate = True
    with pytest.raises(ValueError, match="distinct durable"):
        _run(lifecycle)
    assert lifecycle.events == ["save-0", "fwd_bwd", "step", "save-1"]
    assert not list(lifecycle.output.iterdir())


def test_existing_output_never_overwritten_or_stepped(lifecycle):
    existing = _manifest(lifecycle.output / "states")
    original = existing.read_bytes()
    with pytest.raises(FileExistsError):
        _run(lifecycle)
    assert existing.read_bytes() == original
    assert lifecycle.events == []


def test_decoder_must_return_manifest_inside_its_owned_output(lifecycle, artifact_stubs):
    outside = _manifest(lifecycle.output / "unrelated")
    artifact_stubs[1].decode_optimizer_capture = lambda *args, **kwargs: outside
    with pytest.raises(ValueError, match="must return its output manifest"):
        _run(lifecycle)
    assert outside.is_file()
    assert not list(lifecycle.output.glob("snapshots-*"))
    assert not (lifecycle.output / "states").exists()


def test_uninitialized_parent_job_rejected_before_save(lifecycle):
    lifecycle.job.client.transport.job_id = None
    with pytest.raises(ValueError, match="initialized parent job"):
        _run(lifecycle)
    assert lifecycle.events == []


@pytest.mark.parametrize("degree", [0, -1, True, 1.0])
def test_invalid_dp_degree_rejected_before_save(lifecycle, degree):
    with pytest.raises(ValueError, match="positive integer"):
        capture.capture_optimizer_step(
            lifecycle.session,
            lifecycle.job,
            {},
            learning_rate=0.01,
            output_dir=lifecycle.output,
            expected_dp_world_size=degree,
            optimizer_config=OPTIMIZER,
        )
    assert lifecycle.events == []


def test_onprem_cannot_use_managed_capture(lifecycle):
    lifecycle.session = GatewaySession(lifecycle.output, 4, OnPremConfig())
    with pytest.raises(ValueError, match="requires CortexConfig"):
        _run(lifecycle)
    assert lifecycle.events == []


@pytest.mark.parametrize("order", ["batched", "interleaved"])
@pytest.mark.parametrize("failure", [None, "save-0", "fwd_bwd", "step", "save-1", "export-0", "export-1", "decode"])
def test_runner_public_cortex_capture_keeps_fixture_budget_gradient_and_cleanup(
    tmp_path, monkeypatch, artifact_stubs, capsys, order, failure
):
    cfg = load_config(CONFIG)
    spec = Spec.read(SPEC)
    original = CONFIG.read_bytes(), SPEC.read_bytes()
    creates, cancels, calls, references, snapshots = [], [], [], [], []
    transports, closed = [], []
    replies = {}
    counts = {}
    exports, decodes = artifact_stubs
    cfg.training["debug"] = {"optimizer_state_output_dir": "/local-only-ray-hook"}
    original_close = requests.Session.close

    def close(session):
        closed.append(session)
        return original_close(session)

    monkeypatch.setattr(requests.Session, "close", close)

    def record(job_id, action):
        calls.append((job_id, action))
        if action == failure:
            raise RuntimeError("injected capture failure")

    def reference(model, batch, out, **kwargs):
        assert model == spec.model.cache_path
        assert kwargs["token_budget"] == 10240
        assert kwargs["learning_rate"] == 0.01
        assert kwargs["optimizer_config"] == cfg.training["optimizer"]
        references.append(model)
        return {
            "loss": 2.0,
            "grad_norms": {"weight": 1.0},
            "microbatches": 1,
            "optimizer_state_manifest": str(_manifest(kwargs["optimizer_output_dir"])),
        }

    def send(transport, method, url, **kwargs):
        if method == "POST" and url == transport._prefix:
            transports.append(transport)
            creates.append(kwargs["json"])
            parent = f"job-{len(creates)}"
            counts[parent] = 0
            return {"job_id": parent}
        if method == "GET" and url == f"{transport._prefix}/{transport.job_id}":
            return {"status": "RUNNING"}
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
        manifest = _manifest(output_dir)
        if failure == "decode":
            raise RuntimeError("injected capture failure")
        return manifest

    exports.export_checkpoint = export
    decodes.decode_optimizer_capture = decode
    monkeypatch.setattr(cortex_fixture, "load_fixture", lambda *args: {"model_name": "published-reviewed"})
    monkeypatch.setattr(runner, "run_reference", reference)
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
            result.outcome is Outcome.FAIL
            and result.arm == "gas1"
            and result.gpus == cfg.n_gpus
            and result.reason == "injected capture failure"
            and result.summary == "execution error: RuntimeError: injected capture failure"
            for result in results
        )
        actions = ["save-0", "fwd_bwd", "step", "save-1", "export-0", "export-1"]
        expected = actions if failure == "decode" else actions[: actions.index(failure) + 1]
        assert calls == [("job-1", action) for action in expected]
        traceback = capsys.readouterr().err
        assert "Traceback (most recent call last):" in traceback
        assert "capture_optimizer_step" in traceback
        assert "RuntimeError: injected capture failure" in traceback
        assert cancels == ["job-1"]
        assert all(transport.job_id is None and transport.session in closed for transport in transports)
        assert all(not snapshot.exists() for snapshot in snapshots)
        assert not list((tmp_path / "optimizer" / "dss" / "gas1").iterdir())
        assert CONFIG.read_bytes() == original[0] and SPEC.read_bytes() == original[1]
        return
    results = execute()
    assert len(results) == 4 and all(result.outcome is Outcome.PASS for result in results)
    assert len(references) == len(creates) == len(cancels) == 2
    assert cancels == ["job-1", "job-2"]
    assert all(transport.job_id is None and transport.session in closed for transport in transports)
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
    assert all(not snapshot.exists() for snapshot in snapshots)
    assert (CONFIG.read_bytes(), SPEC.read_bytes()) == original


@pytest.mark.parametrize("order", ["batched", "interleaved"])
def test_runner_onprem_preserves_local_optimizer_capture(tmp_path, monkeypatch, order):
    cfg = load_config(CONFIG)
    spec = Spec.read(SPEC)
    observed = []
    backend = OnPremConfig()

    def reference(model, batch, out, **kwargs):
        assert kwargs["token_budget"] == 10240 and kwargs["learning_rate"] == 0.01
        return {
            "loss": 2.0,
            "microbatches": 1,
            "optimizer_state_manifest": str(_manifest(kwargs["optimizer_output_dir"])),
        }

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
        return StepResult(2.0, {}, optimizer_state_manifest=str(_manifest(path)))

    monkeypatch.setattr(runner, "run_reference", reference)
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
    assert observed[0]["model_name"] == spec.model.cache_path
    assert "hardware" not in observed[0]
