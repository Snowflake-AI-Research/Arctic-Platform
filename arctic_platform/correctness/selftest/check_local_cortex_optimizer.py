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

"""Existing-gateway local optimizer checkpoints retain real client framing and decoder gates."""

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
import requests
import torch

from arctic_platform import wire
from arctic_platform.client import CortexConfig
from arctic_platform.client.transports.cortex import _DEFAULT
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.client.transports.cortex import _SendPolicy
from arctic_platform.correctness.harness import cortex_fixture
from arctic_platform.correctness.harness import cortex_optimizer_capture as capture
from arctic_platform.correctness.harness import execution as cli
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness.arms import ArmDefinition
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.cortex import CorrectnessClient
from arctic_platform.correctness.harness.dss_driver import ArcticJob
from arctic_platform.correctness.harness.dss_driver import GatewaySession
from arctic_platform.correctness.harness.dss_driver import StepResult
from arctic_platform.correctness.harness.dss_driver import _training_client_config
from arctic_platform.correctness.harness.dss_driver import build_payload
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.spec import TestSpec as Spec
from arctic_platform.correctness.selftest.check_cortex_optimizer_capture import CONFIG
from arctic_platform.correctness.selftest.check_cortex_optimizer_capture import OPTIMIZER
from arctic_platform.correctness.selftest.check_cortex_optimizer_capture import SPEC
from arctic_platform.correctness.selftest.check_cortex_optimizer_capture import _manifest
from arctic_platform.correctness.selftest.check_cortex_optimizer_capture import _prepared_references
from arctic_platform.correctness.selftest.check_cortex_optimizer_decode import _pair


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
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
def local_capture(tmp_path, monkeypatch, no_network):
    pair = _pair(tmp_path / "source", world_size=4)
    backend = CortexConfig(base_url="http://127.0.0.1:8003")
    session = GatewaySession(tmp_path, 4, backend)
    events, replies, paths = [], {}, []
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
    client_config = _training_client_config(session, build_payload(load_config(CONFIG).training, "local-fixture", 42))
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
    assert len(no_network) == 1 and no_network[0].job_id is None


def _capture(case):
    return capture.capture_optimizer_step(
        case.session,
        case.job,
        {"input_ids": torch.tensor([[1, 2]])},
        learning_rate=0.01,
        output_dir=case.output,
        expected_dp_world_size=4,
        optimizer_config=OPTIMIZER,
        local_checkpoint=True,
    )


def test_local_save_wire_and_real_dense_decoder_bracket_one_step(local_capture):
    result = _capture(local_capture)
    assert local_capture.events == ["save-0", "fwd_bwd", "step", "save-1"]
    assert result.avg_loss == 2.0 and result.grad_norms == {"weight": 1.0}
    manifest = Path(result.optimizer_state_manifest)
    assert manifest == local_capture.output / "states" / "manifest.json"
    payload = json.loads(manifest.read_text())
    assert payload["step"] == 1
    for entry in payload["parameters"]:
        values = torch.load(manifest.parent / entry["file"], weights_only=True)
        for key in ("parameter_update", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(values[key], local_capture.pair.expected[entry["name"]][key], rtol=0, atol=0)
    assert all(not path.exists() for path in local_capture.paths)
    assert not list(local_capture.output.glob("snapshots-*"))


@pytest.mark.parametrize("failure", ["save-0", "step", "save-1"])
def test_local_failure_cleans_only_owned_snapshots(local_capture, failure):
    local_capture.state.failure = failure
    with pytest.raises(RuntimeError, match="injected"):
        _capture(local_capture)
    assert not list(local_capture.output.iterdir())
    assert local_capture.pair.before.is_dir() and local_capture.pair.after.is_dir()


def test_invalid_after_save_never_publishes_partial_artifact(local_capture):
    def receipt(result, root):
        if local_capture.state.step:
            result["version"] = 2
        return result

    local_capture.state.receipt = receipt
    with pytest.raises(ValueError, match="receipt"):
        _capture(local_capture)
    assert local_capture.events == ["save-0", "fwd_bwd", "step", "save-1"]
    assert not list(local_capture.output.iterdir())


def test_local_decoder_failure_preserves_unrelated_data_and_cleans_owned_snapshots(local_capture, monkeypatch):
    from arctic_platform.correctness.harness import cortex_optimizer_decode as decoder

    def fail(before, after, output, **kwargs):
        output.mkdir()
        (output / "partial.pt").write_bytes(b"not publishable")
        raise ValueError("injected invalid optimizer schema")

    monkeypatch.setattr(decoder, "decode_optimizer_capture", fail)
    with pytest.raises(ValueError, match="schema"):
        _capture(local_capture)
    assert not list(local_capture.output.iterdir())
    assert local_capture.pair.before.is_dir() and local_capture.pair.after.is_dir()


def test_existing_local_artifact_never_overwritten_or_stepped(local_capture):
    path = _manifest(local_capture.output / "states")
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        _capture(local_capture)
    assert path.read_bytes() == original
    assert local_capture.events == []


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
def test_local_receipt_mismatch_rejected_before_step(local_capture, field, value):
    local_capture.state.receipt = lambda result, path: {**result, field: value}
    with pytest.raises(ValueError):
        _capture(local_capture)
    assert local_capture.events == ["save-0"]
    assert not list(local_capture.output.iterdir())
    assert local_capture.pair.before.is_dir()


@pytest.mark.parametrize("bad", ["missing-rank", "extra-rank", "symlink", "empty", "wrong-tag"])
def test_incomplete_local_checkpoint_never_reaches_step(local_capture, bad):
    def damage(result, root):
        tag = root / "global_step0"
        shard = next(tag.glob("*rank_0_*optim_states.pt"))
        if bad == "missing-rank":
            shard.unlink()
        elif bad == "extra-rank":
            shutil.copyfile(shard, tag / "bf16_zero_pp_rank_4_mp_rank_00_optim_states.pt")
        elif bad == "symlink":
            shard.unlink()
            shard.symlink_to(next((local_capture.pair.before / "global_step0").glob("*optim_states.pt")))
        elif bad == "empty":
            shard.write_bytes(b"")
        else:
            tag.rename(root / "global_step99")
        return result

    local_capture.state.receipt = damage
    with pytest.raises(ValueError):
        _capture(local_capture)
    assert local_capture.events == ["save-0"]
    assert not list(local_capture.output.iterdir())


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
def test_local_checkpoint_flag_rejects_nonloopback_or_auth_before_writes(local_capture, monkeypatch, url, credentials):
    local_capture.session = GatewaySession(local_capture.output, 4, CortexConfig(base_url=url, **credentials))
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: pytest.fail("unsafe backend reached mkdir"))
    with pytest.raises(ValueError, match="loopback"):
        _capture(local_capture)
    assert local_capture.events == []


@pytest.mark.parametrize("order", ["batched", "interleaved"])
def test_local_fixture_optimizer_runner_uses_same_model_and_frozen_budget(tmp_path, monkeypatch, no_network, order):
    cfg, spec = load_config(CONFIG), Spec.read(SPEC)
    original = CONFIG.read_bytes(), SPEC.read_bytes()
    captures, creates = [], []
    cancelled = set()
    references = _prepared_references(monkeypatch, spec, cfg, tmp_path)
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
        return StepResult(2.0, {}, optimizer_state_manifest=str(_manifest(kwargs["output_dir"] / "states")))

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
    assert all(result.outcome is Outcome.PASS and result.gpus == 4 for result in results)
    assert all(result.metrics["stated_criterion_abs"] == 1e-3 for result in results)
    assert references == [("single-step-optimizer", "gas1"), ("single-step-optimizer", "gas4")]
    assert len(references) == len(captures) == len(creates) == 2
    assert cancelled == {f"parent-{index}" for index in (1, 2)}
    assert len(no_network) == 2 and all(transport.job_id is None for transport in no_network)
    for create in creates:
        child = create["sub_job_configs"][0]
        assert child["model_name"] == model and child["training_config"]["n_gpus"] == 4
        assert "optimizer_state_output_dir" not in child["training_config"]["debug"]
    assert (CONFIG.read_bytes(), SPEC.read_bytes()) == original


@pytest.mark.parametrize("tests", [["single-step-optimizer"], ["single-step-grads", "single-step-optimizer"]])
def test_cli_local_fixture_accepts_optimizer_selection_before_dispatch(monkeypatch, tests):
    args = SimpleNamespace(
        local_cortex_fixture=True, cortex_fixture=None, config=str(CONFIG), all_configs=False, test=tests
    )
    monkeypatch.setattr(cli, "_client_backend", lambda args: CortexConfig(base_url="http://127.0.0.1:8003"))

    def stop(*args, **kwargs):
        raise RuntimeError("eligible dispatch")

    monkeypatch.setattr(cli, "_resolve_configs", stop)
    with pytest.raises(RuntimeError, match="eligible dispatch"):
        cli.cmd_run(args)


@pytest.mark.parametrize(
    "selected", [None, ["e2e-sft-train-validate"], ["single-step-optimizer", "inference-checkpoint-loss"]]
)
def test_cli_local_fixture_does_not_implicitly_enable_other_checks(monkeypatch, selected):
    args = SimpleNamespace(
        local_cortex_fixture=True, cortex_fixture=None, config=str(CONFIG), all_configs=False, test=selected
    )
    monkeypatch.setattr(cli, "_client_backend", lambda args: CortexConfig(base_url="http://127.0.0.1:8003"))
    monkeypatch.setattr(cli, "_resolve_configs", lambda *args, **kwargs: pytest.fail("invalid selection dispatched"))
    with pytest.raises(SystemExit, match="single-step"):
        cli.cmd_run(args)


def test_runner_rejects_different_frozen_optimizer_reference_before_batch_or_job(tmp_path, monkeypatch):
    cfg, spec = load_config(CONFIG), Spec.read(SPEC)
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


def test_cli_local_fixture_never_expands_declared_four_gpu_width(tmp_path, monkeypatch):
    cfg, spec = load_config(CONFIG), Spec.read(SPEC)
    args = SimpleNamespace(
        local_cortex_fixture=True,
        cortex_fixture=None,
        config=str(CONFIG),
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
    monkeypatch.setattr(cli, "_resolve_configs", lambda *args, **kwargs: [CONFIG])
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
