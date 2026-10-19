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

"""Verify reduced and explicit full-model identity without allocating gateway or GPU workers."""

import asyncio
import base64
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
import requests
import torch

from arctic_platform import wire
from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.client.transports.cortex import _DEFAULT
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.client.transports.cortex import _SendPolicy
from arctic_platform.correctness.checks import e2e_sft_train_validate as check
from arctic_platform.correctness.harness import dss_driver
from arctic_platform.correctness.harness import gsm8k
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.spec import TestSpec as ConfigSpec

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config"
FULL_SNAPSHOT = "/data-fast/tmp/e2e-full-test/hf/hub/models--Qwen--Qwen3-8B/snapshots/pinned"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unit tests must not access the network")

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


def context(tmp_path, width=4):
    cfg = load_config(CONFIG).at_gpu_width(width)
    return SimpleNamespace(
        cfg=cfg,
        spec=ConfigSpec.read(ROOT / f"specs/{cfg.config_id}.json"),
        workdir=tmp_path / "work",
        attn_implementation=cfg.attention_implementation,
        client_backend=CortexConfig(base_url="http://unused"),
        target_model_name="ap-correctness/reviewed-fixture",
    )


def replay(tmp_path):
    rows = [
        gsm8k.Example(
            question=f"row-{index}", prompt="q:", completion=" a", input_ids=(10 + index, 30, 40), n_prompt=1
        )
        for index in range(6)
    ]
    return check.materialize_replay(
        check.replay_batches(rows, 2, 3, pad_id=0),
        gsm8k.build_batch("validation", rows[:2], pad_id=0),
        tmp_path / "replay",
    )


@pytest.mark.parametrize("width", [4, 8])
@pytest.mark.parametrize("failure", [None, "backward", "forward"])
def test_e2e_replay_uses_validated_fixture_and_cleans_up(tmp_path, monkeypatch, no_network, width, failure):
    ctx = context(tmp_path, width)
    before = json.dumps(ctx.cfg.training, sort_keys=True)
    recorded = replay(tmp_path)
    expected_train, expected_validation = check.request_bodies(recorded, model_provider="huggingface")
    creates, cancels, backward, steps, forward, replies = [], [], [], [], [], {}

    def forbidden(*args, **kwargs):
        pytest.fail("Cortex replay must not enter the local Ray gateway")

    monkeypatch.setattr(dss_driver, "_gateway_runtime", forbidden)

    async def send(transport, method, url, *, policy=_DEFAULT, **kwargs):
        assert asyncio.get_running_loop() is transport._loop
        assert isinstance(policy, _SendPolicy)
        if method == "POST" and url == transport._prefix:
            creates.append(kwargs["json"])
            return {"job_id": "e2e-job"}
        if url.endswith(":cancel"):
            cancels.append(transport.job_id)
            return {}
        if method == "GET" and url == f"{transport._prefix}/{transport.job_id}":
            return {"status": "RUNNING", "sub_jobs": [{"job_type": "training", "sub_job_id": "trainer"}]}
        if method == "GET":
            return {"status": "REQUEST_STATE_DONE", "result": replies[url.rsplit("/", 1)[-1]]}
        if url.endswith("/forward-backward"):
            body = wire.loads(wire.decode_byte_chunks([kwargs["data"]], kind="request"))
            assert "processing" not in body
            backward.append(body)
            if failure == "backward":
                raise RuntimeError("mock backward failed")
            result = {"avg_loss": float(len(backward))}
        elif url.endswith("/step"):
            steps.append(kwargs["json"])
            result = {}
        elif url.endswith("/operation"):
            envelope = kwargs["json"]
            assert envelope["operation_type"] == "forward"
            assert envelope["sub_job_id"] == "trainer"
            body = wire.loads(base64.b64decode(envelope["payload"]["payload_b64"]))
            assert "processing" not in body
            assert body["context"] == {"labels_are_shifted": True}
            forward.append(body)
            if failure == "forward":
                raise RuntimeError("mock forward failed")
            result = {"logprobs": torch.full(body["kwargs"]["input_ids"].shape, -1.25)}
        else:
            pytest.fail(f"unexpected request: {method} {url}")
        request_id = f"request-{len(replies)}"
        replies[request_id] = result
        return {"request_id": request_id}

    monkeypatch.setattr(CortexTransport, "_send", send)
    if failure:
        with pytest.raises(RuntimeError, match=f"mock {failure} failed"):
            check.run_dss_trajectory(ctx, recorded, tmp_path / "target")
    else:
        losses, logprobs = check.run_dss_trajectory(ctx, recorded, tmp_path / "target")
        assert losses == [1.0, 2.0, 3.0]
        assert torch.all(logprobs == -1.25)
        assert steps == [{"learning_rate": ctx.cfg.training["optimizer"]["lr"]}] * 3
        for actual, expected in zip(backward, expected_train):
            for key in ("input_ids", "labels", "position_ids"):
                torch.testing.assert_close(actual["kwargs"][key], expected["batch"][key], rtol=0, atol=0)
        torch.testing.assert_close(
            forward[0]["kwargs"]["labels"], expected_validation["batch"]["labels"], rtol=0, atol=0
        )
    assert cancels == ["e2e-job"]
    assert no_network and all(transport.job_id is None for transport in no_network)
    assert creates[0]["hardware"] == "H200"
    sub = creates[0]["sub_job_configs"][0]
    assert sub["model_name"] == ctx.target_model_name
    assert sub["training_config"]["n_gpus"] == width
    assert sub["training_config"]["ds_config"] == ctx.cfg.training["ds_config"]
    assert json.dumps(ctx.cfg.training, sort_keys=True) == before


@pytest.mark.parametrize("name", [None, ""])
def test_e2e_rejects_missing_fixture_before_allocation(tmp_path, monkeypatch, name):
    ctx = context(tmp_path)
    ctx.target_model_name = name
    monkeypatch.setattr(check, "gateway", lambda *args, **kwargs: pytest.fail("must reject before allocation"))
    with pytest.raises(ValueError, match="validated published Cortex fixture"):
        check.run_dss_trajectory(ctx, replay(tmp_path), tmp_path / "target")


def full_context(tmp_path):
    ctx = context(tmp_path)
    ctx.client_backend = CortexConfig(base_url="http://127.0.0.1:8003")
    ctx.local_gateway_full_model = True
    ctx.local_full_model_path = FULL_SNAPSHOT
    # A fixture binding must not change the full-model request.
    ctx.target_model_name = "ap-correctness/not-the-full-model"
    return ctx


@pytest.mark.parametrize("full", [None, False, True])
def test_e2e_reference_command_selects_identity_without_changing_contract(tmp_path, monkeypatch, full):
    ctx = full_context(tmp_path) if full else context(tmp_path)
    if full is not None:
        ctx.local_gateway_full_model = full
    expected = FULL_SNAPSHOT if full else ctx.spec.model.cache_path
    before_config = json.dumps(ctx.cfg.sub_jobs, sort_keys=True)
    before_spec = ctx.spec.to_json()
    inspected = []
    monkeypatch.setattr(check, "uses_mixer_packing", lambda path: inspected.append(path) or False)
    recorded = replay(tmp_path)

    command = check.reference_command(ctx, recorded, tmp_path / "replay.json", tmp_path / "out.json", None)

    assert command[command.index("--model") + 1] == expected
    assert command[command.index("--learning-rate") + 1] == str(ctx.cfg.training["optimizer"]["lr"])
    assert json.loads(command[command.index("--optimizer-config") + 1]) == ctx.cfg.training["optimizer"]
    if full:
        assert command.count("--optimizer-chunk-elements") == 1
        assert command[command.index("--optimizer-chunk-elements") + 1] == "1048576"
    else:
        assert "--optimizer-chunk-elements" not in command
    assert inspected == [expected]
    assert json.dumps(ctx.cfg.sub_jobs, sort_keys=True) == before_config
    assert ctx.spec.to_json() == before_spec
    assert ctx.cfg.n_gpus == 4
    assert check.TRAIN_STEPS == 100
    assert check.TOLERANCE == 0.03


@pytest.mark.parametrize("full", [False, True])
def test_e2e_tokenizer_uses_selected_reference_identity(tmp_path, monkeypatch, full):
    ctx = full_context(tmp_path) if full else context(tmp_path)
    expected = FULL_SNAPSHOT if full else ctx.spec.model.cache_path
    seen = []

    def model_identity(path):
        seen.append(("model", path))
        return {"assets": {}, "sha256": "mock-selected-model-content"}

    def tokenizer(path):
        seen.append(("tokenizer", path))
        raise RuntimeError("stop before materializing any data")

    def forbidden(*args, **kwargs):
        pytest.fail("invalid tokenizer must stop before records, HF, or target allocation")

    monkeypatch.setattr(check.reference_records, "model_identity", model_identity)
    monkeypatch.setattr(gsm8k, "load_tokenizer", tokenizer)
    monkeypatch.setattr(check.reference_records, "load_record", forbidden)
    monkeypatch.setattr(check, "run_reference_trajectory", forbidden)
    monkeypatch.setattr(check, "run_dss_trajectory", forbidden)
    with pytest.raises(RuntimeError, match="stop before materializing"):
        check._run(ctx)
    assert seen == [("model", expected), ("tokenizer", expected)]


@pytest.mark.parametrize("failure", [None, "backward", "forward"])
def test_full_e2e_requests_original_model_and_preserves_cleanup(tmp_path, monkeypatch, failure):
    ctx = full_context(tmp_path)
    ctx.target_model_name = None
    before = json.dumps(ctx.cfg.training, sort_keys=True)
    recorded = replay(tmp_path)
    created, exited, trained = [], [], []
    session, job = object(), object()

    @contextlib.contextmanager
    def gateway(directory, slots, *, backend):
        assert slots == 4
        assert backend is ctx.client_backend
        yield session

    @contextlib.contextmanager
    def running_job(actual_session, payload):
        assert actual_session is session
        created.append(payload)
        try:
            yield job
        finally:
            exited.append(job)

    def train(actual_session, actual_job, body, *, learning_rate):
        assert actual_session is session and actual_job is job
        assert learning_rate == ctx.cfg.training["optimizer"]["lr"]
        if failure == "backward":
            raise RuntimeError("mock backward failed")
        trained.append(body)
        return SimpleNamespace(avg_loss=float(len(trained)))

    def forward(actual_session, actual_job, body):
        assert actual_session is session and actual_job is job
        assert "return_logprobs" not in body["processing"]
        if failure == "forward":
            raise RuntimeError("mock forward failed")
        return torch.full(body["batch"]["input_ids"].shape, -1.25)

    monkeypatch.setattr(check, "gateway", gateway)
    monkeypatch.setattr(check, "running_job", running_job)
    monkeypatch.setattr(check, "fwd_bwd_step", train)
    monkeypatch.setattr(check, "forward_logprobs", forward)
    if failure:
        with pytest.raises(RuntimeError, match=f"mock {failure} failed"):
            check.run_dss_trajectory(ctx, recorded, tmp_path / "target")
    else:
        losses, logprobs = check.run_dss_trajectory(ctx, recorded, tmp_path / "target")
        assert losses == [1.0, 2.0, 3.0]
        assert torch.all(logprobs == -1.25)
    assert exited == [job]
    assert created[0]["model_name"] == ctx.cfg.model_name == "Qwen/Qwen3-8B"
    assert created[0]["training_config"]["n_gpus"] == 4
    assert created[0]["training_config"]["ds_config"] == ctx.cfg.training["ds_config"]
    assert json.dumps(ctx.cfg.training, sort_keys=True) == before


@pytest.mark.parametrize("path", [None, "", "/code/not-fast/full-model"])
def test_full_e2e_rejects_missing_or_nonfast_snapshot_before_allocation(tmp_path, monkeypatch, path):
    ctx = full_context(tmp_path)
    ctx.local_full_model_path = path
    monkeypatch.setattr(check, "gateway", lambda *args, **kwargs: pytest.fail("must reject before allocation"))
    with pytest.raises(ValueError, match="validated local full-model snapshot|/data-fast"):
        check.run_dss_trajectory(ctx, replay(tmp_path), tmp_path / "target")


@pytest.mark.parametrize(
    "backend",
    [
        CortexConfig(base_url="https://hosted.example"),
        CortexConfig(base_url="http://other-host:8003"),
        CortexConfig(base_url="http://127.0.0.1:8003", host="hosted.example"),
        OnPremConfig(),
    ],
)
def test_full_e2e_never_bypasses_hosted_or_non_gateway_guards(tmp_path, monkeypatch, backend):
    ctx = full_context(tmp_path)
    ctx.client_backend = backend
    monkeypatch.setattr(check, "gateway", lambda *args, **kwargs: pytest.fail("must reject before allocation"))
    with pytest.raises(ValueError, match="local Cortex gateway|HTTP loopback"):
        check.run_dss_trajectory(ctx, replay(tmp_path), tmp_path / "target")
