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

"""Static correctness selftests: runtime clients."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from argparse import Namespace
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import aiohttp
import pytest
import requests
import torch
from pydantic import ValidationError

from arctic_platform import wire
from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.client import TrainingConfig
from arctic_platform.client.transport import JobHandles
from arctic_platform.client.transports.cortex import _DEFAULT
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.client.transports.cortex import _SendPolicy
from arctic_platform.correctness.checks import e2e_sft_train_validate as check
from arctic_platform.correctness.checks import e2e_sft_train_validate as e2e
from arctic_platform.correctness.checks import inference_checkpoint_loss as inference
from arctic_platform.correctness.harness import cortex as cortex_fixture
from arctic_platform.correctness.harness import dss_driver
from arctic_platform.correctness.harness import execution as cli
from arctic_platform.correctness.harness import hosted
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness import sft as gsm8k
from arctic_platform.correctness.harness.batches import load as load_batch
from arctic_platform.correctness.harness.config import LoadedConfig
from arctic_platform.correctness.harness.config import TestSpec as ConfigSpec
from arctic_platform.correctness.harness.config import TestSpec as Spec
from arctic_platform.correctness.harness.config import arms_for
from arctic_platform.correctness.harness.config import config_checksum
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.cortex import CorrectnessCortexTransport
from arctic_platform.correctness.harness.cortex import CortexTrainingConfig
from arctic_platform.correctness.harness.dss_driver import GatewaySession
from arctic_platform.correctness.harness.dss_driver import _training_client_config
from arctic_platform.correctness.harness.dss_driver import build_payload
from arctic_platform.correctness.harness.hosted import HostedConnection
from arctic_platform.correctness.harness.hosted import HostedTrainingJob
from arctic_platform.correctness.harness.hosted import HostedTransport
from arctic_platform.correctness.harness.hosted import is_local_cortex_gateway
from arctic_platform.correctness.harness.hosted import require_local_cortex_gateway
from arctic_platform.correctness.harness.hosted import save_and_resolve_checkpoint_id
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.registry import TestResult as Result
from arctic_platform.correctness.harness.transport import Checkpoint
from arctic_platform.testing_utils import torch_assert_equal

from .check_catalog_configuration import _applicable_tests_spec
from .check_operator_interface import _cli_config_selection_write_config

# Client backends.


def _client_backends_remote():
    return CortexConfig(base_url="http://cortex.test")


@pytest.fixture(autouse=False)
def _client_backends_no_network(monkeypatch):

    def forbidden(*args, **kwargs):
        pytest.fail("unit tests must not access the network")

    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(aiohttp.ClientSession, "request", forbidden)
    transports = []
    original_start = CortexTransport._start_loop

    def start(transport):
        original_start(transport)
        transports.append(transport)

    monkeypatch.setattr(CortexTransport, "_start_loop", start)
    yield
    for transport in transports:
        transport._on_loop(transport._close_session()).result()
        transport._loop.call_soon_threadsafe(transport._loop.stop)


def _client_backends_session_with_close_spy(transport, monkeypatch, closed):

    async def create():
        assert asyncio.get_running_loop() is transport._loop
        transport._session = aiohttp.ClientSession(headers=transport._auth_headers(), trust_env=False)
        return transport._session

    session = transport._on_loop(create()).result()
    original_close = transport._close_session

    async def close():
        assert asyncio.get_running_loop() is transport._loop
        current = transport._session
        await original_close()
        assert transport._session is None
        assert current is None or current.closed
        if current is not None:
            closed.append("close")

    monkeypatch.setattr(transport, "_close_session", close)
    return session


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_cli_defaults_to_cortex_training_and_accepts_legacy_cortex(monkeypatch):
    selected = []
    monkeypatch.setattr(cli, "cmd_run", lambda args: selected.append(args.backend) or 0)
    assert cli.main(["run", "--all-configs"]) == 0
    assert cli.main(["run", "--all-configs", "--backend", "cortex-training"]) == 0
    assert cli.main(["run", "--all-configs", "--backend", "cortex"]) == 0
    assert selected == ["cortex", "cortex", "cortex"]
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["run", "--all-configs", "--backend", "onprem"])
    assert exc.value.code == 2


def test_client_backends_onboarding_uses_the_same_cortex_training_default():
    args = cli.parse_args(["onboard", "--config", "./work/config.json"])
    assert args.backend == "cortex"
    args = cli.parse_args(["onboard", "--config", "./work/config.json", "--backend", "cortex-training"])
    assert args.backend == "cortex"


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_cortex_connection_uses_environment_or_pat_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://from-env.test")
    assert cli._client_backend(Namespace()).base_url == "http://from-env.test"
    token = tmp_path / "pat"
    token.write_text("file-token\n")
    backend = cli._client_backend(Namespace(pat_file=token, host="host", database="db", schema="sc"))
    assert backend.pat.get_secret_value() == "file-token"
    assert "file-token" not in repr(backend)
    assert isinstance(cli._client_backend(Namespace(backend="onprem")), OnPremConfig)


@pytest.mark.usefixtures("_client_backends_no_network")
@pytest.mark.parametrize(
    "base_url",
    ["http://127.0.0.1:7001", "http://localhost:7001", "http://[::1]:7001", "http://remote.test:7001"],
)
@pytest.mark.parametrize("from_environment", [False, True])
@pytest.mark.parametrize("settings", ["missing", "explicit", "environment", "empty"])
def test_client_backends_local_cortex_route_segments(monkeypatch, base_url, from_environment, settings):
    for name in ("BASE_URL", "HOST", "PAT", "DATABASE", "SCHEMA"):
        monkeypatch.delenv(f"ARCTIC_CORTEX_{name}", raising=False)
    args = Namespace(backend="cortex-training")
    if from_environment:
        monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", base_url)
    else:
        args.base_url = base_url
    if settings == "explicit":
        args.database, args.schema = "USER_DB", "USER_SCHEMA"
    elif settings == "environment":
        monkeypatch.setenv("ARCTIC_CORTEX_DATABASE", "USER_DB")
        monkeypatch.setenv("ARCTIC_CORTEX_SCHEMA", "USER_SCHEMA")
    elif settings == "empty":
        args.database, args.schema = "", ""

    backend = cli._client_backend(args)
    if settings in {"explicit", "environment"}:
        expected = ("USER_DB", "USER_SCHEMA")
    elif base_url == "http://remote.test:7001":
        expected = ("", "")
    else:
        expected = ("dev", "dev")
    assert (backend.database, backend.schema_) == expected
    transport = CortexTransport(backend)
    try:
        assert transport._prefix == f"{base_url}/api/v2/databases/{expected[0]}/schemas/{expected[1]}/cortex-training"
    finally:
        transport.shutdown()


@pytest.mark.usefixtures("_client_backends_no_network")
@pytest.mark.parametrize(
    "base_url", ["http://127.0.0.1:7001", "http://localhost:7001", "http://[::1]:7001", "http://127.0.0.1:7001/"]
)
@pytest.mark.parametrize("from_environment", [False, True])
def test_client_backends_local_cortex_uses_direct_url_without_hosted_auth(monkeypatch, base_url, from_environment):
    monkeypatch.setenv("ARCTIC_CORTEX_HOST", "hosted-account.test")
    monkeypatch.setenv("ARCTIC_CORTEX_PAT", "unused-hosted-token")
    monkeypatch.setenv("ARCTIC_CORTEX_DATABASE", "HOSTED_DB")
    monkeypatch.setenv("ARCTIC_CORTEX_SCHEMA", "HOSTED_SCHEMA")
    if from_environment:
        monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", base_url)
    else:
        monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://old-gateway.test")
    args = Namespace(backend="cortex", database="LOCAL_DB", schema="LOCAL_SCHEMA")
    if not from_environment:
        args.base_url = base_url
    backend = cli._client_backend(args)
    assert isinstance(backend, CortexConfig)
    assert backend.base_url == base_url
    transport = CortexTransport(backend)
    calls = []

    class Reply:
        status = 200

        async def __aenter__(self):
            assert asyncio.get_running_loop() is transport._loop
            return self

        async def __aexit__(self, *args):
            return False

        async def read(self):
            return b'{"ok": true}'

    def request(session, method, url, **kwargs):
        assert asyncio.get_running_loop() is transport._loop
        assert session is transport._session
        assert session._loop is transport._loop
        assert session.trust_env is False
        calls.append((method, url, dict(session.headers), kwargs))
        return Reply()

    monkeypatch.setattr(aiohttp.ClientSession, "request", request)
    try:
        expected = f"{base_url.rstrip('/')}/api/v2/databases/LOCAL_DB/schemas/LOCAL_SCHEMA/cortex-training"
        assert transport._prefix == expected
        assert transport._auth_headers() == {}
        assert transport._on_loop(transport._send("GET", transport._prefix)).result() == {"ok": True}
        assert len(calls) == 1
        method, url, headers, kwargs = calls[0]
        assert (method, url) == ("GET", expected)
        assert urlsplit(url).netloc == urlsplit(base_url).netloc
        assert "Authorization" not in headers
        assert "X-Snowflake-Authorization-Token-Type" not in headers
        assert set(kwargs) == {"timeout"}
        assert isinstance(kwargs["timeout"], aiohttp.ClientTimeout)
        assert kwargs["timeout"].total == backend.request_timeout
    finally:
        session = transport._session
        transport.shutdown()
        assert session is None or session.closed
        assert transport._session is None


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_remote_run_never_probes_local_allocation(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    _cli_config_selection_write_config(config_dir, "h200")
    _cli_config_selection_write_config(config_dir, "b200")
    seen = []

    def forbidden(*args, **kwargs):
        pytest.fail("Cortex must not probe or clean the local allocation")

    monkeypatch.setattr(cli, "local_gpu_type", forbidden)
    monkeypatch.setattr(cli, "available_gpus", forbidden)
    monkeypatch.setattr(cli, "validate_against_host", forbidden)
    monkeypatch.setattr(cli, "ensure_no_interrupted_builds", forbidden)
    monkeypatch.setattr(cli, "_frozen_spec", lambda cfg, *args, **kwargs: seen.append(cfg.config_id))
    monkeypatch.setattr(cli, "_finish", lambda *args, **kwargs: 0)
    args = cli.parse_args(
        ["run", "--all-configs", "--config-dir", str(config_dir), "--base-url", _client_backends_remote().base_url]
    )
    assert cli._run_configs(args) == 0
    assert len(seen) == 2


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_remote_job_skips_ray_and_preserves_backend(tmp_path, monkeypatch):
    backend = _client_backends_remote()
    seen = []

    def forbidden(*args):
        pytest.fail("Cortex must not start or stop Ray")

    class Client:

        def __init__(self, config):
            seen.append(config)
            self.jobs = JobHandles(training="parent:training:0")

        def shutdown(self):
            seen.append("shutdown")

    monkeypatch.setattr(dss_driver, "_gateway_runtime", forbidden)
    monkeypatch.setattr(dss_driver, "ArcticClient", Client)
    payload = {"model_name": "model", "training_config": {"n_gpus": 8, "max_seq_len": 8}}
    with dss_driver.gateway(tmp_path, 8, backend=backend) as session:
        with dss_driver.running_job(session, payload) as job:
            assert job.job_id == "parent:training:0"
    assert seen[0].backend is backend
    assert seen[0].training.checkpoint_path is None
    assert seen[1] == "shutdown"


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_cortex_batch_preserves_microbatch_order_labels_and_context():
    chunks = [
        {"input_ids": torch.tensor([[index, index + 1]]), "labels": torch.tensor([[index + 1, -100]])}
        for index in range(4)
    ]
    body = {"batch": chunks, "meta": {"labels_are_shifted": True}, "processing": {"loss_fn": "sft"}}
    lowered = dss_driver.cortex_training_body(body)
    torch_assert_equal(lowered["kwargs"]["input_ids"], torch.cat([chunk["input_ids"] for chunk in chunks]))
    torch_assert_equal(lowered["kwargs"]["labels"], torch.cat([chunk["labels"] for chunk in chunks]))
    assert lowered["context"] == body["meta"]
    assert "processing" not in lowered
    assert body["processing"]["loss_fn"] == "sft"
    assert len(body["batch"]) == 4


@pytest.mark.usefixtures("_client_backends_no_network")
@pytest.mark.parametrize("batches", [[], [{"x": torch.ones(1)}, {"y": torch.ones(1)}]])
def test_client_backends_cortex_rejects_malformed_microbatch_lists(batches):
    with pytest.raises(ValueError, match="identical tensor keys"):
        dss_driver.cortex_training_body({"batch": batches})


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_onprem_batch_and_cortex_rpc_are_not_rewritten():
    body = {"batch": {"input_ids": torch.ones(1, 2)}}
    job = SimpleNamespace(client=SimpleNamespace(config=SimpleNamespace(backend=OnPremConfig())))
    assert dss_driver.training_body(job, body) is body
    rpc = {"args": (), "kwargs": {"input_ids": torch.ones(1, 2)}}
    assert dss_driver.cortex_training_body(rpc) is rpc


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_cortex_config_forwards_worker_settings_without_mutation():
    worker = {
        "debug": {"full_determinism_must_comply": False},
        "sp_size": 8,
        "prime_rl": {"fused_cross_entropy": "liger"},
        "enable_gradient_checkpointing": True,
    }
    config = CortexTrainingConfig(
        model_name="model",
        training_gpus=8,
        backend=_client_backends_remote(),
        training=TrainingConfig(ds_worker_config=worker, full_determinism=True),
        source_checkpoint_info={"checkpoint_id": "cp_saved", "source_job_id": "parent"},
    )
    wire = config.to_cortex()[0]
    assert wire["source_checkpoint_info"] == config.source_checkpoint_info
    assert wire["training_config"]["sp_size"] == 8
    assert wire["training_config"]["prime_rl"] == worker["prime_rl"]
    assert wire["training_config"]["debug"] == {"full_determinism": True, "full_determinism_must_comply": False}
    assert wire["training_config"]["activation_checkpointing"] is True
    assert "enable_gradient_checkpointing" not in wire["training_config"]
    assert "full_determinism" not in worker["debug"]


@pytest.mark.usefixtures("_client_backends_no_network")
@pytest.mark.parametrize("source", [{}, {"checkpoint_id": "cp_saved"}, {"checkpoint_id": "", "source_job_id": "p"}])
def test_client_backends_cortex_resume_requires_complete_source_identity(source):
    with pytest.raises(ValidationError, match="checkpoint_id and source_job_id"):
        CortexTrainingConfig(model_name="model", backend=_client_backends_remote(), source_checkpoint_info=source)


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_onprem_refuses_cortex_checkpoint_identity():
    with pytest.raises(ValidationError, match="requires CortexConfig"):
        CortexTrainingConfig(
            model_name="model", source_checkpoint_info={"checkpoint_id": "cp_saved", "source_job_id": "p"}
        )


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_cortex_transport_sends_hardware_and_lists_durable_checkpoints(monkeypatch):
    transport = CorrectnessCortexTransport(CortexConfig(base_url="http://cortex.test"))
    calls = []

    async def send(self, method, path, *, policy=_DEFAULT, **kwargs):
        assert asyncio.get_running_loop() is self._loop
        assert isinstance(policy, _SendPolicy)
        calls.append((method, path, kwargs))
        if method == "POST":
            return {"job_id": "parent"}
        return {"checkpoints": [{"checkpoint_id": "cp_saved"}]}

    monkeypatch.setattr(CortexTransport, "_send", send)
    monkeypatch.setattr(transport, "_wait_running", lambda: None)
    monkeypatch.setattr(transport, "_capture_sub_jobs", lambda: {"training": "parent:training:0"})
    config = CortexTrainingConfig(model_name="model", training_gpus=8, backend=transport.config, hardware="H200")
    try:
        transport.initialize(client_config=config)
        assert calls[0][2]["json"] == {"sub_job_configs": config.to_cortex(), "hardware": "H200"}
        assert transport.list_checkpoints() == [{"checkpoint_id": "cp_saved"}]
        assert calls[1][:2] == ("GET", f"{transport._prefix}/parent/checkpoints")
    finally:
        transport.shutdown()


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_cortex_shutdown_closes_session_when_cancel_fails(monkeypatch):
    transport = CorrectnessCortexTransport(_client_backends_remote())
    transport.job_id = "parent"
    closed = []
    session = _client_backends_session_with_close_spy(transport, monkeypatch, closed)
    attempts = []

    async def send(method, url, *, policy=_DEFAULT, **kwargs):
        assert asyncio.get_running_loop() is transport._loop
        assert isinstance(policy, _SendPolicy)
        assert method == "POST" and url.endswith("/parent:cancel")
        attempts.append(url)
        raise aiohttp.ClientConnectionError("cancel unavailable")

    monkeypatch.setattr(transport, "_send", send)
    transport.shutdown()
    transport.shutdown()
    assert transport.job_id is None
    assert attempts == [f"{transport._prefix}/parent:cancel"]
    assert closed == ["close"]
    assert session.closed and transport._session is None


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_inapplicable_check_is_not_also_a_cortex_failure(tmp_path, monkeypatch):
    cfg = SimpleNamespace(config_id="unit", n_gpus=8, training={}, at_gpu_width=lambda width: cfg)
    spec = _applicable_tests_spec([])
    results = runner.execute(
        cfg,
        spec,
        [],
        vocab_size=128,
        select=["single-step-optimizer"],
        client_backend=_client_backends_remote(),
        workdir=tmp_path,
    )
    assert len(results) == 1
    assert results[0].outcome is Outcome.INAPPLICABLE


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_cortex_unsupported_checks_fail_before_batch_or_reference(tmp_path, monkeypatch):
    cfg = SimpleNamespace(config_id="unit", n_gpus=8, training={}, at_gpu_width=lambda width: cfg)
    spec = _applicable_tests_spec(["single-step-optimizer"])
    monkeypatch.setattr(runner, "build_batch", lambda *args, **kwargs: pytest.fail("must reject before batches"))
    results = runner.execute(
        cfg,
        spec,
        [],
        vocab_size=128,
        select=["single-step-optimizer"],
        client_backend=_client_backends_remote(),
        workdir=tmp_path,
    )
    assert len(results) == 1
    assert results[0].outcome is Outcome.FAIL
    assert results[0].summary == "published Cortex fixture required"
    assert "--cortex-fixture" in results[0].reason


@pytest.mark.usefixtures("_client_backends_no_network")
@pytest.mark.parametrize("check", ["single-step-grads", "e2e-sft-train-validate"])
def test_client_backends_cortex_refuses_checks_requiring_local_model_identity(tmp_path, monkeypatch, check):
    cfg = SimpleNamespace(config_id="unit", n_gpus=8, training={}, at_gpu_width=lambda width: cfg)
    monkeypatch.setattr(runner, "build_batch", lambda *args, **kwargs: pytest.fail("must reject before batches"))
    monkeypatch.setattr(runner, "run_reference", lambda *args, **kwargs: pytest.fail("must reject before reference"))
    results = runner.execute(
        cfg,
        _applicable_tests_spec([check]),
        [],
        vocab_size=128,
        select=[check],
        client_backend=_client_backends_remote(),
        workdir=tmp_path,
    )
    assert len(results) == 1
    assert results[0].outcome is Outcome.FAIL
    if check == "single-step-grads":
        assert "--cortex-fixture" in results[0].reason
        assert "--backend onprem" not in results[0].reason
        assert results[0].summary == "published Cortex fixture required"
    else:
        assert results[0].reason == runner.cortex_unavailable_reason(runner.registered_tests()[check])
        assert "--backend onprem" not in results[0].reason


@pytest.mark.usefixtures("_client_backends_no_network")
@pytest.mark.parametrize("check", ["rl-weight-sync", "rl-router-replay"])
def test_client_backends_cortex_rl_refuses_local_instrumentation_before_check(tmp_path, monkeypatch, check):
    cfg = SimpleNamespace(
        config_id="unit", n_gpus=8, training={}, attention_implementation="sdpa", at_gpu_width=lambda width: cfg
    )
    monkeypatch.setattr(runner, "rl_gpu_count", lambda cfg: 16)
    tests = runner.registered_tests()
    monkeypatch.setattr(tests[check], "fn", lambda ctx: pytest.fail("must reject before tokenizer or client"))
    results = runner.execute_rl(
        cfg,
        _applicable_tests_spec([check]),
        vocab_size=128,
        select=[check],
        client_backend=_client_backends_remote(),
        workdir=tmp_path,
    )
    assert len(results) == 1
    assert results[0].outcome is Outcome.FAIL
    assert results[0].reason == runner.cortex_unavailable_reason(tests[check])
    assert "--backend onprem" not in results[0].reason


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_cli_rejects_unsupported_cortex_checks_without_reading_model(tmp_path, monkeypatch):
    import transformers

    path = _cli_config_selection_write_config(tmp_path / "configs", "h200")
    spec = _applicable_tests_spec(["single-step-grads"])
    captured = []

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported Cortex check must not load local or catalog model metadata")

    monkeypatch.setattr(cli, "_frozen_spec", lambda cfg, path, arms, **kwargs: (spec, arms))
    monkeypatch.setattr(cli, "_catalog_vocab_size", forbidden)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", forbidden)
    monkeypatch.setattr(cli, "_finish", lambda results, *args, **kwargs: captured.extend(results) or 1)
    assert (
        cli.main(
            [
                "run",
                "--config",
                str(path),
                "--test",
                "single-step-grads",
                "--base-url",
                _client_backends_remote().base_url,
            ]
        )
        == 1
    )
    assert captured and all((result.outcome is Outcome.FAIL for result in captured))
    assert "--cortex-fixture" in captured[0].reason


@pytest.mark.usefixtures("_client_backends_no_network")
@pytest.mark.parametrize("with_fixture", [False, True])
def test_client_backends_cortex_cli_inference_runs_full_model_without_reduced_metadata(
    tmp_path, monkeypatch, with_fixture
):
    import transformers

    path = _cli_config_selection_write_config(tmp_path / "configs", "h200")
    cfg = load_config(path)
    test_id = "inference-checkpoint-loss"
    spec = _applicable_tests_spec([test_id])
    spec.model.cache_path = "/data-fast/unused-reduced-reference"
    observed = []
    vocab_requests = []
    finished = []
    metadata = {}

    def forbidden(*args, **kwargs):
        pytest.fail("full-model inference must not read reduced metadata, build reference batches, or run HF")

    def catalog_vocab(model_name):
        vocab_requests.append(model_name)
        assert model_name == cfg.model_name
        return 128

    def measure(ctx):
        observed.append(ctx)
        assert isinstance(ctx.client_backend, CortexConfig)
        assert ctx.cfg.model_name == cfg.model_name
        assert inference._full_model_name(ctx) == cfg.model_name
        assert ctx.vocab_size == 128
        assert ctx.batch_paths == {}
        assert ctx.spec.model.cache_path == spec.model.cache_path
        yield Result(test_id=test_id, config_id=ctx.config_id, outcome=Outcome.PASS)

    def execute(run_cfg, run_spec, arms, **kwargs):
        return runner.execute(run_cfg, run_spec, arms, workdir=tmp_path / f"width-{run_cfg.n_gpus}", **kwargs)

    def finish(results, *args, **kwargs):
        finished.extend(results)
        metadata.update(kwargs["execution"])
        return 0

    monkeypatch.setattr(cli, "_frozen_spec", lambda cfg, path, arms, **kwargs: (spec, []))
    monkeypatch.setattr(cli, "_catalog_vocab_size", catalog_vocab)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", forbidden)
    monkeypatch.setattr(runner, "build_batch", forbidden)
    monkeypatch.setattr(runner, "run_reference", forbidden)
    monkeypatch.setattr(runner.registered_tests()[test_id], "fn", measure)
    monkeypatch.setattr(cli, "execute", execute)
    monkeypatch.setattr(cli, "_finish", finish)
    fixture = {
        "model_name": "published-exact-reviewed-weights",
        "fixture_sha256": "fixture-hash",
        "publication": {"stage_path": "snow://fixture-stage/exact"},
    }
    monkeypatch.setattr(cortex_fixture, "load_fixture", lambda *args: fixture)
    args = [
        "run",
        "--config",
        str(path),
        "--test",
        test_id,
        "--gpu-pool",
        "16",
        "--base-url",
        _client_backends_remote().base_url,
    ]
    if with_fixture:
        args.extend(["--cortex-fixture", str(tmp_path / "receipt.json")])
    assert cli.main(args) == 0
    assert vocab_requests == [cfg.model_name]
    assert [ctx.cfg.n_gpus for ctx in observed] == [8, 16]
    assert [result.outcome for result in finished] == [Outcome.PASS, Outcome.PASS]
    assert [result.gpus for result in finished] == [8, 16]
    expected = {
        "backend": "cortex",
        "requested_model_name": fixture["model_name"] if with_fixture else cfg.model_name,
        "inference_model_name": cfg.model_name,
    }
    if with_fixture:
        expected.update(
            fixture_sha256=fixture["fixture_sha256"], fixture_stage_path=fixture["publication"]["stage_path"]
        )
    assert metadata == {cfg.config_id: expected}


@pytest.mark.usefixtures("_client_backends_no_network")
@pytest.mark.parametrize("include_inference", [False, True])
def test_client_backends_cortex_cli_fixture_provenance_keeps_e2e_and_full_inference_distinct(
    tmp_path, monkeypatch, include_inference
):
    path = _cli_config_selection_write_config(tmp_path / "configs", "h200")
    cfg = load_config(path)
    selected = ["e2e-sft-train-validate"]
    if include_inference:
        selected.append("inference-checkpoint-loss")
    spec = _applicable_tests_spec(selected)
    fixture = {
        "model_name": "published-exact-reviewed-weights",
        "fixture_sha256": "fixture-hash",
        "publication": {"stage_path": "snow://fixture-stage/exact"},
    }
    metadata = {}
    calls = []
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(cli, "_frozen_spec", lambda cfg, path, arms, **kwargs: (spec, []))
    monkeypatch.setattr(cortex_fixture, "load_fixture", lambda *args: fixture)
    monkeypatch.setattr(cli, "_run_vocab_size", lambda *args: 128)

    def execute(cfg, *args, **kwargs):
        calls.append(kwargs)
        return [Result(test_id, cfg.config_id, Outcome.PASS) for test_id in kwargs["select"]]

    monkeypatch.setattr(cli, "execute", execute)
    monkeypatch.setattr(cli, "_finish", lambda *args, **kwargs: metadata.update(kwargs["execution"]) or 0)
    args = [
        "run",
        "--config",
        str(path),
        "--base-url",
        _client_backends_remote().base_url,
        "--cortex-fixture",
        str(receipt),
    ]
    for test_id in selected:
        args.extend(["--test", test_id])
    assert cli.main(args) == 0
    expected = {
        "backend": "cortex",
        "requested_model_name": fixture["model_name"],
        "fixture_sha256": fixture["fixture_sha256"],
        "fixture_stage_path": fixture["publication"]["stage_path"],
    }
    if include_inference:
        expected["inference_model_name"] = cfg.model_name
    assert metadata == {cfg.config_id: expected}
    assert [call["select"] for call in calls] == [[test_id] for test_id in selected]
    assert all((call["cortex_fixture"] == receipt for call in calls))


@pytest.mark.usefixtures("_client_backends_no_network")
@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("with_fixture", [False, True])
def test_client_backends_inference_only_vocabulary_uses_full_model_on_both_backends(
    tmp_path, monkeypatch, remote, with_fixture
):
    import transformers

    cfg = load_config(_cli_config_selection_write_config(tmp_path / "configs", "h200"))
    test_id = "inference-checkpoint-loss"
    spec = _applicable_tests_spec([test_id])
    spec.model.cache_path = "/data-fast/unused-reduced-reference"
    observed = []
    backend = _client_backends_remote() if remote else OnPremConfig()
    fixture = {"model_name": "published-exact-reviewed-weights"} if with_fixture else None

    def catalog_vocab(model_name):
        observed.append(model_name)
        return 128

    monkeypatch.setattr(cli, "_catalog_vocab_size", catalog_vocab)
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("inference-only vocabulary must not read reduced metadata"),
    )
    assert cli._run_vocab_size(cfg, spec, [test_id], backend, fixture) == 128
    assert observed == [cfg.model_name]


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_cortex_e2e_only_keeps_reduced_model_vocabulary(tmp_path, monkeypatch):
    import transformers

    cfg = load_config(_cli_config_selection_write_config(tmp_path / "configs", "h200"))
    spec = _applicable_tests_spec(["e2e-sft-train-validate"])
    spec.model.cache_path = "/data-fast/unused-reduced-reference"
    observed = []

    def reduced_config(model_name, **kwargs):
        observed.append((model_name, kwargs))
        return SimpleNamespace(vocab_size=64)

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", reduced_config)
    monkeypatch.setattr(cli, "_catalog_vocab_size", lambda *args: pytest.fail("E2E must keep reduced vocabulary"))
    assert (
        cli._run_vocab_size(
            cfg,
            spec,
            ["e2e-sft-train-validate"],
            _client_backends_remote(),
            {"model_name": "published-exact-reviewed-weights"},
        )
        == 64
    )
    assert observed == [(spec.model.cache_path, {"trust_remote_code": True})]


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_cortex_sft_body_does_not_select_the_rl_processor():
    from arctic_platform import wire

    body = {
        "batch": {"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[1, 2, -100]])},
        "processing": {"loss_fn": "sft"},
    }
    lowered = dss_driver.cortex_training_body(body)
    assert "processing" not in lowered
    decoded = wire.loads(wire.dumps(lowered))
    torch_assert_equal(decoded["kwargs"]["labels"], body["batch"]["labels"])


@pytest.mark.usefixtures("_client_backends_no_network")
@pytest.mark.parametrize(
    "processing",
    [
        {"loss_fn": "sft_ce", "config": {"log_prob_micro_batch_size": 128}},
        {"loss_fn": "sft", "config": {"unknown": True}},
    ],
)
def test_client_backends_cortex_does_not_silently_drop_sft_processing_options(processing):
    with pytest.raises(ValueError, match="use --backend onprem"):
        dss_driver.cortex_training_body({"batch": {"input_ids": torch.ones(1, 2)}, "processing": processing})


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_remote_cli_checkpoint_uses_catalog_and_explicit_placement_pool(tmp_path, monkeypatch):
    path = _cli_config_selection_write_config(tmp_path / "configs", "h200")
    spec = _applicable_tests_spec(["checkpoint-resume-loss"])
    measured = []
    metadata = {}
    monkeypatch.setattr(cli, "_frozen_spec", lambda cfg, path, arms, **kwargs: (spec, arms))
    monkeypatch.setattr(cli, "_catalog_vocab_size", lambda name: 128)

    def execute(cfg, *args, **kwargs):
        measured.append((cfg, kwargs))
        return [Result("checkpoint-resume-loss", cfg.config_id, Outcome.PASS, gpus=cfg.n_gpus)]

    monkeypatch.setattr(cli, "execute", execute)
    monkeypatch.setattr(cli, "_finish", lambda *args, **kwargs: metadata.update(kwargs["execution"]) or 0)
    assert (
        cli.main(
            [
                "run",
                "--config",
                str(path),
                "--test",
                "checkpoint-resume-loss",
                "--gpu-pool",
                "16",
                "--base-url",
                _client_backends_remote().base_url,
            ]
        )
        == 0
    )
    assert [cfg.n_gpus for cfg, _ in measured] == [8, 16]
    assert all((isinstance(options["client_backend"], CortexConfig) for _, options in measured))
    assert all((options["vocab_size"] == 128 for _, options in measured))
    assert metadata["model-h200-train"] == {"backend": "cortex", "requested_model_name": measured[0][0].model_name}


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_hosted_alias_uses_the_same_cortex_run_path(monkeypatch):
    selected = []
    monkeypatch.setattr(cli, "cmd_run", lambda args: selected.append(args.backend) or 0)
    assert cli.main(["hosted", "--all-configs", "--test", "checkpoint-resume-loss"]) == 0
    assert selected == ["cortex"]


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_cortex_report_does_not_claim_the_reduced_fixture_was_measured(tmp_path, monkeypatch):
    import json

    from arctic_platform.correctness.harness import report

    monkeypatch.setattr(report, "collect", lambda: {"host": "client-host"})
    spec = _applicable_tests_spec(["checkpoint-resume-loss"])
    execution = {"unit": {"backend": "cortex", "requested_model_name": "catalog-model"}}
    report.write([], {"unit": spec}, tmp_path, execution=execution)
    markdown = (tmp_path / "report.md").read_text()
    assert "requested catalog model: `catalog-model`" in markdown
    assert "reviewed reference fixture (not loaded on Cortex)" in markdown
    assert "holding identical weights and data" not in markdown
    assert "client process, not the remote worker" in markdown
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["execution"] == execution
    assert payload["specs"]["unit"]["model"]["content_hash"] == spec.model.content_hash


@pytest.mark.usefixtures("_client_backends_no_network")
@pytest.mark.parametrize("case,microbatches", [("gas1", 1), ("gas4", 4)])
def test_client_backends_checkpoint_check_uses_three_real_cortex_clients(tmp_path, monkeypatch, case, microbatches):
    from arctic_platform import wire
    from arctic_platform.client import base
    from arctic_platform.correctness.checks import checkpoint_resume
    from arctic_platform.correctness.harness.config import load_config

    jobs = {}
    replies = {}
    events = []
    checkpoint_id = "cp_aabb"

    async def send(transport, method, url, *, policy=_DEFAULT, **kwargs):
        assert asyncio.get_running_loop() is transport._loop
        assert isinstance(policy, _SendPolicy)
        if method == "POST" and url == transport._prefix:
            body = kwargs["json"]
            config = body["sub_job_configs"][0]
            job_id = f"job-{len(jobs) + 1}"
            source = config.get("source_checkpoint_info")
            if source:
                assert source == {"checkpoint_id": checkpoint_id, "source_job_id": "job-2"}
                assert ("cancel", "job-2") in events
            jobs[job_id] = {
                "config": config,
                "hardware": body["hardware"],
                "steps": 10 if source else 0,
                "batches": [],
                "rates": [],
                "checkpoints": [],
            }
            events.append(("create", job_id))
            return {"job_id": job_id}
        job = jobs[transport.job_id]
        if method == "GET":
            if url.endswith("/checkpoints"):
                return {"checkpoints": job["checkpoints"]}
            if url == f"{transport._prefix}/{transport.job_id}":
                return {"status": "RUNNING"}
            return {"status": "REQUEST_STATE_DONE", "result": replies[url.rsplit("/", 1)[-1]]}
        if url.endswith(":cancel"):
            events.append(("cancel", transport.job_id))
            return {}
        operation = urlsplit(url).path.rsplit("/", 1)[-1]
        if operation == "forward-backward":
            frame = wire.decode_byte_chunks([kwargs["data"]], kind="request")
            body = wire.loads(frame)
            assert "processing" not in body
            job["batches"].append(body["kwargs"]["input_ids"])
            result = {"avg_loss": float(body["kwargs"]["input_ids"].sum()) + job["steps"]}
        elif operation == "step":
            job["steps"] += 1
            job["rates"].append(kwargs["json"]["learning_rate"])
            result = {}
        elif operation == "save":
            assert kwargs["json"]["checkpoint_type"] == "resumable"
            assert job["steps"] == 10
            job["checkpoints"].append({"checkpoint_id": checkpoint_id})
            result = {"stage_path": f"snow://stage/checkpoints/{checkpoint_id}/global_step10/"}
            events.append(("save", transport.job_id))
        else:
            pytest.fail(f"unexpected Cortex request: {method} {url}")
        request_id = f"req-{len(replies) + 1}"
        replies[request_id] = result
        return {"request_id": request_id}

    monkeypatch.setattr(base, "require_any_dep_group", lambda *args: None)
    monkeypatch.setattr(CortexTransport, "_send", send)
    cfg = load_config(_cli_config_selection_write_config(tmp_path / "configs", "h200")).at_gpu_width(16)
    arm = SimpleNamespace(name=case, global_batch_size=16 * microbatches, max_seq_len=8, dss_microbatches=microbatches)
    ctx = runner.RunContext(
        config_id="unit",
        cfg=cfg,
        spec=SimpleNamespace(tolerance_for=lambda test_id: 0.001),
        arms=[arm],
        attn_implementation=cfg.attention_implementation,
        workdir=tmp_path,
        batch_paths={},
        vocab_size=128,
        client_backend=_client_backends_remote(),
    )
    results = checkpoint_resume.run(ctx)
    assert [result.outcome for result in results] == [Outcome.PASS]
    assert results[0].metrics["max_abs_diff"] == 0
    assert events == [
        ("create", "job-1"),
        ("cancel", "job-1"),
        ("create", "job-2"),
        ("save", "job-2"),
        ("cancel", "job-2"),
        ("create", "job-3"),
        ("cancel", "job-3"),
    ]
    uninterrupted, first, resumed = jobs.values()
    assert [len(job["batches"]) for job in jobs.values()] == [20, 10, 10]
    for full, split in zip(uninterrupted["batches"], first["batches"] + resumed["batches"]):
        torch_assert_equal(full, split)
    for job in jobs.values():
        assert job["hardware"] == "H200"
        assert job["config"]["model_name"] == cfg.model_name
        assert job["config"]["training_config"]["n_gpus"] == 16
        assert job["config"]["training_config"].get("ds_config", {}).get("gradient_accumulation_steps", 1) == 1
        assert set(job["rates"]) == {checkpoint_resume.LEARNING_RATE}


@pytest.mark.usefixtures("_client_backends_no_network")
@pytest.mark.parametrize("path", ["hosted", "driver"])
@pytest.mark.parametrize("cancel_fails", [False, True])
def test_client_backends_remote_startup_failure_releases_job_and_session(tmp_path, monkeypatch, path, cancel_fails):
    from arctic_platform.correctness.harness import cortex as cortex_transport
    from arctic_platform.correctness.harness import hosted
    from arctic_platform.correctness.harness.config import load_config

    transport = CorrectnessCortexTransport(_client_backends_remote())
    events = []

    async def send(method, url, *, policy=_DEFAULT, **kwargs):
        assert asyncio.get_running_loop() is transport._loop
        assert isinstance(policy, _SendPolicy)
        if method == "POST" and url == transport._prefix:
            return {"job_id": "failed-job"}
        if method == "GET":
            return {"status": "FAILED", "reason": "no capacity"}
        assert url.endswith("/failed-job:cancel")
        events.append("cancel")
        if cancel_fails:
            raise aiohttp.ClientConnectionError("cancel unavailable")
        return {}

    session = _client_backends_session_with_close_spy(transport, monkeypatch, events)
    monkeypatch.setattr(transport, "_send", send)
    monkeypatch.setattr(cortex_transport, "CorrectnessCortexTransport", lambda *args, **kwargs: transport)
    with pytest.raises(RuntimeError, match="no capacity"):
        if path == "hosted":
            cfg = load_config(_cli_config_selection_write_config(tmp_path / "configs", "h200"))
            with hosted.HostedTransport(_client_backends_remote(), cfg) as hosted_transport:
                with hosted_transport.job():
                    pytest.fail("failed startup must not yield a job")
        else:
            payload = {"model_name": "model", "training_config": {"n_gpus": 8}}
            with dss_driver.gateway(tmp_path, 8, backend=_client_backends_remote()) as gateway_session:
                with dss_driver.running_job(gateway_session, payload):
                    pytest.fail("failed startup must not yield a job")
    assert events == ["cancel", "close"]
    assert transport.job_id is None
    assert session.closed and transport._session is None


# Cortex e2e.


_cortex_e2e_ROOT = Path(__file__).resolve().parents[1]

_cortex_e2e_CONFIG = _cortex_e2e_ROOT / "configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config"

_cortex_e2e_FULL_SNAPSHOT = "/data-fast/tmp/e2e-full-test/hf/hub/models--Qwen--Qwen3-8B/snapshots/pinned"


@pytest.fixture(autouse=False)
def _cortex_e2e_no_network(monkeypatch):

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


def _cortex_e2e_context(tmp_path, width=4):
    cfg = load_config(_cortex_e2e_CONFIG).at_gpu_width(width)
    return SimpleNamespace(
        cfg=cfg,
        spec=ConfigSpec.read(_cortex_e2e_ROOT / f"specs/{cfg.config_id}.json"),
        workdir=tmp_path / "work",
        attn_implementation=cfg.attention_implementation,
        client_backend=CortexConfig(base_url="http://unused"),
        target_model_name="ap-correctness/reviewed-fixture",
    )


def _cortex_e2e_replay(tmp_path):
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


@pytest.mark.usefixtures("_cortex_e2e_no_network")
@pytest.mark.parametrize("width", [4, 8])
@pytest.mark.parametrize("failure", [None, "backward", "forward"])
def test_cortex_e2e_e2e_replay_uses_validated_fixture_and_cleans_up(
    tmp_path, monkeypatch, _cortex_e2e_no_network, width, failure
):
    ctx = _cortex_e2e_context(tmp_path, width)
    before = json.dumps(ctx.cfg.training, sort_keys=True)
    recorded = _cortex_e2e_replay(tmp_path)
    expected_train, expected_validation = check.request_bodies(recorded, model_provider="huggingface")
    creates, cancels, backward, steps, forward, replies = ([], [], [], [], [], {})

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
    assert _cortex_e2e_no_network and all((transport.job_id is None for transport in _cortex_e2e_no_network))
    assert creates[0]["hardware"] == "H200"
    sub = creates[0]["sub_job_configs"][0]
    assert sub["model_name"] == ctx.target_model_name
    assert sub["training_config"]["n_gpus"] == width
    assert sub["training_config"]["ds_config"] == ctx.cfg.training["ds_config"]
    assert json.dumps(ctx.cfg.training, sort_keys=True) == before


@pytest.mark.usefixtures("_cortex_e2e_no_network")
@pytest.mark.parametrize("name", [None, ""])
def test_cortex_e2e_e2e_rejects_missing_fixture_before_allocation(tmp_path, monkeypatch, name):
    ctx = _cortex_e2e_context(tmp_path)
    ctx.target_model_name = name
    monkeypatch.setattr(check, "gateway", lambda *args, **kwargs: pytest.fail("must reject before allocation"))
    with pytest.raises(ValueError, match="validated published Cortex fixture"):
        check.run_dss_trajectory(ctx, _cortex_e2e_replay(tmp_path), tmp_path / "target")


def _cortex_e2e_full_context(tmp_path):
    ctx = _cortex_e2e_context(tmp_path)
    ctx.client_backend = CortexConfig(base_url="http://127.0.0.1:8003")
    ctx.local_gateway_full_model = True
    ctx.local_full_model_path = _cortex_e2e_FULL_SNAPSHOT
    ctx.target_model_name = "ap-correctness/not-the-full-model"
    return ctx


@pytest.mark.usefixtures("_cortex_e2e_no_network")
@pytest.mark.parametrize("full", [None, False, True])
def test_cortex_e2e_e2e_reference_command_selects_identity_without_changing_contract(tmp_path, monkeypatch, full):
    ctx = _cortex_e2e_full_context(tmp_path) if full else _cortex_e2e_context(tmp_path)
    if full is not None:
        ctx.local_gateway_full_model = full
    expected = _cortex_e2e_FULL_SNAPSHOT if full else ctx.spec.model.cache_path
    before_config = json.dumps(ctx.cfg.sub_jobs, sort_keys=True)
    before_spec = ctx.spec.to_json()
    inspected = []
    monkeypatch.setattr(check, "uses_mixer_packing", lambda path: inspected.append(path) or False)
    recorded = _cortex_e2e_replay(tmp_path)
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


@pytest.mark.usefixtures("_cortex_e2e_no_network")
@pytest.mark.parametrize("full", [False, True])
def test_cortex_e2e_e2e_tokenizer_uses_selected_reference_identity(tmp_path, monkeypatch, full):
    ctx = _cortex_e2e_full_context(tmp_path) if full else _cortex_e2e_context(tmp_path)
    expected = _cortex_e2e_FULL_SNAPSHOT if full else ctx.spec.model.cache_path
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


@pytest.mark.usefixtures("_cortex_e2e_no_network")
@pytest.mark.parametrize("failure", [None, "backward", "forward"])
def test_cortex_e2e_full_e2e_requests_original_model_and_preserves_cleanup(tmp_path, monkeypatch, failure):
    ctx = _cortex_e2e_full_context(tmp_path)
    ctx.target_model_name = None
    before = json.dumps(ctx.cfg.training, sort_keys=True)
    recorded = _cortex_e2e_replay(tmp_path)
    created, exited, trained = ([], [], [])
    session, job = (object(), object())

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


@pytest.mark.usefixtures("_cortex_e2e_no_network")
@pytest.mark.parametrize("path", [None, "", "/code/not-fast/full-model"])
def test_cortex_e2e_full_e2e_rejects_missing_or_nonfast_snapshot_before_allocation(tmp_path, monkeypatch, path):
    ctx = _cortex_e2e_full_context(tmp_path)
    ctx.local_full_model_path = path
    monkeypatch.setattr(check, "gateway", lambda *args, **kwargs: pytest.fail("must reject before allocation"))
    with pytest.raises(ValueError, match="validated local full-model snapshot|/data-fast"):
        check.run_dss_trajectory(ctx, _cortex_e2e_replay(tmp_path), tmp_path / "target")


@pytest.mark.usefixtures("_cortex_e2e_no_network")
@pytest.mark.parametrize(
    "backend",
    [
        CortexConfig(base_url="https://hosted.example"),
        CortexConfig(base_url="http://other-host:8003"),
        CortexConfig(base_url="http://127.0.0.1:8003", host="hosted.example"),
        OnPremConfig(),
    ],
)
def test_cortex_e2e_full_e2e_never_bypasses_hosted_or_non_gateway_guards(tmp_path, monkeypatch, backend):
    ctx = _cortex_e2e_full_context(tmp_path)
    ctx.client_backend = backend
    monkeypatch.setattr(check, "gateway", lambda *args, **kwargs: pytest.fail("must reject before allocation"))
    with pytest.raises(ValueError, match="local Cortex gateway|HTTP loopback"):
        check.run_dss_trajectory(ctx, _cortex_e2e_replay(tmp_path), tmp_path / "target")


# Cortex model identity.


_cortex_model_identity_ROOT = Path(__file__).resolve().parents[1]

_cortex_model_identity_CONFIG = _cortex_model_identity_ROOT / "configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config"

_cortex_model_identity_SPEC = _cortex_model_identity_ROOT / "specs/qwen3-8b-h200-train-sft-full-4gpus-64k.json"


def test_cortex_model_identity_validated_fixture_reaches_e2e_context_without_changing_reference(tmp_path, monkeypatch):
    test_id = "e2e-sft-train-validate"
    cfg = load_config(_cortex_model_identity_CONFIG)
    spec = Spec.read(_cortex_model_identity_SPEC)
    entry = runner.registered_tests()[test_id]
    observed = []

    def measure(ctx):
        observed.append(ctx)
        assert ctx.target_model_name == "published-exact-reviewed-weights"
        assert ctx.spec.model.cache_path == spec.model.cache_path
        yield Result(test_id=test_id, config_id=ctx.config_id, outcome=Outcome.PASS)

    monkeypatch.setattr(entry, "fn", measure)
    monkeypatch.setattr(
        cortex_fixture, "load_fixture", lambda *args: {"model_name": "published-exact-reviewed-weights"}
    )
    results = runner.execute(
        cfg,
        spec,
        [],
        vocab_size=128,
        select=[test_id],
        client_backend=CortexConfig(base_url="http://unused"),
        cortex_fixture=tmp_path / "receipt.json",
        workdir=tmp_path,
    )
    assert len(observed) == 1
    assert [result.outcome for result in results] == [Outcome.PASS]


def test_cortex_model_identity_cortex_runner_inference_without_fixture_uses_full_configured_model(
    tmp_path, monkeypatch
):
    import transformers

    cfg = load_config(_cortex_model_identity_CONFIG)
    test_id = "inference-checkpoint-loss"
    spec = _applicable_tests_spec([test_id])
    spec.model.cache_path = "/data-fast/unused-reduced-reference"
    observed = []

    def forbidden(*args, **kwargs):
        pytest.fail("inference without a fixture must not read reduced metadata or prepare HF reference work")

    def measure(ctx):
        observed.append(ctx)
        assert ctx.target_model_name is None
        assert ctx.cfg.model_name == cfg.model_name
        assert inference._full_model_name(ctx) == cfg.model_name
        assert ctx.batch_paths == {}
        yield Result(test_id=test_id, config_id=ctx.config_id, outcome=Outcome.PASS)

    monkeypatch.setattr(cortex_fixture, "load_fixture", forbidden)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", forbidden)
    monkeypatch.setattr(runner, "build_batch", forbidden)
    monkeypatch.setattr(runner, "run_reference", forbidden)
    monkeypatch.setattr(runner.registered_tests()[test_id], "fn", measure)
    results = runner.execute(
        cfg,
        spec,
        [],
        vocab_size=128,
        select=[test_id],
        client_backend=CortexConfig(base_url="http://unused"),
        workdir=tmp_path,
    )
    assert len(observed) == 1
    assert [result.outcome for result in results] == [Outcome.PASS]


def test_cortex_model_identity_mixed_cortex_run_keeps_e2e_fixture_but_inference_ignores_it(tmp_path, monkeypatch):
    import transformers

    cfg = load_config(_cortex_model_identity_CONFIG)
    test_ids = ["e2e-sft-train-validate", "inference-checkpoint-loss"]
    spec = _applicable_tests_spec(test_ids)
    spec.model.cache_path = "/data-fast/unused-reduced-reference"
    observed = {}

    def forbidden(*args, **kwargs):
        pytest.fail("whole-run dispatch must not read reduced metadata or prepare HF reference work")

    def measure(test_id):

        def check(ctx):
            observed[test_id] = ctx
            assert ctx.target_model_name == "published-exact-reviewed-weights"
            assert ctx.spec.model.cache_path == spec.model.cache_path
            if test_id == "inference-checkpoint-loss":
                assert inference._full_model_name(ctx) == cfg.model_name
            yield Result(test_id=test_id, config_id=ctx.config_id, outcome=Outcome.PASS)

        return check

    for test_id in test_ids:
        monkeypatch.setattr(runner.registered_tests()[test_id], "fn", measure(test_id))
    monkeypatch.setattr(
        cortex_fixture, "load_fixture", lambda *args: {"model_name": "published-exact-reviewed-weights"}
    )
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", forbidden)
    monkeypatch.setattr(runner, "build_batch", forbidden)
    monkeypatch.setattr(runner, "run_reference", forbidden)
    results = runner.execute(
        cfg,
        spec,
        [],
        vocab_size=128,
        select=test_ids,
        client_backend=CortexConfig(base_url="http://unused"),
        cortex_fixture=tmp_path / "receipt.json",
        workdir=tmp_path,
    )
    assert set(observed) == set(test_ids)
    assert observed[test_ids[0]] is observed[test_ids[1]]
    assert {result.test_id: result.outcome for result in results} == {test_id: Outcome.PASS for test_id in test_ids}


@pytest.mark.parametrize("target_model_name", [None, "published-exact-reviewed-weights"])
def test_cortex_model_identity_cortex_sampling_checkpoint_identity_uses_full_model_not_e2e_fixture(target_model_name):
    cfg = load_config(_cortex_model_identity_CONFIG)
    backend = CortexConfig(base_url="http://unused")
    ctx = SimpleNamespace(
        cfg=cfg,
        spec=SimpleNamespace(model=SimpleNamespace(cache_path="/data-fast/unused-reduced-reference")),
        client_backend=backend,
        target_model_name=target_model_name,
    )
    config = inference.cortex_sampling_config(ctx, backend, "cp_full_saved", "full-training-job", 10240)
    assert config.hardware == "H200"
    assert config.backend == backend
    payload = config.to_cortex()[0]
    assert payload["model_name"] == cfg.model_name
    assert payload["source_checkpoint_info"] == {
        "checkpoint_id": "cp_full_saved",
        "source_job_id": "full-training-job",
    }


@pytest.mark.parametrize("remote", [False, True])
def test_cortex_model_identity_e2e_target_identity_hardware_and_backend(tmp_path, monkeypatch, remote):
    cfg = load_config(_cortex_model_identity_CONFIG)
    spec = Spec.read(_cortex_model_identity_SPEC)
    backend = CortexConfig(base_url="http://unused") if remote else OnPremConfig()
    ctx = SimpleNamespace(
        cfg=cfg,
        spec=spec,
        attn_implementation="sdpa",
        client_backend=backend,
        target_model_name="published-exact-reviewed-weights" if remote else None,
    )
    replay = SimpleNamespace(train_batch_paths=[tmp_path / "batch.pt"])
    captured = []

    @contextmanager
    def gateway(directory, slots, **kwargs):
        assert slots == cfg.n_gpus
        assert kwargs["backend"] is backend
        yield "session"

    @contextmanager
    def running_job(session, payload):
        captured.append(payload)
        yield "job"

    monkeypatch.setattr(
        e2e,
        "request_bodies",
        lambda *args, **kwargs: (
            [{"train": 1}],
            {"held-out": 1, "processing": {"loss_fn": "sft", "return_logprobs": True}},
        ),
    )
    monkeypatch.setattr(e2e, "gateway", gateway)
    monkeypatch.setattr(e2e, "running_job", running_job)
    monkeypatch.setattr(e2e, "fwd_bwd_step", lambda *args, **kwargs: SimpleNamespace(avg_loss=2.0))

    def forward_logprobs(session, job, body):
        assert ("return_logprobs" in body["processing"]) is not remote
        return torch.tensor([[-1.0]])

    monkeypatch.setattr(e2e, "forward_logprobs", forward_logprobs)
    losses, _ = e2e.run_dss_trajectory(ctx, replay, tmp_path)
    assert losses == [2.0]
    assert len(captured) == 1
    assert captured[0]["model_name"] == (ctx.target_model_name if remote else spec.model.cache_path)
    if remote:
        assert captured[0]["hardware"] == "H200"
    else:
        assert "hardware" not in captured[0]


def test_cortex_model_identity_cortex_e2e_without_published_identity_never_creates_job(tmp_path, monkeypatch):
    cfg = load_config(_cortex_model_identity_CONFIG)
    ctx = SimpleNamespace(
        cfg=cfg,
        spec=Spec.read(_cortex_model_identity_SPEC),
        attn_implementation="sdpa",
        client_backend=CortexConfig(base_url="http://unused"),
        target_model_name=None,
    )
    monkeypatch.setattr(e2e, "gateway", lambda *args, **kwargs: pytest.fail("must reject before allocation"))
    monkeypatch.setattr(e2e, "request_bodies", lambda *args, **kwargs: ([], {}))
    with pytest.raises(ValueError, match="published Cortex fixture"):
        e2e.run_dss_trajectory(ctx, SimpleNamespace(), tmp_path)


# Cortex port.

_driver_reviewed_root = Path(__file__).resolve().parents[1]
_driver_reviewed_rl_config = _driver_reviewed_root / "configs/qwen3.8-flash-next/h200/train-rl-32gpus-sp16-256k.config"


@pytest.mark.parametrize("weight_format", [None, "vllm", "hf", "lora"])
def test_driver_reviewed_build_payload_keeps_format_outside_worker_config(weight_format):
    cfg = load_config(_driver_reviewed_root / "configs/qwen3.8-flash-next/h200/train-sft-full-8gpus-sp8-256k.config")
    before = deepcopy(cfg.training)
    payload = dss_driver.build_payload(cfg.training, cfg.model_name, cfg.native.seed, weight_format=weight_format)
    assert payload.get("weight_format") == weight_format
    assert ("weight_format" in payload) == (weight_format is not None)
    assert "weight_format" not in payload["training_config"]
    assert cfg.training == before


@pytest.mark.parametrize("weight_format", ["vllm", "hf", "lora"])
def test_driver_reviewed_training_preserves_explicit_format_without_mutating_payload(weight_format):
    cfg = load_config(_driver_reviewed_rl_config)
    payload = deepcopy(cfg.sub_job)
    payload["weight_format"] = weight_format
    before = deepcopy(payload)
    session = GatewaySession(Path("."), 32, CortexConfig(base_url="http://unused"))
    native = dss_driver._training_client_config(session, payload)
    assert native.training.weight_format == weight_format
    assert native.to_cortex()[0]["weight_format"] == weight_format
    assert payload == before


@pytest.mark.parametrize("job_type", ["sampling", "log_prob"])
def test_driver_reviewed_inference_preserves_format_topology_and_independent_replay_cache(job_type):
    cfg = load_config(_driver_reviewed_rl_config)
    payload = dss_driver.rl_sampling_payload(cfg, "/data-fast/models/reviewed")
    payload["job_type"] = job_type
    if job_type == "log_prob":
        payload["inference_config"]["n_gpus"] = 8
        payload["inference_config"].pop("num_replicas")
        payload["inference_config"].pop("router_replay")
    before = deepcopy(payload)
    native = dss_driver._inference_client_config(payload, backend=CortexConfig(base_url="http://unused"))
    sub = native.to_cortex()[0]
    assert isinstance(native, CortexTrainingConfig)
    assert native.sampling.weight_format == sub["weight_format"] == "vllm"
    assert sub["inference_config"]["vllm_config"]["tensor_parallel_size"] == 8
    assert sub["inference_config"]["n_gpus"] == (16 if job_type == "sampling" else 8)
    if job_type == "sampling":
        assert sub["inference_config"]["num_replicas"] == 2
        assert sub["inference_config"]["router_replay"]["max_cache_bytes"] == 17179869184
    else:
        assert "num_replicas" not in sub["inference_config"]
        assert "router_replay" not in sub["inference_config"]
    assert payload == before


@pytest.mark.parametrize("failure", [None, "bootstrap", "body"])
def test_driver_reviewed_rl_bootstraps_before_yield_and_always_shuts_down(monkeypatch, tmp_path, failure):
    cfg = load_config(_driver_reviewed_rl_config)
    original = cfg.native.model_dump()
    payload = dss_driver.build_payload(cfg.training, "/data-fast/models/reviewed", cfg.native.seed)
    before = deepcopy(payload)
    events = []

    class Client:
        def __init__(self, native):
            events.append("create")
            assert isinstance(native, CortexTrainingConfig)
            assert native.model_name == "/data-fast/models/reviewed"
            assert native.training_gpus == native.sampling_gpus == 16
            subs = native.to_cortex()
            assert [sub["weight_format"] for sub in subs] == ["vllm", "vllm"]
            sampling = next(sub["inference_config"] for sub in subs if sub["job_type"] == "sampling")
            assert sampling["num_replicas"] == 2
            assert sampling["vllm_config"]["tensor_parallel_size"] == 8
            assert sampling["router_replay"]["max_cache_bytes"] == 17179869184

        def bootstrap_router_replay(self, *, max_cache_bytes):
            assert max_cache_bytes == 2147483648
            events.append("bootstrap")
            if failure == "bootstrap":
                raise RuntimeError("bootstrap failed")

        def shutdown(self):
            events.append("shutdown")

    monkeypatch.setattr(dss_driver, "RLClient", Client)

    def run():
        with dss_driver.running_rl_client(
            cfg, payload, "/data-fast/models/reviewed", tmp_path, backend=CortexConfig(base_url="http://unused")
        ):
            events.append("body")
            if failure == "body":
                raise RuntimeError("body failed")

    if failure is None:
        run()
    else:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            run()
    assert events == ["create", "bootstrap", *(["body"] if failure != "bootstrap" else []), "shutdown"]
    assert cfg.native.model_dump() == original
    assert payload == before


@pytest.mark.parametrize("disabled_side", ["training", "sampling"])
def test_driver_reviewed_rl_does_not_bootstrap_when_either_side_is_disabled(monkeypatch, tmp_path, disabled_side):
    cfg = load_config(_driver_reviewed_rl_config)
    payload = dss_driver.build_payload(cfg.training, cfg.model_name, cfg.native.seed)
    if disabled_side == "training":
        payload["training_config"]["router_replay"]["enabled"] = False
    else:
        cfg.native.sampling.arctic_inference_config["router_replay"]["enabled"] = False
    events = []

    class Client:
        def __init__(self, native):
            events.append("create")

        def bootstrap_router_replay(self, **kwargs):
            pytest.fail("disabled replay must not bootstrap")

        def shutdown(self):
            events.append("shutdown")

    monkeypatch.setattr(dss_driver, "RLClient", Client)
    with dss_driver.running_rl_client(
        cfg, payload, cfg.model_name, tmp_path, backend=CortexConfig(base_url="http://unused")
    ):
        events.append("body")
    assert events == ["create", "body", "shutdown"]


def test_driver_reviewed_rl_revalidates_incompatible_formats_before_client_creation(monkeypatch, tmp_path):
    cfg = load_config(_driver_reviewed_rl_config)
    payload = dss_driver.build_payload(cfg.training, cfg.model_name, cfg.native.seed)
    payload["weight_format"] = "hf"
    monkeypatch.setattr(dss_driver, "RLClient", lambda config: pytest.fail("invalid config must not create jobs"))
    with pytest.raises(ValueError, match="weight formats must agree"):
        with dss_driver.running_rl_client(
            cfg, payload, cfg.model_name, tmp_path, backend=CortexConfig(base_url="http://unused")
        ):
            pytest.fail("invalid config must not yield")


def test_driver_reviewed_rl_rejects_unsupported_onprem_fields_before_transport(monkeypatch):
    cfg = load_config(_driver_reviewed_rl_config)
    payload = dss_driver.build_payload(cfg.training, cfg.model_name, cfg.native.seed)
    monkeypatch.setattr(
        cortex_fixture, "initialize_or_cleanup", lambda *args, **kwargs: pytest.fail("must not create jobs")
    )
    with pytest.raises(ValueError, match="require the Cortex correctness route"):
        with dss_driver.running_rl_client(
            cfg, payload, cfg.model_name, Path("/data-fast/tmp/reviewed-rl-cpu-control"), backend=OnPremConfig()
        ):
            pytest.fail("unsupported fields must not yield")


@pytest.mark.parametrize("job_type", ["training", "sampling"])
def test_driver_reviewed_rejects_unsupported_format_before_gateway_runtime(monkeypatch, job_type):
    cfg = load_config(_driver_reviewed_rl_config)
    payload = (
        deepcopy(cfg.sub_job)
        if job_type == "training"
        else dss_driver.rl_sampling_payload(cfg, "/data-fast/models/reviewed")
    )
    session = GatewaySession(Path("/data-fast/tmp/reviewed-prestart-control"), 32, OnPremConfig())
    monkeypatch.setattr(
        dss_driver, "_gateway_runtime", lambda *args: pytest.fail("invalid recipe must not touch the runtime")
    )
    monkeypatch.setattr(dss_driver, "ArcticClient", lambda *args: pytest.fail("invalid recipe must not create jobs"))
    with pytest.raises(ValueError, match="require the Cortex correctness route"):
        with dss_driver.running_job(session, payload):
            pytest.fail("unsupported fields must not yield")


_cortex_port_ROOT = Path(__file__).resolve().parents[1]

_cortex_port_CONFIG = _cortex_port_ROOT / "configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config"


@pytest.fixture(autouse=False)
def _cortex_port_no_network(monkeypatch):
    monkeypatch.setattr(
        aiohttp.ClientSession, "request", lambda *args, **kwargs: pytest.fail("unit tests must not access the network")
    )
    monkeypatch.setattr(
        requests.Session, "request", lambda *args, **kwargs: pytest.fail("unit tests must not access the network")
    )


@pytest.mark.usefixtures("_cortex_port_no_network")
def test_cortex_port_retained_config_loads_without_changing_the_reviewed_identity():
    original = _cortex_port_CONFIG.read_bytes()
    cfg = load_config(_cortex_port_CONFIG)
    spec = json.loads((_cortex_port_ROOT / f"specs/{cfg.config_id}.json").read_text())
    assert config_checksum(cfg) == spec["config_checksum"]
    assert _cortex_port_CONFIG.read_bytes() == original


@pytest.mark.usefixtures("_cortex_port_no_network")
@pytest.mark.parametrize("protocol", ["http", "other"])
def test_cortex_port_config_intake_does_not_ignore_unknown_protocols(tmp_path, protocol):
    raw = json.loads(_cortex_port_CONFIG.read_text())
    raw["backend"]["protocol"] = protocol
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.usefixtures("_cortex_port_no_network")
def test_cortex_port_cortex_training_preserves_dss_extra_fields_and_determinism():
    training = {
        "n_gpus": 8,
        "max_seq_len": 32,
        "train_batch_size": 8,
        "optimizer": {"name": "AdamW", "lr": 0.0001},
        "sp_size": 8,
        "prime_rl": {"fused_cross_entropy": "liger"},
        "activation_checkpointing": True,
        "custom_worker_option": {"value": 3},
    }
    before = json.dumps(training, sort_keys=True)
    payload = build_payload(training, "published-fixture", 42)
    session = GatewaySession(Path("."), 8, CortexConfig(base_url="http://unused"))
    cfg = _training_client_config(session, payload)
    wire = cfg.to_cortex()[0]
    for key in ("sp_size", "prime_rl", "custom_worker_option", "activation_checkpointing", "mb_spec", "debug"):
        assert wire["training_config"][key] == payload["training_config"][key]
    assert wire["training_config"]["debug"]["gradient_norms_per_param"] is True
    assert wire["training_config"]["debug"]["full_determinism"] is True
    assert "enable_gradient_checkpointing" not in wire["training_config"]
    assert json.dumps(training, sort_keys=True) == before


@pytest.mark.usefixtures("_cortex_port_no_network")
def test_cortex_port_hosted_config_does_not_depend_on_removed_public_backend_fields():
    cfg = load_config(_cortex_port_CONFIG)
    config = hosted.training_client_config(cfg, CortexConfig(base_url="http://unused"))
    assert config.hardware == "H200"
    assert config.to_cortex()[0]["training_config"]["debug"]["full_determinism"] is True


@pytest.mark.usefixtures("_cortex_port_no_network")
@pytest.mark.parametrize("order", ["batched", "interleaved"])
@pytest.mark.parametrize("startup_error", [False, True])
def test_cortex_port_reviewed_gradient_cases_send_dss_requests_not_onprem_workarounds(
    tmp_path, monkeypatch, order, startup_error
):
    from arctic_platform.correctness.harness import cortex as cortex_fixture
    from arctic_platform.correctness.harness import reference_records

    cfg = load_config(_cortex_port_CONFIG)
    spec = ConfigSpec.read(_cortex_port_ROOT / f"specs/{cfg.config_id}.json")
    references = []
    targets = []
    creates = []
    cancels = []
    replies = {}

    def reference(ctx, identity):
        assert identity["check_id"] == "single-step-grads"
        name = identity["recipe"]["synthetic_inputs"]["batch"]["name"]
        references.append(load_batch(ctx.batch_paths[name]))
        return {"loss": 2.0, "grad_norms": {"weight": 1.0}, "microbatches": 1, "active_tokens": 1}

    async def send(transport, method, url, **kwargs):
        if method == "POST" and url == transport._prefix:
            creates.append(kwargs["json"])
            return {"job_id": f"job-{len(creates)}"}
        if method == "GET" and url == f"{transport._prefix}/{transport.job_id}":
            if startup_error:
                return {"status": "failed", "reason": "unsupported native worker options: ['mb_spec']"}
            return {"status": "RUNNING"}
        if method == "GET":
            return {"status": "REQUEST_STATE_DONE", "result": replies[url.rsplit("/", 1)[-1]]}
        if url.endswith(":cancel"):
            cancels.append(transport.job_id)
            return {}
        if url.endswith("/forward-backward"):
            body = wire.loads(wire.decode_byte_chunks([kwargs["data"]], kind="request"))
            assert "processing" not in body
            targets.append(body["kwargs"])
            result = {"avg_loss": 2.0}
        elif url.endswith("/step"):
            assert kwargs["json"]["learning_rate"] == 0.0
            result = {"gradient_norms_per_param": {"weight": 1.0}}
        else:
            pytest.fail(f"unexpected request: {method} {url}")
        request_id = f"request-{len(replies)}"
        replies[request_id] = result
        return {"request_id": request_id}

    monkeypatch.setattr(cortex_fixture, "load_fixture", lambda *args: {"model_name": "published-reviewed-fixture"})
    monkeypatch.setattr(runner, "run_reference", lambda *a, **k: pytest.fail("regression must not run HF"))
    monkeypatch.setattr(reference_records, "model_identity", lambda *a: {"sha256": "unit-model"})
    monkeypatch.setattr(reference_records, "load_record", reference)
    monkeypatch.setattr(runner, "uses_mixer_packing", lambda *args: False)
    assert not hasattr(runner, "ap_batch_processing_options")
    monkeypatch.setattr(CortexTransport, "_send", send)
    results = runner.execute(
        cfg,
        spec,
        arms_for(cfg.max_seq_len, cfg.n_gpus),
        vocab_size=128,
        select=["single-step-grads"],
        client_backend=CortexConfig(base_url="http://unused"),
        cortex_fixture=tmp_path / "receipt.json",
        workdir=tmp_path,
        order=order,
    )
    assert [result.arm for result in results] == ["gas1", "gas4"]
    if startup_error:
        assert all((result.outcome.value == "fail" for result in results))
        assert all(("execution error:" in result.summary and "mb_spec" in result.summary for result in results))
        assert all((not result.metrics and (not result.mismatches) for result in results))
        assert targets == []
    else:
        assert all((result.outcome.value == "pass" for result in results))
    assert cancels == ["job-1", "job-2"]
    for create in creates:
        assert create["hardware"] == "H200"
        sub = create["sub_job_configs"][0]
        assert sub["model_name"] == "published-reviewed-fixture"
        assert sub["training_config"]["ds_config"]["gradient_accumulation_steps"] == 1
        assert sub["training_config"]["debug"]["gradient_norms_per_param"] is True
    for target, reference_batch in zip(targets, references):
        torch.testing.assert_close(target["input_ids"], reference_batch.input_ids, rtol=0, atol=0)
        torch.testing.assert_close(target["labels"], reference_batch.labels, rtol=0, atol=0)
        torch.testing.assert_close(target["position_ids"], reference_batch.position_ids, rtol=0, atol=0)


def _cortex_port_mock_inference_inputs(monkeypatch, cfg):
    training = [
        gsm8k.Example(
            question=f"train-{index}",
            prompt=f"train-{index}:",
            completion=" answer",
            input_ids=(index + 1, 100, 101, 102),
            n_prompt=2,
        )
        for index in range(inference.TRAIN_STEPS * cfg.training["train_batch_size"])
    ]
    validation = [
        gsm8k.Example(
            question=f"held-{index}",
            prompt=f"held-{index}:",
            completion=" answer",
            input_ids=tuple(range(200 + index * 10, 204 + index * 10 + index % 2)),
            n_prompt=2,
        )
        for index in range(cfg.dp_size)
    ]
    tokenizer_paths = []
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)

    def load_tokenizer(path):
        tokenizer_paths.append(path)
        return tokenizer

    def read_rows(path, *, count=None):
        if path == inference.TRAIN_SPLIT:
            assert count == len(training)
            return training
        assert path == inference.VALIDATION_SPLIT
        assert count is None
        return validation

    monkeypatch.setattr(gsm8k, "load_tokenizer", load_tokenizer)
    monkeypatch.setattr(gsm8k, "read_rows", read_rows)
    monkeypatch.setattr(gsm8k, "tokenize_examples", lambda actual, rows: rows if actual is tokenizer else None)
    return (training, validation, tokenizer_paths)


@pytest.mark.usefixtures("_cortex_port_no_network")
@pytest.mark.parametrize(
    "mode,receipt",
    [
        ("success", "checkpoint_id"),
        ("success", "stage_path"),
        ("success", "catalog"),
        ("malformed_generate", "checkpoint_id"),
        ("generate_error", "checkpoint_id"),
        ("save_error", "checkpoint_id"),
        ("forward_error", "checkpoint_id"),
        ("sampling_start_error", "checkpoint_id"),
    ],
)
def test_cortex_port_managed_inference_uses_public_transport_and_cleans_up(tmp_path, monkeypatch, mode, receipt):
    cfg = load_config(_cortex_port_CONFIG)
    spec = ConfigSpec.read(_cortex_port_ROOT / f"specs/{cfg.config_id}.json")
    training, validation, tokenizer_paths = _cortex_port_mock_inference_inputs(monkeypatch, cfg)
    ctx = SimpleNamespace(
        cfg=cfg,
        spec=spec,
        config_id=cfg.config_id,
        workdir=tmp_path,
        attn_implementation=cfg.attention_implementation,
        client_backend=CortexConfig(base_url="http://unused"),
        target_model_name="ap-correctness/reviewed-fixture",
    )
    creates, cancels, backward, steps, forwards, generates = ([], [], [], [], [], [])
    transports, closed, replies = ([], [], {})
    saved = False
    checkpoint_id = "cp_1234abcd"
    original_close = CortexTransport._close_session

    async def close(transport):
        await original_close(transport)
        closed.append(transport)

    def forbidden(*args, **kwargs):
        pytest.fail("managed inference must not enter Ray or use a local checkpoint export/tokenizer write")

    monkeypatch.setattr(dss_driver, "_gateway_runtime", forbidden)
    monkeypatch.setattr(inference, "save_weights_only_checkpoint", forbidden)
    monkeypatch.setattr(inference, "materialize_checkpoint_tokenizer", forbidden)
    monkeypatch.setattr(inference, "_onprem_sampling_and_reload", forbidden)
    monkeypatch.setattr(inference.shutil, "rmtree", forbidden)
    monkeypatch.setattr(CortexTransport, "_close_session", close)

    async def send(transport, method, url, **kwargs):
        nonlocal saved
        path = url.split("?", 1)[0]
        if method == "POST" and url == transport._prefix:
            creates.append(kwargs["json"])
            transports.append(transport)
            assert len(creates) <= 2, "Cortex must not create a training reload diagnostic job"
            return {"job_id": f"job-{len(creates)}"}
        if path.endswith(":cancel"):
            cancels.append(transport.job_id)
            return {}
        if method == "GET" and path.endswith("/checkpoints"):
            return {"checkpoints": [{"checkpoint_id": checkpoint_id}] if saved else []}
        if method == "GET" and url == f"{transport._prefix}/{transport.job_id}":
            if mode == "sampling_start_error" and transport.job_id == "job-2":
                return {"status": "FAILED", "reason": "mock sampling startup failed"}
            role = creates[int(transport.job_id.rsplit("-", 1)[1]) - 1]["sub_job_configs"][0]["job_type"]
            return {"status": "RUNNING", "sub_jobs": [{"job_type": role, "sub_job_id": f"opaque-{role}-id"}]}
        if method == "GET":
            return {"status": "REQUEST_STATE_DONE", "result": replies[url.rsplit("/", 1)[-1]]}
        if path.endswith("/forward-backward"):
            body = wire.loads(wire.decode_byte_chunks([kwargs["data"]], kind="request"))
            assert "processing" not in body
            backward.append(body)
            result = {"avg_loss": 2.0}
        elif path.endswith("/step"):
            steps.append(kwargs["json"])
            result = {}
        elif path.endswith("/operation"):
            envelope = kwargs["json"]
            assert envelope["operation_type"] == "forward"
            assert envelope["sub_job_id"] == "opaque-training-id"
            body = wire.loads(base64.b64decode(envelope["payload"]["payload_b64"]))
            assert "processing" not in body
            assert body["context"] == {"labels_are_shifted": True}
            forwards.append(body)
            if mode == "forward_error":
                raise RuntimeError("mock forward failed")
            result = {"logprobs": torch.full(body["kwargs"]["input_ids"].shape, -1.25)}
        elif path.endswith("/save"):
            assert kwargs["json"]["checkpoint_type"] == "weights-only"
            assert kwargs["json"].get("path") is None
            assert kwargs["json"].get("export_hf") is not True
            if mode == "save_error":
                raise RuntimeError("mock save failed")
            saved = True
            result = {"checkpoint_id": "global_step10"}
            if receipt == "checkpoint_id":
                result["checkpoint_id"] = checkpoint_id
            elif receipt == "stage_path":
                result["stage_path"] = f"snow://experiment/run/checkpoints/{checkpoint_id}/weights-only"
        elif path.endswith("/generate"):
            assert cancels == ["job-1"], "sampling must use the durable save after source teardown"
            body = wire.loads(wire.decode_byte_chunks([kwargs["data"]], kind="request"))
            generates.append(body)
            if mode == "generate_error":
                raise RuntimeError("mock generate failed")
            rows = [
                {
                    "prompt_len": example.length,
                    "prompt_logprobs": [
                        None,
                        *[
                            {9999: {"logprob": -99.0}, str(token): {"logprob": -1.25}}
                            for token in example.input_ids[1:]
                        ],
                    ],
                    "token_ids": [9999],
                }
                for example in validation
            ]
            if mode == "malformed_generate":
                rows[0]["prompt_logprobs"][1] = None
            result = {"results": rows}
        else:
            pytest.fail(f"unexpected managed request: {method} {url}")
        request_id = f"request-{len(replies)}"
        replies[request_id] = result
        return {"request_id": request_id}

    monkeypatch.setattr(CortexTransport, "_send", send)
    result = inference.run(ctx)[0]
    expected_jobs = 1 if mode in {"save_error", "forward_error"} else 2
    assert cancels == [f"job-{index}" for index in range(1, expected_jobs + 1)]
    assert len(transports) == expected_jobs
    assert all(
        (transport.job_id is None and transport._session is None and (transport in closed) for transport in transports)
    )
    assert tokenizer_paths == [cfg.model_name]
    assert len(backward) == len(steps) == inference.TRAIN_STEPS == 10
    assert len(forwards) == 1
    assert all((step == {"learning_rate": cfg.training["optimizer"]["lr"]} for step in steps))
    assert not (tmp_path / inference.TEST_ID / "weights-only").exists()
    for create in creates:
        assert create["hardware"] == "H200"
        assert create["sub_job_configs"][0]["model_name"] == cfg.model_name
    assert creates[0]["sub_job_configs"][0]["job_type"] == "training"
    if expected_jobs == 2:
        sampling = creates[1]["sub_job_configs"][0]
        assert sampling["job_type"] == "sampling"
        assert sampling["source_checkpoint_info"] == {"checkpoint_id": checkpoint_id, "source_job_id": "job-1"}
        assert sampling["inference_config"]["n_gpus"] == 1
        assert sampling["inference_config"]["max_seq_len"] == cfg.max_tokens_per_mb + 1
    expected_validation = inference.pack_logit_aligned(gsm8k.build_batch("validation", validation, 0))
    torch.testing.assert_close(forwards[0]["kwargs"]["labels"], expected_validation["batch"]["labels"], rtol=0, atol=0)
    for body, batch in zip(backward, inference.training_batches(training, cfg.training["train_batch_size"], 0)):
        torch.testing.assert_close(body["kwargs"]["input_ids"], batch.input_ids, rtol=0, atol=0)
        torch.testing.assert_close(body["kwargs"]["labels"], batch.labels, rtol=0, atol=0)
    if generates:
        assert generates[0]["prompts"] == [example.prompt + example.completion for example in validation]
        assert generates[0]["sampling_params"] == {"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": 0}
    if mode == "success":
        assert result.outcome.value == "pass"
        assert result.metrics["reference_cross_entropy"] == result.metrics["target_cross_entropy"] == 1.25
        assert result.metrics["scored_answer_tokens"] == sum((example.answer_tokens for example in validation))
        assert result.metrics["train_steps"] == 10
        assert result.metrics["stated_criterion_abs"] == 0.001
    else:
        assert result.outcome.value == "fail"


@pytest.mark.usefixtures("_cortex_port_no_network")
def test_cortex_port_sampling_config_forwards_checkpoint_identity_without_mutating_training():
    cfg = load_config(_cortex_port_CONFIG)
    before = json.dumps(cfg.training, sort_keys=True)
    ctx = SimpleNamespace(cfg=cfg, target_model_name="ap-correctness/reviewed-fixture")
    backend = CortexConfig(base_url="http://unused")
    config = inference.cortex_sampling_config(ctx, backend, "cp_saved", "source-parent", 16384)
    assert config.backend == backend
    assert config.sampling_gpus == 1
    assert config.training_gpus == config.log_prob_gpus == 0
    assert config.hardware == "H200"
    sub = config.to_cortex()[0]
    assert sub["model_name"] == cfg.model_name
    assert sub["job_type"] == "sampling"
    assert sub["source_checkpoint_info"] == {"checkpoint_id": "cp_saved", "source_job_id": "source-parent"}
    assert sub["inference_config"]["vllm_config"]["enforce_eager"] is True
    assert json.dumps(cfg.training, sort_keys=True) == before


@pytest.mark.usefixtures("_cortex_port_no_network")
def test_cortex_port_onprem_inference_retains_local_export_and_reload_diagnostic(tmp_path, monkeypatch, capsys):
    cfg = load_config(_cortex_port_CONFIG)
    spec = ConfigSpec.read(_cortex_port_ROOT / f"specs/{cfg.config_id}.json")
    training, validation, tokenizer_paths = _cortex_port_mock_inference_inputs(monkeypatch, cfg)
    ctx = SimpleNamespace(
        cfg=cfg,
        spec=spec,
        config_id=cfg.config_id,
        workdir=tmp_path,
        attn_implementation=cfg.attention_implementation,
        client_backend=OnPremConfig(),
        target_model_name=None,
    )
    clients, steps, tokenizer_saves = ([], [], [])
    weights_dir = tmp_path / inference.TEST_ID / "weights-only"

    def forbidden(*args, **kwargs):
        pytest.fail("onprem must not use managed sampling or checkpoint IDs")

    class Client:

        def __init__(self, config):
            self.config = config
            self.shutdown_called = False
            self.jobs = JobHandles(training="training" if config.training_gpus else None, log_prob="log_prob")
            clients.append(self)

        def shutdown(self):
            self.shutdown_called = True

        def fwd_bwd(self, body):
            index = len(steps)
            expected = inference.pack(
                inference.training_batches(training, cfg.training["train_batch_size"], 0)[index],
                model_provider=cfg.training["model_provider"],
            )
            torch.testing.assert_close(body["batch"]["input_ids"], expected["batch"]["input_ids"], rtol=0, atol=0)
            torch.testing.assert_close(body["batch"]["labels"], expected["batch"]["labels"], rtol=0, atol=0)
            return {"avg_loss": 2.0}

        def step(self, learning_rate):
            steps.append(learning_rate)
            return {}

        def fwd_no_grad(self, body):
            assert body["processing"] == {"loss_fn": "sft", "return_logprobs": True}
            assert body["meta"] == {"labels_are_shifted": True}
            return {"logprobs": torch.full(body["batch"]["input_ids"].shape, -1.25)}

        def save_checkpoint(self, **kwargs):
            assert kwargs == {"path": str(weights_dir), "export_hf": True}
            weights_dir.mkdir(parents=True)
            (weights_dir / "config.json").write_text("{}")
            return {"hf_path": str(weights_dir)}

        def _call(self, request):
            assert request.op == "log-probs"
            assert request.body["prompts"] == [example.prompt for example in validation]
            assert request.body["completions"] == [example.completion for example in validation]
            assert request.body["top_k"] == 0
            return {
                "results": [
                    {"token_ids": list(example.input_ids[1:]), "logprobs": [-1.25] * (example.length - 1)}
                    for example in validation
                ]
            }

    monkeypatch.setattr(dss_driver, "ArcticClient", Client)
    monkeypatch.setattr(dss_driver, "_gateway_runtime", lambda *args: contextlib.nullcontext())
    monkeypatch.setattr(inference, "CorrectnessClient", forbidden)
    monkeypatch.setattr(inference, "save_and_resolve_checkpoint_id", forbidden)
    monkeypatch.setattr(
        inference, "materialize_checkpoint_tokenizer", lambda tokenizer, path: tokenizer_saves.append(path)
    )
    result = inference._run(ctx)
    assert result.outcome.value == "pass"
    assert len(clients) == 3
    assert [client.config.model_name for client in clients] == [cfg.model_name, str(weights_dir), str(weights_dir)]
    assert all((client.shutdown_called and isinstance(client.config.backend, OnPremConfig) for client in clients))
    assert steps == [cfg.training["optimizer"]["lr"]] * 10
    assert tokenizer_paths == [cfg.model_name]
    assert tokenizer_saves == [str(weights_dir)]
    assert result.metrics["reference_cross_entropy"] == result.metrics["target_cross_entropy"] == 1.25
    assert "INFERENCE_DIAGNOSTIC" in capsys.readouterr().out


# Hosted.


_hosted_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "qwen3-8b" / "h200" / "train-sft-full-4gpus-64k.config"
)

_hosted_CONNECTION = HostedConnection(host="server", database="DB", schema="SCHEMA", pat="not-a-token")

_hosted_LOCAL_CHECKPOINT_ID = "cp_00000000-0000-0000-0000-000000000001"

_hosted_LOCAL_STAGE = {
    "location_type": "S3",
    "bucket": "synthetic-control-bucket",
    "prefix": f"jobs/saving-parent/checkpoints/{_hosted_LOCAL_CHECKPOINT_ID}/",
}


def _hosted_config():
    return load_config(_hosted_CONFIG)


def _hosted_wire(cfg, **kwargs) -> dict:
    return hosted.training_client_config(cfg, CortexConfig(base_url="http://unused"), **kwargs).to_cortex()[0]


class _hosted_Client:
    """Enough of the hosted client to record what the transport asked it to do."""

    def __init__(self, *, losses=(), save_result=None, catalog=None, added=()) -> None:
        self.losses = list(losses)
        self.save_result = {} if save_result is None else dict(save_result)
        self.catalog = [dict(entry) for entry in catalog or []]
        self.added = [dict(entry) for entry in added]
        self.calls: list = []
        self.created: list = []
        self.cancelled: list = []
        self.closed = False
        self.transport = self
        self.jobs = JobHandles(training="job-1:training:0")

    def create(self, config):
        self.config = config
        self.created.append((config.to_cortex()[0], config.hardware))
        self.jobs = JobHandles(training=f"job-{len(self.created)}:training:0")
        return self

    def fwd_bwd(self, body):
        self.calls.append(("fwd-bwd", body))
        return {"avg_loss": self.losses.pop(0) if self.losses else 0.0}

    def step(self, learning_rate=None):
        self.calls.append(("step", learning_rate))
        return {"global_steps": len(self.calls)}

    def save_checkpoint(self, checkpoint_type=None, *, path=None):
        self.calls.append(("save", checkpoint_type) if path is None else ("save", checkpoint_type, path))
        self.catalog = self.catalog + self.added
        return self.save_result

    def load_checkpoint(self, *, path):
        self.calls.append(("load", path))
        return {"checkpoint_path": path, "checkpoint_tag": "global_step10", "version": 10}

    def list_checkpoints(self):
        return [dict(entry) for entry in self.catalog]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cancelled.append(str(self.jobs.training).split(":", 1)[0])
        self.closed = True


def test_hosted_the_created_sub_job_is_the_config_file_s_training_sub_job() -> None:
    """Whatever the operator wrote reaches the server: typed fields as themselves, the rest as passthrough."""
    cfg = _hosted_config()
    wire = _hosted_wire(cfg)
    assert wire["job_type"] == "training"
    assert wire["model_name"] == cfg.model_name
    assert wire["dtype"] == cfg.sub_job["dtype"]
    assert wire["seed"] == cfg.sub_job["seed"]
    for key, value in cfg.training.items():
        if key in {"optimizer", "debug"}:
            continue
        assert wire["training_config"][key] == value, key
    for key, value in cfg.training["optimizer"].items():
        assert wire["training_config"]["optimizer"]["name" if key == "type" else key] == value, key


def test_hosted_building_a_sub_job_leaves_the_loaded_config_alone() -> None:
    """The client canonicalizes the optimizer block it is handed, and the rest of the run reads that block."""
    cfg = _hosted_config()
    before = json.dumps(cfg.sub_job, sort_keys=True)
    _hosted_wire(cfg)
    assert json.dumps(cfg.sub_job, sort_keys=True) == before


def test_hosted_optimizer_state_loading_is_left_at_the_server_default() -> None:
    """A resume that restores the moments is what is wanted: both jobs read one config, so DP size is equal."""
    assert "load_optimizer_states" not in _hosted_wire(_hosted_config())["training_config"]


def test_hosted_an_ordinary_sub_job_carries_no_source_checkpoint() -> None:
    assert "source_checkpoint_info" not in _hosted_wire(_hosted_config())


def test_hosted_a_resumed_sub_job_names_the_checkpoint_and_the_job_that_saved_it() -> None:
    wire = _hosted_wire(_hosted_config(), source_checkpoint_info={"checkpoint_id": "cp_1", "source_job_id": "job-a"})
    source = wire["source_checkpoint_info"]
    assert source["checkpoint_id"] == "cp_1"
    assert source["source_job_id"] == "job-a"
    assert source.get("is_external", False) is False


def test_hosted_the_checkpoint_id_is_the_stage_path_segment() -> None:
    """The DeepSpeed tag beside it names a directory inside the saving job and resolves from nowhere else."""
    client = _hosted_Client(
        save_result={
            "stage_path": "snow://stage/run/checkpoints/cp_4c1f0a2b/global_step10/",
            "checkpoint_tag": "global_step10",
        }
    )
    assert save_and_resolve_checkpoint_id(client, "job-1", "resumable") == "cp_4c1f0a2b"
    assert ("save", "resumable") in client.calls


def test_hosted_the_checkpoint_id_falls_back_to_what_the_save_added_to_the_catalog() -> None:
    """The catalog's order is not specified, so the id is the entry that appeared, not the last listed."""
    client = _hosted_Client(
        save_result={"checkpoint_tag": "global_step10"},
        catalog=[{"checkpoint_id": "cp_older"}],
        added=[{"checkpoint_id": "cp_newer"}],
    )
    assert save_and_resolve_checkpoint_id(client, "job-1", "resumable") == "cp_newer"


def test_hosted_a_save_that_cannot_be_identified_is_an_error() -> None:
    client = _hosted_Client(save_result={}, catalog=[{"checkpoint_id": "cp_older"}])
    with pytest.raises(RuntimeError, match="cannot be identified"):
        save_and_resolve_checkpoint_id(client, "job-1", "resumable")


def test_hosted_one_iteration_is_a_forward_backward_then_a_step_at_the_requested_rate() -> None:
    """The loss is the forward-backward's, so it is read before the step that follows changes a parameter."""
    client = _hosted_Client(losses=[2.5])
    body = {"args": (), "kwargs": {}}
    loss = HostedTrainingJob(client, "job-1").fwd_bwd_step(body, learning_rate=0.001)
    assert loss == 2.5
    assert client.calls == [("fwd-bwd", body), ("step", 0.001)]


def test_hosted_a_forward_backward_that_reports_no_loss_is_an_error() -> None:

    class Silent(_hosted_Client):

        def fwd_bwd(self, body):
            return {"metrics": {}}

    with pytest.raises(RuntimeError, match="no avg_loss"):
        HostedTrainingJob(Silent(), "job-1").fwd_bwd_step({}, learning_rate=0.001)


def test_hosted_a_job_releases_its_gpus_when_its_block_ends(monkeypatch) -> None:
    client = _hosted_Client()
    monkeypatch.setattr(hosted, "ArcticClient", client.create)
    with HostedTransport(_hosted_CONNECTION, _hosted_config()) as transport:
        with transport.job() as job:
            assert job.job_id == "job-1"
            assert client.cancelled == []
        assert client.cancelled == ["job-1"]
    assert client.closed


def test_hosted_a_job_releases_its_gpus_when_the_block_raises(monkeypatch) -> None:
    client = _hosted_Client()
    monkeypatch.setattr(hosted, "ArcticClient", client.create)
    with pytest.raises(ZeroDivisionError):
        with HostedTransport(_hosted_CONNECTION, _hosted_config()) as transport:
            with transport.job():
                raise ZeroDivisionError
    assert client.cancelled == ["job-1"]


def test_hosted_a_resumed_job_is_created_from_the_checkpoint_it_was_given(monkeypatch) -> None:
    client = _hosted_Client()
    monkeypatch.setattr(hosted, "ArcticClient", client.create)
    with HostedTransport(_hosted_CONNECTION, _hosted_config()) as transport:
        with transport.job(resume_from=Checkpoint(reference="cp_1", source_job="job-a")):
            pass
    body, hardware = client.created[0]
    assert body["source_checkpoint_info"] == {"checkpoint_id": "cp_1", "source_job_id": "job-a"}
    assert hardware == "H200"


def test_hosted_creating_a_job_before_the_transport_is_entered_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="entered first"):
        with HostedTransport(_hosted_CONNECTION, _hosted_config()).job():
            pass


def test_hosted_the_connection_does_not_render_its_token() -> None:
    assert "not-a-token" not in repr(_hosted_CONNECTION)


def test_hosted_explicit_checkpoint_id_does_not_use_a_deepspeed_tag() -> None:
    client = _hosted_Client(save_result={"checkpoint_id": "cp_new", "checkpoint_tag": "global_step10"})
    assert save_and_resolve_checkpoint_id(client, "job-1", "resumable") == "cp_new"


def test_hosted_an_ambiguous_catalog_delta_is_refused() -> None:
    client = _hosted_Client(added=[{"checkpoint_id": "cp_one"}, {"checkpoint_id": "cp_two"}])
    with pytest.raises(RuntimeError, match="cannot be identified"):
        save_and_resolve_checkpoint_id(client, "job-1", "resumable")


def test_hosted_checkpoint_jobs_preserve_dss_engine_gas_and_width(monkeypatch) -> None:
    client = _hosted_Client()
    monkeypatch.setattr(hosted, "ArcticClient", client.create)
    cfg = _hosted_config().at_gpu_width(8)
    with HostedTransport(_hosted_CONNECTION, cfg) as transport:
        with transport.job():
            pass
        with transport.job(resume_from=Checkpoint("cp_new", "job-1")):
            pass
    assert len(client.created) == 2
    for payload, hardware in client.created:
        assert hardware == "H200"
        assert payload["training_config"]["n_gpus"] == 8
        assert payload["training_config"]["ds_config"]["gradient_accumulation_steps"] == 1
    assert client.cancelled == ["job-1", "job-2"]


def _hosted_local_backend(url="http://127.0.0.1:8003", **kwargs):
    return CortexConfig(base_url=url, host=None, pat=None, **kwargs)


@pytest.mark.parametrize("url", ["http://127.0.0.1:8003", "http://localhost:8003/", "http://[::1]:8003"])
def test_hosted_local_gateway_accepts_only_explicit_http_loopback(url) -> None:
    backend = _hosted_local_backend(url)
    assert is_local_cortex_gateway(backend)
    require_local_cortex_gateway(backend)


@pytest.mark.parametrize(
    "url",
    [
        "https://localhost:8003",
        "http://localhost.example:8003",
        "http://127.0.0.2:8003",
        "http://0.0.0.0:8003",
        "http://10.0.0.1:8003",
        "http://user:secret@localhost:8003",
        "http://localhost:8003/api",
        "http://localhost:8003/?value=secret",
        "http://localhost:8003/#secret",
    ],
)
def test_hosted_local_gateway_rejects_nonloopback_and_decorated_urls(url) -> None:
    with pytest.raises(ValueError, match="unauthenticated HTTP loopback") as exc:
        require_local_cortex_gateway(_hosted_local_backend(url))
    assert "secret" not in str(exc.value)


@pytest.mark.parametrize("options", [{"host": "server"}, {"pat": "not-a-token"}])
def test_hosted_local_gateway_refuses_hosted_credentials(options) -> None:
    backend = CortexConfig(base_url="http://127.0.0.1:8003", **options)
    with pytest.raises(ValueError, match="unauthenticated HTTP loopback") as exc:
        require_local_cortex_gateway(backend)
    assert "not-a-token" not in str(exc.value)


class _hosted_LocalClient(_hosted_Client):

    def list_checkpoints(self):
        raise AssertionError("the supplied local gateway has no checkpoint catalog")

    def load_checkpoint(self, checkpoint_id, *, stage_info=None):
        self.calls.append(("load", checkpoint_id, deepcopy(stage_info)))
        return hosted._local_load_metadata(
            {"checkpoint_id": checkpoint_id, "checkpoint_tag": "global_step10", "version": 10}, checkpoint_id
        )


def _hosted_local_checkpoint(root):
    return hosted.LocalCortexCheckpoint(
        _hosted_LOCAL_CHECKPOINT_ID,
        "owned-source",
        str(root),
        "global_step10",
        10,
        stage_info=deepcopy(_hosted_LOCAL_STAGE),
    )


@pytest.mark.parametrize("checkpoint_type", ["resumable", "weights-only"])
def test_hosted_local_save_uses_a_fresh_fast_path_without_catalog(monkeypatch, tmp_path, checkpoint_type) -> None:
    root = tmp_path / "retained-checkpoint"
    root.mkdir()
    marker = root / "retained-evidence"
    marker.write_text("preserve", encoding="ascii")
    monkeypatch.setattr(hosted, "correctness_workdir", lambda prefix: root)
    client = _hosted_LocalClient(
        save_result={"checkpoint_path": str(root), "checkpoint_tag": "global_step10", "version": 10}
    )
    saved_root, result = hosted._save_local_checkpoint(client, checkpoint_type)
    assert saved_root == root.resolve()
    assert result is client.save_result
    assert client.calls == [("save", checkpoint_type, str(root))]
    assert marker.read_text(encoding="ascii") == "preserve"


def test_hosted_local_resumable_checkpoint_keeps_the_saving_job_identity(monkeypatch, tmp_path) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    monkeypatch.setattr(hosted, "correctness_workdir", lambda prefix: root)
    client = _hosted_LocalClient(
        save_result={
            "checkpoint_path": str(root),
            "checkpoint_tag": "global_step10",
            "version": 10,
            "checkpoint_id": _hosted_LOCAL_CHECKPOINT_ID,
            "stage_info": deepcopy(_hosted_LOCAL_STAGE),
        }
    )
    checkpoint = HostedTrainingJob(client, "owned-parent", local_gateway=True).save_resumable()
    assert checkpoint == hosted.LocalCortexCheckpoint(
        _hosted_LOCAL_CHECKPOINT_ID, "owned-parent", str(root), "global_step10", 10, stage_info=_hosted_LOCAL_STAGE
    )
    assert checkpoint.stage_info is not client.save_result["stage_info"]
    assert client.calls == [("save", "resumable", str(root))]


def test_hosted_local_resumable_save_without_durable_identity_retains_output_and_releases_job(
    monkeypatch, tmp_path
) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    marker = root / "retained-evidence"
    marker.write_text("preserve", encoding="ascii")
    monkeypatch.setattr(hosted, "correctness_workdir", lambda prefix: root)
    client = _hosted_LocalClient(
        save_result={"checkpoint_path": str(root), "checkpoint_tag": "global_step10", "version": 10}
    )
    monkeypatch.setattr(hosted, "ArcticClient", client.create)
    with pytest.raises(RuntimeError, match="only a FAST path/tag"):
        with HostedTransport(_hosted_local_backend(), _hosted_config()) as transport:
            with transport.job() as job:
                job.save_resumable()
    assert client.calls == [("save", "resumable", str(root))]
    assert client.cancelled == ["job-1"] and client.closed
    assert marker.read_text(encoding="ascii") == "preserve"


@pytest.mark.parametrize(
    "changes",
    [
        {"checkpoint_path": None},
        {"checkpoint_path": "relative/checkpoint"},
        {"checkpoint_path": "/tmp/checkpoint"},
        {"checkpoint_path": "/data-fast"},
        {"checkpoint_path": "/data-fast/tmp/not-this-save"},
        {"checkpoint_tag": None},
        {"checkpoint_tag": "cp_123"},
        {"checkpoint_tag": "global_step-1"},
        {"checkpoint_tag": "global_step010"},
        {"checkpoint_tag": "../global_step10"},
        {"version": None},
        {"version": "10"},
        {"version": 10.0},
        {"version": True},
        {"version": 11},
    ],
)
def test_hosted_local_save_rejects_inconsistent_metadata_and_retains_output(monkeypatch, tmp_path, changes) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    marker = root / "retained-evidence"
    marker.write_text("preserve", encoding="ascii")
    monkeypatch.setattr(hosted, "correctness_workdir", lambda prefix: root)
    result = {"checkpoint_path": str(root), "checkpoint_tag": "global_step10", "version": 10, **changes}
    with pytest.raises((ValueError, RuntimeError)):
        hosted._save_local_checkpoint(_hosted_LocalClient(save_result=result), "resumable")
    assert marker.read_text(encoding="ascii") == "preserve"


@pytest.mark.parametrize("result", [None, [], "global_step10"])
def test_hosted_local_save_rejects_nonmapping_response(monkeypatch, tmp_path, result) -> None:
    monkeypatch.setattr(hosted, "correctness_workdir", lambda prefix: tmp_path)
    client = _hosted_LocalClient()
    client.save_result = result
    with pytest.raises(RuntimeError, match="no checkpoint metadata"):
        hosted._save_local_checkpoint(client, "resumable")


def test_hosted_local_save_rejects_invalid_checkpoint_type_before_creating_storage(monkeypatch) -> None:

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid type must not allocate storage")

    monkeypatch.setattr(hosted, "correctness_workdir", unexpected)
    with pytest.raises(ValueError, match="checkpoint_type"):
        hosted._save_local_checkpoint(_hosted_LocalClient(), "unknown")


def test_hosted_local_resume_creates_a_fresh_full_model_job_then_loads_before_training(monkeypatch, tmp_path) -> None:
    cfg = _hosted_config()
    before = json.dumps(cfg.sub_job, sort_keys=True)
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "latest").write_text("global_step10\n", encoding="ascii")
    client = _hosted_LocalClient(
        losses=[2.5],
        save_result={
            "checkpoint_path": str(root),
            "checkpoint_tag": "global_step10",
            "version": 10,
            "checkpoint_id": _hosted_LOCAL_CHECKPOINT_ID,
            "stage_info": deepcopy(_hosted_LOCAL_STAGE),
        },
    )
    monkeypatch.setattr(hosted, "correctness_workdir", lambda prefix: root)
    monkeypatch.setattr(hosted, "ArcticClient", client.create)
    with HostedTransport(_hosted_local_backend(), cfg) as transport:
        with transport.job() as first:
            checkpoint = first.save_resumable()
        assert client.cancelled == ["job-1"]
        with transport.job(resume_from=checkpoint) as resumed:
            assert resumed.job_id == "job-2"
            assert client.calls[-1] == ("load", _hosted_LOCAL_CHECKPOINT_ID, _hosted_LOCAL_STAGE)
            assert resumed.fwd_bwd_step({"args": (), "kwargs": {}}, learning_rate=0.001) == 2.5
    assert client.cancelled == ["job-1", "job-2"]
    assert client.calls == [
        ("save", "resumable", str(root)),
        ("load", _hosted_LOCAL_CHECKPOINT_ID, _hosted_LOCAL_STAGE),
        ("fwd-bwd", {"args": (), "kwargs": {}}),
        ("step", 0.001),
    ]
    for payload, hardware in client.created:
        assert payload["model_name"] == cfg.model_name == "Qwen/Qwen3-8B"
        assert "source_checkpoint_info" not in payload
        assert hardware == "H200"
        assert payload["training_config"]["n_gpus"] == cfg.n_gpus == 4
        assert payload["training_config"]["ds_config"]["gradient_accumulation_steps"] == 1
    assert json.dumps(cfg.sub_job, sort_keys=True) == before
    assert root.is_dir()
    assert (root / "latest").read_text(encoding="ascii") == "global_step10\n"
    assert client.closed


def test_hosted_local_resume_load_failure_releases_fresh_job_and_retains_checkpoint(monkeypatch, tmp_path) -> None:

    class FailingLoad(_hosted_LocalClient):

        def load_checkpoint(self, checkpoint_id, *, stage_info=None):
            super().load_checkpoint(checkpoint_id, stage_info=stage_info)
            raise RuntimeError("runtime load unavailable")

    client = FailingLoad()
    (tmp_path / "latest").write_text("global_step10\n", encoding="ascii")
    monkeypatch.setattr(hosted, "ArcticClient", client.create)
    with pytest.raises(RuntimeError, match="runtime load unavailable"):
        with HostedTransport(_hosted_local_backend(), _hosted_config()) as transport:
            with transport.job(resume_from=_hosted_local_checkpoint(tmp_path)):
                raise AssertionError("must not train before a successful load")
    assert client.cancelled == ["job-1"]
    assert client.calls == [("load", _hosted_LOCAL_CHECKPOINT_ID, _hosted_LOCAL_STAGE)]
    assert tmp_path.is_dir()
    assert (tmp_path / "latest").read_text(encoding="ascii") == "global_step10\n"
    assert client.closed


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"checkpoint_tag": "global_step0", "version": 0},
        {"checkpoint_tag": "global_step10", "version": True},
        {"checkpoint_tag": "global_step10", "version": 9},
    ],
)
def test_hosted_local_resume_rejects_unconfirmed_or_wrong_step_and_releases_fresh_job(
    monkeypatch, tmp_path, result
) -> None:

    class IncorrectLoad(_hosted_LocalClient):

        def load_checkpoint(self, checkpoint_id, *, stage_info=None):
            super().load_checkpoint(checkpoint_id, stage_info=stage_info)
            return hosted._local_load_metadata({"checkpoint_id": checkpoint_id, **result}, checkpoint_id)

    (tmp_path / "latest").write_text("global_step10\n", encoding="ascii")
    client = IncorrectLoad()
    monkeypatch.setattr(hosted, "ArcticClient", client.create)
    with pytest.raises(RuntimeError, match="local checkpoint load"):
        with HostedTransport(_hosted_local_backend(), _hosted_config()) as transport:
            with transport.job(resume_from=_hosted_local_checkpoint(tmp_path)):
                raise AssertionError("must not train before confirmed restored step")
    assert client.cancelled == ["job-1"]
    assert client.calls == [("load", _hosted_LOCAL_CHECKPOINT_ID, _hosted_LOCAL_STAGE)]
    assert tmp_path.is_dir()
    assert client.closed


@pytest.mark.parametrize("marker", [None, b"", b"cp_123", b"global_step-1", b"\xff", b"global_step10" + b" " * 129])
def test_hosted_local_resume_rejects_path_with_missing_or_invalid_marker_before_allocating_job(
    monkeypatch, tmp_path, marker
) -> None:

    def unexpected(*args, **kwargs):
        raise AssertionError("a FAST marker must not authorize cross-job resume")

    if marker is not None:
        (tmp_path / "latest").write_bytes(marker)
    monkeypatch.setattr(hosted, "ArcticClient", unexpected)
    with pytest.raises(ValueError, match="durable checkpoint identity"):
        with HostedTransport(_hosted_local_backend(), _hosted_config()) as transport:
            with transport.job(resume_from=Checkpoint(str(tmp_path), "owned-source")):
                pass


def test_hosted_local_resume_refuses_a_path_with_a_symlink_latest_marker(monkeypatch, tmp_path) -> None:
    target = tmp_path / "real-marker"
    target.write_text("global_step10\n", encoding="ascii")
    (tmp_path / "latest").symlink_to(target)
    client = _hosted_LocalClient()
    monkeypatch.setattr(hosted, "ArcticClient", client.create)
    with pytest.raises(ValueError, match="durable checkpoint identity"):
        with HostedTransport(_hosted_local_backend(), _hosted_config()) as transport:
            with transport.job(resume_from=Checkpoint(str(tmp_path), "owned-source")):
                pass
    assert client.created == client.calls == client.cancelled == []
    assert target.read_text(encoding="ascii") == "global_step10\n"


def test_hosted_local_resume_refuses_a_path_with_a_nonregular_latest_marker(monkeypatch, tmp_path) -> None:
    (tmp_path / "latest").mkdir()
    client = _hosted_LocalClient()
    monkeypatch.setattr(hosted, "ArcticClient", client.create)
    with pytest.raises(ValueError, match="durable checkpoint identity"):
        with HostedTransport(_hosted_local_backend(), _hosted_config()) as transport:
            with transport.job(resume_from=Checkpoint(str(tmp_path), "owned-source")):
                pass
    assert client.created == client.calls == client.cancelled == []
    assert (tmp_path / "latest").is_dir()


@pytest.mark.parametrize("reference", ["cp_hosted", "relative/checkpoint", "/tmp/checkpoint", "/data-fast"])
def test_hosted_local_resume_rejects_nonfast_or_nonabsolute_reference_before_job_creation(
    monkeypatch, reference
) -> None:

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid reference must not create a training job")

    monkeypatch.setattr(hosted, "ArcticClient", unexpected)
    with pytest.raises(ValueError, match="durable checkpoint identity"):
        with HostedTransport(_hosted_local_backend(), _hosted_config()) as transport:
            with transport.job(resume_from=Checkpoint(reference, "owned-source")):
                pass


def test_hosted_local_resume_refuses_a_valid_fast_marker_before_allocating_job(monkeypatch, tmp_path) -> None:
    (tmp_path / "latest").write_text("global_step10\n", encoding="ascii")
    client = _hosted_LocalClient()
    monkeypatch.setattr(hosted, "ArcticClient", client.create)
    with pytest.raises(ValueError, match="durable checkpoint identity"):
        with HostedTransport(_hosted_local_backend(), _hosted_config()) as transport:
            with transport.job(resume_from=Checkpoint(str(tmp_path), "owned-source")):
                raise AssertionError("a valid FAST marker is not a durable identity")
    assert client.created == client.calls == client.cancelled == []
    assert (tmp_path / "latest").read_text(encoding="ascii") == "global_step10\n"


def test_hosted_local_resume_requires_explicit_stage_before_allocating_job(monkeypatch, tmp_path) -> None:
    checkpoint = hosted.LocalCortexCheckpoint(
        _hosted_LOCAL_CHECKPOINT_ID, "owned-source", str(tmp_path), "global_step10", 10
    )
    client = _hosted_LocalClient()
    monkeypatch.setattr(hosted, "ArcticClient", client.create)
    with pytest.raises(RuntimeError, match="explicit stage_info"):
        with HostedTransport(_hosted_local_backend(), _hosted_config()) as transport:
            with transport.job(resume_from=checkpoint):
                raise AssertionError("a checkpoint id alone cannot resolve another parent's store")
    assert client.created == client.calls == client.cancelled == []


# Hosted determinism.


def test_hosted_determinism_training_client_config_pins_determinism() -> None:
    cfg = LoadedConfig(
        config_id="x",
        path=Path("x"),
        sub_job={
            "model_name": "model",
            "training_config": {
                "optimizer": {"type": "adamw"},
                "max_seq_len": 8,
                "train_batch_size": 1,
                "n_gpus": 1,
                "debug": {"full_determinism": False, "full_determinism_must_comply": True},
            },
            "global_batch_size": 1,
            "dtype": "bfloat16",
            "seed": 1,
        },
    )
    wire = _hosted_wire(cfg)
    debug = wire["training_config"]["debug"]
    assert debug["full_determinism"] is True
    assert debug["full_determinism_must_comply"] is False
