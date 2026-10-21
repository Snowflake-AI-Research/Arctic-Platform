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
import json
from argparse import Namespace
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
from arctic_platform.client.transport import JobHandles
from arctic_platform.client.transports.cortex import _DEFAULT
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.client.transports.cortex import _SendPolicy
from arctic_platform.correctness.harness import dss_driver
from arctic_platform.correctness.harness import execution as cli
from arctic_platform.correctness.harness import hosted
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness.batches import load as load_batch
from arctic_platform.correctness.harness.config import LoadedConfig
from arctic_platform.correctness.harness.config import ReviewedTrainingConfig
from arctic_platform.correctness.harness.config import TestSpec as ConfigSpec
from arctic_platform.correctness.harness.config import arms_for
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.cortex import CorrectnessCortexTransport
from arctic_platform.correctness.harness.cortex import CortexTrainingConfig
from arctic_platform.correctness.harness.dss_driver import GatewaySession
from arctic_platform.correctness.harness.dss_driver import _training_client_config
from arctic_platform.correctness.harness.dss_driver import build_payload
from arctic_platform.correctness.harness.hosted import HostedConnection
from arctic_platform.correctness.harness.hosted import is_local_cortex_gateway
from arctic_platform.correctness.harness.hosted import require_local_cortex_gateway
from arctic_platform.correctness.harness.hosted import save_and_resolve_checkpoint_id
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.testing_utils import torch_assert_equal

from .check_catalog_configuration import _applicable_tests_spec
from .check_operator_interface import _cli_config_selection_write_config
from .helpers import _synthetic_config_spec


@pytest.fixture(autouse=True)
def _synthetic_runtime_inputs(tmp_path, monkeypatch):
    config, spec = _synthetic_config_spec(tmp_path / "inputs")
    monkeypatch.setitem(globals(), "_cortex_port_CONFIG", config)
    monkeypatch.setitem(globals(), "_cortex_port_ROOT", config.parents[3])
    monkeypatch.setitem(globals(), "_driver_reviewed_config", config)


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
def test_client_backends_cli_defaults_to_onprem_and_accepts_explicit_cortex(monkeypatch):
    selected = []
    monkeypatch.setattr(cli, "cmd_run", lambda args: selected.append(args.backend) or 0)
    assert cli.main(["run", "--all-configs"]) == 0
    assert cli.main(["run", "--all-configs", "--backend", "cortex"]) == 0
    assert selected == ["onprem", "cortex"]


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_cortex_connection_uses_environment_or_pat_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://from-env.test")
    assert cli._client_backend(Namespace(backend="cortex")).base_url == "http://from-env.test"
    token = tmp_path / "pat"
    token.write_text("file-token\n")
    backend = cli._client_backend(Namespace(backend="cortex", pat_file=token, host="host", database="db", schema="sc"))
    assert backend.pat.get_secret_value() == "file-token"
    assert "file-token" not in repr(backend)
    assert isinstance(cli._client_backend(Namespace(backend="onprem")), OnPremConfig)


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
        [
            "run",
            "--backend",
            "cortex",
            "--all-configs",
            "--config-dir",
            str(config_dir),
            "--base-url",
            _client_backends_remote().base_url,
        ]
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
        training=ReviewedTrainingConfig(ds_worker_config=worker, full_determinism=True),
    )
    wire = config.to_cortex()[0]
    assert wire["training_config"]["sp_size"] == 8
    assert wire["training_config"]["prime_rl"] == worker["prime_rl"]
    assert wire["training_config"]["debug"] == {"full_determinism": True, "full_determinism_must_comply": False}
    assert wire["training_config"]["activation_checkpointing"] is True
    assert "enable_gradient_checkpointing" not in wire["training_config"]
    assert "full_determinism" not in worker["debug"]


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_rejects_checkpoint_resume_options_not_in_part_one():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
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
@pytest.mark.parametrize("check", ["single-step-grads", "single-step-optimizer"])
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
def test_client_backends_hosted_alias_uses_the_same_cortex_run_path(monkeypatch):
    selected = []
    monkeypatch.setattr(cli, "cmd_run", lambda args: selected.append(args.backend) or 0)
    assert cli.main(["hosted", "--all-configs", "--test", "single-step-grads"]) == 0
    assert selected == ["cortex"]


@pytest.mark.usefixtures("_client_backends_no_network")
def test_client_backends_cortex_report_does_not_claim_the_reduced_fixture_was_measured(tmp_path, monkeypatch):
    import json

    from arctic_platform.correctness.harness import report

    monkeypatch.setattr(report, "collect", lambda: {"host": "client-host"})
    spec = _applicable_tests_spec(["single-step-grads"])
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
@pytest.mark.parametrize("cancel_fails", [False, True])
def test_client_backends_remote_startup_failure_releases_job_and_session(tmp_path, monkeypatch, cancel_fails):
    from arctic_platform.correctness.harness import cortex as cortex_transport

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
        payload = {"model_name": "model", "training_config": {"n_gpus": 8}}
        with dss_driver.gateway(tmp_path, 8, backend=_client_backends_remote()) as gateway_session:
            with dss_driver.running_job(gateway_session, payload):
                pytest.fail("failed startup must not yield a job")
    assert events == ["cancel", "close"]
    assert transport.job_id is None
    assert session.closed and transport._session is None


_driver_reviewed_config: Path | None = None


@pytest.mark.parametrize("weight_format", [None, "vllm", "hf", "lora"])
def test_driver_reviewed_build_payload_keeps_format_outside_worker_config(weight_format):
    cfg = load_config(_driver_reviewed_config)
    before = deepcopy(cfg.training)
    payload = dss_driver.build_payload(cfg.training, cfg.model_name, cfg.native.seed, weight_format=weight_format)
    assert payload.get("weight_format") == weight_format
    assert ("weight_format" in payload) == (weight_format is not None)
    assert "weight_format" not in payload["training_config"]
    assert cfg.training == before


@pytest.mark.parametrize("weight_format", ["vllm", "hf", "lora"])
def test_driver_reviewed_training_preserves_explicit_format_without_mutating_payload(weight_format):
    cfg = load_config(_driver_reviewed_config)
    payload = deepcopy(cfg.sub_job)
    payload["weight_format"] = weight_format
    before = deepcopy(payload)
    session = GatewaySession(Path("."), 32, CortexConfig(base_url="http://unused"))
    native = dss_driver._training_client_config(session, payload)
    assert native.training.weight_format == weight_format
    assert native.to_cortex()[0]["weight_format"] == weight_format
    assert payload == before


def test_driver_reviewed_rejects_unsupported_format_before_gateway_runtime(monkeypatch):
    cfg = load_config(_driver_reviewed_config)
    payload = deepcopy(cfg.sub_job)
    payload["weight_format"] = "vllm"
    session = GatewaySession(Path("/data-fast/tmp/reviewed-prestart-control"), 32, OnPremConfig())
    monkeypatch.setattr(
        dss_driver, "_gateway_runtime", lambda *args: pytest.fail("invalid recipe must not touch the runtime")
    )
    monkeypatch.setattr(dss_driver, "ArcticClient", lambda *args: pytest.fail("invalid recipe must not create jobs"))
    with pytest.raises(ValueError, match="require the Cortex correctness route"):
        with dss_driver.running_job(session, payload):
            pytest.fail("unsupported fields must not yield")


_cortex_port_ROOT: Path | None = None
_cortex_port_CONFIG: Path | None = None


@pytest.fixture(autouse=False)
def _cortex_port_no_network(monkeypatch):
    monkeypatch.setattr(
        aiohttp.ClientSession, "request", lambda *args, **kwargs: pytest.fail("unit tests must not access the network")
    )
    monkeypatch.setattr(
        requests.Session, "request", lambda *args, **kwargs: pytest.fail("unit tests must not access the network")
    )


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


_hosted_CONNECTION = HostedConnection(host="server", database="DB", schema="SCHEMA", pat="not-a-token")


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


def test_hosted_the_connection_does_not_render_its_token() -> None:
    assert "not-a-token" not in repr(_hosted_CONNECTION)


def test_hosted_explicit_checkpoint_id_does_not_use_a_deepspeed_tag() -> None:
    client = _hosted_Client(save_result={"checkpoint_id": "cp_new", "checkpoint_tag": "global_step10"})
    assert save_and_resolve_checkpoint_id(client, "job-1", "resumable") == "cp_new"


def test_hosted_an_ambiguous_catalog_delta_is_refused() -> None:
    client = _hosted_Client(added=[{"checkpoint_id": "cp_one"}, {"checkpoint_id": "cp_two"}])
    with pytest.raises(RuntimeError, match="cannot be identified"):
        save_and_resolve_checkpoint_id(client, "job-1", "resumable")


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
