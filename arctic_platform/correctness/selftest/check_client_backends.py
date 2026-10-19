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

"""Backend selection and current Cortex wire contracts without GPUs or network."""

import asyncio
from argparse import Namespace
from types import SimpleNamespace
from urllib.parse import urlsplit

import aiohttp
import pytest
import requests
import torch
from pydantic import ValidationError

from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.client import TrainingConfig
from arctic_platform.client.transport import JobHandles
from arctic_platform.client.transports.cortex import _DEFAULT
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.client.transports.cortex import _SendPolicy
from arctic_platform.correctness import __main__ as cli
from arctic_platform.correctness.checks import inference_checkpoint_loss as inference
from arctic_platform.correctness.harness import cortex_fixture
from arctic_platform.correctness.harness import dss_driver
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.cortex import CortexTrainingConfig
from arctic_platform.correctness.harness.cortex_transport import CorrectnessCortexTransport
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.registry import TestResult as Result
from arctic_platform.correctness.selftest.check_applicable_tests import _spec
from arctic_platform.correctness.selftest.check_cli_config_selection import _write_config
from arctic_platform.testing_utils import torch_assert_equal


def _remote():
    return CortexConfig(base_url="http://cortex.test")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
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


def _session_with_close_spy(transport, monkeypatch, closed):
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


def test_cli_defaults_to_cortex_and_accepts_explicit_onprem(monkeypatch):
    selected = []
    monkeypatch.setattr(cli, "cmd_run", lambda args: selected.append(args.backend) or 0)
    assert cli.main(["run", "--all-configs"]) == 0
    assert cli.main(["run", "--all-configs", "--backend", "onprem"]) == 0
    assert selected == ["cortex", "onprem"]


def test_cortex_connection_uses_environment_or_pat_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://from-env.test")
    assert cli._client_backend(Namespace()).base_url == "http://from-env.test"
    token = tmp_path / "pat"
    token.write_text("file-token\n")
    backend = cli._client_backend(Namespace(pat_file=token, host="host", database="db", schema="sc"))
    assert backend.pat.get_secret_value() == "file-token"
    assert "file-token" not in repr(backend)
    assert isinstance(cli._client_backend(Namespace(backend="onprem")), OnPremConfig)


@pytest.mark.parametrize(
    "base_url", ["http://127.0.0.1:7001", "http://localhost:7001", "http://[::1]:7001", "http://127.0.0.1:7001/"]
)
@pytest.mark.parametrize("from_environment", [False, True])
def test_local_cortex_uses_direct_url_without_hosted_auth(monkeypatch, base_url, from_environment):
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


def test_remote_run_never_probes_local_allocation(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    _write_config(config_dir, "h200")
    _write_config(config_dir, "b200")
    seen = []

    def forbidden(*args, **kwargs):
        pytest.fail("Cortex must not probe or clean the local allocation")

    monkeypatch.setattr(cli, "local_gpu_type", forbidden)
    monkeypatch.setattr(cli, "available_gpus", forbidden)
    monkeypatch.setattr(cli, "validate_against_host", forbidden)
    monkeypatch.setattr(cli, "ensure_no_interrupted_builds", forbidden)
    monkeypatch.setattr(cli, "_frozen_spec", lambda cfg, *args, **kwargs: seen.append(cfg.config_id))
    monkeypatch.setattr(cli, "_finish", lambda *args, **kwargs: 0)
    assert cli.main(["run", "--all-configs", "--config-dir", str(config_dir), "--base-url", _remote().base_url]) == 0
    assert len(seen) == 2


def test_remote_job_skips_ray_and_preserves_backend(tmp_path, monkeypatch):
    backend = _remote()
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


def test_cortex_batch_preserves_microbatch_order_labels_and_context():
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


@pytest.mark.parametrize("batches", [[], [{"x": torch.ones(1)}, {"y": torch.ones(1)}]])
def test_cortex_rejects_malformed_microbatch_lists(batches):
    with pytest.raises(ValueError, match="identical tensor keys"):
        dss_driver.cortex_training_body({"batch": batches})


def test_onprem_batch_and_cortex_rpc_are_not_rewritten():
    body = {"batch": {"input_ids": torch.ones(1, 2)}}
    job = SimpleNamespace(client=SimpleNamespace(config=SimpleNamespace(backend=OnPremConfig())))
    assert dss_driver.training_body(job, body) is body
    rpc = {"args": (), "kwargs": {"input_ids": torch.ones(1, 2)}}
    assert dss_driver.cortex_training_body(rpc) is rpc


def test_cortex_config_forwards_worker_settings_without_mutation():
    worker = {
        "debug": {"full_determinism_must_comply": False},
        "sp_size": 8,
        "prime_rl": {"fused_cross_entropy": "liger"},
        "enable_gradient_checkpointing": True,
    }
    config = CortexTrainingConfig(
        model_name="model",
        training_gpus=8,
        backend=_remote(),
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


@pytest.mark.parametrize("source", [{}, {"checkpoint_id": "cp_saved"}, {"checkpoint_id": "", "source_job_id": "p"}])
def test_cortex_resume_requires_complete_source_identity(source):
    with pytest.raises(ValidationError, match="checkpoint_id and source_job_id"):
        CortexTrainingConfig(model_name="model", backend=_remote(), source_checkpoint_info=source)


def test_onprem_refuses_cortex_checkpoint_identity():
    with pytest.raises(ValidationError, match="requires CortexConfig"):
        CortexTrainingConfig(
            model_name="model", source_checkpoint_info={"checkpoint_id": "cp_saved", "source_job_id": "p"}
        )


def test_cortex_transport_sends_hardware_and_lists_durable_checkpoints(monkeypatch):
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


def test_cortex_shutdown_closes_session_when_cancel_fails(monkeypatch):
    transport = CorrectnessCortexTransport(_remote())
    transport.job_id = "parent"
    closed = []
    session = _session_with_close_spy(transport, monkeypatch, closed)
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


def test_inapplicable_check_is_not_also_a_cortex_failure(tmp_path, monkeypatch):
    cfg = SimpleNamespace(config_id="unit", n_gpus=8, training={}, at_gpu_width=lambda width: cfg)
    spec = _spec([])
    results = runner.execute(
        cfg, spec, [], vocab_size=128, select=["single-step-optimizer"], client_backend=_remote(), workdir=tmp_path
    )
    assert len(results) == 1
    assert results[0].outcome is Outcome.INAPPLICABLE


def test_cortex_unsupported_checks_fail_before_batch_or_reference(tmp_path, monkeypatch):
    cfg = SimpleNamespace(config_id="unit", n_gpus=8, training={}, at_gpu_width=lambda width: cfg)
    spec = _spec(["single-step-optimizer"])
    monkeypatch.setattr(runner, "build_batch", lambda *args, **kwargs: pytest.fail("must reject before batches"))
    results = runner.execute(
        cfg, spec, [], vocab_size=128, select=["single-step-optimizer"], client_backend=_remote(), workdir=tmp_path
    )
    assert len(results) == 1
    assert results[0].outcome is Outcome.FAIL
    assert results[0].summary == "published Cortex fixture required"
    assert "--cortex-fixture" in results[0].reason


@pytest.mark.parametrize("check", ["single-step-grads", "e2e-sft-train-validate"])
def test_cortex_refuses_checks_requiring_local_model_identity(tmp_path, monkeypatch, check):
    cfg = SimpleNamespace(config_id="unit", n_gpus=8, training={}, at_gpu_width=lambda width: cfg)
    monkeypatch.setattr(runner, "build_batch", lambda *args, **kwargs: pytest.fail("must reject before batches"))
    monkeypatch.setattr(runner, "run_reference", lambda *args, **kwargs: pytest.fail("must reject before reference"))
    results = runner.execute(
        cfg, _spec([check]), [], vocab_size=128, select=[check], client_backend=_remote(), workdir=tmp_path
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


@pytest.mark.parametrize("check", ["rl-weight-sync", "rl-router-replay"])
def test_cortex_rl_refuses_local_instrumentation_before_check(tmp_path, monkeypatch, check):
    cfg = SimpleNamespace(
        config_id="unit", n_gpus=8, training={}, attention_implementation="sdpa", at_gpu_width=lambda width: cfg
    )
    monkeypatch.setattr(runner, "rl_gpu_count", lambda cfg: 16)
    tests = runner.registered_tests()
    monkeypatch.setattr(tests[check], "fn", lambda ctx: pytest.fail("must reject before tokenizer or client"))
    results = runner.execute_rl(
        cfg, _spec([check]), vocab_size=128, select=[check], client_backend=_remote(), workdir=tmp_path
    )
    assert len(results) == 1
    assert results[0].outcome is Outcome.FAIL
    assert results[0].reason == runner.cortex_unavailable_reason(tests[check])
    assert "--backend onprem" not in results[0].reason


def test_cli_rejects_unsupported_cortex_checks_without_reading_model(tmp_path, monkeypatch):
    import transformers

    path = _write_config(tmp_path / "configs", "h200")
    spec = _spec(["single-step-grads"])
    captured = []

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported Cortex check must not load local or catalog model metadata")

    monkeypatch.setattr(cli, "_frozen_spec", lambda cfg, path, arms, **kwargs: (spec, arms))
    monkeypatch.setattr(cli, "_catalog_vocab_size", forbidden)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", forbidden)
    monkeypatch.setattr(cli, "_finish", lambda results, *args, **kwargs: captured.extend(results) or 1)
    assert (
        cli.main(["run", "--config", str(path), "--test", "single-step-grads", "--base-url", _remote().base_url]) == 1
    )
    assert captured and all(result.outcome is Outcome.FAIL for result in captured)
    assert "--cortex-fixture" in captured[0].reason


@pytest.mark.parametrize("with_fixture", [False, True])
def test_cortex_cli_inference_runs_full_model_without_reduced_metadata(tmp_path, monkeypatch, with_fixture):
    import transformers

    path = _write_config(tmp_path / "configs", "h200")
    cfg = load_config(path)
    test_id = "inference-checkpoint-loss"
    spec = _spec([test_id])
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
        _remote().base_url,
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
            fixture_sha256=fixture["fixture_sha256"],
            fixture_stage_path=fixture["publication"]["stage_path"],
        )
    assert metadata == {cfg.config_id: expected}


@pytest.mark.parametrize("include_inference", [False, True])
def test_cortex_cli_fixture_provenance_keeps_e2e_and_full_inference_distinct(tmp_path, monkeypatch, include_inference):
    path = _write_config(tmp_path / "configs", "h200")
    cfg = load_config(path)
    selected = ["e2e-sft-train-validate"]
    if include_inference:
        selected.append("inference-checkpoint-loss")
    spec = _spec(selected)
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
    monkeypatch.setattr(cli, "execute", lambda *args, **kwargs: calls.append(kwargs) or [])
    monkeypatch.setattr(cli, "_finish", lambda *args, **kwargs: metadata.update(kwargs["execution"]) or 0)
    args = ["run", "--config", str(path), "--base-url", _remote().base_url, "--cortex-fixture", str(receipt)]
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
    assert len(calls) == 1
    assert calls[0]["select"] == selected
    assert calls[0]["cortex_fixture"] == receipt


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("with_fixture", [False, True])
def test_inference_only_vocabulary_uses_full_model_on_both_backends(tmp_path, monkeypatch, remote, with_fixture):
    import transformers

    cfg = load_config(_write_config(tmp_path / "configs", "h200"))
    test_id = "inference-checkpoint-loss"
    spec = _spec([test_id])
    spec.model.cache_path = "/data-fast/unused-reduced-reference"
    observed = []
    backend = _remote() if remote else OnPremConfig()
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


def test_cortex_e2e_only_keeps_reduced_model_vocabulary(tmp_path, monkeypatch):
    import transformers

    cfg = load_config(_write_config(tmp_path / "configs", "h200"))
    spec = _spec(["e2e-sft-train-validate"])
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
            _remote(),
            {"model_name": "published-exact-reviewed-weights"},
        )
        == 64
    )
    assert observed == [(spec.model.cache_path, {"trust_remote_code": True})]


def test_cortex_sft_body_does_not_select_the_rl_processor():
    from arctic_platform import wire

    body = {
        "batch": {"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[1, 2, -100]])},
        "processing": {"loss_fn": "sft"},
    }
    lowered = dss_driver.cortex_training_body(body)
    # No processing dictionary selects the zone's supervised dispatch, including its label shift.
    assert "processing" not in lowered
    decoded = wire.loads(wire.dumps(lowered))
    torch_assert_equal(decoded["kwargs"]["labels"], body["batch"]["labels"])


@pytest.mark.parametrize(
    "processing",
    [
        {"loss_fn": "sft_ce", "config": {"log_prob_micro_batch_size": 128}},
        {"loss_fn": "sft", "config": {"unknown": True}},
    ],
)
def test_cortex_does_not_silently_drop_sft_processing_options(processing):
    with pytest.raises(ValueError, match="use --backend onprem"):
        dss_driver.cortex_training_body({"batch": {"input_ids": torch.ones(1, 2)}, "processing": processing})


def test_remote_cli_checkpoint_uses_catalog_and_explicit_placement_pool(tmp_path, monkeypatch):
    path = _write_config(tmp_path / "configs", "h200")
    spec = _spec(["checkpoint-resume-loss"])
    measured = []
    metadata = {}
    monkeypatch.setattr(cli, "_frozen_spec", lambda cfg, path, arms, **kwargs: (spec, arms))
    monkeypatch.setattr(cli, "_catalog_vocab_size", lambda name: 128)
    monkeypatch.setattr(cli, "execute", lambda cfg, *args, **kwargs: measured.append((cfg, kwargs)) or [])
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
                _remote().base_url,
            ]
        )
        == 0
    )
    assert [cfg.n_gpus for cfg, _ in measured] == [8, 16]
    assert all(isinstance(options["client_backend"], CortexConfig) for _, options in measured)
    assert all(options["vocab_size"] == 128 for _, options in measured)
    assert metadata["model-h200-train"] == {"backend": "cortex", "requested_model_name": measured[0][0].model_name}


def test_hosted_alias_uses_the_same_cortex_run_path(monkeypatch):
    selected = []
    monkeypatch.setattr(cli, "cmd_run", lambda args: selected.append(args.backend) or 0)
    assert cli.main(["hosted", "--all-configs", "--test", "checkpoint-resume-loss"]) == 0
    assert selected == ["cortex"]


def test_cortex_report_does_not_claim_the_reduced_fixture_was_measured(tmp_path, monkeypatch):
    import json

    from arctic_platform.correctness.harness import report

    monkeypatch.setattr(report, "collect", lambda: {"host": "client-host"})
    spec = _spec(["checkpoint-resume-loss"])
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


@pytest.mark.parametrize("case,microbatches", [("gas1", 1), ("gas4", 4)])
def test_checkpoint_check_uses_three_real_cortex_clients(tmp_path, monkeypatch, case, microbatches):
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
    cfg = load_config(_write_config(tmp_path / "configs", "h200")).at_gpu_width(16)
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
        client_backend=_remote(),
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


@pytest.mark.parametrize("path", ["hosted", "driver"])
@pytest.mark.parametrize("cancel_fails", [False, True])
def test_remote_startup_failure_releases_job_and_session(tmp_path, monkeypatch, path, cancel_fails):
    from arctic_platform.correctness.harness import cortex_transport
    from arctic_platform.correctness.harness import hosted
    from arctic_platform.correctness.harness.config import load_config

    transport = CorrectnessCortexTransport(_remote())
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

    session = _session_with_close_spy(transport, monkeypatch, events)
    monkeypatch.setattr(transport, "_send", send)
    monkeypatch.setattr(cortex_transport, "CorrectnessCortexTransport", lambda *args, **kwargs: transport)
    with pytest.raises(RuntimeError, match="no capacity"):
        if path == "hosted":
            cfg = load_config(_write_config(tmp_path / "configs", "h200"))
            with hosted.HostedTransport(_remote(), cfg) as hosted_transport:
                with hosted_transport.job():
                    pytest.fail("failed startup must not yield a job")
        else:
            payload = {"model_name": "model", "training_config": {"n_gpus": 8}}
            with dss_driver.gateway(tmp_path, 8, backend=_remote()) as gateway_session:
                with dss_driver.running_job(gateway_session, payload):
                    pytest.fail("failed startup must not yield a job")
    assert events == ["cancel", "close"]
    assert transport.job_id is None
    assert session.closed and transport._session is None
