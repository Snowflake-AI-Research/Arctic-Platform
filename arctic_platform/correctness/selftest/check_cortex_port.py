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

"""Regression coverage for the DSS training contract against the clean public client."""

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
from arctic_platform.client.transport import JobHandles
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.correctness.checks import inference_checkpoint_loss as inference
from arctic_platform.correctness.harness import dss_driver
from arctic_platform.correctness.harness import gsm8k
from arctic_platform.correctness.harness import hosted
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness.arms import arms_for
from arctic_platform.correctness.harness.batches import load as load_batch
from arctic_platform.correctness.harness.config import config_checksum
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.dss_driver import GatewaySession
from arctic_platform.correctness.harness.dss_driver import _training_client_config
from arctic_platform.correctness.harness.dss_driver import build_payload
from arctic_platform.correctness.harness.spec import TestSpec as ConfigSpec

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(
        aiohttp.ClientSession, "request", lambda *args, **kwargs: pytest.fail("unit tests must not access the network")
    )
    monkeypatch.setattr(
        requests.Session, "request", lambda *args, **kwargs: pytest.fail("unit tests must not access the network")
    )


def test_retained_config_loads_without_changing_the_reviewed_identity():
    original = CONFIG.read_bytes()
    cfg = load_config(CONFIG)
    spec = json.loads((ROOT / f"specs/{cfg.config_id}.json").read_text())
    assert config_checksum(cfg) == spec["config_checksum"]
    assert CONFIG.read_bytes() == original


@pytest.mark.parametrize("protocol", ["http", "other"])
def test_config_intake_does_not_ignore_unknown_protocols(tmp_path, protocol):
    raw = json.loads(CONFIG.read_text())
    raw["backend"]["protocol"] = protocol
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_config(path)


def test_cortex_training_preserves_dss_extra_fields_and_determinism():
    training = {
        "n_gpus": 8,
        "max_seq_len": 32,
        "train_batch_size": 8,
        "optimizer": {"name": "AdamW", "lr": 1e-4},
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


def test_hosted_config_does_not_depend_on_removed_public_backend_fields():
    cfg = load_config(CONFIG)
    config = hosted.training_client_config(cfg, CortexConfig(base_url="http://unused"))
    assert config.hardware == "H200"
    assert config.to_cortex()[0]["training_config"]["debug"]["full_determinism"] is True


@pytest.mark.parametrize("order", ["batched", "interleaved"])
@pytest.mark.parametrize("startup_error", [False, True])
def test_reviewed_gradient_cases_send_dss_requests_not_onprem_workarounds(tmp_path, monkeypatch, order, startup_error):
    from arctic_platform.correctness.harness import cortex_fixture
    from arctic_platform.correctness.harness import reference_records

    cfg = load_config(CONFIG)
    spec = ConfigSpec.read(ROOT / f"specs/{cfg.config_id}.json")
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
        assert all(result.outcome.value == "fail" for result in results)
        assert all("execution error:" in result.summary and "mb_spec" in result.summary for result in results)
        assert all(not result.metrics and not result.mismatches for result in results)
        assert targets == []
    else:
        assert all(result.outcome.value == "pass" for result in results)
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


def _mock_inference_inputs(monkeypatch, cfg):
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
    return training, validation, tokenizer_paths


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
def test_managed_inference_uses_public_transport_and_cleans_up(tmp_path, monkeypatch, mode, receipt):
    cfg = load_config(CONFIG)
    spec = ConfigSpec.read(ROOT / f"specs/{cfg.config_id}.json")
    training, validation, tokenizer_paths = _mock_inference_inputs(monkeypatch, cfg)
    ctx = SimpleNamespace(
        cfg=cfg,
        spec=spec,
        config_id=cfg.config_id,
        workdir=tmp_path,
        attn_implementation=cfg.attention_implementation,
        client_backend=CortexConfig(base_url="http://unused"),
        target_model_name="ap-correctness/reviewed-fixture",
    )
    creates, cancels, backward, steps, forwards, generates = [], [], [], [], [], []
    transports, closed, replies = [], [], {}
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
        transport.job_id is None and transport._session is None and transport in closed for transport in transports
    )
    assert tokenizer_paths == [cfg.model_name]
    assert len(backward) == len(steps) == inference.TRAIN_STEPS == 10
    assert len(forwards) == 1
    assert all(step == {"learning_rate": cfg.training["optimizer"]["lr"]} for step in steps)
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
        assert result.metrics["scored_answer_tokens"] == sum(example.answer_tokens for example in validation)
        assert result.metrics["train_steps"] == 10
        assert result.metrics["stated_criterion_abs"] == 1e-3
    else:
        assert result.outcome.value == "fail"


def test_sampling_config_forwards_checkpoint_identity_without_mutating_training():
    cfg = load_config(CONFIG)
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


def test_onprem_inference_retains_local_export_and_reload_diagnostic(tmp_path, monkeypatch, capsys):
    cfg = load_config(CONFIG)
    spec = ConfigSpec.read(ROOT / f"specs/{cfg.config_id}.json")
    training, validation, tokenizer_paths = _mock_inference_inputs(monkeypatch, cfg)
    ctx = SimpleNamespace(
        cfg=cfg,
        spec=spec,
        config_id=cfg.config_id,
        workdir=tmp_path,
        attn_implementation=cfg.attention_implementation,
        client_backend=OnPremConfig(),
        target_model_name=None,
    )
    clients, steps, tokenizer_saves = [], [], []
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
    assert [client.config.model_name for client in clients] == [
        cfg.model_name,
        str(weights_dir),
        str(weights_dir),
    ]
    assert all(client.shutdown_called and isinstance(client.config.backend, OnPremConfig) for client in clients)
    assert steps == [cfg.training["optimizer"]["lr"]] * 10
    assert tokenizer_paths == [cfg.model_name]
    assert tokenizer_saves == [str(weights_dir)]
    assert result.metrics["reference_cross_entropy"] == result.metrics["target_cross_entropy"] == 1.25
    assert "INFERENCE_DIAGNOSTIC" in capsys.readouterr().out
