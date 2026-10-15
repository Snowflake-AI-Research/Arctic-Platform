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

"""Full-model inference and published E2E targets retain their distinct model identities."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.correctness.checks import e2e_sft_train_validate as e2e
from arctic_platform.correctness.checks import inference_checkpoint_loss as inference
from arctic_platform.correctness.harness import cortex_fixture
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.registry import TestResult as Result
from arctic_platform.correctness.harness.spec import TestSpec as Spec
from arctic_platform.correctness.selftest.check_applicable_tests import _spec

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config"
SPEC = ROOT / "specs/qwen3-8b-h200-train-sft-full-4gpus-64k.json"


def test_validated_fixture_reaches_e2e_context_without_changing_reference(tmp_path, monkeypatch):
    test_id = "e2e-sft-train-validate"
    cfg = load_config(CONFIG)
    spec = Spec.read(SPEC)
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


def test_cortex_runner_inference_without_fixture_uses_full_configured_model(tmp_path, monkeypatch):
    import transformers

    cfg = load_config(CONFIG)
    test_id = "inference-checkpoint-loss"
    spec = _spec([test_id])
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


def test_mixed_cortex_run_keeps_e2e_fixture_but_inference_ignores_it(tmp_path, monkeypatch):
    import transformers

    cfg = load_config(CONFIG)
    test_ids = ["e2e-sft-train-validate", "inference-checkpoint-loss"]
    spec = _spec(test_ids)
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
def test_cortex_sampling_checkpoint_identity_uses_full_model_not_e2e_fixture(target_model_name):
    cfg = load_config(CONFIG)
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
def test_e2e_target_identity_hardware_and_backend(tmp_path, monkeypatch, remote):
    cfg = load_config(CONFIG)
    spec = Spec.read(SPEC)
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


def test_cortex_e2e_without_published_identity_never_creates_job(tmp_path, monkeypatch):
    cfg = load_config(CONFIG)
    ctx = SimpleNamespace(
        cfg=cfg,
        spec=Spec.read(SPEC),
        attn_implementation="sdpa",
        client_backend=CortexConfig(base_url="http://unused"),
        target_model_name=None,
    )
    monkeypatch.setattr(e2e, "gateway", lambda *args, **kwargs: pytest.fail("must reject before allocation"))
    monkeypatch.setattr(e2e, "request_bodies", lambda *args, **kwargs: ([], {}))
    with pytest.raises(ValueError, match="published Cortex fixture"):
        e2e.run_dss_trajectory(ctx, SimpleNamespace(), tmp_path)
