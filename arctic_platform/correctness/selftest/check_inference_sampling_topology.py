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

"""CPU-only request topology controls, without model, native vLLM or network execution."""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.correctness.checks import inference_checkpoint_loss as check
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.cortex import CortexTrainingConfig
from arctic_platform.correctness.harness.dss_driver import _inference_client_config
from arctic_platform.correctness.harness.dss_driver import sampling_payload
from arctic_platform.correctness.harness.seeds import SEED

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
FLASH_CONFIG = CONFIG_ROOT / "qwen3.8-flash-next/h200/train-sft-full-16gpus-sp16-256k.config"
ORDINARY_CONFIG = CONFIG_ROOT / "qwen3-8b/h200/train-sft-full-4gpus-64k.config"
FLASH_ID = "qwen3.8-flash-next-h200-train-sft-full-16gpus-sp16-256k"
TOPOLOGIES = [(8, 8, 1), (16, 8, 2)]
TOKEN_BUDGET = 10240


def context(config=FLASH_CONFIG, topology=None):
    cfg = load_config(config)
    return SimpleNamespace(
        cfg=cfg,
        config_id=cfg.config_id,
        inference_sampling_topology=topology,
        client_backend=OnPremConfig(),
        attn_implementation="qsa_flex",
    )


def assert_topology(payload, topology):
    allocation, tp, pp = topology
    inf = payload["inference_config"]
    engine = inf["vllm_config"]
    assert inf["n_gpus"] == allocation == tp * pp
    assert engine["tensor_parallel_size"] == tp
    assert engine["pipeline_parallel_size"] == pp
    assert engine["enable_expert_parallel"] is True
    extra = engine["extra_engine_kwargs"]
    assert extra["distributed_executor_backend"] == "ray"
    assert extra["data_parallel_size"] == 1
    assert extra["prefill_context_parallel_size"] == 1
    assert extra["decode_context_parallel_size"] == 1
    assert extra["enable_eplb"] is False
    assert extra["enable_elastic_ep"] is False
    assert extra.get("nnodes", 1) == 1
    assert extra["engram_config"] == {
        "cpu_offload": False,
        "embedding_across_dp": False,
        "dp_shared_memory": False,
        "use_thp": False,
    }


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_original_flashnext_request_separates_allocation_from_tp_and_pp(topology):
    ctx = context(topology=topology)
    assert ctx.cfg.config_id == FLASH_ID
    before = deepcopy(ctx.cfg)
    payload = check._sampling_payload(ctx, "/saved/full-checkpoint", TOKEN_BUDGET)
    assert check.sampling_gpu_count(ctx) == topology[0]
    assert_topology(payload, topology)
    assert payload["model_name"] == "/saved/full-checkpoint"
    assert payload["job_type"] == "log_prob"
    assert payload["seed"] == SEED
    assert payload["dtype"] == "bfloat16"
    inf = payload["inference_config"]
    assert inf["max_seq_len"] == TOKEN_BUDGET + 1
    engine = inf["vllm_config"]
    assert engine["max_model_len"] == TOKEN_BUDGET + 1
    assert engine["dtype"] == "bfloat16"
    assert engine["enforce_eager"] is True
    assert engine["disable_custom_all_reduce"] is True
    assert engine["gpu_memory_utilization"] == 0.85
    assert engine["trust_remote_code"] is True
    assert "model" not in engine
    assert "speculative_config" not in engine
    assert "speculative_config" not in engine["extra_engine_kwargs"]
    assert ctx.cfg == before


@pytest.mark.parametrize("config", [FLASH_CONFIG, ORDINARY_CONFIG])
@pytest.mark.parametrize("missing_attribute", [False, True])
def test_without_opt_in_the_exact_one_gpu_payload_is_unchanged(config, missing_attribute):
    ctx = context(config)
    if missing_attribute:
        del ctx.inference_sampling_topology
    assert check.sampling_gpu_count(ctx) == 1
    assert check._sampling_payload(ctx, "/saved/full-checkpoint", TOKEN_BUDGET) == sampling_payload(
        "/saved/full-checkpoint", SEED, dtype="bfloat16", n_gpus=1, max_seq_len=TOKEN_BUDGET + 1
    )


@pytest.mark.parametrize(
    "topology",
    [
        (),
        (16, 8),
        (16, 8, 2, 1),
        [16, 8, 2],
        {"allocation": 16, "tp": 8, "pp": 2},
        "16,8,2",
        (True, 8, 2),
        (16, True, 2),
        (16, 8, True),
        (0, 8, 2),
        (16, -8, 2),
        (16, 8.0, 2),
        (16, "8", 2),
        (8, 8, 2),
        (16, 16, 1),
        (8, 4, 2),
        (1, 1, 1),
    ],
)
def test_malformed_or_unreviewed_topology_is_rejected(topology):
    ctx = context(topology=topology)
    with pytest.raises(ValueError):
        check.sampling_gpu_count(ctx)
    with pytest.raises(ValueError):
        check._sampling_payload(ctx, "/saved/full-checkpoint", TOKEN_BUDGET)


@pytest.mark.parametrize("different_identity", ["config_id", "model_name", "training_width"])
def test_flashnext_opt_in_requires_exact_original_training_identity(different_identity):
    ctx = context(topology=(16, 8, 2))
    if different_identity == "config_id":
        ctx.cfg.config_id += "-alias"
        ctx.config_id = ctx.cfg.config_id
    elif different_identity == "model_name":
        ctx.cfg.sub_job["model_name"] = "Qwen/Qwen3-8B"
        ctx.cfg.native.model_name = "Qwen/Qwen3-8B"
    else:
        ctx.cfg.native.training_gpus = 8
        ctx.cfg.training["n_gpus"] = 8
    with pytest.raises(ValueError):
        check.sampling_gpu_count(ctx)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_managed_sampling_uses_helper_and_preserves_topology_on_cortex_wire(monkeypatch, topology):
    ctx = context(topology=topology)
    backend = CortexConfig(base_url="https://unused.invalid")
    calls = []
    original = check._sampling_payload

    def observe(actual_ctx, model_path, token_budget):
        calls.append((actual_ctx, model_path, token_budget))
        return original(actual_ctx, model_path, token_budget)

    monkeypatch.setattr(check, "_sampling_payload", observe)
    config = check.cortex_sampling_config(ctx, backend, "cp_full_saved", "owned-training", TOKEN_BUDGET)
    assert calls == [(ctx, ctx.cfg.model_name, TOKEN_BUDGET)]
    before = deepcopy(config.model_dump())
    sub = config.to_cortex()[0]
    assert_topology(sub, topology)
    assert sub["model_name"] == ctx.cfg.model_name
    assert sub["source_checkpoint_info"] == {"checkpoint_id": "cp_full_saved", "source_job_id": "owned-training"}
    assert sub["inference_config"]["max_seq_len"] == TOKEN_BUDGET + 1
    assert "max_model_len" not in sub["inference_config"]["vllm_config"]
    assert config.backend == backend
    assert config.hardware == "H200"
    sub["inference_config"]["vllm_config"]["extra_engine_kwargs"]["engram_config"]["cpu_offload"] = True
    assert config.model_dump() == before
    assert_topology(config.to_cortex()[0], topology)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_local_sampling_validates_export_then_uses_same_helper(monkeypatch, tmp_path, topology):
    ctx = context(topology=topology)
    backend = CortexConfig(base_url="http://127.0.0.1:8003")
    weights = tmp_path / "saved-full-export"
    events = []
    original = check._sampling_payload

    def validate(actual_ctx, actual_weights):
        assert actual_ctx is ctx
        assert actual_weights == weights
        events.append("validate")
        return actual_weights

    def observe(actual_ctx, model_path, token_budget):
        assert actual_ctx is ctx
        assert str(model_path) == str(weights)
        assert token_budget == TOKEN_BUDGET
        events.append("helper")
        return original(actual_ctx, model_path, token_budget)

    monkeypatch.setattr(check, "validate_local_inference_export", validate)
    monkeypatch.setattr(check, "_sampling_payload", observe)
    config = check.local_cortex_sampling_config(ctx, backend, weights, TOKEN_BUDGET)
    assert events == ["validate", "helper"]
    sub = config.to_cortex()[0]
    assert_topology(sub, topology)
    assert sub["model_name"] == str(weights)
    assert "source_checkpoint_info" not in sub
    assert config.source_checkpoint_info is None
    assert config.backend == backend


def test_local_topology_never_bypasses_full_export_rejection(monkeypatch, tmp_path):
    ctx = context(topology=(16, 8, 2))
    backend = CortexConfig(base_url="http://127.0.0.1:8003")

    def invalid_export(*args):
        raise RuntimeError("full configured tensor inventory missing")

    def forbidden_helper(*args):
        pytest.fail("sampling helper called after rejected full export")

    monkeypatch.setattr(check, "validate_local_inference_export", invalid_export)
    monkeypatch.setattr(check, "_sampling_payload", forbidden_helper)
    with pytest.raises(RuntimeError, match="full configured tensor inventory"):
        check.local_cortex_sampling_config(ctx, backend, tmp_path / "incomplete", TOKEN_BUDGET)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_onprem_sampling_uses_same_helper_before_engine_creation(monkeypatch, tmp_path, topology):
    ctx = context(topology=topology)
    session = object()
    weights = tmp_path / "saved-full-export"
    calls = []
    original = check._sampling_payload

    class StopBeforeEngine(Exception):
        pass

    def observe(actual_ctx, model_path, token_budget):
        assert actual_ctx is ctx
        assert model_path == weights
        assert token_budget == TOKEN_BUDGET
        calls.append("helper")
        return original(actual_ctx, model_path, token_budget)

    @contextmanager
    def engine(actual_session, payload):
        assert actual_session is session
        assert calls == ["helper"]
        assert_topology(payload, topology)
        raise StopBeforeEngine
        yield

    monkeypatch.setattr(check, "_sampling_payload", observe)
    monkeypatch.setattr(check, "running_job", engine)
    with pytest.raises(StopBeforeEngine):
        check._onprem_sampling_and_reload(ctx, session, weights, TOKEN_BUDGET, [], {})


def test_production_client_cortex_serializer_remains_outside_correctness_override():
    payload = sampling_payload("Qwen/Qwen3-8B", SEED, dtype="bfloat16", n_gpus=4, max_seq_len=128)
    payload["job_type"] = "sampling"
    config = _inference_client_config(payload, backend=CortexConfig(base_url="https://unused.invalid"))
    before = deepcopy(config.model_dump())
    sub = config.to_cortex()[0]
    assert sub["inference_config"]["n_gpus"] == 4
    assert "tensor_parallel_size" not in sub["inference_config"]["vllm_config"]
    assert config.model_dump() == before


@pytest.mark.parametrize("config_path", [FLASH_CONFIG, ORDINARY_CONFIG])
def test_correctness_one_gpu_wire_does_not_opt_in_implicitly(config_path):
    ctx = context(config_path)
    backend = CortexConfig(base_url="https://unused.invalid")
    config = check.cortex_sampling_config(ctx, backend, "cp_full_saved", "owned-training", TOKEN_BUDGET)
    before = deepcopy(config.model_dump())
    sub = config.to_cortex()[0]
    assert config.inference_tensor_parallel_size is None
    assert config.sampling_gpus == sub["inference_config"]["n_gpus"] == 1
    assert "tensor_parallel_size" not in sub["inference_config"]["vllm_config"]
    assert "pipeline_parallel_size" not in sub["inference_config"]["vllm_config"]
    assert "enable_expert_parallel" not in sub["inference_config"]["vllm_config"]
    assert config.model_dump() == before


@pytest.mark.parametrize(
    "invalid",
    [
        "tensor_bool",
        "tensor_float",
        "tensor_zero",
        "pipeline_bool",
        "allocation_mismatch",
        "engine_tensor_mismatch",
        "engine_tensor_bool",
        "no_inference_allocation",
        "non_cortex_backend",
    ],
)
def test_scoped_cortex_field_rejects_inconsistent_or_non_cortex_topology(invalid):
    ctx = context(topology=(16, 8, 2))
    backend = CortexConfig(base_url="https://unused.invalid")
    config = check.cortex_sampling_config(ctx, backend, "cp_full_saved", "owned-training", TOKEN_BUDGET)
    values = deepcopy(config.model_dump())
    if invalid == "tensor_bool":
        values["inference_tensor_parallel_size"] = True
    elif invalid == "tensor_float":
        values["inference_tensor_parallel_size"] = 8.0
    elif invalid == "tensor_zero":
        values["inference_tensor_parallel_size"] = 0
    elif invalid == "pipeline_bool":
        values["sampling"]["vllm"]["pipeline_parallel_size"] = True
    elif invalid == "allocation_mismatch":
        values["sampling_gpus"] = 8
    elif invalid == "engine_tensor_mismatch":
        values["sampling"]["vllm"]["tensor_parallel_size"] = 4
    elif invalid == "engine_tensor_bool":
        values["inference_tensor_parallel_size"] = 1
        values["sampling"]["vllm"]["tensor_parallel_size"] = True
        values["sampling"]["vllm"]["pipeline_parallel_size"] = 16
    elif invalid == "no_inference_allocation":
        values["sampling_gpus"] = 0
    else:
        values["backend"] = OnPremConfig().model_dump()
    with pytest.raises(ValueError):
        CortexTrainingConfig.model_validate(values)


@pytest.mark.parametrize("invalid_identity", [False, True])
def test_invalid_opt_in_fails_before_training_or_gateway_creation(monkeypatch, tmp_path, invalid_identity):
    ctx = context(ORDINARY_CONFIG if invalid_identity else FLASH_CONFIG, topology=(16, 8, 2))
    if not invalid_identity:
        ctx.inference_sampling_topology = (16, 16, 1)
    ctx.workdir = tmp_path

    def forbidden(*args, **kwargs):
        pytest.fail("invalid sampling topology reached preparation, gateway or training")

    monkeypatch.setattr(check, "_full_model_name", forbidden)
    monkeypatch.setattr(check.gsm8k, "load_tokenizer", forbidden)
    monkeypatch.setattr(check, "gateway", forbidden)
    monkeypatch.setattr(check, "running_job", forbidden)
    with pytest.raises(ValueError):
        check._run(ctx)


@pytest.mark.parametrize(
    "config_path,topology,expected_slots",
    [(FLASH_CONFIG, (8, 8, 1), 16), (FLASH_CONFIG, (16, 8, 2), 16), (ORDINARY_CONFIG, None, 5)],
)
def test_opt_in_gateway_reuses_training_capacity_only_after_destruction(
    monkeypatch, tmp_path, config_path, topology, expected_slots
):
    ctx = context(config_path, topology=topology)
    ctx.workdir = tmp_path
    events = []
    steps = []
    rows = [[-2.0]]
    distribution = {"examples": 1, "minimum_tokens": 4, "median_tokens": 4, "maximum_tokens": 4}
    monkeypatch.setattr(check.gsm8k, "load_tokenizer", lambda path: object())
    monkeypatch.setattr(check.gsm8k, "padding_token_id", lambda tokenizer: 0)
    monkeypatch.setattr(check.gsm8k, "read_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(check.gsm8k, "tokenize_examples", lambda *args: [])
    monkeypatch.setattr(check.gsm8k, "length_distribution", lambda rows: distribution)
    monkeypatch.setattr(check.gsm8k, "select_by_token_budget", lambda *args: ([object()], 4))
    monkeypatch.setattr(check.gsm8k, "assert_disjoint", lambda *args: None)
    monkeypatch.setattr(check.gsm8k, "build_batch", lambda *args: object())
    monkeypatch.setattr(check, "training_batches", lambda *args: [object()] * check.TRAIN_STEPS)
    monkeypatch.setattr(check, "pack_logit_aligned", lambda batch: {})
    monkeypatch.setattr(check, "pack", lambda *args, **kwargs: {})
    monkeypatch.setattr(check, "build_payload", lambda *args, **kwargs: {"job_type": "training"})
    monkeypatch.setattr(check, "forward_logprobs", lambda *args: None)
    monkeypatch.setattr(check, "reference_answer_logprobs", lambda *args: rows)
    monkeypatch.setattr(check, "save_weights_only_checkpoint", lambda *args: tmp_path / "full-export")
    monkeypatch.setattr(check, "materialize_checkpoint_tokenizer", lambda *args: None)
    monkeypatch.setattr(check, "verdict", lambda *args: "synthetic-routing-only")

    def step(*args, **kwargs):
        steps.append(kwargs["learning_rate"])
        return SimpleNamespace(avg_loss=2.0)

    @contextmanager
    def gateway(path, slots, *, backend):
        assert slots == expected_slots
        assert isinstance(backend, OnPremConfig)
        events.append("gateway")
        yield object()

    @contextmanager
    def training(session, payload):
        assert payload["job_type"] == "training"
        events.append("training_start")
        try:
            yield object()
        finally:
            events.append("training_destroyed")

    def sampling(actual_ctx, session, weights_dir, token_budget, examples, validation_body):
        assert actual_ctx is ctx
        assert events == ["gateway", "training_start", "training_destroyed"]
        events.append("sampling")
        return rows, rows

    monkeypatch.setattr(check, "gateway", gateway)
    monkeypatch.setattr(check, "running_job", training)
    monkeypatch.setattr(check, "fwd_bwd_step", step)
    monkeypatch.setattr(check, "_onprem_sampling_and_reload", sampling)
    assert check._run(ctx) == "synthetic-routing-only"
    assert events == ["gateway", "training_start", "training_destroyed", "sampling"]
    assert steps == [check.configured_learning_rate(ctx.cfg)] * check.TRAIN_STEPS
