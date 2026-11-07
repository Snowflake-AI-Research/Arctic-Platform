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

"""CPU admission controls; no model, reference subprocess or gateway is started."""

import copy
import json
from pathlib import Path

import pytest

from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.correctness.harness import config as config_module
from arctic_platform.correctness.harness import execution
from arctic_platform.correctness.harness import gateway
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness.registry import TestOutcome
from arctic_platform.correctness.onboarding import onboard

MISSING = object()
CHECKS = ["single-step-grads", "single-step-optimizer"]


@pytest.fixture(params=["native", "cortex"])
def recipe(tmp_path, request):
    def create(dtype="bfloat16", *, training_dtype=MISSING, accumulation="fp32"):
        worker = {"sp_size": 1, "attn_implementation": "sdpa"}
        if training_dtype is not MISSING:
            worker["dtype"] = training_dtype
        ds = {
            "bf16": {"enabled": True},
            "data_types": {"grad_accum_dtype": accumulation},
        }
        if request.param == "native":
            raw = {
                "model_name": "Qwen/Qwen3-0.6B",
                "training_gpus": 1,
                "max_seq_len": 128,
                "training": {"ds_config": ds, "ds_worker_config": worker},
            }
            if dtype is not MISSING:
                raw["dtype"] = dtype
        else:
            sub_job = {
                "job_type": "training",
                "model_name": "Qwen/Qwen3-0.6B",
                "training_config": {"n_gpus": 1, "max_seq_len": 128, "ds_config": ds, **worker},
            }
            if dtype is not MISSING:
                sub_job["dtype"] = dtype
            raw = {"sub_job_configs": [sub_job]}
        source = tmp_path / "supplied.json"
        source.write_text(json.dumps(raw) + "\n")
        return config_module.load_config(source)

    return create


def untouched(config):
    return (
        config.path.read_bytes(),
        copy.deepcopy(config.sub_job),
        config.native.model_dump() if config.native is not None else None,
        config_module.config_checksum(config),
    )


def forbidden(*args, **kwargs):
    pytest.fail("dtype admission reached model preparation, placement or target execution")


@pytest.mark.parametrize("dtype", ["float32", "fp32", "float16", "fp16", "float64", "not-a-dtype"])
def test_declared_non_bf16_model_dtype_is_refused_without_mutation(recipe, dtype):
    config = recipe(dtype)
    before = untouched(config)
    with pytest.raises(ValueError, match="support only BF16 model dtype") as caught:
        config_module.validate_first_two_model_dtype(config)
    assert config.config_id in str(caught.value)
    assert f"sub_job.dtype={dtype!r}" in str(caught.value)
    assert "FP32 gradient accumulation is a separate setting" in str(caught.value)
    assert untouched(config) == before


@pytest.mark.parametrize("dtype", [MISSING, None, "bfloat16", "bf16", "BF16", "BFloat16"])
def test_bf16_aliases_and_default_keep_fp32_accumulation_and_identity(recipe, dtype):
    config = recipe(dtype)
    before = untouched(config)
    config_module.validate_first_two_model_dtype(config)
    assert config.training["ds_config"]["data_types"]["grad_accum_dtype"] == "fp32"
    assert untouched(config) == before


@pytest.mark.parametrize("dtype, training_dtype", [("float32", "bf16"), ("bf16", "float32"), (None, "fp32")])
def test_training_override_cannot_hide_non_bf16_model_dtype(recipe, dtype, training_dtype):
    config = recipe(dtype, training_dtype=training_dtype)
    before = untouched(config)
    with pytest.raises(ValueError, match="support only BF16 model dtype"):
        config_module.validate_first_two_model_dtype(config)
    assert untouched(config) == before


@pytest.mark.parametrize("dtype", ["", " bf16 ", "torch.bfloat16", 0, False, ["bf16"]])
def test_invalid_cortex_dtype_is_not_treated_as_default(tmp_path, dtype):
    config = config_module.LoadedConfig(
        config_id="invalid-dtype",
        path=tmp_path / "unused.json",
        sub_job={"dtype": dtype, "training_config": {}},
    )
    with pytest.raises(ValueError, match="support only BF16 model dtype"):
        config_module.validate_first_two_model_dtype(config)


@pytest.mark.parametrize("test_id", CHECKS)
def test_onboarding_preflight_refuses_fp32_before_fixture_or_cached_prerequisites(recipe, monkeypatch, test_id):
    config = recipe("float32")
    args = execution.parse_args(["onboard", "--config", str(config.path), "--test-id", test_id])
    args._prerequisites_checked = True
    monkeypatch.setattr(onboard, "_configure_fixture_target", forbidden)
    before = untouched(config)
    with pytest.raises(ValueError, match="support only BF16 model dtype"):
        onboard._onboarding_preflight(args, config)
    assert untouched(config) == before


@pytest.mark.parametrize("dtype", ["bfloat16", "BF16"])
def test_onboarding_bf16_with_fp32_accumulation_still_reaches_fixture_admission(recipe, monkeypatch, dtype):
    config = recipe(dtype)
    args = execution.parse_args(["onboard", "--config", str(config.path), "--test-id", CHECKS[0]])
    args._prerequisites_checked = True
    calls = []
    monkeypatch.setattr(onboard, "_configure_fixture_target", lambda *args: calls.append("fixture"))
    before = untouched(config)
    onboard._onboarding_preflight(args, config)
    assert calls == ["fixture"]
    assert untouched(config) == before


def test_bf16_still_requires_fp32_gradient_accumulation(recipe, monkeypatch):
    config = recipe(accumulation="bf16")
    args = execution.parse_args(["onboard", "--config", str(config.path), "--test-id", CHECKS[0]])
    args._prerequisites_checked = True
    monkeypatch.setattr(onboard, "_configure_fixture_target", forbidden)
    before = untouched(config)
    with pytest.raises(ValueError, match="Set data_types.grad_accum_dtype"):
        onboard._onboarding_preflight(args, config)
    assert untouched(config) == before


@pytest.mark.parametrize("operation", ["calibration", "reference-preparation", "probe", "completed-summary"])
def test_onboarding_run_refuses_fp32_before_preparation_or_completion_shortcuts(recipe, monkeypatch, operation):
    config = recipe("float32")
    args = execution.parse_args(["onboard", "--config", str(config.path), "--test-id", CHECKS[0]])
    if operation == "reference-preparation":
        args.prepare_references = True
        args.reference_tests = CHECKS
    if operation == "probe":
        args.probe_layers = 1
    if operation == "completed-summary":
        args.test_spec = config.path
    monkeypatch.setattr(config_module, "local_config_path", lambda path: Path(path))
    for name in ("_prepare_references", "_probe_worker", "_onboard_configuration", "_completed_onboarding", "_parent"):
        monkeypatch.setattr(onboard, name, forbidden)
    before = untouched(config)
    with pytest.raises(ValueError, match="support only BF16 model dtype"):
        onboard.run(args)
    assert untouched(config) == before


def test_onboarding_run_admits_bf16_fp32_accumulator_without_changing_recipe(recipe, monkeypatch):
    config = recipe()
    args = execution.parse_args(["onboard", "--config", str(config.path)])
    monkeypatch.setattr(config_module, "local_config_path", lambda path: Path(path))
    monkeypatch.setattr(onboard, "_onboard_configuration", lambda args: 23)
    before = untouched(config)
    assert onboard.run(args) == 23
    assert untouched(config) == before


@pytest.mark.parametrize("test_id", CHECKS)
@pytest.mark.parametrize("prepare_references", [False, True])
def test_direct_runner_refuses_fp32_before_workdir_placement_reference_or_target(
    recipe, monkeypatch, test_id, prepare_references
):
    config = recipe("float32")
    monkeypatch.setattr(config, "at_gpu_width", forbidden)
    for name in ("correctness_workdir", "run_reference", "build_payload", "gateway"):
        monkeypatch.setattr(runner, name, forbidden)
    before = untouched(config)
    with pytest.raises(ValueError, match="support only BF16 model dtype"):
        runner.execute(
            config,
            None,
            [],
            vocab_size=2,
            select=[test_id],
            prepare_references=prepare_references,
            workdir=Path("/code/not-an-admitted-model-workdir"),
        )
    assert untouched(config) == before


def test_direct_runner_admits_bf16_before_existing_placement(recipe, monkeypatch):
    config = recipe()
    calls = []

    def placed(width):
        calls.append(width)
        raise RuntimeError("admitted placement sentinel")

    monkeypatch.setattr(config, "at_gpu_width", placed)
    before = untouched(config)
    with pytest.raises(RuntimeError, match="admitted placement sentinel"):
        runner.execute(config, None, [], vocab_size=2, select=CHECKS)
    assert calls == [1]
    assert untouched(config) == before


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_regression_operator_admits_dtype_before_spec_or_target(recipe, monkeypatch, tmp_path, remote, dtype):
    config = recipe(dtype)
    args = execution.parse_args(["run", "--config", str(config.path), "--test", CHECKS[0], "--any-gpu"])
    args.out = tmp_path / "report"
    backend = CortexConfig(base_url="http://unused.invalid") if remote else OnPremConfig()
    monkeypatch.setattr(execution, "_configure_regression_target", lambda args: args)
    monkeypatch.setattr(execution, "_client_backend", lambda args: backend)
    monkeypatch.setattr(execution, "_resolve_configs", lambda *args, **kwargs: [config.path])
    monkeypatch.setattr(execution, "ensure_no_interrupted_builds", lambda: None)
    monkeypatch.setattr(execution, "local_cortex_gateway", forbidden)
    monkeypatch.setattr(execution, "_run_vocab_size", forbidden)
    monkeypatch.setattr(execution, "available_gpus", forbidden if dtype == "float32" else lambda: 1)
    specs = []
    results = []
    monkeypatch.setattr(execution, "_frozen_spec", lambda *args, **kwargs: specs.append("spec"))

    def finish(produced, *args, **kwargs):
        results.extend(produced)
        return 1 if produced else 0

    monkeypatch.setattr(execution, "_finish", finish)
    before = untouched(config)
    assert execution._run_configs(args, executor=forbidden) == (1 if dtype == "float32" else 0)
    if dtype == "float32":
        assert specs == []
        assert len(results) == 1
        assert results[0].outcome is TestOutcome.FAIL
        assert "support only BF16 model dtype" in results[0].reason
    else:
        assert specs == ["spec"]
        assert results == []
    assert untouched(config) == before


@pytest.mark.parametrize("dtype", [None, "bf16", "BF16", "bfloat16", "BFloat16"])
def test_payload_uses_canonical_bf16_without_changing_frozen_recipe(recipe, dtype):
    config = recipe(training_dtype=dtype)
    before = untouched(config)
    config_module.validate_first_two_model_dtype(config)
    payload = gateway.build_payload(config.training, "/unused/model", seed=0, determinism="off")
    assert payload["dtype"] == "bfloat16"
    assert payload["training_config"]["dtype"] == "bfloat16"
    assert untouched(config) == before


@pytest.mark.parametrize("dtype", ["float32", "fp32", "float16", "fp16"])
def test_training_dtype_override_is_rejected_before_target_work(recipe, dtype):
    config = recipe(training_dtype=dtype)
    before = untouched(config)
    with pytest.raises(ValueError, match="training_config.dtype"):
        config_module.validate_first_two_model_dtype(config)
    assert untouched(config) == before
