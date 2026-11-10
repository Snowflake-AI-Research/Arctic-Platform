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
import hashlib
import io
import json
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest

from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.correctness.harness import config as config_module
from arctic_platform.correctness.harness import engine_admission
from arctic_platform.correctness.harness import execution
from arctic_platform.correctness.harness import gateway
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness import storage
from arctic_platform.correctness.harness.registry import TestOutcome
from arctic_platform.correctness.onboarding import onboard

MISSING = object()
CHECKS = ["single-step-grads", "single-step-optimizer"]


@pytest.fixture
def fast_storage(tmp_path, monkeypatch):
    """Exercise the real storage validator without requiring a host /data-fast mount."""
    root = tmp_path / "data-fast"
    root.mkdir()
    monkeypatch.setattr(storage, "_DATA_FAST_ROOT", root)
    return root


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
def test_onboarding_run_refuses_fp32_before_preparation_or_completion_shortcuts(
    recipe, monkeypatch, operation, fast_storage
):
    config = recipe("float32")
    args = execution.parse_args(
        ["onboard", "--config", str(config.path), "--test-id", CHECKS[0], "--cache-root", str(fast_storage / "models")]
    )
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


def test_onboarding_run_admits_bf16_fp32_accumulator_without_changing_recipe(recipe, monkeypatch, fast_storage):
    config = recipe()
    args = execution.parse_args(
        ["onboard", "--config", str(config.path), "--cache-root", str(fast_storage / "models")]
    )
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


@pytest.mark.parametrize("accumulation", ["fp32", "bf16", None])
@pytest.mark.parametrize("automatic", [False, True])
def test_runtime_default_admission_preserves_explicit_settings_and_recipe(
    recipe, monkeypatch, accumulation, automatic
):
    config = recipe(accumulation=accumulation)
    before = untouched(config)
    calls = []

    def resolved(config, *, automatically_owned_local):
        calls.append(automatically_owned_local)
        return {"bf16": {"enabled": True}, "data_types": {"grad_accum_dtype": "fp32"}}

    monkeypatch.setattr(engine_admission, "_owned_local_engine_config", resolved)
    if accumulation == "bf16":
        with pytest.raises(ValueError, match="requires grad_accum_dtype='fp32'"):
            engine_admission.validate_reference_accumulation(config, automatically_owned_local=automatic)
    elif accumulation is None and not automatic:
        with pytest.raises(ValueError, match="unverified defaults"):
            engine_admission.validate_reference_accumulation(config)
    else:
        engine_admission.validate_reference_accumulation(config, automatically_owned_local=automatic)
    assert calls == ([True] if accumulation is None and automatic else [])
    assert untouched(config) == before


@pytest.mark.parametrize(
    "resolved, message",
    [
        ({"bf16": {"enabled": True}}, "not FP32"),
        ({"bf16": {"enabled": True}, "data_types": {"grad_accum_dtype": None}}, "not FP32"),
        ({"bf16": {"enabled": True}, "data_types": {"grad_accum_dtype": "bf16"}}, "not FP32"),
        ({"bf16": {"enabled": False}}, "requires BF16"),
        ({"bf16": {"enabled": True}, "fp16": {"enabled": True}}, "without FP16"),
        ({"bf16": {"enabled": True}, "torch_autocast": {"enabled": True}}, "without FP16 or autocast"),
        (
            {"bf16": {"enabled": True}, "data_types": {"param_dtype": "fp32", "grad_accum_dtype": "fp32"}},
            "BF16 model parameters",
        ),
    ],
)
def test_runtime_default_admission_refuses_incompatible_resolved_config(recipe, monkeypatch, resolved, message):
    config = recipe(accumulation=None)
    before = untouched(config)
    monkeypatch.setattr(
        engine_admission, "_owned_local_engine_config", lambda *args, **kwargs: copy.deepcopy(resolved)
    )
    with pytest.raises(ValueError, match=message):
        engine_admission.validate_reference_accumulation(config, automatically_owned_local=True)
    assert untouched(config) == before


@pytest.fixture
def installed_runtime(monkeypatch):
    """Use portable schema/builder doubles without importing a separately installed DSS service."""
    import torch

    calls = []
    schema = ModuleType("dss.ray_dss.models")

    class JobConfig:
        @staticmethod
        def model_validate(payload):
            calls.append(("payload", copy.deepcopy(payload)))
            return SimpleNamespace(to_engine_dict=lambda name: {"name": name, "submitted": payload})

    schema.JobConfig = JobConfig
    runtime = ModuleType("dss.ray_dss.jobs.gpu.training")

    def build(config, width, sp_size):
        calls.append(("builder", config, width, sp_size))
        return {"bf16": {"enabled": True}, "data_types": {"grad_accum_dtype": "fp32"}}

    runtime._build_ds_config = build
    package = ModuleType("dss.ray_dss.jobs.gpu")
    package.training = runtime
    monkeypatch.setitem(sys.modules, schema.__name__, schema)
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(gateway, "_allocation_hostfile", lambda: None)
    return SimpleNamespace(runtime=runtime, calls=calls)


def test_owned_runtime_resolution_uses_submitted_payload_without_overrides(recipe, installed_runtime):
    config = recipe(accumulation=None)
    before = untouched(config)
    resolved = engine_admission._owned_local_engine_config(config, automatically_owned_local=True)
    expected = gateway.training_client_config(config, CortexConfig(base_url="http://127.0.0.1:1")).to_cortex()[0]
    assert installed_runtime.calls[0] == ("payload", expected)
    assert installed_runtime.calls[1][2:] == (config.n_gpus, config.sp_size)
    assert expected["training_config"]["ds_config"]["data_types"]["grad_accum_dtype"] is None
    assert resolved["data_types"]["grad_accum_dtype"] == "fp32"
    assert untouched(config) == before


@pytest.mark.parametrize("fault", ["unowned", "missing-builder", "nonlocal-width", "builder-error"])
def test_owned_runtime_resolution_fails_closed(recipe, installed_runtime, monkeypatch, fault):
    config = recipe(accumulation=None)
    before = untouched(config)
    message = {
        "unowned": "automatically owned local",
        "missing-builder": "does not expose _build_ds_config",
        "nonlocal-width": "requires a same-job hostfile for nonlocal workers",
        "builder-error": "synthetic builder failure",
    }[fault]
    if fault == "missing-builder":
        del installed_runtime.runtime._build_ds_config
    elif fault == "nonlocal-width":
        import torch

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    elif fault == "builder-error":

        def broken(*args):
            raise RuntimeError("synthetic builder failure")

        installed_runtime.runtime._build_ds_config = broken
    with pytest.raises(ValueError, match=message):
        engine_admission._owned_local_engine_config(config, automatically_owned_local=fault != "unowned")
    assert untouched(config) == before


@pytest.fixture(params=[(False, 0), (True, 1)])
def selected_hostfile_placement(installed_runtime, monkeypatch, tmp_path, request):
    local, gpu = request.param
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("selected-worker slots=2\n")
    node = SimpleNamespace(host="selected-worker", gpu_ids=[gpu])
    placement = ModuleType("dss.ray_dss.placement.local")

    def backend(*, hostfile):
        assert hostfile == str(tmp_path / "hostfile")
        return SimpleNamespace(_pick_hosts=lambda width: [node])

    placement.LocalPlacementBackend = backend
    utilities = ModuleType("dss.ray_dss.utils")
    utilities.is_local = lambda host: local
    monkeypatch.setitem(sys.modules, placement.__name__, placement)
    monkeypatch.setitem(sys.modules, utilities.__name__, utilities)
    monkeypatch.setattr(gateway, "_allocation_hostfile", lambda: hostfile)
    return SimpleNamespace(hostfile=hostfile, gpu=gpu)


def test_owned_runtime_resolution_refuses_unverified_hostfile_placement(
    recipe, installed_runtime, selected_hostfile_placement, monkeypatch
):
    config = recipe(accumulation=None)
    before = untouched(config)
    calls = []

    def reject(request, hostfile):
        calls.append(request)
        assert hostfile == selected_hostfile_placement.hostfile
        assert storage.parse_hostfile(hostfile)[0].host == "selected-worker"
        raise ValueError("unverified allocation worker runtime")

    monkeypatch.setattr(engine_admission, "_allocation_runtime_receipts", reject)
    with pytest.raises(ValueError, match="unverified allocation worker runtime"):
        engine_admission._owned_local_engine_config(config, automatically_owned_local=True)
    assert len(calls) == 1
    assert installed_runtime.calls == []
    assert untouched(config) == before


@pytest.mark.parametrize("gpu_count", [1, 2])
def test_owned_runtime_resolution_uses_verified_hostfile_placement(
    recipe, installed_runtime, selected_hostfile_placement, monkeypatch, gpu_count
):
    config = recipe(accumulation=None)
    before = untouched(config)
    expected = gateway.training_client_config(config, CortexConfig(base_url="http://127.0.0.1:1")).to_cortex()[0]
    resolved = {"bf16": {"enabled": True}, "data_types": {"grad_accum_dtype": "fp32"}}
    calls = []

    def receipts(request, hostfile):
        calls.append(request)
        assert hostfile == selected_hostfile_placement.hostfile
        assert request == {"payload": expected, "n_gpus": config.n_gpus, "sp_size": config.sp_size}
        return {"selected-worker": {"gpu_count": gpu_count, "resolved": copy.deepcopy(resolved)}}

    monkeypatch.setattr(engine_admission, "_allocation_runtime_receipts", receipts)
    if selected_hostfile_placement.gpu >= gpu_count:
        with pytest.raises(ValueError, match="selected unavailable GPUs"):
            engine_admission.validate_reference_accumulation(config, automatically_owned_local=True)
    else:
        engine_admission.validate_reference_accumulation(config, automatically_owned_local=True)
    assert len(calls) == 1
    assert installed_runtime.calls == []
    assert untouched(config) == before


def test_allocation_runtime_resolution_refuses_disagreeing_workers(monkeypatch, tmp_path):
    layout = [SimpleNamespace(host=host, gpu_ids=[0]) for host in ("head", "worker")]
    receipts = {
        host: {"gpu_count": 1, "resolved": {"bf16": {"enabled": True}, "data_types": {"grad_accum_dtype": dtype}}}
        for host, dtype in (("head", "fp32"), ("worker", "bf16"))
    }
    monkeypatch.setattr(engine_admission, "_allocation_runtime_receipts", lambda *args: receipts)
    with pytest.raises(ValueError, match="workers resolve different submitted engine configurations"):
        engine_admission._allocation_engine_configs({}, tmp_path / "hostfile", layout)


@pytest.mark.asyncio
@pytest.mark.parametrize("admitted", [False, True])
async def test_gateway_runtime_scope_does_not_load_dss_for_nontraining_jobs(monkeypatch, tmp_path, admitted):
    monkeypatch.delenv(engine_admission.RUNTIME_ADMISSION_ENV, raising=False)
    if admitted:
        monkeypatch.setenv(engine_admission.RUNTIME_ADMISSION_ENV, "{}")
    monkeypatch.setitem(sys.modules, "dss.ray_dss.jobs.gpu", None)
    monkeypatch.setitem(sys.modules, "dss.ray_dss.stage_info", None)
    calls = []

    async def original(zone, config, job_id):
        calls.append((zone, config, job_id))
        return "created"

    service = SimpleNamespace(_create_job_v0=original)
    config = SimpleNamespace(training_config=None)
    with engine_admission.gateway_runtime_scope(service, tmp_path / "unused-hostfile"):
        assert await service._create_job_v0("zone", config, "child") == "created"
    assert calls == [("zone", config, "child")]
    assert service._create_job_v0 is original


@pytest.mark.asyncio
async def test_gateway_runtime_scope_loads_stage_info_only_for_admitted_training(monkeypatch, tmp_path):
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("worker slots=1\n")
    receipt = {"host": "worker", "boot_id": "boot", "python": sys.executable, "python_sha256": "a" * 64, "sources": {}}
    allocation = {
        "nodes": [{"host": "worker", "slots": 1}],
        "hostfile_sha256": hashlib.sha256(hostfile.read_bytes()).hexdigest(),
        "runtime_receipts": {"worker": receipt},
        "ssh": {"port": 22},
    }
    monkeypatch.setenv(engine_admission.RUNTIME_ADMISSION_ENV, json.dumps(allocation))
    payload = {"job_id": "child", "training_config": {"ds_config": {"bf16": {"enabled": True}}}}
    config = SimpleNamespace(job_id="child", num_gpus=1, training_config=SimpleNamespace(debug=None))
    stripped = []
    stage_info = ModuleType("dss.ray_dss.stage_info")

    def strip(request):
        stripped.append(request)
        return SimpleNamespace(to_engine_dict=lambda job_id: copy.deepcopy(payload))

    stage_info.strip_stage_info_from_job_config = strip
    monkeypatch.setitem(sys.modules, stage_info.__name__, stage_info)
    training = ModuleType("dss.ray_dss.jobs.gpu.training")
    training.sp = SimpleNamespace(sp_size_of=lambda config: 1)
    package = ModuleType("dss.ray_dss.jobs.gpu")
    package.training = training
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, training.__name__, training)
    requests = []

    def receipts(request, selected_hostfile, transport):
        requests.append(request)
        assert selected_hostfile == hostfile
        assert transport == allocation["ssh"]
        return allocation["runtime_receipts"]

    monkeypatch.setattr(engine_admission, "_allocation_runtime_receipts", receipts)
    calls = []

    async def original(zone, request, job_id):
        calls.append(request)
        return "created"

    service = SimpleNamespace(_create_job_v0=original)
    zone = SimpleNamespace(nodes=[SimpleNamespace(host="worker", gpu_ids=[0])], zone_id="zone")
    with engine_admission.gateway_runtime_scope(service, hostfile):
        assert await service._create_job_v0(zone, config, "child") == "created"
    assert stripped == calls == [config]
    assert requests == [{"payload": payload, "n_gpus": 1, "sp_size": 1}]
    admission = config.training_config.debug[engine_admission.RUNTIME_ADMISSION_KEY]
    assert admission["training_child_id"] == "child"
    assert admission["ranks"] == [("worker", 0)]
    assert service._create_job_v0 is original


@pytest.mark.parametrize("test_id", CHECKS)
@pytest.mark.parametrize("automatic", [False, True])
def test_onboarding_validation_child_retains_its_gateway_ownership(recipe, monkeypatch, tmp_path, test_id, automatic):
    config = recipe()
    before = untouched(config)
    monkeypatch.setattr(execution, "_resolve_configs", lambda *args, **kwargs: [config.path])
    monkeypatch.delenv("ARCTIC_CORTEX_HOST", raising=False)
    monkeypatch.delenv("ARCTIC_CORTEX_BASE_URL", raising=False)
    monkeypatch.delenv("ARCTIC_CORTEX_PAT", raising=False)
    arguments = ["onboard", "--config", str(config.path), "--test-id", test_id]
    if not automatic:
        arguments += ["--base-url", "http://127.0.0.1:8123"]
    args = execution.parse_args(arguments)
    args.output_dir = tmp_path / "validation"
    args.checkout_root = Path(__file__).resolve().parents[2]
    onboard._configure_fixture_target(args, config, [test_id])
    assert bool(getattr(args, "_automatic_cortex_gateway", False)) == automatic
    commands = []
    if automatic:
        monkeypatch.setattr(onboard, "_validation_gateway", forbidden)

    def child(command, **kwargs):
        commands.append(command)
        parsed = execution.parse_args(command[3:])
        assert parsed.test == [test_id]
        assert parsed.local_cortex_fixture
        assert parsed.base_url == (None if automatic else args.base_url)
        parsed = execution._configure_regression_target(parsed)
        assert bool(getattr(parsed, "_automatic_cortex_gateway", False)) == automatic
        output = Path(command[command.index("--out") + 1])
        (output / "report.json").write_text(json.dumps({"totals": {"pass": 2, "fail": 0}}))
        return SimpleNamespace(stdout=io.StringIO("controlled child result\n"), wait=lambda: 0)

    monkeypatch.setattr(onboard, "subprocess", SimpleNamespace(Popen=child, PIPE=-1, STDOUT=-2))
    assert onboard._run_final_regression(args, {"layers": 4})["status"] == "pass"
    assert len(commands) == 1
    assert ("--base-url" in commands[0]) is not automatic
    assert untouched(config) == before
