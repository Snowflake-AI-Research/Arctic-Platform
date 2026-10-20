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

"""Static correctness selftests: onboarding storage."""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import sys
from decimal import ROUND_CEILING
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers
from safetensors.torch import save_file

from arctic_platform.client import CortexConfig
from arctic_platform.correctness.checks import checkpoint_resume
from arctic_platform.correctness.harness import batches
from arctic_platform.correctness.harness import config as _fixed_tolerance_onboarding_SPEC
from arctic_platform.correctness.harness import dss_driver
from arctic_platform.correctness.harness import reference_records
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness import storage as workdir
from arctic_platform.correctness.harness.config import ArmDefinition
from arctic_platform.correctness.harness.config import TestSpec as ConfigSpec
from arctic_platform.correctness.harness.config import TestSpec as Spec
from arctic_platform.correctness.harness.config import config_checksum
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.registry import registered_tests
from arctic_platform.correctness.onboarding import calibration as RENDER_MODULE
from arctic_platform.correctness.onboarding import onboard
from arctic_platform.correctness.onboarding import onboard as MODULE
from arctic_platform.correctness.onboarding import synth_model
from arctic_platform.correctness.reference import hf_single_gpu
from arctic_platform.correctness.reference import optimizer_step

from .check_catalog_configuration import _applicable_tests_spec
from .helpers import _config_factory_native_config

# Data fast storage.


@pytest.fixture
def _data_fast_storage_fast_root(tmp_path, monkeypatch):
    root = tmp_path / "data-fast"
    root.mkdir()
    monkeypatch.setattr(workdir, "_DATA_FAST_ROOT", root)
    monkeypatch.setattr(workdir, "_DATA_FAST_TMPDIR", root / "tmp")
    for name in (workdir.CORRECTNESS_TMPDIR_ENV, "TMPDIR", "TEMP", "TMP"):
        monkeypatch.delenv(name, raising=False)
    return root


@pytest.mark.parametrize("as_string", [False, True])
def test_data_fast_storage_require_data_fast_path_returns_canonical_path_without_writes(
    _data_fast_storage_fast_root, as_string
):
    destination = _data_fast_storage_fast_root / "new" / ".." / "checkpoints"
    result = workdir.require_data_fast_path(str(destination) if as_string else destination)
    assert result == _data_fast_storage_fast_root / "checkpoints"
    assert not result.exists()
    assert list(_data_fast_storage_fast_root.iterdir()) == []


def test_data_fast_storage_require_data_fast_path_expands_home_and_relative_paths(
    _data_fast_storage_fast_root, monkeypatch
):
    monkeypatch.setenv("HOME", str(_data_fast_storage_fast_root))
    monkeypatch.chdir(_data_fast_storage_fast_root)
    assert workdir.require_data_fast_path("~/states") == _data_fast_storage_fast_root / "states"
    assert workdir.require_data_fast_path("states") == _data_fast_storage_fast_root / "states"
    assert not (_data_fast_storage_fast_root / "states").exists()


@pytest.mark.parametrize(
    "destination", ["/code/users/stas/bulk", "/code/other/bulk", "/tmp/bulk", "/data-fast-other/bulk"]
)
def test_data_fast_storage_require_data_fast_path_rejects_other_storage(_data_fast_storage_fast_root, destination):
    with pytest.raises(ValueError, match="checkpoint export.* /data-fast"):
        workdir.require_data_fast_path(destination, label="checkpoint export")
    assert list(_data_fast_storage_fast_root.iterdir()) == []


@pytest.mark.parametrize("kind", ["root", "sibling", "traversal", "relative"])
def test_data_fast_storage_require_data_fast_path_rejects_root_and_escaping_paths(
    _data_fast_storage_fast_root, monkeypatch, kind
):
    monkeypatch.chdir(_data_fast_storage_fast_root.parent)
    destination = {
        "root": _data_fast_storage_fast_root,
        "sibling": _data_fast_storage_fast_root.with_name("data-fast-other") / "bulk",
        "traversal": _data_fast_storage_fast_root / ".." / "outside" / "bulk",
        "relative": Path("outside/bulk"),
    }[kind]
    with pytest.raises(ValueError, match="/data-fast"):
        workdir.require_data_fast_path(destination)
    assert list(_data_fast_storage_fast_root.iterdir()) == []
    assert not (_data_fast_storage_fast_root.parent / "outside").exists()


def test_data_fast_storage_require_data_fast_path_rejects_symlink_escape(_data_fast_storage_fast_root):
    outside = _data_fast_storage_fast_root.parent / "outside"
    outside.mkdir()
    (_data_fast_storage_fast_root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="/data-fast"):
        workdir.require_data_fast_path(_data_fast_storage_fast_root / "escape" / "bulk")
    assert list(outside.iterdir()) == []


def test_data_fast_storage_require_data_fast_path_accepts_internal_symlink_as_canonical_path(
    _data_fast_storage_fast_root,
):
    target = _data_fast_storage_fast_root / "target"
    target.mkdir()
    (_data_fast_storage_fast_root / "alias").symlink_to(target, target_is_directory=True)
    assert workdir.require_data_fast_path(_data_fast_storage_fast_root / "alias" / "states") == target / "states"
    assert list(target.iterdir()) == []


def test_data_fast_storage_require_data_fast_path_rejects_symlinked_fast_root(
    _data_fast_storage_fast_root, monkeypatch
):
    outside = _data_fast_storage_fast_root.parent / "outside"
    outside.mkdir()
    alias = _data_fast_storage_fast_root.parent / "fake-fast"
    alias.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(workdir, "_DATA_FAST_ROOT", alias)
    with pytest.raises(ValueError, match="/data-fast"):
        workdir.require_data_fast_path(alias / "bulk")
    assert list(outside.iterdir()) == []


def test_data_fast_storage_require_data_fast_path_rejects_missing_fast_storage_without_creating_it(
    _data_fast_storage_fast_root, monkeypatch
):
    missing = _data_fast_storage_fast_root.parent / "missing-fast"
    monkeypatch.setattr(workdir, "_DATA_FAST_ROOT", missing)
    with pytest.raises(ValueError, match="existing /data-fast"):
        workdir.require_data_fast_path(missing / "states")
    assert not missing.exists()


@pytest.mark.parametrize("generic_tmpdir", ["/code/users/stas/tmp", "/tmp", "/home/other/tmp"])
def test_data_fast_storage_correctness_workdir_ignores_generic_temp_environment(
    _data_fast_storage_fast_root, monkeypatch, generic_tmpdir
):
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, generic_tmpdir)
    first = workdir.correctness_workdir("capture-")
    second = workdir.correctness_workdir("capture-")
    assert first.parent == second.parent == _data_fast_storage_fast_root / "tmp"
    assert first != second
    assert first.is_dir() and second.is_dir()
    assert first.name.startswith("capture-")


def test_data_fast_storage_correctness_workdir_accepts_only_fast_explicit_override(
    _data_fast_storage_fast_root, monkeypatch
):
    override = _data_fast_storage_fast_root / "captures"
    monkeypatch.setenv(workdir.CORRECTNESS_TMPDIR_ENV, str(override))
    created = workdir.correctness_workdir("capture-")
    assert created.parent == override
    assert created.is_dir()
    assert not (_data_fast_storage_fast_root / "tmp").exists()


@pytest.mark.parametrize("kind", ["code", "tmp", "root", "traversal", "symlink"])
def test_data_fast_storage_correctness_workdir_rejects_override_before_allocation(
    _data_fast_storage_fast_root, monkeypatch, kind
):
    outside = _data_fast_storage_fast_root.parent / "outside"
    outside.mkdir()
    (_data_fast_storage_fast_root / "escape").symlink_to(outside, target_is_directory=True)
    override = {
        "code": Path("/code/users/stas/bulk"),
        "tmp": Path("/tmp/bulk"),
        "root": _data_fast_storage_fast_root,
        "traversal": _data_fast_storage_fast_root / ".." / "outside" / "bulk",
        "symlink": _data_fast_storage_fast_root / "escape" / "bulk",
    }[kind]
    monkeypatch.setenv(workdir.CORRECTNESS_TMPDIR_ENV, str(override))

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("invalid storage must be rejected before workdir allocation")

    monkeypatch.setattr(workdir.tempfile, "mkdtemp", unexpected_allocation)
    with pytest.raises(ValueError, match=f"{workdir.CORRECTNESS_TMPDIR_ENV}.* /data-fast"):
        workdir.correctness_workdir("capture-")
    assert list(outside.iterdir()) == []
    assert not (_data_fast_storage_fast_root / "tmp").exists()


def test_data_fast_storage_correctness_workdir_missing_fast_storage_never_falls_back(
    _data_fast_storage_fast_root, monkeypatch
):
    missing = _data_fast_storage_fast_root.parent / "missing-fast"
    monkeypatch.setattr(workdir, "_DATA_FAST_ROOT", missing)
    monkeypatch.setattr(workdir, "_DATA_FAST_TMPDIR", missing / "tmp")
    monkeypatch.setenv("TMPDIR", str(_data_fast_storage_fast_root))
    with pytest.raises(ValueError, match="existing /data-fast"):
        workdir.correctness_workdir("capture-")
    assert not missing.exists()
    assert list(_data_fast_storage_fast_root.iterdir()) == []


@pytest.mark.parametrize("prefix", ["../capture-", "/code/users/stas/capture-", "nested/capture-"])
def test_data_fast_storage_correctness_workdir_rejects_prefix_traversal_before_allocation(
    _data_fast_storage_fast_root, prefix
):
    with pytest.raises(ValueError, match="prefix"):
        workdir.correctness_workdir(prefix)
    assert list(_data_fast_storage_fast_root.iterdir()) == []


# Fixed tolerance onboarding.


_fixed_tolerance_onboarding_LOCAL_TEST = "inference-checkpoint-loss"

_fixed_tolerance_onboarding_RESUME_TEST = "checkpoint-resume-loss"


def _fixed_tolerance_onboarding_write_config(tmp_path: Path) -> Path:
    path = tmp_path / "configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            _config_factory_native_config(
                {
                    "n_gpus": 4,
                    "sp_size": 4,
                    "max_seq_len": 32768,
                    "attn_implementation": "flash_attention_3",
                    "train_batch_size": 4,
                    "optimizer": {"lr": 2e-05},
                }
            )
        )
    )
    return path


def _fixed_tolerance_onboarding_write_spec(config_path: Path, spec_path: Path) -> None:
    """A reviewed spec in the state the local check needs: onboarded, with a tolerance and settings."""
    cfg = load_config(config_path)
    _fixed_tolerance_onboarding_SPEC.TestSpec(
        config_id=cfg.config_id,
        config_path=str(config_path),
        model=_fixed_tolerance_onboarding_SPEC.ModelSpec(
            source_checkpoint="Qwen/Qwen3-8B",
            num_hidden_layers=4,
            vision_depth=0,
            seed=1234,
            param_count=1000000,
            cache_path="/data-fast/base-models/synthetic/Qwen3-8B-4L",
            content_hash="0123456789abcdef",
            sized_against_gib=140.0,
        ),
        arms=[
            _fixed_tolerance_onboarding_SPEC.ArmSpec(
                name="gas1",
                global_batch_size=1,
                max_seq_len=10240,
                total_tokens=10240,
                active_tokens=10240,
                dss_microbatches=1,
                pad_fraction=0.0,
            )
        ],
        attn_implementations=["flash_attention_3"],
        applicable_tests=["single-step-grads", "single-step-optimizer"],
        config_checksum=config_checksum(cfg),
        test_tolerances={
            "single-step-grads": _fixed_tolerance_onboarding_SPEC.TestTolerance(
                absolute=0.004,
                calibration_runs=16,
                raw_max_same_tensor_range=0.0017,
                multiplier=2.0,
                computed_gate=0.0034,
                worst_tensor="embed_tokens.weight",
                status="calibrated",
            )
        },
        test_settings={"single-step-optimizer": {"reference_token_budget": 10240}},
    ).write(spec_path)


def _fixed_tolerance_onboarding_onboarding_args(config_path: Path, spec_path: Path, tmp_path: Path, test_id: str):
    output_dir = tmp_path / "onboarding-output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return argparse.Namespace(
        config=config_path, test_spec=spec_path, test_id=test_id, output_dir=output_dir, checkout_root=tmp_path
    )


def _fixed_tolerance_onboarding_prepared(tmp_path: Path, test_id: str = _fixed_tolerance_onboarding_LOCAL_TEST):
    config_path = _fixed_tolerance_onboarding_write_config(tmp_path)
    spec_path = tmp_path / "spec.json"
    _fixed_tolerance_onboarding_write_spec(config_path, spec_path)
    return (
        _fixed_tolerance_onboarding_onboarding_args(config_path, spec_path, tmp_path, test_id),
        load_config(config_path),
        spec_path,
    )


def _fixed_tolerance_onboarding_regression(status: str):

    def run(args, selected, **kwargs):
        return {
            "layers": selected["layers"],
            "status": status,
            "returncode": 0 if status == "pass" else 1,
            "log": "/tmp/run.log",
            "report": "/tmp/report.json" if status == "pass" else None,
        }

    return run


def _fixed_tolerance_onboarding_must_not_run(args, selected, **kwargs):
    raise AssertionError("the check was run for a config whose spec was already refused")


def test_fixed_tolerance_onboarding_unknown_test_id_is_still_refused() -> None:
    with pytest.raises(ValueError, match="onboarding does not support test id"):
        MODULE._onboarding_kind("no-such-check")


def test_fixed_tolerance_onboarding_a_check_with_no_reference_takes_the_fixed_tolerance_path() -> None:
    assert MODULE._onboarding_kind(_fixed_tolerance_onboarding_LOCAL_TEST) == MODULE.FIXED_TOLERANCE


def test_fixed_tolerance_onboarding_checkpoint_resume_measures_its_own_gate() -> None:
    assert MODULE._onboarding_kind(_fixed_tolerance_onboarding_RESUME_TEST) == _fixed_tolerance_onboarding_RESUME_TEST


def test_fixed_tolerance_onboarding_reference_checks_keep_their_own_paths() -> None:
    assert MODULE._onboarding_kind("single-step-grads") == "single-step-grads"
    assert MODULE._onboarding_kind("single-step-optimizer") == "single-step-optimizer"


def test_fixed_tolerance_onboarding_no_check_declares_it_needs_the_hosted_control_plane() -> None:
    tests = registered_tests()
    assert not tests[_fixed_tolerance_onboarding_RESUME_TEST].requires_hosted_control_plane
    assert not tests[_fixed_tolerance_onboarding_LOCAL_TEST].requires_hosted_control_plane


def test_fixed_tolerance_onboarding_passing_check_adds_one_id_and_changes_nothing_else(
    monkeypatch, tmp_path: Path
) -> None:
    args, cfg, spec_path = _fixed_tolerance_onboarding_prepared(tmp_path)
    before = json.loads(spec_path.read_text())
    monkeypatch.setattr(MODULE, "_run_final_regression", _fixed_tolerance_onboarding_regression("pass"))
    assert MODULE._onboard_fixed_tolerance_test(args, cfg) == 0
    after = json.loads(spec_path.read_text())
    assert set(after["applicable_tests"]) - set(before["applicable_tests"]) == {_fixed_tolerance_onboarding_LOCAL_TEST}
    assert after["applicable_tests"] == sorted(after["applicable_tests"])
    assert after["test_tolerances"] == before["test_tolerances"]
    assert after["test_settings"] == before["test_settings"]
    assert _fixed_tolerance_onboarding_LOCAL_TEST not in after["test_tolerances"]
    assert _fixed_tolerance_onboarding_LOCAL_TEST not in after["test_settings"]
    assert {key: value for key, value in after.items() if key != "applicable_tests"} == {
        key: value for key, value in before.items() if key != "applicable_tests"
    }


def test_fixed_tolerance_onboarding_failing_check_restores_the_previous_spec_bytes(
    monkeypatch, tmp_path: Path
) -> None:
    args, cfg, spec_path = _fixed_tolerance_onboarding_prepared(tmp_path)
    before = spec_path.read_bytes()
    monkeypatch.setattr(MODULE, "_run_final_regression", _fixed_tolerance_onboarding_regression("failed"))
    with pytest.raises(RuntimeError, match=f"{_fixed_tolerance_onboarding_LOCAL_TEST} failed"):
        MODULE._onboard_fixed_tolerance_test(args, cfg)
    assert spec_path.read_bytes() == before


def test_fixed_tolerance_onboarding_an_exception_during_the_run_restores_the_previous_spec_bytes(
    monkeypatch, tmp_path: Path
) -> None:
    args, cfg, spec_path = _fixed_tolerance_onboarding_prepared(tmp_path)
    before = spec_path.read_bytes()

    def explode(args, selected, **kwargs):
        raise RuntimeError("the gateway never came up")

    monkeypatch.setattr(MODULE, "_run_final_regression", explode)
    with pytest.raises(RuntimeError, match="the gateway never came up"):
        MODULE._onboard_fixed_tolerance_test(args, cfg)
    assert spec_path.read_bytes() == before


def test_fixed_tolerance_onboarding_a_config_with_no_spec_is_refused(monkeypatch, tmp_path: Path) -> None:
    args, cfg, spec_path = _fixed_tolerance_onboarding_prepared(tmp_path)
    spec_path.unlink()
    monkeypatch.setattr(MODULE, "_run_final_regression", _fixed_tolerance_onboarding_must_not_run)
    with pytest.raises(ValueError, match="requires an existing reviewed spec"):
        MODULE._onboard_fixed_tolerance_test(args, cfg)
    assert not spec_path.exists()


def test_fixed_tolerance_onboarding_a_stale_checksum_is_refused_rather_than_repaired(
    monkeypatch, tmp_path: Path
) -> None:
    args, cfg, spec_path = _fixed_tolerance_onboarding_prepared(tmp_path)
    before = spec_path.read_bytes()
    edited = json.loads(args.config.read_text())
    edited["max_seq_len"] = 16384
    args.config.write_text(json.dumps(edited))
    monkeypatch.setattr(MODULE, "_run_final_regression", _fixed_tolerance_onboarding_must_not_run)
    with pytest.raises(ValueError, match="hashes to"):
        MODULE._onboard_fixed_tolerance_test(args, load_config(args.config))
    assert spec_path.read_bytes() == before


def test_fixed_tolerance_onboarding_a_hosted_only_check_is_refused_and_names_the_hosted_command(
    monkeypatch, tmp_path: Path
) -> None:
    args, cfg, spec_path = _fixed_tolerance_onboarding_prepared(tmp_path, _fixed_tolerance_onboarding_RESUME_TEST)
    before = spec_path.read_bytes()
    monkeypatch.setattr(
        registered_tests()[_fixed_tolerance_onboarding_RESUME_TEST], "requires_hosted_control_plane", True
    )
    monkeypatch.setattr(MODULE, "_run_final_regression", _fixed_tolerance_onboarding_must_not_run)
    with pytest.raises(ValueError) as refusal:
        MODULE._onboard_fixed_tolerance_test(args, cfg)
    assert "arctic_platform.correctness hosted" in str(refusal.value)
    assert _fixed_tolerance_onboarding_RESUME_TEST in str(refusal.value)
    assert spec_path.read_bytes() == before


# Helper storage.


_helper_storage_ROOT = Path(__file__).resolve().parents[1]

_helper_storage_CONFIG = _helper_storage_ROOT / "configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config"

_helper_storage_SPEC = _helper_storage_ROOT / "specs/qwen3-8b-h200-train-sft-full-4gpus-64k.json"


@pytest.fixture(params=["/tmp", "/code", "/data-fast-shadow", "/data-fast", "symlink"])
def _helper_storage_unsafe_destination(request, tmp_path):
    if request.param != "symlink":
        return Path(request.param)
    alias = tmp_path / "escape"
    alias.symlink_to("/tmp", target_is_directory=True)
    return alias / "output"


@pytest.mark.parametrize("helper", ["gateway", "transport", "job", "weights", "adapter", "payload", "batch"])
def test_helper_storage_direct_helpers_reject_unsafe_writes_before_side_effects(
    tmp_path, monkeypatch, _helper_storage_unsafe_destination, helper
):

    def side_effect(*args, **kwargs):
        pytest.fail("unsafe destination reached a write or operational side effect")

    checkpoint = tmp_path / "source"
    source = checkpoint / "hf"
    source.mkdir(parents=True)
    (source / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    save_file({"weight": torch.ones(2)}, source / "adapter_model.safetensors")
    payload = {"training_config": {"n_gpus": 1}, "model_name": "read-only-model"}
    monkeypatch.setattr(Path, "mkdir", side_effect)
    monkeypatch.setattr(dss_driver.shutil, "rmtree", side_effect)
    monkeypatch.setattr(dss_driver.shutil, "copytree", side_effect)
    monkeypatch.setattr(dss_driver, "_gateway_runtime", side_effect)
    monkeypatch.setattr(dss_driver, "ArcticClient", side_effect)
    client = SimpleNamespace(save_checkpoint=side_effect)
    with pytest.raises(ValueError, match="/data-fast"):
        if helper == "gateway":
            with dss_driver.gateway(_helper_storage_unsafe_destination, 1):
                pytest.fail("unsafe gateway entered")
        elif helper == "transport":
            with dss_driver.GatewayTransport(
                _helper_storage_unsafe_destination, {"n_gpus": 1}, "read-only-model", 42, slots=1
            ):
                pytest.fail("unsafe transport entered")
        elif helper == "job":
            with dss_driver.running_job(dss_driver.GatewaySession(_helper_storage_unsafe_destination, 1), payload):
                pytest.fail("unsafe job entered")
        elif helper == "weights":
            dss_driver.save_weights_only_checkpoint(
                None, SimpleNamespace(client=client), _helper_storage_unsafe_destination
            )
        elif helper == "adapter":
            dss_driver.copy_checkpoint_peft_adapter(checkpoint, _helper_storage_unsafe_destination)
        elif helper == "payload":
            dss_driver.build_payload(
                {"n_gpus": 1}, "read-only-model", 42, optimizer_state_output_dir=_helper_storage_unsafe_destination
            )
        else:
            batches.save(SimpleNamespace(), _helper_storage_unsafe_destination)


def test_helper_storage_transport_revalidates_destination_before_enter_mkdir(tmp_path, monkeypatch):
    destination = tmp_path / "transport"
    transport = dss_driver.GatewayTransport(destination, {"n_gpus": 1}, "read-only-model", 42, slots=1)
    destination.symlink_to("/tmp", target_is_directory=True)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: pytest.fail("unsafe transport reached mkdir"))
    with pytest.raises(ValueError, match="/data-fast"):
        with transport:
            pytest.fail("unsafe transport entered")


def test_helper_storage_direct_training_job_rejects_checkpoint_child_escape_before_allocation(tmp_path, monkeypatch):
    (tmp_path / "checkpoints").symlink_to("/tmp", target_is_directory=True)
    monkeypatch.setattr(
        dss_driver, "_gateway_runtime", lambda *args, **kwargs: pytest.fail("unsafe checkpoint reached runtime")
    )
    payload = {"training_config": {"n_gpus": 1}, "model_name": "read-only-model"}
    with pytest.raises(ValueError, match="/data-fast"):
        with dss_driver.running_job(dss_driver.GatewaySession(tmp_path, 1), payload):
            pytest.fail("unsafe checkpoint reached a job")


@pytest.mark.parametrize("branch", ["reference", "dss"])
@pytest.mark.parametrize("order", ["batched", "interleaved"])
def test_helper_storage_runner_rejects_derived_optimizer_symlink_before_mkdir_or_payload(
    tmp_path, monkeypatch, capsys, branch, order
):
    optimizer = tmp_path / "optimizer"
    optimizer.mkdir()
    (optimizer / branch).symlink_to("/tmp", target_is_directory=True)
    original_mkdir = Path.mkdir
    references = []

    def mkdir(path, *args, **kwargs):
        if not path.resolve().is_relative_to(tmp_path):
            pytest.fail("derived optimizer path escaped before validation")
        return original_mkdir(path, *args, **kwargs)

    def reference(*args, **kwargs):
        references.append(kwargs["optimizer_output_dir"])
        if branch == "reference":
            pytest.fail("unsafe optimizer output reached the reference")
        return {"loss": 2.0, "grad_norms": {}, "microbatches": 1}

    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(runner, "run_reference", reference)
    monkeypatch.setattr(runner, "uses_mixer_packing", lambda *args: False)
    monkeypatch.setattr(reference_records, "model_identity", lambda path: {"sha256": "unit-model"})

    def prepared_record(ctx, identity):
        if branch == "reference":
            raise reference_records.ReferencePrerequisiteError("prepared HF reference missing: unit-record")
        return {"loss": 2.0, "grad_norms": {}, "microbatches": 1}

    monkeypatch.setattr(reference_records, "load_record", prepared_record)
    monkeypatch.setattr(runner, "gateway", lambda *args, **kwargs: contextlib.nullcontext(None))
    monkeypatch.setattr(
        runner, "running_job", lambda *args, **kwargs: pytest.fail("unsafe optimizer path reached job payload")
    )

    def execute():
        return runner.execute(
            load_config(_helper_storage_CONFIG),
            Spec.read(_helper_storage_SPEC),
            [ArmDefinition("gas1", 4, 32)],
            vocab_size=128,
            select=["single-step-optimizer"],
            workdir=tmp_path,
            order=order,
            prepare_references=branch == "reference",
        )

    if branch == "reference":
        with pytest.raises(ValueError, match="/data-fast"):
            execute()
    else:
        results = execute()
        assert len(results) == 1
        result = results[0]
        assert result.test_id == "single-step-optimizer"
        assert result.outcome is Outcome.FAIL and result.arm == "gas1" and (result.gpus == 4)
        assert "/data-fast" in result.reason
        assert result.summary == f"execution error: ValueError: {result.reason}"
        traceback = capsys.readouterr().err
        assert "Traceback (most recent call last):" in traceback
        assert "payload_for" in traceback and "require_data_fast_path" in traceback
        assert f"ValueError: {result.reason}" in traceback
    assert not references


def test_helper_storage_fast_batch_destination_is_canonical_and_read_inputs_remain_unrestricted(tmp_path, monkeypatch):
    fast_root = tmp_path / "fast"
    fast_root.mkdir()
    monkeypatch.setattr(workdir, "_DATA_FAST_ROOT", fast_root)
    batch = batches.build_batch("small", 1, 4, 128)
    output = fast_root / "unused" / ".." / "batch.pt"
    digest = batches.save(batch, output)
    assert len(digest) == 16
    assert (fast_root / "batch.pt").is_file()
    assert not (fast_root / "unused").exists()
    outside_source = tmp_path / "read-only-batch.pt"
    outside_source.write_bytes((fast_root / "batch.pt").read_bytes())
    loaded = batches.load(outside_source)
    assert torch.equal(loaded.input_ids, batch.input_ids)


def test_helper_storage_adapter_source_outside_fast_root_is_read_and_safe_destination_behavior_preserved(
    tmp_path, monkeypatch
):
    fast_root = tmp_path / "fast"
    fast_root.mkdir()
    checkpoint = tmp_path / "read-only-source"
    source = checkpoint / "hf"
    source.mkdir(parents=True)
    (source / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    save_file({"weight": torch.ones(2)}, source / "adapter_model.safetensors")
    destination = fast_root / "unused" / ".." / "adapter"
    existing = fast_root / "adapter"
    existing.mkdir()
    (existing / "stale").write_text("replace valid destination as before")
    monkeypatch.setattr(workdir, "_DATA_FAST_ROOT", fast_root)
    assert dss_driver.copy_checkpoint_peft_adapter(checkpoint, destination) == existing
    assert not (existing / "stale").exists()
    assert not (fast_root / "unused").exists()
    assert (existing / "adapter_model.safetensors").read_bytes() == (source / "adapter_model.safetensors").read_bytes()
    assert (source / "adapter_config.json").is_file()


def test_helper_storage_remote_training_config_without_local_writes_allows_read_only_workdir():
    backend = CortexConfig(base_url="http://unused")
    session = dss_driver.GatewaySession(Path("/code/read-only"), 1, backend)
    payload = dss_driver.build_payload({"n_gpus": 1}, "/code/read-only-model", 42)
    config = dss_driver._training_client_config(session, payload)
    assert config.training.checkpoint_path is None
    assert config.model_name == "/code/read-only-model"


def test_helper_storage_weights_save_keeps_public_framing_and_does_not_restrict_returned_source(tmp_path, monkeypatch):
    fast_root = tmp_path / "fast"
    fast_root.mkdir()
    source = tmp_path / "read-only-export"
    source.mkdir()
    (source / "config.json").write_text("{}")
    calls = []

    def save_checkpoint(**kwargs):
        calls.append(kwargs)
        return {"path": str(source), "hf_path": str(source)}

    monkeypatch.setattr(workdir, "_DATA_FAST_ROOT", fast_root)
    destination = fast_root / "unused" / ".." / "checkpoint"
    job = SimpleNamespace(client=SimpleNamespace(save_checkpoint=save_checkpoint))
    assert dss_driver.save_weights_only_checkpoint(None, job, destination) == str(source)
    assert calls == [{"path": str(fast_root / "checkpoint"), "export_hf": True}]
    assert not (fast_root / "checkpoint").exists()


# Hf reference onboarding.


_hf_reference_onboarding_CONFIG = Path(
    "arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config"
)


@pytest.fixture
def _hf_reference_onboarding_prepared_args(tmp_path, monkeypatch):
    parser = argparse.ArgumentParser()
    onboard.add_arguments(parser)
    args = parser.parse_args(["--config", str(_hf_reference_onboarding_CONFIG), "--prepare-references"])
    cfg = load_config(_hf_reference_onboarding_CONFIG)
    spec = _applicable_tests_spec(list(reference_records.CRITERIA))
    spec.config_id = cfg.config_id
    spec.config_checksum = config_checksum(cfg)
    spec.model.cache_path = str(tmp_path / "external-model")
    monkeypatch.setattr(ConfigSpec, "read", lambda path: spec)
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda path, **kwargs: SimpleNamespace(vocab_size=128)
    )
    return (args, spec)


@pytest.mark.parametrize("test_id", list(reference_records.CRITERIA))
def test_hf_reference_onboarding_preparation_dispatches_frozen_workload_without_target_or_sizing(
    _hf_reference_onboarding_prepared_args, monkeypatch, test_id
):
    args, spec = _hf_reference_onboarding_prepared_args
    args.test_id = test_id
    calls = []
    monkeypatch.setattr(runner, "execute", lambda *a, **k: calls.append((a, k)) or [])
    monkeypatch.setattr(onboard, "_run_probe", lambda *a: pytest.fail("preparation must not resize"))
    monkeypatch.setattr(onboard, "_write_initial_spec", lambda *a: pytest.fail("preparation must not rewrite spec"))
    assert onboard.run(args) == 0
    assert len(calls) == 1
    positional, keywords = calls[0]
    assert positional[1] is spec
    assert keywords["prepare_references"] is True
    assert keywords["select"] == [test_id]
    assert keywords["vocab_size"] == 128
    assert keywords["reference_calibration_runs"] == 16
    assert keywords["reference_calibration_workers"] == 1
    assert keywords["client_backend"].base_url == "http://127.0.0.1:8003"
    assert keywords["reference_records_dir"] == reference_records.DEFAULT_RECORDS_DIR


@pytest.mark.parametrize("field", ["config_checksum", "config_id"])
def test_hf_reference_onboarding_preparation_rejects_mismatched_frozen_spec(
    _hf_reference_onboarding_prepared_args, monkeypatch, field
):
    args, spec = _hf_reference_onboarding_prepared_args
    setattr(spec, field, "wrong")
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("mismatched workload executed"))
    with pytest.raises(ValueError, match="matching reviewed"):
        onboard.run(args)


@pytest.mark.parametrize("test_id", ["checkpoint-resume-loss", "inference-checkpoint-loss", "rl-weight-sync", "bad"])
def test_hf_reference_onboarding_preparation_rejects_live_reference_checks(
    _hf_reference_onboarding_prepared_args, monkeypatch, test_id
):
    args, _ = _hf_reference_onboarding_prepared_args
    args.test_id = test_id
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("normal reference prepared"))
    with pytest.raises(ValueError, match="three HF"):
        onboard.run(args)


def test_hf_reference_onboarding_preparation_rejects_inapplicable_check(
    _hf_reference_onboarding_prepared_args, monkeypatch
):
    args, spec = _hf_reference_onboarding_prepared_args
    spec.applicable_tests = []
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("inapplicable workload executed"))
    with pytest.raises(ValueError, match="not applicable"):
        onboard.run(args)


def test_hf_reference_onboarding_preparation_does_not_accept_model_sizing_override(
    _hf_reference_onboarding_prepared_args, monkeypatch
):
    args, _ = _hf_reference_onboarding_prepared_args
    args.probe_layers = 4
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("model override executed"))
    with pytest.raises(ValueError, match="model sizing"):
        onboard.run(args)


def test_hf_reference_onboarding_preparation_requires_git_record_destination(
    _hf_reference_onboarding_prepared_args, tmp_path, monkeypatch
):
    args, _ = _hf_reference_onboarding_prepared_args
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    monkeypatch.setattr(
        reference_records, "__file__", str(checkout / "arctic_platform/correctness/harness/reference_records.py")
    )
    args.reference_records_dir = tmp_path / "untracked-outside-checkout"
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("outside Git preparation executed"))
    with pytest.raises(reference_records.ReferencePrerequisiteError, match="Git checkout"):
        onboard.run(args)


def test_hf_reference_onboarding_full_model_preparation_is_only_for_e2e(
    _hf_reference_onboarding_prepared_args, monkeypatch
):
    args, _ = _hf_reference_onboarding_prepared_args
    args.local_gateway_full_model = True
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("wrong full-model workload executed"))
    with pytest.raises(ValueError, match="only for E2E"):
        onboard.run(args)


def test_hf_reference_onboarding_preparation_passes_explicit_calibration_and_full_e2e_options(
    _hf_reference_onboarding_prepared_args, monkeypatch
):
    args, _ = _hf_reference_onboarding_prepared_args
    args.test_id = "e2e-sft-train-validate"
    args.local_gateway_full_model = True
    args.reference_calibration_workers = 8
    calls = []
    monkeypatch.setattr(runner, "execute", lambda *a, **k: calls.append(k) or [])
    assert onboard.run(args) == 0
    assert calls[0]["local_gateway_full_model"] is True
    assert calls[0]["reference_calibration_workers"] == 8


def test_hf_reference_onboarding_preparation_preserves_combined_check_workload(
    _hf_reference_onboarding_prepared_args, monkeypatch
):
    args, _ = _hf_reference_onboarding_prepared_args
    args.reference_tests = ["single-step-grads", "single-step-optimizer"]
    calls = []
    monkeypatch.setattr(runner, "execute", lambda *a, **k: calls.append(k) or [])
    assert onboard.run(args) == 0
    assert calls[0]["select"] == args.reference_tests


def test_hf_reference_onboarding_combined_selection_cannot_prepare_live_reference(
    _hf_reference_onboarding_prepared_args, monkeypatch
):
    args, _ = _hf_reference_onboarding_prepared_args
    args.reference_tests = ["single-step-grads", "checkpoint-resume-loss"]
    monkeypatch.setattr(runner, "execute", lambda *a, **k: pytest.fail("live reference prepared"))
    with pytest.raises(ValueError, match="three HF"):
        onboard.run(args)


def test_hf_reference_onboarding_reference_selection_requires_preparation_mode(
    _hf_reference_onboarding_prepared_args, monkeypatch
):
    args, _ = _hf_reference_onboarding_prepared_args
    args.prepare_references = False
    args.reference_tests = ["single-step-grads", "single-step-optimizer"]
    monkeypatch.setattr(onboard, "_parent", lambda *a: pytest.fail("normal onboarding executed"))
    with pytest.raises(ValueError, match="requires --prepare-references"):
        onboard.run(args)


def test_hf_reference_onboarding_failed_reference_retains_complete_traceback_on_fast_storage(tmp_path, monkeypatch):
    output = tmp_path / "reference.json"
    stderr = "Traceback (most recent call last):\n" + "detail\n" * 1000 + "RuntimeError: actual failure\n"
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="actual stdout\n", stderr=stderr),
    )
    with pytest.raises(RuntimeError, match="full stderr:") as error:
        runner.run_reference(
            "external-model",
            tmp_path / "batch.pt",
            output,
            token_budget=64,
            ce_chunk=1,
            attn="sdpa",
            fp32_lm_head=False,
        )
    assert str(tmp_path / "reference.stderr.log") in str(error.value)
    assert (tmp_path / "reference.stderr.log").read_text() == stderr
    assert (tmp_path / "reference.stdout.log").read_text() == "actual stdout\n"


@pytest.mark.parametrize(
    "selection", [None, [], ["checkpoint-resume-loss"], ["single-step-grads", "rl-router-replay"]]
)
def test_hf_reference_onboarding_runner_preparation_cannot_execute_normal_references(selection, tmp_path, monkeypatch):
    cfg = load_config(_hf_reference_onboarding_CONFIG)
    monkeypatch.setattr(runner, "registered_tests", lambda: pytest.fail("invalid selection reached execution"))
    with pytest.raises(ValueError, match="explicit HF"):
        runner.execute(
            cfg,
            _applicable_tests_spec([]),
            [],
            vocab_size=128,
            prepare_references=True,
            select=selection,
            workdir=tmp_path,
        )


# Onboarding.


def _onboarding_render_config(tmp_path: Path, prime_rl: dict | None, peft_config: dict | None = None) -> str:
    config = tmp_path / "configs/model/h200/workload.config"
    config.parent.mkdir(parents=True)
    training = {"n_gpus": 8, "sp_size": 8, "attn_implementation": "flash_attention_3"}
    if prime_rl is not None:
        training["prime_rl"] = prime_rl
    if peft_config is not None:
        training["peft_config"] = peft_config
    config.write_text(json.dumps(_config_factory_native_config(training)))
    args = argparse.Namespace(
        config=config,
        source_checkpoint="source",
        layers=4,
        model_path=tmp_path / "model",
        output_json=tmp_path / "output.json",
        output_script=tmp_path / "driver.py",
        test_spec=tmp_path / "spec.json",
        test_id="single-step-grads",
        tokens=2048,
        reference_token_budget=65536,
    )
    rendered = RENDER_MODULE.render(args)
    compile(rendered, "<generated-reference-calibration>", "exec")
    return rendered


def test_onboarding_renderer_reads_nested_prime_rl_chunk_size(tmp_path: Path) -> None:
    rendered = _onboarding_render_config(
        tmp_path, {"fused_lm_head_token_chunk_size": 8192, "fused_cross_entropy": False}
    )
    assert "lm_head_token_chunk_size=8192" in rendered


def test_onboarding_renderer_preserves_runtime_default_without_chunking(tmp_path: Path) -> None:
    rendered = _onboarding_render_config(tmp_path, None)
    assert "lm_head_token_chunk_size=None" in rendered


def test_onboarding_renderer_uses_configured_fused_cross_entropy(tmp_path: Path) -> None:
    rendered = _onboarding_render_config(
        tmp_path, {"fused_lm_head_token_chunk_size": "disabled", "fused_cross_entropy": "liger"}
    )
    assert "fused_cross_entropy='liger'" in rendered


def test_onboarding_renderer_passes_peft_config_to_reference(tmp_path: Path) -> None:
    rendered = _onboarding_render_config(
        tmp_path, None, {"peft_type": "Lora", "r": 32, "target_modules": ["q_proj", "v_proj"]}
    )
    assert "peft_config={'peft_type': 'Lora', 'r': 32, 'target_modules': ['q_proj', 'v_proj']}" in rendered


def test_onboarding_renderer_gate_metadata_uses_the_computed_safety_factor(tmp_path: Path) -> None:
    rendered = _onboarding_render_config(tmp_path, None)
    tree = ast.parse(rendered)
    multiplier_values = [
        value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "multiplier"
    ]
    assert len(multiplier_values) == 2
    assert all((isinstance(value, ast.Name) and value.id == "GATE_SAFETY_FACTOR" for value in multiplier_values))
    assert "computed {GATE_SAFETY_FACTOR:g}x gate" in rendered
    assert "{GATE_SAFETY_FACTOR:g}x rounded upward" in rendered
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            (
                isinstance(target, ast.Name) and target.id in {"GATE_QUANTUM", "GATE_SAFETY_FACTOR"}
                for target in node.targets
            )
        )
        or (isinstance(node, ast.FunctionDef) and node.name == "_select_gate")
    ]
    namespace = {"Decimal": Decimal, "ROUND_CEILING": ROUND_CEILING}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "<calibration-gate>", "exec"), namespace)
    assert namespace["GATE_SAFETY_FACTOR"] == 4.0
    assert namespace["_select_gate"](0.01) == (0.04, 0.04)
    assert namespace["_select_gate"](0.0) == (0.0, 0.001)


def test_onboarding_model_shape_uses_first_prefix_containing_every_attention_type(monkeypatch) -> None:

    class TextConfig:
        num_hidden_layers = 8
        layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2

    class Config:
        text_config = TextConfig()

    import transformers

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: Config())
    assert MODULE._model_shape("unused") == (8, 4, 4)


def test_onboarding_model_shape_applies_four_layer_floor(monkeypatch) -> None:

    class TextConfig:
        num_hidden_layers = 8
        layer_types = ["full_attention"] * 8

    class Config:
        text_config = TextConfig()

    import transformers

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: Config())
    assert MODULE._model_shape("unused") == (8, 1, 4)


def test_onboarding_optimizer_reference_model_does_not_replace_test1_model(monkeypatch, tmp_path: Path) -> None:
    test1_path = tmp_path / "Qwen3.8-27B-36L"
    spec = SimpleNamespace(
        model=SimpleNamespace(source_checkpoint="Qwen/Qwen3.8-27B", cache_path=str(test1_path), num_hidden_layers=36)
    )
    monkeypatch.setattr(MODULE, "_model_shape", lambda unused: (64, 4, 4))
    model_path, layers = MODULE._optimizer_reference_model(spec, tmp_path)
    assert model_path == tmp_path / "Qwen3.8-27B-4L"
    assert layers == 4
    assert spec.model.cache_path == str(test1_path)
    assert spec.model.num_hidden_layers == 36


def test_onboarding_cuda_oom_classification_does_not_swallow_other_failures() -> None:
    assert MODULE._is_cuda_oom(RuntimeError("CUDA out of memory while allocating tensor"))
    assert not MODULE._is_cuda_oom(RuntimeError("shape mismatch"))


def test_onboarding_final_validation_requires_semantic_pass() -> None:
    report = {"totals": {"pass": 2, "fail": 0, "inapplicable": 0}}
    assert MODULE._validation_outcome(0, "Overall: 2 passed, 0 failed", report) == "pass"


def test_onboarding_final_validation_classifies_cuda_oom_for_retry() -> None:
    assert MODULE._validation_outcome(1, "torch.OutOfMemoryError: CUDA out of memory", None) == "oom"


def test_onboarding_final_validation_does_not_retry_correctness_failure() -> None:
    report = {"totals": {"pass": 1, "fail": 1, "inapplicable": 0}}
    assert MODULE._validation_outcome(1, "Overall: 1 passed, 1 failed", report) == "failed"


def test_onboarding_source_checkpoint_defaults_to_training_model_name() -> None:
    config = argparse.Namespace(model_name="Qwen/Qwen3-8B")
    assert MODULE._select_source_checkpoint(None, config) == "Qwen/Qwen3-8B"


def test_onboarding_source_checkpoint_override_wins() -> None:
    config = argparse.Namespace(model_name="Qwen/Qwen3-8B")
    assert MODULE._select_source_checkpoint("private/model", config) == "private/model"


# Onboarding storage.


@pytest.fixture
def _onboarding_storage_fast_root(tmp_path, monkeypatch):
    root = tmp_path / "data-fast"
    root.mkdir()
    monkeypatch.setattr(workdir, "_DATA_FAST_ROOT", root)
    monkeypatch.setattr(workdir, "_DATA_FAST_TMPDIR", root / "tmp")
    monkeypatch.delenv(workdir.CORRECTNESS_TMPDIR_ENV, raising=False)
    return root


@pytest.fixture(params=["code", "tmp", "root", "sibling", "traversal", "relative", "symlink"])
def _onboarding_storage_invalid_destination(request, _onboarding_storage_fast_root, monkeypatch):
    outside = _onboarding_storage_fast_root.parent / "outside"
    outside.mkdir()
    monkeypatch.chdir(_onboarding_storage_fast_root.parent)
    if request.param == "symlink":
        (_onboarding_storage_fast_root / "escape").symlink_to(outside, target_is_directory=True)
    return {
        "code": Path("/code/users/stas/onboarding-storage-test"),
        "tmp": Path("/tmp/onboarding-storage-test"),
        "root": _onboarding_storage_fast_root,
        "sibling": _onboarding_storage_fast_root.with_name("data-fast-other") / "model",
        "traversal": _onboarding_storage_fast_root / ".." / "outside" / "model",
        "relative": Path("outside/model"),
        "symlink": _onboarding_storage_fast_root / "escape" / "model",
    }[request.param]


def _onboarding_storage_unexpected(*args, **kwargs):
    pytest.fail("invalid storage must be rejected before allocation or model/config/copy work")


@pytest.mark.parametrize("operation", ["materialize", "materialize_pretrained", "copy_sidecars"])
def test_onboarding_storage_model_destination_rejected_before_config_or_sidecar_resolution(
    _onboarding_storage_fast_root, _onboarding_storage_invalid_destination, monkeypatch, operation
):
    monkeypatch.setattr(synth_model, "reduce_config", _onboarding_storage_unexpected)
    monkeypatch.setattr(synth_model, "sidecar_files", _onboarding_storage_unexpected)
    monkeypatch.setattr(synth_model, "count_parameters", _onboarding_storage_unexpected)
    with pytest.raises(ValueError, match="destination.* /data-fast"):
        if operation == "copy_sidecars":
            synth_model.copy_sidecars("not-a-local-source", _onboarding_storage_invalid_destination)
        else:
            getattr(synth_model, operation)("not-a-local-source", str(_onboarding_storage_invalid_destination), 4)
    assert list((_onboarding_storage_fast_root.parent / "outside").iterdir()) == []


@pytest.mark.parametrize("field", ["cache_root", "workdir"])
def test_onboarding_storage_build_spec_rejects_paths_before_allocation_or_model_work(
    _onboarding_storage_fast_root, _onboarding_storage_invalid_destination, monkeypatch, field
):
    monkeypatch.setattr(onboard, "correctness_workdir", _onboarding_storage_unexpected)
    monkeypatch.setattr(synth_model, "size_for_single_gpu", _onboarding_storage_unexpected)
    monkeypatch.setattr(synth_model, "materialize_pretrained", _onboarding_storage_unexpected)
    kwargs = {
        "cache_root": str(_onboarding_storage_fast_root / "cache"),
        "workdir": _onboarding_storage_fast_root / "work",
    }
    kwargs[field] = _onboarding_storage_invalid_destination
    with pytest.raises(ValueError, match="/data-fast"):
        onboard.build_spec(SimpleNamespace(), "unused", arm_defs=[], **kwargs)
    assert not (_onboarding_storage_fast_root / "work").exists()
    assert not (_onboarding_storage_fast_root / "cache").exists()


@pytest.mark.parametrize("entrypoint", ["run", "_parent", "_probe_worker"])
@pytest.mark.parametrize("field", ["cache_root", "output_dir"])
def test_onboarding_storage_onboarding_entrypoints_reject_storage_before_checkout_or_config(
    _onboarding_storage_fast_root, _onboarding_storage_invalid_destination, monkeypatch, entrypoint, field
):
    monkeypatch.setattr(onboard, "_checkout_root", _onboarding_storage_unexpected)
    args = argparse.Namespace(
        cache_root=_onboarding_storage_fast_root / "cache", output_dir=_onboarding_storage_fast_root / "work"
    )
    setattr(args, field, _onboarding_storage_invalid_destination)
    with pytest.raises(ValueError, match="/data-fast"):
        getattr(onboard, entrypoint)(args)
    assert not (_onboarding_storage_fast_root / "work").exists()


@pytest.mark.parametrize("field", ["output_dir", "optimizer_model_path"])
def test_onboarding_storage_optimizer_calibration_rejects_storage_before_config_batch_or_model(
    _onboarding_storage_fast_root, _onboarding_storage_invalid_destination, field
):
    args = argparse.Namespace(
        output_dir=_onboarding_storage_fast_root / "work", optimizer_model_path=_onboarding_storage_fast_root / "model"
    )
    setattr(args, field, _onboarding_storage_invalid_destination)
    with pytest.raises(ValueError, match="/data-fast"):
        onboard._run_optimizer_calibration(args)
    assert not (_onboarding_storage_fast_root / "work").exists()


@pytest.mark.parametrize("operation", ["materialize", "materialize_pretrained"])
def test_onboarding_storage_materializers_forward_canonical_fast_destination(
    _onboarding_storage_fast_root, tmp_path, monkeypatch, operation
):
    import transformers

    source = tmp_path / "source"
    source.mkdir()
    (source / "tokenizer_config.json").write_text("{}")
    target = _onboarding_storage_fast_root / "models"
    target.mkdir()
    (_onboarding_storage_fast_root / "alias").symlink_to(target, target_is_directory=True)
    destination = _onboarding_storage_fast_root / "alias" / "reduced"
    calls = []
    config = SimpleNamespace(architectures=["StorageTestModel"])

    class FakeModel:

        def __init__(self, cfg):
            assert cfg is config
            calls.append("model")

        @classmethod
        def from_pretrained(cls, value, **kwargs):
            assert value == str(source)
            return cls(kwargs["config"])

        def to(self, dtype):
            return self

        def save_pretrained(self, path, **kwargs):
            assert path == target / "reduced"
            calls.append(path)
            (path / "config.json").write_text("{}")

    monkeypatch.setattr(transformers, "StorageTestModel", FakeModel, raising=False)
    monkeypatch.setattr(synth_model, "reduce_config", lambda *args: (config, 8, 1))
    monkeypatch.setattr(synth_model, "count_parameters", lambda cfg: 1)
    result = getattr(synth_model, operation)(str(source), str(destination), 4)
    assert result.cache_path == str(target / "reduced")
    assert calls == ["model", target / "reduced"]
    assert (target / "reduced" / "tokenizer_config.json").read_text() == "{}"


def test_onboarding_storage_copy_sidecars_canonical_fast_destination_and_idempotency(
    _onboarding_storage_fast_root, tmp_path
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "tokenizer_config.json").write_text('{"version": 1}')
    target = _onboarding_storage_fast_root / "target"
    target.mkdir()
    (_onboarding_storage_fast_root / "alias").symlink_to(target, target_is_directory=True)
    for _ in range(2):
        copied = synth_model.copy_sidecars(str(source), _onboarding_storage_fast_root / "alias")
        assert copied == [target / "tokenizer_config.json"]
        assert copied[0].read_text() == '{"version": 1}'


def test_onboarding_storage_build_spec_accepts_fast_work_cache_and_separate_spec_output(
    _onboarding_storage_fast_root, tmp_path, monkeypatch
):
    import transformers

    from arctic_platform.correctness.harness import config as config_module
    from arctic_platform.correctness.harness.config import ModelSpec

    captured = []

    def materialize(source, destination, layers):
        captured.append(Path(destination))
        return ModelSpec(source, layers, 2, 0, 1, destination, "hash", 100.0)

    monkeypatch.setattr(synth_model, "materialize_pretrained", materialize)
    monkeypatch.setattr(config_module, "config_checksum", lambda cfg: "checksum")
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(vocab_size=8)
    )
    cfg = SimpleNamespace(attention_implementation="eager", config_id="test", path="input.config")
    spec = onboard.build_spec(
        cfg,
        "source/model",
        str(_onboarding_storage_fast_root / "cache"),
        [],
        num_layers=4,
        workdir=_onboarding_storage_fast_root / "work",
    )
    assert captured == [_onboarding_storage_fast_root / "cache" / "model-4L"]
    assert (_onboarding_storage_fast_root / "work").is_dir()
    separate_spec = tmp_path / "checkout" / "spec.json"
    spec.write(separate_spec)
    assert json.loads(separate_spec.read_text())["model"]["cache_path"] == str(captured[0])


@pytest.mark.parametrize("probe", [False, True])
def test_onboarding_storage_run_keeps_separate_small_spec_and_probe_reports_unrestricted(
    _onboarding_storage_fast_root, tmp_path, monkeypatch, probe
):
    args = argparse.Namespace(
        config=tmp_path / "checkout" / "input.config",
        cache_root=_onboarding_storage_fast_root / "cache",
        output_dir=_onboarding_storage_fast_root / "work",
        test_spec=tmp_path / "checkout" / "spec.json",
        probe_layers=4 if probe else None,
        probe_output=tmp_path / "checkout" / "probe.json",
    )
    monkeypatch.setattr(onboard, "_checkout_root", lambda config: tmp_path / "checkout")
    monkeypatch.setattr(onboard.sys, "path", list(onboard.sys.path))
    calls = []
    monkeypatch.setattr(onboard, "_parent", lambda value: calls.append(value) or 0)
    monkeypatch.setattr(onboard, "_probe_worker", lambda value: calls.append(value) or 0)
    assert onboard.run(args) == 0
    assert calls == [args]
    assert args.test_spec == tmp_path / "checkout" / "spec.json"
    assert args.probe_output == tmp_path / "checkout" / "probe.json"
    assert not args.output_dir.exists()


def test_onboarding_storage_optimizer_calibration_missing_fast_root_never_falls_back(
    _onboarding_storage_fast_root, monkeypatch
):
    _onboarding_storage_fast_root.rmdir()
    args = argparse.Namespace(
        output_dir=_onboarding_storage_fast_root / "work", optimizer_model_path=_onboarding_storage_fast_root / "model"
    )
    with pytest.raises(ValueError, match="existing /data-fast"):
        onboard._run_optimizer_calibration(args)
    assert not _onboarding_storage_fast_root.exists()


@pytest.mark.parametrize("fail_reference", [False, True])
def test_onboarding_storage_optimizer_calibration_allocates_fresh_root_and_cleans_only_owned_tree(
    _onboarding_storage_fast_root, tmp_path, monkeypatch, fail_reference
):
    import torch
    import transformers

    from arctic_platform.correctness.checks import optimizer_state
    from arctic_platform.correctness.harness import batches
    from arctic_platform.correctness.harness import config as config_module
    from arctic_platform.correctness.harness import config as spec_module
    from arctic_platform.correctness.harness import runner

    output = _onboarding_storage_fast_root / "output"
    output.mkdir()
    prior = _onboarding_storage_fast_root / "dss-correctness-calibration" / "test"
    prior.mkdir(parents=True)
    sentinel = prior / "sentinel"
    sentinel.write_text("preexisting")
    args = argparse.Namespace(
        output_dir=output,
        optimizer_model_path=_onboarding_storage_fast_root / "model",
        config=tmp_path / "input.config",
        test_spec=tmp_path / "checkout" / "spec.json",
        optimizer_layers=4,
        reference_token_budget=8,
    )
    config = SimpleNamespace(
        config_id="test",
        effective_training={},
        training={"optimizer": {"name": "AdamW"}},
        attention_implementation="eager",
        lm_head_token_chunk_size=None,
        optimizer_dtype="fp32",
    )
    spec = SimpleNamespace(
        config_checksum="checksum",
        arms=[SimpleNamespace(name="gas1", global_batch_size=1, max_seq_len=4)],
        test_tolerances={},
        test_settings={},
        applicable_tests=[],
    )

    def write_spec(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

    spec.write = write_spec
    monkeypatch.setattr(config_module, "load_config", lambda path: config)
    monkeypatch.setattr(config_module, "config_checksum", lambda cfg: "checksum")
    monkeypatch.setattr(spec_module.TestSpec, "read", lambda path: spec)
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(vocab_size=8)
    )
    monkeypatch.setattr(batches, "build_batch", lambda *args, **kwargs: object())
    monkeypatch.setattr(batches, "save", lambda batch, path: path.write_text("mock batch"))
    monkeypatch.setattr(runner, "materialize_dss_peft_adapter", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(onboard, "CALIBRATION_RUNS", 2)
    monkeypatch.setattr(optimizer_state, "max_pairwise_optimizer_delta", lambda *args, **kwargs: None)
    created = []
    deleted = []
    real_rmtree = onboard.shutil.rmtree

    def allocate(prefix):
        root = workdir.correctness_workdir(prefix)
        created.append(root)
        return root

    def cleanup(path, **kwargs):
        deleted.append(path)
        real_rmtree(path, **kwargs)

    def reference(*positional, **kwargs):
        destination = kwargs["optimizer_output_dir"]
        assert destination.parent.parent in created
        if fail_reference:
            raise RuntimeError("reference failed")
        destination.mkdir()
        manifest = destination / "manifest.json"
        manifest.write_text('{"parameters": [{"name": "mock"}]}')
        return {"optimizer_state_manifest": str(manifest), "loss": 1.0, "peak_gib": 0.0}

    monkeypatch.setattr(onboard, "correctness_workdir", allocate)
    monkeypatch.setattr(onboard.shutil, "rmtree", cleanup)
    monkeypatch.setattr(runner, "run_reference", reference)
    for _ in range(2):
        if fail_reference:
            with pytest.raises(RuntimeError, match="reference failed"):
                onboard._run_optimizer_calibration(args)
        else:
            report = onboard._run_optimizer_calibration(args)
            assert report == output / "optimizer-reference-repeatability.json"
            assert args.test_spec.exists()
    assert len(created) == 2 and created[0] != created[1]
    assert deleted == created
    assert all((path.parent == _onboarding_storage_fast_root / "tmp" and (not path.exists()) for path in created))
    assert sentinel.read_text() == "preexisting"


# Reference storage.


class _reference_storage_AllocationReached(RuntimeError):
    pass


@pytest.mark.parametrize("entrypoint", ["reference", "optimizer"])
@pytest.mark.parametrize(
    "destination", ["/code/optimizer", "/tmp/optimizer", "/data-fast-other/optimizer", "/data-fast"]
)
def test_reference_storage_reference_tensor_output_rejected_before_allocation(monkeypatch, entrypoint, destination):

    def allocate(*args, **kwargs):
        raise _reference_storage_AllocationReached("unsafe output reached model or optimizer allocation")

    monkeypatch.setattr(hf_single_gpu, "build_engine", allocate)
    monkeypatch.setattr(optimizer_step, "master_dtype", allocate)
    with pytest.raises(ValueError, match="/data-fast"):
        if entrypoint == "reference":
            hf_single_gpu.run("unused", None, optimizer_output_dir=destination)
        else:
            optimizer_step.run_adamw_step(
                None,
                {},
                {},
                learning_rate=0.001,
                gradient_clipping=None,
                optimizer_dtype="float32",
                output_dir=Path(destination),
            )


@pytest.mark.parametrize("entrypoint", ["reference", "optimizer"])
def test_reference_storage_reference_tensor_output_rejects_fast_symlink_escape(tmp_path, monkeypatch, entrypoint):
    alias = tmp_path / "escape"
    alias.symlink_to("/tmp", target_is_directory=True)
    test_reference_storage_reference_tensor_output_rejected_before_allocation(
        monkeypatch, entrypoint, str(alias / "optimizer")
    )


def test_reference_storage_reference_without_optimizer_output_still_reaches_engine(monkeypatch):

    def allocate(*args, **kwargs):
        raise _reference_storage_AllocationReached("reference engine reached")

    monkeypatch.setattr(hf_single_gpu, "build_engine", allocate)
    with pytest.raises(_reference_storage_AllocationReached, match="reference engine"):
        hf_single_gpu.run("unused", None)


def test_reference_storage_reference_fast_optimizer_output_still_reaches_engine(tmp_path, monkeypatch):

    def allocate(*args, **kwargs):
        raise _reference_storage_AllocationReached("reference engine reached")

    monkeypatch.setattr(hf_single_gpu, "build_engine", allocate)
    with pytest.raises(_reference_storage_AllocationReached, match="reference engine"):
        hf_single_gpu.run("unused", None, optimizer_output_dir=str(tmp_path / "optimizer"))
    assert not (tmp_path / "optimizer").exists()


# Checkpoint onboarding.


@pytest.fixture
def _checkpoint_onboarding_calibration(monkeypatch, tmp_path):
    args, config, spec_path = _fixed_tolerance_onboarding_prepared(tmp_path, _fixed_tolerance_onboarding_RESUME_TEST)
    original = spec_path.read_bytes()
    calls = []

    def model_config(path, **kwargs):
        calls.append(("model", path))
        return SimpleNamespace(text_config=SimpleNamespace(vocab_size=256))

    def measure(training, model, **kwargs):
        assert spec_path.read_bytes() == original
        calls.append(("measure", model, kwargs))
        return (0.1001, "gas4 loss at iteration 10")

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=model_config))
    )
    monkeypatch.setattr(checkpoint_resume, "measure_undeterministic_loss_range", measure)
    monkeypatch.setattr(onboard, "_run_final_regression", _fixed_tolerance_onboarding_regression("pass"))
    return (args, config, spec_path, original, calls)


def test_checkpoint_onboarding_missing_checkpoint_contract_preserves_existing_reviewed_fields(
    _checkpoint_onboarding_calibration,
):
    args, config, path, original, calls = _checkpoint_onboarding_calibration
    before = json.loads(original)
    assert onboard._onboard_checkpoint_resume(args, config) == 0
    after = json.loads(path.read_text())
    assert set(after["applicable_tests"]) - set(before["applicable_tests"]) == {
        _fixed_tolerance_onboarding_RESUME_TEST
    }
    assert after["applicable_tests"] == sorted(after["applicable_tests"])
    assert {key: value for key, value in after.items() if key not in {"applicable_tests", "test_tolerances"}} == {
        key: value for key, value in before.items() if key not in {"applicable_tests", "test_tolerances"}
    }
    assert {
        key: value for key, value in after["test_tolerances"].items() if key != _fixed_tolerance_onboarding_RESUME_TEST
    } == before["test_tolerances"]
    assert after["test_tolerances"][_fixed_tolerance_onboarding_RESUME_TEST]["absolute"] == 0.201
    assert [(call[0], call[1]) for call in calls] == [
        ("model", before["model"]["cache_path"]),
        ("measure", before["model"]["cache_path"]),
    ]
    assert calls[1][2]["slots"] == config.n_gpus
    assert calls[1][2]["vocab_size"] == 256
    assert calls[1][2]["workdir"] == args.output_dir / "checkpoint-resume-calibration"


@pytest.mark.parametrize("status", ["failed", "error"])
def test_checkpoint_onboarding_unsuccessful_validation_restores_exact_reviewed_bytes(
    _checkpoint_onboarding_calibration, monkeypatch, status
):
    args, config, path, original, _ = _checkpoint_onboarding_calibration
    monkeypatch.setattr(onboard, "_run_final_regression", _fixed_tolerance_onboarding_regression(status))
    with pytest.raises(RuntimeError, match=f"{_fixed_tolerance_onboarding_RESUME_TEST} {status}"):
        onboard._onboard_checkpoint_resume(args, config)
    assert path.read_bytes() == original


@pytest.mark.parametrize("exception", [RuntimeError("gateway unavailable"), KeyboardInterrupt()])
def test_checkpoint_onboarding_interrupted_validation_restores_exact_reviewed_bytes(
    _checkpoint_onboarding_calibration, monkeypatch, exception
):
    args, config, path, original, _ = _checkpoint_onboarding_calibration

    def fail(*args, **kwargs):
        assert _fixed_tolerance_onboarding_RESUME_TEST in json.loads(path.read_text())["applicable_tests"]
        raise exception

    monkeypatch.setattr(onboard, "_run_final_regression", fail)
    with pytest.raises(type(exception)):
        onboard._onboard_checkpoint_resume(args, config)
    assert path.read_bytes() == original


def test_checkpoint_onboarding_stale_config_is_rejected_before_measurement_or_spec_write(
    _checkpoint_onboarding_calibration,
):
    args, _, path, original, calls = _checkpoint_onboarding_calibration
    config = json.loads(args.config.read_text())
    config["max_seq_len"] = 16384
    args.config.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="hashes to"):
        onboard._onboard_checkpoint_resume(args, load_config(args.config))
    assert calls == []
    assert path.read_bytes() == original


def test_checkpoint_onboarding_calibration_failure_does_not_modify_reviewed_spec(
    _checkpoint_onboarding_calibration, monkeypatch
):
    args, config, path, original, _ = _checkpoint_onboarding_calibration

    def fail(*args, **kwargs):
        raise RuntimeError("calibration failed")

    monkeypatch.setattr(checkpoint_resume, "measure_undeterministic_loss_range", fail)
    with pytest.raises(RuntimeError, match="calibration failed"):
        onboard._onboard_checkpoint_resume(args, config)
    assert path.read_bytes() == original
