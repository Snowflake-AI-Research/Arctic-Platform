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
import hashlib
import io
import json
import logging
import sys
from decimal import ROUND_CEILING
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers
from safetensors.torch import load_file
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


@pytest.fixture(autouse=True)
def _synthetic_onboarding_environment(monkeypatch):
    # These calibration/spec controls use synthetic models; operator controls cover environment preflight.
    monkeypatch.setattr(onboard, "_onboarding_preflight", lambda *args: None)
    monkeypatch.setattr(_fixed_tolerance_onboarding_SPEC, "local_config_path", lambda path: Path(path).resolve())


@pytest.fixture
def _onboarding_checkout_roots(tmp_path, monkeypatch):
    library = tmp_path / "published-library"
    onboarding = library / "arctic_platform" / "correctness" / "onboarding"
    onboarding.mkdir(parents=True)
    harness = onboarding.parent / "harness"
    harness.mkdir()
    external = tmp_path / "operator-inputs"
    external.mkdir()
    monkeypatch.setattr(onboard, "__file__", str(onboarding / "onboard.py"))
    monkeypatch.setattr(reference_records, "__file__", str(harness / "reference_records.py"))
    return library, external


@pytest.mark.parametrize("suffix", [".config", ".json"])
@pytest.mark.parametrize("foreign_checkout", [False, True])
def test_onboarding_checkout_root_uses_library_for_external_configs(
    _onboarding_checkout_roots, suffix, foreign_checkout
):
    library, external = _onboarding_checkout_roots
    onboard.subprocess.run(["git", "init", "--quiet", str(library)], check=True)
    if foreign_checkout:
        onboard.subprocess.run(["git", "init", "--quiet", str(external)], check=True)
    config = external / ("train" + suffix)
    raw = b'{"synthetic":"external config, no model execution"}\n'
    config.write_bytes(raw)

    assert onboard._checkout_root(config) == library
    assert config.read_bytes() == raw
    assert reference_records.require_git_records_dir(library / "generated-records") == library / "generated-records"
    with pytest.raises(reference_records.ReferencePrerequisiteError, match="Git checkout"):
        reference_records.require_git_records_dir(external / "generated-records")
    assert not (library / "generated-records").exists()
    assert not (external / "generated-records").exists()


def test_onboarding_checkout_root_does_not_adopt_config_git_when_library_is_not_git(
    _onboarding_checkout_roots,
):
    library, external = _onboarding_checkout_roots
    onboard.subprocess.run(["git", "init", "--quiet", str(external)], check=True)
    config = external / "train.config"
    config.write_text('{"synthetic":"no published library Git checkout"}\n')

    with pytest.raises(onboard.subprocess.CalledProcessError):
        onboard._checkout_root(config)
    assert not (library / ".git").exists()


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


@pytest.mark.parametrize("local", [False, True])
def test_inference_forwarding_onboarding_subprocess_preserves_candidate_backend_and_scope(
    monkeypatch, tmp_path, local, capsys
):
    from arctic_platform.correctness.harness.execution import parse_args

    extra = (
        [
            "--backend",
            "cortex",
            "--base-url",
            "http://127.0.0.1:8003",
            "--database",
            "dev",
            "--schema",
            "dev",
            "--gpu-pool",
            "16",
            "--local-gateway-full-model",
            "--inference-sampling-topology",
            "16",
            "8",
            "2",
        ]
        if local
        else []
    )
    args = parse_args(
        [
            "onboard",
            "--config",
            str(tmp_path / "input.config"),
            "--test-id",
            "inference-checkpoint-loss",
            "--test-spec",
            str(tmp_path / "candidate.json"),
            "--output-dir",
            str(tmp_path / "out"),
            *extra,
        ]
    )
    args.checkout_root = tmp_path
    captured = []

    def process(command, **kwargs):
        captured.append(command)
        out = Path(command[command.index("--out") + 1])
        (out / "report.json").write_text(json.dumps({"totals": {"pass": 1, "fail": 0}}))
        return SimpleNamespace(stdout=io.StringIO("measured\n"), wait=lambda: 0)

    monkeypatch.setattr(onboard.subprocess, "Popen", process)
    result = onboard._run_final_regression(
        args, {"layers": 4, "model_description": "configured full model Qwen/Qwen3.8-Flash-Next"}
    )
    command = captured[0]
    assert command[command.index("--test-spec") + 1] == str(args.test_spec)
    assert command[command.index("--backend") + 1] == "cortex"
    if local:
        assert command[-4:] == ["--inference-sampling-topology", "16", "8", "2"]
        assert "--local-gateway-full-model" in command
        assert command[command.index("--base-url") + 1] == "http://127.0.0.1:8003"
    else:
        assert "--inference-sampling-topology" not in command
        assert "--local-gateway-full-model" not in command
    assert result["status"] == "pass"
    assert "configured full model" in capsys.readouterr().out


@pytest.mark.parametrize("status", ["pass", "failed", "exception"])
def test_inference_forwarding_fixed_gate_preserves_fixture_and_rollback(monkeypatch, tmp_path, status, capsys):
    args, cfg, spec_path = _fixed_tolerance_onboarding_prepared(tmp_path, "inference-checkpoint-loss")
    original = spec_path.read_bytes()
    before = json.loads(original)

    def validate(args, selected, **kwargs):
        assert selected["model_description"] == f"configured full model {cfg.model_name}"
        assert json.loads(spec_path.read_text())["model"] == before["model"]
        if status == "exception":
            raise RuntimeError("test-only unavailable export")
        return _fixed_tolerance_onboarding_regression(status)(args, selected, **kwargs)

    monkeypatch.setattr(onboard, "_run_final_regression", validate)
    if status != "pass":
        with pytest.raises(RuntimeError):
            onboard._onboard_fixed_tolerance_test(args, cfg)
        assert spec_path.read_bytes() == original
        assert not (args.output_dir / "onboarding-inference-checkpoint-loss.json").exists()
    else:
        assert onboard._onboard_fixed_tolerance_test(args, cfg) == 0
        after = json.loads(spec_path.read_text())
        assert {k: v for k, v in after.items() if k != "applicable_tests"} == {
            k: v for k, v in before.items() if k != "applicable_tests"
        }
        manifest = json.loads((args.output_dir / "onboarding-inference-checkpoint-loss.json").read_text())
        assert manifest["validated_model_scope"] == "configured-full-model"
        assert manifest["validated_model_name"] == cfg.model_name
        assert manifest["frozen_reference_num_hidden_layers"] == 4
        assert "inference-checkpoint-loss" not in after["test_tolerances"]
    assert "configured full model" in capsys.readouterr().out


def test_inference_forwarding_onboarding_local_route_never_admits_gpu_pool_from_author_host(monkeypatch, tmp_path):
    from arctic_platform.correctness.harness import config as config_module

    args, cfg, _ = _fixed_tolerance_onboarding_prepared(tmp_path, "inference-checkpoint-loss")
    args.backend = "cortex"
    args.base_url = "http://127.0.0.1:8003"
    args.database = args.schema = "dev"
    args.local_gateway_full_model = True
    args.source_checkpoint = None
    args.tokens = None
    args.reference_token_budget = None
    monkeypatch.setattr(onboard, "_validate_storage", lambda *a: None)
    monkeypatch.setattr(onboard, "_checkout_root", lambda *a: tmp_path)
    monkeypatch.setattr(
        config_module, "validate_against_host", lambda *a: pytest.fail("no author-host GPU validation")
    )
    monkeypatch.setattr(dss_driver, "available_gpus", lambda: pytest.fail("no author GPU probe"))
    monkeypatch.setattr(workdir, "ensure_no_interrupted_builds", lambda: pytest.fail("no onprem build probe"))
    calls = []
    monkeypatch.setattr(onboard, "_onboard_fixed_tolerance_test", lambda args, cfg: calls.append(cfg) or 0)
    assert onboard._parent(args) == 0
    assert calls == [cfg]


@pytest.mark.parametrize(
    ("test_id", "full_model", "pool", "base_url", "error"),
    [
        ("inference-checkpoint-loss", False, 32, "http://127.0.0.1:8003", None),
        ("inference-checkpoint-loss", False, None, "http://127.0.0.1:8003", None),
        ("inference-checkpoint-loss", False, 16, "http://127.0.0.1:8003", "required 32 GPUs"),
        ("inference-checkpoint-loss", True, 32, "http://127.0.0.1:8003", "reference-free RL inference"),
        ("inference-checkpoint-loss", False, 32, "https://example.com", "local"),
        ("rl-weight-sync", False, 32, "http://127.0.0.1:8003", "reference-free RL inference"),
        ("rl-router-replay", False, 32, "http://127.0.0.1:8003", "reference-free RL inference"),
        ("checkpoint-resume-loss", False, 32, "http://127.0.0.1:8003", "reference-free RL inference"),
    ],
)
def test_rl_local_onboarding_admission_preserves_checks_and_combined_gpu_width(
    monkeypatch, tmp_path, test_id, full_model, pool, base_url, error
):
    config_module = _fixed_tolerance_onboarding_SPEC
    config_path = (
        Path(__file__).resolve().parents[1] / "configs/glm5.3-flash-bf16/h200/train-rl-32gpus-sp16-256k.config"
    )
    cfg = load_config(config_path)
    before = cfg.native.model_dump()
    args = SimpleNamespace(
        config=config_path,
        test_id=test_id,
        backend="cortex",
        base_url=base_url,
        database="dev",
        schema="dev",
        local_gateway_full_model=full_model,
        gpu_pool=pool,
        source_checkpoint="/data-fast/base-models/glm-original",
        tokens=None,
        reference_token_budget=None,
        output_dir=tmp_path / "onboarding",
        test_spec=tmp_path / "candidate.json",
    )
    monkeypatch.setattr(onboard, "_validate_storage", lambda *a: None)
    monkeypatch.setattr(onboard, "_checkout_root", lambda *a: tmp_path)
    monkeypatch.setattr(config_module, "load_config", lambda *a: cfg)

    def unexpected(*args, **kwargs):
        pytest.fail("RL local admission must not probe GPUs, size, calibrate or start a model")

    monkeypatch.setattr(config_module, "validate_against_host", unexpected)
    monkeypatch.setattr(dss_driver, "available_gpus", unexpected)
    monkeypatch.setattr(workdir, "ensure_no_interrupted_builds", unexpected)
    monkeypatch.setattr(onboard, "_run_probe", unexpected)
    monkeypatch.setattr(onboard, "_render_and_run_calibration", unexpected)
    calls = []
    monkeypatch.setattr(onboard, "_onboard_fixed_tolerance_test", lambda args, cfg: calls.append(cfg) or 0)
    if error is None:
        assert onboard._parent(args) == 0
        assert calls == [cfg]
    else:
        with pytest.raises(ValueError, match=error):
            onboard._parent(args)
        assert calls == []
    assert cfg.native.model_dump() == before
    assert not args.test_spec.exists()


@pytest.mark.parametrize(
    ("test_id", "backend", "local_fixture", "full_model", "pool", "base_url", "error"),
    [
        ("single-step-grads", "cortex", True, False, 8, "http://127.0.0.1:8003", None),
        (
            "single-step-optimizer",
            "cortex",
            True,
            False,
            None,
            "http://127.0.0.1:8003",
            None,
        ),
        ("single-step-grads", "cortex", True, False, 4, "http://127.0.0.1:8003", "required 8 GPUs"),
        ("single-step-grads", "cortex", True, False, 8, "https://example.com", "local"),
        ("single-step-grads", "cortex", True, True, 8, "http://127.0.0.1:8003", "supervised Cortex"),
        ("single-step-grads", "onprem", True, False, 8, "http://127.0.0.1:8003", "supervised Cortex"),
        ("single-step-grads", "cortex", False, False, 8, "http://127.0.0.1:8003", None),
        ("inference-checkpoint-loss", "cortex", True, False, 8, "http://127.0.0.1:8003", "supervised Cortex"),
    ],
)
def test_supervised_local_fixture_onboarding_admits_only_explicit_native_route(
    monkeypatch, tmp_path, test_id, backend, local_fixture, full_model, pool, base_url, error
):
    config_path = (
        Path(__file__).resolve().parents[1] / "configs/glm5.3-flash/h200/train-sft-lora-8gpus-sp8-256k.config"
    )
    cfg = load_config(config_path)
    before = cfg.native.model_dump()
    args = SimpleNamespace(
        config=config_path,
        test_id=test_id,
        backend=backend,
        base_url=base_url,
        database="dev",
        schema="dev",
        local_cortex_fixture=local_fixture,
        local_gateway_full_model=full_model,
        gpu_pool=pool,
        source_checkpoint="/data-fast/base-models/glm-original",
        tokens=None,
        reference_token_budget=None,
        output_dir=tmp_path / "onboarding",
        test_spec=tmp_path / "candidate.json",
        reuse_sizing=None,
    )
    monkeypatch.setattr(onboard, "_validate_storage", lambda *a: None)
    monkeypatch.setattr(onboard, "_checkout_root", lambda *a: tmp_path)
    monkeypatch.setattr(_fixed_tolerance_onboarding_SPEC, "load_config", lambda *a: cfg)

    def unexpected(*args, **kwargs):
        pytest.fail("local fixture admission must not probe the author host or start a model")

    monkeypatch.setattr(_fixed_tolerance_onboarding_SPEC, "validate_against_host", unexpected)
    monkeypatch.setattr(dss_driver, "available_gpus", unexpected)
    monkeypatch.setattr(workdir, "ensure_no_interrupted_builds", unexpected)
    monkeypatch.setattr(onboard, "_render_and_run_calibration", unexpected)
    monkeypatch.setattr(onboard, "_run_final_regression", unexpected)
    reached = []

    def admitted(*args):
        reached.append(True)
        raise RuntimeError("admitted without preparing a model")

    monkeypatch.setattr(onboard, "_model_shape", admitted)
    monkeypatch.setattr(onboard, "_onboard_optimizer_test", lambda *a: reached.append(True) or 0)
    if error is not None:
        with pytest.raises(ValueError, match=error):
            onboard._parent(args)
        assert reached == []
    elif test_id == "single-step-optimizer":
        assert onboard._parent(args) == 0
        assert reached == [True]
    else:
        with pytest.raises(RuntimeError, match="admitted without preparing"):
            onboard._parent(args)
        assert reached == [True]
    assert cfg.native.model_dump() == before
    assert not args.test_spec.exists()


def test_supervised_local_fixture_onboarding_final_command_preserves_native_selector(monkeypatch, tmp_path):
    from arctic_platform.correctness.harness.execution import parse_args

    args = parse_args(
        [
            "onboard",
            "--config",
            str(tmp_path / "input.config"),
            "--test-id",
            "single-step-grads",
            "--test-spec",
            str(tmp_path / "candidate.json"),
            "--output-dir",
            str(tmp_path / "out"),
            "--backend",
            "cortex",
            "--base-url",
            "http://127.0.0.1:8003",
            "--gpu-pool",
            "8",
            "--local-cortex-fixture",
        ]
    )
    args.checkout_root = tmp_path
    captured = []

    def process(command, **kwargs):
        captured.append(command)
        output = Path(command[command.index("--out") + 1])
        (output / "report.json").write_text(json.dumps({"totals": {"pass": 1, "fail": 0}}))
        return SimpleNamespace(stdout=io.StringIO(), wait=lambda: 0)

    monkeypatch.setattr(onboard.subprocess, "Popen", process)
    assert onboard._run_final_regression(args, {"layers": 4})["status"] == "pass"
    command = captured[0]
    assert "--local-cortex-fixture" in command
    assert "--local-gateway-full-model" not in command
    parsed = parse_args(command[3:])
    assert parsed.local_cortex_fixture is True
    assert parsed.test == ["single-step-grads"]
    assert parsed.test_spec == args.test_spec
    assert parsed.backend == "cortex"
    assert parsed.gpu_pool == 8


@pytest.mark.parametrize("preparation", ["--prepare-references", "--probe-layers"])
def test_supervised_local_fixture_selector_rejects_preparation_only_modes(monkeypatch, tmp_path, preparation):
    parser = argparse.ArgumentParser()
    onboard.add_arguments(parser)
    extra = [preparation, "4"] if preparation == "--probe-layers" else [preparation]
    args = parser.parse_args(["--config", str(tmp_path / "input.config"), "--local-cortex-fixture", *extra])
    monkeypatch.setattr(onboard, "_validate_storage", lambda *a: pytest.fail("must refuse before storage checks"))
    with pytest.raises(ValueError, match="final target validation"):
        onboard.run(args)


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
    assert keywords["reference_calibration_runs"] == 5
    assert keywords["reference_calibration_workers"] == 1
    assert keywords["client_backend"].base_url == "http://127.0.0.1:1"
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


@pytest.mark.parametrize(
    "relative",
    [
        "glm5.3-flash-bf16/h200/train-sft-16gpus-sp16-256k.config",
        "glm5.3-flash/h200/train-sft-lora-8gpus-sp8-256k.config",
    ],
)
def test_reference_attention_sizing_and_rendered_calibration_use_glm_sdpa(
    _onboarding_storage_fast_root, monkeypatch, relative
):
    config_path = Path(__file__).resolve().parents[1] / "configs" / relative
    config = load_config(config_path)
    args = argparse.Namespace(
        config=config_path,
        cache_root=_onboarding_storage_fast_root / "cache",
        output_dir=_onboarding_storage_fast_root / "output",
        source_checkpoint="synthetic-source",
        probe_layers=4,
        probe_output=_onboarding_storage_fast_root / "sizing.json",
        tokens=8,
        reference_token_budget=8,
    )
    calls = []
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "synthetic")
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *args: None)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("synthetic sizing must not inspect a GPU"))
    monkeypatch.setattr(
        synth_model,
        "materialize_pretrained",
        lambda *args: SimpleNamespace(param_count=1, content_hash="synthetic"),
    )
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(vocab_size=16)
    )

    def reference(model, batch, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(peak_gib=1.0)

    monkeypatch.setattr(hf_single_gpu, "run", reference)
    assert onboard._probe_worker(args) == 0
    assert calls[0]["attn_implementation"] == "sdpa"
    assert config.attention_implementation == "sparse_mla"
    assert calls[0]["peft_config"] == config.effective_training.get("peft_config")
    rendered = RENDER_MODULE.render(
        argparse.Namespace(
            **{
                **vars(args),
                "layers": 4,
                "model_path": args.cache_root / "synthetic-source-4L",
                "output_json": args.output_dir / "calibration.json",
                "test_spec": args.output_dir / "synthetic-spec.json",
                "test_id": "single-step-grads",
            }
        )
    )
    compile(rendered, "<synthetic-glm-calibration>", "exec")
    assert "attn='sdpa'" in rendered
    assert "\"attention_implementation\": 'sdpa'" in rendered


def test_reference_attention_sizing_note_records_upstream_backend(_onboarding_storage_fast_root, monkeypatch):
    config_path = (
        Path(__file__).resolve().parents[1] / "configs/glm5.3-flash-bf16/h200/train-sft-16gpus-sp16-256k.config"
    )
    spec = _applicable_tests_spec([])
    spec.notes = []
    written = []
    spec.write = lambda path: written.append(path)
    monkeypatch.setattr(onboard, "build_spec", lambda *args, **kwargs: spec)
    args = argparse.Namespace(
        config=config_path,
        checkout_root=Path(__file__).resolve().parents[3],
        source_checkpoint="synthetic-source",
        cache_root=_onboarding_storage_fast_root / "cache",
        output_dir=_onboarding_storage_fast_root / "output",
        test_spec=_onboarding_storage_fast_root / "synthetic-spec.json",
        tokens=8,
    )
    onboard._write_initial_spec(args, {"layers": 4, "peak_gib": 1.0}, None)
    assert written == [args.test_spec]
    assert spec.notes == ["Single-GPU reference uses sdpa with 4 complete-cycle layers and peaks at 1.000 GiB."]


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


@pytest.fixture
def _rendered_calibration_workers(monkeypatch):
    calls = []
    monkeypatch.setattr(synth_model, "materialize_pretrained", lambda *args: None)
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(num_hidden_layers=4, vocab_size=128),
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(_fixed_tolerance_onboarding_SPEC, "hash_directory", lambda *args: "synthetic-model-hash")

    def reference(model, batch_path, output, **kwargs):
        batch = batches.load(batch_path)
        assert batch.input_ids.shape == (1, 2048)
        result = {"loss": 1.0, "grad_norms": {"model.weight": 2.0}, "peak_gib": 0.0}
        output.write_text(json.dumps(result))
        calls.append((batch_path, output))
        return result

    monkeypatch.setattr(runner, "run_reference", reference)
    return calls


@pytest.mark.parametrize("explicit", [False, True])
def test_onboarding_renderer_keeps_batch_tensors_separate_from_durable_diagnostics(
    tmp_path, _onboarding_storage_fast_root, _rendered_calibration_workers, explicit, capsys
):
    rendered = _onboarding_render_config(tmp_path, None)
    namespace = {"__name__": "synthetic_calibration"}
    exec(compile(rendered, "<generated-reference-calibration>", "exec"), namespace)
    diagnostics = tmp_path / "work" / "logs" / "calibration"
    namespace["OUTPUT"] = diagnostics / "reference-repeatability.json"
    scratch = _onboarding_storage_fast_root / "explicit-calibration"
    if explicit:
        namespace["WORK_DIR"] = str(scratch)
    namespace["SPEC_PATH"].write_text("{}")

    assert namespace["main"]() == 0
    lines = capsys.readouterr().out.splitlines()
    header = next(line for line in lines if line.split() == ["weight", "minimum", "maximum", "range"])
    assert header == f"{'weight':64} {'minimum':>13} {'maximum':>13} {'range':>13}"
    assert not any(line.split() == ["tensor", "minimum", "maximum", "range"] for line in lines)

    calls = _rendered_calibration_workers
    assert len(calls) == 5
    batch_paths = {batch_path for batch_path, output in calls}
    assert len(batch_paths) == 1
    batch_path = batch_paths.pop()
    assert batch_path.is_relative_to(_onboarding_storage_fast_root)
    if explicit:
        assert batch_path.parent == scratch
    else:
        assert batch_path.parent.parent == _onboarding_storage_fast_root / "tmp"
    assert all(output.parent == diagnostics / "calibration-runs" for batch_path, output in calls)
    assert not list(diagnostics.rglob("*.pt"))
    payload = json.loads(namespace["OUTPUT"].read_text())
    assert payload["configuration"]["runs"] == 5
    assert payload["tolerance"]["selected_gate"] == 0.001
    spec = json.loads(namespace["SPEC_PATH"].read_text())
    assert spec["test_tolerances"]["single-step-grads"]["calibration_runs"] == 5
    assert spec["test_tolerances"]["single-step-grads"]["worst_tensor"] == "model.weight"
    assert payload["tolerance"]["raw_max_same_tensor_range"] == 0.0


def test_onboarding_renderer_rejects_diagnostic_directory_for_tensor_scratch(
    tmp_path, _onboarding_storage_fast_root, _rendered_calibration_workers
):
    namespace = {"__name__": "synthetic_calibration"}
    exec(compile(_onboarding_render_config(tmp_path, None), "<generated-reference-calibration>", "exec"), namespace)
    namespace["WORK_DIR"] = str(tmp_path / "work" / "logs")
    with pytest.raises(ValueError, match="gradient calibration tensors"):
        namespace["main"]()
    assert _rendered_calibration_workers == []
    assert not list(tmp_path.rglob("*.pt"))


def test_onboarding_calibration_orchestration_uses_fast_scratch_and_durable_diagnostics(
    tmp_path, _onboarding_storage_fast_root, _rendered_calibration_workers, monkeypatch
):
    _onboarding_render_config(tmp_path, None)
    args = argparse.Namespace(
        config=tmp_path / "configs/model/h200/workload.config",
        source_checkpoint="synthetic-source",
        cache_root=_onboarding_storage_fast_root / "cache",
        output_dir=_onboarding_storage_fast_root / "scratch",
        _diagnostics_dir=tmp_path / "work" / "logs" / "calibration",
        test_spec=tmp_path / "spec.json",
        test_id="single-step-grads",
        tokens=2048,
        reference_token_budget=65536,
        checkout_root=tmp_path,
    )
    args.output_dir.mkdir()
    args.test_spec.write_text("{}")

    def execute(command, *, cwd, env):
        namespace = {"__name__": "synthetic_calibration"}
        exec(compile(Path(command[1]).read_text(), command[1], "exec"), namespace)
        assert Path(namespace["WORK_DIR"]) == args.output_dir / "calibration-runs"
        return namespace["main"]()

    monkeypatch.setattr(onboard, "_stream_subprocess", execute)
    result = onboard._render_and_run_calibration(args, {"layers": 4})
    assert result == args._diagnostics_dir / "reference-repeatability.json"
    assert result.is_file()
    assert (args.output_dir / "calibration-runs" / "batch.pt").is_file()
    assert len(_rendered_calibration_workers) == 5
    assert not list(args._diagnostics_dir.rglob("*.pt"))


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


@pytest.mark.parametrize(
    ("layer_types", "expected"),
    [
        ([], 0),
        (["full_attention"] * 8, 1),
        (["linear", "linear", "linear", "sparse"] * 2, 4),
        (["linear", "linear", "linear", "sparse"] * 11 + ["linear"], 4),
        (["linear", "linear", "linear", "sparse"] * 2 + ["other"], 9),
    ],
)
def test_onboarding_block_period_accepts_trailing_partial_block(layer_types, expected) -> None:
    assert synth_model.detect_block_period(layer_types) == expected


def test_onboarding_reduce_config_keeps_complete_block_from_partial_source(monkeypatch) -> None:
    text = SimpleNamespace(
        num_hidden_layers=45,
        layer_types=["linear", "linear", "linear", "sparse"] * 11 + ["linear"],
        layer_settings=list(range(45)),
        other_settings=["preserve"],
        hidden_size=6144,
    )
    config = SimpleNamespace(text_config=text)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)

    reduced, original, period = synth_model.reduce_config("unused", 4)

    assert reduced is config
    assert (original, period) == (45, 4)
    assert text.num_hidden_layers == 4
    assert text.layer_types == ["linear", "linear", "linear", "sparse"]
    assert text.layer_settings == [0, 1, 2, 3]
    assert text.other_settings == ["preserve"]
    assert text.hidden_size == 6144


@pytest.mark.parametrize("layers", [3, 5])
def test_onboarding_reduce_config_still_rejects_partial_selected_block(monkeypatch, layers) -> None:
    text = SimpleNamespace(
        num_hidden_layers=45,
        layer_types=["linear", "linear", "linear", "sparse"] * 11 + ["linear"],
    )
    config = SimpleNamespace(text_config=text)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    with pytest.raises(ValueError, match="multiple of the block period 4"):
        synth_model.reduce_config("unused", layers)
    assert text.num_hidden_layers == 45
    assert len(text.layer_types) == 45


def test_onboarding_model_shape_accepts_source_ending_mid_block(monkeypatch) -> None:
    config = SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=45,
            layer_types=["linear", "linear", "linear", "sparse"] * 11 + ["linear"],
        )
    )
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    assert MODULE._model_shape("unused") == (45, 4, 4)


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


@pytest.fixture
def _sizing_reuse_case(_onboarding_storage_fast_root, monkeypatch):
    root = _onboarding_storage_fast_root
    model = root / "models" / "original-4L"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"num_hidden_layers": 4}')
    (model / "model.safetensors").write_bytes(b"retained-weights")
    args = argparse.Namespace(
        config=root / "recipe.config",
        source_checkpoint=str(root / "original"),
        cache_root=root / "models",
        tokens=8192,
        reference_token_budget=65536,
        reuse_sizing=root / "reuse.json",
    )
    request = {
        "config": "original",
        "seed": 42,
        "tokens": 8192,
        "reference_token_budget": 65536,
        "layers": 4,
        "source_checkpoint": "original",
        "reference_sources": {"hf_single_gpu.py": "original"},
        "torch": "original",
        "transformers": "original",
    }
    selected = {
        "status": "fit",
        "layers": 4,
        "model_path": str(model),
        "model_hash": _fixed_tolerance_onboarding_SPEC.hash_directory(model),
        "param_count": 17,
        "peak_gib": 28.34,
    }
    original = root / "sizing.json"
    original.write_text(json.dumps(selected))
    receipt = {
        "schema": "correctness-sizing-reuse-v1",
        "request": request,
        "sizing_receipt": {"path": str(original), "sha256": hashlib.sha256(original.read_bytes()).hexdigest()},
        "model_identity": reference_records.model_identity(model),
    }
    monkeypatch.setattr(onboard, "_sizing_request", lambda *unused: request.copy())
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(num_hidden_layers=4)
    )
    monkeypatch.setattr(synth_model, "count_parameters", lambda unused: 17)

    def seal():
        original.write_text(json.dumps(selected))
        receipt["sizing_receipt"]["sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
        args.reuse_sizing.write_text(json.dumps(receipt))
        args.reuse_sizing_sha256 = hashlib.sha256(args.reuse_sizing.read_bytes()).hexdigest()

    seal()
    return args, request, selected, receipt, model, original, seal


def test_sizing_reuse_admits_retained_fit_without_sizing_execution(_sizing_reuse_case, monkeypatch):
    args, _, selected, _, _, _, _ = _sizing_reuse_case
    monkeypatch.setattr(onboard, "_run_probe", lambda *a: pytest.fail("sizing replay"))
    monkeypatch.setattr(synth_model, "materialize_pretrained", lambda *a: pytest.fail("model replay"))
    assert onboard._reuse_sizing(args, 4) == selected


@pytest.mark.parametrize(
    "field",
    [
        "config",
        "seed",
        "tokens",
        "reference_token_budget",
        "layers",
        "source_checkpoint",
        "reference_sources",
        "torch",
        "transformers",
    ],
)
def test_sizing_reuse_rejects_mismatched_workload(_sizing_reuse_case, field):
    args, _, _, receipt, _, _, seal = _sizing_reuse_case
    receipt["request"] = {**receipt["request"], field: "changed"}
    seal()
    with pytest.raises(ValueError, match="workload/source/runtime mismatch"):
        onboard._reuse_sizing(args, 4)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("status", "oom", "successful fit"),
        ("status", "error", "successful fit"),
        ("layers", 8, "original depth"),
        ("layers", True, "successful fit"),
        ("peak_gib", float("nan"), "finite positive"),
        ("peak_gib", float("inf"), "finite positive"),
        ("peak_gib", 0, "finite positive"),
        ("param_count", 19, "parameter count"),
        ("model_hash", "different", "model identity"),
        ("model_path", "other-model", "model path"),
    ],
)
def test_sizing_reuse_rejects_failed_or_mismatched_result(_sizing_reuse_case, field, value, message):
    args, _, selected, _, _, _, seal = _sizing_reuse_case
    selected[field] = str(args.cache_root.parent / value) if field == "model_path" else value
    seal()
    with pytest.raises(ValueError, match=message):
        onboard._reuse_sizing(args, 4)


@pytest.mark.parametrize("artifact", ["reuse", "original"])
def test_sizing_reuse_rejects_unsealed_bytes(_sizing_reuse_case, artifact):
    args, _, _, _, _, original, _ = _sizing_reuse_case
    path = args.reuse_sizing if artifact == "reuse" else original
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        onboard._reuse_sizing(args, 4)


def test_sizing_reuse_rejects_same_size_model_change(_sizing_reuse_case):
    args, _, _, _, model, _, _ = _sizing_reuse_case
    (model / "model.safetensors").write_bytes(b"changed--weights")
    with pytest.raises(ValueError, match="contents changed"):
        onboard._reuse_sizing(args, 4)


def test_sizing_reuse_rejects_missing_weights(_sizing_reuse_case):
    args, _, _, _, model, _, _ = _sizing_reuse_case
    (model / "model.safetensors").rename(model / "model.backup")
    with pytest.raises(ValueError, match="retained model weights"):
        onboard._reuse_sizing(args, 4)


def test_sizing_reuse_rejects_wrong_model_depth(_sizing_reuse_case, monkeypatch):
    args, _, _, _, _, _, _ = _sizing_reuse_case
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(num_hidden_layers=8)
    )
    with pytest.raises(ValueError, match="model depth"):
        onboard._reuse_sizing(args, 4)


def test_sizing_reuse_request_binds_actual_recipe_and_reference_sources(_sizing_reuse_case, monkeypatch):
    args, _, _, _, _, _, _ = _sizing_reuse_case
    monkeypatch.undo()
    args.config.write_text("original recipe")
    Path(args.source_checkpoint).mkdir()
    (Path(args.source_checkpoint) / "config.json").write_text("original source config")
    monkeypatch.setattr(
        _fixed_tolerance_onboarding_SPEC,
        "load_config",
        lambda unused: SimpleNamespace(reference_attention_implementation="sdpa"),
    )
    request = onboard._sizing_request(args, 4)
    assert request["config_sha256"] == hashlib.sha256(b"original recipe").hexdigest()
    assert request["source_config_sha256"] == hashlib.sha256(b"original source config").hexdigest()
    assert request["source_checkpoint"] == args.source_checkpoint
    assert request["layers"] == 4
    assert request["seed"] == batches.SEED
    assert request["tokens"] == 8192
    assert request["reference_token_budget"] == 65536
    assert request["reference_attention_implementation"] == "sdpa"
    assert request["torch"] == str(torch.__version__)
    assert request["transformers"] == str(transformers.__version__)
    assert request["interpreter"] == sys.executable
    assert len(request["probe_source_sha256"]) == 64
    assert (
        request["reference_sources"]["reference/hf_single_gpu.py"]
        == reference_records.file_identity(Path(hf_single_gpu.__file__))["sha256"]
    )


@pytest.mark.parametrize("rejected", [False, True])
def test_sizing_reuse_parent_continues_only_admitted_sizing(_sizing_reuse_case, monkeypatch, rejected):
    args, _, selected, receipt, _, _, seal = _sizing_reuse_case
    args.output_dir = args.cache_root.parent / "continuation"
    args.test_spec = args.cache_root.parent / "spec.json"
    args.test_id = "single-step-grads"
    args.config.write_text("synthetic")
    config = SimpleNamespace(
        config_id="synthetic",
        is_rl=False,
        n_gpus=8,
        max_seq_len=262144,
        reference_attention_implementation="sdpa",
    )
    monkeypatch.setattr(_fixed_tolerance_onboarding_SPEC, "load_config", lambda *a: config)
    monkeypatch.setattr(_fixed_tolerance_onboarding_SPEC, "validate_against_host", lambda *a: None)
    monkeypatch.setattr(dss_driver, "available_gpus", lambda: 8)
    monkeypatch.setattr(workdir, "ensure_no_interrupted_builds", lambda: None)
    monkeypatch.setattr(onboard, "_checkout_root", lambda unused: args.cache_root.parent)
    monkeypatch.setattr(onboard, "_model_shape", lambda unused: (45, 4, 4))
    monkeypatch.setattr(onboard, "_run_probe", lambda *a: pytest.fail("sizing replay"))
    calls = []
    monkeypatch.setattr(onboard, "_write_initial_spec", lambda *a: calls.append("spec"))

    def calibrate(unused, admitted):
        assert admitted == selected
        calls.append("calibration")
        path = args.output_dir / "calibration.json"
        path.write_text('{"tolerance": {"selected_gate": 0.005}}')
        return path

    monkeypatch.setattr(onboard, "_render_and_run_calibration", calibrate)
    monkeypatch.setattr(
        onboard,
        "_run_final_regression",
        lambda *a: calls.append("validation") or {"status": "pass", "report": "synthetic-report"},
    )
    if rejected:
        receipt["request"] = {"stale": True}
        seal()
        with pytest.raises(ValueError, match="workload/source/runtime mismatch"):
            onboard._parent(args)
        assert calls == []
        assert not args.test_spec.exists()
    else:
        assert onboard._parent(args) == 0
        assert calls == ["spec", "calibration", "validation"]
        manifest = json.loads((args.output_dir / "onboarding.json").read_text())
        assert manifest["sizing_observations"] == [selected]
        assert manifest["reused_sizing"]["sha256"] == args.reuse_sizing_sha256


@pytest.mark.parametrize(
    "options",
    [
        ["--reuse-sizing", "receipt.json"],
        ["--reuse-sizing-sha256", "a" * 64],
        ["--reuse-sizing", "receipt.json", "--reuse-sizing-sha256", "a" * 64, "--prepare-references"],
        ["--reuse-sizing", "receipt.json", "--reuse-sizing-sha256", "a" * 64, "--test-id", "single-step-optimizer"],
        ["--reuse-sizing", "receipt.json", "--reuse-sizing-sha256", "a" * 64, "--probe-layers", "4"],
    ],
)
def test_sizing_reuse_cli_rejects_incomplete_or_wrong_mode(options, monkeypatch):
    parser = argparse.ArgumentParser()
    onboard.add_arguments(parser)
    args = parser.parse_args(["--config", "unused.config", *options])
    monkeypatch.setattr(onboard, "_validate_storage", lambda *a: pytest.fail("wrong mode reached admission"))
    with pytest.raises(ValueError, match="supplied together|only supported for gradient"):
        onboard.run(args)


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
        cache_root=_onboarding_storage_fast_root / "cache",
        output_dir=_onboarding_storage_fast_root / "work",
        test_id="single-step-grads",
        probe_layers=None,
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


@pytest.mark.parametrize(
    "generation_options",
    [
        {"do_sample": False, "top_p": 0.95, "temperature": 0.7, "max_new_tokens": 8},
        {"do_sample": True, "top_p": 0.95, "temperature": 0.7, "max_new_tokens": 8},
        {"do_sample": False, "eos_token_id": [1, 2], "pad_token_id": 0, "max_new_tokens": 8},
    ],
    ids=["legacy-greedy-defaults", "sampling", "greedy"],
)
def test_onboarding_pretrained_generation_defaults_round_trip(
    _onboarding_storage_fast_root, tmp_path, generation_options
) -> None:
    config = transformers.Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
    )
    model = transformers.Qwen3ForCausalLM(config).to(torch.bfloat16)
    source = tmp_path / "source"
    model.save_pretrained(source)
    expected = {
        name: tensor.clone() for name, tensor in model.state_dict().items() if not name.startswith("model.layers.1.")
    }
    (source / "generation_config.json").write_text(json.dumps(generation_options))
    (source / "tokenizer_config.json").write_text("{}")
    destination = _onboarding_storage_fast_root / "reduced"

    synth_model.materialize_pretrained(str(source), str(destination), 1)

    saved_options = transformers.GenerationConfig.from_dict(
        json.loads((destination / "generation_config.json").read_text())
    )
    reopened = transformers.Qwen3ForCausalLM.from_pretrained(destination, dtype=torch.bfloat16)
    for name, value in generation_options.items():
        assert getattr(saved_options, name) == value
        assert getattr(reopened.generation_config, name) == value
    saved = load_file(destination / "model.safetensors")
    assert saved.keys() == expected.keys()
    for name, tensor in saved.items():
        torch.testing.assert_close(tensor, expected[name], atol=0, rtol=0, msg=name)


def test_onboarding_pretrained_generation_defaults_restore_after_save_failure(tmp_path) -> None:
    generation = transformers.GenerationConfig(do_sample=False, top_p=0.95)

    class FailingModel:
        generation_config = generation

        def save_pretrained(self, path, **kwargs):
            assert self.generation_config is not generation
            raise OSError("disk full")

    model = FailingModel()
    with pytest.raises(OSError, match="disk full"):
        synth_model._save_reduced_pretrained(model, tmp_path)
    assert model.generation_config is generation
    assert generation.do_sample is False
    assert generation.top_p == 0.95
    assert not (tmp_path / "generation_config.json").exists()


def test_onboarding_pretrained_generation_defaults_reject_invalid_values_before_save(tmp_path) -> None:
    generation = transformers.GenerationConfig()
    generation.max_new_tokens = 0
    model = SimpleNamespace(generation_config=generation, save_pretrained=_onboarding_storage_unexpected)

    with pytest.raises(ValueError, match="max_new_tokens"):
        synth_model._save_reduced_pretrained(model, tmp_path)
    assert model.generation_config is generation
    assert not (tmp_path / "generation_config.json").exists()


def test_onboarding_pretrained_fp8_materialization_saves_dequantized_weights(
    _onboarding_storage_fast_root, tmp_path
) -> None:
    source = tmp_path / "fp8-source"
    source.mkdir()
    config = transformers.Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
    )
    config.architectures = ["Qwen3ForCausalLM"]
    model = transformers.Qwen3ForCausalLM(config).to(torch.bfloat16)
    state = model.state_dict()
    expected = {}
    for name, tensor in list(state.items()):
        if tensor.ndim != 2 or name in ("model.embed_tokens.weight", "lm_head.weight"):
            continue
        scales = torch.full((tensor.shape[0] // 8, tensor.shape[1] // 8), 0.25)
        scales[0, 0] = 0.5
        state[name] = torch.full_like(tensor, 2, dtype=torch.float8_e4m3fn)
        state[name.removesuffix(".weight") + ".weight_scale_inv"] = scales
        expected[name] = (2 * scales.repeat_interleave(8, 0).repeat_interleave(8, 1)).to(torch.bfloat16)
    config.quantization_config = {"quant_method": "fp8", "weight_block_size": [8, 8]}
    config.save_pretrained(source)
    save_file(state, source / "model.safetensors", metadata={"format": "pt"})
    (source / "tokenizer_config.json").write_text("{}")
    destination = _onboarding_storage_fast_root / "reduced"

    result = synth_model.materialize_pretrained(str(source), str(destination), 1)
    saved_config = json.loads((destination / "config.json").read_text())
    assert "quantization_config" not in saved_config
    assert saved_config["num_hidden_layers"] == 1
    assert result.source_checkpoint == str(source)
    reopened = transformers.Qwen3ForCausalLM.from_pretrained(destination, dtype=torch.bfloat16)
    assert not getattr(reopened, "is_quantized", False)
    saved = reopened.state_dict()
    assert not any(name.endswith("weight_scale_inv") for name in saved)
    for name, tensor in saved.items():
        assert tensor.dtype == torch.bfloat16
        if name in expected:
            torch.testing.assert_close(tensor, expected[name], atol=0, rtol=0)
        else:
            torch.testing.assert_close(tensor, state[name], atol=0, rtol=0)
    assert not any(name.startswith("model.layers.1.") for name in saved)


def test_onboarding_pretrained_glm_fp8_preserves_native_weight_layout(_onboarding_storage_fast_root, tmp_path) -> None:
    from transformers.core_model_loading import revert_weight_conversion

    from arctic_platform.model.implementations.glm53.converting_glm5_next import convert_hf_layer_to_prime

    text = transformers.Glm5NextTextConfig(
        vocab_size=16,
        pad_token_id=0,
        hidden_size=16,
        intermediate_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=2,
        num_experts_per_tok=1,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=0,
        v_head_dim=8,
        index_n_heads=2,
        index_head_dim=8,
        linear_num_heads=2,
        linear_head_dim=8,
        hc_mult=2,
        layer_types=["linear_attention"] * 2,
        mlp_layer_types=["sparse"] * 2,
    )
    vision = transformers.Glm5NextVisionConfig(
        depth=1,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        out_hidden_size=16,
        projection_intermediate_size=16,
        image_size=14,
        patch_size=14,
    )
    config = transformers.Glm5NextConfig(text_config=text.to_dict(), vision_config=vision.to_dict())
    config.architectures = ["Glm5NextForConditionalGeneration"]
    original = transformers.Glm5NextForConditionalGeneration(config).to(torch.bfloat16)
    source = tmp_path / "glm-fp8-source"
    original.save_pretrained(source)
    state = load_file(source / "model.safetensors")
    for name, tensor in list(state.items()):
        if name.endswith("e_score_correction_bias"):
            state[name] = tensor.float()
    expected = dict(state)
    for index, (name, tensor) in enumerate(list(state.items())):
        if (
            tensor.ndim != 2
            or not name.endswith(".weight")
            or name.endswith(("embed_tokens.weight", "lm_head.weight"))
        ):
            continue
        scale = torch.full(tensor.shape, (index + 1) / 16)
        state[name] = torch.full_like(tensor, 2, dtype=torch.float8_e4m3fn)
        state[name.removesuffix(".weight") + ".weight_scale_inv"] = scale
        expected[name] = (2 * scale).to(torch.bfloat16)
    config.quantization_config = {"quant_method": "fp8", "weight_block_size": [1, 1]}
    config.save_pretrained(source)
    save_file(state, source / "model.safetensors", metadata={"format": "pt"})
    (source / "tokenizer_config.json").write_text("{}")
    destination = _onboarding_storage_fast_root / "glm-reduced"

    synth_model.materialize_pretrained(str(source), str(destination), 1, vision_depth=1)
    saved = load_file(destination / "model.safetensors")
    expected = {
        name: tensor for name, tensor in expected.items() if not name.startswith("model.language_model.layers.1.")
    }
    assert saved.keys() == expected.keys()
    assert not any(name.endswith("weight_scale_inv") for name in saved)
    for name, tensor in saved.items():
        torch.testing.assert_close(tensor, expected[name], atol=0, rtol=0)
    reopened = transformers.Glm5NextForConditionalGeneration.from_pretrained(destination, dtype=torch.bfloat16)
    reopened_state = revert_weight_conversion(reopened, reopened.state_dict())
    assert reopened_state.keys() == saved.keys()
    fp32_suffixes = ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight", "A_log", "dt_bias")
    for name, tensor in reopened_state.items():
        expected_tensor = saved[name].float() if name.endswith(fp32_suffixes) else saved[name]
        torch.testing.assert_close(tensor, expected_tensor, atol=0, rtol=0, msg=name)
    native = convert_hf_layer_to_prime(dict(saved), 0)
    prefix = "model.language_model.layers.0"
    for native_name, hf_name in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
        for expert in range(2):
            torch.testing.assert_close(
                native[f"{prefix}.mlp.experts.{native_name}"][expert],
                saved[f"{prefix}.mlp.experts.{expert}.{hf_name}.weight"],
                atol=0,
                rtol=0,
            )


@pytest.mark.parametrize("problem", ["unconfirmed", "fp8-weight", "remaining-scale"])
def test_onboarding_pretrained_fp8_materialization_rejects_incomplete_dequantization(
    _onboarding_storage_fast_root, monkeypatch, problem
) -> None:
    import transformers

    config = SimpleNamespace(
        architectures=["IncompleteDequantizationModel"],
        quantization_config={"quant_method": "fp8", "weight_block_size": [8, 8]},
    )
    weight = torch.ones(8, 8, dtype=torch.bfloat16)
    if problem == "fp8-weight":
        weight = weight.to(torch.float8_e4m3fn)

    class FakeModel:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            assert kwargs["quantization_config"].dequantize is True
            model = cls()
            model.config = config if problem == "unconfirmed" else SimpleNamespace()
            model.hf_quantizer = SimpleNamespace() if problem == "unconfirmed" else None
            return model

        def named_parameters(self):
            return [("projection.weight", weight)]

        def named_buffers(self):
            return [("projection.weight_scale_inv", torch.ones(1, 1))] if problem == "remaining-scale" else []

        def save_pretrained(self, *args, **kwargs):
            pytest.fail("an incompletely dequantized fixture must not be saved")

    monkeypatch.setattr(transformers, "IncompleteDequantizationModel", FakeModel, raising=False)
    monkeypatch.setattr(synth_model, "reduce_config", lambda *args: (config, 2, 1))
    monkeypatch.setattr(synth_model, "copy_sidecars", _onboarding_storage_unexpected)
    destination = _onboarding_storage_fast_root / "reduced"
    with pytest.raises(ValueError, match="dequantiz"):
        synth_model.materialize_pretrained("unused", str(destination), 1)
    assert not (destination / "config.json").exists()
    assert config.quantization_config["quant_method"] == "fp8"


def test_onboarding_pretrained_materialization_rejects_other_quantizers_before_loading(
    _onboarding_storage_fast_root, monkeypatch
) -> None:
    config = SimpleNamespace(architectures=["UnusedModel"], quantization_config={"quant_method": "gptq"})
    monkeypatch.setattr(synth_model, "reduce_config", lambda *args: (config, 2, 1))
    destination = _onboarding_storage_fast_root / "reduced"
    with pytest.raises(ValueError, match="explicitly dequantized FP8"):
        synth_model.materialize_pretrained("unused", str(destination), 1)
    assert not destination.exists()


def test_onboarding_pretrained_materialization_rejects_quantized_cache(
    _onboarding_storage_fast_root, monkeypatch
) -> None:
    config = SimpleNamespace(architectures=["UnusedModel"])
    monkeypatch.setattr(synth_model, "reduce_config", lambda *args: (config, 2, 1))
    monkeypatch.setattr(synth_model, "copy_sidecars", _onboarding_storage_unexpected)
    destination = _onboarding_storage_fast_root / "reduced"
    destination.mkdir()
    cached = '{"quantization_config": {"quant_method": "fp8"}}'
    (destination / "config.json").write_text(cached)
    with pytest.raises(ValueError, match="cached reduced fixture is quantized"):
        synth_model.materialize_pretrained("unused", str(destination), 1)
    assert (destination / "config.json").read_text() == cached


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
        test_id="single-step-grads",
        reference_records_dir=None,
    )
    args.config.parent.mkdir(parents=True)
    args.config.write_text("{}")
    args.test_spec.write_text("{}")
    monkeypatch.setattr(_fixed_tolerance_onboarding_SPEC, "local_config_path", lambda path: path)
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
        reference_attention_implementation="sdpa",
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

    def adapter(*args, **kwargs):
        assert kwargs["attn_implementation"] == "eager"
        return None

    monkeypatch.setattr(runner, "materialize_dss_peft_adapter", adapter)
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
        assert kwargs["attn"] == "sdpa"
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


# Local Cortex compact reference publication.


def _onboarding_config(tmp_path: Path, *, gpus: int = 4, glm: bool = False, lora: bool = False) -> Path:
    path = tmp_path / "configs" / "synthetic" / "h200" / "onboarding.config"
    path.parent.mkdir(parents=True)
    training = {
        "n_gpus": gpus,
        "sp_size": gpus,
        "max_seq_len": 32768,
        "attn_implementation": "sparse_mla" if glm else "flash_attention_3",
        "train_batch_size": gpus,
        "optimizer": {"lr": 2e-05},
    }
    if lora:
        training["peft_config"] = {
            "peft_type": "LORA",
            "r": 2,
            "lora_alpha": 4,
            "target_modules": ["q_proj"],
            "task_type": "CAUSAL_LM",
        }
    path.write_text(
        json.dumps(
            _config_factory_native_config(
                training,
                model_name="zai-org/GLM-5.3-Flash" if glm else "Synthetic/Onboarding",
            )
        )
    )
    return path


@pytest.fixture
def _local_cortex_reference_publication(tmp_path, monkeypatch):
    config_path = _onboarding_config(tmp_path, gpus=8, glm=True, lora=True)
    parser = argparse.ArgumentParser()
    onboard.add_arguments(parser)
    args = parser.parse_args(
        [
            "--config",
            str(config_path),
            "--test-id",
            "single-step-grads",
            "--backend",
            "cortex",
            "--base-url",
            "http://127.0.0.1:8003",
            "--database",
            "dev",
            "--schema",
            "dev",
            "--local-cortex-fixture",
            "--output-dir",
            str(tmp_path / "out"),
            "--test-spec",
            str(tmp_path / "candidate.json"),
            "--reference-records-dir",
            str(reference_records.DEFAULT_RECORDS_DIR / "synthetic-publication"),
        ]
    )
    cfg = load_config(config_path)
    spec = _applicable_tests_spec(["single-step-grads"])
    spec.config_id = cfg.config_id
    spec.config_path = str(config_path)
    spec.config_checksum = config_checksum(cfg)
    spec.model.cache_path = str(tmp_path / "model")
    state = SimpleNamespace(
        args=args,
        cfg=cfg,
        spec=spec,
        selected={"status": "fit", "layers": 4, "peak_gib": 1.0},
        events=[],
        calls=[],
        frozen=None,
        record=tmp_path / "synthetic-record.json",
        failure=None,
    )
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(onboard, "_validate_storage", lambda *a: None)
    monkeypatch.setattr(onboard, "_checkout_root", lambda *a: tmp_path)
    monkeypatch.setattr(onboard, "_model_shape", lambda *a: (45, 4, 4))
    monkeypatch.setattr(onboard, "_run_probe", lambda *a: state.selected)
    monkeypatch.setattr(_fixed_tolerance_onboarding_SPEC, "validate_against_host", lambda *a: None)
    monkeypatch.setattr(dss_driver, "available_gpus", lambda: 8)
    monkeypatch.setattr(workdir, "ensure_no_interrupted_builds", lambda: None)

    def initial(actual, selected, first_oom):
        assert actual is args and selected is state.selected and first_oom is None
        state.events.append("spec")
        spec.write(args.test_spec)

    def calibrate(actual, selected):
        assert actual is args and selected is state.selected
        state.events.append("calibration")
        spec.test_tolerances["single-step-grads"] = _fixed_tolerance_onboarding_SPEC.TestTolerance(
            absolute=0.005, calibration_runs=onboard.CALIBRATION_RUNS
        )
        spec.write(args.test_spec)
        state.frozen = args.test_spec.read_bytes()
        if state.failure == "calibration":
            raise RuntimeError("calibration failed")
        path = args.output_dir / "calibration.json"
        path.write_text('{"tolerance": {"selected_gate": 0.005}}')
        return path

    def model_config(path, **kwargs):
        assert path == spec.model.cache_path
        assert args.test_spec.read_bytes() == state.frozen
        return SimpleNamespace(vocab_size=128)

    def publish(*positional, **keywords):
        state.events.append("publication")
        assert args.test_spec.read_bytes() == state.frozen
        assert positional[1].tolerance_for("single-step-grads") == 0.005
        assert positional[1].test_tolerances["single-step-grads"].calibration_runs == onboard.CALIBRATION_RUNS
        state.calls.append((positional, keywords))
        if isinstance(state.failure, BaseException):
            raise state.failure
        state.record.write_text('{"synthetic": true}')
        return []

    def validate(actual, selected):
        assert actual is args and selected is state.selected
        state.events.append("validation")
        assert args.test_spec.read_bytes() == state.frozen
        assert state.record.exists() is args.local_cortex_fixture
        if state.failure == "validation":
            raise RuntimeError("validation failed")
        return {"status": "pass", "report": "synthetic-report"}

    monkeypatch.setattr(onboard, "_write_initial_spec", initial)
    monkeypatch.setattr(onboard, "_render_and_run_calibration", calibrate)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", model_config)
    monkeypatch.setattr(runner, "execute", publish)
    monkeypatch.setattr(onboard, "_run_final_regression", validate)
    return state


@pytest.mark.parametrize("local_fixture", [True, False])
def test_local_cortex_reference_publication_runs_after_frozen_gate_before_validation(
    _local_cortex_reference_publication, local_fixture
):
    state = _local_cortex_reference_publication
    state.args.local_cortex_fixture = local_fixture
    if not local_fixture:
        state.args.backend = "onprem"
    before = state.cfg.native.model_dump()
    assert onboard._parent(state.args) == 0
    assert state.events == (
        ["spec", "calibration", "publication", "validation"]
        if local_fixture
        else ["spec", "calibration", "validation"]
    )
    assert state.args.test_spec.read_bytes() == state.frozen
    assert load_config(state.args.config).native.model_dump() == before
    if local_fixture:
        positional, keywords = state.calls[0]
        assert positional[0].config_id == state.cfg.config_id
        assert keywords["prepare_references"] is True
        assert keywords["select"] == ["single-step-grads"]
        assert keywords["reference_records_dir"] == state.args.reference_records_dir
        assert keywords["reference_token_budget"] == state.args.reference_token_budget
        assert keywords["workdir"] == state.args.output_dir
        assert keywords["local_gateway_full_model"] is False
    else:
        assert state.calls == []
    manifest = json.loads((state.args.output_dir / "onboarding.json").read_text())
    assert manifest["selected_gate"] == 0.005
    assert manifest["validation_attempts"] == [{"status": "pass", "report": "synthetic-report"}]


@pytest.mark.parametrize("existing", [True, False])
@pytest.mark.parametrize("exception", [RuntimeError("publication failed"), KeyboardInterrupt()])
def test_local_cortex_reference_publication_failure_rolls_back_before_target(
    _local_cortex_reference_publication, existing, exception
):
    state = _local_cortex_reference_publication
    previous = b'{ "retained": "exact prior bytes" }\n'
    if existing:
        state.args.test_spec.write_bytes(previous)
    state.failure = exception
    with pytest.raises(type(exception)):
        onboard._parent(state.args)
    assert state.events == ["spec", "calibration", "publication"]
    assert not state.record.exists()
    assert not (state.args.output_dir / "onboarding.json").exists()
    if existing:
        assert state.args.test_spec.read_bytes() == previous
    else:
        assert not state.args.test_spec.exists()


@pytest.mark.parametrize("existing", [True, False])
@pytest.mark.parametrize("failure", ["calibration", "validation"])
def test_local_cortex_reference_publication_keeps_calibration_and_target_rollback(
    _local_cortex_reference_publication, existing, failure
):
    state = _local_cortex_reference_publication
    previous = b'{ "retained": "exact prior bytes" }\n'
    if existing:
        state.args.test_spec.write_bytes(previous)
    state.failure = failure
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        onboard._parent(state.args)
    assert state.events == (
        ["spec", "calibration"] if failure == "calibration" else ["spec", "calibration", "publication", "validation"]
    )
    assert state.record.exists() is (failure == "validation")
    assert not (state.args.output_dir / "onboarding.json").exists()
    if existing:
        assert state.args.test_spec.read_bytes() == previous
    else:
        assert not state.args.test_spec.exists()


def test_local_cortex_reference_publication_final_command_uses_same_record_directory(tmp_path, monkeypatch):
    from arctic_platform.correctness.harness.execution import parse_args

    parser = argparse.ArgumentParser()
    onboard.add_arguments(parser)
    args = parser.parse_args(["--config", str(tmp_path / "job.config"), "--local-cortex-fixture"])
    args.output_dir = tmp_path / "out"
    args.checkout_root = tmp_path
    args.reference_records_dir = reference_records.DEFAULT_RECORDS_DIR / "synthetic-publication"
    captured = []

    def process(command, **kwargs):
        captured.append(command)
        output = Path(command[command.index("--out") + 1])
        (output / "report.json").write_text('{"totals": {"pass": 1, "fail": 0}}')
        return SimpleNamespace(stdout=io.StringIO(), wait=lambda: 0)

    monkeypatch.setattr(onboard.subprocess, "Popen", process)
    assert onboard._run_final_regression(args, {"layers": 4})["status"] == "pass"
    parsed = parse_args(captured[0][3:])
    assert parsed.reference_records_dir == args.reference_records_dir
    assert parsed.local_cortex_fixture is True


@pytest.fixture
def _local_cortex_optimizer_onboarding(tmp_path, monkeypatch):
    from dataclasses import replace

    from arctic_platform.correctness.harness.config import TestTolerance

    config_path = Path(__file__).resolve().parents[1] / "configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config"
    parser = argparse.ArgumentParser()
    onboard.add_arguments(parser)
    args = parser.parse_args(
        [
            "--config",
            str(config_path),
            "--backend",
            "cortex",
            "--base-url",
            "http://127.0.0.1:8003",
            "--database",
            "dev",
            "--schema",
            "dev",
            "--local-cortex-fixture",
            "--test-id",
            "single-step-optimizer",
            "--reference-calibration-workers",
            "8",
            "--reference-token-budget",
            "64",
            "--cache-root",
            str(tmp_path / "models"),
            "--output-dir",
            str(tmp_path / "work"),
            "--test-spec",
            str(tmp_path / "reviewed.json"),
            "--reference-records-dir",
            str(tmp_path / "records"),
        ]
    )
    cfg = load_config(config_path)
    model = args.cache_root / "Qwen3-8B-4L"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"num_hidden_layers": 4, "vocab_size": 32}')
    (model / "model.safetensors").write_bytes(b"synthetic CPU fixture, no model payload execution")
    spec = _applicable_tests_spec(["single-step-grads"])
    spec.config_id = cfg.config_id
    spec.config_path = str(config_path)
    spec.config_checksum = config_checksum(cfg)
    spec.model.source_checkpoint = cfg.model_name
    spec.model.cache_path = str(model)
    spec.model.num_hidden_layers = 4
    template = spec.arms[0]
    spec.arms = [
        replace(template, name="gas1", global_batch_size=4),
        replace(template, name="gas4", global_batch_size=16),
    ]
    spec.test_tolerances["single-step-grads"] = TestTolerance(absolute=0.005)
    spec.test_tolerances["single-step-optimizer"] = TestTolerance(absolute=42.0, status="retained full-tensor gate")
    spec.test_settings["single-step-grads"] = {"retained": "frozen Test 1 settings"}
    spec.write(args.test_spec)
    state = SimpleNamespace(
        args=args, cfg=cfg, spec=spec, previous=args.test_spec.read_bytes(), events=[], failure=None
    )
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(onboard, "_validate_storage", lambda *a: None)
    monkeypatch.setattr(onboard, "_checkout_root", lambda *a: tmp_path)
    monkeypatch.setattr(onboard, "_model_shape", lambda *a: (32, 1, 4))
    monkeypatch.setattr(reference_records, "require_git_records_dir", lambda path: Path(path))

    def forbidden(*args, **kwargs):
        pytest.fail(
            "local scalar-norm onboarding replayed sizing, materialization, full-tensor calibration or a target"
        )

    monkeypatch.setattr(onboard, "_run_probe", forbidden)
    monkeypatch.setattr(onboard, "_run_optimizer_calibration", forbidden)
    monkeypatch.setattr(synth_model, "materialize_pretrained", forbidden)
    monkeypatch.setattr(runner, "run_reference", forbidden)
    monkeypatch.setattr(dss_driver, "available_gpus", forbidden)
    monkeypatch.setattr(workdir, "ensure_no_interrupted_builds", forbidden)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: SimpleNamespace(vocab_size=32))

    def prepare(actual_cfg, actual_spec, arms, **kwargs):
        state.events.append("calibration")
        assert actual_cfg.config_id == cfg.config_id
        assert actual_spec.model == spec.model
        assert actual_spec.arms == spec.arms
        assert args.test_spec.read_bytes() == state.previous
        assert kwargs["select"] == ["single-step-optimizer"]
        assert kwargs["prepare_references"] is True
        assert kwargs["reference_calibration_runs"] == 5
        assert kwargs["reference_calibration_workers"] == args.reference_calibration_workers
        assert kwargs["reference_token_budget"] == 64
        assert isinstance(kwargs["client_backend"], CortexConfig)
        staging = kwargs["reference_records_dir"]
        assert staging.parent == args.reference_records_dir
        assert staging.name.startswith(".optimizer-onboarding-")
        assert list(staging.iterdir()) == []
        if state.failure == "calibration":
            raise RuntimeError("calibration failed")
        if state.failure == "target-results":
            return ["not reference-only"]
        ctx = SimpleNamespace(cfg=cfg, spec=actual_spec, config_id=cfg.config_id, reference_records_dir=staging)
        training = cfg.effective_training
        for index, arm in enumerate(arms):
            if state.failure == "missing-arm" and arm.name == "gas4":
                continue
            batch = batches.build_batch(arm.name, arm.global_batch_size, arm.max_seq_len, 32, seed=batches.SEED)
            batches.save(batch, args.output_dir / f"batch-{arm.name}.pt")
            recipe = {
                "model": reference_records.model_identity(model),
                "initialization": "external-model",
                "synthetic_inputs": {
                    "version": "private-cpu-generator-v1",
                    "seed": batches.SEED,
                    "rows": arm.global_batch_size,
                    "max_seq_len": arm.max_seq_len,
                    "vocab_size": 32,
                    "batch": reference_records.batch_identity(batch),
                },
                "attention": cfg.reference_attention_implementation,
                "token_budget": 64,
                "ce_chunk": 2048,
                "training": {
                    "fp32_lm_head": bool(training.get("fp32_lm_head", False)),
                    "fused_cross_entropy": cfg.fused_cross_entropy,
                    "mixer_packing": False,
                    "matmul_precision": training.get("matmul_precision", "highest"),
                    "peft_config": training.get("peft_config"),
                    "lm_head_token_chunk_size": cfg.lm_head_token_chunk_size,
                    "lm_head_vocab_chunk_size": training.get("fused_lm_head_vocab_chunk_size", 8192),
                    "deterministic": True,
                },
                "optimizer": {
                    "config": cfg.training["optimizer"],
                    "learning_rate": runner.OPTIMIZER_LEARNING_RATE,
                    "gradient_clipping": cfg.training.get("gradient_clipping"),
                    "dtype": cfg.optimizer_dtype,
                },
            }
            if state.failure == "wrong-inputs":
                recipe["synthetic_inputs"]["rows"] += 1
            if state.failure == "wrong-training":
                recipe["training"]["deterministic"] = False
            identity = reference_records.workload_identity(ctx, "single-step-optimizer", recipe)
            if state.failure == "source-drift":
                identity["reference_sources"] = {}
            spread = 0.0002 if index == 0 else 0.001
            gate = {
                "criterion": "optimizer-state-l2-norm-v1",
                "absolute": 0.001 if index == 0 else 0.004,
                "status": "reviewed",
                "provenance": {
                    "policy": "onboarding-norm-repeatability-4x-upward-1e-3-v1",
                    "calibration_runs": 5,
                    "maximum_pairwise_norm_delta": spread,
                    "worst_quantity": "layer.weight::exp_avg",
                    "multiplier": 4,
                    "minimum_gate": 0.001,
                },
            }
            if state.failure == "legacy-provenance":
                gate["provenance"] = "retained full-tensor gate"
            if state.failure == "wrong-runs":
                gate["provenance"]["calibration_runs"] = 8
            if state.failure == "wrong-gate":
                gate["absolute"] = 0.002
            if state.failure == "negative-spread":
                gate["provenance"]["maximum_pairwise_norm_delta"] = -0.1
            measurements = {
                "loss": 1.0,
                "microbatches": 1,
                "active_tokens": batch.active_tokens,
                "grad_norms": {"layer.weight": 1.0},
                "optimizer_norms": {
                    "criterion": "optimizer-state-l2-norm-v1",
                    "step": 1,
                    "state_keys": ["parameter_update_residual", "exp_avg", "exp_avg_sq"],
                    "parameter_count": 1,
                    "optimizer_values_count": 3,
                    "parameters": {
                        "layer.weight": {
                            "shape": [2],
                            "parameter_update_residual": 0.0,
                            "exp_avg": 0.0,
                            "exp_avg_sq": 0.0,
                        }
                    },
                },
                "optimizer_norm_gate": gate,
            }
            path = reference_records.publish_record(ctx, identity, measurements)
            if state.failure == "publication-conflict":
                retained = json.loads(json.dumps(measurements))
                retained["loss"] = 2.0
                old_ctx = SimpleNamespace(reference_records_dir=args.reference_records_dir)
                reference_records.publish_record(old_ctx, identity, retained)
            if state.failure == "corrupt-record":
                data = json.loads(path.read_text())
                data["payload_sha256"] = "corrupt"
                path.write_text(json.dumps(data))
        if state.failure == "wrong-model":
            (model / "model.safetensors").write_bytes(b"changed fixture after preparation")
        return []

    def validate(actual_args, selected):
        assert actual_args is args and selected == {"layers": 4}
        state.events.append("validation")
        published = ConfigSpec.read(args.test_spec)
        assert published.tolerance_for("single-step-optimizer") == 0.004
        assert published.test_settings["single-step-optimizer"]["norm_gates_by_arm"] == {"gas1": 0.001, "gas4": 0.004}
        assert len(list(args.reference_records_dir.glob(f"{cfg.config_id}/single-step-optimizer/*.json"))) == 2
        if state.failure == "validation":
            raise RuntimeError("validation failed")
        if state.failure == "interrupt":
            raise KeyboardInterrupt()
        if state.failure == "target-fail":
            return {"status": "failed", "log": "synthetic mismatch"}
        return {"status": "pass", "report": "synthetic-report"}

    monkeypatch.setattr(runner, "execute", prepare)
    monkeypatch.setattr(onboard, "_run_final_regression", validate)
    return state


@pytest.mark.parametrize("workers", [1, 8])
def test_local_cortex_optimizer_scalar_norm_publication_preserves_test1_and_per_arm_gates(
    _local_cortex_optimizer_onboarding, workers
):
    state = _local_cortex_optimizer_onboarding
    state.args.reference_calibration_workers = workers
    native = state.cfg.native.model_dump()
    assert onboard._parent(state.args) == 0
    assert state.events == ["calibration", "validation"]
    actual = ConfigSpec.read(state.args.test_spec)
    assert actual.model == state.spec.model
    assert actual.arms == state.spec.arms
    assert actual.test_tolerances["single-step-grads"] == state.spec.test_tolerances["single-step-grads"]
    assert actual.test_settings["single-step-grads"] == state.spec.test_settings["single-step-grads"]
    assert state.cfg.native.model_dump() == native
    assert actual.test_settings["single-step-optimizer"]["calibration_metric"] == "optimizer-state-l2-norm-v1"
    manifest = json.loads((state.args.output_dir / "onboarding-t02.json").read_text())
    assert manifest["criterion"] == "optimizer-state-l2-norm-v1"
    assert manifest["selected_gate"] == 0.004
    assert set(manifest["reference_records_by_arm"]) == {"gas1", "gas4"}
    for name, receipt in manifest["reference_records_by_arm"].items():
        path = Path(receipt["record"])
        assert reference_records.file_identity(path) == receipt["record_identity"]
        record = json.loads(path.read_text())
        assert record["identity"] == receipt["identity"]
        assert record["measurements"]["optimizer_norm_gate"] == receipt["gate"]
        assert receipt["gate"]["absolute"] == (0.001 if name == "gas1" else 0.004)
        assert receipt["gate"]["provenance"]["calibration_runs"] == 5
    assert list(state.args.reference_records_dir.glob(".optimizer-onboarding-*")) == []


@pytest.mark.parametrize(
    "failure",
    [
        "calibration",
        "target-results",
        "missing-arm",
        "wrong-inputs",
        "wrong-training",
        "source-drift",
        "wrong-model",
        "legacy-provenance",
        "wrong-runs",
        "wrong-gate",
        "negative-spread",
        "corrupt-record",
        "publication-conflict",
        "validation",
        "target-fail",
        "interrupt",
    ],
)
def test_local_cortex_optimizer_scalar_norm_failure_restores_exact_spec_and_no_verdict(
    _local_cortex_optimizer_onboarding, failure
):
    state = _local_cortex_optimizer_onboarding
    state.failure = failure
    exception = KeyboardInterrupt if failure == "interrupt" else (ValueError, RuntimeError)
    with pytest.raises(exception):
        onboard._parent(state.args)
    assert state.args.test_spec.read_bytes() == state.previous
    assert not (state.args.output_dir / "onboarding-t02.json").exists()
    assert list(state.args.reference_records_dir.glob(".optimizer-onboarding-*")) == []
    if failure in {"validation", "target-fail", "interrupt"}:
        assert state.events == ["calibration", "validation"]
        assert len(list(state.args.reference_records_dir.glob("*/single-step-optimizer/*.json"))) == 2
    else:
        assert state.events == ["calibration"]


@pytest.mark.parametrize("failure", ["checksum", "fixture-path", "fixture-depth", "no-test1"])
def test_local_cortex_optimizer_scalar_norm_rejects_unreviewed_fixture_before_calibration(
    _local_cortex_optimizer_onboarding, failure
):
    state = _local_cortex_optimizer_onboarding
    if failure == "checksum":
        state.spec.config_checksum = "wrong"
    elif failure == "fixture-path":
        state.spec.model.cache_path += "-other"
    elif failure == "fixture-depth":
        state.spec.model.num_hidden_layers = 8
    else:
        state.spec.applicable_tests = []
    state.spec.write(state.args.test_spec)
    prior = state.args.test_spec.read_bytes()
    with pytest.raises(ValueError):
        onboard._parent(state.args)
    assert state.events == []
    assert state.args.test_spec.read_bytes() == prior
    assert not (state.args.output_dir / "onboarding-t02.json").exists()


@pytest.mark.parametrize(("runs", "workers"), [(16, 1), (8, 8), (0, 8), (5, 0), (5, 9)])
def test_local_cortex_optimizer_scalar_norm_rejects_invalid_repeat_policy_before_reference(
    _local_cortex_optimizer_onboarding, runs, workers
):
    state = _local_cortex_optimizer_onboarding
    state.args.reference_calibration_runs = runs
    state.args.reference_calibration_workers = workers
    with pytest.raises(ValueError, match="5 repeats"):
        onboard._parent(state.args)
    assert state.events == []
    assert state.args.test_spec.read_bytes() == state.previous


def _optimizer_child_files(args, tmp_path):
    bootstrap = tmp_path / "caller.py"
    bootstrap.write_text("# Synthetic child source; never executed by these CPU controls.\n")
    config = tmp_path / "caller.json"
    config.write_text('{"private_token": "never include configuration contents in the receipt"}\n')
    for name, path in (
        ("optimizer_validation_bootstrap", bootstrap),
        ("optimizer_validation_config", config),
    ):
        setattr(args, name, path)
        setattr(args, name + "_sha256", hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def _optimizer_validation_child(tmp_path):
    parser = argparse.ArgumentParser()
    onboard.add_arguments(parser)
    args = parser.parse_args(
        [
            "--config",
            str(tmp_path / "job.config"),
            "--backend",
            "cortex",
            "--base-url",
            "http://127.0.0.1:8003",
            "--database",
            "dev",
            "--schema",
            "dev",
            "--test-id",
            "single-step-optimizer",
            "--local-cortex-fixture",
            "--test-spec",
            str(tmp_path / "reviewed.json"),
            "--reference-records-dir",
            str(tmp_path / "fresh-records"),
        ]
    )
    args.output_dir = tmp_path / "out"
    args.checkout_root = tmp_path
    _optimizer_child_files(args, tmp_path)
    return args


def test_optimizer_validation_child_forwards_original_regression_and_retains_source_receipt(
    _optimizer_validation_child, monkeypatch
):
    from arctic_platform.correctness.harness.execution import parse_args

    args = _optimizer_validation_child
    admission = onboard._optimizer_validation_bootstrap(args)
    args._admitted_optimizer_validation_bootstrap = admission
    captured = []

    def process(command, **kwargs):
        captured.append((command, kwargs))
        out = Path(command[command.index("--out") + 1])
        (out / "report.json").write_text('{"totals": {"pass": 2, "fail": 0}}')
        return SimpleNamespace(stdout=io.StringIO("synthetic child\n"), wait=lambda: 0)

    monkeypatch.setattr(onboard.subprocess, "Popen", process)
    result = onboard._run_final_regression(args, {"layers": 4})
    command, kwargs = captured[0]
    assert command[:9] == [
        sys.executable,
        "-B",
        "-u",
        str(args.optimizer_validation_bootstrap),
        "--bridge-config",
        str(args.optimizer_validation_config),
        "--config-sha",
        args.optimizer_validation_config_sha256,
        "--",
    ]
    parsed = parse_args(command[9:])
    assert parsed.test == ["single-step-optimizer"]
    assert parsed.test_spec == args.test_spec
    assert parsed.reference_records_dir == args.reference_records_dir
    assert parsed.local_cortex_fixture is True
    assert parsed.backend == "cortex"
    assert parsed.base_url == args.base_url
    assert kwargs["cwd"] == args.checkout_root
    assert kwargs["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert result["status"] == "pass"
    assert result["optimizer_validation_bootstrap"] == admission
    persisted = (args.output_dir / "validation/4-layers/validation.json").read_text()
    assert json.loads(persisted) == result
    assert "private_token" not in persisted


@pytest.mark.parametrize(
    "missing",
    [
        "optimizer_validation_bootstrap",
        "optimizer_validation_bootstrap_sha256",
        "optimizer_validation_config",
        "optimizer_validation_config_sha256",
    ],
)
def test_optimizer_validation_child_rejects_incomplete_source_pins(_optimizer_validation_child, missing):
    setattr(_optimizer_validation_child, missing, None)
    with pytest.raises(ValueError, match="both bootstrap/config"):
        onboard._optimizer_validation_bootstrap(_optimizer_validation_child)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("test_id", "single-step-grads"),
        ("backend", "onprem"),
        ("local_cortex_fixture", False),
        ("prepare_references", True),
        ("probe_layers", 4),
        ("local_gateway_full_model", True),
        ("reference_tests", ["single-step-optimizer"]),
    ],
)
def test_optimizer_validation_child_rejects_other_modes_before_storage_or_reference(
    _optimizer_validation_child, monkeypatch, name, value
):
    args = _optimizer_validation_child
    setattr(args, name, value)
    monkeypatch.setattr(
        onboard, "_validate_storage", lambda *a: pytest.fail("unsupported child route reached storage")
    )
    with pytest.raises(ValueError, match="only local Cortex Test 2"):
        onboard.run(args)


@pytest.mark.parametrize("role", ["bootstrap", "config"])
@pytest.mark.parametrize("fault", ["changed-bytes", "replacement", "symlink", "hardlink", "bad-sha"])
def test_optimizer_validation_child_rejects_source_faults_before_target(
    _optimizer_validation_child, monkeypatch, role, fault
):
    args = _optimizer_validation_child
    args._admitted_optimizer_validation_bootstrap = onboard._optimizer_validation_bootstrap(args)
    path = getattr(args, "optimizer_validation_" + role)
    if fault == "changed-bytes":
        path.write_text("different source\n")
    elif fault == "replacement":
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    elif fault == "symlink":
        moved = path.with_suffix(".original")
        path.rename(moved)
        path.symlink_to(moved)
    elif fault == "hardlink":
        path.with_suffix(".linked").hardlink_to(path)
    else:
        setattr(args, "optimizer_validation_" + role + "_sha256", "bad")
    monkeypatch.setattr(onboard.subprocess, "Popen", lambda *a, **k: pytest.fail("source fault reached target child"))
    with pytest.raises(ValueError, match="optimizer validation"):
        onboard._run_final_regression(args, {"layers": 4})
    assert not args.output_dir.exists()


@pytest.mark.parametrize("role", ["bootstrap", "config"])
def test_optimizer_validation_child_rejects_source_drift_after_child_exit(
    _optimizer_validation_child, monkeypatch, role
):
    args = _optimizer_validation_child

    def process(command, **kwargs):
        out = Path(command[command.index("--out") + 1])
        (out / "report.json").write_text('{"totals": {"pass": 2, "fail": 0}}')

        def wait():
            getattr(args, "optimizer_validation_" + role).write_text("drift while target ran\n")
            return 0

        return SimpleNamespace(stdout=io.StringIO(), wait=wait)

    monkeypatch.setattr(onboard.subprocess, "Popen", process)
    with pytest.raises(ValueError, match="optimizer validation.*changed"):
        onboard._run_final_regression(args, {"layers": 4})
    assert not (args.output_dir / "validation/4-layers/validation.json").exists()


@pytest.mark.parametrize("failure", ["pre-calibration-pin", "pre-child-drift", "child-failure", "child-interrupt"])
def test_optimizer_validation_child_failure_restores_exact_test1_without_reference_replay(
    request, monkeypatch, tmp_path, failure
):
    final_regression = onboard._run_final_regression
    state = request.getfixturevalue("_local_cortex_optimizer_onboarding")
    state.args.checkout_root = tmp_path
    _optimizer_child_files(state.args, tmp_path)
    monkeypatch.setattr(onboard, "_run_final_regression", final_regression)
    calibrations = []

    def calibration(args, cfg, spec):
        calibrations.append("synthetic calibration, no runner/HF call")
        spec.applicable_tests.append("single-step-optimizer")
        spec.write(args.test_spec)
        if failure == "pre-child-drift":
            args.optimizer_validation_bootstrap.write_text("changed after synthetic calibration\n")
        return args.output_dir / "not-consumed-on-failure.json"

    def process(command, **kwargs):
        if failure == "child-interrupt":
            raise KeyboardInterrupt()
        assert failure == "child-failure"
        return SimpleNamespace(stdout=io.StringIO("synthetic child failure\n"), wait=lambda: 1)

    monkeypatch.setattr(onboard, "_run_local_optimizer_calibration", calibration)
    monkeypatch.setattr(onboard.subprocess, "Popen", process)
    if failure == "pre-calibration-pin":
        state.args.optimizer_validation_bootstrap_sha256 = "0" * 64
    expected = (
        KeyboardInterrupt
        if failure == "child-interrupt"
        else RuntimeError if failure == "child-failure" else ValueError
    )
    with pytest.raises(expected):
        onboard._onboard_optimizer_test(state.args, state.cfg)
    assert state.args.test_spec.read_bytes() == state.previous
    assert len(calibrations) == (0 if failure == "pre-calibration-pin" else 1)
    assert state.events == []
    assert not (state.args.output_dir / "onboarding-t02.json").exists()


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "unexpected",
        "retained-unexpected",
        "outside-source",
        "missing",
        "mismatched",
    ],
)
def test_onboarding_pretrained_omitted_layers_do_not_warn(
    _onboarding_storage_fast_root, tmp_path, monkeypatch, caplog, problem
) -> None:
    config = transformers.Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
    )
    model = transformers.Qwen3ForCausalLM(config).to(torch.bfloat16)
    source = tmp_path / "source"
    model.save_pretrained(source)
    (source / "tokenizer_config.json").write_text("{}")
    state = load_file(source / "model.safetensors")
    key = "model.layers.0.input_layernorm.weight"
    if problem == "unexpected":
        key = "unrelated.weight"
        state[key] = torch.ones(16, dtype=torch.bfloat16)
    elif problem == "retained-unexpected":
        key = "model.layers.0.unrelated.weight"
        state[key] = torch.ones(16, dtype=torch.bfloat16)
    elif problem == "outside-source":
        key = "model.layers.3.input_layernorm.weight"
        state[key] = torch.ones(16, dtype=torch.bfloat16)
    elif problem == "missing":
        del state[key]
    elif problem == "mismatched":
        state[key] = torch.ones(17, dtype=torch.bfloat16)
    save_file(state, source / "model.safetensors", metadata={"format": "pt"})
    destination = _onboarding_storage_fast_root / "reduced"
    original_ignore = transformers.Qwen3ForCausalLM._keys_to_ignore_on_load_unexpected
    monkeypatch.setattr(logging.getLogger("transformers"), "propagate", True)
    caplog.set_level(logging.WARNING, logger="transformers")
    save_reduced = synth_model._save_reduced_pretrained

    def save_original_class(loaded, path):
        assert type(loaded) is transformers.Qwen3ForCausalLM
        save_reduced(loaded, path)

    monkeypatch.setattr(synth_model, "_save_reduced_pretrained", save_original_class)

    if problem == "mismatched":
        with pytest.raises(RuntimeError, match="ignore_mismatched_sizes"):
            synth_model.materialize_pretrained(str(source), str(destination), 1)
        assert not (destination / "config.json").exists()
    else:
        synth_model.materialize_pretrained(str(source), str(destination), 1)
        saved = load_file(destination / "model.safetensors")
        assert not any(name.startswith(("model.layers.1.", "model.layers.2.")) for name in saved)
        assert json.loads((destination / "config.json").read_text())["architectures"] == ["Qwen3ForCausalLM"]
        for name, tensor in saved.items():
            if problem == "missing" and name == key:
                continue
            torch.testing.assert_close(tensor, state[name], atol=0, rtol=0, msg=name)

    report = caplog.text
    assert "model.layers.1." not in report
    assert "model.layers.2." not in report
    assert "model.layers.{1, 2}." not in report
    if problem is None:
        assert "LOAD REPORT" not in report
        assert "UNEXPECTED" not in report
    else:
        assert key in report
        expected_status = {
            "unexpected": "UNEXPECTED",
            "retained-unexpected": "UNEXPECTED",
            "outside-source": "UNEXPECTED",
            "missing": "MISSING",
            "mismatched": "MISMATCH",
        }
        assert expected_status[problem] in report
    assert transformers.Qwen3ForCausalLM._keys_to_ignore_on_load_unexpected is original_ignore

    caplog.clear()
    reduced, _, _ = synth_model.reduce_config(str(source), 1)
    if problem != "mismatched":
        direct, info = transformers.Qwen3ForCausalLM.from_pretrained(
            source, config=reduced, dtype=torch.bfloat16, output_loading_info=True
        )
        assert type(direct) is transformers.Qwen3ForCausalLM
        assert any(name.startswith("model.layers.1.") for name in info["unexpected_keys"])
