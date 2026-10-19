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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prepared references retain compact, immutable, content-bound CPU measurements."""

import builtins
import copy
import hashlib
import json
import os
import shlex
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier
from types import SimpleNamespace

import pytest

from arctic_platform.correctness.harness import reference_records as records

GRADS = "single-step-grads"
OPTIMIZER = "single-step-optimizer"
E2E = "e2e-sft-train-validate"
PARAMETER = "layers.0.self_attn.q_proj.lora_A.default.weight"


@dataclass
class _Tolerance:
    absolute: float = 1e-3
    calibration_runs: int = 16


@pytest.fixture
def ctx(tmp_path):
    return SimpleNamespace(
        config_id="unit-lora",
        cfg=SimpleNamespace(gpu_type="H200"),
        spec=SimpleNamespace(
            config_checksum="c" * 64,
            config_path="unit-lora.config",
            test_tolerances={GRADS: _Tolerance()},
        ),
        reference_records_dir=tmp_path / "references",
        reference_record_paths={},
    )


@pytest.fixture
def identity(ctx, monkeypatch):
    monkeypatch.setattr(records.importlib.metadata, "version", lambda package: "unit-version")
    monkeypatch.setattr(records, "file_identity", lambda path: {"sha256": "a" * 64, "size": 1})
    return records.workload_identity(
        ctx,
        GRADS,
        {"arm": "gas1", "model": {"sha256": "b" * 64}, "batch": {"sha256": "d" * 64}},
    )


def _identity_for(identity, check_id):
    result = copy.deepcopy(identity)
    result["check_id"] = check_id
    result["criterion"] = records.CRITERIA[check_id]
    if check_id != GRADS:
        result.pop("gate", None)
    return result


def _measurements(check_id=GRADS):
    if check_id == E2E:
        return {
            "train_losses": [1.25 - step / 1000 for step in range(100)],
            "train_gradient_norms": [0.5] * 100,
            "optimizer_steps": 100,
            "validation_cross_entropy": 0.75,
            "scored_answer_tokens": 32,
        }
    measurements = {
        "loss": 1.25,
        "microbatches": 4,
        "active_tokens": 16,
        "grad_norms": {PARAMETER: 0.5},
    }
    if check_id == OPTIMIZER:
        measurements.update(
            optimizer_norms={
                "criterion": records.CRITERIA[OPTIMIZER],
                "step": 1,
                "state_keys": ["parameter_update_residual", "exp_avg", "exp_avg_sq"],
                "parameter_count": 1,
                "optimizer_values_count": 3,
                "parameters": {
                    PARAMETER: {
                        "shape": [2, 3],
                        "parameter_update_residual": 0.1,
                        "exp_avg": 0.05,
                        "exp_avg_sq": 0.001,
                    }
                },
            },
            optimizer_norm_gate={
                "criterion": records.CRITERIA[OPTIMIZER],
                "absolute": 1e-3,
                "status": "reviewed",
                "provenance": {"review": "unit-onboarding", "runs": 16},
            },
        )
    return measurements


def _no_hf(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"transformers", "peft"} or name.startswith("arctic_platform.correctness.reference"):
            raise AssertionError(f"prepared record attempted an HF import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


@pytest.mark.parametrize("check_id", [GRADS, OPTIMIZER, E2E])
def test_publish_load_compact_measurements_without_hf(ctx, identity, monkeypatch, check_id):
    identity = _identity_for(identity, check_id)
    measurements = _measurements(check_id)
    _no_hf(monkeypatch)

    path = records.publish_record(ctx, identity, measurements)
    loaded = records.load_record(ctx, identity)

    assert loaded == measurements
    assert "measurements" not in loaded
    assert path.parent == ctx.reference_records_dir / ctx.config_id / check_id
    assert path.suffix == ".json"
    assert ctx.reference_record_paths == {path.stem: str(path)}
    payload = path.read_bytes()
    assert payload.endswith(b"\n")
    assert b"\n " not in payload
    assert len(payload) < 8192
    assert set(json.loads(payload)) == {"schema", "identity", "measurements", "payload_sha256"}
    assert not list(ctx.reference_records_dir.rglob("*.pt"))
    assert not list(ctx.reference_records_dir.rglob(".prepare-*"))


def test_missing_record_has_actionable_onboarding_command_without_hf(ctx, identity, monkeypatch):
    _no_hf(monkeypatch)

    with pytest.raises(records.ReferencePrerequisiteError) as error:
        records.load_record(ctx, identity)

    assert "prepared HF reference missing" in str(error.value)
    assert (
        "python -m arctic_platform.correctness onboard "
        "--config unit-lora.config --prepare-references --test-id single-step-grads"
        in str(error.value)
    )
    assert str(ctx.reference_records_dir) in str(error.value)
    assert not ctx.reference_records_dir.exists()
    assert ctx.reference_record_paths == {}


def test_missing_combined_reference_command_roundtrips_selection_and_workload(ctx, identity, monkeypatch):
    _no_hf(monkeypatch)
    ctx.spec.config_path = "configs/unit lora.config"
    ctx.reference_records_dir /= "compact records"
    ctx.local_gateway_full_model = True
    identity["recipe"].update(optimizer={"lr": 1e-4}, token_budget=64)

    with pytest.raises(records.ReferencePrerequisiteError) as error:
        records.load_record(ctx, identity)

    command = shlex.split(str(error.value).split("; run ", 1)[1])
    assert command == [
        "python",
        "-m",
        "arctic_platform.correctness",
        "onboard",
        "--config",
        ctx.spec.config_path,
        "--prepare-references",
        "--reference-tests",
        GRADS,
        OPTIMIZER,
        "--reference-records-dir",
        str(ctx.reference_records_dir),
        "--local-gateway-full-model",
        "--reference-token-budget",
        "64",
    ]
    assert not ctx.reference_records_dir.exists()
    assert not ctx.reference_record_paths


def test_repeat_publication_is_byte_identical_and_does_not_replace_file(ctx, identity):
    measurements = _measurements()
    path = records.publish_record(ctx, identity, measurements)
    before = path.stat()
    payload = path.read_bytes()

    repeated = records.publish_record(ctx, identity, dict(reversed(list(measurements.items()))))

    assert repeated == path
    assert path.read_bytes() == payload
    assert path.stat().st_ino == before.st_ino
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert list(path.parent.glob("*.json")) == [path]


def test_conflicting_publication_preserves_original_record(ctx, identity):
    original = _measurements()
    path = records.publish_record(ctx, identity, original)
    payload = path.read_bytes()
    changed = {**original, "loss": 2.0}

    with pytest.raises(records.ReferencePrerequisiteError, match="conflicting immutable"):
        records.publish_record(ctx, identity, changed)

    assert path.read_bytes() == payload
    assert records.load_record(ctx, identity) == original
    assert not list(path.parent.glob(".prepare-*"))


@pytest.mark.parametrize("corruption", ["measurement", "payload_hash", "schema", "identity"])
def test_corrupt_or_incompatible_record_requires_onboarding(ctx, identity, corruption):
    path = records.publish_record(ctx, identity, _measurements())
    record = json.loads(path.read_text())
    if corruption == "measurement":
        record["measurements"]["loss"] += 1
    elif corruption == "payload_hash":
        record["payload_sha256"] = "0" * 64
    elif corruption == "schema":
        record["schema"] = records.SCHEMA_VERSION + 1
    else:
        record["identity"]["config_checksum"] = "e" * 64
    path.write_text(json.dumps(record))
    ctx.reference_record_paths.clear()

    with pytest.raises(records.ReferencePrerequisiteError, match="corrupt or incompatible.*run"):
        records.load_record(ctx, identity)

    assert ctx.reference_record_paths == {}


@pytest.mark.parametrize("payload", ["{", "[]", "null"])
def test_malformed_record_is_an_actionable_prerequisite_error(ctx, identity, payload):
    path = records.publish_record(ctx, identity, _measurements())
    path.write_text(payload)

    with pytest.raises(records.ReferencePrerequisiteError, match="run.*--prepare-references"):
        records.load_record(ctx, identity)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_record_payload_is_an_actionable_prerequisite_error(ctx, identity, nonfinite):
    path = records.publish_record(ctx, identity, _measurements())
    record = json.loads(path.read_text())
    record["measurements"]["loss"] = nonfinite
    path.write_text(json.dumps(record))
    ctx.reference_record_paths.clear()

    with pytest.raises(records.ReferencePrerequisiteError, match="run.*--prepare-references"):
        records.load_record(ctx, identity)

    assert not ctx.reference_record_paths


@pytest.mark.parametrize("measurements", [None, [], 7, "measurements"])
def test_nonobject_measurements_with_matching_hash_require_onboarding(ctx, identity, measurements):
    path = records.publish_record(ctx, identity, _measurements())
    record = json.loads(path.read_text())
    record["measurements"] = measurements
    record["payload_sha256"] = records._digest(measurements)
    path.write_text(json.dumps(record))
    ctx.reference_record_paths.clear()

    with pytest.raises(records.ReferencePrerequisiteError, match="object.*run.*--prepare-references"):
        records.load_record(ctx, identity)

    assert not ctx.reference_record_paths


def test_changed_workload_identity_does_not_reuse_old_record(ctx, identity):
    path = records.publish_record(ctx, identity, _measurements())
    changed = copy.deepcopy(identity)
    changed["recipe"]["arm"] = "gas4"
    ctx.reference_record_paths.clear()

    with pytest.raises(records.ReferencePrerequisiteError, match="missing.*--prepare-references"):
        records.load_record(ctx, changed)

    assert list(ctx.reference_records_dir.rglob("*.json")) == [path]
    assert not ctx.reference_record_paths


@pytest.mark.parametrize("check_id", [GRADS, OPTIMIZER, E2E])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_nested_measurements_are_not_published(ctx, identity, check_id, nonfinite):
    identity = _identity_for(identity, check_id)
    measurements = _measurements(check_id)
    measurements["diagnostic"] = {"nested": [nonfinite]}

    with pytest.raises(records.ReferencePrerequisiteError, match="nonfinite.*diagnostic.nested"):
        records.publish_record(ctx, identity, measurements)

    assert not ctx.reference_records_dir.exists()


@pytest.mark.parametrize("field", ["loss", "microbatches", "active_tokens"])
def test_incomplete_single_step_measurements_are_rejected(ctx, identity, field):
    measurements = _measurements()
    del measurements[field]

    with pytest.raises(records.ReferencePrerequisiteError, match=f"missing {field}"):
        records.publish_record(ctx, identity, measurements)


@pytest.mark.parametrize("norms", [{}, None, {PARAMETER: -0.1}, {PARAMETER: "0.1"}, {7: 0.1}])
def test_gradient_norms_require_named_nonnegative_scalars(ctx, identity, norms):
    measurements = {**_measurements(), "grad_norms": norms}

    with pytest.raises(records.ReferencePrerequisiteError, match="gradient norms"):
        records.publish_record(ctx, identity, measurements)


@pytest.mark.parametrize("field", ["train_losses", "train_gradient_norms"])
@pytest.mark.parametrize("length", [0, 99, 101])
def test_e2e_requires_exactly_100_losses_and_norms(ctx, identity, field, length):
    identity = _identity_for(identity, E2E)
    measurements = _measurements(E2E)
    measurements[field] = [0.5] * length

    with pytest.raises(records.ReferencePrerequisiteError, match=f"all 100 scalar {field}"):
        records.publish_record(ctx, identity, measurements)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("train_losses", [1.0] * 99 + ["1.0"]),
        ("train_gradient_norms", [0.5] * 99 + [None]),
        ("optimizer_steps", 99),
        ("validation_cross_entropy", None),
        ("scored_answer_tokens", 0),
        ("scored_answer_tokens", 1.5),
    ],
)
def test_incomplete_e2e_scoring_or_steps_are_rejected(ctx, identity, field, invalid):
    identity = _identity_for(identity, E2E)
    measurements = {**_measurements(E2E), field: invalid}

    with pytest.raises(records.ReferencePrerequisiteError):
        records.publish_record(ctx, identity, measurements)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("criterion", "old-criterion"),
        ("status", "unreviewed"),
        ("provenance", {}),
        ("absolute", -1e-3),
        ("absolute", float("inf")),
    ],
)
def test_optimizer_gate_requires_reviewed_finite_norm_criterion(ctx, identity, field, invalid):
    identity = _identity_for(identity, OPTIMIZER)
    measurements = _measurements(OPTIMIZER)
    measurements["optimizer_norm_gate"][field] = invalid

    with pytest.raises(records.ReferencePrerequisiteError):
        records.publish_record(ctx, identity, measurements)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("absolute", True),
        ("absolute", False),
        ("provenance", " "),
        ("provenance", True),
        ("provenance", 7),
        ("provenance", ["unit-review"]),
    ],
)
def test_optimizer_gate_rejects_bool_tolerance_and_invalid_provenance(ctx, identity, field, invalid):
    identity = _identity_for(identity, OPTIMIZER)
    measurements = _measurements(OPTIMIZER)
    measurements["optimizer_norm_gate"][field] = invalid

    with pytest.raises(records.ReferencePrerequisiteError, match="gate|provenance"):
        records.publish_record(ctx, identity, measurements)

    assert not ctx.reference_records_dir.exists()
    assert not ctx.reference_record_paths


def test_optimizer_requires_parameter_norms(ctx, identity):
    identity = _identity_for(identity, OPTIMIZER)
    measurements = _measurements(OPTIMIZER)
    measurements["optimizer_norms"]["parameters"] = {}

    with pytest.raises(records.ReferencePrerequisiteError, match="no parameter norms"):
        records.publish_record(ctx, identity, measurements)


def test_optimizer_gate_does_not_accept_legacy_reviewed_boolean(ctx, identity):
    identity = _identity_for(identity, OPTIMIZER)
    measurements = _measurements(OPTIMIZER)
    del measurements["optimizer_norm_gate"]["status"]
    measurements["optimizer_norm_gate"]["reviewed"] = True

    with pytest.raises(records.ReferencePrerequisiteError, match="review provenance"):
        records.publish_record(ctx, identity, measurements)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("step", 2),
        ("state_keys", ["parameter_update", "exp_avg", "exp_avg_sq"]),
        ("parameter_count", 2),
        ("optimizer_values_count", 2),
        ("parameters", {PARAMETER: {"shape": [2, 3], "exp_avg": 0.05}}),
        (
            "parameters",
            {
                PARAMETER: {
                    "shape": [2, 3],
                    "exp_avg": 0.05,
                    "exp_avg_sq": 0.001,
                    "parameter_update_residual": -0.1,
                }
            },
        ),
    ],
)
def test_optimizer_summary_is_validated_before_publication(ctx, identity, field, invalid):
    identity = _identity_for(identity, OPTIMIZER)
    measurements = _measurements(OPTIMIZER)
    measurements["optimizer_norms"][field] = invalid

    with pytest.raises(ValueError, match="optimizer norm"):
        records.publish_record(ctx, identity, measurements)

    assert not ctx.reference_records_dir.exists()


def test_load_revalidates_measurements_even_when_payload_hash_matches(ctx, identity):
    path = records.publish_record(ctx, identity, _measurements())
    record = json.loads(path.read_text())
    record["measurements"]["grad_norms"] = {}
    record["payload_sha256"] = records._digest(record["measurements"])
    path.write_text(json.dumps(record))

    with pytest.raises(records.ReferencePrerequisiteError, match="gradient norms"):
        records.load_record(ctx, identity)


def test_concurrent_identical_writers_publish_one_complete_record(ctx, identity):
    barrier = Barrier(4)
    measurements = _measurements()

    def publish():
        barrier.wait(timeout=10)
        return records.publish_record(ctx, identity, measurements)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(publish) for _ in range(4)]
        paths = [future.result(timeout=20) for future in futures]

    assert len(set(paths)) == 1
    assert records.load_record(ctx, identity) == measurements
    assert list(paths[0].parent.iterdir()) == [paths[0]]


def test_concurrent_conflicting_writers_do_not_overwrite_winner(ctx, identity):
    barrier = Barrier(2)

    def publish(loss):
        measurements = {**_measurements(), "loss": loss}
        barrier.wait(timeout=10)
        try:
            return "published", loss, records.publish_record(ctx, identity, measurements)
        except records.ReferencePrerequisiteError as error:
            return "conflict", loss, str(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(publish, loss) for loss in (1.25, 2.5)]
        results = [future.result(timeout=20) for future in futures]

    winner = next(result for result in results if result[0] == "published")
    conflict = next(result for result in results if result[0] == "conflict")
    assert "conflicting immutable" in conflict[2]
    assert records.load_record(ctx, identity)["loss"] == winner[1]
    assert list(winner[2].parent.iterdir()) == [winner[2]]


def test_publication_is_visible_only_after_complete_bytes_are_synced(ctx, identity, monkeypatch):
    measurements = _measurements()
    original_fsync = records.os.fsync
    original_link = records.os.link
    synced = []
    observed = []

    def fsync(fd):
        original_fsync(fd)
        synced.append(fd)

    def link(source, destination):
        assert synced
        assert not destination.exists()
        with open(source, "rb") as stream:
            pending = json.load(stream)
        assert pending["measurements"] == measurements
        assert pending["payload_sha256"] == records._digest(measurements)
        original_link(source, destination)
        observed.append(records.load_record(ctx, identity))

    monkeypatch.setattr(records.os, "fsync", fsync)
    monkeypatch.setattr(records.os, "link", link)

    path = records.publish_record(ctx, identity, measurements)

    assert observed == [measurements]
    assert not list(path.parent.glob(".prepare-*"))


@pytest.mark.parametrize("failure", ["fsync", "link"])
def test_failed_publication_removes_temporary_files(ctx, identity, monkeypatch, failure):
    def fail(*args, **kwargs):
        raise OSError("unit publication failure")

    monkeypatch.setattr(records.os, failure, fail)

    with pytest.raises(OSError, match="unit publication failure"):
        records.publish_record(ctx, identity, _measurements())

    assert not list(ctx.reference_records_dir.rglob("*.json"))
    assert not list(ctx.reference_records_dir.rglob(".prepare-*"))
    assert not ctx.reference_record_paths


def test_byte_cap_is_checked_before_publication_creates_files(ctx, identity, monkeypatch):
    monkeypatch.setattr(records, "MAX_RECORD_BYTES", 1)

    with pytest.raises(records.ReferencePrerequisiteError, match="size limit"):
        records.publish_record(ctx, identity, _measurements())

    assert not ctx.reference_records_dir.exists()
    assert not ctx.reference_record_paths


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("config_id", "../outside"),
        ("config_id", "."),
        ("config_id", ".."),
        ("config_id", "/outside"),
        ("config_id", "nested/config"),
        ("config_id", "nested\\config"),
        ("config_id", ""),
        ("config_id", None),
        ("check_id", "../outside"),
        ("check_id", "/outside"),
        ("check_id", "checkpoint-resume-loss"),
    ],
)
@pytest.mark.parametrize("operation", ["publish", "load"])
def test_invalid_record_path_identity_is_rejected_before_filesystem_writes(ctx, identity, field, invalid, operation):
    changed = {**identity, field: invalid}

    with pytest.raises(records.ReferencePrerequisiteError, match="invalid reference record"):
        if operation == "publish":
            records.publish_record(ctx, changed, _measurements())
        else:
            records.load_record(ctx, changed)

    assert not ctx.reference_records_dir.exists()
    assert not ctx.reference_record_paths


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / ".git").mkdir()
    monkeypatch.setattr(
        records, "__file__", str(root / "arctic_platform" / "correctness" / "harness" / "reference_records.py")
    )
    return root


def test_git_records_directory_accepts_checkout_path_without_creating_it(checkout):
    destination = checkout / "arctic_platform" / "correctness" / "reference_data"
    assert records.require_git_records_dir(destination) == destination
    assert not destination.exists()


@pytest.mark.parametrize("relative", ["../outside", "../checkout-sibling", ".git", ".git/objects"])
def test_git_records_directory_rejects_outside_checkout_and_git_metadata(checkout, relative):
    with pytest.raises(records.ReferencePrerequisiteError, match="Git checkout"):
        records.require_git_records_dir(checkout / relative)


@pytest.mark.parametrize("target", ["outside", "git"])
def test_git_records_directory_rejects_symlink_escape(checkout, target):
    destination = checkout.parent / "outside" if target == "outside" else checkout / ".git"
    destination.mkdir(exist_ok=True)
    alias = checkout / "records"
    alias.symlink_to(destination, target_is_directory=True)

    with pytest.raises(records.ReferencePrerequisiteError, match="Git checkout"):
        records.require_git_records_dir(alias / "new-references")

    assert not (destination / "new-references").exists()


def test_exact_byte_cap_loads_but_one_byte_over_cap_fails(ctx, identity, monkeypatch):
    measurements = _measurements()
    path = records.publish_record(ctx, identity, measurements)
    size = path.stat().st_size
    monkeypatch.setattr(records, "MAX_RECORD_BYTES", size)
    assert records.publish_record(ctx, identity, measurements) == path
    assert records.load_record(ctx, identity) == measurements
    monkeypatch.setattr(records, "MAX_RECORD_BYTES", size - 1)
    ctx.reference_record_paths.clear()

    with pytest.raises(records.ReferencePrerequisiteError, match="oversized"):
        records.load_record(ctx, identity)

    assert not ctx.reference_record_paths


def test_workload_identity_binds_inputs_config_and_gate_not_local_record_root(ctx, identity):
    recipe = copy.deepcopy(identity["recipe"])
    same = records.workload_identity(ctx, GRADS, recipe)
    relocated = copy.deepcopy(ctx)
    relocated.reference_records_dir = ctx.reference_records_dir / "another-machine"
    relocated.spec.config_path = "/another-machine/unit-lora.config"
    assert records.workload_identity(relocated, GRADS, recipe) == same
    changed = copy.deepcopy(recipe)
    changed["batch"]["sha256"] = "e" * 64
    assert records.workload_identity(ctx, GRADS, changed) != same
    ctx.spec.config_checksum = "f" * 64
    assert records.workload_identity(ctx, GRADS, recipe) != same
    ctx.spec.config_checksum = identity["config_checksum"]
    ctx.spec.test_tolerances[GRADS].absolute = 2e-3
    assert records.workload_identity(ctx, GRADS, recipe) != same


def test_workload_identity_supports_missing_optional_dependencies(ctx, identity, monkeypatch):
    import torch

    def absent(package):
        raise records.importlib.metadata.PackageNotFoundError(package)

    monkeypatch.setattr(records.importlib.metadata, "version", absent)
    result = records.workload_identity(ctx, GRADS, identity["recipe"])
    assert result["dependencies"]
    assert result["dependencies"]["torch"] == str(torch.__version__)
    assert all(value is None for package, value in result["dependencies"].items() if package != "torch")


def test_workload_identity_uses_loaded_torch_build_not_distribution_metadata(ctx, identity, monkeypatch):
    import torch

    monkeypatch.setattr(torch, "__version__", "2.13.0+cu130")
    monkeypatch.setattr(
        records.importlib.metadata, "version", lambda package: "2.13.0" if package == "torch" else "unit-version"
    )
    without_suffix = records.workload_identity(ctx, E2E, identity["recipe"])
    monkeypatch.setattr(
        records.importlib.metadata,
        "version",
        lambda package: "2.13.0+cu130" if package == "torch" else "unit-version",
    )
    with_suffix = records.workload_identity(ctx, E2E, identity["recipe"])

    assert without_suffix == with_suffix
    assert with_suffix["dependencies"]["torch"] == "2.13.0+cu130"

    monkeypatch.setattr(torch, "__version__", "2.13.0+cu128")
    assert records.workload_identity(ctx, E2E, identity["recipe"]) != with_suffix


def test_metadata_only_torch_drift_reuses_record_without_hf(ctx, identity, monkeypatch):
    import torch

    monkeypatch.setattr(torch, "__version__", "2.13.0+cu130")
    monkeypatch.setattr(
        records.importlib.metadata,
        "version",
        lambda package: "2.13.0+cu130" if package == "torch" else "unit-version",
    )
    prepared = records.workload_identity(ctx, E2E, identity["recipe"])
    measurements = _measurements(E2E)
    records.publish_record(ctx, prepared, measurements)
    monkeypatch.setattr(
        records.importlib.metadata, "version", lambda package: "2.13.0" if package == "torch" else "unit-version"
    )
    _no_hf(monkeypatch)
    target = records.workload_identity(ctx, E2E, identity["recipe"])
    assert records.load_record(ctx, target) == measurements


def test_workload_identity_rejects_non_reference_check(ctx):
    with pytest.raises(ValueError, match="not an HF reference check"):
        records.workload_identity(ctx, "checkpoint-resume-loss", {})


def test_tensor_semantic_hash_is_independent_of_torch_save_archive(tmp_path, monkeypatch):
    import torch

    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    torch.save(tensor, first)
    torch.save(tensor.clone(), second)
    assert hashlib.sha256(first.read_bytes()).digest() != hashlib.sha256(second.read_bytes()).digest()
    first_tensor = torch.load(first, map_location="cpu", weights_only=True)
    second_tensor = torch.load(second, map_location="cpu", weights_only=True)

    def forbidden_save(*args, **kwargs):
        raise AssertionError("semantic tensor hashing must not serialize a torch archive")

    monkeypatch.setattr(torch, "save", forbidden_save)
    assert records.tensor_identity(first_tensor) == records.tensor_identity(second_tensor)


def test_tensor_semantic_hash_does_not_consume_or_depend_on_global_rng():
    import torch

    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        before = torch.random.get_rng_state().clone()
        identity = records.tensor_identity(tensor)
        assert torch.equal(torch.random.get_rng_state(), before)
        torch.rand(32)
        advanced = torch.random.get_rng_state().clone()
        assert records.tensor_identity(tensor) == identity
        assert torch.equal(torch.random.get_rng_state(), advanced)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "int64"])
def test_tensor_semantic_hash_normalizes_storage_layout_and_detaches(dtype):
    import torch

    tensor = torch.arange(6, dtype=getattr(torch, dtype)).reshape(2, 3)
    noncontiguous = tensor.t().contiguous().t()
    assert not noncontiguous.is_contiguous()
    assert records.tensor_identity(noncontiguous) == records.tensor_identity(tensor)
    if tensor.is_floating_point():
        tensor.requires_grad_()
        assert records.tensor_identity(tensor)["dtype"] == f"torch.{dtype}"
        assert tensor.grad is None


def test_tensor_semantic_hash_binds_shape_dtype_and_values():
    import torch

    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    identity = records.tensor_identity(tensor)
    assert identity["shape"] == [2, 3]
    assert identity["dtype"] == "torch.float32"
    assert len(identity["sha256"]) == 64
    assert records.tensor_identity(tensor.reshape(6))["sha256"] != identity["sha256"]
    assert records.tensor_identity(tensor.view(torch.int32))["sha256"] != identity["sha256"]
    changed = tensor.clone()
    changed[0, 0] = 0.5
    assert records.tensor_identity(changed)["sha256"] != identity["sha256"]


@pytest.mark.parametrize("field", ["input_ids", "position_ids", "labels"])
def test_batch_identity_binds_all_token_semantics(field):
    import torch

    batch = SimpleNamespace(
        name="train-step-001",
        active_tokens=2,
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.int64),
        position_ids=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        labels=torch.tensor([[-100, 2, 3]], dtype=torch.int64),
    )
    identity = records.batch_identity(batch)
    assert identity["name"] == batch.name
    assert identity["active_tokens"] == 2
    assert set(identity["tensors"]) == {"input_ids", "position_ids", "labels"}
    changed = copy.deepcopy(batch)
    getattr(changed, field)[0, 1] += 1
    assert records.batch_identity(changed)["tensors"][field] != identity["tensors"][field]


def _model_assets(root):
    root.mkdir()
    (root / "config.json").write_text('{"num_hidden_layers":4}\n')
    (root / "model.safetensors").write_bytes(b"unit model weights A")
    return root


def test_model_identity_detects_same_size_content_drift_with_restored_mtime(tmp_path):
    root = _model_assets(tmp_path / "model")
    before = records.model_identity(root)
    weights = root / "model.safetensors"
    old_stat = weights.stat()
    # Separate ctime ticks so filesystem timestamp granularity cannot mask the edit.
    time.sleep(0.02)
    weights.write_bytes(b"unit model weights B")
    os.utime(weights, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    after = records.model_identity(root)

    assert weights.stat().st_size == old_stat.st_size
    assert weights.stat().st_mtime_ns == old_stat.st_mtime_ns
    assert before["assets"]["model.safetensors"]["sha256"] != after["assets"]["model.safetensors"]["sha256"]
    assert before["sha256"] != after["sha256"]


def test_model_identity_is_independent_of_asset_root(tmp_path):
    first = _model_assets(tmp_path / "first")
    second = _model_assets(tmp_path / "second")
    assert records.model_identity(first) == records.model_identity(second)


@pytest.mark.parametrize("missing", ["directory", "config.json", "model.safetensors"])
def test_model_identity_requires_complete_external_assets(tmp_path, missing):
    root = tmp_path / "model"
    if missing != "directory":
        _model_assets(root)
        (root / missing).unlink()

    with pytest.raises(records.ReferencePrerequisiteError, match="assets unavailable|complete model"):
        records.model_identity(root)
