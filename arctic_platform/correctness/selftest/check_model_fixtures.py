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

"""Static correctness selftests: model fixtures."""


from __future__ import annotations

import hashlib
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from arctic_platform.client.config import CortexConfig as _local_cortex_fixture_CortexConfig
from arctic_platform.correctness.harness import cortex as cortex_fixture
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness.config import ArmDefinition
from arctic_platform.correctness.harness.config import TestTolerance as Tolerance
from arctic_platform.correctness.harness.config import hash_directory
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.registry import TestResult as Result

from .check_catalog_configuration import _applicable_tests_spec
from .helpers import _synthetic_config_spec

_cortex_fixture_STAGE = "@FIXTURE_DB.FIXTURE_SCHEMA.REVIEWED_MODELS"


def _cortex_fixture_module():
    from arctic_platform.correctness.harness import cortex as cortex_fixture

    return cortex_fixture


def _cortex_fixture_fixture(tmp_path, *, legacy=False):
    spec = _applicable_tests_spec(["single-step-grads"])
    root = tmp_path / "reviewed"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"num_hidden_layers": 1, "vocab_size": 128}))
    (root / "generation_config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"reviewed-weight-bytes")
    spec.model.content_hash = hash_directory(root)
    (root / "tokenizer.json").write_text('{"version": "1.0"}')
    (root / "tokenizer_config.json").write_text("{}")
    (root / "merges.txt").write_text("a b\n")
    spec.model.cache_path = str(root)
    if not legacy:
        spec.model.content_hash = hash_directory(root)
    return (spec, root)


def _cortex_fixture_uploaded(module, staging, binding, *, stage=_cortex_fixture_STAGE):
    manifest = {
        "schema_version": 1,
        "model_name": binding["model_name"],
        "stage_path": f"{stage}/{binding['model_name']}",
        "files": binding["files"],
    }
    (staging / "_neutrino_manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize("unsafe", ["/tmp", "/code", "/data-fast-shadow", "/data-fast", "symlink"])
def test_cortex_fixture_fixture_requires_data_fast_before_hashing_or_staging(tmp_path, monkeypatch, unsafe):
    module = _cortex_fixture_module()
    spec, _ = _cortex_fixture_fixture(tmp_path)
    destination = Path(unsafe)
    if unsafe == "symlink":
        destination = tmp_path / "escape"
        destination.symlink_to("/tmp", target_is_directory=True)
        destination /= "output"
    monkeypatch.setattr(
        module, "describe_fixture", lambda *args, **kwargs: pytest.fail("unsafe staging reached source hashing")
    )
    with pytest.raises(ValueError, match="/data-fast"):
        module.prepare_fixture(spec, destination)
    assert not list(tmp_path.glob("ap-cortex-fixture-*"))


def test_cortex_fixture_preparation_preserves_reviewed_tree_and_uses_separate_content_bound_id(tmp_path):
    module = _cortex_fixture_module()
    spec, root = _cortex_fixture_fixture(tmp_path)
    staging, binding = module.prepare_fixture(spec, tmp_path / "staging")
    assert staging != root
    assert binding["model_name"].startswith("ap-correctness/")
    assert binding["model_name"] != spec.model.source_checkpoint
    assert binding["reviewed_hash"] == spec.model.content_hash
    assert not (root / "_neutrino_manifest.json").exists()
    assert {item["path"] for item in binding["files"]} == {p.name for p in root.iterdir()}
    assert (staging / "model.safetensors").read_bytes() == (root / "model.safetensors").read_bytes()


@pytest.mark.parametrize("extra_sidecar", [None, "special_tokens_map.json", "added_tokens.json"])
def test_cortex_fixture_legacy_hash_accepts_only_tokenizer_sidecar_additions_and_binds_all_files(
    tmp_path, extra_sidecar
):
    module = _cortex_fixture_module()
    spec, root = _cortex_fixture_fixture(tmp_path, legacy=True)
    if extra_sidecar:
        (root / extra_sidecar).write_text("{}")
    assert hash_directory(root) != spec.model.content_hash
    staging, binding = module.prepare_fixture(spec, tmp_path / "staging")
    assert binding["reviewed_hash"] == spec.model.content_hash
    assert {item["path"] for item in binding["files"]} == {path.name for path in root.iterdir()}
    assert all((len(item["sha256"]) == 64 for item in binding["files"]))
    binding["publication"] = module.validate_publication(_cortex_fixture_uploaded(module, staging, binding), binding)
    receipt = tmp_path / "binding.json"
    receipt.write_text(json.dumps(binding))
    assert module.load_fixture(receipt, spec) == binding


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("changed", ["config.json", "generation_config.json", "model.safetensors"])
def test_cortex_fixture_reviewed_hash_rejects_size_changes_to_non_tokenizer_assets(tmp_path, legacy, changed):
    module = _cortex_fixture_module()
    spec, root = _cortex_fixture_fixture(tmp_path, legacy=legacy)
    path = root / changed
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="file-name/size hash"):
        module.describe_fixture(spec, root)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "added",
    [
        "extra.safetensors",
        "model.safetensors.index.json",
        "extra_config.json",
        "tokenizer_extra.json",
        "chat_template.json",
    ],
)
def test_cortex_fixture_reviewed_hash_rejects_unreviewed_non_tokenizer_files(tmp_path, legacy, added):
    module = _cortex_fixture_module()
    spec, root = _cortex_fixture_fixture(tmp_path, legacy=legacy)
    (root / added).write_text("{}")
    with pytest.raises(ValueError, match="file-name/size hash"):
        module.describe_fixture(spec, root)


@pytest.mark.parametrize("same_size", [False, True])
@pytest.mark.parametrize(
    "changed", ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"]
)
def test_cortex_fixture_legacy_binding_rejects_tokenizer_drift_after_publication(tmp_path, same_size, changed):
    module = _cortex_fixture_module()
    spec, root = _cortex_fixture_fixture(tmp_path, legacy=True)
    path = root / changed
    if not path.exists():
        path.write_text("{}")
    staging, binding = module.prepare_fixture(spec, tmp_path / "staging")
    binding["publication"] = module.validate_publication(_cortex_fixture_uploaded(module, staging, binding), binding)
    receipt = tmp_path / "binding.json"
    receipt.write_text(json.dumps(binding))
    path.write_bytes(b"x" * path.stat().st_size if same_size else path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="no longer match the published fixture"):
        module.load_fixture(receipt, spec)


def test_cortex_fixture_equal_size_weight_changes_produce_different_remote_identity(tmp_path):
    module = _cortex_fixture_module()
    spec, root = _cortex_fixture_fixture(tmp_path)
    _, first = module.prepare_fixture(spec, tmp_path / "staging")
    weights = root / "model.safetensors"
    weights.write_bytes(b"x" * weights.stat().st_size)
    assert hash_directory(root) == spec.model.content_hash
    _, second = module.prepare_fixture(spec, tmp_path / "staging")
    assert first["model_name"] != second["model_name"]


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "changed", ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "merges.txt"]
)
def test_cortex_fixture_binding_rejects_modified_reference_assets_even_with_same_size(tmp_path, legacy, changed):
    module = _cortex_fixture_module()
    spec, root = _cortex_fixture_fixture(tmp_path, legacy=legacy)
    staging, binding = module.prepare_fixture(spec, tmp_path / "staging")
    binding["publication"] = module.validate_publication(_cortex_fixture_uploaded(module, staging, binding), binding)
    receipt = tmp_path / "binding.json"
    receipt.write_text(json.dumps(binding))
    path = root / changed
    original_size = path.stat().st_size
    if changed == "config.json":
        config = json.loads(path.read_text())
        config["vocab_size"] += 1
        path.write_text(json.dumps(config))
    else:
        path.write_bytes(b"x" * original_size)
    assert path.stat().st_size == original_size
    with pytest.raises(ValueError, match="reference files"):
        module.load_fixture(receipt, spec)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("missing", ["model.safetensors", "tokenizer_config.json"])
def test_cortex_fixture_compatible_hash_still_requires_weights_and_tokenizer(tmp_path, legacy, missing):
    module = _cortex_fixture_module()
    spec, root = _cortex_fixture_fixture(tmp_path, legacy=legacy)
    (root / missing).unlink()
    if not legacy or missing == "model.safetensors":
        spec.model.content_hash = hash_directory(root)
    with pytest.raises(ValueError, match="requires safetensors weights and tokenizer sidecars"):
        module.describe_fixture(spec, root)


@pytest.mark.parametrize("legacy", [False, True])
def test_cortex_fixture_compatible_hash_still_requires_reviewed_layer_count(tmp_path, legacy):
    module = _cortex_fixture_module()
    spec, root = _cortex_fixture_fixture(tmp_path, legacy=legacy)
    spec.model.num_hidden_layers = 2
    with pytest.raises(ValueError, match="layer count"):
        module.describe_fixture(spec, root)


@pytest.mark.parametrize("field", ["stage_path", "files"])
def test_cortex_fixture_binding_refuses_mismatched_upload_manifest(tmp_path, field):
    module = _cortex_fixture_module()
    spec, _ = _cortex_fixture_fixture(tmp_path)
    staging, binding = module.prepare_fixture(spec, tmp_path / "staging")
    _cortex_fixture_uploaded(module, staging, binding)
    path = staging / "_neutrino_manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = [] if field == "files" else "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="publication"):
        module.validate_publication(manifest, binding)


def test_cortex_fixture_unpublished_binding_cannot_enable_cortex(tmp_path):
    module = _cortex_fixture_module()
    spec, _ = _cortex_fixture_fixture(tmp_path)
    _, binding = module.prepare_fixture(spec, tmp_path / "staging")
    receipt = tmp_path / "binding.json"
    receipt.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="publication"):
        module.load_fixture(receipt, spec)


def test_cortex_fixture_publication_failure_never_writes_receipt_or_manifest_to_reference(tmp_path, monkeypatch):
    from arctic_platform.correctness.onboarding import cortex_fixture_upload as module

    spec, root = _cortex_fixture_fixture(tmp_path)
    commands = []

    def failed(cmd, **kwargs):
        commands.append(cmd)
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(module.subprocess, "run", failed)
    receipt = tmp_path / "binding.json"
    with pytest.raises(subprocess.CalledProcessError):
        module.publish_fixture(
            spec, tmp_path / "staging", Path("upload_model.py"), "python", receipt, stage=_cortex_fixture_STAGE
        )
    assert len(commands) == 1
    assert "--overwrite" not in commands[0]
    assert "--dry-run" not in commands[0]
    assert commands[0][commands[0].index("--local-path") + 1] != str(root)
    assert not receipt.exists()
    assert not (root / "_neutrino_manifest.json").exists()


def test_cortex_fixture_successful_publication_receipt_is_checked_against_reference(tmp_path, monkeypatch):
    module = _cortex_fixture_module()
    from arctic_platform.correctness.onboarding import cortex_fixture_upload as upload

    spec, root = _cortex_fixture_fixture(tmp_path)

    def uploaded(cmd, **kwargs):
        staging = Path(cmd[cmd.index("--local-path") + 1])
        binding = module.describe_fixture(spec, staging)
        _cortex_fixture_uploaded(module, staging, binding)

    monkeypatch.setattr(upload.subprocess, "run", uploaded)
    receipt = tmp_path / "binding.json"
    upload.publish_fixture(
        spec, tmp_path / "staging", Path("upload_model.py"), "python", receipt, stage=_cortex_fixture_STAGE
    )
    binding = module.load_fixture(receipt, spec)
    assert binding["reference_path"] == str(root)
    assert binding["publication"]["stage_path"].startswith(_cortex_fixture_STAGE + "/ap-correctness/")
    assert hash_directory(root) == spec.model.content_hash


def test_cortex_fixture_authentication_is_delegated_to_optional_uploader(tmp_path, monkeypatch):
    from arctic_platform.correctness.onboarding import cortex_fixture_upload as module

    spec, _ = _cortex_fixture_fixture(tmp_path)
    for name in ("SNOWFLAKE_USER", "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_PASSWORD"):
        monkeypatch.delenv(name, raising=False)

    def uploader(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(module.subprocess, "run", uploader)
    with pytest.raises(subprocess.CalledProcessError):
        module.publish_fixture(
            spec,
            tmp_path / "staging",
            Path("upload_model.py"),
            "python",
            tmp_path / "receipt",
            stage=_cortex_fixture_STAGE,
        )
    assert not (tmp_path / "receipt").exists()


def test_cortex_fixture_receipt_cannot_be_written_into_reference_checkpoint(tmp_path, monkeypatch):
    from arctic_platform.correctness.onboarding import cortex_fixture_upload as module

    spec, root = _cortex_fixture_fixture(tmp_path)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("must reject before upload"))
    with pytest.raises(ValueError, match="receipt.*outside"):
        module.publish_fixture(
            spec,
            tmp_path / "staging",
            Path("upload_model.py"),
            "python",
            root / "receipt.json",
            stage=_cortex_fixture_STAGE,
        )


@pytest.mark.parametrize("stage", [_cortex_fixture_STAGE, "@ANOTHER_DB.PUBLIC.SERVICE_MODELS"])
def test_cortex_fixture_binding_accepts_verified_publication_without_upload_credentials(tmp_path, monkeypatch, stage):
    module = _cortex_fixture_module()
    spec, _ = _cortex_fixture_fixture(tmp_path)
    binding = module.describe_fixture(spec, Path(spec.model.cache_path))
    binding["publication"] = {"stage_path": f"{stage}/{binding['model_name']}", "files": binding["files"]}
    receipt = tmp_path / "binding.json"
    receipt.write_text(json.dumps(binding))
    monkeypatch.delenv("SNOWFLAKE_ACCOUNT", raising=False)
    assert module.load_fixture(receipt, spec) == binding


def test_cortex_fixture_published_fixture_enables_only_supported_reference_check():
    _cortex_fixture_module()
    registered = runner.registered_tests()
    binding = {"model_name": "ap-correctness/reviewed"}
    assert set(registered) == {"single-step-grads", "single-step-optimizer"}
    for check in ("single-step-grads", "single-step-optimizer"):
        assert runner.cortex_unavailable_reason(registered[check], binding) is None
        missing = runner.cortex_unavailable_reason(registered[check], None)
        assert missing and "--cortex-fixture" in missing
        assert runner.cortex_unavailable_reason(registered[check]) == missing


def _cortex_fixture_receipt(tmp_path):
    module = _cortex_fixture_module()
    spec, _ = _cortex_fixture_fixture(tmp_path)
    staging, binding = module.prepare_fixture(spec, tmp_path / "staging")
    binding["publication"] = module.validate_publication(_cortex_fixture_uploaded(module, staging, binding), binding)
    receipt = tmp_path / "binding.json"
    receipt.write_text(json.dumps(binding))
    return (spec, receipt, binding)


@pytest.mark.parametrize("include_gas4", [False, True])
def test_cortex_fixture_gradient_runner_loads_prepared_reference_and_routes_only_target_to_cortex(
    tmp_path, monkeypatch, include_gas4
):
    from arctic_platform.client import CortexConfig
    from arctic_platform.correctness.harness import reference_records

    spec, receipt, binding = _cortex_fixture_receipt(tmp_path)
    spec.test_tolerances["single-step-grads"] = Tolerance(absolute=0.01)
    cfg = load_config(_synthetic_config_spec(tmp_path / "inputs")[0])
    backend = CortexConfig(base_url="http://cortex.test")
    events = []

    @contextmanager
    def remote_gateway(workdir, gpus, **kwargs):
        assert kwargs["backend"] is backend
        assert gpus == 4
        events.append("cortex")
        yield "remote-session"

    @contextmanager
    def remote_job(session, payload):
        assert session == "remote-session"
        assert payload["model_name"] == binding["model_name"]
        assert payload["training_config"]["debug"]["gradient_norms_per_param"]
        yield SimpleNamespace()

    def reference(ctx, identity):
        assert identity["check_id"] == "single-step-grads"
        events.append("prepared")
        return {"loss": 1.0, "grad_norms": {"weight": 1.0}, "microbatches": 1, "active_tokens": 1}

    monkeypatch.setattr(runner, "run_reference", lambda *a, **k: pytest.fail("regression must not run HF"))
    monkeypatch.setattr(reference_records, "load_record", reference)
    monkeypatch.setattr(runner, "gateway", remote_gateway)
    monkeypatch.setattr(runner, "running_job", remote_job)
    monkeypatch.setattr(runner, "fwd_bwd_step", lambda *a, **k: {})
    monkeypatch.setattr(runner, "uses_mixer_packing", lambda *a: False)
    monkeypatch.setattr(
        runner.registered_tests()["single-step-grads"].fn,
        "per_arm",
        lambda ctx, arm: Result(
            "single-step-grads", ctx.config_id, Outcome.PASS, summary="routing control", arm=arm.name
        ),
    )
    results = runner.execute(
        cfg,
        spec,
        [ArmDefinition("gas1", 4, 8)] + ([ArmDefinition("gas4", 16, 8)] if include_gas4 else []),
        vocab_size=128,
        select=["single-step-grads"],
        client_backend=backend,
        cortex_fixture=receipt,
        workdir=tmp_path / "run",
    )
    if include_gas4:
        assert [(result.arm, result.outcome) for result in results] == [("gas1", Outcome.PASS), ("gas4", Outcome.PASS)]
        assert events == ["prepared", "prepared", "cortex"]
    else:
        assert [result.outcome for result in results] == [Outcome.PASS]
        assert events == ["prepared", "cortex"]


def test_cortex_fixture_fixture_cli_uses_local_vocab_and_records_published_target(tmp_path, monkeypatch):
    import transformers

    from arctic_platform.correctness.harness import execution as cli

    spec, receipt, binding = _cortex_fixture_receipt(tmp_path)
    captured = []
    metadata = {}
    path = _synthetic_config_spec(tmp_path / "inputs")[0]
    monkeypatch.setattr(cli, "_frozen_spec", lambda cfg, path, arms, **k: (spec, arms))
    monkeypatch.setattr(cli, "_catalog_vocab_size", lambda *a: pytest.fail("must not resolve custom ID on HF Hub"))

    def local_vocab(name, **kwargs):
        assert name == spec.model.cache_path
        return SimpleNamespace(vocab_size=128)

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", local_vocab)

    def execute_fixture(cfg, *args, **kwargs):
        from arctic_platform.correctness.harness.registry import TestOutcome
        from arctic_platform.correctness.harness.registry import TestResult

        captured.append(kwargs)
        return [TestResult("single-step-grads", cfg.config_id, TestOutcome.PASS)]

    monkeypatch.setattr(cli, "execute", execute_fixture)
    monkeypatch.setattr(cli, "_finish", lambda *a, **k: metadata.update(k["execution"]) or 0)
    assert (
        cli.main(
            [
                "run",
                "--backend",
                "cortex",
                "--config",
                str(path),
                "--test",
                "single-step-grads",
                "--cortex-fixture",
                str(receipt),
                "--base-url",
                "http://cortex.test",
                "--out",
                str(tmp_path / "report"),
            ]
        )
        == 0
    )
    assert captured[0]["vocab_size"] == 128
    assert captured[0]["cortex_fixture"] == receipt
    execution = metadata[load_config(path).config_id]
    assert execution["requested_model_name"] == binding["model_name"]
    assert execution["fixture_sha256"] == binding["fixture_sha256"]


def test_cortex_fixture_report_distinguishes_publication_from_worker_byte_verification(tmp_path, monkeypatch):
    from arctic_platform.correctness.harness import report

    spec, _, binding = _cortex_fixture_receipt(tmp_path)
    monkeypatch.setattr(report, "collect", lambda: {"host": "hf-worker"})
    execution = {
        spec.config_id: {
            "backend": "cortex",
            "requested_model_name": binding["model_name"],
            "fixture_sha256": binding["fixture_sha256"],
            "fixture_stage_path": binding["publication"]["stage_path"],
        }
    }
    markdown = report.to_markdown([], {spec.config_id: spec}, execution=execution)
    assert "published reduced model" in markdown
    assert binding["fixture_sha256"] in markdown
    assert "not independently verified" in markdown
    assert "not loaded on Cortex" not in markdown


class _local_cortex_fixture_Session:

    def __init__(self, health, *, status=200):
        self.health = health
        self.status = status
        self.calls = []
        self.trust_env = True
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return SimpleNamespace(status_code=self.status, json=lambda: self.health)

    def post(self, url, **kwargs):
        pytest.fail("existing gateway must not receive an invented fixture verification POST")


def _local_cortex_fixture_session(monkeypatch):
    session = _local_cortex_fixture_Session({"status": "OK"})
    monkeypatch.setattr(requests, "Session", lambda: session)
    return session


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]"])
def test_local_cortex_fixture_local_fixture_binds_local_bytes_with_existing_health_only(tmp_path, monkeypatch, host):
    spec, root = _cortex_fixture_fixture(tmp_path, legacy=True)
    session = _local_cortex_fixture_session(monkeypatch)
    binding = cortex_fixture.load_local_fixture(
        _local_cortex_fixture_CortexConfig(base_url=f"http://{host}:8003/"), spec
    )
    assert binding["model_name"] == str(root.resolve())
    assert binding["distribution"] == "local-gateway"
    assert "publication" not in binding
    assert session.closed
    assert session.trust_env is False
    assert [method for method, _, _ in session.calls] == ["GET"]
    assert session.calls[0][1] == f"http://{host}:8003/health"
    assert all((call[2]["allow_redirects"] is False for call in session.calls))
    assert binding["files"] == [
        {"path": path.name, "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(root.iterdir())
    ]
    assert (
        binding["fixture_sha256"]
        == hashlib.sha256(json.dumps(binding["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:8003",
        "http://qa6.example:8003",
        "http://127.0.0.1:8003/proxy",
        "http://127.0.0.1:8003?target=qa6",
        "http://127.0.0.1:8003#target",
        "http://user:secret@127.0.0.1:8003",
    ],
)
def test_local_cortex_fixture_local_fixture_rejects_nonlocal_or_ambiguous_urls_before_io(tmp_path, monkeypatch, url):
    spec, _ = _cortex_fixture_fixture(tmp_path)
    monkeypatch.setattr(requests, "Session", lambda: pytest.fail("invalid URL reached network"))
    with pytest.raises(ValueError, match="loopback"):
        cortex_fixture.load_local_fixture(_local_cortex_fixture_CortexConfig(base_url=url), spec)


@pytest.mark.parametrize(
    "field,value",
    [("backend", "dss"), ("source_root", "/different/AP/checkout"), ("supports_local_model_paths", False)],
)
def test_local_cortex_fixture_local_fixture_does_not_require_or_attest_backend_identity(
    tmp_path, monkeypatch, field, value
):
    spec, _ = _cortex_fixture_fixture(tmp_path)
    session = _local_cortex_fixture_session(monkeypatch)
    session.health[field] = value
    binding = cortex_fixture.load_local_fixture(
        _local_cortex_fixture_CortexConfig(base_url="http://127.0.0.1:8003"), spec
    )
    assert binding["distribution"] == "local-gateway"
    assert not {"backend", "source_root", "supports_local_model_paths"} & binding.keys()
    assert len(session.calls) == 1
    assert session.closed


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("changed", ["config.json", "generation_config.json", "model.safetensors"])
def test_local_cortex_fixture_local_fixture_rejects_protected_asset_size_changes(
    tmp_path, monkeypatch, legacy, changed
):
    spec, root = _cortex_fixture_fixture(tmp_path, legacy=legacy)
    session = _local_cortex_fixture_session(monkeypatch)
    path = root / changed
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="file-name/size hash"):
        cortex_fixture.load_local_fixture(_local_cortex_fixture_CortexConfig(base_url="http://127.0.0.1:8003"), spec)
    assert session.closed


@pytest.mark.parametrize("added", ["extra.safetensors", "extra_config.json"])
def test_local_cortex_fixture_local_fixture_rejects_added_unreviewed_assets(tmp_path, monkeypatch, added):
    spec, root = _cortex_fixture_fixture(tmp_path, legacy=True)
    _local_cortex_fixture_session(monkeypatch)
    (root / added).write_text("{}")
    with pytest.raises(ValueError, match="file-name/size hash"):
        cortex_fixture.load_local_fixture(_local_cortex_fixture_CortexConfig(base_url="http://127.0.0.1:8003"), spec)


def test_local_cortex_fixture_local_fixture_requires_reviewed_layer_count(tmp_path, monkeypatch):
    spec, root = _cortex_fixture_fixture(tmp_path)
    _local_cortex_fixture_session(monkeypatch)
    path = root / "config.json"
    config = json.loads(path.read_text())
    config["num_hidden_layers"] = 2
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="layer count"):
        cortex_fixture.load_local_fixture(_local_cortex_fixture_CortexConfig(base_url="http://127.0.0.1:8003"), spec)


def test_local_cortex_fixture_local_fixture_requires_tokenizer_even_with_legacy_hash(tmp_path, monkeypatch):
    spec, root = _cortex_fixture_fixture(tmp_path, legacy=True)
    _local_cortex_fixture_session(monkeypatch)
    (root / "tokenizer_config.json").unlink()
    with pytest.raises(ValueError, match="tokenizer sidecars"):
        cortex_fixture.load_local_fixture(_local_cortex_fixture_CortexConfig(base_url="http://127.0.0.1:8003"), spec)


@pytest.mark.parametrize("unsafe", ["/tmp", "/code", "/data-fast-shadow", "/data-fast", "symlink"])
def test_local_cortex_fixture_local_fixture_rejects_unsafe_model_root_before_io(tmp_path, monkeypatch, unsafe):
    spec, _ = _cortex_fixture_fixture(tmp_path)
    root = Path(unsafe)
    if unsafe == "symlink":
        root = tmp_path / "escape"
        root.symlink_to("/tmp", target_is_directory=True)
    spec.model.cache_path = str(root)
    monkeypatch.setattr(requests, "Session", lambda: pytest.fail("unsafe root reached network"))
    monkeypatch.setattr(cortex_fixture, "describe_fixture", lambda *args: pytest.fail("unsafe root reached hashing"))
    with pytest.raises(ValueError, match="/data-fast"):
        cortex_fixture.load_local_fixture(_local_cortex_fixture_CortexConfig(base_url="http://127.0.0.1:8003"), spec)


@pytest.mark.parametrize("credentials", [{"pat": "not-a-real-token"}, {"host": "qa6.example"}])
def test_local_cortex_fixture_local_fixture_rejects_authentication_before_io(tmp_path, monkeypatch, credentials):
    spec, _ = _cortex_fixture_fixture(tmp_path)
    monkeypatch.setattr(requests, "Session", lambda: pytest.fail("credentials reached network"))
    with pytest.raises(ValueError, match="unauthenticated"):
        cortex_fixture.load_local_fixture(
            _local_cortex_fixture_CortexConfig(base_url="http://127.0.0.1:8003", **credentials), spec
        )


@pytest.mark.parametrize("status", [301, 401, 500])
def test_local_cortex_fixture_local_fixture_does_not_follow_redirects_or_authenticate(tmp_path, monkeypatch, status):
    spec, _ = _cortex_fixture_fixture(tmp_path)
    session = _local_cortex_fixture_session(monkeypatch)
    session.status = status
    with pytest.raises(ValueError, match="not healthy"):
        cortex_fixture.load_local_fixture(_local_cortex_fixture_CortexConfig(base_url="http://127.0.0.1:8003"), spec)
    assert len(session.calls) == 1
    assert session.closed


@pytest.mark.parametrize("payload", [None, [], "not an object", {}, {"status": "pending"}])
def test_local_cortex_fixture_local_fixture_rejects_malformed_health(tmp_path, monkeypatch, payload):
    spec, _ = _cortex_fixture_fixture(tmp_path)
    session = _local_cortex_fixture_session(monkeypatch)
    session.health = payload
    with pytest.raises(ValueError, match="not healthy"):
        cortex_fixture.load_local_fixture(_local_cortex_fixture_CortexConfig(base_url="http://127.0.0.1:8003"), spec)


def test_local_cortex_fixture_local_fixture_normalizes_connection_failure(tmp_path, monkeypatch):
    spec, _ = _cortex_fixture_fixture(tmp_path)
    session = _local_cortex_fixture_session(monkeypatch)

    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("gateway unavailable")

    monkeypatch.setattr(session, "get", unavailable)
    with pytest.raises(ValueError, match="verification failed: ConnectionError"):
        cortex_fixture.load_local_fixture(_local_cortex_fixture_CortexConfig(base_url="http://127.0.0.1:8003"), spec)


def test_local_cortex_fixture_execution_transport_stays_on_unauthenticated_loopback(monkeypatch):
    from arctic_platform.client.transports.cortex import CortexTransport
    from arctic_platform.correctness.harness.cortex import CorrectnessCortexTransport

    transport = CorrectnessCortexTransport(_local_cortex_fixture_CortexConfig(base_url="http://127.0.0.1:8003"))
    sent = {}

    async def send(self, method, url, **kwargs):
        assert method == "GET" and url == f"{transport._prefix}/p1"
        sent.update(kwargs)
        return {}

    monkeypatch.setattr(CortexTransport, "_send", send)
    try:
        assert transport._on_loop(transport._send("GET", f"{transport._prefix}/p1")).result() == {}
        session = transport._session
        assert session is not None and session.trust_env is False
        assert "Authorization" not in session.headers
        assert sent["allow_redirects"] is False
    finally:
        transport.shutdown()
    assert session.closed
    assert transport._session is None
