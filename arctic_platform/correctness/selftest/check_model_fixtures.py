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
import math
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import safetensors
import torch
from safetensors.torch import save_file
from transformers.utils import hub

from arctic_platform.client import CortexConfig as _local_cortex_full_27b_CortexConfig
from arctic_platform.client import CortexConfig as _local_cortex_full_flashnext_CortexConfig
from arctic_platform.client import CortexConfig as _local_cortex_full_model_CortexConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.client.config import CortexConfig as _local_cortex_fixture_CortexConfig
from arctic_platform.correctness.harness import cortex as cortex_fixture
from arctic_platform.correctness.harness import execution as cli
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness.config import ArmDefinition
from arctic_platform.correctness.harness.config import TestSpec as Spec
from arctic_platform.correctness.harness.config import TestTolerance as Tolerance
from arctic_platform.correctness.harness.config import hash_directory
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.registry import TestResult as Result
from arctic_platform.correctness.models import original

from .check_catalog_configuration import _applicable_tests_spec
from .check_optimizer_capture import _cortex_optimizer_capture_CONFIG
from .check_optimizer_capture import _cortex_optimizer_capture_SPEC

# Cortex fixture.


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
    for check in ("single-step-grads", "single-step-optimizer", "e2e-sft-train-validate"):
        assert runner.cortex_unavailable_reason(registered[check], binding) is None
        missing = runner.cortex_unavailable_reason(registered[check], None)
        assert missing and "--cortex-fixture" in missing
        assert runner.cortex_unavailable_reason(registered[check]) == missing
    inference = registered["inference-checkpoint-loss"]
    assert runner.cortex_unavailable_reason(inference, binding) is None
    assert runner.cortex_unavailable_reason(inference, None) is None
    assert runner.cortex_unavailable_reason(inference) is None


def _reduced_fixture_mtp_producer_case(tmp_path, *, model_type="qwen3_5_moe_text"):
    from arctic_platform.correctness.harness.dss_driver import build_payload

    spec = _applicable_tests_spec(["single-step-grads"])
    root = tmp_path / f"reviewed-{model_type}"
    root.mkdir()
    config = {
        "model_type": "qwen3_5_moe" if model_type == "qwen3_5_moe_text" else model_type,
        "text_config": {
            "model_type": model_type,
            "num_hidden_layers": 2,
            "mtp_num_hidden_layers": 1,
        },
        "vision_config": {"depth": 2},
    }
    if model_type == "glm5_next_text":
        config["text_config"].pop("mtp_num_hidden_layers")
        config["text_config"]["num_nextn_predict_layers"] = 1
    (root / "config.json").write_text(json.dumps(config))
    save_file({"model.weight": torch.ones(2, 2)}, root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": "model.safetensors"}})
    )
    (root / "tokenizer_config.json").write_text("{}")
    spec.model.source_checkpoint = "Qwen/Qwen3.5-Source"
    spec.model.num_hidden_layers = 2
    spec.model.vision_depth = 2
    spec.model.seed = 0
    spec.model.cache_path = str(root.resolve())
    spec.model.content_hash = hash_directory(root)
    binding = cortex_fixture.describe_fixture(spec, root)
    payload = build_payload(
        {
            "model_provider": "prime_rl",
            "ep_size": 2,
            "prime_rl": {"debug": {"existing_debug": "preserved"}},
        },
        binding["model_name"],
        42,
        attn_implementation="sdpa",
    )
    return spec, binding, payload


@pytest.mark.parametrize("model_type", ["qwen3_5_moe_text", "qwen4_exp_text", "glm5_next_text"])
def test_reduced_fixture_mtp_producer_binds_validated_publication_without_overwriting_debug(tmp_path, model_type):
    from arctic_platform.debug.reduced_fixture_mtp import maybe_bind_reduced_fixture_mtp

    spec, binding, payload = _reduced_fixture_mtp_producer_case(tmp_path, model_type=model_type)

    maybe_bind_reduced_fixture_mtp(payload, spec, binding)

    debug = payload["training_config"]["prime_rl"]["debug"]
    assert debug["existing_debug"] == "preserved"
    assert debug["reduced_fixture_mtp"] == {
        "schema_version": 1,
        "config_id": spec.config_id,
        "fixture_locator": binding["model_name"],
        "reference_path": spec.model.cache_path,
        "reviewed_hash": spec.model.content_hash,
        "fixture_sha256": binding["fixture_sha256"],
        "files": binding["files"],
        "source_checkpoint": spec.model.source_checkpoint,
        "num_hidden_layers": 2,
        "vision_depth": 2,
        "materialization": "pretrained-prefix-reduction",
    }


def test_reduced_fixture_mtp_producer_rejects_preexisting_descriptor(tmp_path):
    from arctic_platform.debug.reduced_fixture_mtp import maybe_bind_reduced_fixture_mtp

    spec, binding, payload = _reduced_fixture_mtp_producer_case(tmp_path)
    original = {"caller_supplied": True}
    payload["training_config"]["prime_rl"]["debug"]["reduced_fixture_mtp"] = original

    with pytest.raises(ValueError, match="must not be supplied or overwritten"):
        maybe_bind_reduced_fixture_mtp(payload, spec, binding)
    assert payload["training_config"]["prime_rl"]["debug"]["reduced_fixture_mtp"] is original


def test_reduced_fixture_mtp_producer_is_noop_during_reference_preparation_without_publication(tmp_path):
    from arctic_platform.debug.reduced_fixture_mtp import maybe_bind_reduced_fixture_mtp

    spec, _, payload = _reduced_fixture_mtp_producer_case(tmp_path)
    before = json.loads(json.dumps(payload))

    maybe_bind_reduced_fixture_mtp(payload, spec, None)

    assert payload == before


def test_reduced_fixture_mtp_producer_is_noop_for_non_qwen_fixture(tmp_path):
    from arctic_platform.debug.reduced_fixture_mtp import maybe_bind_reduced_fixture_mtp

    spec, binding, payload = _reduced_fixture_mtp_producer_case(tmp_path, model_type="llama")
    before = json.loads(json.dumps(payload))

    maybe_bind_reduced_fixture_mtp(payload, spec, binding)

    assert payload == before


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
    cfg = load_config(Path("arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config"))
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
    path = Path("arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config")
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


# Local cortex full 27b.


_local_cortex_full_27b_BASE = Path(__file__).resolve().parents[1]

_local_cortex_full_27b_CONFIG = _local_cortex_full_27b_BASE / "configs/qwen3.8-27b/b200/train-sft-8gpus-2k.config"

_local_cortex_full_27b_SPEC = _local_cortex_full_27b_BASE / "specs/qwen3.8-27b-b200-train-sft-8gpus-2k.json"

_local_cortex_full_27b_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

_local_cortex_full_27b_CONFIG_SHA = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"

_local_cortex_full_27b_INDEX_SHA = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"

_local_cortex_full_27b_SCHEMA_SHA = "9c0eb904ed294a62882adb287b81e36fdb48e5fb8e006ee25b084341ac78fd55"

_local_cortex_full_27b_TENSOR_BYTES = 55562855904


def _local_cortex_full_27b_schema_bytes(schema):
    return json.dumps(schema, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


@pytest.fixture
def _local_cortex_full_27b_full_model(tmp_path, monkeypatch):
    cfg, spec = (load_config(_local_cortex_full_27b_CONFIG), Spec.read(_local_cortex_full_27b_SPEC))
    cache = tmp_path / "hub"
    root = cache / "models--Qwen--Qwen3.8-27B" / "snapshots" / _local_cortex_full_27b_REVISION
    root.mkdir(parents=True)
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {"num_hidden_layers": 64, "hidden_size": 5120, "vocab_size": 248320},
        "vision_config": {"depth": 27},
    }
    schema = {
        **{f"model.language_model.layers.{i}.input_layernorm.weight": ["BF16", [5120]] for i in range(64)},
        **{f"model.visual.blocks.{i}.norm.weight": ["BF16", [4]] for i in range(27)},
        "lm_head.weight": ["BF16", [248320, 5120]],
        "model.language_model.embed_tokens.weight": ["BF16", [248320, 5120]],
        **{f"mtp.synthetic.{i}.weight": ["BF16", [4]] for i in range(1106)},
    }
    used = sum((2 * _local_cortex_full_27b_elements(shape) for _, shape in schema.values()))
    schema["mtp.synthetic.0.weight"][1] = [(_local_cortex_full_27b_TENSOR_BYTES - used) // 2 + 4]
    mapping = {name: f"model-{i % 18 + 1:05d}-of-00018.safetensors" for i, name in enumerate(schema)}
    index = {"metadata": {"total_size": _local_cortex_full_27b_TENSOR_BYTES}, "weight_map": mapping}
    config_bytes = json.dumps(config).encode()
    index_bytes = json.dumps(index).encode()
    (root / "config.json").write_bytes(config_bytes)
    (root / "model.safetensors.index.json").write_bytes(index_bytes)
    for filename in {"tokenizer.json", "tokenizer_config.json", *mapping.values()}:
        (root / filename).write_text("{}")
    pins = {
        config_bytes: _local_cortex_full_27b_CONFIG_SHA,
        index_bytes: _local_cortex_full_27b_INDEX_SHA,
        _local_cortex_full_27b_schema_bytes(schema): _local_cortex_full_27b_SCHEMA_SHA,
    }
    real_sha256 = hashlib.sha256

    def digest(data):
        if data in pins:
            return SimpleNamespace(hexdigest=lambda: pins[data])
        return real_sha256(data)

    monkeypatch.setattr(original, "hashlib", SimpleNamespace(sha256=digest))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("DSS_ALLOW_REMOTE_MODELS", "1")
    calls = []

    def cached_file(model, filename, **kwargs):
        assert model == cfg.model_name and filename == "config.json"
        assert kwargs == {
            "cache_dir": str(cache),
            "revision": _local_cortex_full_27b_REVISION,
            "local_files_only": True,
        }
        return str(root / filename)

    def safe_open(path, **kwargs):
        assert kwargs == {"framework": "pt", "device": "cpu"}
        shard = Path(path).name
        calls.append(shard)

        class Headers:

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def keys(self):
                return [name for name in schema if mapping[name] == shard]

            def get_slice(self, name):
                dtype, shape = schema[name]
                return SimpleNamespace(get_shape=lambda: shape, get_dtype=lambda: dtype)

            def get_tensor(self, name):
                pytest.fail("full-model admission must never load tensor payloads")

        return Headers()

    monkeypatch.setattr(hub, "cached_file", cached_file)
    monkeypatch.setattr(safetensors, "safe_open", safe_open)
    return SimpleNamespace(root=root, cfg=cfg, spec=spec, schema=schema, mapping=mapping, calls=calls)


def _local_cortex_full_27b_elements(shape):
    result = 1
    for size in shape:
        result *= size
    return result


def _local_cortex_full_27b_validate(case):
    return runner.validate_local_full_model(
        _local_cortex_full_27b_CortexConfig(base_url="http://127.0.0.1:8003"), case.cfg, case.spec
    )


@pytest.mark.parametrize("source", ["Qwen/Qwen3.8-27B", "/data-fast/base-models/Qwen/Qwen3.8-27B"])
def test_local_cortex_full_27b_full_27b_original_source_is_admitted_header_only(
    _local_cortex_full_27b_full_model, source
):
    original = (_local_cortex_full_27b_CONFIG.read_bytes(), _local_cortex_full_27b_SPEC.read_bytes())
    _local_cortex_full_27b_full_model.spec.model.source_checkpoint = source
    assert _local_cortex_full_27b_validate(_local_cortex_full_27b_full_model) == _local_cortex_full_27b_full_model.root
    assert len(_local_cortex_full_27b_full_model.calls) == 18
    assert (_local_cortex_full_27b_CONFIG.read_bytes(), _local_cortex_full_27b_SPEC.read_bytes()) == original


@pytest.mark.parametrize(
    "failure",
    [
        "source",
        "config",
        "index",
        "missing-shard",
        "missing-tokenizer",
        "missing-text-layer",
        "missing-vision-layer",
        "vocabulary-shape",
        "same-size-shape",
        "same-size-vision-shape",
        "same-size-mtp-shape",
        "dtype",
        "same-size-dtype",
    ],
)
def test_local_cortex_full_27b_full_27b_invalid_source_is_rejected(_local_cortex_full_27b_full_model, failure):
    root, schema = (_local_cortex_full_27b_full_model.root, _local_cortex_full_27b_full_model.schema)
    if failure == "source":
        _local_cortex_full_27b_full_model.spec.model.source_checkpoint = (
            "/data-fast/base-models/synthetic/Qwen3.8-27B-4L"
        )
    elif failure in {"config", "index"}:
        filename = "config.json" if failure == "config" else "model.safetensors.index.json"
        with (root / filename).open("ab") as stream:
            stream.write(b"\n")
    elif failure == "missing-shard":
        (root / next(iter(_local_cortex_full_27b_full_model.mapping.values()))).unlink()
    elif failure == "missing-tokenizer":
        (root / "tokenizer.json").unlink()
    elif failure in {"missing-text-layer", "missing-vision-layer"}:
        name = (
            "model.language_model.layers.63.input_layernorm.weight"
            if failure == "missing-text-layer"
            else "model.visual.blocks.26.norm.weight"
        )
        del schema[name]
    elif failure == "vocabulary-shape":
        schema["lm_head.weight"][1] = [248319, 5120]
    elif failure == "same-size-shape":
        schema["model.language_model.layers.0.input_layernorm.weight"][1] = [2560, 2]
    elif failure == "same-size-vision-shape":
        schema["model.visual.blocks.0.norm.weight"][1] = [2, 2]
    elif failure == "same-size-mtp-shape":
        schema["mtp.synthetic.1.weight"][1] = [2, 2]
    elif failure == "dtype":
        schema["model.language_model.layers.0.input_layernorm.weight"][0] = "F16"
    else:
        schema["model.language_model.layers.0.input_layernorm.weight"] = ["F32", [2560]]
    with pytest.raises((ValueError, OSError)):
        _local_cortex_full_27b_validate(_local_cortex_full_27b_full_model)


@pytest.mark.parametrize("failure", ["revision", "repository", "cache-escape"])
def test_local_cortex_full_27b_full_27b_snapshot_root_is_pinned(
    _local_cortex_full_27b_full_model, monkeypatch, failure
):
    root = _local_cortex_full_27b_full_model.root
    wrong = {
        "revision": root.with_name("0" * 40),
        "repository": root.parent.parent.with_name("models--other"),
        "cache-escape": Path("/tmp/unapproved-full-model"),
    }[failure]
    monkeypatch.setattr(hub, "cached_file", lambda *args, **kwargs: str(wrong / "config.json"))
    with pytest.raises(ValueError):
        _local_cortex_full_27b_validate(_local_cortex_full_27b_full_model)


def test_local_cortex_full_27b_full_27b_rejects_other_config_before_cache_lookup(
    _local_cortex_full_27b_full_model, monkeypatch
):
    _local_cortex_full_27b_full_model.cfg.config_id = "unreviewed-full-model-config"
    monkeypatch.setattr(hub, "cached_file", lambda *args, **kwargs: pytest.fail("unreviewed source lookup"))
    with pytest.raises(ValueError, match="supported pinned original"):
        _local_cortex_full_27b_validate(_local_cortex_full_27b_full_model)


def test_local_cortex_full_27b_full_27b_shard_symlink_cannot_escape_cache(_local_cortex_full_27b_full_model, tmp_path):
    shard = _local_cortex_full_27b_full_model.root / next(iter(_local_cortex_full_27b_full_model.mapping.values()))
    shard.unlink()
    outside = tmp_path / "outside.safetensors"
    outside.write_text("{}")
    shard.symlink_to(outside)
    with pytest.raises(ValueError, match="outside its cache"):
        _local_cortex_full_27b_validate(_local_cortex_full_27b_full_model)


def test_local_cortex_full_27b_full_27b_inference_dispatch_never_runs_hf_reference(
    _local_cortex_full_27b_full_model, tmp_path, monkeypatch
):
    observed = []

    def measure(ctx):
        observed.append(ctx)
        assert ctx.local_gateway_full_model is True
        assert ctx.local_full_model_path == str(_local_cortex_full_27b_full_model.root)
        assert ctx.target_model_name is None
        assert ctx.cfg.model_name == "Qwen/Qwen3.8-27B"
        assert ctx.vocab_size == 248320
        yield Result(test_id="inference-checkpoint-loss", config_id=ctx.config_id, outcome=Outcome.PASS)

    monkeypatch.setattr(runner.registered_tests()["inference-checkpoint-loss"], "fn", measure)
    monkeypatch.setattr(runner, "run_reference", lambda *args, **kwargs: pytest.fail("no HF reference for inference"))
    results = runner.execute(
        _local_cortex_full_27b_full_model.cfg,
        _local_cortex_full_27b_full_model.spec,
        [],
        vocab_size=248320,
        select=["inference-checkpoint-loss"],
        workdir=tmp_path,
        client_backend=_local_cortex_full_27b_CortexConfig(base_url="http://127.0.0.1:8003"),
        local_gateway_full_model=True,
    )
    assert len(observed) == 1
    assert [result.outcome for result in results] == [Outcome.PASS]


# Local cortex full flashnext.


_local_cortex_full_flashnext_BASE = Path(__file__).resolve().parents[1]

_local_cortex_full_flashnext_CONFIG = (
    _local_cortex_full_flashnext_BASE / "configs/qwen3.8-flash-next/h200/train-sft-full-16gpus-sp16-256k.config"
)

_local_cortex_full_flashnext_SPEC = (
    _local_cortex_full_flashnext_BASE / "specs/qwen3.8-flash-next-h200-train-sft-full-16gpus-sp16-256k.json"
)

_local_cortex_full_flashnext_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"

_local_cortex_full_flashnext_CONFIG_SHA = "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"

_local_cortex_full_flashnext_INDEX_SHA = "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de"

_local_cortex_full_flashnext_SCHEMA_SHA = "1d55a6d3f64b1b2b004302864c0e74e11871ddf9f3481b10fc21fc8b0ec53d26"

_local_cortex_full_flashnext_PAYLOAD_BYTES = 359999963128

_local_cortex_full_flashnext_PLE = "model.language_model.layers.1.ple.ple_embedding."

_local_cortex_full_flashnext_BUFFERS = {
    "layer_multipliers": [3],
    "ngram_heads_offsets": [16],
    "ngram_heads_vocab_sizes": [16],
}


def _local_cortex_full_flashnext_schema_bytes(schema):
    return json.dumps(schema, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


@pytest.fixture
def _local_cortex_full_flashnext_full_model(tmp_path, monkeypatch):
    cfg, spec = (load_config(_local_cortex_full_flashnext_CONFIG), Spec.read(_local_cortex_full_flashnext_SPEC))
    cache = tmp_path / "hub"
    root = cache / "models--Qwen--Qwen3.8-Flash-Next" / "snapshots" / _local_cortex_full_flashnext_REVISION
    root.mkdir(parents=True)
    config = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": {"num_hidden_layers": 48, "hidden_size": 2560, "vocab_size": 248320},
        "vision_config": {"depth": 27},
    }
    schema = {
        **{f"model.language_model.layers.{i}.norm.weight": ["BF16", [2560]] for i in range(48)},
        **{f"model.visual.blocks.{i}.norm.weight": ["BF16", [4]] for i in range(27)},
        "lm_head.weight": ["BF16", [248320, 2560]],
        "model.language_model.embed_tokens.weight": ["BF16", [248320, 2560]],
        **{
            _local_cortex_full_flashnext_PLE + name: ["I64", shape.copy()]
            for name, shape in _local_cortex_full_flashnext_BUFFERS.items()
        },
        **{f"mtp.synthetic.{i}.weight": ["BF16", [4]] for i in range(1578)},
    }
    used = sum((math.prod(shape) * (8 if dtype == "I64" else 2) for dtype, shape in schema.values()))
    schema["mtp.synthetic.0.weight"][1] = [(_local_cortex_full_flashnext_PAYLOAD_BYTES - used) // 2 + 4]
    mapping = {name: f"model-{i % 131 + 1:05d}-of-00131.safetensors" for i, name in enumerate(schema)}
    index = {"metadata": {"total_size": _local_cortex_full_flashnext_PAYLOAD_BYTES}, "weight_map": mapping}
    raw_config, raw_index = (json.dumps(config).encode(), json.dumps(index).encode())
    (root / "config.json").write_bytes(raw_config)
    (root / "model.safetensors.index.json").write_bytes(raw_index)
    for name in {"tokenizer.json", "tokenizer_config.json", *mapping.values()}:
        (root / name).write_text("{}")
    pins = {
        raw_config: _local_cortex_full_flashnext_CONFIG_SHA,
        raw_index: _local_cortex_full_flashnext_INDEX_SHA,
        _local_cortex_full_flashnext_schema_bytes(schema): _local_cortex_full_flashnext_SCHEMA_SHA,
    }
    real_sha = hashlib.sha256

    def digest(raw):
        return SimpleNamespace(hexdigest=lambda: pins[raw]) if raw in pins else real_sha(raw)

    monkeypatch.setattr(original, "hashlib", SimpleNamespace(sha256=digest))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("DSS_ALLOW_REMOTE_MODELS", "1")
    monkeypatch.setattr(hub, "cached_file", lambda *args, **kwargs: str(root / "config.json"))
    calls = []

    def safe_open(path, **kwargs):
        assert kwargs == {"framework": "pt", "device": "cpu"}
        shard = Path(path).name
        calls.append(shard)

        class Headers:

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def keys(self):
                return [name for name in schema if mapping.get(name) == shard]

            def get_slice(self, name):
                dtype, shape = schema[name]
                return SimpleNamespace(get_dtype=lambda: dtype, get_shape=lambda: shape)

            def get_tensor(self, name):
                pytest.fail("source admission must not materialize tensor payloads")

        return Headers()

    monkeypatch.setattr(safetensors, "safe_open", safe_open)
    return SimpleNamespace(root=root, cfg=cfg, spec=spec, schema=schema, mapping=mapping, calls=calls)


def _local_cortex_full_flashnext_validate(case):
    return runner.validate_local_full_model(
        _local_cortex_full_flashnext_CortexConfig(base_url="http://127.0.0.1:8003"), case.cfg, case.spec
    )


@pytest.mark.parametrize("placement", [8, 16])
@pytest.mark.parametrize("source", ["model-id", "pinned-snapshot"])
def test_local_cortex_full_flashnext_original_flashnext_is_admitted_header_only(
    _local_cortex_full_flashnext_full_model, source, placement
):
    config = (
        _local_cortex_full_flashnext_BASE
        / f"configs/qwen3.8-flash-next/h200/train-sft-full-{placement}gpus-sp{placement}-256k.config"
    )
    spec = (
        _local_cortex_full_flashnext_BASE
        / f"specs/qwen3.8-flash-next-h200-train-sft-full-{placement}gpus-sp{placement}-256k.json"
    )
    original = (config.read_bytes(), spec.read_bytes())
    _local_cortex_full_flashnext_full_model.cfg = load_config(config)
    _local_cortex_full_flashnext_full_model.spec = Spec.read(spec)
    if source == "model-id":
        _local_cortex_full_flashnext_full_model.spec.model.source_checkpoint = (
            _local_cortex_full_flashnext_full_model.cfg.model_name
        )
    assert (
        _local_cortex_full_flashnext_validate(_local_cortex_full_flashnext_full_model)
        == _local_cortex_full_flashnext_full_model.root
    )
    assert len(_local_cortex_full_flashnext_full_model.calls) == 131
    assert (config.read_bytes(), spec.read_bytes()) == original


@pytest.mark.parametrize("legacy_cache", ["HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"])
def test_local_cortex_full_flashnext_explicit_cache_and_revision_override_legacy_defaults(
    _local_cortex_full_flashnext_full_model, tmp_path, monkeypatch, legacy_cache
):
    case = _local_cortex_full_flashnext_full_model
    monkeypatch.setenv(legacy_cache, str(tmp_path / "unrelated-cache"))
    observed = []

    def cached_file(model, filename, **kwargs):
        observed.append(kwargs)
        return hub.cached_files(model, [filename], **kwargs)[0]

    monkeypatch.setattr(hub, "cached_file", cached_file)
    assert _local_cortex_full_flashnext_validate(case) == case.root
    assert observed == [
        {
            "cache_dir": str(case.root.parents[2]),
            "revision": _local_cortex_full_flashnext_REVISION,
            "local_files_only": True,
        }
    ]


@pytest.mark.parametrize("buffer", sorted(_local_cortex_full_flashnext_BUFFERS))
@pytest.mark.parametrize("failure", ["missing", "dtype", "shape"])
def test_local_cortex_full_flashnext_every_original_integer_buffer_is_pinned(
    _local_cortex_full_flashnext_full_model, buffer, failure
):
    name = _local_cortex_full_flashnext_PLE + buffer
    if failure == "missing":
        _local_cortex_full_flashnext_full_model.schema.pop(name)
    elif failure == "dtype":
        _local_cortex_full_flashnext_full_model.schema[name][0] = "BF16"
    else:
        _local_cortex_full_flashnext_full_model.schema[name][1] = [
            1,
            *_local_cortex_full_flashnext_full_model.schema[name][1],
        ]
    with pytest.raises(ValueError):
        _local_cortex_full_flashnext_validate(_local_cortex_full_flashnext_full_model)


@pytest.mark.parametrize("placement", [8, 16])
@pytest.mark.parametrize(
    "failure",
    [
        "source",
        "config-id",
        "model-id",
        "config",
        "index",
        "missing-shard",
        "tokenizer",
        "text-layer",
        "vision-layer",
        "vocab",
        "shape",
        "dtype",
        "same-byte-dtype",
    ],
)
def test_local_cortex_full_flashnext_original_flashnext_rejects_drift(
    _local_cortex_full_flashnext_full_model, failure, placement
):
    case = _local_cortex_full_flashnext_full_model
    case.cfg = load_config(
        _local_cortex_full_flashnext_BASE
        / f"configs/qwen3.8-flash-next/h200/train-sft-full-{placement}gpus-sp{placement}-256k.config"
    )
    case.spec = Spec.read(
        _local_cortex_full_flashnext_BASE
        / f"specs/qwen3.8-flash-next-h200-train-sft-full-{placement}gpus-sp{placement}-256k.json"
    )
    if failure == "source":
        case.spec.model.source_checkpoint = "/data-fast/base-models/synthetic/FlashNext-4L"
    elif failure == "config-id":
        case.cfg.config_id = "unreviewed-flashnext"
    elif failure == "model-id":
        case.cfg.sub_job["model_name"] = "Qwen/other"
    elif failure in {"config", "index"}:
        name = "config.json" if failure == "config" else "model.safetensors.index.json"
        with (case.root / name).open("ab") as stream:
            stream.write(b"\n")
    elif failure in {"missing-shard", "tokenizer"}:
        name = next(iter(case.mapping.values())) if failure == "missing-shard" else "tokenizer.json"
        (case.root / name).rename(case.root / (name + ".retained"))
    elif failure in {"text-layer", "vision-layer"}:
        name = (
            "model.language_model.layers.47.norm.weight"
            if failure == "text-layer"
            else "model.visual.blocks.26.norm.weight"
        )
        case.schema.pop(name)
    elif failure == "vocab":
        case.schema["lm_head.weight"][1] = [248319, 2560]
    elif failure == "shape":
        case.schema["model.language_model.layers.0.norm.weight"][1] = [1280, 2]
    elif failure == "dtype":
        case.schema["model.language_model.layers.0.norm.weight"][0] = "F32"
    else:
        case.schema["model.language_model.layers.0.norm.weight"] = ["I64", [640]]
    with pytest.raises((ValueError, OSError)):
        _local_cortex_full_flashnext_validate(case)


@pytest.mark.parametrize("failure", ["revision", "repository", "offline", "remote"])
def test_local_cortex_full_flashnext_original_flashnext_rejects_wrong_cache_or_online_mode(
    _local_cortex_full_flashnext_full_model, monkeypatch, failure
):
    if failure in {"revision", "repository"}:
        root = _local_cortex_full_flashnext_full_model.root
        wrong = root.with_name("0" * 40) if failure == "revision" else root.parent.parent.with_name("other")
        monkeypatch.setattr(hub, "cached_file", lambda *args, **kwargs: str(wrong / "config.json"))
    else:
        monkeypatch.setenv("HF_HUB_OFFLINE" if failure == "offline" else "DSS_ALLOW_REMOTE_MODELS", "0")
    with pytest.raises(ValueError):
        _local_cortex_full_flashnext_validate(_local_cortex_full_flashnext_full_model)


def test_local_cortex_full_flashnext_original_flashnext_inference_dispatch_never_runs_hf(
    _local_cortex_full_flashnext_full_model, tmp_path, monkeypatch
):
    observed = []
    _local_cortex_full_flashnext_full_model.spec.applicable_tests.append("inference-checkpoint-loss")

    def measure(ctx):
        assert ctx.local_gateway_full_model is True
        assert ctx.local_full_model_path == str(_local_cortex_full_flashnext_full_model.root)
        observed.append(ctx)
        yield Result(test_id="inference-checkpoint-loss", config_id=ctx.config_id, outcome=Outcome.PASS)

    monkeypatch.setattr(runner.registered_tests()["inference-checkpoint-loss"], "fn", measure)
    monkeypatch.setattr(runner, "run_reference", lambda *args, **kwargs: pytest.fail("no inference HF reference"))
    results = runner.execute(
        _local_cortex_full_flashnext_full_model.cfg,
        _local_cortex_full_flashnext_full_model.spec,
        [],
        vocab_size=248320,
        select=["inference-checkpoint-loss"],
        workdir=tmp_path,
        client_backend=_local_cortex_full_flashnext_CortexConfig(base_url="http://127.0.0.1:8003"),
        local_gateway_full_model=True,
    )
    assert len(observed) == 1
    assert [result.outcome for result in results] == [Outcome.PASS]


def test_local_cortex_full_flashnext_original_flashnext_source_admission_does_not_change_applicability(
    _local_cortex_full_flashnext_full_model, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        runner.registered_tests()["inference-checkpoint-loss"],
        "fn",
        lambda *args: pytest.fail("source admission cannot make a check applicable"),
    )
    monkeypatch.setattr(runner, "run_reference", lambda *args, **kwargs: pytest.fail("no inference HF reference"))
    results = runner.execute(
        _local_cortex_full_flashnext_full_model.cfg,
        _local_cortex_full_flashnext_full_model.spec,
        [],
        vocab_size=248320,
        select=["inference-checkpoint-loss"],
        workdir=tmp_path,
        client_backend=_local_cortex_full_flashnext_CortexConfig(base_url="http://127.0.0.1:8003"),
        local_gateway_full_model=True,
    )
    assert [result.outcome for result in results] == [Outcome.INAPPLICABLE]


# Local cortex full model.


@pytest.fixture
def _local_cortex_full_model_offline_model(tmp_path, monkeypatch):
    from transformers.utils import hub

    cfg, spec = (load_config(_cortex_optimizer_capture_CONFIG), Spec.read(_cortex_optimizer_capture_SPEC))
    root = tmp_path / "hub" / "models--Qwen--Qwen3-8B" / "snapshots" / "b968826d9c46dd6066d109eabc6255188de91218"
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "attention_bias": False,
                "attention_dropout": 0.0,
                "bos_token_id": 151643,
                "eos_token_id": 151645,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 4096,
                "initializer_range": 0.02,
                "intermediate_size": 12288,
                "max_position_embeddings": 40960,
                "max_window_layers": 36,
                "model_type": "qwen3",
                "num_attention_heads": 32,
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-06,
                "rope_scaling": None,
                "rope_theta": 1000000,
                "sliding_window": None,
                "tie_word_embeddings": False,
                "torch_dtype": "bfloat16",
                "transformers_version": "4.51.0",
                "use_cache": True,
                "use_sliding_window": False,
                "vocab_size": 151936,
            },
            indent=2,
        )
    )
    (root / "tokenizer_config.json").write_text("{}")
    (root / "tokenizer.json").write_text("{}")
    tensors = {f"model.layers.{i}.weight": torch.ones(1) for i in range(36)}
    save_file(tensors, root / "model-00001-of-00001.safetensors")
    index = {"weight_map": {name: "model-00001-of-00001.safetensors" for name in tensors}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("DSS_ALLOW_REMOTE_MODELS", "1")
    calls = []

    def cached_file(name, filename, **kwargs):
        assert name == cfg.model_name
        assert filename == "config.json"
        assert kwargs["local_files_only"] is True
        calls.append(name)
        return str(root / filename)

    monkeypatch.setattr(hub, "cached_file", cached_file)
    monkeypatch.setattr(requests.Session, "request", lambda *args, **kwargs: pytest.fail("no network"))
    return SimpleNamespace(root=root, cfg=cfg, spec=spec, calls=calls, index=index)


def _local_cortex_full_model_execute(case, tmp_path, **kwargs):
    return runner.execute(
        case.cfg,
        case.spec,
        [],
        vocab_size=128,
        select=["e2e-sft-train-validate"],
        workdir=tmp_path,
        client_backend=_local_cortex_full_model_CortexConfig(base_url="http://127.0.0.1:8003"),
        local_gateway_full_model=True,
        **kwargs,
    )


def test_local_cortex_full_model_full_runner_preserves_catalog_target_and_spec_with_distinct_reference(
    _local_cortex_full_model_offline_model, tmp_path, monkeypatch
):
    original = (_cortex_optimizer_capture_CONFIG.read_bytes(), _cortex_optimizer_capture_SPEC.read_bytes())
    observed = []

    def measure(ctx):
        observed.append(ctx)
        assert ctx.local_gateway_full_model is True
        assert ctx.local_full_model_path == str(_local_cortex_full_model_offline_model.root)
        assert ctx.target_model_name is None
        assert ctx.cfg.model_name == "Qwen/Qwen3-8B"
        assert ctx.spec.model.cache_path == _local_cortex_full_model_offline_model.spec.model.cache_path
        yield Result(test_id="e2e-sft-train-validate", config_id=ctx.config_id, outcome=Outcome.PASS)

    monkeypatch.setattr(runner.registered_tests()["e2e-sft-train-validate"], "fn", measure)
    monkeypatch.setattr(cortex_fixture, "load_fixture", lambda *args: pytest.fail("no fabricated publication"))
    monkeypatch.setattr(cortex_fixture, "load_local_fixture", lambda *args: pytest.fail("no reduced local binding"))
    results = _local_cortex_full_model_execute(_local_cortex_full_model_offline_model, tmp_path)
    assert len(observed) == 1 and _local_cortex_full_model_offline_model.calls == ["Qwen/Qwen3-8B"]
    assert [result.outcome for result in results] == [Outcome.PASS]
    assert (_cortex_optimizer_capture_CONFIG.read_bytes(), _cortex_optimizer_capture_SPEC.read_bytes()) == original


@pytest.mark.parametrize(
    "failure",
    [
        "reduced",
        "five-layers",
        "missing-shard",
        "index-escape",
        "symlink-escape",
        "missing-layer",
        "wrong-revision",
        "wrong-repository",
        "missing-tokenizer",
        "changed-rope",
    ],
)
def test_local_cortex_full_model_full_snapshot_invalidity_rejected_before_test(
    _local_cortex_full_model_offline_model, tmp_path, monkeypatch, failure
):
    root = _local_cortex_full_model_offline_model.root
    if failure in {"reduced", "five-layers"}:
        config = json.loads((root / "config.json").read_text())
        config["num_hidden_layers"] = 4 if failure == "reduced" else 5
        (root / "config.json").write_text(json.dumps(config))
    elif failure == "missing-shard":
        (root / "model-00001-of-00001.safetensors").unlink()
    elif failure == "index-escape":
        _local_cortex_full_model_offline_model.index["weight_map"]["model.layers.0.weight"] = "../outside.safetensors"
        (root / "model.safetensors.index.json").write_text(json.dumps(_local_cortex_full_model_offline_model.index))
    elif failure == "symlink-escape":
        (root / "config.json").unlink()
        (root / "config.json").symlink_to("/tmp/nonexistent-full-model.json")
    elif failure in {"wrong-revision", "wrong-repository"}:
        from transformers.utils import hub

        wrong = (
            root.with_name("b" * 40) if failure == "wrong-revision" else root.parent.parent.with_name("models--other")
        )
        monkeypatch.setattr(hub, "cached_file", lambda *args, **kwargs: str(wrong / "config.json"))
    elif failure == "missing-tokenizer":
        (root / "tokenizer.json").unlink()
    elif failure == "changed-rope":
        config = json.loads((root / "config.json").read_text())
        config["rope_theta"] = 10000
        (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    else:
        tensors = {f"model.layers.{i}.weight": torch.ones(1) for i in range(35)}
        save_file(tensors, root / "model-00001-of-00001.safetensors")
    monkeypatch.setattr(
        runner.registered_tests()["e2e-sft-train-validate"],
        "fn",
        lambda *args: pytest.fail("invalid full snapshot reached test"),
    )
    with pytest.raises((ValueError, OSError)):
        _local_cortex_full_model_execute(_local_cortex_full_model_offline_model, tmp_path)


@pytest.mark.parametrize("variable", ["HF_HUB_OFFLINE", "DSS_ALLOW_REMOTE_MODELS"])
def test_local_cortex_full_model_full_mode_requires_operational_offline_contract(
    _local_cortex_full_model_offline_model, tmp_path, monkeypatch, variable
):
    monkeypatch.delenv(variable)
    with pytest.raises(ValueError, match="offline"):
        _local_cortex_full_model_execute(_local_cortex_full_model_offline_model, tmp_path)
    assert _local_cortex_full_model_offline_model.calls == []


@pytest.mark.parametrize(
    "backend", [OnPremConfig(), _local_cortex_full_model_CortexConfig(base_url="http://remote.example")]
)
def test_local_cortex_full_model_full_model_mode_is_only_for_verified_local_gateway(
    _local_cortex_full_model_offline_model, tmp_path, backend
):
    with pytest.raises(ValueError):
        runner.execute(
            _local_cortex_full_model_offline_model.cfg,
            _local_cortex_full_model_offline_model.spec,
            [],
            vocab_size=128,
            workdir=tmp_path,
            select=["e2e-sft-train-validate"],
            client_backend=backend,
            local_gateway_full_model=True,
        )
    assert _local_cortex_full_model_offline_model.calls == []


@pytest.mark.parametrize(
    "selected", [None, ["single-step-optimizer"], ["e2e-sft-train-validate", "single-step-grads"]]
)
def test_local_cortex_full_model_full_mode_cannot_bypass_other_fixture_guards(
    _local_cortex_full_model_offline_model, tmp_path, selected
):
    with pytest.raises(ValueError):
        runner.execute(
            _local_cortex_full_model_offline_model.cfg,
            _local_cortex_full_model_offline_model.spec,
            [],
            vocab_size=128,
            workdir=tmp_path,
            select=selected,
            client_backend=_local_cortex_full_model_CortexConfig(base_url="http://127.0.0.1:8003"),
            local_gateway_full_model=True,
        )
    assert _local_cortex_full_model_offline_model.calls == []


def test_local_cortex_full_model_hosted_e2e_still_requires_reviewed_published_fixture(tmp_path):
    cfg, spec = (load_config(_cortex_optimizer_capture_CONFIG), Spec.read(_cortex_optimizer_capture_SPEC))
    results = runner.execute(
        cfg,
        spec,
        [],
        vocab_size=128,
        select=["e2e-sft-train-validate"],
        client_backend=_local_cortex_full_model_CortexConfig(base_url="http://remote.example"),
        workdir=tmp_path,
    )
    assert len(results) == 1 and results[0].outcome is Outcome.FAIL
    assert "published Cortex fixture" in results[0].summary


def test_local_cortex_full_model_cli_full_mode_explicit_selection_is_eligible(monkeypatch):
    args = SimpleNamespace(
        local_gateway_full_model=True,
        local_cortex_fixture=False,
        cortex_fixture=None,
        config=str(_cortex_optimizer_capture_CONFIG),
        all_configs=False,
        test=["e2e-sft-train-validate"],
    )
    monkeypatch.setattr(
        cli, "_client_backend", lambda args: _local_cortex_full_model_CortexConfig(base_url="http://127.0.0.1:8003")
    )

    def stop(*args, **kwargs):
        raise RuntimeError("eligible full-model dispatch")

    monkeypatch.setattr(cli, "_resolve_configs", stop)
    with pytest.raises(RuntimeError, match="eligible full-model"):
        cli.cmd_run(args)


@pytest.mark.parametrize(
    "changes",
    [
        {"test": ["single-step-optimizer"]},
        {"test": None},
        {"local_cortex_fixture": True},
        {"all_configs": True},
        {"cortex_fixture": Path("receipt.json")},
    ],
)
def test_local_cortex_full_model_cli_full_mode_invalid_combinations_rejected_before_dispatch(monkeypatch, changes):
    args = SimpleNamespace(
        local_gateway_full_model=True,
        local_cortex_fixture=False,
        cortex_fixture=None,
        config=str(_cortex_optimizer_capture_CONFIG),
        all_configs=False,
        test=["e2e-sft-train-validate"],
    )
    vars(args).update(changes)
    monkeypatch.setattr(
        cli, "_client_backend", lambda args: _local_cortex_full_model_CortexConfig(base_url="http://127.0.0.1:8003")
    )
    monkeypatch.setattr(cli, "_resolve_configs", lambda *args, **kwargs: pytest.fail("invalid full mode dispatched"))
    with pytest.raises(SystemExit):
        cli.cmd_run(args)


def test_local_cortex_full_model_full_model_vocab_uses_validated_local_snapshot_without_catalog_lookup(
    _local_cortex_full_model_offline_model, monkeypatch
):
    import transformers

    seen = []

    def config(path, **kwargs):
        assert path == _local_cortex_full_model_offline_model.root
        assert kwargs["local_files_only"] is True
        seen.append(path)
        return SimpleNamespace(vocab_size=151936)

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", config)
    monkeypatch.setattr(cli, "_catalog_vocab_size", lambda *args: pytest.fail("full mode attempted catalog lookup"))
    assert (
        cli._run_vocab_size(
            _local_cortex_full_model_offline_model.cfg,
            _local_cortex_full_model_offline_model.spec,
            ["e2e-sft-train-validate"],
            _local_cortex_full_model_CortexConfig(base_url="http://127.0.0.1:8003"),
            local_full_model_path=_local_cortex_full_model_offline_model.root,
        )
        == 151936
    )
    assert seen == [_local_cortex_full_model_offline_model.root]


# Local cortex fixture.


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


# Reference-free original-source metadata.


@pytest.fixture
def _reference_free_metadata_workspace():
    import tempfile

    with tempfile.TemporaryDirectory(prefix="reference-free-metadata-", dir="/data-fast/tmp") as directory:
        yield Path(directory).resolve()


def _reference_free_metadata_source(workspace, *, indexed=True, vision="depth"):
    from arctic_platform.correctness.harness.inference_export import schema_bytes

    root = workspace / "original"
    root.mkdir()
    config = {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {
            "model_type": "glm5_next_text",
            "num_hidden_layers": 45,
            "vocab_size": 128,
            "dtype": "bfloat16",
        },
    }
    if vision is not None:
        config["vision_config"] = {"model_type": "vision_fixture", vision: 24}
    (root / "config.json").write_text(json.dumps(config))
    tensors = {
        "model.embed_tokens.weight": torch.zeros((4, 3), dtype=torch.bfloat16),
        "model.layers.44.norm.weight": torch.ones(3, dtype=torch.float32),
        "model.layer_multipliers": torch.tensor(1, dtype=torch.int64),
    }
    schema = {
        "model.embed_tokens.weight": ["BF16", [4, 3]],
        "model.layers.44.norm.weight": ["F32", [3]],
        "model.layer_multipliers": ["I64", []],
    }
    if indexed:
        shards = {
            "model-00001-of-00002.safetensors": {"model.embed_tokens.weight": tensors["model.embed_tokens.weight"]},
            "model-00002-of-00002.safetensors": {
                name: tensor for name, tensor in tensors.items() if name != "model.embed_tokens.weight"
            },
        }
        for name, values in shards.items():
            save_file(values, str(root / name))
        index = {
            "metadata": {"total_size": schema_bytes(schema)},
            "weight_map": {tensor: shard for shard, values in shards.items() for tensor in values},
        }
        (root / "model.safetensors.index.json").write_text(json.dumps(index))
    else:
        save_file(tensors, str(root / "model.safetensors"))
        index = None
    return SimpleNamespace(root=root, config=config, schema=schema, index=index)


def _reference_free_metadata_unchanged(root):
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_dev,
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
            path.stat().st_ctime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.iterdir())
    }


@pytest.fixture
def _reference_free_metadata_no_model(monkeypatch):
    import huggingface_hub
    import transformers
    from safetensors import torch as safetensors_torch

    from arctic_platform.correctness.onboarding import onboard
    from arctic_platform.correctness.onboarding import synth_model

    def forbidden(*args, **kwargs):
        pytest.fail("reference-free metadata reached model construction, sizing, payloads or HF references")

    for name in ("_model_shape", "build_spec", "_prepare_references", "_render_and_run_calibration"):
        monkeypatch.setattr(onboard, name, forbidden)
    for name in ("reduce_config", "count_parameters", "size_for_single_gpu", "materialize_pretrained"):
        monkeypatch.setattr(synth_model, name, forbidden)
    for name in ("AutoConfig", "AutoModel", "AutoModelForCausalLM"):
        cls = getattr(transformers, name)
        monkeypatch.setattr(cls, "from_pretrained", forbidden)
        if hasattr(cls, "from_config"):
            monkeypatch.setattr(cls, "from_config", forbidden)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(safetensors_torch, "load_file", forbidden)
    actual_safe_open = safetensors.safe_open
    header_files = []

    @contextmanager
    def headers_only(path, **kwargs):
        header_files.append(Path(path).name)
        with actual_safe_open(path, **kwargs) as stream:
            yield SimpleNamespace(keys=stream.keys, get_slice=stream.get_slice, get_tensor=forbidden)

    monkeypatch.setattr(safetensors, "safe_open", headers_only)
    actual_open = Path.open
    actual_read_bytes = Path.read_bytes

    def metadata_open(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if path.suffix == ".safetensors" and ("r" in mode or "+" in mode):
            forbidden()
        return actual_open(path, *args, **kwargs)

    def metadata_read_bytes(path):
        if path.suffix == ".safetensors":
            forbidden()
        return actual_read_bytes(path)

    monkeypatch.setattr(Path, "open", metadata_open)
    monkeypatch.setattr(Path, "read_bytes", metadata_read_bytes)
    return header_files


@pytest.mark.parametrize("indexed", [True, False], ids=["indexed", "unsharded"])
@pytest.mark.parametrize(
    "vision", ["depth", "num_hidden_layers", None], ids=["vision-depth", "vision-layers", "text-only"]
)
def test_reference_free_metadata_original_schema_no_constructor_or_source_rewrite(
    _reference_free_metadata_workspace, monkeypatch, indexed, vision
):
    from arctic_platform.correctness.harness.inference_export import schema_bytes
    from arctic_platform.correctness.harness.inference_export import schema_sha
    from arctic_platform.correctness.onboarding import synth_model

    case = _reference_free_metadata_source(_reference_free_metadata_workspace, indexed=indexed, vision=vision)
    before = _reference_free_metadata_unchanged(case.root)
    # Activate the same no-construction/payload sentinels after hashing the tiny test fixtures.
    with monkeypatch.context() as guard:
        headers = _reference_free_metadata_no_model.__wrapped__(guard)
        model, vocab_size, evidence = synth_model.describe_source_model(str(case.root))
    assert _reference_free_metadata_unchanged(case.root) == before
    assert vocab_size == 128
    assert model.num_hidden_layers == 45
    assert model.vision_depth == (24 if vision is not None else None)
    assert model.cache_path == str(case.root)
    assert model.source_checkpoint == str(case.root)
    assert model.content_hash == hash_directory(case.root)
    assert model.seed is model.param_count is model.sized_against_gib is None
    assert model.measured_reference_peak_gib is None
    assert evidence == {
        "config_sha256": before["config.json"][-1],
        "index_sha256": before["model.safetensors.index.json"][-1] if indexed else None,
        "tensor_schema_sha256": schema_sha(case.schema),
        "tensor_count": 3,
        "stored_tensor_elements": 16,
        "tensor_payload_bytes": schema_bytes(case.schema),
        "tensor_payloads_read": False,
    }
    assert set(headers) == {path.name for path in case.root.glob("*.safetensors")}
    assert (case.root / "model.safetensors.index.json").exists() is indexed


@pytest.mark.parametrize(
    "location,field,value",
    [
        ("root", "text_config", None),
        ("root", "text_config", []),
        ("root", "text_config", "not-an-object"),
        ("text", "num_hidden_layers", None),
        ("text", "num_hidden_layers", 0),
        ("text", "num_hidden_layers", -1),
        ("text", "num_hidden_layers", True),
        ("text", "num_hidden_layers", 45.0),
        ("text", "vocab_size", None),
        ("text", "vocab_size", 0),
        ("text", "vocab_size", False),
        ("text", "vocab_size", "128"),
        ("root", "vision_config", []),
        ("root", "vision_config", {}),
        ("vision", "depth", 0),
        ("vision", "depth", True),
        ("vision", "depth", 24.0),
    ],
)
def test_reference_free_metadata_rejects_invalid_original_config(
    _reference_free_metadata_workspace, _reference_free_metadata_no_model, location, field, value
):
    from arctic_platform.correctness.onboarding.synth_model import describe_source_model

    case = _reference_free_metadata_source(_reference_free_metadata_workspace)
    holder = {"root": case.config, "text": case.config["text_config"], "vision": case.config["vision_config"]}[
        location
    ]
    holder[field] = value
    (case.root / "config.json").write_text(json.dumps(case.config))
    with pytest.raises(ValueError, match="original .*required"):
        describe_source_model(str(case.root))


def _reference_free_metadata_fp8_source(workspace, *, indexed):
    case = _reference_free_metadata_source(workspace, indexed=False)
    case.config["quantization_config"] = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    (case.root / "config.json").write_text(json.dumps(case.config))
    weight = "model.layers.44.mlp.down_proj.weight"
    scale = "model.layers.44.mlp.down_proj.weight_scale_inv"
    tensors = {
        "model.embed_tokens.weight": torch.zeros((4, 3), dtype=torch.bfloat16),
        "model.layers.44.norm.weight": torch.ones(3, dtype=torch.float32),
        "model.layer_multipliers": torch.tensor(1, dtype=torch.int64),
        weight: torch.ones((129, 257), dtype=torch.bfloat16).to(torch.float8_e4m3fn),
        scale: torch.ones((2, 3), dtype=torch.float32),
    }
    save_file(tensors, str(case.root / "model.safetensors"))
    case.schema.update({weight: ["F8_E4M3", [129, 257]], scale: ["F32", [2, 3]]})
    case.payload_bytes = 24 + 12 + 8 + 129 * 257 + 2 * 3 * 4
    if indexed:
        case.index = {
            "metadata": {"total_size": case.payload_bytes},
            "weight_map": {name: "model.safetensors" for name in tensors},
        }
        (case.root / "model.safetensors.index.json").write_text(json.dumps(case.index))
    return case


@pytest.mark.parametrize("indexed", [True, False], ids=["indexed", "unsharded"])
def test_reference_free_metadata_declared_block_fp8_source_preserves_strict_exports(
    _reference_free_metadata_workspace, monkeypatch, indexed
):
    from arctic_platform.correctness.harness import inference_export
    from arctic_platform.correctness.onboarding.synth_model import describe_source_model

    case = _reference_free_metadata_fp8_source(_reference_free_metadata_workspace, indexed=indexed)
    before = _reference_free_metadata_unchanged(case.root)
    with monkeypatch.context() as guard:
        _reference_free_metadata_no_model.__wrapped__(guard)
        model, vocab, evidence = describe_source_model(str(case.root))
        with pytest.raises(ValueError, match="invalid tensor dtype"):
            inference_export.schema_bytes(case.schema)
        if indexed:
            with pytest.raises(ValueError, match="invalid tensor dtype"):
                inference_export.read_inventory(case.root, case.index)
    assert _reference_free_metadata_unchanged(case.root) == before
    assert model.seed is model.param_count is model.sized_against_gib is None
    assert vocab == 128
    assert evidence["tensor_count"] == 5
    assert evidence["tensor_payload_bytes"] == case.payload_bytes
    assert evidence["tensor_schema_sha256"] == inference_export.schema_sha(case.schema)
    assert evidence["original_quantization_config"] == case.config["quantization_config"]
    assert evidence["tensor_payloads_read"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("quant_method", "int8"),
        ("fmt", "e5m2"),
        ("activation_scheme", "static"),
        ("weight_block_size", [64, 128]),
        ("weight_block_size", [128.0, 128]),
        ("weight_block_size", [128, 128.0]),
        ("weight_block_size", [True, 128]),
        ("weight_block_size", None),
        ("weight_block_size", "128,128"),
    ],
)
def test_reference_free_metadata_rejects_unsupported_fp8_declaration(
    _reference_free_metadata_workspace, _reference_free_metadata_no_model, field, value
):
    from arctic_platform.correctness.onboarding.synth_model import describe_source_model

    case = _reference_free_metadata_fp8_source(_reference_free_metadata_workspace, indexed=True)
    case.config["quantization_config"][field] = value
    (case.root / "config.json").write_text(json.dumps(case.config))
    with pytest.raises(ValueError, match="supported dynamic block-FP8"):
        describe_source_model(str(case.root))


@pytest.mark.parametrize("corruption", ["missing-declaration", "missing-scale", "scale-shape", "scale-dtype", "size"])
def test_reference_free_metadata_rejects_fp8_inventory_corruption(_reference_free_metadata_workspace, corruption):
    from arctic_platform.correctness.harness.inference_export import original_schema_bytes
    from arctic_platform.correctness.harness.inference_export import read_inventory

    case = _reference_free_metadata_fp8_source(_reference_free_metadata_workspace, indexed=True)
    quantization = case.config["quantization_config"]
    scale = "model.layers.44.mlp.down_proj.weight_scale_inv"
    if corruption == "missing-declaration":
        quantization = None
    elif corruption == "missing-scale":
        case.schema.pop(scale)
    elif corruption == "scale-shape":
        case.schema[scale] = ["F32", [1, 3]]
    elif corruption == "scale-dtype":
        case.schema[scale] = ["BF16", [2, 3]]
    else:
        case.index["metadata"]["total_size"] += 1
        with pytest.raises(ValueError, match="byte inventory disagrees"):
            read_inventory(case.root, case.index, original_quantization_config=quantization)
        return
    with pytest.raises(ValueError, match="invalid tensor dtype|matching FP32 inverse block scales"):
        original_schema_bytes(case.schema, quantization)


@pytest.mark.parametrize(
    "bad_json",
    [
        "[]",
        '{"text_config": {}, "text_config": {"num_hidden_layers": 45, "vocab_size": 128}}',
        '{"text_config":',
    ],
    ids=["nonobject", "duplicate-key", "malformed"],
)
def test_reference_free_metadata_rejects_noncanonical_json(
    _reference_free_metadata_workspace, _reference_free_metadata_no_model, bad_json
):
    from arctic_platform.correctness.onboarding.synth_model import describe_source_model

    case = _reference_free_metadata_source(_reference_free_metadata_workspace)
    (case.root / "config.json").write_text(bad_json)
    with pytest.raises(ValueError):
        describe_source_model(str(case.root))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-config",
        "missing-shard",
        "extra-shard",
        "missing-index-and-single",
        "empty-map",
        "wrong-byte-count",
        "wrong-header-name",
        "wrong-shard",
        "traversal",
        "nonobject-index",
        "duplicate-index-key",
        "bad-header",
    ],
)
def test_reference_free_metadata_rejects_inventory_or_metadata_mismatch(
    _reference_free_metadata_workspace, _reference_free_metadata_no_model, mutation
):
    from arctic_platform.correctness.onboarding.synth_model import describe_source_model

    case = _reference_free_metadata_source(_reference_free_metadata_workspace)
    index_path = case.root / "model.safetensors.index.json"
    shard = case.root / "model-00001-of-00002.safetensors"
    if mutation == "missing-config":
        (case.root / "config.json").unlink()
    elif mutation == "missing-shard":
        shard.unlink()
    elif mutation == "extra-shard":
        save_file({"extra": torch.zeros(1, dtype=torch.bfloat16)}, str(case.root / "extra.safetensors"))
    elif mutation == "missing-index-and-single":
        index_path.unlink()
    elif mutation == "empty-map":
        case.index["weight_map"] = {}
    elif mutation == "wrong-byte-count":
        case.index["metadata"]["total_size"] += 2
    elif mutation == "wrong-header-name":
        case.index["weight_map"]["wrong.tensor"] = case.index["weight_map"].pop("model.embed_tokens.weight")
    elif mutation == "wrong-shard":
        case.index["weight_map"]["model.embed_tokens.weight"] = "model-00002-of-00002.safetensors"
    elif mutation == "traversal":
        case.index["weight_map"]["model.embed_tokens.weight"] = "../outside.safetensors"
    elif mutation == "bad-header":
        shard.write_bytes(b"invalid-safetensors-header")
    if mutation == "nonobject-index":
        index_path.write_text("[]")
    elif mutation == "duplicate-index-key":
        index_path.write_text('{"weight_map": {}, "weight_map": {}}')
    elif index_path.exists():
        index_path.write_text(json.dumps(case.index))
    with pytest.raises((ValueError, OSError, safetensors.SafetensorError)):
        describe_source_model(str(case.root))


@pytest.mark.parametrize(
    "tensors,total",
    [
        ({"weight": torch.zeros(1, dtype=torch.float16)}, 2),
        ({"unknown_buffer": torch.zeros(1, dtype=torch.int64)}, 8),
        ({"weight": torch.empty((0, 3), dtype=torch.bfloat16)}, 0),
    ],
    ids=["unreviewed-f16", "unreviewed-i64", "zero-dimension"],
)
@pytest.mark.parametrize("indexed", [True, False], ids=["indexed", "unsharded"])
def test_reference_free_metadata_rejects_unreviewed_header_schema(
    _reference_free_metadata_workspace, _reference_free_metadata_no_model, tensors, total, indexed
):
    from arctic_platform.correctness.onboarding.synth_model import describe_source_model

    case = _reference_free_metadata_source(_reference_free_metadata_workspace, indexed=False)
    save_file(tensors, str(case.root / "model.safetensors"))
    if indexed:
        (case.root / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": total}, "weight_map": {name: "model.safetensors" for name in tensors}}
            )
        )
    with pytest.raises(ValueError, match="invalid tensor dtype or shape"):
        describe_source_model(str(case.root))


@pytest.mark.parametrize("target", ["root", "ancestor", "config.json", "model.safetensors.index.json", "shard"])
def test_reference_free_metadata_rejects_symlinks(
    _reference_free_metadata_workspace, _reference_free_metadata_no_model, target
):
    from arctic_platform.correctness.onboarding.synth_model import describe_source_model

    workspace = _reference_free_metadata_workspace
    case = _reference_free_metadata_source(workspace)
    source = case.root
    if target == "root":
        source = workspace / "alias"
        source.symlink_to(case.root, target_is_directory=True)
    elif target == "ancestor":
        alias = workspace / "parent-alias"
        alias.symlink_to(workspace, target_is_directory=True)
        source = alias / "original"
    else:
        name = "model-00001-of-00002.safetensors" if target == "shard" else target
        path = case.root / name
        moved = workspace / name
        path.rename(moved)
        path.symlink_to(moved)
    with pytest.raises(ValueError, match="canonical regular|regular original"):
        describe_source_model(str(source))


@pytest.mark.parametrize("target", ["config.json", "model.safetensors.index.json", "shard", "membership", "inode"])
def test_reference_free_metadata_rejects_source_drift_during_header_admission(
    _reference_free_metadata_workspace, _reference_free_metadata_no_model, monkeypatch, target
):
    from arctic_platform.correctness.harness import inference_export
    from arctic_platform.correctness.onboarding.synth_model import describe_source_model

    case = _reference_free_metadata_source(_reference_free_metadata_workspace)
    actual_inventory = inference_export.read_inventory

    def drifting_inventory(root, index, *, original_quantization_config=None):
        result = actual_inventory(root, index, original_quantization_config=original_quantization_config)
        if target == "membership":
            save_file({"extra": torch.zeros(1, dtype=torch.bfloat16)}, str(case.root / "new.safetensors"))
        else:
            name = "model-00001-of-00002.safetensors" if target in ("shard", "inode") else target
            path = case.root / name
            if target == "inode":
                path.rename(case.root / "replaced.safetensors")
                save_file({"model.embed_tokens.weight": torch.zeros((4, 3), dtype=torch.bfloat16)}, str(path))
            else:
                with path.open("ab") if path.suffix != ".safetensors" else open(path, "ab") as stream:
                    stream.write(b" ")
        return result

    monkeypatch.setattr(inference_export, "read_inventory", drifting_inventory)
    with pytest.raises(ValueError, match="changed during preparation"):
        describe_source_model(str(case.root))


def _reference_free_metadata_rl_args(workspace, monkeypatch):
    from arctic_platform.correctness.harness.config import LoadedConfig
    from arctic_platform.correctness.onboarding import onboard

    case = _reference_free_metadata_source(workspace)
    training = {
        "job_type": "training",
        "model_name": "zai-org/GLM-5.3-Flash-BF16",
        "dtype": "bfloat16",
        "training_config": {
            "n_gpus": 2,
            "sp_size": 2,
            "max_seq_len": 128,
            "mb_spec": {"max_tokens_per_mb": 128},
            "attn_implementation": "sparse_mla",
        },
    }
    cfg = LoadedConfig(
        config_id="metadata-fixture-rl",
        path=workspace / "rl.config",
        sub_job=training,
        sub_jobs=[training, {"job_type": "sampling"}],
    )
    # Relocate only the canonical publication root; never write the maintained specs directory.
    monkeypatch.setattr(onboard, "__file__", str(workspace / "overlay" / "onboarding" / "onboard.py"))
    args = SimpleNamespace(
        test_id="inference-checkpoint-loss",
        source_checkpoint=str(case.root),
        test_spec=workspace / "overlay" / "specs" / f"{cfg.config_id}.json",
        tokens=128,
        config=cfg.path,
        output_dir=workspace / "output",
    )
    args.output_dir.mkdir()
    return args, cfg, case


def test_reference_free_metadata_null_roundtrip_preserves_full_depth_and_evidence(
    _reference_free_metadata_workspace, _reference_free_metadata_no_model, monkeypatch
):
    from arctic_platform.correctness.onboarding import onboard

    args, cfg, _ = _reference_free_metadata_rl_args(_reference_free_metadata_workspace, monkeypatch)
    spec = onboard._build_reference_free_rl_spec(args, cfg)
    spec.write(args.test_spec)
    raw = json.loads(args.test_spec.read_text())
    assert {field: raw["model"][field] for field in ("seed", "param_count", "sized_against_gib")} == {
        "seed": None,
        "param_count": None,
        "sized_against_gib": None,
    }
    assert raw["model"]["vision_depth"] == 24
    restored = Spec.read(args.test_spec)
    assert restored == spec
    assert restored.test_settings[args.test_id]["source_metadata"]["tensor_payloads_read"] is False


def test_reference_free_metadata_generated_arms_checksum_and_only_selected_rl_check(
    _reference_free_metadata_workspace, _reference_free_metadata_no_model, monkeypatch
):
    from arctic_platform.correctness.harness.batches import SEED
    from arctic_platform.correctness.harness.batches import build_batch
    from arctic_platform.correctness.harness.config import arms_for
    from arctic_platform.correctness.harness.config import config_checksum
    from arctic_platform.correctness.harness.config import correctness_microbatch_tokens
    from arctic_platform.correctness.harness.registry import RL_INAPPLICABLE_REASON
    from arctic_platform.correctness.harness.registry import registered_tests
    from arctic_platform.correctness.onboarding import onboard

    args, cfg, _ = _reference_free_metadata_rl_args(_reference_free_metadata_workspace, monkeypatch)
    expected_checksum = config_checksum(cfg)
    spec = onboard._build_reference_free_rl_spec(args, cfg)
    assert spec.config_id == cfg.config_id
    assert spec.config_path == str(cfg.path)
    assert spec.config_checksum == expected_checksum == config_checksum(cfg)
    assert spec.attn_implementations == [cfg.attention_implementation]
    assert spec.applicable_tests == [args.test_id]
    assert spec.test_tolerances == {}
    assert spec.inapplicable == {
        test_id: RL_INAPPLICABLE_REASON for test_id, entry in registered_tests().items() if entry.compares_to_reference
    }
    assert list(spec.test_settings) == [args.test_id]
    definitions = arms_for(cfg.max_seq_len, cfg.n_gpus)
    assert len(spec.arms) == len(definitions) == 2
    for arm, definition in zip(spec.arms, definitions):
        batch = build_batch(definition.name, definition.global_batch_size, definition.max_seq_len, 128, seed=SEED)
        assert vars(arm) == {
            "name": definition.name,
            "global_batch_size": definition.global_batch_size,
            "max_seq_len": definition.max_seq_len,
            "total_tokens": batch.total_tokens,
            "active_tokens": batch.active_tokens,
            "dss_microbatches": definition.microbatches(
                correctness_microbatch_tokens(cfg.max_tokens_per_mb), cfg.dp_size
            ),
            "pad_fraction": batch.pad_fraction,
        }
    assert not args.test_spec.exists()
    assert "not an architecture parameter count" in " ".join(spec.notes)


@pytest.mark.parametrize("invalid", ["custom-destination", "smaller-tokens", "sft", "reference-check", "hosted-check"])
def test_reference_free_metadata_bootstrap_prerequisites_fail_before_source_admission(
    _reference_free_metadata_workspace, _reference_free_metadata_no_model, monkeypatch, invalid
):
    from arctic_platform.correctness.onboarding import onboard
    from arctic_platform.correctness.onboarding import synth_model

    args, cfg, _ = _reference_free_metadata_rl_args(_reference_free_metadata_workspace, monkeypatch)
    if invalid == "custom-destination":
        args.test_spec = args.output_dir / "custom.json"
    elif invalid == "smaller-tokens":
        args.tokens -= 1
    elif invalid == "sft":
        cfg.sub_jobs = [cfg.sub_job]
    elif invalid == "reference-check":
        args.test_id = "single-step-grads"
    else:
        from arctic_platform.correctness.harness.registry import registered_tests

        # No current check requires hosted execution; exercise the existing admission flag directly.
        monkeypatch.setattr(registered_tests()[args.test_id], "requires_hosted_control_plane", True)
    monkeypatch.setattr(
        synth_model, "describe_source_model", lambda *args: pytest.fail("invalid bootstrap reached metadata admission")
    )
    with pytest.raises(ValueError, match="initial RL onboarding"):
        onboard._build_reference_free_rl_spec(args, cfg)
    assert not args.test_spec.exists()


@pytest.mark.parametrize("result", ["pass", "fail", "exception"])
def test_reference_free_metadata_publication_requires_final_regression_and_rolls_back(
    _reference_free_metadata_workspace, _reference_free_metadata_no_model, monkeypatch, result
):
    from arctic_platform.correctness.onboarding import onboard

    args, cfg, _ = _reference_free_metadata_rl_args(_reference_free_metadata_workspace, monkeypatch)
    calls = []

    def final_regression(actual_args, selected, *, phase):
        assert actual_args is args
        candidate = Spec.read(args.test_spec)
        assert candidate.model.num_hidden_layers == 45
        assert candidate.model.param_count is candidate.model.seed is candidate.model.sized_against_gib is None
        assert candidate.test_tolerances == {}
        assert candidate.applicable_tests == [args.test_id]
        assert candidate.test_settings[args.test_id]["source_metadata"]["tensor_payloads_read"] is False
        assert selected == {"layers": 45, "model_description": f"configured full model {cfg.model_name}"}
        assert phase == "phase 1/1"
        calls.append(candidate)
        if result == "exception":
            raise RuntimeError("final regression sentinel")
        return {"status": result, "report": str(args.output_dir / "report.json"), "log": "retained-final.log"}

    monkeypatch.setattr(onboard, "_run_final_regression", final_regression)
    manifest = args.output_dir / f"onboarding-{args.test_id}.json"
    if result == "pass":
        assert onboard._onboard_fixed_tolerance_test(args, cfg) == 0
        assert Spec.read(args.test_spec) == calls[0]
        record = json.loads(manifest.read_text())
        assert record["bootstrapped_reference_free_rl"] is True
        assert record["validated_model_scope"] == "configured-full-model"
        assert record["num_hidden_layers"] == 45
        assert record["frozen_reference_num_hidden_layers"] is None
        assert "not calibrated" in record["tolerance"]
        assert not list(args.output_dir.glob("unvalidated-rl-spec-*.json"))
    else:
        with pytest.raises(RuntimeError, match="final regression sentinel|inference-checkpoint-loss fail"):
            onboard._onboard_fixed_tolerance_test(args, cfg)
        assert not args.test_spec.exists()
        assert not manifest.exists()
        quarantined = list(args.output_dir.glob("unvalidated-rl-spec-*.json"))
        assert len(quarantined) == 1
        assert Spec.read(quarantined[0]) == calls[0]
    assert len(calls) == 1


def test_reference_free_metadata_existing_spec_checksum_drift_does_not_regress_or_restamp(
    _reference_free_metadata_workspace, _reference_free_metadata_no_model, monkeypatch
):
    from arctic_platform.correctness.onboarding import onboard

    args, cfg, _ = _reference_free_metadata_rl_args(_reference_free_metadata_workspace, monkeypatch)
    spec = onboard._build_reference_free_rl_spec(args, cfg)
    spec.write(args.test_spec)
    before = args.test_spec.read_bytes()
    cfg.training["max_seq_len"] += 1
    monkeypatch.setattr(
        onboard, "_run_final_regression", lambda *args, **kwargs: pytest.fail("checksum drift regressed")
    )
    with pytest.raises(ValueError, match="reviewed spec records training config checksum"):
        onboard._onboard_fixed_tolerance_test(args, cfg)
    assert args.test_spec.read_bytes() == before


def test_reference_free_metadata_existing_reviewed_spec_restored_after_final_failure(
    _reference_free_metadata_workspace, _reference_free_metadata_no_model, monkeypatch
):
    from arctic_platform.correctness.onboarding import onboard

    args, cfg, _ = _reference_free_metadata_rl_args(_reference_free_metadata_workspace, monkeypatch)
    spec = onboard._build_reference_free_rl_spec(args, cfg)
    spec.applicable_tests = []
    spec.notes.append("retained reviewed sentinel")
    spec.write(args.test_spec)
    before = args.test_spec.read_bytes()

    def final_regression(*args, **kwargs):
        raise RuntimeError("existing final regression sentinel")

    monkeypatch.setattr(onboard, "_run_final_regression", final_regression)
    with pytest.raises(RuntimeError, match="existing final regression sentinel"):
        onboard._onboard_fixed_tolerance_test(args, cfg)
    assert args.test_spec.read_bytes() == before
    assert not list(args.output_dir.glob("unvalidated-rl-spec-*.json"))


def test_reference_free_metadata_supervised_build_spec_retains_reduced_numeric_recipe(
    _reference_free_metadata_workspace, monkeypatch
):
    import transformers

    from arctic_platform.correctness.harness.batches import SEED
    from arctic_platform.correctness.harness.config import ModelSpec
    from arctic_platform.correctness.harness.config import arms_for
    from arctic_platform.correctness.onboarding import onboard
    from arctic_platform.correctness.onboarding import synth_model

    workspace = _reference_free_metadata_workspace
    args, cfg, _ = _reference_free_metadata_rl_args(workspace, monkeypatch)
    cfg.sub_jobs = [cfg.sub_job]
    observed = []
    model = ModelSpec(
        source_checkpoint=args.source_checkpoint,
        num_hidden_layers=2,
        vision_depth=2,
        seed=SEED,
        param_count=456,
        cache_path=str(workspace / "reduced"),
        content_hash="numeric-reduced-fixture",
        sized_against_gib=100.0,
    )

    def materialize(source, destination, layers):
        observed.append((source, destination, layers))
        return model

    monkeypatch.setattr(synth_model, "materialize_pretrained", materialize)
    monkeypatch.setattr(synth_model, "size_for_single_gpu", lambda source: (2, 456))
    monkeypatch.setattr(
        synth_model,
        "describe_source_model",
        lambda *args: pytest.fail("supervised sizing used RL metadata descriptor"),
    )
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(vocab_size=128)
    )
    spec = onboard.build_spec(
        cfg,
        args.source_checkpoint,
        str(workspace / "cache"),
        arms_for(cfg.max_seq_len, cfg.n_gpus),
        workdir=workspace / "work",
    )
    assert observed == [(args.source_checkpoint, str(workspace / "cache" / "original-2L"), 2)]
    assert spec.model is model
    assert spec.model.param_count == 456
    assert spec.model.seed == SEED
    assert spec.model.sized_against_gib == 100.0
    assert spec.applicable_tests == ["single-step-grads"]
    assert spec.inapplicable == spec.test_settings == {}
