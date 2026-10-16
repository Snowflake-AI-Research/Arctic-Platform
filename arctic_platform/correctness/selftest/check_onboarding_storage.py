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

"""Onboarding rejects heavyweight storage outside fast disk before doing work."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arctic_platform.correctness.harness import workdir
from arctic_platform.correctness.onboarding import onboard
from arctic_platform.correctness.onboarding import synth_model


@pytest.fixture
def fast_root(tmp_path, monkeypatch):
    root = tmp_path / "data-fast"
    root.mkdir()
    monkeypatch.setattr(workdir, "_DATA_FAST_ROOT", root)
    monkeypatch.setattr(workdir, "_DATA_FAST_TMPDIR", root / "tmp")
    monkeypatch.delenv(workdir.CORRECTNESS_TMPDIR_ENV, raising=False)
    return root


@pytest.fixture(params=["code", "tmp", "root", "sibling", "traversal", "relative", "symlink"])
def invalid_destination(request, fast_root, monkeypatch):
    outside = fast_root.parent / "outside"
    outside.mkdir()
    monkeypatch.chdir(fast_root.parent)
    if request.param == "symlink":
        (fast_root / "escape").symlink_to(outside, target_is_directory=True)
    return {
        "code": Path("/code/users/stas/onboarding-storage-test"),
        "tmp": Path("/tmp/onboarding-storage-test"),
        "root": fast_root,
        "sibling": fast_root.with_name("data-fast-other") / "model",
        "traversal": fast_root / ".." / "outside" / "model",
        "relative": Path("outside/model"),
        "symlink": fast_root / "escape" / "model",
    }[request.param]


def _unexpected(*args, **kwargs):
    pytest.fail("invalid storage must be rejected before allocation or model/config/copy work")


@pytest.mark.parametrize("operation", ["materialize", "materialize_pretrained", "copy_sidecars"])
def test_model_destination_rejected_before_config_or_sidecar_resolution(
    fast_root, invalid_destination, monkeypatch, operation
):
    monkeypatch.setattr(synth_model, "reduce_config", _unexpected)
    monkeypatch.setattr(synth_model, "sidecar_files", _unexpected)
    monkeypatch.setattr(synth_model, "count_parameters", _unexpected)
    with pytest.raises(ValueError, match="destination.* /data-fast"):
        if operation == "copy_sidecars":
            synth_model.copy_sidecars("not-a-local-source", invalid_destination)
        else:
            getattr(synth_model, operation)("not-a-local-source", str(invalid_destination), 4)
    assert list((fast_root.parent / "outside").iterdir()) == []


@pytest.mark.parametrize("field", ["cache_root", "workdir"])
def test_build_spec_rejects_paths_before_allocation_or_model_work(fast_root, invalid_destination, monkeypatch, field):
    monkeypatch.setattr(onboard, "correctness_workdir", _unexpected)
    monkeypatch.setattr(synth_model, "size_for_single_gpu", _unexpected)
    monkeypatch.setattr(synth_model, "materialize_pretrained", _unexpected)
    kwargs = {"cache_root": str(fast_root / "cache"), "workdir": fast_root / "work"}
    kwargs[field] = invalid_destination
    with pytest.raises(ValueError, match="/data-fast"):
        onboard.build_spec(SimpleNamespace(), "unused", arm_defs=[], **kwargs)
    assert not (fast_root / "work").exists()
    assert not (fast_root / "cache").exists()


@pytest.mark.parametrize("entrypoint", ["run", "_parent", "_probe_worker"])
@pytest.mark.parametrize("field", ["cache_root", "output_dir"])
def test_onboarding_entrypoints_reject_storage_before_checkout_or_config(
    fast_root, invalid_destination, monkeypatch, entrypoint, field
):
    monkeypatch.setattr(onboard, "_checkout_root", _unexpected)
    args = argparse.Namespace(cache_root=fast_root / "cache", output_dir=fast_root / "work")
    setattr(args, field, invalid_destination)
    with pytest.raises(ValueError, match="/data-fast"):
        getattr(onboard, entrypoint)(args)
    assert not (fast_root / "work").exists()


@pytest.mark.parametrize("field", ["output_dir", "optimizer_model_path"])
def test_optimizer_calibration_rejects_storage_before_config_batch_or_model(fast_root, invalid_destination, field):
    args = argparse.Namespace(output_dir=fast_root / "work", optimizer_model_path=fast_root / "model")
    setattr(args, field, invalid_destination)
    # No config/spec attributes: reaching any calibration work would fail this test.
    with pytest.raises(ValueError, match="/data-fast"):
        onboard._run_optimizer_calibration(args)
    assert not (fast_root / "work").exists()


@pytest.mark.parametrize("operation", ["materialize", "materialize_pretrained"])
def test_materializers_forward_canonical_fast_destination(fast_root, tmp_path, monkeypatch, operation):
    import transformers

    source = tmp_path / "source"
    source.mkdir()
    (source / "tokenizer_config.json").write_text("{}")
    target = fast_root / "models"
    target.mkdir()
    (fast_root / "alias").symlink_to(target, target_is_directory=True)
    destination = fast_root / "alias" / "reduced"
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


def test_copy_sidecars_canonical_fast_destination_and_idempotency(fast_root, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "tokenizer_config.json").write_text('{"version": 1}')
    target = fast_root / "target"
    target.mkdir()
    (fast_root / "alias").symlink_to(target, target_is_directory=True)
    for _ in range(2):
        copied = synth_model.copy_sidecars(str(source), fast_root / "alias")
        assert copied == [target / "tokenizer_config.json"]
        assert copied[0].read_text() == '{"version": 1}'


def test_build_spec_accepts_fast_work_cache_and_separate_spec_output(fast_root, tmp_path, monkeypatch):
    import transformers

    from arctic_platform.correctness.harness import config as config_module
    from arctic_platform.correctness.harness.spec import ModelSpec

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
        cfg, "source/model", str(fast_root / "cache"), [], num_layers=4, workdir=fast_root / "work"
    )
    assert captured == [fast_root / "cache" / "model-4L"]
    assert (fast_root / "work").is_dir()
    separate_spec = tmp_path / "checkout" / "spec.json"
    spec.write(separate_spec)
    assert json.loads(separate_spec.read_text())["model"]["cache_path"] == str(captured[0])


@pytest.mark.parametrize("probe", [False, True])
def test_run_keeps_separate_small_spec_and_probe_reports_unrestricted(fast_root, tmp_path, monkeypatch, probe):
    args = argparse.Namespace(
        config=tmp_path / "checkout" / "input.config",
        cache_root=fast_root / "cache",
        output_dir=fast_root / "work",
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


def test_optimizer_calibration_missing_fast_root_never_falls_back(fast_root, monkeypatch):
    fast_root.rmdir()
    args = argparse.Namespace(output_dir=fast_root / "work", optimizer_model_path=fast_root / "model")
    with pytest.raises(ValueError, match="existing /data-fast"):
        onboard._run_optimizer_calibration(args)
    assert not fast_root.exists()


@pytest.mark.parametrize("fail_reference", [False, True])
def test_optimizer_calibration_allocates_fresh_root_and_cleans_only_owned_tree(
    fast_root, tmp_path, monkeypatch, fail_reference
):
    import torch
    import transformers

    from arctic_platform.correctness.checks import optimizer_state
    from arctic_platform.correctness.harness import batches
    from arctic_platform.correctness.harness import config as config_module
    from arctic_platform.correctness.harness import runner
    from arctic_platform.correctness.harness import spec as spec_module

    output = fast_root / "output"
    output.mkdir()
    prior = fast_root / "dss-correctness-calibration" / "test"
    prior.mkdir(parents=True)
    sentinel = prior / "sentinel"
    sentinel.write_text("preexisting")
    args = argparse.Namespace(
        output_dir=output,
        optimizer_model_path=fast_root / "model",
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
    assert all(path.parent == fast_root / "tmp" and not path.exists() for path in created)
    assert sentinel.read_text() == "preexisting"
