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

"""Direct correctness helpers validate write destinations without restricting source reads."""

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from arctic_platform.client import CortexConfig
from arctic_platform.correctness.harness import batches
from arctic_platform.correctness.harness import dss_driver
from arctic_platform.correctness.harness import reference_records
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness import workdir
from arctic_platform.correctness.harness.arms import ArmDefinition
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.spec import TestSpec as Spec

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config"
SPEC = ROOT / "specs/qwen3-8b-h200-train-sft-full-4gpus-64k.json"


@pytest.fixture(params=["/tmp", "/code", "/data-fast-shadow", "/data-fast", "symlink"])
def unsafe_destination(request, tmp_path):
    if request.param != "symlink":
        return Path(request.param)
    alias = tmp_path / "escape"
    alias.symlink_to("/tmp", target_is_directory=True)
    return alias / "output"


@pytest.mark.parametrize("helper", ["gateway", "transport", "job", "weights", "adapter", "payload", "batch"])
def test_direct_helpers_reject_unsafe_writes_before_side_effects(tmp_path, monkeypatch, unsafe_destination, helper):
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
            with dss_driver.gateway(unsafe_destination, 1):
                pytest.fail("unsafe gateway entered")
        elif helper == "transport":
            with dss_driver.GatewayTransport(unsafe_destination, {"n_gpus": 1}, "read-only-model", 42, slots=1):
                pytest.fail("unsafe transport entered")
        elif helper == "job":
            with dss_driver.running_job(dss_driver.GatewaySession(unsafe_destination, 1), payload):
                pytest.fail("unsafe job entered")
        elif helper == "weights":
            dss_driver.save_weights_only_checkpoint(None, SimpleNamespace(client=client), unsafe_destination)
        elif helper == "adapter":
            dss_driver.copy_checkpoint_peft_adapter(checkpoint, unsafe_destination)
        elif helper == "payload":
            dss_driver.build_payload(
                {"n_gpus": 1}, "read-only-model", 42, optimizer_state_output_dir=unsafe_destination
            )
        else:
            batches.save(SimpleNamespace(), unsafe_destination)


def test_transport_revalidates_destination_before_enter_mkdir(tmp_path, monkeypatch):
    destination = tmp_path / "transport"
    transport = dss_driver.GatewayTransport(destination, {"n_gpus": 1}, "read-only-model", 42, slots=1)
    destination.symlink_to("/tmp", target_is_directory=True)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: pytest.fail("unsafe transport reached mkdir"))
    with pytest.raises(ValueError, match="/data-fast"):
        with transport:
            pytest.fail("unsafe transport entered")


def test_direct_training_job_rejects_checkpoint_child_escape_before_allocation(tmp_path, monkeypatch):
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
def test_runner_rejects_derived_optimizer_symlink_before_mkdir_or_payload(
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
            load_config(CONFIG),
            Spec.read(SPEC),
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
        assert result.outcome is Outcome.FAIL and result.arm == "gas1" and result.gpus == 4
        assert "/data-fast" in result.reason
        assert result.summary == f"execution error: ValueError: {result.reason}"
        traceback = capsys.readouterr().err
        assert "Traceback (most recent call last):" in traceback
        assert "payload_for" in traceback and "require_data_fast_path" in traceback
        assert f"ValueError: {result.reason}" in traceback
    assert not references


def test_fast_batch_destination_is_canonical_and_read_inputs_remain_unrestricted(tmp_path, monkeypatch):
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


def test_adapter_source_outside_fast_root_is_read_and_safe_destination_behavior_preserved(tmp_path, monkeypatch):
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


def test_remote_training_config_without_local_writes_allows_read_only_workdir():
    backend = CortexConfig(base_url="http://unused")
    session = dss_driver.GatewaySession(Path("/code/read-only"), 1, backend)
    payload = dss_driver.build_payload({"n_gpus": 1}, "/code/read-only-model", 42)
    config = dss_driver._training_client_config(session, payload)
    assert config.training.checkpoint_path is None
    assert config.model_name == "/code/read-only-model"


def test_weights_save_keeps_public_framing_and_does_not_restrict_returned_source(tmp_path, monkeypatch):
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
