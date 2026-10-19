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

"""Arctic Platform result normalization covers all expert-parallel shards."""

import asyncio
import contextlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from arctic_platform.correctness.diagnostics.execution_controls import pack_microbatches
from arctic_platform.correctness.harness.batches import Batch
from arctic_platform.correctness.harness.dss_driver import _gateway_runtime
from arctic_platform.correctness.harness.dss_driver import _globalize_expert_norms
from arctic_platform.correctness.harness.dss_driver import _labels_for_provider
from arctic_platform.correctness.harness.dss_driver import _training_client_config
from arctic_platform.correctness.harness.dss_driver import available_gpus
from arctic_platform.correctness.harness.dss_driver import build_payload
from arctic_platform.correctness.harness.dss_driver import copy_checkpoint_peft_adapter
from arctic_platform.correctness.harness.dss_driver import pack


def test_globalize_expert_norms_replaces_rank_local_value() -> None:
    norms = _globalize_expert_norms(
        {"layers.0.mlp.experts.w1": 3.0, "layers.0.input_layernorm.weight": 7.0},
        {"layers.0.mlp.experts.w1": [3.0, 4.0]},
    )

    assert norms == {
        "layers.0.mlp.experts.w1": 5.0,
        "layers.0.input_layernorm.weight": 7.0,
    }


def test_fwd_bwd_step_reads_rank_owned_norms_from_merged_list() -> None:
    from arctic_platform.correctness.harness.dss_driver import fwd_bwd_step

    class Client:
        def fwd_bwd(self, body):
            return {"avg_loss": 1.25, "metrics": {}}

        def step(self, learning_rate):
            return {
                "metrics": {
                    "gradient_norms_per_param": [{}, {"layers.0.input_layernorm.weight": 2.0}],
                    "gradient_norms_per_expert": [],
                }
            }

    class Job:
        client = Client()

    result = fwd_bwd_step(None, Job(), {}, learning_rate=1e-3)

    assert result.avg_loss == 1.25
    assert result.grad_norms == {"layers.0.input_layernorm.weight": 2.0}


def test_fwd_bwd_step_unwraps_rank_owned_optimizer_manifest() -> None:
    from arctic_platform.correctness.harness.dss_driver import fwd_bwd_step

    class Client:
        def fwd_bwd(self, body):
            return {"avg_loss": 1.25, "metrics": {}}

        def step(self, learning_rate):
            return {
                "metrics": {
                    "optimizer_state_manifest": ["/tmp/manifest.json"],
                }
            }

    class Job:
        client = Client()

    result = fwd_bwd_step(None, Job(), {}, learning_rate=1e-3)

    assert result.optimizer_state_manifest == "/tmp/manifest.json"


def test_optimizer_capture_worker_patches_imported_server_worker_bindings(monkeypatch) -> None:
    import arctic_platform.common as common_package
    from arctic_platform.correctness.harness.optimizer_capture import _patched_deepspeed_worker_bindings

    worker_module = types.ModuleType("arctic_platform.common.deepspeed_worker")
    original_worker = object()
    stale_worker = object()
    capture_worker = object()
    common_server = types.ModuleType("arctic_platform.common.ray_server")
    rl_server = types.ModuleType("arctic_platform.rl.ray_server")
    worker_module.DeepSpeedWorker = original_worker
    common_server.DeepSpeedWorker = stale_worker
    rl_server.DeepSpeedWorker = stale_worker
    monkeypatch.setitem(sys.modules, "arctic_platform.common.deepspeed_worker", worker_module)
    monkeypatch.setattr(common_package, "deepspeed_worker", worker_module, raising=False)
    monkeypatch.setitem(sys.modules, "arctic_platform.common.ray_server", common_server)
    monkeypatch.setitem(sys.modules, "arctic_platform.rl.ray_server", rl_server)

    with _patched_deepspeed_worker_bindings(capture_worker):
        assert worker_module.DeepSpeedWorker is capture_worker
        assert common_server.DeepSpeedWorker is capture_worker
        assert rl_server.DeepSpeedWorker is capture_worker

    assert worker_module.DeepSpeedWorker is original_worker
    assert common_server.DeepSpeedWorker is stale_worker
    assert rl_server.DeepSpeedWorker is stale_worker


def test_optimizer_capture_server_patches_workers_inside_actor(monkeypatch) -> None:
    import arctic_platform.common as common_package
    from arctic_platform.correctness.harness import optimizer_capture

    worker_module = types.ModuleType("arctic_platform.common.deepspeed_worker")
    server_module = types.ModuleType("arctic_platform.common.ray_server")
    original_worker, capture_worker = object(), object()
    worker_module.DeepSpeedWorker = original_worker
    server_module.DeepSpeedWorker = original_worker
    events = []

    class ServerState:
        async def initialize(self, payload):
            events.append((worker_module.DeepSpeedWorker, server_module.DeepSpeedWorker))
            if payload.get("fail"):
                raise RuntimeError("initialize failed")
            return {"job_id": 7}

    server_module.ArcticRLRayServerState = ServerState
    server_module.ArcticRLRayServer = type("RequestServer", (), {})
    monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)
    monkeypatch.setitem(sys.modules, server_module.__name__, server_module)
    monkeypatch.setattr(common_package, "deepspeed_worker", worker_module, raising=False)
    monkeypatch.setattr(common_package, "ray_server", server_module, raising=False)
    monkeypatch.setattr(optimizer_capture, "_optimizer_capture_worker_class", lambda: capture_worker)

    with optimizer_capture.optimizer_capture_worker():
        capture_state = server_module.ArcticRLRayServerState
    assert server_module.ArcticRLRayServerState is ServerState
    # A fresh actor imports production bindings, not the driver's patched globals.
    actor = capture_state()
    assert asyncio.run(actor.initialize({})) == {"job_id": 7}
    with pytest.raises(RuntimeError, match="initialize failed"):
        asyncio.run(actor.initialize({"fail": True}))

    assert events == [(capture_worker, capture_worker)] * 2
    assert worker_module.DeepSpeedWorker is original_worker
    assert server_module.DeepSpeedWorker is original_worker


def test_prime_rl_receives_prediction_aligned_labels() -> None:
    batch = Batch(
        name="one",
        input_ids=torch.tensor([[4, 5, 6]]),
        position_ids=torch.tensor([[0, 1, 2]]),
        labels=torch.tensor([[4, 5, 6]]),
    )

    assert _labels_for_provider(batch, "prime_rl").tolist() == [[5, 6, -100]]
    assert _labels_for_provider(batch, "huggingface").tolist() == [[4, 5, 6]]


def test_prime_rl_uses_fused_model_loss_with_aligned_labels() -> None:
    batch = Batch(
        name="one",
        input_ids=torch.tensor([[4, 5, 6]]),
        position_ids=torch.tensor([[0, 1, 2]]),
        labels=torch.tensor([[4, 5, 6]]),
    )

    from arctic_platform.correctness.harness.dss_driver import pack

    prime = pack(batch, model_provider="prime_rl")
    assert prime["processing"]["loss_fn"] == "sft"
    assert prime["meta"]["labels_are_shifted"] is True
    assert prime["batch"]["labels"].tolist() == [[5, 6, -100]]
    assert pack(batch, model_provider="huggingface")["processing"]["loss_fn"] == "sft"


def test_pack_microbatches_emits_exact_gas_list() -> None:
    batch = Batch(
        name="gas4",
        input_ids=torch.arange(15).view(5, 3),
        position_ids=torch.arange(3).unsqueeze(0).expand(5, -1).contiguous(),
        labels=torch.arange(15).view(5, 3),
    )

    body = pack_microbatches(batch, 4, model_provider="huggingface")

    assert body["meta"] == {}
    assert [micro["input_ids"].shape[0] for micro in body["batch"]] == [2, 1, 1, 1]
    assert body["batch"][0]["input_ids"].tolist() == [[0, 1, 2], [3, 4, 5]]
    assert body["batch"][-1]["labels"].tolist() == [[12, 13, 14]]


def test_pack_microbatches_preserves_prime_rl_shifted_labels() -> None:
    batch = Batch(
        name="prime",
        input_ids=torch.tensor([[4, 5, 6], [7, 8, 9]]),
        position_ids=torch.arange(3).unsqueeze(0).expand(2, -1).contiguous(),
        labels=torch.tensor([[4, 5, 6], [7, 8, 9]]),
    )

    body = pack_microbatches(batch, 2, model_provider="prime_rl")

    assert body["meta"]["labels_are_shifted"] is True
    assert body["batch"][0]["labels"].tolist() == [[5, 6, -100]]
    assert body["batch"][1]["labels"].tolist() == [[8, 9, -100]]


def test_pack_microbatches_can_request_memory_sft_ce_without_mutating_config() -> None:
    batch = Batch(
        name="gas4",
        input_ids=torch.arange(12).view(4, 3),
        position_ids=torch.arange(3).unsqueeze(0).expand(4, -1).contiguous(),
        labels=torch.arange(12).view(4, 3),
    )
    processing_config = {
        "logits_optimization": "memory",
        "logits_optimization_peak_mem_size_in_gib": 4,
    }

    body = pack_microbatches(batch, 4, loss_fn="sft_ce", processing_config=processing_config)
    processing_config["logits_optimization"] = "compute"

    assert body["processing"] == {
        "loss_fn": "sft_ce",
        "config": {
            "logits_optimization": "memory",
            "logits_optimization_peak_mem_size_in_gib": 4,
        },
    }


def test_normal_packing_uses_plain_supervised_loss() -> None:
    from arctic_platform.correctness.harness.batches import build_batch

    body = pack(build_batch("gas4", 16, 8, 128))
    assert isinstance(body["batch"], dict)
    assert body["processing"] == {"loss_fn": "sft"}


def test_reference_runner_passes_the_harness_seed(tmp_path, monkeypatch) -> None:
    from arctic_platform.correctness.harness import runner

    output = tmp_path / "reference.json"
    captured = {}

    def run_subprocess(command, **kwargs):
        captured["command"] = command
        output.write_text('{"loss": 0.0}')
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(runner.subprocess, "run", run_subprocess)
    result = runner.run_reference(
        "model",
        tmp_path / "batch.pt",
        output,
        token_budget=1,
        ce_chunk=1,
        attn="sdpa",
        fp32_lm_head=False,
        seed=1234,
        peft_adapter_path=tmp_path / "adapter",
    )

    seed_index = captured["command"].index("--seed")
    adapter_index = captured["command"].index("--peft-adapter")
    assert captured["command"][seed_index + 1] == "1234"
    assert captured["command"][adapter_index + 1] == str(tmp_path / "adapter")
    assert result == {"loss": 0.0}


@pytest.mark.parametrize("unsafe", ["/tmp", "/code", "/data-fast-shadow", "/data-fast", "symlink"])
@pytest.mark.parametrize("boundary", ["execute", "execute_rl", "adapter", "reference_optimizer"])
def test_runner_requires_data_fast_before_operational_side_effects(tmp_path, monkeypatch, unsafe, boundary):
    from arctic_platform.correctness.harness import runner

    destination = Path(unsafe)
    if unsafe == "symlink":
        destination = tmp_path / "escape"
        destination.symlink_to("/tmp", target_is_directory=True)
        destination /= "output"
    cfg = SimpleNamespace(
        training={"peft_config": {"peft_type": "LORA"}},
        n_gpus=1,
        at_gpu_width=lambda *args: pytest.fail("unsafe workdir reached config preparation"),
    )
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("unsafe output reached subprocess")
    )
    with pytest.raises(ValueError, match="/data-fast"):
        if boundary == "execute":
            runner.execute(cfg, None, [], vocab_size=128, workdir=destination)
        elif boundary == "execute_rl":
            runner.execute_rl(cfg, None, vocab_size=128, workdir=destination)
        elif boundary == "adapter":
            runner.materialize_dss_peft_adapter(cfg, "read-only-model", destination, attn_implementation="eager")
        else:
            runner.run_reference(
                "read-only-model",
                tmp_path / "batch.pt",
                tmp_path / "reference.json",
                token_budget=1,
                ce_chunk=1,
                attn="sdpa",
                fp32_lm_head=False,
                optimizer_output_dir=destination,
            )
    assert not (tmp_path / "reference.json").exists()


def test_reference_allows_small_repo_report_with_data_fast_optimizer_output(tmp_path, monkeypatch):
    from arctic_platform.correctness.harness import runner

    output = Path(__file__).resolve().parents[3] / ".codex-run-state" / f"reference-report-{tmp_path.name}.json"
    optimizer_output = tmp_path / "optimizer"
    commands = []

    def run_subprocess(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    original_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == output:
            return '{"loss": 0.0}'
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", run_subprocess)
    monkeypatch.setattr(Path, "read_text", read_text)
    assert not output.exists()
    result = runner.run_reference(
        "read-only-model",
        tmp_path / "batch.pt",
        output,
        token_budget=1,
        ce_chunk=1,
        attn="sdpa",
        fp32_lm_head=False,
        optimizer_config={"name": "AdamW"},
        learning_rate=0.01,
        optimizer_output_dir=optimizer_output,
    )
    assert result == {"loss": 0.0}
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--out") + 1] == str(output)
    assert command[command.index("--optimizer-output-dir") + 1] == str(optimizer_output)
    assert not output.exists()


def test_copy_checkpoint_peft_adapter_accepts_a_checkpoint_adapter_pack(tmp_path) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    source = checkpoint_root / "global_step0" / "default"
    source.mkdir(parents=True)
    (source / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    save_file({"base_model.model.q_proj.lora_A.weight": torch.ones(2, 2)}, source / "adapter_model.safetensors")
    destination = tmp_path / "adapter"

    assert copy_checkpoint_peft_adapter(checkpoint_root, destination) == destination
    assert (destination / "adapter_model.safetensors").read_bytes() == (
        source / "adapter_model.safetensors"
    ).read_bytes()


def test_copy_checkpoint_peft_adapter_accepts_a_weights_only_hf_pack(tmp_path) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    source = checkpoint_root / "hf"
    source.mkdir(parents=True)
    (source / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    save_file({"base_model.model.q_proj.lora_A.weight": torch.ones(2, 2)}, source / "adapter_model.safetensors")
    destination = tmp_path / "adapter"

    assert copy_checkpoint_peft_adapter(checkpoint_root, destination) == destination
    assert (destination / "adapter_model.safetensors").read_bytes() == (
        source / "adapter_model.safetensors"
    ).read_bytes()


def test_copy_checkpoint_peft_adapter_rejects_empty_tensors_before_copy(tmp_path) -> None:
    source = tmp_path / "checkpoint" / "hf"
    source.mkdir(parents=True)
    (source / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    save_file({}, source / "adapter_model.safetensors")
    destination = tmp_path / "adapter"
    destination.mkdir()
    (destination / "sentinel").write_text("preserve")

    with pytest.raises(RuntimeError, match="no adapter tensors"):
        copy_checkpoint_peft_adapter(source.parent, destination)
    assert (destination / "sentinel").read_text() == "preserve"


def test_materialize_peft_adapter_accepts_adapter_only_export(tmp_path, monkeypatch) -> None:
    from arctic_platform.correctness.harness import runner

    cfg = SimpleNamespace(training={"peft_config": {"peft_type": "LORA"}}, n_gpus=1)
    cfg.at_gpu_width = lambda width: cfg
    source = tmp_path / "checkpoint" / "hf"
    source.mkdir(parents=True)
    (source / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    save_file({"base_model.model.q_proj.lora_A.weight": torch.ones(2, 2)}, source / "adapter_model.safetensors")
    calls = []

    def save_checkpoint(**kwargs):
        calls.append(kwargs)
        return {"path": str(source.parent), "hf_path": str(source)}

    monkeypatch.setattr(runner, "build_payload", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "gateway", lambda *args: contextlib.nullcontext(object()))
    job = SimpleNamespace(client=SimpleNamespace(save_checkpoint=save_checkpoint))
    monkeypatch.setattr(runner, "running_job", lambda *args: contextlib.nullcontext(job))

    exported = runner.materialize_dss_peft_adapter(cfg, "unused", tmp_path, attn_implementation="eager")

    assert exported == tmp_path / "adapter"
    assert (exported / "adapter_model.safetensors").read_bytes() == (source / "adapter_model.safetensors").read_bytes()
    assert calls == [{"path": str(source.parent), "export_hf": True}]


def test_full_model_weights_only_export_still_requires_model_config(tmp_path) -> None:
    from arctic_platform.correctness.harness.dss_driver import save_weights_only_checkpoint

    client = SimpleNamespace(save_checkpoint=lambda **kwargs: {"hf_path": str(tmp_path), "path": str(tmp_path)})
    with pytest.raises(RuntimeError, match="no model config"):
        save_weights_only_checkpoint(None, SimpleNamespace(client=client), tmp_path)


def test_copy_checkpoint_peft_adapter_rejects_a_full_model_pack(tmp_path) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    source = checkpoint_root / "global_step0" / "default"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"full-model")

    with pytest.raises(RuntimeError, match="no PEFT pack"):
        copy_checkpoint_peft_adapter(checkpoint_root, tmp_path / "adapter")


def test_payload_routes_optimizer_capture_to_debug_for_harness(tmp_path) -> None:
    destination = tmp_path / "optimizer"
    payload = build_payload(
        {"n_gpus": 1, "mb_spec": {"max_tokens_per_mb": 1024}},
        "model",
        42,
        optimizer_state_output_dir=destination,
    )
    client_config = _training_client_config(type("Session", (), {"workdir": tmp_path})(), payload)
    worker_config = client_config.to_onprem("training")["ds_worker_config"]

    assert worker_config["debug"]["optimizer_state_output_dir"] == str(destination.resolve())


@pytest.mark.parametrize(
    "job_config",
    [
        {},
        {"training_config": None, "ds_worker_config": None},
        {"training_config": {"debug": None}, "ds_worker_config": {"debug": {}}},
        {"debug": {"optimizer_state_output_dir": "/ignored"}},
    ],
)
def test_optimizer_capture_debug_config_defaults(job_config) -> None:
    from arctic_platform.correctness.harness.optimizer_capture import _optimizer_capture_debug_config

    assert _optimizer_capture_debug_config(job_config) == {}


def test_optimizer_capture_debug_config_merges_supported_sections_without_mutation() -> None:
    from arctic_platform.correctness.harness.optimizer_capture import _optimizer_capture_debug_config

    training_debug = {"optimizer_state_output_dir": "/training", "model_state_output_dir": "/model"}
    worker_debug = {"optimizer_state_output_dir": "/worker"}
    debug = _optimizer_capture_debug_config(
        {"training_config": {"debug": training_debug}, "ds_worker_config": {"debug": worker_debug}}
    )

    assert debug == {"optimizer_state_output_dir": "/worker", "model_state_output_dir": "/model"}
    debug["optimizer_state_output_dir"] = "/changed"
    assert training_debug == {"optimizer_state_output_dir": "/training", "model_state_output_dir": "/model"}
    assert worker_debug == {"optimizer_state_output_dir": "/worker"}


@pytest.mark.parametrize("section", ["training_config", "ds_worker_config"])
@pytest.mark.parametrize("nested_debug", [False, True], ids=["config", "debug"])
@pytest.mark.parametrize("value", [[], "", False])
def test_optimizer_capture_debug_config_rejects_malformed_sections(section, nested_debug, value) -> None:
    from arctic_platform.correctness.harness.optimizer_capture import _optimizer_capture_debug_config

    config = {section: {"debug": value} if nested_debug else value}
    field = f"{section}.debug" if nested_debug else section
    with pytest.raises(TypeError, match=rf"{field} must be a dict"):
        _optimizer_capture_debug_config(config)


@pytest.fixture
def optimizer_capture_initializer(monkeypatch):
    import arctic_platform.common as common_package
    from arctic_platform.correctness.harness import optimizer_capture

    class Worker:
        def initialize(self, master_addr, job_config):
            self.production_initialization = (master_addr, job_config)
            return True

    worker_module = types.ModuleType("arctic_platform.common.deepspeed_worker")
    worker_module.DeepSpeedWorker = SimpleNamespace(__ray_metadata__=SimpleNamespace(modified_class=Worker))
    ray_module = types.ModuleType("ray")
    ray_module.remote = lambda worker: worker
    monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)
    monkeypatch.setitem(sys.modules, ray_module.__name__, ray_module)
    monkeypatch.setattr(common_package, "deepspeed_worker", worker_module, raising=False)
    monkeypatch.setattr(optimizer_capture, "_OPTIMIZER_CAPTURE_WORKER", None)
    return optimizer_capture._optimizer_capture_worker_class()()


@pytest.mark.parametrize(
    "fields",
    [
        (),
        ("optimizer_state_output_dir",),
        ("model_state_output_dir",),
        ("optimizer_state_output_dir", "model_state_output_dir"),
    ],
    ids=["disabled", "optimizer", "model", "both"],
)
def test_optimizer_capture_initialize_uses_current_client_job_config(optimizer_capture_initializer, tmp_path, fields):
    payload = build_payload({"n_gpus": 1}, "model", 42)
    payload["training_config"]["debug"].update({field: str(tmp_path / field) for field in fields})
    client_config = _training_client_config(SimpleNamespace(workdir=tmp_path), payload)
    job_config = client_config.to_onprem("training")
    worker = optimizer_capture_initializer

    assert worker.initialize("master", job_config) is True
    assert worker._optimizer_state_output_dir == (
        str(tmp_path / "optimizer_state_output_dir") if "optimizer_state_output_dir" in fields else None
    )
    assert worker._model_state_output_dir == (
        str(tmp_path / "model_state_output_dir") if "model_state_output_dir" in fields else None
    )
    assert worker.production_initialization[0] == "master"
    assert worker.production_initialization[1] is job_config


def test_optimizer_capture_initialize_accepts_path_objects(optimizer_capture_initializer, tmp_path):
    worker = optimizer_capture_initializer
    job_config = {
        "training_config": {
            "debug": {
                "optimizer_state_output_dir": tmp_path / "optimizer",
                "model_state_output_dir": tmp_path / "model",
            }
        }
    }

    assert worker.initialize("master", job_config) is True
    assert worker._optimizer_state_output_dir == str(tmp_path / "optimizer")
    assert worker._model_state_output_dir == str(tmp_path / "model")


def test_optimizer_capture_initialize_worker_settings_can_disable_training_capture(optimizer_capture_initializer):
    worker = optimizer_capture_initializer
    job_config = {
        "training_config": {"debug": {"optimizer_state_output_dir": "/training", "model_state_output_dir": "/model"}},
        "ds_worker_config": {"debug": {"optimizer_state_output_dir": None, "model_state_output_dir": None}},
    }

    assert worker.initialize("master", job_config) is True
    assert worker._optimizer_state_output_dir is None
    assert worker._model_state_output_dir is None


@pytest.mark.parametrize("field", ["optimizer_state_output_dir", "model_state_output_dir"])
@pytest.mark.parametrize("value", ["", "relative/path"])
def test_optimizer_capture_initialize_rejects_relative_paths(optimizer_capture_initializer, field, value):
    worker = optimizer_capture_initializer

    with pytest.raises(ValueError, match=rf"debug.{field} must be an absolute path"):
        worker.initialize("master", {"ds_worker_config": {"debug": {field: value}}})
    assert not hasattr(worker, "production_initialization")


@pytest.mark.parametrize("field", ["optimizer_state_output_dir", "model_state_output_dir"])
def test_optimizer_capture_initialize_rejects_non_path_values(optimizer_capture_initializer, field):
    worker = optimizer_capture_initializer

    with pytest.raises(TypeError):
        worker.initialize("master", {"ds_worker_config": {"debug": {field: 123}}})
    assert not hasattr(worker, "production_initialization")


def test_optimizer_capture_worker_context_restores_production_worker() -> None:
    pytest.importorskip("ray")
    from arctic_platform.common import deepspeed_worker as deepspeed_worker_module
    from arctic_platform.correctness.harness.optimizer_capture import _optimizer_capture_worker_class
    from arctic_platform.correctness.harness.optimizer_capture import optimizer_capture_worker

    original = deepspeed_worker_module.DeepSpeedWorker
    capture_worker = _optimizer_capture_worker_class()
    with optimizer_capture_worker():
        assert deepspeed_worker_module.DeepSpeedWorker is capture_worker
    assert deepspeed_worker_module.DeepSpeedWorker is original


def test_payload_can_skip_unused_gradient_norm_collection() -> None:
    payload = build_payload({}, "model", 42, gradient_norms_per_param=False)

    assert payload["training_config"]["debug"]["gradient_norms_per_param"] is False


def test_payload_preserves_configured_deepspeed_accumulation() -> None:
    payload = build_payload(
        {
            "n_gpus": 4,
            "ds_config": {
                "train_batch_size": 4,
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
            },
        },
        "model",
        42,
    )

    ds_config = payload["training_config"]["ds_config"]
    assert ds_config["gradient_accumulation_steps"] == 1
    assert ds_config["train_batch_size"] == 4


def test_payload_caps_large_microbatch_budget_at_64k() -> None:
    payload = build_payload(
        {"mb_spec": {"max_tokens_per_mb": 262144}},
        "model",
        42,
    )

    assert payload["training_config"]["mb_spec"]["max_tokens_per_mb"] == 65536


def test_payload_applies_spec_lm_head_chunk_without_overriding_config() -> None:
    injected = build_payload({}, "model", 42, lm_head_token_chunk_size=2048)
    configured = build_payload(
        {"fused_lm_head_token_chunk_size": 4096},
        "model",
        42,
        lm_head_token_chunk_size=2048,
    )

    assert injected["training_config"]["fused_lm_head_token_chunk_size"] == 2048
    assert configured["training_config"]["fused_lm_head_token_chunk_size"] == 4096


def test_payload_preserves_smaller_microbatch_budget() -> None:
    payload = build_payload(
        {"mb_spec": {"max_tokens_per_mb": 10240}},
        "model",
        42,
    )

    assert payload["training_config"]["mb_spec"]["max_tokens_per_mb"] == 10240


def test_payload_does_not_choose_deepep_chunks() -> None:
    payload = build_payload(
        {"ep_size": 4, "mb_spec": {"max_tokens_per_mb": 10240}},
        "model",
        42,
    )

    assert "deepep_token_chunk_size" not in payload["training_config"]


def test_pool_is_the_local_devices_when_the_harness_owns_the_gateway(monkeypatch) -> None:
    """With no external gateway the harness starts one on this node, so its devices are the whole pool."""
    monkeypatch.delenv("DSS_GATEWAY_URL", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)

    assert available_gpus() == 8


def test_pool_is_the_hostfile_slots_when_an_external_gateway_owns_placement(monkeypatch, tmp_path) -> None:
    """A gateway the harness did not start places jobs across its hostfile, which this node cannot see."""
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("10.0.0.1 slots=8\n10.0.0.2 slots=8\n")
    monkeypatch.setenv("DSS_GATEWAY_URL", "http://gateway:8000")
    monkeypatch.setenv("DSS_GATEWAY_HOSTFILE", str(hostfile))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)

    assert available_gpus() == 16


def test_pool_falls_back_to_local_devices_when_the_named_hostfile_is_absent(monkeypatch, tmp_path) -> None:
    """An unreadable hostfile leaves the pool unknown, so the guard keeps the number it can verify."""
    monkeypatch.setenv("DSS_GATEWAY_URL", "http://gateway:8000")
    monkeypatch.setenv("DSS_GATEWAY_HOSTFILE", str(tmp_path / "missing"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)

    assert available_gpus() == 8


def test_payload_pins_determinism_over_a_config_that_asks_otherwise() -> None:
    payload = build_payload(
        {
            "debug": {"full_determinism": False, "full_determinism_must_comply": True},
            "mb_spec": {"max_tokens_per_mb": 1024},
        },
        "model",
        42,
    )

    debug = payload["training_config"]["debug"]
    assert debug["full_determinism"] is True
    assert debug["full_determinism_must_comply"] is False


def test_payload_determinism_off_clears_a_config_that_asks_for_the_pin() -> None:
    payload = build_payload(
        {
            "debug": {"full_determinism": True, "full_determinism_must_comply": True},
            "mb_spec": {"max_tokens_per_mb": 1024},
        },
        "model",
        42,
        determinism="off",
    )

    debug = payload["training_config"]["debug"]
    assert debug["full_determinism"] is False
    assert debug["full_determinism_must_comply"] is False


@pytest.mark.parametrize("auth_mode", [None, "disabled", "token", "legacy-mode"])
def test_gateway_runtime_cleans_before_and_after_standard_port(monkeypatch, auth_mode) -> None:
    import torch

    from arctic_platform.correctness.harness import dss_driver

    events = []
    monkeypatch.setenv("MASTER_PORT", "12345")
    if auth_mode is None:
        monkeypatch.delenv("RAY_AUTH_MODE", raising=False)
    else:
        monkeypatch.setenv("RAY_AUTH_MODE", auth_mode)
    monkeypatch.setattr(dss_driver, "_stop_and_verify_gateway_runtime", lambda: events.append("clean"))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)

    def start(*, auto_attach, include_peers):
        events.append(
            ("start", auto_attach, include_peers, os.environ["MASTER_PORT"], os.environ.get("RAY_AUTH_MODE"))
        )

    class RayCluster:
        @staticmethod
        def init_ray_cluster(*, auto_attach, include_peers):
            start(auto_attach=auto_attach, include_peers=include_peers)

        @staticmethod
        def _shutdown():
            events.append("shutdown")

    def load_ray_cluster():
        events.append(("load", os.environ["MASTER_PORT"], os.environ.get("RAY_AUTH_MODE")))
        return RayCluster

    monkeypatch.setattr(dss_driver, "_load_gateway_ray_cluster", load_ray_cluster)

    with _gateway_runtime(8):
        events.append(("body", os.environ["MASTER_PORT"], os.environ.get("RAY_AUTH_MODE")))

    assert events == [
        ("load", "29500", auth_mode),
        "clean",
        ("start", False, False, "29500", auth_mode),
        ("body", "29500", auth_mode),
        "shutdown",
        "clean",
    ]
    assert os.environ["MASTER_PORT"] == "12345"
    assert os.environ.get("RAY_AUTH_MODE") == auth_mode


def test_allocation_runtime_command_uses_hostfile_ds_ssh(monkeypatch, tmp_path) -> None:
    from arctic_platform.correctness.harness import dss_driver

    hostfile = tmp_path / "hostfile"
    hostfile.write_text("10.0.0.1 slots=8\n10.0.0.2 slots=8\n")
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setenv("DSS_GATEWAY_HOSTFILE", str(hostfile))
    monkeypatch.setattr(dss_driver.subprocess, "run", run)

    dss_driver._run_on_allocation("ray stop --force")

    assert captured["command"][1:3] == ["-f", str(hostfile)]
    assert captured["command"][0].endswith("/ds_ssh")
    assert captured["command"][3] == "ray stop --force"
    assert captured["kwargs"] == {"capture_output": True, "text": True, "timeout": 180}


def test_runtime_cleanup_stops_ray_then_checks_processes_and_port(monkeypatch) -> None:
    from arctic_platform.correctness.harness import dss_driver

    scripts = []
    monkeypatch.setattr(dss_driver, "_run_on_allocation", lambda script: scripts.append(script))

    dss_driver._stop_and_verify_gateway_runtime()

    assert len(scripts) == 2
    assert scripts[0].endswith("/ray stop --force")
    assert "pgrep -af" in scripts[1]
    assert "29500" in scripts[1]


def test_gateway_runtime_restores_port_when_preflight_fails(monkeypatch) -> None:
    from arctic_platform.correctness.harness import dss_driver

    calls = 0

    def cleanup():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("port is still bound")

    monkeypatch.setenv("MASTER_PORT", "12345")
    monkeypatch.setattr(dss_driver, "_stop_and_verify_gateway_runtime", cleanup)

    with pytest.raises(RuntimeError, match="port is still bound"):
        with _gateway_runtime(1):
            pytest.fail("preflight must fail before launch")

    assert os.environ["MASTER_PORT"] == "12345"


def test_running_job_keeps_cleanup_outside_client_shutdown(monkeypatch, tmp_path) -> None:
    from arctic_platform.correctness.harness import dss_driver

    events = []

    @contextlib.contextmanager
    def runtime(slots):
        events.append(("runtime-enter", slots))
        try:
            yield
        finally:
            events.append("runtime-exit")

    class Client:
        def __init__(self, config):
            events.append("client-start")
            self.jobs = type("Jobs", (), {"training": 7})()

        def shutdown(self):
            events.append("client-shutdown")

    monkeypatch.setattr(dss_driver, "_gateway_runtime", runtime)
    monkeypatch.setattr(dss_driver, "ArcticClient", Client)
    payload = {
        "model_name": "model",
        "job_type": "training",
        "training_config": {"n_gpus": 1, "mb_spec": {"max_tokens_per_mb": 1}},
    }

    with dss_driver.running_job(dss_driver.GatewaySession(tmp_path, 1), payload) as job:
        events.append(("job", job.job_id))

    assert events == [("runtime-enter", 1), "client-start", ("job", 7), "client-shutdown", "runtime-exit"]
