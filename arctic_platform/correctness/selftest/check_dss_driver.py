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

import contextlib
import os

import pytest
import torch

from arctic_platform.correctness.harness.batches import Batch
from arctic_platform.correctness.harness.dss_driver import _gateway_runtime
from arctic_platform.correctness.harness.dss_driver import _globalize_expert_norms
from arctic_platform.correctness.harness.dss_driver import _labels_for_provider
from arctic_platform.correctness.harness.dss_driver import _training_client_config
from arctic_platform.correctness.harness.dss_driver import available_gpus
from arctic_platform.correctness.harness.dss_driver import build_payload
from arctic_platform.correctness.harness.dss_driver import copy_checkpoint_peft_adapter
from arctic_platform.correctness.harness.dss_driver import export_initial_peft_adapter


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


def test_export_initial_peft_adapter_accepts_the_live_adapter_pack(tmp_path) -> None:
    destination = tmp_path / "adapter"
    destination.mkdir()
    (destination / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    (destination / "adapter_model.safetensors").write_bytes(b"adapter")

    assert export_initial_peft_adapter(destination) == destination


def test_copy_checkpoint_peft_adapter_accepts_a_checkpoint_adapter_pack(tmp_path) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    source = checkpoint_root / "global_step0" / "default"
    source.mkdir(parents=True)
    (source / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    (source / "adapter_model.safetensors").write_bytes(b"adapter")
    destination = tmp_path / "adapter"

    assert copy_checkpoint_peft_adapter(checkpoint_root, destination) == destination
    assert (destination / "adapter_model.safetensors").read_bytes() == b"adapter"


def test_copy_checkpoint_peft_adapter_rejects_a_full_model_pack(tmp_path) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    source = checkpoint_root / "global_step0" / "default"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"full-model")

    with pytest.raises(RuntimeError, match="no PEFT pack"):
        copy_checkpoint_peft_adapter(checkpoint_root, tmp_path / "adapter")


def test_payload_routes_initial_adapter_export_to_the_worker(tmp_path) -> None:
    destination = tmp_path / "adapter"
    payload = build_payload(
        {"n_gpus": 1, "mb_spec": {"max_tokens_per_mb": 1024}},
        "model",
        42,
        initial_peft_adapter_output_dir=destination,
    )
    client_config = _training_client_config(type("Session", (), {"workdir": tmp_path})(), payload)
    worker_config = client_config.to_onprem("training")["ds_worker_config"]

    assert worker_config["debug"]["initial_peft_adapter_output_dir"] == str(destination.resolve())


def test_payload_can_skip_unused_gradient_norm_collection() -> None:
    payload = build_payload({}, "model", 42, gradient_norms_per_param=False)

    assert payload["training_config"]["debug"]["gradient_norms_per_param"] is False


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


def test_payload_uses_microbatch_budget_for_deepep_chunks() -> None:
    payload = build_payload(
        {"ep_size": 4, "mb_spec": {"max_tokens_per_mb": 10240}},
        "model",
        42,
    )

    assert payload["training_config"]["deepep_token_chunk_size"] == 10240


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


def test_gateway_runtime_cleans_before_and_after_standard_port(monkeypatch) -> None:
    import torch

    from arctic_platform.common import ray_cluster
    from arctic_platform.correctness.harness import dss_driver

    events = []
    monkeypatch.setenv("MASTER_PORT", "12345")
    monkeypatch.setenv("RAY_AUTH_MODE", "legacy-mode")
    monkeypatch.setattr(dss_driver, "_stop_and_verify_gateway_runtime", lambda: events.append("clean"))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)

    def start(*, auto_attach, include_peers):
        events.append(("start", auto_attach, include_peers, os.environ["MASTER_PORT"], os.environ["RAY_AUTH_MODE"]))

    monkeypatch.setattr(ray_cluster, "init_ray_cluster", start)
    monkeypatch.setattr(ray_cluster, "_shutdown", lambda: events.append("shutdown"))

    with _gateway_runtime(8):
        events.append(("body", os.environ["MASTER_PORT"], os.environ["RAY_AUTH_MODE"]))

    assert events == [
        "clean",
        ("start", False, False, "29500", "token"),
        ("body", "29500", "token"),
        "shutdown",
        "clean",
    ]
    assert os.environ["MASTER_PORT"] == "12345"
    assert os.environ["RAY_AUTH_MODE"] == "legacy-mode"


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
