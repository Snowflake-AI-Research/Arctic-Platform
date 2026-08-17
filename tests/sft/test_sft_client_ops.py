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
"""ArcticSFTClient op surface — FakeTransport, no live backend or GPUs."""

from __future__ import annotations

import pytest

from arctic_platform.client import JobHandles
from arctic_platform.client import Request
from arctic_platform.client import Transport
from arctic_platform.sft import ArcticSFTClient
from arctic_platform.sft import ArcticSFTClientConfig
from arctic_platform.sft import client as sft_client_module

TRAINING = 1
SAMPLING = 2


class FakeTransport(Transport):
    def __init__(self, config) -> None:
        self.config = config
        self.jobs = JobHandles()
        self.calls: list[Request] = []
        self.shutdown_calls = 0

    def initialize(self) -> JobHandles:
        sampling = SAMPLING if getattr(self.config, "sampling_gpus", 0) else None
        self.jobs = JobHandles(training=TRAINING, sampling=sampling)
        return self.jobs

    def call(self, request: Request) -> dict:
        self.calls.append(request)
        return {"avg_loss": 0.5, "metrics": {"loss": 0.5}, "results": ["ok"], "status": "ok"}

    async def acall(self, request: Request) -> dict:
        return self.call(request)

    def shutdown(self) -> None:
        self.shutdown_calls += 1


@pytest.fixture
def client(monkeypatch) -> ArcticSFTClient:
    monkeypatch.setattr(sft_client_module, "_make_transport", FakeTransport)
    cfg = ArcticSFTClientConfig(model_name="m", training_gpus=2, checkpoint_path="/tmp/c")
    return ArcticSFTClient(cfg)


@pytest.fixture
def sampling_client(monkeypatch) -> ArcticSFTClient:
    monkeypatch.setattr(sft_client_module, "_make_transport", FakeTransport)
    cfg = ArcticSFTClientConfig(
        model_name="m",
        training_gpus=1,
        sampling_gpus=1,
        colocate=True,
        checkpoint_path="/tmp/c",
    )
    return ArcticSFTClient(cfg)


def _last(client: ArcticSFTClient) -> Request:
    return client.transport.calls[-1]


class TestSFTOpMapping:
    def test_fwd_bwd_defaults_sft_loss_fn(self, client):
        client.fwd_bwd({"batch": {"input_ids": [1]}, "meta": {}})
        req = _last(client)
        assert req.op == "forward-backward"
        assert req.job_id == TRAINING
        assert req.binary is True
        assert req.body["processing"] == {"loss_fn": "sft"}
        assert req.body["meta"] == {}

    def test_fwd_bwd_folds_explicit_processing(self, client):
        client.fwd_bwd({"batch": {}}, processing={"loss_fn": "sft_ce"})
        assert _last(client).body["processing"] == {"loss_fn": "sft_ce"}

    def test_fwd_no_grad_targets_training_binary(self, client):
        client.fwd_no_grad({"batch": {"input_ids": [1]}})
        req = _last(client)
        assert req.op == "forward"
        assert req.job_id == TRAINING
        assert req.binary is True
        assert req.body["processing"] == {"loss_fn": "sft"}

    def test_step_targets_training(self, client):
        client.step()
        req = _last(client)
        assert req.op == "step"
        assert req.job_id == TRAINING
        assert req.binary is False
        # LR is server-authoritative (init/scheduler), so step sends no override.
        assert req.body == {}

    def test_save_checkpoint_targets_training(self, client):
        client.save_checkpoint(path="/tmp/ckpt", step=3, export_hf=True, save_total_limit=2)
        req = _last(client)
        assert req.op == "save"
        assert req.job_id == TRAINING
        assert req.body == {
            "path": "/tmp/ckpt",
            "step": 3,
            "export_hf": True,
            "save_total_limit": 2,
        }

    def test_load_checkpoint_targets_training(self, client):
        client.load_checkpoint(path="/tmp/ckpt", step=3)
        req = _last(client)
        assert req.op == "load-checkpoint"
        assert req.job_id == TRAINING
        assert req.body == {"path": "/tmp/ckpt", "step": 3}

    def test_generate_targets_sampling(self, sampling_client):
        out = sampling_client.generate(["hi"], sampling_params={"max_tokens": 8})
        assert out == ["ok"]
        req = _last(sampling_client)
        assert req.op == "generate"
        assert req.job_id == SAMPLING
        assert req.body["prompts"] == ["hi"]
        assert req.body["sampling_params"] == {"max_tokens": 8}

    def test_sync_weights_stages_wake_and_reset(self, sampling_client):
        # Explicit cuda_ipc is a per-call override (present); low_memory is left to the job's init-time default
        # (omitted); colocate is never on the wire.
        sampling_client.sync_weights(cuda_ipc=True)
        ops = [c.op for c in sampling_client.transport.calls]
        assert ops == [
            "wake-inference",
            "operation",
            "wake-inference",
            "operation",
        ]
        sync_req = sampling_client.transport.calls[1]
        assert sync_req.job_id == TRAINING
        assert sync_req.body["operation_type"] == "weight-sync"
        payload = sync_req.body["payload"]
        assert payload["cuda_ipc"] is True
        assert "colocate" not in payload
        assert "low_memory" not in payload
        assert payload["source_sub_job_id"] == TRAINING
        assert payload["target_sub_job_ids"] == [SAMPLING]

    def test_sync_weights_body_omits_strategy_by_default(self, sampling_client):
        """No override → only job ids on the wire; the server uses the strategy baked onto the training job at init."""
        sampling_client.sync_weights()
        payload = sampling_client.transport.calls[1].body["payload"]
        assert payload == {"source_sub_job_id": TRAINING, "target_sub_job_ids": [SAMPLING]}


class TestSFTLifecycle:
    def test_reconnect_config_copies_training_job_id(self, client):
        cfg = client.reconnect_config()
        assert cfg.training_job_id == TRAINING
        assert cfg.sampling_job_id is None
        assert cfg.model_name == "m"

    def test_reconnect_config_copies_sampling_job_id(self, sampling_client):
        cfg = sampling_client.reconnect_config()
        assert cfg.training_job_id == TRAINING
        assert cfg.sampling_job_id == SAMPLING

    def test_context_manager_shuts_down(self, monkeypatch):
        monkeypatch.setattr(sft_client_module, "_make_transport", FakeTransport)
        cfg = ArcticSFTClientConfig(model_name="m", training_gpus=1, checkpoint_path="/tmp/c")
        with ArcticSFTClient(cfg) as c:
            assert c.jobs.training == TRAINING
        assert c.transport.shutdown_calls == 1

    def test_initialize_failure_shuts_down_transport(self, monkeypatch):
        """A launched local server must not be orphaned if initialize() fails."""

        class BoomTransport(FakeTransport):
            def initialize(self):
                raise RuntimeError("init failed")

        monkeypatch.setattr(sft_client_module, "_make_transport", BoomTransport)
        cfg = ArcticSFTClientConfig(model_name="m", training_gpus=1, checkpoint_path="/tmp/c")
        with pytest.raises(RuntimeError, match="init failed"):
            ArcticSFTClient(cfg)


class TestSFTTransportSelection:
    def test_make_transport_selects_http(self):
        from arctic_platform.client.transports.onprem_http import HttpTransport

        cfg = ArcticSFTClientConfig(model_name="m", training_gpus=1, checkpoint_path="/tmp/c", comm_protocol="http")
        assert isinstance(sft_client_module._make_transport(cfg), HttpTransport)

    def test_make_transport_selects_ray(self, monkeypatch):
        import arctic_platform.client.transports.onprem_ray as ray_mod

        class DummyRay:
            def __init__(self, config):
                self.config = config

        monkeypatch.setattr(ray_mod, "RayTransport", DummyRay)
        cfg = ArcticSFTClientConfig(model_name="m", training_gpus=1, checkpoint_path="/tmp/c", comm_protocol="ray")
        assert isinstance(sft_client_module._make_transport(cfg), DummyRay)

    def test_to_rl_config_disables_sampling_and_log_prob(self):
        cfg = ArcticSFTClientConfig(
            model_name="m",
            training_gpus=2,
            host="h",
            port=9,
            launch_local_server=True,
            server_cuda_visible_devices="0,1",
            checkpoint_path="/tmp/c",
        )
        rl = cfg.to_rl_config()
        assert rl.training_gpus == 2
        assert rl.sampling_gpus == 0
        assert rl.log_prob_gpus == 0
        assert rl.backend_config.host == "h"
        assert rl.backend_config.port == 9
        assert rl.backend_config.launch_local_server is True
        assert rl.backend_config.server_cuda_visible_devices == "0,1"
        assert rl.training.checkpoint_path == "/tmp/c"

    def test_to_rl_config_routes_weight_sync_strategy(self):
        cfg = ArcticSFTClientConfig(
            model_name="m",
            training_gpus=1,
            sampling_gpus=1,
            colocate=True,
            cuda_ipc=True,
            low_memory=True,
            checkpoint_path="/tmp/c",
        )
        rl = cfg.to_rl_config()
        assert rl.training.cuda_ipc is True
        assert rl.training.low_memory is True
