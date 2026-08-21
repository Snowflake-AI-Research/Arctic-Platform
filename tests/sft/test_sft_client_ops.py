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
"""ArcticSFTClient op surface — FakeTransport, no live backend or GPUs.

ArcticSFTClient shares every op with ArcticClient; what is asserted here is the
SFT-specific part (the ``loss_fn: sft`` default on the forward bodies) plus the
shared ops as an SFT caller sees them.
"""

from __future__ import annotations

import pytest

from arctic_platform.client import ArcticSFTClientConfig
from arctic_platform.client import JobHandles
from arctic_platform.client import OnPremConfig
from arctic_platform.client import Request
from arctic_platform.client import TrainingConfig
from arctic_platform.client import Transport
from arctic_platform.client import base as base_module
from arctic_platform.sft import ArcticSFTClient

TRAINING = 1
SAMPLING = 2


class FakeTransport(Transport):
    def __init__(self, config, server_state=None) -> None:
        self.config = config
        self.jobs = JobHandles()
        self.calls: list[Request] = []
        self.shutdown_calls = 0

    def initialize(self) -> JobHandles:
        sampling = SAMPLING if self.config.sampling_gpus else None
        self.jobs = JobHandles(training=TRAINING, sampling=sampling)
        return self.jobs

    def call(self, request: Request) -> dict:
        self.calls.append(request)
        return {"avg_loss": 0.5, "metrics": {"loss": 0.5}, "results": ["ok"], "status": "ok"}

    async def acall(self, request: Request) -> dict:
        return self.call(request)

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _config(**kwargs) -> ArcticSFTClientConfig:
    kwargs.setdefault("training_gpus", 2)
    kwargs.setdefault("backend", OnPremConfig())
    kwargs.setdefault("training", TrainingConfig(checkpoint_path="/tmp/c"))
    return ArcticSFTClientConfig(model_name="m", **kwargs)


@pytest.fixture
def client(monkeypatch) -> ArcticSFTClient:
    monkeypatch.setattr(base_module, "make_transport", FakeTransport)
    return ArcticSFTClient(_config())


@pytest.fixture
def sampling_client(monkeypatch) -> ArcticSFTClient:
    monkeypatch.setattr(base_module, "make_transport", FakeTransport)
    cfg = _config(training_gpus=1, sampling_gpus=1, backend=OnPremConfig(colocate=True))
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

    def test_fwd_bwd_keeps_processing_from_the_batch(self, client):
        """A batch that already carries `processing` is not overwritten by the sft default."""
        client.fwd_bwd({"batch": {}, "processing": {"loss_fn": "sft_ce"}})
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
            "checkpoint_id": None,
            "checkpoint_type": "resumable",
            "path": "/tmp/ckpt",
            "step": 3,
            "export_hf": True,
            "save_total_limit": 2,
            "stage_info": None,
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
        monkeypatch.setattr(base_module, "make_transport", FakeTransport)
        with ArcticSFTClient(_config(training_gpus=1)) as c:
            assert c.jobs.training == TRAINING
        assert c.transport.shutdown_calls == 1

    def test_initialize_failure_shuts_down_transport(self, monkeypatch):
        """A launched local server must not be orphaned if initialize() fails."""

        class BoomTransport(FakeTransport):
            def initialize(self):
                raise RuntimeError("init failed")

        monkeypatch.setattr(base_module, "make_transport", BoomTransport)
        with pytest.raises(RuntimeError, match="init failed"):
            ArcticSFTClient(_config(training_gpus=1))
