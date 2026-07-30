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
"""The op surface of ArcticRLClient must lower to one canonical Request each.

A FakeTransport records the Request every op produces so we can assert the
mapping (target job, body, binary flag) with no live backend or GPUs.
"""

from __future__ import annotations

import pytest

from arctic_platform.client import OPS
from arctic_platform.client import ArcticRLClient
from arctic_platform.client import ArcticRLClientConfig
from arctic_platform.client import JobHandles
from arctic_platform.client import Request
from arctic_platform.client import Transport
from arctic_platform.client import client as client_module
from arctic_platform.client import unresolved_ops
from arctic_platform.client.transport import method_name

TRAINING, SAMPLING, LOG_PROB = 1, 2, 3


class FakeTransport(Transport):
    def __init__(self, config: ArcticRLClientConfig) -> None:
        self.config = config
        self.jobs = JobHandles()
        self.calls: list[Request] = []

    def initialize(self) -> JobHandles:
        self.jobs = JobHandles(training=TRAINING, sampling=SAMPLING, log_prob=LOG_PROB)
        return self.jobs

    def call(self, request: Request) -> dict:
        self.calls.append(request)
        return {"results": ["ok"], "loss": 0.0}

    def shutdown(self) -> None:
        pass


@pytest.fixture
def client(monkeypatch) -> ArcticRLClient:
    monkeypatch.setattr(client_module, "make_transport", FakeTransport)
    cfg = ArcticRLClientConfig(model_name="m", training_gpus=1, sampling_gpus=1, log_prob_gpus=1)
    return ArcticRLClient(cfg)


def _last(client: ArcticRLClient) -> Request:
    return client.transport.calls[-1]


class TestOpMapping:
    def test_fwd_bwd_targets_training_binary(self, client):
        """fwd_bwd -> training job, binary payload, body carries the batch."""
        client.fwd_bwd({"input_ids": [1, 2]})
        req = _last(client)
        assert req.op == "forward-backward"
        assert req.job_id == TRAINING
        assert req.binary is True
        assert req.body["input_ids"] == [1, 2]

    def test_fwd_bwd_folds_processing_and_router_replay(self, client):
        """Non-None processing/router_replay are folded into the body."""
        client.fwd_bwd({"input_ids": [1]}, processing={"p": 1}, router_replay={"r": 2})
        body = _last(client).body
        assert body["processing"] == {"p": 1}
        assert body["router_replay"] == {"r": 2}

    def test_fwd_no_grad_targets_training_binary(self, client):
        """fwd_no_grad -> training job, binary payload."""
        client.fwd_no_grad({"input_ids": [1]})
        req = _last(client)
        assert req.op == "forward"
        assert req.job_id == TRAINING
        assert req.binary is True

    def test_step_targets_training(self, client):
        """step -> training job, learning_rate in body."""
        client.step(learning_rate=0.5)
        req = _last(client)
        assert req.op == "step"
        assert req.job_id == TRAINING
        assert req.binary is False
        assert req.body == {"learning_rate": 0.5}

    def test_save_checkpoint_targets_training(self, client):
        """save_checkpoint -> training job (save op, Cortex-shaped body)."""
        client.save_checkpoint(checkpoint_id="cp1", checkpoint_type="weights-only")
        req = _last(client)
        assert req.op == "save"
        assert req.job_id == TRAINING
        assert req.body == {"checkpoint_id": "cp1", "checkpoint_type": "weights-only"}

    def test_generate_targets_sampling_and_unwraps_results(self, client):
        """generate -> sampling job; returns the 'results' list."""
        out = client.generate(["hi"], sampling_params={"n": 1}, routing_key="k", strict=True)
        req = _last(client)
        assert req.op == "generate"
        assert req.job_id == SAMPLING
        assert req.body == {"prompts": ["hi"], "sampling_params": {"n": 1}, "routing_key": "k", "strict": True}
        assert out == ["ok"]

    def test_log_probs_targets_log_prob(self, client):
        """log_probs -> log_prob job."""
        client.log_probs(["hi"], completions=["there"], top_k=3)
        req = _last(client)
        assert req.op == "log-probs"
        assert req.job_id == LOG_PROB
        assert req.body == {"prompts": ["hi"], "completions": ["there"], "top_k": 3}

    def test_reset_prefix_cache_targets_sampling(self, client):
        """reset_prefix_cache -> sampling job."""
        client.reset_prefix_cache(drain=False, timeout_s=5.0, retry_interval_s=0.2)
        req = _last(client)
        assert req.op == "reset-prefix-cache"
        assert req.job_id == SAMPLING
        assert req.body == {"drain": False, "timeout_s": 5.0, "retry_interval_s": 0.2}

    def test_sync_weights_targets_training_with_sub_job_ids(self, client):
        """sync_weights -> training job; Cortex-shaped source/target sub-job ids in body."""
        client.sync_weights()
        req = _last(client)
        assert req.op == "weight-sync"
        assert req.job_id == TRAINING
        assert req.body == {"source_sub_job_id": TRAINING, "target_sub_job_ids": [SAMPLING]}


class TestLifecycle:
    def test_reconnect_config_copies_job_ids(self, client):
        """reconnect_config bakes the live job ids into a fresh config."""
        cfg = client.reconnect_config()
        assert cfg.training_job_id == TRAINING
        assert cfg.sampling_job_id == SAMPLING
        assert cfg.log_prob_job_id == LOG_PROB

    def test_require_raises_for_uninitialized_job(self):
        """JobHandles.require guards against calling an op before its job exists."""
        with pytest.raises(ValueError, match="No training job"):
            JobHandles().require("training")


class TestOpRegistry:
    """Contract: the client's op vocabulary and OPS stay in lockstep, and a
    transport's op coverage is checkable without a live backend."""

    def test_client_emits_exactly_the_registered_ops(self, client):
        """Driving every client op must produce exactly the canonical OPS set."""
        client.fwd_bwd({"input_ids": [1]})
        client.fwd_no_grad({"input_ids": [1]})
        client.step()
        client.save_checkpoint()
        client.generate(["hi"])
        client.log_probs(["hi"])
        client.sync_weights()
        client.reset_prefix_cache()
        assert {req.op for req in client.transport.calls} == OPS

    def test_unresolved_ops_flags_a_missing_method(self):
        """A target missing one op's method is reported (renamed/dropped op -> caught early)."""

        class PartialServer:
            pass

        server = PartialServer()
        for op in OPS - {"step"}:
            setattr(server, method_name(op), lambda *a, **k: None)
        assert unresolved_ops(server) == ["step"]

    def test_unresolved_ops_empty_when_fully_covered(self):
        """A target exposing every op resolves cleanly."""

        class FullServer:
            pass

        server = FullServer()
        for op in OPS:
            setattr(server, method_name(op), lambda *a, **k: None)
        assert unresolved_ops(server) == []


class TestTransportSelection:
    def test_make_transport_selects_ray(self, monkeypatch):
        """onprem + ray routes to RayTransport (constructed lazily, so patch it)."""
        import arctic_platform.client.transports.onprem_ray as ray_mod

        class DummyRay:
            def __init__(self, config):
                self.config = config

        monkeypatch.setattr(ray_mod, "RayTransport", DummyRay)
        cfg = ArcticRLClientConfig(model_name="m", comm_protocol="ray", training_gpus=1)
        assert isinstance(client_module.make_transport(cfg), DummyRay)

    def test_make_transport_selects_http_for_onprem(self):
        """onprem + http (the default) routes to HttpTransport."""
        from arctic_platform.client.transports.onprem_http import HttpTransport

        cfg = ArcticRLClientConfig(model_name="m", comm_protocol="http", training_gpus=1)
        assert isinstance(client_module.make_transport(cfg), HttpTransport)
