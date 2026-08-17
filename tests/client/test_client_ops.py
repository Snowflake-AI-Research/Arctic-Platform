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
"""The op surface of ArcticRLClient / SyncArcticRLClient must lower to one
canonical Request each.

A FakeTransport records the Request every op produces so we can assert the
mapping (target job, body, binary flag) with no live backend or GPUs.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from arctic_platform.client import OPS
from arctic_platform.client import ArcticRLClient
from arctic_platform.client import ArcticRLClientConfig
from arctic_platform.client import JobHandles
from arctic_platform.client import OnPremConfig
from arctic_platform.client import Request
from arctic_platform.client import SyncArcticRLClient
from arctic_platform.client import Transport
from arctic_platform.client import client as client_module
from arctic_platform.client import create_arctic_rl_client
from arctic_platform.client import unresolved_ops
from arctic_platform.client.transport import method_name

TRAINING, SAMPLING, LOG_PROB = 1, 2, 3


class FakeTransport(Transport):
    def __init__(self, config: ArcticRLClientConfig, server_state=None) -> None:
        self.config = config
        self.server_state = server_state
        self.jobs = JobHandles()
        self.calls: list[Request] = []

    def initialize(self) -> JobHandles:
        self.jobs = JobHandles(training=TRAINING, sampling=SAMPLING, log_prob=LOG_PROB)
        return self.jobs

    def call(self, request: Request) -> dict:
        self.calls.append(request)
        return {"results": ["ok"], "loss": 0.0}

    async def acall(self, request: Request) -> dict:
        return self.call(request)

    def shutdown(self) -> None:
        pass


def _call(client, method: str, *args, **kwargs):
    """Drive a sync or async client method from a sync test."""
    result = getattr(client, method)(*args, **kwargs)
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result


@pytest.fixture(params=[ArcticRLClient, SyncArcticRLClient], ids=["async", "sync"])
def client(monkeypatch, request) -> ArcticRLClient | SyncArcticRLClient:
    monkeypatch.setattr(client_module, "make_transport", FakeTransport)
    cfg = ArcticRLClientConfig(model_name="m", training_gpus=1, sampling_gpus=1, log_prob_gpus=1)
    return request.param(cfg)


def _last(client: ArcticRLClient | SyncArcticRLClient) -> Request:
    return client.transport.calls[-1]


class TestOpMapping:
    def test_fwd_bwd_targets_training_binary(self, client):
        """fwd_bwd -> training job, binary payload, body carries the batch."""
        _call(client, "fwd_bwd", {"input_ids": [1, 2]})
        req = _last(client)
        assert req.op == "forward-backward"
        assert req.job_id == TRAINING
        assert req.binary is True
        assert req.body["input_ids"] == [1, 2]

    def test_fwd_bwd_folds_processing_and_router_replay(self, client):
        """Non-None processing/router_replay are folded into the body."""
        _call(client, "fwd_bwd", {"input_ids": [1]}, processing={"p": 1}, router_replay={"r": 2})
        body = _last(client).body
        assert body["processing"] == {"p": 1}
        assert body["router_replay"] == {"r": 2}

    def test_fwd_no_grad_targets_training_binary(self, client):
        """fwd_no_grad -> training job, binary payload."""
        _call(client, "fwd_no_grad", {"input_ids": [1]})
        req = _last(client)
        assert req.op == "forward"
        assert req.job_id == TRAINING
        assert req.binary is True

    def test_fwd_no_grad_reference_targets_log_prob(self, client):
        """fwd_no_grad(reference_model=True) -> log_prob job (reference log-probs)."""
        _call(client, "fwd_no_grad", {"input_ids": [1]}, reference_model=True)
        req = _last(client)
        assert req.op == "forward"
        assert req.job_id == LOG_PROB
        assert req.binary is True

    def test_step_targets_training(self, client):
        """step -> training job, learning_rate in body."""
        _call(client, "step", learning_rate=0.5)
        req = _last(client)
        assert req.op == "step"
        assert req.job_id == TRAINING
        assert req.binary is False
        assert req.body == {"learning_rate": 0.5}

    def test_save_checkpoint_targets_training(self, client):
        """save_checkpoint -> training job (save op, Cortex + on-prem SFT body)."""
        _call(client, "save_checkpoint", checkpoint_id="cp1", checkpoint_type="weights-only")
        req = _last(client)
        assert req.op == "save"
        assert req.job_id == TRAINING
        assert req.body["checkpoint_id"] == "cp1"
        assert req.body["checkpoint_type"] == "weights-only"

    def test_load_checkpoint_targets_training(self, client):
        _call(client, "load_checkpoint", path="/tmp/x", step=2)
        req = _last(client)
        assert req.op == "load-checkpoint"
        assert req.job_id == TRAINING
        assert req.body == {"path": "/tmp/x", "step": 2}

    def test_generate_targets_sampling_and_unwraps_results(self, client):
        """generate -> sampling job; returns the 'results' list."""
        out = _call(client, "generate", ["hi"], sampling_params={"n": 1}, routing_key="k", strict=True)
        req = _last(client)
        assert req.op == "generate"
        assert req.job_id == SAMPLING
        assert req.body == {"prompts": ["hi"], "sampling_params": {"n": 1}, "routing_key": "k", "strict": True}
        assert out == ["ok"]

    def test_log_probs_targets_log_prob(self, client):
        """log_probs -> log_prob job."""
        _call(client, "log_probs", ["hi"], completions=["there"], top_k=3)
        req = _last(client)
        assert req.op == "log-probs"
        assert req.job_id == LOG_PROB
        assert req.body == {"prompts": ["hi"], "completions": ["there"], "top_k": 3}

    def test_reset_prefix_cache_targets_sampling(self, client):
        """reset_prefix_cache -> sampling job via the operation envelope."""
        _call(client, "reset_prefix_cache", drain=False, timeout_s=5.0, retry_interval_s=0.2)
        req = _last(client)
        assert req.op == "operation"
        assert req.job_id == SAMPLING
        assert req.body == {
            "operation_type": "reset-prefix-cache",
            "sub_job_type": "sampling",
            "payload": {"drain": False, "timeout_s": 5.0, "retry_interval_s": 0.2},
        }

    def test_sync_weights_staged_wake_and_operation(self, client):
        """sync_weights: wake → weight-sync operation → wake → reset-prefix-cache.

        Explicit cuda_ipc / low_memory are per-call overrides included in the
        payload; colocate is never sent (the server owns it via launch state).
        """
        _call(client, "sync_weights", cuda_ipc=True, low_memory=False)
        ops = [r.op for r in client.transport.calls[-4:]]
        assert ops == ["wake-inference", "operation", "wake-inference", "operation"]
        sync_req = client.transport.calls[-3]
        assert sync_req.job_id == TRAINING
        assert sync_req.body["operation_type"] == "weight-sync"
        assert sync_req.body["payload"] == {
            "source_sub_job_id": TRAINING,
            "target_sub_job_ids": [SAMPLING],
            "cuda_ipc": True,
            "low_memory": False,
        }

    def test_sync_weights_body_omits_strategy_by_default(self, client):
        """With no override, the payload carries only the job ids: the server uses
        the strategy baked onto the training job at init (no colocate on the wire)."""
        _call(client, "sync_weights")
        sync_req = client.transport.calls[-3]
        assert sync_req.body["payload"] == {
            "source_sub_job_id": TRAINING,
            "target_sub_job_ids": [SAMPLING],
        }

    def test_sleep_wake_training_and_inference(self, client):
        _call(client, "sleep_inference", level=2)
        assert _last(client).op == "sleep-inference"
        _call(client, "wake_inference", tags=["weights"])
        assert _last(client).op == "wake-inference"
        _call(client, "sleep_training", mode="non_lp")
        assert _last(client).op == "sleep-training"
        _call(client, "wake_training")
        assert _last(client).op == "wake-training"


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


class TestServerState:
    """Reconnect handoff: a client can expose (and reattach via) the transport's
    server-state handle, but only transports that actually own one."""

    def test_get_server_state_raises_without_transport_support(self, client):
        """FakeTransport has no server state -> get_server_state() raises."""
        with pytest.raises(NotImplementedError, match="no server state"):
            client.get_server_state()

    def test_get_server_state_returns_transport_state(self, monkeypatch):
        """When the transport exposes get_server_state, the client forwards it."""
        sentinel = object()

        class StatefulTransport(FakeTransport):
            def get_server_state(self):
                return sentinel

        monkeypatch.setattr(client_module, "make_transport", StatefulTransport)
        cfg = ArcticRLClientConfig(model_name="m", training_gpus=1, sampling_gpus=1, log_prob_gpus=1)
        assert ArcticRLClient(cfg).get_server_state() is sentinel

    def test_create_client_forwards_server_state_for_reconnect(self, monkeypatch):
        """create_arctic_rl_client(cfg, server_state=...) reattaches via the Ray transport."""
        import arctic_platform.client.transports.onprem_ray as ray_mod

        sentinel = object()

        class DummyRay:
            def __init__(self, config, server_state=None):
                self.config = config
                self.server_state = server_state
                self.jobs = JobHandles(training=TRAINING, sampling=SAMPLING, log_prob=LOG_PROB)

            def initialize(self):
                return self.jobs

            def get_server_state(self):
                return self.server_state

        monkeypatch.setattr(ray_mod, "RayTransport", DummyRay)
        cfg = ArcticRLClientConfig(
            model_name="m",
            backend_config=OnPremConfig(comm_protocol="ray"),
            training_gpus=1,
            sampling_gpus=1,
            log_prob_gpus=1,
            training_job_id=TRAINING,
            sampling_job_id=SAMPLING,
            log_prob_job_id=LOG_PROB,
        )
        client = create_arctic_rl_client(cfg, server_state=sentinel)
        assert client.transport.server_state is sentinel
        assert client.get_server_state() is sentinel


class TestOpRegistry:
    """Contract: the client's op vocabulary and OPS stay in lockstep, and a
    transport's op coverage is checkable without a live backend."""

    def test_client_emits_exactly_the_registered_ops(self, client):
        """Driving every client op must cover the canonical OPS set."""
        _call(client, "fwd_bwd", {"input_ids": [1]})
        _call(client, "fwd_no_grad", {"input_ids": [1]})
        _call(client, "step")
        _call(client, "save_checkpoint")
        _call(client, "load_checkpoint")
        _call(client, "generate", ["hi"])
        _call(client, "log_probs", ["hi"])
        # sync_weights expands to wake + operation + wake + reset(operation)
        n_before = len(client.transport.calls)
        _call(client, "sync_weights")
        assert {"wake-inference", "operation"} <= {r.op for r in client.transport.calls[n_before:]}
        _call(client, "reset_prefix_cache")
        _call(client, "sleep_inference")
        _call(client, "wake_inference")
        _call(client, "sleep_training")
        _call(client, "wake_training")
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
            def __init__(self, config, server_state=None):
                self.config = config
                self.server_state = server_state

        monkeypatch.setattr(ray_mod, "RayTransport", DummyRay)
        cfg = ArcticRLClientConfig(model_name="m", backend_config=OnPremConfig(comm_protocol="ray"), training_gpus=1)
        assert isinstance(client_module.make_transport(cfg), DummyRay)

    def test_make_transport_selects_http_for_onprem(self):
        """onprem + http (the default) routes to HttpTransport."""
        from arctic_platform.client.transports.onprem_http import HttpTransport

        cfg = ArcticRLClientConfig(model_name="m", backend_config=OnPremConfig(comm_protocol="http"), training_gpus=1)
        assert isinstance(client_module.make_transport(cfg), HttpTransport)

    def test_make_transport_forwards_server_state_to_ray(self, monkeypatch):
        """make_transport threads server_state into the Ray transport (reconnect path)."""
        import arctic_platform.client.transports.onprem_ray as ray_mod

        class DummyRay:
            def __init__(self, config, server_state=None):
                self.config = config
                self.server_state = server_state

        monkeypatch.setattr(ray_mod, "RayTransport", DummyRay)
        sentinel = object()
        cfg = ArcticRLClientConfig(model_name="m", backend_config=OnPremConfig(comm_protocol="ray"), training_gpus=1)
        transport = client_module.make_transport(cfg, server_state=sentinel)
        assert transport.server_state is sentinel

    def test_make_transport_rejects_server_state_for_http(self):
        """server_state reconnect is Ray-only; HTTP transport must reject it."""
        cfg = ArcticRLClientConfig(model_name="m", backend_config=OnPremConfig(comm_protocol="http"), training_gpus=1)
        with pytest.raises(ValueError, match="server_state reconnect"):
            client_module.make_transport(cfg, server_state=object())


class TestWeightSyncStrategyInit:
    """The static weight-sync strategy (cuda_ipc / low_memory) rides the training /initialize payload, so /weight-sync
    need not resend it."""

    def test_training_init_payload_carries_strategy(self):
        from arctic_platform.client import TrainingConfig

        cfg = ArcticRLClientConfig(
            model_name="m",
            training_gpus=1,
            training=TrainingConfig(checkpoint_path="/tmp/c", cuda_ipc=True, low_memory=True),
        )
        payload = cfg.to_onprem("training")
        assert payload["cuda_ipc"] is True
        assert payload["low_memory"] is True

    def test_non_training_init_payload_omits_strategy(self):
        cfg = ArcticRLClientConfig(model_name="m", sampling_gpus=1)
        payload = cfg.to_onprem("sampling")
        assert "cuda_ipc" not in payload
        assert "low_memory" not in payload
