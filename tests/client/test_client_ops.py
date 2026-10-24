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
"""The op surface of the unified client must lower to one canonical Request each.

A FakeTransport records the Request every op produces so we can assert the
mapping (target job, body, binary flag) with no live backend or GPUs. The shared
ops are driven across all three frontends (async RL, sync RL, SFT) so a subclass
cannot silently diverge from the base; op-registry coverage and log_probs stay on
the RL clients, and SFT's loss-fn defaulting is asserted in tests/sft.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Literal
from typing import Optional

import pytest
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PositiveInt
from pydantic import ValidationError

from arctic_platform.client import OPS
from arctic_platform.client import ArcticClient
from arctic_platform.client import ArcticClientConfig
from arctic_platform.client import ArcticRLClient
from arctic_platform.client import ArcticSFTClient
from arctic_platform.client import ArcticSFTClientConfig
from arctic_platform.client import AsyncArcticClient
from arctic_platform.client import AsyncArcticRLClient
from arctic_platform.client import JobHandles
from arctic_platform.client import OnPremConfig
from arctic_platform.client import Request
from arctic_platform.client import TrainingConfig
from arctic_platform.client import Transport
from arctic_platform.client import base as base_module
from arctic_platform.client import requests as cluster_requests
from arctic_platform.client import unresolved_ops
from arctic_platform.client.transport import CLUSTER_OPS
from arctic_platform.client.transport import method_name

TRAINING, SAMPLING, LOG_PROB = 1, 2, 3


class FakeTransport(Transport):
    def __init__(self, config: ArcticClientConfig, server_state=None) -> None:
        self.config = config
        self.server_state = server_state
        self.jobs = JobHandles()
        self.calls: list[Request] = []

    def initialize(self, client_config: ArcticClientConfig) -> JobHandles:
        self.jobs = JobHandles(training=TRAINING, sampling=SAMPLING, log_prob=LOG_PROB)
        return self.jobs

    def call(self, request: Request) -> dict:
        self.calls.append(request)
        return {"results": ["ok"], "loss": 0.0, "request_id": "req", "state": "running", "events": [{}]}

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


def _stop_stream_after_one_event(stream) -> None:
    """Read one event from a sync or async stream, then close it early."""
    if not inspect.isasyncgen(stream):
        next(stream)
        stream.close()
        return

    async def drive():
        await anext(stream)
        await stream.aclose()

    asyncio.run(drive())


def _build(monkeypatch, cls):
    monkeypatch.setattr(base_module, "make_transport", FakeTransport)
    return cls(ArcticClientConfig(model_name="m", training_gpus=1, sampling_gpus=1, log_prob_gpus=1))


@pytest.fixture(params=[AsyncArcticRLClient, ArcticRLClient, ArcticSFTClient], ids=["async", "sync", "sft"])
def client(monkeypatch, request):
    """Every frontend, for the ops they all share."""
    return _build(monkeypatch, request.param)


@pytest.fixture(params=[AsyncArcticRLClient, ArcticRLClient], ids=["async", "sync"])
def rl_client(monkeypatch, request) -> AsyncArcticRLClient | ArcticRLClient:
    """RL-only surface (log_probs) and full op-registry coverage."""
    return _build(monkeypatch, request.param)


def _last(client) -> Request:
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

    def test_fwd_no_grad_reference_targets_log_prob(self, rl_client):
        """fwd_no_grad(reference_model=True) -> log_prob job (reference log-probs)."""
        _call(rl_client, "fwd_no_grad", {"input_ids": [1]}, reference_model=True)
        req = _last(rl_client)
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
        assert req.body == {"path": "/tmp/x", "step": 2, "checkpoint_id": None, "source_job_id": None}

    def test_generate_targets_sampling_and_unwraps_results(self, client):
        """generate -> sampling job; returns the 'results' list."""
        out = _call(client, "generate", ["hi"], sampling_params={"n": 1}, routing_key="k", strict=True)
        req = _last(client)
        assert req.op == "generate"
        assert req.job_id == SAMPLING
        assert req.body == {"prompts": ["hi"], "sampling_params": {"n": 1}, "routing_key": "k", "strict": True}
        assert out == ["ok"]

    def test_log_probs_targets_log_prob(self, rl_client):
        """log_probs -> log_prob job. RL-only: SFT never allocates a log-prob engine."""
        _call(rl_client, "log_probs", ["hi"], completions=["there"], top_k=3)
        req = _last(rl_client)
        assert req.op == "log-probs"
        assert req.job_id == LOG_PROB
        assert req.body == {"prompts": ["hi"], "completions": ["there"], "top_k": 3}

    def test_sft_client_has_no_log_probs(self, monkeypatch):
        """log_probs stays on the RL subclass, not the shared base."""
        assert not hasattr(_build(monkeypatch, ArcticSFTClient), "log_probs")

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

        Explicit cuda_ipc / low_memory are per-call overrides included in the payload; colocate is never sent (the
        server owns it via launch state).
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
        """With no override, the payload carries only the job ids: the server uses the strategy baked onto the training
        job at init (no colocate on the wire)."""
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

    def test_require_any_returns_the_first_live_job(self):
        """require_any picks any initialized job and fails when none exists."""
        assert JobHandles(sampling=SAMPLING, log_prob=LOG_PROB).require_any() == SAMPLING
        with pytest.raises(ValueError, match="No job initialized"):
            JobHandles().require_any()


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

        monkeypatch.setattr(base_module, "make_transport", StatefulTransport)
        cfg = ArcticClientConfig(model_name="m", training_gpus=1, sampling_gpus=1, log_prob_gpus=1)
        assert AsyncArcticRLClient(cfg).get_server_state() is sentinel

    def test_create_client_forwards_server_state_for_reconnect(self, monkeypatch):
        """AsyncArcticRLClient(cfg, server_state=...) reattaches via the on-prem transport."""
        import arctic_platform.client.transports.onprem as onprem_mod

        sentinel = object()

        class DummyRay:
            def __init__(self, config, server_state=None):
                self.config = config
                self.server_state = server_state
                self.jobs = JobHandles(training=TRAINING, sampling=SAMPLING, log_prob=LOG_PROB)

            def initialize(self, client_config):
                return self.jobs

            def get_server_state(self):
                return self.server_state

        monkeypatch.setattr(onprem_mod, "OnPremTransport", DummyRay)
        cfg = ArcticClientConfig(
            model_name="m",
            training_gpus=1,
            sampling_gpus=1,
            log_prob_gpus=1,
            training_job_id=TRAINING,
            sampling_job_id=SAMPLING,
            log_prob_job_id=LOG_PROB,
        )
        client = AsyncArcticRLClient(cfg, server_state=sentinel)
        assert client.transport.server_state is sentinel
        assert client.get_server_state() is sentinel


class TestOpRegistry:
    """Contract: the client's op vocabulary and OPS stay in lockstep, and a
    transport's op coverage is checkable without a live backend."""

    def test_client_emits_exactly_the_registered_ops(self, rl_client):
        """Driving every client op must cover the canonical OPS set."""
        _call(rl_client, "fwd_bwd", {"input_ids": [1]})
        _call(rl_client, "fwd_no_grad", {"input_ids": [1]})
        _call(rl_client, "step")
        _call(rl_client, "save_checkpoint")
        _call(rl_client, "load_checkpoint")
        _call(rl_client, "generate", ["hi"])
        _stop_stream_after_one_event(rl_client.generate_stream(["hi"]))  # submit, poll, cancel
        _call(rl_client, "log_probs", ["hi"])
        # sync_weights expands to wake + operation + wake + reset(operation)
        n_before = len(rl_client.transport.calls)
        _call(rl_client, "sync_weights")
        assert {"wake-inference", "operation"} <= {r.op for r in rl_client.transport.calls[n_before:]}
        _call(rl_client, "reset_prefix_cache")
        _call(rl_client, "sleep_inference")
        _call(rl_client, "wake_inference")
        _call(rl_client, "sleep_training")
        _call(rl_client, "wake_training")
        assert {req.op for req in rl_client.transport.calls} == OPS

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


class TestClusterRequestBuilders:
    @pytest.mark.parametrize(
        ("actual", "expected"),
        [
            (cluster_requests.list_jobs_request("running"), Request("list-jobs", body={"status": "running"})),
            (cluster_requests.get_job_request("job-1"), Request("get-job", "job-1")),
            (
                cluster_requests.get_experiment_run_request("job-1"),
                Request("get-experiment-run", "job-1"),
            ),
            (cluster_requests.cancel_job_request("job-1"), Request("cancel-job", "job-1")),
            (cluster_requests.capacity_request("H100"), Request("capacity", body={"hardware": "H100"})),
            (
                cluster_requests.capacity_request("H200", ("job_type", "submitted_by")),
                Request(
                    "capacity",
                    body={"hardware": "H200", "breakdown": ("job_type", "submitted_by")},
                ),
            ),
            (cluster_requests.list_checkpoints_request("job-1"), Request("list-checkpoints", "job-1")),
            (
                cluster_requests.export_checkpoint_request("job-1", "step-1"),
                Request("export-checkpoint", "job-1", {"checkpoint_id": "step-1"}),
            ),
            (
                cluster_requests.delete_checkpoint_request("job-1", "step-1"),
                Request("delete-checkpoint", "job-1", {"checkpoint_id": "step-1"}),
            ),
            (
                cluster_requests.download_artifacts_request("job-1", "stdout", "/tmp/logs"),
                Request("download-artifacts", "job-1", {"kind": "stdout", "output_dir": "/tmp/logs"}),
            ),
        ],
    )
    def test_context_free_request_shape(self, actual, expected):
        """Cluster builders lower arguments without client or transport state."""
        assert actual == expected

    def test_optional_filters_are_omitted(self):
        """Absent account filters do not become explicit null query values."""
        assert cluster_requests.list_jobs_request().body == {}
        assert cluster_requests.capacity_request().body == {}

    def test_cluster_vocabulary_is_separate_from_live_ops(self):
        """Cluster builders have their own registry and do not expand on-prem coverage."""
        requests = (
            cluster_requests.list_jobs_request(),
            cluster_requests.get_job_request("job"),
            cluster_requests.get_experiment_run_request("job"),
            cluster_requests.cancel_job_request("job"),
            cluster_requests.capacity_request(),
            cluster_requests.list_checkpoints_request("job"),
            cluster_requests.export_checkpoint_request("job", "checkpoint"),
            cluster_requests.delete_checkpoint_request("job", "checkpoint"),
            cluster_requests.download_artifacts_request("job", "stdout", "/tmp/logs"),
        )
        assert {request.op for request in requests} == CLUSTER_OPS
        assert CLUSTER_OPS.isdisjoint(OPS)


class TestClusterClientMethods:
    def test_account_ops_use_the_given_arguments(self, client):
        """Account-level ops need no stored handle."""
        _call(client, "list_jobs", "running")
        assert _last(client) == cluster_requests.list_jobs_request("running")
        _call(client, "capacity", "H200", breakdown=("job_type",))
        assert _last(client) == cluster_requests.capacity_request("H200", ("job_type",))
        _call(client, "cancel_job", "job-9")
        assert _last(client) == cluster_requests.cancel_job_request("job-9")

    def test_get_job_targets_a_stored_handle(self, client):
        _call(client, "get_job")
        assert _last(client) == cluster_requests.get_job_request(TRAINING)

    def test_checkpoint_ops_target_the_training_job(self, client):
        _call(client, "list_checkpoints")
        assert _last(client) == cluster_requests.list_checkpoints_request(TRAINING)
        _call(client, "export_checkpoint", "cp-1")
        assert _last(client) == cluster_requests.export_checkpoint_request(TRAINING, "cp-1")
        _call(client, "delete_checkpoint", "cp-1")
        assert _last(client) == cluster_requests.delete_checkpoint_request(TRAINING, "cp-1")

    def test_onprem_rejects_cluster_ops(self):
        """On-prem has no job persistence, so cluster ops fail instead of reaching the Ray server."""
        from arctic_platform.client.transports.onprem import OnPremTransport

        transport = OnPremTransport(OnPremConfig())
        with pytest.raises(NotImplementedError, match="list-jobs"):
            transport.call(cluster_requests.list_jobs_request())


class TestTransportSelection:
    def test_make_transport_passes_cortex_config(self, monkeypatch):
        """Stateful clients pass the complete Cortex transport config unchanged."""
        pytest.importorskip("tenacity")
        import arctic_platform.client.transports.cortex as cortex_mod

        class DummyCortex:
            def __init__(self, config):
                self.config = config

        monkeypatch.setattr(cortex_mod, "CortexTransport", DummyCortex)
        from arctic_platform.client import CortexConfig

        backend = CortexConfig(base_url="http://mock")
        cfg = ArcticClientConfig(
            model_name="m",
            backend=backend,
        )
        transport = base_module.make_transport(cfg)
        assert isinstance(transport, DummyCortex)
        assert transport.config is backend

    def test_make_transport_selects_onprem(self, monkeypatch):
        """onprem routes to OnPremTransport (constructed lazily, so patch it)."""
        import arctic_platform.client.transports.onprem as onprem_mod

        class DummyRay:
            def __init__(self, config, server_state=None):
                self.config = config
                self.server_state = server_state

        monkeypatch.setattr(onprem_mod, "OnPremTransport", DummyRay)
        cfg = ArcticClientConfig(model_name="m", backend=OnPremConfig(), training_gpus=1)
        transport = base_module.make_transport(cfg)
        assert isinstance(transport, DummyRay)
        assert transport.config is cfg.backend

    def test_make_transport_forwards_server_state_to_onprem(self, monkeypatch):
        """make_transport threads server_state into the on-prem transport (reconnect path)."""
        import arctic_platform.client.transports.onprem as onprem_mod

        class DummyRay:
            def __init__(self, config, server_state=None):
                self.config = config
                self.server_state = server_state

        monkeypatch.setattr(onprem_mod, "OnPremTransport", DummyRay)
        sentinel = object()
        cfg = ArcticClientConfig(model_name="m", backend=OnPremConfig(), training_gpus=1)
        transport = base_module.make_transport(cfg, server_state=sentinel)
        assert transport.server_state is sentinel


class TestWeightSyncStrategyInit:
    """The static weight-sync strategy (cuda_ipc / low_memory) rides the training /initialize payload, so /weight-sync
    need not resend it."""

    def test_training_init_payload_carries_strategy(self):
        from arctic_platform.client import TrainingConfig

        cfg = ArcticClientConfig(
            model_name="m",
            training_gpus=1,
            training=TrainingConfig(checkpoint_path="/tmp/c", cuda_ipc=True, low_memory=True),
        )
        payload = cfg.to_onprem("training")
        assert payload["cuda_ipc"] is True
        assert payload["low_memory"] is True

    def test_non_training_init_payload_omits_strategy(self):
        cfg = ArcticClientConfig(model_name="m", sampling_gpus=1)
        payload = cfg.to_onprem("sampling")
        assert "cuda_ipc" not in payload
        assert "low_memory" not in payload


@pytest.fixture(autouse=True)
def _isolate_arctic_env(monkeypatch):
    """Clear ``ARCTIC_CORTEX_*`` before each test.

    ``CortexConfig`` is a ``BaseSettings``, so it reads the ambient environment
    at construction; a var leaked in from the developer's shell would otherwise
    decide the outcome of these tests.
    """
    import os

    for k in list(os.environ):
        if k.startswith("ARCTIC_CORTEX_"):
            monkeypatch.delenv(k, raising=False)


class TestCortexConfigReadsEnv:
    """``CortexConfig`` hydrates from ``ARCTIC_CORTEX_*`` via pydantic-settings,
    and explicit values still win over the environment."""

    def test_base_url_only_bypasses_pat(self, monkeypatch):
        monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://mock")
        from arctic_platform.client import CortexConfig

        cfg = CortexConfig()
        assert cfg.base_url == "http://mock"
        assert cfg.host is None

    def test_host_path_requires_db_schema_pat(self, monkeypatch):
        monkeypatch.setenv("ARCTIC_CORTEX_HOST", "acct.snowflakecomputing.com")
        monkeypatch.setenv("ARCTIC_CORTEX_DATABASE", "db")
        monkeypatch.setenv("ARCTIC_CORTEX_SCHEMA", "sch")
        monkeypatch.setenv("ARCTIC_CORTEX_PAT", "pat-value")
        from arctic_platform.client import CortexConfig

        cfg = CortexConfig()
        assert cfg.host == "acct.snowflakecomputing.com"
        assert cfg.database == "db"
        assert cfg.schema_ == "sch"
        assert cfg.pat.get_secret_value() == "pat-value"

    def test_missing_pat_on_host_auth_is_refused(self, monkeypatch):
        monkeypatch.setenv("ARCTIC_CORTEX_HOST", "acct.snowflakecomputing.com")
        monkeypatch.setenv("ARCTIC_CORTEX_DATABASE", "db")
        monkeypatch.setenv("ARCTIC_CORTEX_SCHEMA", "sch")
        from arctic_platform.client import CortexConfig

        with pytest.raises(ValidationError, match="ARCTIC_CORTEX_PAT"):
            CortexConfig()

    def test_explicit_override_wins(self, monkeypatch):
        monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://env")
        from arctic_platform.client import CortexConfig

        cfg = CortexConfig(base_url="http://explicit")
        assert cfg.base_url == "http://explicit"

    def test_pat_is_not_rendered_anywhere(self, monkeypatch):
        """The PAT must not ride along into a log line or a serialized config.

        Configs get printed, and the whole point of hydrating `pat` from the
        environment is that the token is now a materialized field rather than
        something fetched on demand — so it would otherwise land in every repr.
        """
        monkeypatch.setenv("ARCTIC_CORTEX_HOST", "acct.snowflakecomputing.com")
        monkeypatch.setenv("ARCTIC_CORTEX_DATABASE", "db")
        monkeypatch.setenv("ARCTIC_CORTEX_SCHEMA", "sch")
        monkeypatch.setenv("ARCTIC_CORTEX_PAT", "SUPERSECRET123")
        from arctic_platform.client import CortexConfig

        cfg = CortexConfig()
        assert "SUPERSECRET123" not in repr(cfg)
        assert "SUPERSECRET123" not in str(cfg.model_dump())
        assert "SUPERSECRET123" not in cfg.model_dump_json()
        # ...but it is still readable where it is actually needed.
        assert cfg.pat.get_secret_value() == "SUPERSECRET123"

    def test_exported_but_empty_reads_as_unset(self, monkeypatch):
        """A shell that exports ARCTIC_CORTEX_ENDPOINT= means "unset", not "".

        Without ``env_ignore_empty`` the empty string would beat the default and
        the endpoint would silently become invalid.
        """
        monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://mock")
        monkeypatch.setenv("ARCTIC_CORTEX_ENDPOINT", "")
        from arctic_platform.client import CortexConfig

        assert CortexConfig().endpoint == "cortex-training"


class TestUnifiedConfigDoesNotReadEnv:
    """Only ``CortexConfig`` reads the environment, and only for its own fields.

    Which backend you get is decided by the caller — for SkyRL, by which
    entrypoint the recipe names — never by an ambient variable. A stray
    ``ARCTIC_CORTEX_HOST`` must not turn an on-prem run into a Cortex one.
    """

    def test_cortex_env_does_not_promote_an_onprem_backend(self, monkeypatch):
        monkeypatch.setenv("ARCTIC_CORTEX_HOST", "acct.snowflakecomputing.com")
        cfg = ArcticClientConfig(model_name="m", backend=OnPremConfig(), training_gpus=1)
        assert cfg.backend.type == "onprem"


class TestCortexNoopOffload:
    """``to_cortex`` drops ``offload_*: {device: none}`` from the forwarded
    ds_config. Cortex builds the optimizer from the lifted typed ``optimizer``
    and lands on DeepSpeedCPUAdam, so an explicit no-op offload block moves only
    the parameters to GPU and ``step()`` asserts "CPUAdam param is on cuda:0"."""

    @staticmethod
    def _zero_block(ds_config: dict) -> dict:
        from arctic_platform.client import CortexConfig

        cfg = ArcticClientConfig(
            model_name="m",
            training_gpus=1,
            backend=CortexConfig(base_url="https://x", pat="y"),
            training={"ds_config": ds_config},
        )
        return cfg._cortex_training_sub_job()["training_config"]["ds_config"]["zero_optimization"]

    def test_drops_noop_offload(self):
        # verbatim from the ds_config SkyRL sent on the job that failed to provision
        block = self._zero_block(
            {
                "zero_optimization": {
                    "offload_optimizer": {"device": "none"},
                    "offload_param": {"device": "none"},
                    "stage": 0,
                },
                "optimizer": {"type": "AdamW", "params": {"lr": 1e-6}},
            }
        )
        assert block == {"stage": 0}

    def test_keeps_a_real_cpu_offload_request(self):
        block = self._zero_block({"zero_optimization": {"offload_optimizer": {"device": "cpu"}, "stage": 2}})
        assert block == {"offload_optimizer": {"device": "cpu"}, "stage": 2}

    def test_leaves_untouched_when_nothing_to_drop(self):
        assert self._zero_block({"zero_optimization": {"stage": 2}}) == {"stage": 2}


class TestCortexBackendOnlyConstruction:
    def test_backend_config_does_not_initialize_jobs(self):
        """A bare CortexConfig builds transport plumbing but cannot create a job."""
        pytest.importorskip("tenacity")
        from arctic_platform.client import CortexConfig
        from arctic_platform.client.transports.cortex import CortexTransport

        backend = CortexConfig(
            base_url="http://mock",
            database="db",
            schema="schema",
        )
        transport = CortexTransport(backend)
        assert transport.config is backend
        assert transport._prefix == "http://mock/api/v2/databases/db/schemas/schema/cortex-training"
        assert transport.jobs == JobHandles()
        assert transport.job_id is None

    def test_parent_id_normalizes_sub_job_tokens(self):
        """Cluster routes can address a parent from either parent or sub-job IDs."""
        pytest.importorskip("tenacity")
        from arctic_platform.client.transports.cortex import _parent_id

        assert _parent_id("job-1") == "job-1"
        assert _parent_id("job-1:training:0") == "job-1"
        assert _parent_id(42) == "42"


# Mirror of Neutrino's ``ZorroTrainConfig`` in dss/ray_dss/models.py on the DSS
# branch ``truwase/zorro-train`` (DSS #573, d7ef9b1b). The server model is
# ``extra="forbid"``: a field added or renamed there must be added here and in
# ``arctic_platform.client.config._cortex_zorro_train``.
class _ServerZorroTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable: bool = False
    response_len: Optional[PositiveInt] = None
    temperature: Optional[float] = Field(default=None, gt=0)
    use_unpad: bool = True
    logits_optimization: Literal["none", "memory", "compute"] = "none"
    logits_optimization_peak_mem_size_in_gib: int = Field(default=4, gt=0)
    logits_compute_from_fp32_inputs: bool = False
    logits_compute_in_fp32: bool = False


class TestCortexZorroTrainingConfig:
    """A ZoRRo ds_worker_config must reach Neutrino as ``training_config.pack=false``
    plus a ``zorro_train`` block. Without them the job defaults to packed dispatch,
    never installs the ZoRRo forward, and ``pack_microbatch`` rejects the batch."""

    BASE_WORKER = {
        "use_liger": True,
        "enable_gradient_checkpointing": True,
        "attn_implementation": "flash_attention_2",
    }
    # verbatim key set SkyRL's arctic_rl config emits with use_zorro on
    ZORRO_WORKER = {
        **BASE_WORKER,
        "zorro_train_enable": True,
        "response_len": 1024,
        "max_token_len": 9216,
        "rollout_n": 8,
        "temperature": 1.0,
        "use_unpad": True,
        "logits_optimization": "memory",
        "logits_optimization_peak_mem_size_in_gib": 4.0,
        "logits_compute_from_fp32_inputs": False,
        "logits_compute_in_fp32": True,
    }

    @staticmethod
    def _training_config(worker: dict | None) -> dict:
        from arctic_platform.client import CortexConfig

        cfg = ArcticClientConfig(
            model_name="m",
            training_gpus=1,
            backend=CortexConfig(base_url="https://x", pat="y"),
            training={"ds_worker_config": worker},
        )
        return cfg._cortex_training_sub_job()["training_config"]

    def test_enabled_unpacks_and_forwards_zorro_block(self):
        tc = self._training_config(self.ZORRO_WORKER)
        assert tc["pack"] is False
        assert tc["zorro_train"] == {
            "enable": True,
            "response_len": 1024,
            "temperature": 1.0,
            "use_unpad": True,
            "logits_optimization": "memory",
            "logits_optimization_peak_mem_size_in_gib": 4,
            "logits_compute_from_fp32_inputs": False,
            "logits_compute_in_fp32": True,
        }

    def test_client_only_keys_are_not_forwarded(self):
        tc = self._training_config(self.ZORRO_WORKER)
        for key in ("max_token_len", "rollout_n", "zorro_train_enable"):
            assert key not in tc["zorro_train"]
            assert key not in tc

    def test_whole_float_peak_mem_lands_as_int(self):
        peak = self._training_config(self.ZORRO_WORKER)["zorro_train"]["logits_optimization_peak_mem_size_in_gib"]
        assert peak == 4 and type(peak) is int

    def test_fractional_peak_mem_raises_instead_of_truncating(self):
        worker = {**self.ZORRO_WORKER, "logits_optimization_peak_mem_size_in_gib": 2.5}
        with pytest.raises(ValueError, match="logits_optimization_peak_mem_size_in_gib"):
            self._training_config(worker)

    @pytest.mark.parametrize("key", ["response_len", "temperature"])
    def test_missing_required_field_raises(self, key):
        worker = {k: v for k, v in self.ZORRO_WORKER.items() if k != key}
        with pytest.raises(ValueError, match=key):
            self._training_config(worker)

    @pytest.mark.parametrize(("key", "value"), [("response_len", 0), ("response_len", 1.5), ("temperature", 0.0)])
    def test_invalid_required_field_raises(self, key, value):
        with pytest.raises(ValueError, match=key):
            self._training_config({**self.ZORRO_WORKER, key: value})

    def test_optional_knobs_left_to_server_defaults(self):
        worker = {**self.BASE_WORKER, "zorro_train_enable": True, "response_len": 512, "temperature": 0.7}
        expected = {"enable": True, "response_len": 512, "temperature": 0.7}
        assert self._training_config(worker)["zorro_train"] == expected

    @pytest.mark.parametrize("enable", [None, False])
    def test_disabled_emits_neither_pack_nor_zorro(self, enable):
        worker = {k: v for k, v in self.ZORRO_WORKER.items() if k != "zorro_train_enable"}
        if enable is not None:
            worker["zorro_train_enable"] = enable
        tc = self._training_config(worker)
        assert "pack" not in tc and "zorro_train" not in tc
        # leftover ZoRRo knobs must not change a non-ZoRRo job at all
        assert tc == self._training_config(self.BASE_WORKER)

    def test_emitted_block_validates_against_server_schema(self):
        block = self._training_config(self.ZORRO_WORKER)["zorro_train"]
        parsed = _ServerZorroTrainConfig.model_validate(block)
        assert parsed.model_dump(include=set(block)) == block

    def test_server_schema_mirror_rejects_client_only_keys(self):
        # pins the reason max_token_len / rollout_n must stay off the wire
        block = {**self._training_config(self.ZORRO_WORKER)["zorro_train"], "rollout_n": 8}
        with pytest.raises(ValidationError, match="rollout_n"):
            _ServerZorroTrainConfig.model_validate(block)

    def test_server_schema_mirror_matches_dss(self):
        models = pytest.importorskip("dss.ray_dss.models")
        server = models.ZorroTrainConfig
        assert set(server.model_fields) == set(_ServerZorroTrainConfig.model_fields)
        tc = self._training_config(self.ZORRO_WORKER)
        models.TrainingConfig.model_validate(tc)


class TestCortexTransportNoopOps:
    """``wake_*`` / ``sleep_*`` short-circuit in the Cortex transport so the
    shim doesn't have to wrap each call — including the ones ``sync_weights``
    invokes internally."""

    def test_call_returns_empty_for_noop_op(self, monkeypatch):
        # CortexTransport pulls in tenacity, which lives in the `cortex` extra;
        # CI's unit-tests job installs `.[testing,rl]` only.
        pytest.importorskip("tenacity")
        monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://mock")
        from arctic_platform.client import CortexConfig
        from arctic_platform.client.transport import Request
        from arctic_platform.client.transports.cortex import CortexTransport

        cfg = ArcticClientConfig(model_name="m", backend=CortexConfig(), training_gpus=1, sampling_gpus=1)
        t = CortexTransport(cfg.backend)
        # Would normally raise NotImplementedError on the transport; the noop
        # short-circuit means the shim never has to guard these calls.
        assert t.call(Request("wake-inference", 1, None)) == {}
        assert t.call(Request("sleep-training", 1, {"mode": "all"})) == {}

    def test_wake_inference_tag_list_is_a_noop(self, monkeypatch):
        """sync_weights wakes with a bare tag list as the body; it must not be read as a dict."""
        pytest.importorskip("tenacity")
        from arctic_platform.client import CortexConfig
        from arctic_platform.client.transports.cortex import CortexTransport

        t = CortexTransport(CortexConfig(base_url="http://mock"))
        assert t.call(cluster_requests.wake_inference_request(JobHandles(sampling=SAMPLING), ["weights"])) == {}
        assert asyncio.run(t.acall(Request("wake-inference", SAMPLING, ["kv_cache"]))) == {}


class TestCortexCancelToleratesEmptyBody:
    """``:cancel`` answers 200 with no body. Decoding that as JSON reported a
    release that had in fact succeeded as ``could not release <id>``, which
    points the operator at the GPU cap instead of at their freed GPUs."""

    def test_cancel_request_accepts_empty_response(self, monkeypatch):
        pytest.importorskip("tenacity")
        monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://mock")
        from arctic_platform.client import CortexConfig
        from arctic_platform.client.transports.cortex import CortexTransport

        t = CortexTransport(CortexConfig())

        class _EmptyResp:
            status = 200

            async def read(self):
                return b""

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return None

        sent = {}

        class _Session:
            def request(self, method, url, **kwargs):
                sent["method"], sent["url"] = method, url
                return _EmptyResp()

        t._session = _Session()

        t.call(cluster_requests.cancel_job_request("job-abc"))

        assert sent["method"] == "POST"
        assert sent["url"].endswith("/job-abc:cancel")


class TestCortexSharedHelper:
    """``to_cortex_fwd_bwd_payload`` lives under ``arctic_platform.integrations``
    so the reshape rule sits next to the dispatch shim that uses it."""

    def test_reshape_matches_cortex_wire_shape(self):
        import torch

        from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload

        ids = torch.zeros((2, 10), dtype=torch.int64)
        attn = torch.ones((2, 10), dtype=torch.int64)
        adv = torch.zeros((2, 10))
        resp_mask = torch.ones((2, 10), dtype=torch.int64)
        out = to_cortex_fwd_bwd_payload(
            {
                "batch": {
                    "input_ids": ids,
                    "attention_mask": attn,
                    "advantages": adv,
                    "response_mask": resp_mask,
                },
                "meta": {},
            },
        )
        assert out["args"] == ()
        assert torch.equal(out["kwargs"]["input_ids"], ids)
        assert torch.equal(out["kwargs"]["attention_mask"], attn)
        # Server-side GRPO reads these from context for its preflight.
        assert torch.equal(out["context"]["input_ids"], ids)
        assert "advantages" in out["context"]
        assert "loss_mask" in out["context"]
        assert out["processing"]["loss_fn"] == "grpo"
        assert "old_log_probs" not in out["kwargs"]
        assert "old_log_probs_shifted" not in out["context"]

    def test_balance_stamp_rides_the_wire(self):
        """True and False both reach context. A missing switch stays missing."""
        import torch

        from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload

        ids = torch.zeros((2, 10), dtype=torch.int64)
        attn = torch.ones((2, 10), dtype=torch.int64)
        batch = {
            "input_ids": ids,
            "attention_mask": attn,
            "advantages": torch.zeros((2, 10)),
            "response_mask": torch.ones((2, 10), dtype=torch.int64),
        }
        opted_out = to_cortex_fwd_bwd_payload(
            {"batch": batch, "meta": {"balance_batch": False, "load_balancer": True}}
        )
        assert opted_out["context"]["balance_batch"] is False
        assert "load_balancer" not in opted_out["context"]
        opted_in = to_cortex_fwd_bwd_payload({"batch": batch, "meta": {"balance_batch": True}})
        assert opted_in["context"]["balance_batch"] is True
        absent = to_cortex_fwd_bwd_payload({"batch": batch, "meta": {}})
        assert "balance_batch" not in absent["context"]

    def test_left_pads_are_rewritten_to_trailing_pads(self):
        """SkyRL left-pads to the batch's longest sequence; Cortex's packer
        rejects that outright ("packing requires left-aligned rows")."""
        import torch

        from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload

        # Row 0 is full width; row 1 carries two leading pad columns.
        ids = torch.tensor([[1, 2, 3, 4], [0, 0, 7, 8]])
        attn = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
        adv = torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, -0.25, -0.25]])
        resp_mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])

        out = to_cortex_fwd_bwd_payload(
            {
                "batch": {
                    "input_ids": ids,
                    "attention_mask": attn,
                    "advantages": adv,
                    "response_mask": resp_mask,
                },
                "meta": {},
            },
        )

        assert torch.equal(out["kwargs"]["input_ids"], torch.tensor([[1, 2, 3, 4], [7, 8, 0, 0]]))
        assert torch.equal(out["kwargs"]["attention_mask"], torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]))
        # The scored tokens and their advantages must have moved with the ids.
        loss_mask, moved_adv = out["context"]["loss_mask"], out["context"]["advantages"]
        assert out["kwargs"]["input_ids"][1][loss_mask[1]].tolist() == [7, 8]
        assert moved_adv[1][loss_mask[1]].tolist() == [-0.25, -0.25]
        # Row 0 was already aligned and must be untouched.
        assert out["kwargs"]["input_ids"][0].tolist() == [1, 2, 3, 4]
        assert moved_adv[0][loss_mask[0]].tolist() == [0.5, 0.5]
        # No advantage may survive on a padding column.
        assert torch.equal((moved_adv != 0), loss_mask)

    def test_explicit_old_log_probs_ride_the_alignment(self):
        """A caller-supplied pi_old has to move with the tokens it scored."""
        import torch

        from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload

        ids = torch.tensor([[1, 2, 3, 4], [0, 0, 7, 8]])
        attn = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
        resp_mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
        old = torch.tensor([[0.0, 0.0, -0.1, -0.2], [0.0, 0.0, -0.7, -0.8]], dtype=torch.float64)
        out = to_cortex_fwd_bwd_payload(
            {
                "batch": {
                    "input_ids": ids,
                    "attention_mask": attn,
                    "advantages": resp_mask.to(torch.float32),
                    "response_mask": resp_mask,
                    "old_log_probs": torch.full((2, 4), -9.0),
                },
                "meta": {"temperature": 0.7, "calculate_entropy": True},
            },
            old_log_probs_shifted=old,
        )
        sent = out["context"]["old_log_probs_shifted"]
        assert sent.dtype == torch.float32
        assert out["context"]["temperature"] == 0.7
        assert out["context"]["calculate_entropy"] is True
        assert "temperature" not in out["kwargs"]
        assert "calculate_entropy" not in out["kwargs"]
        loss_mask = out["context"]["loss_mask"]
        assert sent[1][loss_mask[1]].tolist() == pytest.approx([-0.7, -0.8])
        assert sent[0][loss_mask[0]].tolist() == pytest.approx([-0.1, -0.2])
        # The batch's own ``old_log_probs`` are still dropped.
        assert "old_log_probs" not in out["kwargs"]

    def test_already_aligned_batch_is_passed_through(self):
        """An already-conformant batch must not pay for
        a gather -- and must not be perturbed by one."""
        import torch

        from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload

        ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        attn = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        out = to_cortex_fwd_bwd_payload(
            {
                "batch": {
                    "input_ids": ids,
                    "attention_mask": attn,
                    "advantages": torch.zeros((2, 4)),
                    "response_mask": torch.tensor([[0, 0, 1, 0], [0, 1, 0, 0]]),
                },
                "meta": {},
            },
        )

        assert out["kwargs"]["input_ids"] is ids
        assert out["kwargs"]["attention_mask"] is attn

    def test_processing_matches_jae_cookbook_contract(self):
        """The wire contract mirrors ``cortex-client/recipes/rl_loop.py``:
        loss_agg_mode / entropy_coeff / eps_clip explicit, no dp_size."""
        import torch

        from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload

        ids = torch.zeros((2, 10), dtype=torch.int64)
        out = to_cortex_fwd_bwd_payload(
            {
                "batch": {
                    "input_ids": ids,
                    "attention_mask": torch.ones((2, 10), dtype=torch.int64),
                    "advantages": torch.zeros((2, 10)),
                    "response_mask": torch.ones((2, 10), dtype=torch.int64),
                },
                "meta": {},
            },
        )
        cfg = out["processing"]["config"]
        assert cfg["loss_agg_mode"] == "token-mean"
        assert cfg["entropy_coeff"] == 0.0
        assert cfg["eps_clip"] == 0.2
        # dp_size / prox_logp_method are NOT in Jae's cookbook and must not
        # leak into the wire: dp_size acts as an extra LR divisor at scale.
        assert "dp_size" not in cfg
        assert "prox_logp_method" not in cfg

    def test_caller_loss_config_wins_but_loss_fn_stays_pinned(self):
        """Recipe-supplied loss knobs propagate to the server, but ``loss_fn``
        does not: this lowering builds a payload the server's ``grpo`` loss can
        read, so honouring a caller's alias would hand those tensors to a loss
        that expects a different contract."""
        import torch

        from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload

        ids = torch.zeros((2, 10), dtype=torch.int64)
        out = to_cortex_fwd_bwd_payload(
            {
                "batch": {
                    "input_ids": ids,
                    "attention_mask": torch.ones((2, 10), dtype=torch.int64),
                    "advantages": torch.zeros((2, 10)),
                    "response_mask": torch.ones((2, 10), dtype=torch.int64),
                },
                "meta": {"global_batch_size": 128},
            },
            processing={
                "loss_fn": "some_other_grpo",
                "config": {"eps_clip": 0.3, "loss_agg_mode": "seq-mean-token-sum", "entropy_coeff": 0.01},
            },
        )
        cfg = out["processing"]["config"]
        assert cfg["eps_clip"] == 0.3
        assert cfg["loss_agg_mode"] == "seq-mean-token-sum"
        assert cfg["entropy_coeff"] == 0.01
        assert out["processing"]["loss_fn"] == "grpo"
        assert cfg["global_batch_size"] == 128  # meta fallback still applied

    def test_missing_response_mask_fails_loud(self):
        """Falling back to ``attention_mask`` would silently train on prompt
        tokens; refuse instead of producing a wrong-but-plausible gradient."""
        import pytest
        import torch

        from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload

        with pytest.raises(ValueError, match="loss_mask.*response_mask"):
            to_cortex_fwd_bwd_payload(
                {
                    "batch": {
                        "input_ids": torch.zeros((2, 10), dtype=torch.int64),
                        "attention_mask": torch.ones((2, 10), dtype=torch.int64),
                        "advantages": torch.zeros((2, 10)),
                    },
                    "meta": {},
                },
            )

    def test_non_zorro_forward_left_aligns_leading_pads(self):
        import torch

        from arctic_platform.integrations._cortex_shared import to_cortex_fwd_payload

        ids = torch.tensor([[0, 0, 1, 2], [5, 6, 7, 8]])
        attn = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
        out = to_cortex_fwd_payload(
            {"batch": {"input_ids": ids, "attention_mask": attn}, "meta": {"zorro_train_enable": False}}
        )
        assert out["kwargs"]["input_ids"][0].tolist() == [1, 2, 0, 0]
        assert out["kwargs"]["attention_mask"][0].tolist() == [1, 1, 0, 0]

    def test_zorro_fwd_bwd_keeps_the_client_layout_and_the_snapshot(self):
        import torch

        from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload

        ids = torch.tensor([[0, 5, 6, 7, 8], [0, 1, 2, 3, 4]])
        attn = torch.tensor([[0, 1, 1, 1, 1], [0, 1, 1, 1, 1]])
        prompts = torch.tensor([[0, 5, 6], [0, 1, 2]])
        old = torch.arange(10, dtype=torch.float32).reshape(2, 5)
        out = to_cortex_fwd_bwd_payload(
            {
                "batch": {
                    "input_ids": ids,
                    "attention_mask": attn,
                    "advantages": torch.zeros((2, 5)),
                    "response_mask": torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]]),
                    "prompts": prompts,
                    "old_log_probs": old,
                },
                "meta": {
                    "zorro_train_enable": True,
                    "max_prompt_len": 3,
                    "balance_batch": True,
                    "pad_token_id": 151643,
                    "temperature": 0.7,
                },
            }
        )
        assert torch.equal(out["kwargs"]["input_ids"], ids)
        assert torch.equal(out["kwargs"]["prompts"], prompts)
        assert torch.equal(out["kwargs"]["old_log_probs"], old)
        assert out["context"]["zorro_train_enable"] is True
        assert out["context"]["max_prompt_len"] == 3
        assert out["context"]["pad_token_id"] == 151643
        assert out["context"]["temperature"] == 0.7


class TestCortexPredictNextFrame:
    """SkyRL scores ``input_ids[i]`` at column ``i``; Cortex weights column ``i``
    by its prediction of ``input_ids[i + 1]`` (labels are ``roll(input_ids, -1)``)."""

    @staticmethod
    def _skyrl_batch():
        """Three SkyRL rows, ``[pad | prompt | response | pad]`` with P=3, R=3."""
        import torch

        pad = 0
        ids = torch.tensor(
            [
                [11, 12, 13, 21, 22, pad],  # p=3, r=2
                [pad, pad, 14, 31, 32, 33],  # p=1, r=3
                [15, 16, 17, 41, 42, 43],  # p=3, r=3, full width
            ]
        )
        attn = torch.tensor([[1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
        loss_mask = torch.tensor([[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
        adv = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.4, 0.5, 0.0],
                [0.0, 0.0, 0.0, -0.1, -0.2, -0.3],
                [0.0, 0.0, 0.0, 0.7, 0.8, 0.9],
            ]
        )
        return {"input_ids": ids, "attention_mask": attn, "loss_mask": loss_mask, "advantages": adv}

    def test_skyrl_rows_are_aligned_then_shifted(self):
        import torch

        from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload

        out = to_cortex_fwd_bwd_payload(
            {"batch": self._skyrl_batch(), "meta": {"zorro_train_enable": False}}, response_aligned=True
        )
        ids = out["kwargs"]["input_ids"]
        assert ids.tolist() == [[11, 12, 13, 21, 22, 0], [14, 31, 32, 33, 0, 0], [15, 16, 17, 41, 42, 43]]
        loss_mask = out["context"]["loss_mask"]
        assert loss_mask.int().tolist() == [[0, 0, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0]]
        assert torch.equal(
            out["context"]["advantages"],
            torch.tensor(
                [[0.0, 0.0, 0.4, 0.5, 0.0, 0.0], [-0.1, -0.2, -0.3, 0.0, 0.0, 0.0], [0.0, 0.0, 0.7, 0.8, 0.9, 0.0]]
            ),
        )
        # The targets Cortex scores at the active positions are exactly the responses.
        labels = torch.roll(ids, shifts=-1, dims=-1)
        assert [labels[row][loss_mask[row]].tolist() for row in range(3)] == [[21, 22], [31, 32, 33], [41, 42, 43]]
        # The full-width row's last column would score its own first token (15).
        assert labels[2, -1].item() == 15 and not loss_mask[2, -1]

    def test_shift_follows_left_alignment(self):
        """A pad gap inside a row closes up when aligned. Shifting first would
        put the first target on the gap and drop it."""
        import torch

        from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload

        batch = {
            "input_ids": torch.tensor([[5, 6, 0, 7, 8]]),
            "attention_mask": torch.tensor([[1, 1, 0, 1, 1]]),
            "loss_mask": torch.tensor([[0, 0, 0, 1, 1]]),
            "advantages": torch.tensor([[0.0, 0.0, 0.0, 2.0, 3.0]]),
        }
        out = to_cortex_fwd_bwd_payload({"batch": batch, "meta": {}}, response_aligned=True)
        assert out["kwargs"]["input_ids"].tolist() == [[5, 6, 7, 8, 0]]
        assert out["context"]["loss_mask"].int().tolist() == [[0, 1, 1, 0, 0]]
        assert out["context"]["advantages"].tolist() == [[0.0, 2.0, 3.0, 0.0, 0.0]]

    def test_default_frame_is_left_for_callers_already_in_it(self):
        """Tinker builds ``response_mask`` and ``advantages`` in the Cortex frame."""
        from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload

        out = to_cortex_fwd_bwd_payload({"batch": self._skyrl_batch(), "meta": {}})
        assert out["context"]["loss_mask"].int().tolist() == [
            [0, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1],
        ]

    def test_caller_old_log_probs_shifted_are_not_shifted_again(self):
        import torch

        from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload

        old = torch.tensor([[0.0, 0.0, -1.0, -2.0, 0.0, 0.0], [0.0, 0.0, -3.0, -4.0, -5.0, 0.0], [0.0] * 6])
        out = to_cortex_fwd_bwd_payload(
            {"batch": self._skyrl_batch(), "meta": {}}, old_log_probs_shifted=old, response_aligned=True
        )
        assert out["context"]["old_log_probs_shifted"][:2].tolist() == [
            [0.0, 0.0, -1.0, -2.0, 0.0, 0.0],
            [-3.0, -4.0, -5.0, 0.0, 0.0, 0.0],
        ]

    def test_zorro_rows_are_not_shifted(self):
        """The patched forward returns one log-prob per response token."""
        import torch

        from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload

        batch = self._skyrl_batch()
        batch["prompts"] = batch["input_ids"][:, :3]
        out = to_cortex_fwd_bwd_payload(
            {"batch": batch, "meta": {"zorro_train_enable": True, "max_prompt_len": 3}}, response_aligned=True
        )
        assert torch.equal(out["context"]["loss_mask"], batch["loss_mask"].bool())
        assert torch.equal(out["context"]["advantages"], batch["advantages"])

    def test_skyrl_shim_sends_the_predict_next_frame(self):
        import asyncio

        from arctic_platform.integrations._cortex_dispatch import _CortexClientShim

        seen = {}

        class Client:
            async def fwd_bwd(self, payload):
                seen["payload"] = payload
                return {}

        shim = _CortexClientShim.__new__(_CortexClientShim)
        shim._legacy_config = shim._unified_config = None
        shim._client = Client()
        asyncio.run(shim.fwd_bwd({"batch": self._skyrl_batch(), "meta": {}}))
        assert seen["payload"]["context"]["loss_mask"][1].int().tolist() == [1, 1, 1, 0, 0, 0]


class TestCortexActorConfig:
    """SkyRL and verl send their loss settings as ``meta['actor_config']`` and no
    ``processing.config``; the server's ``grpo`` reads them under its own names."""

    # What SkyRL's ``_build_meta`` sends, plus keys grpo has no use for.
    ACTOR = {
        "loss_agg_mode": "seq-mean-token-mean",
        "kl_loss_coef": 0.0,
        "kl_loss_type": "low_var_kl",
        "clip_ratio": 0.2,
        "clip_ratio_low": 0.15,
        "clip_ratio_high": 0.28,
        "clip_ratio_c": 3.0,
        "entropy_coeff": 0.001,
        "use_kl_loss": False,
        "calculate_entropy": True,
        "rollout_n": 4,
    }

    @staticmethod
    def _payload(meta, processing=None, zorro=False):
        import torch

        from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload

        batch = {
            "input_ids": torch.zeros((2, 4), dtype=torch.int64),
            "attention_mask": torch.ones((2, 4), dtype=torch.int64),
            "advantages": torch.zeros((2, 4)),
            "response_mask": torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
        }
        if zorro:
            batch["prompts"] = batch["input_ids"][:, :2]
            meta = {**meta, "zorro_train_enable": True, "max_prompt_len": 2}
        return to_cortex_fwd_bwd_payload({"batch": batch, "meta": meta}, processing=processing)

    def test_skyrl_actor_config_reaches_grpo(self):
        from arctic_platform.rl.processors.grpo import _CORTEX_GRPO_VARIANT
        from arctic_platform.rl.processors.grpo import _validate_plain_config

        skyrl_processing = {"post": ["apply_temperature", "compute_entropy_and_logprobs"], "loss_fn": "verl_grpo"}
        cfg = self._payload({"actor_config": self.ACTOR, "global_batch_size": 8}, skyrl_processing)["processing"][
            "config"
        ]
        assert cfg == {
            "eps_clip": 0.15,
            "eps_clip_higher": 0.28,
            "c_clip": 3.0,
            "loss_agg_mode": "seq-mean-token-mean",
            "entropy_coeff": 0.001,
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "kl_loss_type": "low_var_kl",
            "global_batch_size": 8,
        }
        # The Cortex grpo rejects unknown keys; every forwarded one is known.
        _validate_plain_config(cfg, _CORTEX_GRPO_VARIANT)

    def test_explicit_processing_config_wins_over_actor_config(self):
        cfg = self._payload(
            {"actor_config": self.ACTOR}, {"config": {"eps_clip": 0.3, "loss_agg_mode": "token-mean"}}
        )["processing"]["config"]
        assert cfg["eps_clip"] == 0.3
        assert cfg["loss_agg_mode"] == "token-mean"
        assert cfg["eps_clip_higher"] == 0.28
        assert cfg["entropy_coeff"] == 0.001

    def test_unset_sides_fall_back_to_clip_ratio(self):
        actor = {"clip_ratio": 0.25, "clip_ratio_low": None, "clip_ratio_high": None, "clip_ratio_c": None}
        cfg = self._payload({"actor_config": actor})["processing"]["config"]
        assert cfg == {"eps_clip": 0.25, "eps_clip_higher": 0.25, "loss_agg_mode": "token-mean", "entropy_coeff": 0.0}

    def test_without_actor_config_defaults_are_unchanged(self):
        cfg = self._payload({})["processing"]["config"]
        assert cfg == {"eps_clip": 0.2, "loss_agg_mode": "token-mean", "entropy_coeff": 0.0}

    def test_zorro_path_forwards_actor_config(self):
        actor = {**self.ACTOR, "loss_agg_mode": "token-mean"}
        cfg = self._payload({"actor_config": actor}, zorro=True)["processing"]["config"]
        assert (cfg["eps_clip"], cfg["eps_clip_higher"], cfg["c_clip"]) == (0.15, 0.28, 3.0)
        assert cfg["entropy_coeff"] == 0.001

    def test_forwarded_clip_changes_the_server_loss(self):
        """ratio = e^0.25 = 1.284 with A = 1: clipped at 1.28 by SkyRL's 0.28, at 1.2 by the default."""
        import torch

        from arctic_platform.rl.processors.grpo import _grpo_loss

        actor = {**self.ACTOR, "loss_agg_mode": "token-mean", "entropy_coeff": 0.0}
        context = {
            "advantages": torch.tensor([[1.0]]),
            "loss_mask": torch.tensor([[True]]),
            "old_log_probs_shifted": torch.tensor([[0.0]]),
        }
        outputs = {"logprobs": torch.tensor([[0.25]])}
        forwarded = self._payload({"actor_config": actor})["processing"]["config"]
        default = self._payload({})["processing"]["config"]
        loss, _ = _grpo_loss(outputs, context, forwarded, "cpu")
        assert loss.item() == pytest.approx(-1.28)
        loss, _ = _grpo_loss(outputs, context, default, "cpu")
        assert loss.item() == pytest.approx(-1.2)


class TestCortexShimSaveWeightsFailsLoud:
    """``save_weights`` raises ``NotImplementedError`` — Cortex sub-jobs don't
    share disk, so a silent no-op would leave sampling on stale weights."""

    def test_save_weights_raises(self, monkeypatch):
        import asyncio

        monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://mock")
        from arctic_platform.integrations._cortex_dispatch import _CortexClientShim
        from arctic_platform.rl.config import ArcticRLClientConfig as Legacy

        # Left at SkyRL's own default: the shim is reached by the recipe naming
        # the Cortex entrypoint, not by anything in this config.
        legacy = Legacy(model_name="m", training_gpus=1, sampling_gpus=1)

        # Skip the real ArcticRLClient constructor (needs live transport init).
        shim = _CortexClientShim.__new__(_CortexClientShim)
        shim._legacy_config = legacy
        shim._unified_config = None
        shim._client = None

        with pytest.raises(NotImplementedError, match="Cortex has no disk-based"):
            asyncio.run(shim.save_weights("/tmp/w"))


class TestCortexShimForwardsLikeOnPrem:
    """``fwd_no_grad`` sends the on-prem payload and returns its logprobs.

    A reference-model request is still refused: zeros there would turn a KL
    term into a function of the live policy.
    """

    @staticmethod
    def _shim(client=None):
        from arctic_platform.integrations._cortex_dispatch import _CortexClientShim

        shim = _CortexClientShim.__new__(_CortexClientShim)
        shim._legacy_config = shim._unified_config = None
        shim._client = client
        return shim

    @staticmethod
    def _batch():
        import torch

        return {
            "batch": {
                "input_ids": torch.zeros((2, 5), dtype=torch.long),
                "attention_mask": torch.ones((2, 5), dtype=torch.long),
            },
            "meta": {"pad_token_id": 0, "zorro_train_enable": False},
        }

    def test_reference_model_request_raises(self):
        import asyncio

        with pytest.raises(NotImplementedError, match="reference-model log-probs"):
            asyncio.run(self._shim().fwd_no_grad(self._batch(), reference_model=True))

    def test_zorro_forward_sends_the_client_layout(self):
        import asyncio

        import torch

        seen = {}

        class Client:
            async def fwd_no_grad(self, payload):
                seen["payload"] = payload
                return {"logprobs": torch.arange(8, dtype=torch.float32).reshape(2, 4)}

        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        batch = {
            "batch": {
                "input_ids": ids,
                "attention_mask": torch.ones(2, 4, dtype=torch.long),
                "prompts": ids[:, :2],
                "position_ids": torch.arange(4).repeat(2, 1),
            },
            "meta": {
                "zorro_train_enable": True,
                "max_prompt_len": 2,
                "max_response_len": 2,
                "pad_token_id": 151643,
            },
            "processing": {"post": ["compute_entropy_and_logprobs"], "loss_fn": None},
        }
        out = asyncio.run(self._shim(Client()).fwd_no_grad(batch, reference_model=False))
        sent = seen["payload"]
        assert torch.equal(sent["kwargs"]["prompts"], ids[:, :2])
        assert sent["context"]["zorro_train_enable"] is True
        assert sent["context"]["max_prompt_len"] == 2
        assert sent["context"]["pad_token_id"] == 151643
        assert sent["processing"]["loss_fn"] is None
        assert torch.equal(out["batch"]["logprobs"], out["batch"]["log_probs"])
        assert out["batch"]["logprobs"].shape == (2, 4)

    def test_forward_returns_the_server_logprobs(self):
        import asyncio

        import torch

        class Client:
            async def fwd_no_grad(self, payload):
                return {"logprobs": torch.ones((2, 5))}

        out = asyncio.run(self._shim(Client()).fwd_no_grad(self._batch(), reference_model=False))
        assert out["batch"]["logprobs"].shape == (2, 5)
        assert float(out["batch"]["logprobs"].sum()) == 10.0


def test_zorro_is_only_compatible_with_arctic_rl_client(monkeypatch):
    monkeypatch.setattr(base_module, "make_transport", FakeTransport)
    enabled = {"zorro_train_enable": True}
    sft = ArcticSFTClientConfig(
        model_name="m",
        training_gpus=1,
        training=TrainingConfig(checkpoint_path="/tmp/c", ds_worker_config=enabled),
    )
    plain = ArcticClientConfig(model_name="m", training_gpus=1, training=TrainingConfig(ds_worker_config=enabled))
    nested = ArcticClientConfig(
        model_name="m",
        training_gpus=1,
        training=TrainingConfig(
            ds_worker_config={"zorro_train": {"enable": True, "response_len": 8, "temperature": 1.0}}
        ),
    )

    with pytest.raises(ValueError, match="only compatible with ArcticRLClient"):
        ArcticSFTClient(sft)
    with pytest.raises(ValueError, match="only compatible with ArcticRLClient"):
        ArcticClient(plain)
    with pytest.raises(ValueError, match="only compatible with ArcticRLClient"):
        AsyncArcticClient(nested)

    ArcticSFTClient(
        ArcticSFTClientConfig(
            model_name="m",
            training_gpus=1,
            training=TrainingConfig(checkpoint_path="/tmp/c", ds_worker_config={"zorro_train_enable": False}),
        )
    )
    ArcticRLClient(plain)
    AsyncArcticRLClient(nested)
