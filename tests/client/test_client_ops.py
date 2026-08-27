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

import pytest

from arctic_platform.client import OPS
from arctic_platform.client import ArcticClientConfig
from arctic_platform.client import ArcticRLClient
from arctic_platform.client import ArcticSFTClient
from arctic_platform.client import AsyncArcticRLClient
from arctic_platform.client import JobHandles
from arctic_platform.client import OnPremConfig
from arctic_platform.client import Request
from arctic_platform.client import Transport
from arctic_platform.client import base as base_module
from arctic_platform.client import unresolved_ops
from arctic_platform.client.transport import method_name

TRAINING, SAMPLING, LOG_PROB = 1, 2, 3


class FakeTransport(Transport):
    def __init__(self, config: ArcticClientConfig, server_state=None) -> None:
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
        assert req.body == {"path": "/tmp/x", "step": 2}

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
        """AsyncArcticRLClient(cfg, server_state=...) reattaches via the Ray transport."""
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
        cfg = ArcticClientConfig(
            model_name="m",
            backend=OnPremConfig(protocol="ray"),
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


class TestTransportSelection:
    def test_make_transport_selects_ray(self, monkeypatch):
        """onprem + ray routes to RayTransport (constructed lazily, so patch it)."""
        import arctic_platform.client.transports.onprem_ray as ray_mod

        class DummyRay:
            def __init__(self, config, server_state=None):
                self.config = config
                self.server_state = server_state

        monkeypatch.setattr(ray_mod, "RayTransport", DummyRay)
        cfg = ArcticClientConfig(model_name="m", backend=OnPremConfig(protocol="ray"), training_gpus=1)
        assert isinstance(base_module.make_transport(cfg), DummyRay)

    def test_make_transport_selects_http_for_onprem(self):
        """onprem + http (the default) routes to HttpTransport."""
        from arctic_platform.client.transports.onprem_http import HttpTransport

        cfg = ArcticClientConfig(model_name="m", backend=OnPremConfig(protocol="http"), training_gpus=1)
        assert isinstance(base_module.make_transport(cfg), HttpTransport)

    def test_make_transport_forwards_server_state_to_ray(self, monkeypatch):
        """make_transport threads server_state into the Ray transport (reconnect path)."""
        import arctic_platform.client.transports.onprem_ray as ray_mod

        class DummyRay:
            def __init__(self, config, server_state=None):
                self.config = config
                self.server_state = server_state

        monkeypatch.setattr(ray_mod, "RayTransport", DummyRay)
        sentinel = object()
        cfg = ArcticClientConfig(model_name="m", backend=OnPremConfig(protocol="ray"), training_gpus=1)
        transport = base_module.make_transport(cfg, server_state=sentinel)
        assert transport.server_state is sentinel

    def test_make_transport_rejects_server_state_for_http(self):
        """server_state reconnect is Ray-only; HTTP transport must reject it."""
        cfg = ArcticClientConfig(model_name="m", backend=OnPremConfig(protocol="http"), training_gpus=1)
        with pytest.raises(ValueError, match="server_state reconnect"):
            base_module.make_transport(cfg, server_state=object())


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
    """Clear ARCTIC_BACKEND / ARCTIC_CORTEX_* before each test; a leaked env from
    a parallel process would flip config promotion under our feet."""
    import os

    for k in list(os.environ):
        if k.startswith("ARCTIC_BACKEND") or k.startswith("ARCTIC_CORTEX_"):
            monkeypatch.delenv(k, raising=False)


class TestCortexConfigFromEnv:
    """``CortexConfig.from_env`` reads ``ARCTIC_CORTEX_*`` and lets explicit
    kwargs win. This is the one call-site framework adapters use to flip to
    Cortex from a shell-level env."""

    def test_base_url_only_bypasses_pat(self, monkeypatch):
        monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://mock")
        from arctic_platform.client import CortexConfig

        cfg = CortexConfig.from_env()
        assert cfg.base_url == "http://mock"
        assert cfg.host is None

    def test_host_path_requires_db_schema_pat(self, monkeypatch):
        monkeypatch.setenv("ARCTIC_CORTEX_HOST", "acct.snowflakecomputing.com")
        monkeypatch.setenv("ARCTIC_CORTEX_DATABASE", "db")
        monkeypatch.setenv("ARCTIC_CORTEX_SCHEMA", "sch")
        monkeypatch.setenv("CORTEX_PAT", "pat-value")
        from arctic_platform.client import CortexConfig

        cfg = CortexConfig.from_env()
        assert cfg.host == "acct.snowflakecomputing.com"
        assert cfg.database == "db"
        assert cfg.schema_ == "sch"
        assert cfg.resolve_pat() == "pat-value"

    def test_explicit_override_wins(self, monkeypatch):
        monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://env")
        from arctic_platform.client import CortexConfig

        cfg = CortexConfig.from_env(base_url="http://explicit")
        assert cfg.base_url == "http://explicit"


class TestUnifiedConfigDoesNotReadEnv:
    """``ArcticClientConfig`` does not swap ``backend`` from env vars.
    ``ARCTIC_BACKEND`` is only honored at framework adapter call-sites (verl's
    ``_create_rl_client_config``, the legacy ``arctic_platform.rl`` validator)."""

    def test_no_env_promotion_on_unified_config(self, monkeypatch):
        monkeypatch.setenv("ARCTIC_BACKEND", "cortex")
        cfg = ArcticClientConfig(model_name="m", backend=OnPremConfig(), training_gpus=1)
        assert cfg.backend.type == "onprem"


class TestCortexTransportNoopOps:
    """``wake_*`` / ``sleep_*`` short-circuit in the Cortex transport so the
    shim doesn't have to wrap each call — including the ones ``sync_weights``
    invokes internally."""

    def test_call_returns_empty_for_noop_op(self, monkeypatch):
        monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://mock")
        from arctic_platform.client import CortexConfig
        from arctic_platform.client.transport import Request
        from arctic_platform.client.transports.cortex import CortexTransport

        cfg = ArcticClientConfig(
            model_name="m", backend=CortexConfig.from_env(), training_gpus=1, sampling_gpus=1
        )
        t = CortexTransport(cfg)
        # Would normally raise NotImplementedError on the transport; the noop
        # short-circuit means the shim never has to guard these calls.
        assert t.call(Request("wake-inference", 1, None)) == {}
        assert t.call(Request("sleep-training", 1, {"mode": "all"})) == {}


class TestLegacyBackendEnvPromotion:
    """Legacy ``arctic_platform.rl.config.ArcticRLClientConfig`` (the shape
    SkyRL builds) promotes ``backend="local"`` -> ``"cortex"`` when
    ``ARCTIC_BACKEND=cortex`` is set. Explicit non-default backends still win.
    """

    def test_local_default_gets_promoted(self, monkeypatch):
        monkeypatch.setenv("ARCTIC_BACKEND", "cortex")
        from arctic_platform.rl.config import ArcticRLClientConfig as Legacy

        cfg = Legacy(model_name="m", backend="local", training_gpus=1)
        assert cfg.backend == "cortex"

    def test_explicit_non_default_backend_wins(self, monkeypatch):
        monkeypatch.setenv("ARCTIC_BACKEND", "cortex")
        from arctic_platform.rl.config import ArcticRLClientConfig as Legacy

        cfg = Legacy(model_name="m", backend="dss-platform", training_gpus=1)
        assert cfg.backend == "dss-platform"


class TestCortexSharedHelper:
    """``to_cortex_fwd_bwd_payload`` lives under ``arctic_platform.integrations``
    so the SkyRL shim and the verl adapter share the reshape rule."""

    def test_integration_path_importable(self):
        from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload

        assert callable(to_cortex_fwd_bwd_payload)

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
        # Pin loss_fn="grpo" so verl's "verl_grpo" alias doesn't reach the server.
        assert out["processing"]["loss_fn"] == "grpo"
        assert "old_log_probs" not in out["kwargs"]
        assert "old_log_probs_shifted" not in out["context"]

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

    def test_caller_processing_config_wins(self):
        """Recipe-supplied loss knobs override the defaults, so verl's
        ``actor.entropy_coeff`` / ``loss_agg_mode`` propagate all the way
        to the Cortex server."""
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
                "loss_fn": "verl_grpo",
                "config": {"eps_clip": 0.3, "loss_agg_mode": "seq-mean-token-sum", "entropy_coeff": 0.01},
            },
        )
        cfg = out["processing"]["config"]
        assert cfg["eps_clip"] == 0.3
        assert cfg["loss_agg_mode"] == "seq-mean-token-sum"
        assert cfg["entropy_coeff"] == 0.01
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


class TestCortexShimSaveWeightsFailsLoud:
    """``save_weights`` raises ``NotImplementedError`` — Cortex sub-jobs don't
    share disk, so a silent no-op would leave sampling on stale weights."""

    def test_save_weights_raises(self, monkeypatch):
        import asyncio

        monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://mock")
        from arctic_platform.integrations._cortex_dispatch import _CortexClientShim
        from arctic_platform.rl.config import ArcticRLClientConfig as Legacy

        legacy = Legacy(model_name="m", backend="cortex", training_gpus=1, sampling_gpus=1)

        # Skip the real ArcticRLClient constructor (needs live transport init).
        shim = _CortexClientShim.__new__(_CortexClientShim)
        shim._legacy_config = legacy
        shim._unified_config = None
        shim._client = None

        with pytest.raises(NotImplementedError, match="Cortex has no disk-based"):
            asyncio.run(shim.save_weights("/tmp/w"))
