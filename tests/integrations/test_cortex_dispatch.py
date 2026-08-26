# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cortex routing for the legacy ``arctic_platform.rl`` API, which is what SkyRL
is written against.

SkyRL builds the legacy flat config itself and hard-codes ``backend="local"``,
so the two things worth pinning are that ``ARCTIC_BACKEND=cortex`` can still
redirect it, and that the resulting object answers to the legacy accessor names
its entrypoint reads.
"""

from __future__ import annotations

import sys
import types

import pytest

from arctic_platform.rl.config import ArcticRLClientConfig


@pytest.fixture(autouse=True)
def _cortex_env(monkeypatch):
    monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://mock")
    monkeypatch.setenv("ARCTIC_BACKEND", "cortex")
    # For non-Cortex backends the legacy config derives host/port via ray, and a
    # CPU-only Cortex driver has no ray installed -- which is precisely why the
    # cortex branch returns early. Stub it so the non-cortex cases stay testable
    # in an env without ray.
    stub = types.ModuleType("arctic_platform.rl.ray_cluster")
    stub.primary_ip = lambda: "127.0.0.1"
    monkeypatch.setitem(sys.modules, "arctic_platform.rl.ray_cluster", stub)


def _legacy(**overrides) -> ArcticRLClientConfig:
    base = dict(model_name="Qwen/Qwen3-0.6B", training_gpus=1, sampling_gpus=1)
    return ArcticRLClientConfig(**{**base, **overrides})


class FakeUnifiedClient:
    """Stands in for AsyncArcticRLClient, whose __init__ provisions real jobs."""

    def __init__(self, config):
        self.config = config
        self.jobs = type("Jobs", (), {"training": 11, "sampling": 22, "log_prob": None})()
        self.shutdown_calls = 0

    async def shutdown(self):
        self.shutdown_calls += 1

    async def fwd_bwd(self, batch, **kw):
        return {"metrics": {"loss": 0.5}}


@pytest.fixture
def dispatch(monkeypatch):
    import arctic_platform.client as client_module
    from arctic_platform.integrations import _cortex_dispatch

    monkeypatch.setattr(client_module, "AsyncArcticRLClient", FakeUnifiedClient)
    return _cortex_dispatch


class TestBackendEnvPromotion:
    """SkyRL hard-codes backend="local", so that value is treated as unset."""

    def test_local_is_promoted(self):
        assert _legacy().backend == "cortex"

    def test_explicit_non_default_wins(self):
        assert _legacy(backend="dss-platform").backend == "dss-platform"

    def test_no_env_means_no_promotion(self, monkeypatch):
        monkeypatch.delenv("ARCTIC_BACKEND")
        assert _legacy().backend == "local"

    def test_cortex_skips_host_port_derivation(self):
        """There is no local server to address, and deriving it would import ray."""
        config = _legacy()
        assert (config.host, config.port) == (None, None)


class TestFactoryRouting:
    def test_cortex_backend_routes_to_dispatch(self, dispatch):
        from arctic_platform.rl import create_arctic_rl_client

        client = create_arctic_rl_client(_legacy())
        assert isinstance(client, dispatch._LegacyCortexClient)

    def test_unknown_protocol_still_raises(self):
        """The cortex branch must not have swallowed the old error path."""
        from arctic_platform.rl import create_arctic_rl_client

        config = _legacy(backend="dss-platform")
        object.__setattr__(config, "comm_protocol", "carrier-pigeon")
        with pytest.raises(ValueError, match="Invalid communication protocol"):
            create_arctic_rl_client(config)


class TestConfigTranslation:
    def test_core_fields_carry_over(self):
        from arctic_platform.integrations._cortex_dispatch import _to_unified_config

        unified = _to_unified_config(_legacy(seed=7, log_prob_gpus=2))
        assert unified.model_name == "Qwen/Qwen3-0.6B"
        assert (unified.seed, unified.training_gpus, unified.log_prob_gpus) == (7, 1, 2)
        assert unified.backend.base_url == "http://mock"

    def test_optimizer_folded_into_ds_config(self):
        """Cortex's sub-job builder reads the optimizer out of ds_config."""
        from arctic_platform.integrations._cortex_dispatch import _to_unified_config

        optimizer = {"type": "AdamW", "params": {"lr": 1e-5}}
        unified = _to_unified_config(_legacy(training_config={"optimizer": optimizer}))
        assert unified.training.ds_config["optimizer"] == optimizer

    def test_existing_ds_config_optimizer_is_not_clobbered(self):
        from arctic_platform.integrations._cortex_dispatch import _to_unified_config

        unified = _to_unified_config(
            _legacy(
                ds_config={"optimizer": {"type": "keep-me"}},
                training_config={"optimizer": {"type": "discard-me"}},
            )
        )
        assert unified.training.ds_config["optimizer"]["type"] == "keep-me"

    def test_job_ids_forwarded_for_worker_reattach(self):
        from arctic_platform.integrations._cortex_dispatch import _to_unified_config

        unified = _to_unified_config(_legacy(training_job_id=5, sampling_job_id=6))
        assert (unified.training_job_id, unified.sampling_job_id) == (5, 6)

    def test_absent_job_ids_stay_absent(self):
        from arctic_platform.integrations._cortex_dispatch import _to_unified_config

        assert _to_unified_config(_legacy()).training_job_id is None


class TestLegacyAccessors:
    """What `recipes/rl/skyrl/*/arctic_rl/entrypoint.py` reads off the client."""

    @pytest.fixture
    def client(self, dispatch):
        return dispatch.create_cortex_client(_legacy())

    def test_job_id_properties(self, client):
        assert (client.training_job_id, client.sampling_job_id, client.log_prob_job_id) == (11, 22, None)

    def test_config_is_the_legacy_shape(self, client):
        assert isinstance(client.config, ArcticRLClientConfig)

    def test_reconnect_config_is_legacy_shaped_with_job_ids(self, client):
        """Workers rebuild from this, and they rebuild a legacy config."""
        reconnect = client.reconnect_config()
        assert isinstance(reconnect, ArcticRLClientConfig)
        assert (reconnect.training_job_id, reconnect.sampling_job_id) == (11, 22)

    def test_get_server_state_is_none_not_an_error(self, client):
        """The unified client raises for non-Ray transports; SkyRL wants a value."""
        assert client.get_server_state() is None

    def test_unknown_attributes_delegate(self, client):
        assert callable(client.fwd_bwd)

    def test_shutdown_from_sync_context_completes(self, client):
        assert client.shutdown() is None
        assert client._client.shutdown_calls == 1

    def test_shutdown_inside_running_loop_is_awaitable(self, client):
        import asyncio

        async def main():
            await client.shutdown()

        asyncio.run(main())
        assert client._client.shutdown_calls == 1
