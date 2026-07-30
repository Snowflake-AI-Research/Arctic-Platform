# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `arctic_platform.rl.create_arctic_rl_client` -> Cortex dispatch.

These pin the "zero adapter change" contract:

- Both merged upstream integrations
  (`NovaSky-AI/SkyRL#1837:integrations/arctic_rl/` +
  `arctic_platform/integrations/verl/adapter.py` used by verl#6422) call
  `create_arctic_rl_client(config, server_state)` and read the returned
  client's public surface (async methods + property attrs).
- Flipping `config.backend = "cortex"` in yaml is the only knob those
  integrations touch; no import swap, no `await`/`sync` restructure.
- The returned shim's async surface matches the legacy
  `ArcticRLHTTPClient` / `ArcticRLRayClient` verbatim so both integrations'
  call sites keep working.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

# `arctic_platform.rl` is lazy on the heavy exports (see the `__getattr__`
# in `arctic_platform/rl/__init__.py`). Importing `ArcticRLClientConfig`
# alone is pydantic-only, so this test module runs on any environment,
# Cortex-only drivers included. The `create_arctic_rl_client` symbol
# below only materialises the on-prem-server import chain if a test
# constructs a `backend="local"` config — the Cortex branch short-
# circuits before touching those modules.
from arctic_platform.rl import ArcticRLClientConfig, create_arctic_rl_client  # noqa: E402
from arctic_platform.rl._cortex_dispatch import _CortexClientShim, _to_unified_config  # noqa: E402


# ---------------------------------------------------------------------------
# Fake unified client: records every call so tests can assert forwarding.
# ---------------------------------------------------------------------------


class _FakeUnifiedClient:
    """Stand-in for `arctic_platform.client.ArcticRLClient` inside the shim.

    Records every method call + kwargs. Returns canned bodies shaped like
    the on-prem envelopes the integrations expect.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.training_job_id = "cortex-train-1"
        self.sampling_job_id = "cortex-sample-1"
        self.log_prob_job_id = None

    def _record(self, name: str, args: tuple, kwargs: dict) -> Any:
        self.calls.append((name, args, kwargs))
        # Shape: fwd_bwd / step responses carry a `metrics` dict; fwd_no_grad
        # carries a `batch` dict; the rest return {}.
        if name in {"fwd_bwd", "step"}:
            return {"metrics": {"grad_norm": 0.5, "avg_loss": 0.1}, "loss": 0.1}
        if name == "fwd_no_grad":
            return {"batch": {"logprobs": [[0.0]]}, "model_outputs": {"logprobs": [[0.0]]}}
        return {}

    def fwd_bwd(self, batch, **kw):
        return self._record("fwd_bwd", (batch,), kw)

    def fwd_no_grad(self, batch, **kw):
        return self._record("fwd_no_grad", (batch,), kw)

    def step(self, learning_rate=None):
        return self._record("step", (), {"learning_rate": learning_rate})

    def save_checkpoint(self, stage_info=None, path=None):
        return self._record("save_checkpoint", (), {"stage_info": stage_info, "path": path})

    def save_weights(self, path):
        return self._record("save_weights", (), {"path": path})

    def generate(self, prompts, sampling_params=None, **kw):
        return self._record("generate", (prompts,), {"sampling_params": sampling_params, **kw})

    def sync_weights(self, cuda_ipc=False, low_memory=False):
        return self._record("sync_weights", (), {"cuda_ipc": cuda_ipc, "low_memory": low_memory})

    def reset_prefix_cache(self, drain=True, timeout_s=60.0):
        return self._record("reset_prefix_cache", (), {"drain": drain, "timeout_s": timeout_s})

    def wake_inference(self, **kw):
        return self._record("wake_inference", (), kw)

    def sleep_inference(self, **kw):
        return self._record("sleep_inference", (), kw)

    def wake_training(self, **kw):
        return self._record("wake_training", (), kw)

    def sleep_training(self, **kw):
        return self._record("sleep_training", (), kw)

    def wake_log_prob(self, **kw):
        return self._record("wake_log_prob", (), kw)

    def sleep_log_prob(self, **kw):
        return self._record("sleep_log_prob", (), kw)

    def empty_training_cache(self, **kw):
        return self._record("empty_training_cache", (), kw)

    def weight_norm(self, **kw):
        return self._record("weight_norm", (), kw)

    def shutdown(self):
        self._record("shutdown", (), {})


def _run(coro):
    """Drive a coroutine to completion in a fresh loop.

    Mirrors `_run(...)` in SkyRL's `integrations/arctic_rl/trainer.py`
    (which uses `asyncio.run`) so if this helper works, that call site does.
    """
    return asyncio.run(coro)


@pytest.fixture
def legacy_cfg() -> ArcticRLClientConfig:
    return ArcticRLClientConfig(
        backend="cortex",
        model_name="Qwen/Qwen3-0.6B",
        training_gpus=4,
        sampling_gpus=2,
        log_prob_gpus=0,
        cortex_host="test.snowflakecomputing.com",
        cortex_database="ARCTIC_DB",
        cortex_schema="RL",
        cortex_endpoint="cortex-training",
        max_seq_len=8192,
        seed=1234,
    )


# ---------------------------------------------------------------------------
# Config translation
# ---------------------------------------------------------------------------


class TestConfigTranslation:
    def test_backend_cortex_translates_to_unified_cortex(self, legacy_cfg):
        unified = _to_unified_config(legacy_cfg)
        assert unified.backend == "cortex"
        assert unified.model_name == "Qwen/Qwen3-0.6B"
        assert unified.training_gpus == 4
        assert unified.sampling_gpus == 2
        assert unified.log_prob_gpus == 0
        assert unified.seed == 1234
        assert unified.max_seq_len == 8192

    def test_cortex_fields_threaded(self, legacy_cfg):
        unified = _to_unified_config(legacy_cfg)
        assert unified.cortex_host == "test.snowflakecomputing.com"
        assert unified.cortex_database == "ARCTIC_DB"
        assert unified.cortex_schema == "RL"
        assert unified.cortex_endpoint == "cortex-training"

    def test_omitted_cortex_fields_fall_to_unified_defaults(self):
        cfg = ArcticRLClientConfig(
            backend="cortex", model_name="Qwen/Qwen3-0.6B", training_gpus=1
        )
        unified = _to_unified_config(cfg)
        assert unified.cortex_host is None
        assert unified.cortex_database == ""  # unified default
        assert unified.cortex_endpoint == "cortex-training"  # unified default
        assert unified.cortex_pat_env_var == "CORTEX_PAT"  # unified default

    def test_onprem_fields_are_not_forwarded(self, legacy_cfg):
        """`ds_config` / `ds_worker_config` / `arctic_inference_config` are
        on-prem-only; the shim must not silently forward them because Cortex
        has no local placement to configure and picking them up would mask
        real config drift."""
        legacy_cfg.ds_config = {"train_batch_size": 32}
        legacy_cfg.ds_worker_config = {"use_liger": True}
        legacy_cfg.arctic_inference_config = {"speculative_decoding": {"model": "..."}}
        unified = _to_unified_config(legacy_cfg)
        # Unified config's on-prem sub-fields must be None (they may exist on
        # the unified schema but the translator declines to forward them).
        assert getattr(unified, "ds_config", None) is None
        assert getattr(unified, "ds_worker_config", None) is None
        assert getattr(unified, "arctic_inference_config", None) is None

    def test_reconnect_job_ids_forwarded(self):
        cfg = ArcticRLClientConfig(
            backend="cortex",
            model_name="Qwen/Qwen3-0.6B",
            training_gpus=1,
            training_job_id=42,
            sampling_job_id=43,
            log_prob_job_id=44,
        )
        unified = _to_unified_config(cfg)
        assert unified.training_job_id == 42
        assert unified.sampling_job_id == 43
        assert unified.log_prob_job_id == 44


# ---------------------------------------------------------------------------
# Shim: async surface + property surface
# ---------------------------------------------------------------------------


class TestShimAsyncSurface:
    """Each integration call site must resolve to the underlying unified
    client method with matching kwargs. Missing methods regress `await
    client.foo(...)` in the integrations."""

    @pytest.fixture
    def fake(self, legacy_cfg):
        return _FakeUnifiedClient(), _CortexClientShim(_FakeUnifiedClient(), legacy_cfg)

    @pytest.fixture
    def shim(self, legacy_cfg):
        fake = _FakeUnifiedClient()
        return fake, _CortexClientShim(fake, legacy_cfg)

    def test_fwd_bwd_forwards_batch_and_kwargs(self, shim):
        fake, shim = shim
        result = _run(shim.fwd_bwd({"input_ids": []}, reference_model=True))
        assert result["metrics"]["grad_norm"] == 0.5
        name, args, kwargs = fake.calls[-1]
        assert name == "fwd_bwd"
        assert kwargs == {"reference_model": True}

    def test_fwd_no_grad_forwards_reference_model(self, shim):
        """verl's adapter calls fwd_no_grad(payload, reference_model=True/False)."""
        fake, shim = shim
        _run(shim.fwd_no_grad({"batch": {}}, reference_model=False))
        name, _, kwargs = fake.calls[-1]
        assert name == "fwd_no_grad"
        assert kwargs == {"reference_model": False}

    def test_fwd_no_grad_forwards_post_processors(self, shim):
        """SkyRL's dispatch calls fwd_no_grad(..., post_processors=[...])."""
        fake, shim = shim
        _run(shim.fwd_no_grad({"kwargs": {}}, post_processors=["logprobs"]))
        _, _, kwargs = fake.calls[-1]
        assert kwargs == {"post_processors": ["logprobs"]}

    def test_step_forwards_learning_rate(self, shim):
        fake, shim = shim
        _run(shim.step())
        _run(shim.step(learning_rate=1e-5))
        assert fake.calls[-2] == ("step", (), {"learning_rate": None})
        assert fake.calls[-1] == ("step", (), {"learning_rate": 1e-5})

    def test_sync_weights_cuda_ipc_and_low_memory(self, shim):
        """SkyRL colocated calls `sync_weights(cuda_ipc=True)` and verl reads
        both `cuda_ipc` and `low_memory` off yaml."""
        fake, shim = shim
        _run(shim.sync_weights(cuda_ipc=True, low_memory=True))
        _, _, kwargs = fake.calls[-1]
        assert kwargs == {"cuda_ipc": True, "low_memory": True}

    def test_wake_inference_forwards_tags(self, shim):
        """verl calls wake_inference(tags=[...])."""
        fake, shim = shim
        _run(shim.wake_inference(tags=["vllm"]))
        _, _, kwargs = fake.calls[-1]
        assert kwargs == {"tags": ["vllm"]}

    def test_sleep_inference_forwards_level(self, shim):
        """verl calls sleep_inference(level=2)."""
        fake, shim = shim
        _run(shim.sleep_inference(level=2))
        _, _, kwargs = fake.calls[-1]
        assert kwargs == {"level": 2}

    def test_colocation_lifecycle_ops_all_present(self, shim):
        """SkyRL colocated path calls every wake_/sleep_/empty_/weight_norm op."""
        fake, shim = shim
        _run(shim.empty_training_cache())
        _run(shim.wake_training())
        _run(shim.sleep_training())
        _run(shim.wake_log_prob())
        _run(shim.sleep_log_prob())
        _run(shim.weight_norm())
        names = [c[0] for c in fake.calls]
        assert names == [
            "empty_training_cache",
            "wake_training",
            "sleep_training",
            "wake_log_prob",
            "sleep_log_prob",
            "weight_norm",
        ]

    def test_generate_forwards_prompts_and_params(self, shim):
        fake, shim = shim
        _run(shim.generate(["hi"], sampling_params={"temperature": 0.7}))
        name, args, kwargs = fake.calls[-1]
        assert name == "generate"
        assert args == (["hi"],)
        assert kwargs == {"sampling_params": {"temperature": 0.7}}

    def test_save_checkpoint_forwards_stage_info_and_path(self, shim):
        fake, shim = shim
        _run(shim.save_checkpoint(stage_info={"step": 10}, path="/tmp/ckpt"))
        _, _, kwargs = fake.calls[-1]
        assert kwargs == {"stage_info": {"step": 10}, "path": "/tmp/ckpt"}


class TestShimSyncSurface:
    def test_shutdown_is_sync(self, legacy_cfg):
        fake = _FakeUnifiedClient()
        shim = _CortexClientShim(fake, legacy_cfg)
        shim.shutdown()  # must NOT be a coroutine
        assert fake.calls[-1] == ("shutdown", (), {})

    def test_reconnect_config_returns_legacy_shape_with_job_ids(self, legacy_cfg):
        fake = _FakeUnifiedClient()
        shim = _CortexClientShim(fake, legacy_cfg)
        rc = shim.reconnect_config()
        assert isinstance(rc, ArcticRLClientConfig)
        assert rc.backend == "cortex"
        assert rc.training_job_id == fake.training_job_id
        assert rc.sampling_job_id == fake.sampling_job_id
        assert rc.log_prob_job_id == fake.log_prob_job_id

    def test_get_server_state_is_none_on_cortex(self, legacy_cfg):
        """Cortex has no local Ray state actor; verl's reconnect_handle path
        still calls this and must not crash."""
        shim = _CortexClientShim(_FakeUnifiedClient(), legacy_cfg)
        assert shim.get_server_state() is None


class TestPropertySurface:
    def test_config_returns_legacy_config(self, legacy_cfg):
        """SkyRL reads `client.config.colocate` and verl reads
        `client.config` in reconnect_handle. Must be the legacy shape."""
        shim = _CortexClientShim(_FakeUnifiedClient(), legacy_cfg)
        assert shim.config is legacy_cfg
        assert isinstance(shim.config, ArcticRLClientConfig)

    def test_job_id_properties_pass_through(self, legacy_cfg):
        fake = _FakeUnifiedClient()
        shim = _CortexClientShim(fake, legacy_cfg)
        assert shim.training_job_id == "cortex-train-1"
        assert shim.sampling_job_id == "cortex-sample-1"
        assert shim.log_prob_job_id is None


# ---------------------------------------------------------------------------
# Factory-level dispatch: `create_arctic_rl_client(config)` picks Cortex
# ---------------------------------------------------------------------------


class TestFactoryDispatch:
    def test_backend_cortex_returns_shim(self, monkeypatch, legacy_cfg):
        """`create_arctic_rl_client(config)` must produce a `_CortexClientShim`
        when `config.backend == "cortex"`. Both integrations depend on this
        being the ONLY code path they trip when they set `backend: cortex`."""

        # Stub the unified client so we don't need SnowAPI creds. `_to_unified_config`
        # runs for real (it's just pydantic model construction, no I/O).
        class _StubUnified:
            def __init__(self, cfg, *_a, **_kw):
                self.cfg = cfg
                self.training_job_id = "t"
                self.sampling_job_id = "s"
                self.log_prob_job_id = "l"

            def shutdown(self):  # for teardown paths
                pass

        monkeypatch.setattr("arctic_platform.client.ArcticRLClient", _StubUnified)

        client = create_arctic_rl_client(legacy_cfg)
        assert isinstance(client, _CortexClientShim)
        assert client.training_job_id == "t"
        # Cortex sub-jobs threaded through config translation:
        assert client._client.cfg.backend == "cortex"
        assert client._client.cfg.model_name == "Qwen/Qwen3-0.6B"

    def test_cortex_path_never_imports_onprem_transports(self, monkeypatch, legacy_cfg):
        """The whole point of `create_arctic_rl_client`'s early cortex branch is
        that Cortex users don't drag the on-prem HTTP / Ray / vllm chain into
        their driver. Pin that invariant: after a cortex dispatch, the on-prem
        transport modules must NOT be in `sys.modules`.
        """
        import sys

        # Purge any prior loads so we can observe a clean walk of the cortex path.
        for mod in list(sys.modules):
            if mod in {
                "arctic_platform.rl.http_client",
                "arctic_platform.rl.http_server",
                "arctic_platform.rl.ray_client",
                "arctic_platform.rl.ray_server",
            }:
                monkeypatch.delitem(sys.modules, mod, raising=False)

        class _StubUnified:
            def __init__(self, *_a, **_kw):
                self.training_job_id = None
                self.sampling_job_id = None
                self.log_prob_job_id = None

            def shutdown(self):
                pass

        monkeypatch.setattr("arctic_platform.client.ArcticRLClient", _StubUnified)
        create_arctic_rl_client(legacy_cfg)

        # The on-prem transport modules must NOT have been touched.
        loaded = {m for m in sys.modules if m.startswith("arctic_platform.rl.")}
        onprem_hits = {
            m
            for m in loaded
            if m
            in {
                "arctic_platform.rl.http_client",
                "arctic_platform.rl.http_server",
                "arctic_platform.rl.ray_client",
                "arctic_platform.rl.ray_server",
            }
        }
        assert not onprem_hits, (
            f"Cortex path leaked on-prem transport imports: {sorted(onprem_hits)}. "
            "This means a Cortex-only driver would still need ray / vllm / "
            "arctic_inference installed — regressing the serverless UX."
        )

    def test_onprem_transports_are_lazy_at_module_level(self):
        """`arctic_platform/rl/client.py` must NOT eagerly import
        `ArcticRLHTTPClient` / `ArcticRLRayClient`. Otherwise merely resolving
        `arctic_platform.rl.create_arctic_rl_client` (via the package
        `__getattr__`) pulls the on-prem server chain in.
        """
        import arctic_platform.rl.client as client_module

        assert not hasattr(client_module, "ArcticRLHTTPClient"), (
            "`ArcticRLHTTPClient` is exposed at module scope; it must be "
            "imported inside `create_arctic_rl_client`'s on-prem branch."
        )
        assert not hasattr(client_module, "ArcticRLRayClient"), (
            "`ArcticRLRayClient` is exposed at module scope; it must be "
            "imported inside `create_arctic_rl_client`'s on-prem branch."
        )
