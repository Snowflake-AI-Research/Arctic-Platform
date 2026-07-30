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
"""Cortex dispatch layer for the legacy `arctic_platform.rl` client entry point.

Both merged upstream integrations
(`NovaSky-AI/SkyRL#1837:integrations/arctic_rl/` and the Arctic-specific
`RemoteBackend` under `arctic_platform/integrations/verl/adapter.py` used by
`verl-project/verl#6422`) construct their client via a single call:

    from arctic_platform.rl import ArcticRLClientConfig, create_arctic_rl_client
    client = create_arctic_rl_client(config, server_state)

For Cortex support without touching either integration, `create_arctic_rl_client`
dispatches to this module when `config.backend == "cortex"`. This module:

- Translates the on-prem `ArcticRLClientConfig` into the unified
  `arctic_platform.client.ArcticRLClientConfig` (Cortex-relevant fields only;
  on-prem knobs like `ds_config` / `arctic_inference_config` are dropped
  because Cortex has no local placement to configure).
- Wraps the unified (synchronous) `ArcticRLClient` in `_CortexClientShim`,
  which re-exposes the exact async surface the legacy
  `ArcticRLHTTPClient` / `ArcticRLRayClient` had. Every `async def foo(...)`
  method internally calls the corresponding sync unified method — legal
  Python (`async def` that never awaits is a valid coroutine returning the
  computed value) and semantically identical from the caller's POV because
  the underlying operation is I/O over Cortex, which the unified client
  already handles.

Design rules:

- Method surface is a strict superset of what the two integrations reach
  for (verified against `integrations/arctic_rl/trainer.py` +
  `arctic_platform/integrations/verl/adapter.py`). Any callable the
  integrations hit is here.
- Properties (`config`, `training_job_id`, `sampling_job_id`,
  `log_prob_job_id`) match the legacy client's public names verbatim so
  the integrations' attribute reads work unmodified.
- The compat layer in `arctic_platform.client` (this PR's earlier commits)
  already shapes Cortex responses into the on-prem envelope
  (`metrics.grad_norm` at the top level, `model_outputs` -> `batch` alias,
  `logprobs` naming preserved). This shim doesn't re-shape — it just
  forwards.
"""

from __future__ import annotations

import logging
from typing import Any

from arctic_platform.rl.config import ArcticRLClientConfig as _LegacyConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config translation
# ---------------------------------------------------------------------------


def _to_unified_config(legacy: _LegacyConfig):
    """Translate an on-prem `arctic_platform.rl.ArcticRLClientConfig` into a
    unified `arctic_platform.client.ArcticRLClientConfig` sized for Cortex.

    Only fields Cortex actually consumes are threaded through; on-prem-only
    fields (ds_config, log_prob_ds_config, ds_worker_config,
    arctic_inference_config, colocate, ray_auto_attach, startup_timeout, ...)
    are intentionally dropped because Cortex has no local placement / no
    DeepSpeed worker to configure. `extra="ignore"` on the unified config
    would silently drop them if we forwarded them, but being explicit here
    keeps the wire minimal and makes divergences visible.
    """
    from arctic_platform.client.config import ArcticRLClientConfig as _UnifiedConfig

    kwargs: dict[str, Any] = {
        "backend": "cortex",
        "model_name": legacy.model_name,
        "training_gpus": legacy.training_gpus,
        "sampling_gpus": legacy.sampling_gpus,
        "log_prob_gpus": legacy.log_prob_gpus,
        "seed": legacy.seed,
        # Cortex reconnect: forward pre-existing job ids so the unified client
        # skips job init and attaches instead. Consumed by JobHandles.from_config.
        "training_job_id": legacy.training_job_id,
        "sampling_job_id": legacy.sampling_job_id,
        "log_prob_job_id": legacy.log_prob_job_id,
    }
    for legacy_attr, unified_attr in (
        ("cortex_host", "cortex_host"),
        ("cortex_database", "cortex_database"),
        ("cortex_schema", "cortex_schema"),
        ("cortex_endpoint", "cortex_endpoint"),
        ("cortex_pat_env_var", "cortex_pat_env_var"),
        ("cortex_base_url", "cortex_base_url"),
        ("max_seq_len", "max_seq_len"),
    ):
        val = getattr(legacy, legacy_attr, None)
        if val is not None:
            kwargs[unified_attr] = val

    return _UnifiedConfig(**kwargs)


# ---------------------------------------------------------------------------
# Async facade over the (synchronous) unified client
# ---------------------------------------------------------------------------


class _CortexClientShim:
    """Async facade over `arctic_platform.client.ArcticRLClient` that matches
    the public surface of the legacy `ArcticRLHTTPClient` / `ArcticRLRayClient`.

    The unified client is synchronous by design (the underlying Cortex RPC is
    a single request/response). The legacy client is async because the
    on-prem HTTP path uses `httpx.AsyncClient`. Both merged integrations wire
    the client through both idioms — SkyRL's `_run(coro)` wrapper in the sync
    training loop and direct `await client.foo(...)` in the async paths.
    Exposing `async def` methods that internally call sync unified methods
    makes both idioms work identically.
    """

    def __init__(self, unified_client, legacy_config: _LegacyConfig) -> None:
        self._client = unified_client
        # Both integrations read `client.config.colocate` (SkyRL) and
        # `client.config` for reconnect flows (verl); expose the legacy
        # config, not the translated unified one, so those reads see what
        # the caller passed in.
        self._legacy_config = legacy_config

    # -- Properties preserved verbatim from the legacy client ------------------

    @property
    def config(self) -> _LegacyConfig:
        return self._legacy_config

    @property
    def training_job_id(self):
        return self._client.training_job_id

    @property
    def sampling_job_id(self):
        return self._client.sampling_job_id

    @property
    def log_prob_job_id(self):
        return self._client.log_prob_job_id

    # -- Sync methods (already sync on the legacy client) ---------------------

    def reconnect_config(self) -> _LegacyConfig:
        """Rebuild an on-prem-shaped config that reconnects to the same Cortex
        sub-jobs. Called by SkyRL's driver -> Ray-worker handoff in
        `main_arctic_rl.py` and by verl's `reconnect_handle()`.
        """
        # We can't just call `self._client.reconnect_config()` because that
        # returns a unified config; the caller expects a legacy config. Round-
        # trip via the legacy config with cortex job ids populated.
        return self._legacy_config.model_copy(
            update={
                "training_job_id": self._client.training_job_id,
                "sampling_job_id": self._client.sampling_job_id,
                "log_prob_job_id": self._client.log_prob_job_id,
            }
        )

    def get_server_state(self):
        """Cortex has no local Ray state actor; verl's `reconnect_handle`
        pattern still calls this. Returning None is safe — the reconnect
        path re-attaches via `training_job_id` on the config, not via a
        server-state handle.
        """
        return None

    def shutdown(self) -> None:
        self._client.shutdown()

    # -- Async methods (legacy client is async; shim keeps that signature) ---

    async def fwd_bwd(self, batch: dict, **legacy_kwargs: Any) -> dict:
        return self._client.fwd_bwd(batch, **legacy_kwargs)

    async def fwd_no_grad(self, batch: dict, **legacy_kwargs: Any) -> dict:
        return self._client.fwd_no_grad(batch, **legacy_kwargs)

    async def step(self, learning_rate: float | None = None) -> dict:
        return self._client.step(learning_rate=learning_rate)

    async def save_checkpoint(self, stage_info: dict | None = None, path: str | None = None) -> dict:
        return self._client.save_checkpoint(stage_info=stage_info, path=path)

    async def save_weights(self, path: str) -> dict:
        return self._client.save_weights(path=path)

    async def generate(self, prompts, sampling_params=None, **kwargs) -> list:
        return self._client.generate(prompts=prompts, sampling_params=sampling_params, **kwargs)

    async def sync_weights(self, cuda_ipc: bool = False, low_memory: bool = False) -> dict:
        return self._client.sync_weights(cuda_ipc=cuda_ipc, low_memory=low_memory)

    async def reset_prefix_cache(self, drain: bool = True, timeout_s: float = 60.0) -> dict:
        return self._client.reset_prefix_cache(drain=drain, timeout_s=timeout_s)

    async def wake_inference(self, **kwargs: Any) -> dict:
        return self._client.wake_inference(**kwargs)

    async def sleep_inference(self, **kwargs: Any) -> dict:
        return self._client.sleep_inference(**kwargs)

    async def wake_training(self, **kwargs: Any) -> dict:
        return self._client.wake_training(**kwargs)

    async def sleep_training(self, **kwargs: Any) -> dict:
        return self._client.sleep_training(**kwargs)

    async def wake_log_prob(self, **kwargs: Any) -> dict:
        return self._client.wake_log_prob(**kwargs)

    async def sleep_log_prob(self, **kwargs: Any) -> dict:
        return self._client.sleep_log_prob(**kwargs)

    async def empty_training_cache(self, **kwargs: Any) -> dict:
        return self._client.empty_training_cache(**kwargs)

    async def weight_norm(self, **kwargs: Any) -> dict:
        return self._client.weight_norm(**kwargs)

    async def log_probs(self, batch: dict, **kwargs: Any) -> dict:
        # Unified client exposes log-prob as a fwd_no_grad variant + the
        # compat layer registers a `log-probs` op on CortexTransport; forward
        # via the public `log_probs` accessor when present, else fall back to
        # `fwd_no_grad` with a marker kwarg.
        fn = getattr(self._client, "log_probs", None)
        if callable(fn):
            return fn(batch, **kwargs)
        return self._client.fwd_no_grad(batch, log_probs_only=True, **kwargs)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def build_cortex_client(legacy_config: _LegacyConfig) -> _CortexClientShim:
    """Called from `arctic_platform.rl.client.create_arctic_rl_client` when
    `config.backend == "cortex"`. Kept as a top-level factory so the shim
    can be tested in isolation.
    """
    from arctic_platform.client import ArcticRLClient

    unified_config = _to_unified_config(legacy_config)
    logger.info(
        "arctic_platform.rl -> cortex dispatch: model=%s training_gpus=%d sampling_gpus=%d log_prob_gpus=%d",
        unified_config.model_name,
        unified_config.training_gpus,
        unified_config.sampling_gpus,
        unified_config.log_prob_gpus,
    )
    unified_client = ArcticRLClient(unified_config)
    return _CortexClientShim(unified_client, legacy_config)
