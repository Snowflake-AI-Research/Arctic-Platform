# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cortex dispatch for the legacy ``arctic_platform.rl`` API.

Translates SkyRL's legacy config into the unified client config with a
``CortexConfig`` backend, wraps ``ArcticRLClient``, and rewrites the two
calls whose shapes diverge on Cortex: ``fwd_bwd`` (payload reshape) and
``fwd_no_grad`` (zeros stub — Cortex has no ``/forward``).
"""

from __future__ import annotations

import asyncio
from typing import Any

from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload
from arctic_platform.rl.config import ArcticRLClientConfig as _LegacyConfig

__all__ = ["create_cortex_client"]


def _to_unified_config(legacy: _LegacyConfig):
    """Legacy config -> unified client config. Optimizer is merged into
    ``training.ds_config`` so Cortex's sub-job builder can lift it."""
    from arctic_platform.client.config import ArcticRLClientConfig as U
    from arctic_platform.client.config import CortexConfig
    from arctic_platform.client.config import SamplingConfig
    from arctic_platform.client.config import TrainingConfig

    ds_config = dict(legacy.ds_config or {})
    tc_in = dict(legacy.training_config or {})
    if "optimizer" in tc_in and "optimizer" not in ds_config:
        ds_config["optimizer"] = tc_in["optimizer"]

    fields: dict[str, Any] = {
        "model_name": legacy.model_name,
        "seed": legacy.seed,
        "training_gpus": legacy.training_gpus,
        "sampling_gpus": legacy.sampling_gpus,
        "log_prob_gpus": legacy.log_prob_gpus,
        "job_ready_timeout": legacy.job_ready_timeout,
        "backend": CortexConfig.from_env(),
        "training": TrainingConfig(
            ds_config=ds_config or None,
            ds_worker_config=legacy.ds_worker_config or None,
            checkpoint_path=legacy.checkpoint_path,
            full_determinism=legacy.full_determinism,
        ),
        "sampling": SamplingConfig(
            vllm=dict(legacy.vllm_config or {}),
            arctic_inference_config=legacy.arctic_inference_config or None,
        ),
    }
    for k in ("training_job_id", "sampling_job_id", "log_prob_job_id"):
        v = getattr(legacy, k, None)
        if v is not None:
            fields[k] = v
    return U(**fields)


class _CortexClientShim:
    """Async facade over ``ArcticRLClient`` for the legacy SkyRL adapter.
    Unified-client calls are delegated via ``__getattr__``; only the two
    shape mismatches and ``reconnect_config()`` live here."""

    def __init__(self, legacy_config: _LegacyConfig) -> None:
        from arctic_platform.client import ArcticRLClient

        self._legacy_config = legacy_config
        self._unified_config = _to_unified_config(legacy_config)
        self._client = ArcticRLClient(self._unified_config)

    @property
    def config(self) -> _LegacyConfig:
        return self._legacy_config

    @property
    def training_job_id(self) -> Any:
        return self._client.jobs.training

    @property
    def sampling_job_id(self) -> Any:
        return self._client.jobs.sampling

    @property
    def log_prob_job_id(self) -> Any:
        return self._client.jobs.log_prob

    def reconnect_config(self) -> _LegacyConfig:
        # Forwarder workers reconnect via the legacy-shaped config, not the
        # unified one.
        return self._legacy_config.model_copy(
            update={
                "training_job_id": self._client.jobs.training,
                "sampling_job_id": self._client.jobs.sampling,
                "log_prob_job_id": self._client.jobs.log_prob,
            }
        )

    def get_server_state(self) -> Any:
        return None  # Cortex has no in-process state

    def shutdown(self):
        # SkyRL calls sync, verl awaits: return the coroutine inside a running
        # loop, else drive it to completion so sync callers don't leak jobs.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._client.shutdown())
            return None
        return self._client.shutdown()

    def __getattr__(self, name: str) -> Any:
        # Everything else on ArcticRLClient is exposed as-is.
        return getattr(self._client, name)

    async def fwd_bwd(self, batch: dict, **legacy_kwargs: Any) -> dict:
        processing = legacy_kwargs.pop("processing", None)
        legacy_kwargs.pop("router_replay", None)
        return await self._client.fwd_bwd(
            to_cortex_fwd_bwd_payload(batch, processing=processing)
        )

    async def fwd_no_grad(self, batch: dict, **_: Any) -> dict:
        # Cortex has no /forward op; return zeros. Only correct for single-epoch
        # on-policy GRPO without KL (see docs/cortex-integration.md).
        import torch

        b_data = batch.get("batch") if isinstance(batch, dict) else None
        if not isinstance(b_data, dict):
            b_data = batch if isinstance(batch, dict) else {}
        ids = b_data.get("input_ids")
        b, t = (int(ids.shape[0]), int(ids.shape[-1])) if torch.is_tensor(ids) else (1, 1)
        z = torch.zeros((b, max(t, 1)), dtype=torch.float32)
        return {"batch": {"logprobs": z, "log_probs": z, "entropy": z, "entropies": z}}

    async def save_weights(self, path: str) -> dict:
        # Cortex sub-jobs don't share disk; a silent no-op would leave
        # sampling on stale weights.
        raise NotImplementedError(
            f"Cortex has no disk-based weight reload; use sync_weights() (NCCL) "
            f"or save_checkpoint() instead. Called with path={path!r}."
        )


def create_cortex_client(config: _LegacyConfig) -> _CortexClientShim:
    return _CortexClientShim(config)
