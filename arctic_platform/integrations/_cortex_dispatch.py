# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cortex dispatch shim for the legacy ``arctic_platform.rl`` API.

``arctic_platform.rl.create_arctic_rl_client`` dispatches here when
``backend='cortex'`` is set on the legacy config.

Translates ``arctic_platform.rl.ArcticRLClientConfig`` (what SkyRL builds) into
the unified ``arctic_platform.client.ArcticRLClientConfig`` with a
``CortexConfig`` backend, then wraps ``ArcticRLClient`` and rewrites the two
calls whose shapes diverge on Cortex:

* ``fwd_bwd``: reshape ``{batch, meta, processing}`` -> Cortex's
  ``{args, kwargs, context, processing}``.
* ``fwd_no_grad``: return ``[B, T]`` zeros; Cortex has no ``/forward``.

Cortex deployment settings live on ``CortexConfig``; env-var hydration is
``CortexConfig.from_env()``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload
from arctic_platform.rl.config import ArcticRLClientConfig as _LegacyConfig

__all__ = ["create_cortex_client"]


def _to_unified_config(legacy: _LegacyConfig):
    """Legacy ``arctic_platform.rl`` config -> nested unified client config.

    Optimizer + gradient-clipping are merged into ``training.ds_config`` so
    Cortex's ``to_cortex()`` sub-job builder can lift them; the legacy
    ``training_config`` dict is otherwise opaque to Cortex.
    """
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

    Everything unified-client-compatible is delegated via ``__getattr__``. Only
    the two shape mismatches (``fwd_bwd`` payload reshape, ``fwd_no_grad`` zeros
    stub) and the legacy-config-shaped ``reconnect_config()`` live here.
    """

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
        # SkyRL passes this back into `create_arctic_rl_client` in a forwarder
        # process; the shape must be the legacy config (not the unified one).
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
        # SkyRL calls this sync; verl awaits it. Inside a running loop return
        # the coroutine; otherwise drive it to completion in a fresh loop so
        # sync callers don't leak jobs.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._client.shutdown())
            return None
        return self._client.shutdown()

    # Everything else on ArcticRLClient (generate, step, save_checkpoint,
    # sync_weights, wake_inference, sleep_inference, reset_prefix_cache, ...)
    # is exposed as-is. Cortex-specific behaviour lives in CortexTransport
    # (wake/sleep -> no-op) and the unified client (colocate defaults False).
    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    async def fwd_bwd(self, batch: dict, **legacy_kwargs: Any) -> dict:
        processing = legacy_kwargs.pop("processing", None)
        legacy_kwargs.pop("router_replay", None)
        return await self._client.fwd_bwd(
            to_cortex_fwd_bwd_payload(batch, processing=processing)
        )

    async def fwd_no_grad(self, batch: dict, **_: Any) -> dict:
        # Cortex has no /forward op. Return zero log-probs so SkyRL's
        # approx_kl / clip_ratio display metrics still compute; the server-side
        # GRPO loss restores π_old = π_new via ``old_log_probs = logprobs.detach()``.
        # Only correct for single-epoch on-policy GRPO without KL — see
        # docs/cortex-integration.md.
        import torch

        b_data = batch.get("batch") if isinstance(batch, dict) else None
        if not isinstance(b_data, dict):
            b_data = batch if isinstance(batch, dict) else {}
        ids = b_data.get("input_ids")
        b, t = (int(ids.shape[0]), int(ids.shape[-1])) if torch.is_tensor(ids) else (1, 1)
        z = torch.zeros((b, max(t, 1)), dtype=torch.float32)
        return {"batch": {"logprobs": z, "log_probs": z, "entropy": z, "entropies": z}}

    async def save_weights(self, path: str) -> dict:
        # SkyRL's disk-based inference-side weight reload. Cortex sub-jobs
        # don't share local disk; silent no-op would leave sampling on stale
        # weights forever.
        raise NotImplementedError(
            "Cortex has no disk-based weight reload; use sync_weights() (NCCL) "
            f"or save_checkpoint(). (save_weights(path={path!r}) was called.)"
        )


def create_cortex_client(config: _LegacyConfig) -> _CortexClientShim:
    return _CortexClientShim(config)
