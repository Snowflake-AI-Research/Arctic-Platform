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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cortex dispatch for the legacy ``arctic_platform.rl`` API.

Translates SkyRL's legacy config into the unified client config with a
``CortexConfig`` backend, wraps ``AsyncArcticRLClient``, and rewrites the two
calls whose shapes diverge on Cortex: ``fwd_bwd`` (payload reshape) and
``fwd_no_grad`` (zeros stub — Cortex has no ``/forward``).
"""

from __future__ import annotations

import asyncio
from typing import Any

from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload
from arctic_platform.integrations._cortex_shared import zero_logprobs_like
from arctic_platform.rl.config import ArcticRLClientConfig as _LegacyConfig

__all__ = ["create_cortex_client"]


def _to_unified_config(legacy: _LegacyConfig):
    """Legacy config -> unified client config. Optimizer is merged into
    ``training.ds_config`` so Cortex's sub-job builder can lift it."""
    from arctic_platform.client.config import ArcticClientConfig as U
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
    """Async facade over ``AsyncArcticRLClient`` for the legacy SkyRL adapter.
    Unified-client calls are delegated via ``__getattr__``; only the two
    shape mismatches and ``reconnect_config()`` live here."""

    def __init__(self, legacy_config: _LegacyConfig) -> None:
        # Must be the async client: this shim's callers await every op.
        from arctic_platform.client import AsyncArcticRLClient

        self._legacy_config = legacy_config
        self._unified_config = _to_unified_config(legacy_config)
        self._client = AsyncArcticRLClient(self._unified_config)

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
        # SkyRL calls this synchronously: inside a running loop hand back the
        # coroutine, otherwise drive it so a sync caller doesn't leak the job.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._client.shutdown())
            return None
        return self._client.shutdown()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    async def fwd_bwd(self, batch: dict, **legacy_kwargs: Any) -> dict:
        processing = legacy_kwargs.pop("processing", None)
        legacy_kwargs.pop("router_replay", None)
        return await self._client.fwd_bwd(to_cortex_fwd_bwd_payload(batch, processing=processing))

    async def fwd_no_grad(self, batch: dict, **kwargs: Any) -> dict:
        # Cortex has no /forward; see zero_logprobs_like for why zeros are sound
        # as the policy snapshot and never as pi_ref.
        if kwargs.get("reference_model"):
            raise NotImplementedError(
                "cortex backend cannot serve reference-model log-probs: it has no "
                "/forward sub-job, and substituting zeros would silently corrupt "
                "the KL term. Disable KL (e.g. SkyRL "
                "trainer.algorithm.use_kl_loss=false, use_kl_in_reward=false).\n\n"
                "See docs/cortex-integration.md#supported-recipes."
            )
        # Both spellings of each key: SkyRL reads them inconsistently by call site.
        z = zero_logprobs_like(batch)
        return {"batch": {"logprobs": z, "log_probs": z, "entropy": z, "entropies": z}}

    async def save_weights(self, path: str) -> dict:
        # Cortex sub-jobs don't share disk; a silent no-op would leave
        # sampling on stale weights.
        raise NotImplementedError(
            "Cortex has no disk-based weight reload; use sync_weights() (NCCL) "
            f"or save_checkpoint() instead. Called with path={path!r}."
        )


def create_cortex_client(config: _LegacyConfig) -> _CortexClientShim:
    return _CortexClientShim(config)
