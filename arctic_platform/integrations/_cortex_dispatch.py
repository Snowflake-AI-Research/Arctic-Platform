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
"""Cortex routing for the legacy ``arctic_platform.rl`` client API.

SkyRL's integration is written against the legacy flat ``ArcticRLClientConfig``
and the ``create_arctic_rl_client`` factory, while Cortex is reachable only
through the unified client. This translates the config and hands back an
``AsyncArcticRLClient`` wearing the legacy accessor names.

Deliberately thin. The forward-backward reshape, the ``/forward`` zero-fill and
the response-metric mirroring all live in the Cortex transport, and the unified
client is natively awaitable, so what is left here is config translation plus
the few legacy-shaped accessors SkyRL actually reads.
"""

from __future__ import annotations

import asyncio
from typing import Any

from arctic_platform.rl.config import ArcticRLClientConfig as LegacyConfig

__all__ = ["create_cortex_client"]


def _to_unified_config(legacy: LegacyConfig):
    """Legacy flat config -> unified nested config with a Cortex backend.

    The optimizer block is folded into ``ds_config`` because that is where
    Cortex's sub-job builder looks for it.
    """
    from arctic_platform.client.config import ArcticClientConfig
    from arctic_platform.client.config import CortexConfig
    from arctic_platform.client.config import SamplingConfig
    from arctic_platform.client.config import TrainingConfig

    ds_config = dict(legacy.ds_config or {})
    training_config = dict(legacy.training_config or {})
    if "optimizer" in training_config and "optimizer" not in ds_config:
        ds_config["optimizer"] = training_config["optimizer"]

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
    # Present when a worker is reattaching to jobs the driver already created.
    for job_field in ("training_job_id", "sampling_job_id", "log_prob_job_id"):
        job_id = getattr(legacy, job_field, None)
        if job_id is not None:
            fields[job_field] = job_id
    return ArcticClientConfig(**fields)


class _LegacyCortexClient:
    """An ``AsyncArcticRLClient`` answering to the legacy client's accessors.

    Everything not overridden here is delegated straight through, so the op
    surface SkyRL awaits is the unified client's own.
    """

    def __init__(self, legacy_config: LegacyConfig) -> None:
        from arctic_platform.client import AsyncArcticRLClient

        self._legacy_config = legacy_config
        self._client = AsyncArcticRLClient(_to_unified_config(legacy_config))

    @property
    def config(self) -> LegacyConfig:
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

    def reconnect_config(self) -> LegacyConfig:
        """Job handles in the *legacy* shape, since that is what workers rebuild from."""
        return self._legacy_config.model_copy(
            update={
                "training_job_id": self._client.jobs.training,
                "sampling_job_id": self._client.jobs.sampling,
                "log_prob_job_id": self._client.jobs.log_prob,
            }
        )

    def get_server_state(self) -> None:
        # Only the in-process Ray transport has server state to hand across; the
        # unified client raises for the rest, but SkyRL expects a value it can
        # pass along, so say "nothing to share" instead.
        return None

    def shutdown(self):
        """Usable from both worlds: SkyRL tears down synchronously, verl awaits.

        Returning the coroutine inside a running loop keeps `await` working;
        outside one, drive it to completion so the Cortex jobs are not leaked.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._client.shutdown())
        return self._client.shutdown()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def create_cortex_client(config: LegacyConfig) -> _LegacyCortexClient:
    return _LegacyCortexClient(config)
