# Copyright 2026 Snowflake Inc.
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
"""V1 rollout replica for the Arctic backend.

Thin subclass of :class:`ArcticReplica` that adapts the constructor to
verl V1's ``LLMServerManager.replica_init_kwargs`` forwarding and swaps
in :class:`ArcticV1LLMServer` (which tags
``TokenOutput.extra_fields["global_steps"]``).
"""

from __future__ import annotations

import asyncio

import ray
from omegaconf import DictConfig
from verl.workers.config import RolloutConfig

from arctic_platform.integrations.verl.rollout import ArcticReplica
from arctic_platform.integrations.verl.v1.server import ArcticV1LLMServer


class ArcticV1Replica(ArcticReplica):
    """V1-compatible ArcticReplica. Uses :class:`ArcticV1LLMServer` and
    exposes :meth:`set_global_steps` for the ``remote_backend``
    checkpoint-engine short-circuit fan-out.
    """

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: DictConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
        *,
        main_config: DictConfig,
        backend_handle: dict,
        **_ignored_kwargs,
    ) -> None:
        super().__init__(
            replica_rank=replica_rank,
            config=config,
            model_config=model_config,
            gpus_per_node=gpus_per_node,
            is_reward_model=is_reward_model,
            main_config=main_config,
            backend_handle=backend_handle,
        )
        self.server_class = ray.remote(ArcticV1LLMServer)
        # V1 additions; downstream code may inspect them.
        self.is_teacher_model = is_teacher_model
        self.name_suffix = f"_{name_suffix}" if name_suffix else ""

    async def set_global_steps(self, global_steps: int) -> None:
        """Publish the current policy version to every server in this replica.

        Called from ``CheckpointEngineManager`` after
        ``actor_wg.update_weights`` on the ``remote_backend`` short-circuit.
        """
        if not self.servers:
            return
        await asyncio.gather(*(s.set_global_steps.remote(global_steps) for s in self.servers))
