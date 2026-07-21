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

"""Plugin entry point for the Arctic RL <-> verl integration.

Loaded by verl during ``verl/__init__.py`` bootstrap whenever the user
sets ``VERL_USE_EXTERNAL_MODULES=arctic_platform.integrations.verl.register``.

Registers the ``"arctic"`` backend + forwarder worker + rollout replica.
Auto-selects V0 (Snowflake-AI-Research/verl fork) or V1
(verl-project/verl main) based on whether
:mod:`verl.trainer.ppo.v1.trainer_remote_backend` imports.
"""

from __future__ import annotations


def _register_v0() -> None:
    """V0 (Snowflake-AI-Research/verl fork) registration path."""
    from verl.remote_backend.base import RemoteBackendRegistry
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    # Importing the adapter is what actually registers the backend class:
    # `@RemoteBackendRegistry.register("arctic")` runs at class-definition time.
    from arctic_platform.integrations.verl import adapter as _adapter  # noqa: F401

    def _load_v0_worker() -> type:
        from arctic_platform.integrations.verl.worker import ArcticRLActorRolloutRefWorker

        return ArcticRLActorRolloutRefWorker

    def _load_v0_replica() -> type:
        from arctic_platform.integrations.verl.rollout import ArcticReplica

        return ArcticReplica

    RemoteBackendRegistry.register_worker("arctic", _load_v0_worker)
    RolloutReplicaRegistry.register("arctic", _load_v0_replica)


def _register_v1() -> None:
    """V1 (verl-project/verl main) registration path.

    Points worker + replica loaders at the V1 wrappers so the base-class
    contract matches :class:`ActorRolloutRefWorker` and the V1
    :class:`LLMServerManager`.
    """
    from verl.remote_backend.base import RemoteBackendRegistry
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    # Same adapter (backend business logic is V0/V1-agnostic); the decorator
    # side effect populates RemoteBackendRegistry["arctic"].
    from arctic_platform.integrations.verl import adapter as _adapter  # noqa: F401

    def _load_v1_worker() -> type:
        from arctic_platform.integrations.verl.v1.worker import ArcticV1ActorRolloutRefWorker

        return ArcticV1ActorRolloutRefWorker

    def _load_v1_replica() -> type:
        from arctic_platform.integrations.verl.v1.replica import ArcticV1Replica

        return ArcticV1Replica

    RemoteBackendRegistry.register_worker("arctic", _load_v1_worker)
    RolloutReplicaRegistry.register("arctic", _load_v1_replica)


def _detect_and_register() -> None:
    """Pick V0 or V1 based on what the installed verl exports. V1 wins
    when both are importable (V1 companion PR ships
    :mod:`verl.trainer.ppo.v1.trainer_remote_backend`).
    """
    try:
        import verl.trainer.ppo.v1.trainer_remote_backend  # noqa: F401
    except ImportError:
        _register_v0()
    else:
        _register_v1()


_detect_and_register()
