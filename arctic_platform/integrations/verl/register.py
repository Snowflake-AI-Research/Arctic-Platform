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
sets::

    export VERL_USE_EXTERNAL_MODULES=arctic_platform.integrations.verl.register

Registration is a module-level side effect. Three things wire up under
the name ``"arctic"``:

1. Backend class -- :class:`ArcticRLClientWrapper` registers itself with
   :class:`verl.remote_backend.base.RemoteBackendRegistry` via
   ``@RemoteBackendRegistry.register("arctic")`` applied at
   ``adapter`` import time. That import is eager here because the
   decorator is the mechanism of registration.
2. ActorRollout forwarder worker -- registered lazily via
   :meth:`RemoteBackendRegistry.register_worker`. ``main_ppo`` reads
   the class back through
   :meth:`RemoteBackendRegistry.get_worker` when
   ``trainer.remote_backend=arctic``. Kept lazy so that jobs which
   set ``VERL_USE_EXTERNAL_MODULES`` but use a different backend (or
   just build config trees) don't pay the tensordict / DeepSpeed /
   flops-counter import cost of :mod:`worker`.
3. Rollout replica -- registered lazily with
   :class:`verl.workers.rollout.replica.RolloutReplicaRegistry` so
   ``actor_rollout_ref.rollout.name=arctic`` resolves to
   :class:`ArcticReplica` without eagerly importing vLLM.
"""

from __future__ import annotations

from verl.remote_backend.base import RemoteBackendRegistry
from verl.workers.rollout.replica import RolloutReplicaRegistry

# Importing the adapter is what actually registers the backend class:
# `@RemoteBackendRegistry.register("arctic")` runs at class-definition
# time. Everything below is lazy.
from arctic_platform.integrations.verl import adapter as _adapter  # noqa: F401


def _load_arctic_actor_rollout_worker() -> type:
    from arctic_platform.integrations.verl.worker import ArcticRLActorRolloutRefWorker

    return ArcticRLActorRolloutRefWorker


def _load_arctic_replica() -> type:
    from arctic_platform.integrations.verl.rollout import ArcticReplica

    return ArcticReplica


RemoteBackendRegistry.register_worker("arctic", _load_arctic_actor_rollout_worker)
RolloutReplicaRegistry.register("arctic", _load_arctic_replica)
