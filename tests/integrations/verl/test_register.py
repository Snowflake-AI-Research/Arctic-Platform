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

"""VERL_USE_EXTERNAL_MODULES hook: end-to-end wiring checks.

verl's plugin contract is that importing the module named by
``VERL_USE_EXTERNAL_MODULES`` has the side effect of registering the
integration against verl's registries. Here we assert exactly that: a
fresh import of ``arctic_platform.integrations.verl.register`` (against
a minimal verl stub) registers ``"arctic"`` on both registries, and
does *not* eagerly import the worker / rollout modules or their heavy
transitive deps (tensordict, vLLM, ...).
"""

from __future__ import annotations

import importlib
import inspect
import sys

import pytest

# The adapter module (imported for its @register decorator side effect)
# pulls in omegaconf + transformers at module top level.
pytest.importorskip("omegaconf")
pytest.importorskip("transformers")


def test_register_populates_all_three_slots(verl_stub) -> None:
    backend_registry, rollout_registry = verl_stub

    importlib.import_module("arctic_platform.integrations.verl.register")

    assert "arctic" in backend_registry.list(), (
        "importing arctic_platform.integrations.verl.register must register "
        "'arctic' on verl.remote_backend.RemoteBackendRegistry via the "
        "adapter's @register decorator."
    )
    assert backend_registry.get_worker("arctic") is None or callable(backend_registry.get_worker("arctic")), (
        "worker slot for 'arctic' must be present (lazy loader is "
        "resolved by get_worker); pre-resolution it may return None or "
        "the resolved class -- register_worker stores the loader either "
        "way."
    )
    assert rollout_registry._has_loader("arctic"), (
        "importing arctic_platform.integrations.verl.register must register "
        "'arctic' on verl.workers.rollout.replica.RolloutReplicaRegistry."
    )


def test_register_defers_worker_and_rollout_imports(verl_stub) -> None:
    """The plugin bootstrap must not pull in worker.py / rollout.py.

    Users running with a different backend still get this plugin
    imported (verl reads VERL_USE_EXTERNAL_MODULES unconditionally);
    doing the heavy imports here would tax every unrelated job. The
    lazy worker + replica loaders defer that cost until ``get_worker``
    / ``get`` is actually called by the trainer.
    """
    _, rollout_registry = verl_stub

    for mod in (
        "arctic_platform.integrations.verl.worker",
        "arctic_platform.integrations.verl.rollout",
    ):
        sys.modules.pop(mod, None)

    importlib.import_module("arctic_platform.integrations.verl.register")

    assert not rollout_registry._is_resolved("arctic")
    assert "arctic_platform.integrations.verl.worker" not in sys.modules
    assert "arctic_platform.integrations.verl.rollout" not in sys.modules


def test_registered_loaders_resolve_expected_class_paths(verl_stub) -> None:
    """The loaders under ``arctic`` must target the current worker/rollout classes.

    We don't invoke the loaders here because that pulls in the full
    tensordict / DeepSpeed / vLLM stack -- that path is covered by the
    GPU smoke test. Static-checking the loader source is enough to
    catch a copy-paste regression (e.g. someone points the ``arctic``
    worker loader at a stale ``arctic_verl`` symbol).
    """
    register = importlib.import_module("arctic_platform.integrations.verl.register")

    worker_loader = register._load_arctic_actor_rollout_worker
    replica_loader = register._load_arctic_replica

    worker_src = inspect.getsource(worker_loader)
    replica_src = inspect.getsource(replica_loader)

    assert "arctic_platform.integrations.verl.worker" in worker_src
    assert "ArcticRLActorRolloutRefWorker" in worker_src
    assert "arctic_platform.integrations.verl.rollout" in replica_src
    assert "ArcticReplica" in replica_src


def test_backend_class_resolves_via_decorator(verl_stub) -> None:
    """The adapter registers itself with the registry via the
    ``@RemoteBackendRegistry.register("arctic")`` decorator applied at
    class-definition time -- verify the class object comes back."""
    importlib.import_module("arctic_platform.integrations.verl.register")

    backend_registry, _ = verl_stub

    from arctic_platform.integrations.verl.adapter import ArcticRLClientWrapper

    assert backend_registry.get("arctic") is ArcticRLClientWrapper
