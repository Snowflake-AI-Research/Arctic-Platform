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

"""Fixtures for ``tests/integrations/verl/*``.

Tests here exercise the verl adapter without requiring verl itself to be
installed in the CI image. Fixtures inject minimal stub modules into
``sys.modules`` that mirror only the surface
``arctic_platform.integrations.verl.register`` (and the adapter/rollout
it imports) actually touches.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from typing import Callable

import pytest

# --------------------------------------------------------------------- #
# verl.remote_backend.base.RemoteBackendRegistry stub                    #
# --------------------------------------------------------------------- #
# Mirrors the canonical decorator-based signature from
# Snowflake-AI-Research/verl@arctic_rl_share_v0.7.1 with the
# `register_worker` extension that the paired verl-core PR
# (verl-project/verl#6422) lands: two independent dicts (`_backends`,
# `_workers`), same class object stored under each key, keyed by backend
# name.


class _BackendRegistry:
    """Stand-in for verl's ``RemoteBackendRegistry``.

    * ``register(name)`` is a decorator factory (stores the decorated
      class under ``name``).
    * ``register_worker(name, cls)`` stores the ActorRollout forwarder
      worker class under ``name``.
    * ``get(name)`` / ``get_worker(name)`` return the stored classes.
    """

    def __init__(self) -> None:
        self._backends: dict[str, type] = {}
        self._workers: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type], type]:
        def _decorator(cls: type) -> type:
            existing = self._backends.get(name)
            if existing is not None and existing is not cls:
                raise ValueError(f"Backend name '{name}' already registered to {existing!r}.")
            self._backends[name] = cls
            return cls

        return _decorator

    def register_worker(self, name: str, worker_cls: type) -> None:
        existing = self._workers.get(name)
        if existing is not None and existing is not worker_cls:
            raise ValueError(f"Backend '{name}' worker already registered to {existing!r}.")
        self._workers[name] = worker_cls

    def get(self, name: str) -> type:
        if name not in self._backends:
            raise KeyError(f"Unknown backend '{name}'. Registered: {sorted(self._backends)}.")
        return self._backends[name]

    def get_worker(self, name: str) -> type | None:
        return self._workers.get(name)

    def list(self) -> list[str]:
        return sorted(self._backends)


class _LazyRolloutRegistry:
    """Stand-in for ``RolloutReplicaRegistry``.

    Keeps the lazy-loader signature (``register(name, loader)``) that
    matches the canonical verl-core signature -- ``rollout.name=arctic``
    resolution doesn't materialise the vLLM-bearing class until the
    trainer actually asks for it.
    """

    def __init__(self) -> None:
        self._loaders: dict[str, Callable[[], Any]] = {}
        self._resolved: dict[str, Any] = {}

    def register(self, name: str, loader: Callable[[], Any]) -> None:
        existing = self._loaders.get(name)
        if existing is not None and existing is not loader:
            raise ValueError(f"Rollout name '{name}' already registered to a different loader.")
        self._loaders[name] = loader

    def get(self, name: str) -> Any:
        if name not in self._resolved:
            if name not in self._loaders:
                raise KeyError(f"Unknown rollout '{name}'. Registered: {sorted(self._loaders)}.")
            self._resolved[name] = self._loaders[name]()
        return self._resolved[name]

    def list(self) -> list[str]:
        return sorted(self._loaders)

    def _has_loader(self, name: str) -> bool:
        return name in self._loaders

    def _is_resolved(self, name: str) -> bool:
        return name in self._resolved


class _RemoteBackendBase:
    """Minimal RemoteBackend base class for tests.

    Mirrors just the surface ``ArcticRLClientWrapper`` inherits from so
    ``adapter.py`` can be imported without pulling in real verl.
    """


def _install_verl_stub() -> tuple[_BackendRegistry, _LazyRolloutRegistry]:
    """Install a minimal ``verl.*`` stub into ``sys.modules``.

    Returns the two registry instances so tests can assert on their
    state after the plugin's ``register.py`` module is imported.
    """
    backend_registry = _BackendRegistry()
    rollout_registry = _LazyRolloutRegistry()

    verl = types.ModuleType("verl")
    verl.__path__ = []  # type: ignore[attr-defined]

    verl_remote_backend = types.ModuleType("verl.remote_backend")
    verl_remote_backend.__path__ = []  # type: ignore[attr-defined]

    verl_remote_backend_base = types.ModuleType("verl.remote_backend.base")
    verl_remote_backend_base.RemoteBackendRegistry = backend_registry  # type: ignore[attr-defined]
    verl_remote_backend_base.RemoteBackend = _RemoteBackendBase  # type: ignore[attr-defined]

    verl_workers = types.ModuleType("verl.workers")
    verl_workers.__path__ = []  # type: ignore[attr-defined]

    verl_workers_rollout = types.ModuleType("verl.workers.rollout")
    verl_workers_rollout.__path__ = []  # type: ignore[attr-defined]

    verl_workers_rollout_replica = types.ModuleType("verl.workers.rollout.replica")
    verl_workers_rollout_replica.RolloutReplicaRegistry = rollout_registry  # type: ignore[attr-defined]

    # `adapter.py` uses `verl.utils.tensordict_utils.get_non_tensor_data`
    # to pull `max_prompt_len` / `max_response_len` out of the batch. For
    # tests we just want a plain dict getter with defaults.
    verl_utils = types.ModuleType("verl.utils")
    verl_utils.__path__ = []  # type: ignore[attr-defined]

    verl_utils_tensordict = types.ModuleType("verl.utils.tensordict_utils")

    def _get_non_tensor_data(*, data: dict, key: str, default: Any) -> Any:
        return data.get(key, default) if hasattr(data, "get") else default

    verl_utils_tensordict.get_non_tensor_data = _get_non_tensor_data  # type: ignore[attr-defined]

    for mod in (
        verl,
        verl_remote_backend,
        verl_remote_backend_base,
        verl_workers,
        verl_workers_rollout,
        verl_workers_rollout_replica,
        verl_utils,
        verl_utils_tensordict,
    ):
        sys.modules[mod.__name__] = mod

    return backend_registry, rollout_registry


def _install_arctic_rl_stub() -> None:
    """Install a minimal ``arctic_platform.rl`` stub.

    ``adapter.py`` reads ``ArcticRLClientConfig`` / ``create_arctic_rl_client``
    at module scope; those live behind ``arctic_platform.rl.__init__``, which
    pulls in the ray/vllm/arctic_inference stack. Tests only need the *names*
    to be importable -- the actual RL client is exercised in Layer 4 (GPU).
    """
    arctic_rl = types.ModuleType("arctic_platform.rl")
    arctic_rl.ArcticRLClientConfig = object
    arctic_rl.create_arctic_rl_client = lambda *a, **kw: None
    sys.modules["arctic_platform.rl"] = arctic_rl

    arctic_rl_ray_server = types.ModuleType("arctic_platform.rl.ray_server")
    arctic_rl_ray_server.ArcticRLRayServerState = object
    sys.modules["arctic_platform.rl.ray_server"] = arctic_rl_ray_server


def _uninstall_stubs() -> None:
    for name in list(sys.modules):
        if name == "verl" or name.startswith("verl."):
            del sys.modules[name]
    for name in ("arctic_platform.rl", "arctic_platform.rl.ray_server"):
        # Only drop if the stub is in place, to avoid clobbering a real install.
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None) is None:
            del sys.modules[name]


def _drop_arctic_verl_modules() -> None:
    """Reset the ``arctic_platform.integrations.verl`` submodules so a
    re-import re-runs the module-level ``register()`` calls against a
    fresh stub.

    We also strip the corresponding attributes from the parent package
    -- Python's ``from pkg import sub`` prefers a cached ``pkg.sub``
    attribute over re-importing the submodule, so leaving those in
    place would let the *old* adapter module (bound to the previous
    test's stub registry) silently satisfy the import.
    """
    subnames = ("register", "adapter", "worker", "rollout")
    for name in subnames:
        sys.modules.pop(f"arctic_platform.integrations.verl.{name}", None)
    pkg = sys.modules.get("arctic_platform.integrations.verl")
    if pkg is not None:
        for name in subnames:
            if hasattr(pkg, name):
                delattr(pkg, name)


@pytest.fixture
def verl_stub() -> tuple[_BackendRegistry, _LazyRolloutRegistry]:
    """Provide a fresh verl + arctic_platform.rl stub for each test that opts in."""
    backend, rollout = _install_verl_stub()
    _install_arctic_rl_stub()
    _drop_arctic_verl_modules()
    try:
        yield backend, rollout
    finally:
        _uninstall_stubs()
        _drop_arctic_verl_modules()
