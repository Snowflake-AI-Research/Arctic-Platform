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
"""SkyRL entrypoint that runs training and sampling on Cortex.

Selected from the recipe with::

    trainer.override_entrypoint=arctic_platform.integrations.skyrl.entrypoint

SkyRL's ``main_base`` imports whatever that names and calls its ``main()``, so
this module is the seam. It exists because SkyRL's own Arctic entrypoint is
pinned to the legacy ``arctic_platform.rl`` factory -- it does
``from arctic_platform.rl import create_arctic_rl_client`` at import time and
calls it to build the client. Teaching that factory about Cortex would have
meant extending the client generation we are trying to retire, so instead this
swaps the name in upstream's module for one that builds on
``arctic_platform.client``, and leaves the legacy package untouched.

Choosing the entrypoint *is* choosing Cortex, so there is no environment
variable deciding it: the recipe names this module or it does not.

:func:`main` runs SkyRL's own experiment. A recipe with a different one calls
:func:`launch_driver` and passes the function that builds it.
"""

from __future__ import annotations

import contextlib
import signal
import threading
from typing import Any
from typing import Iterator
from typing import Optional

import ray

__all__ = ["launch_driver", "main", "patch_upstream"]

_RECONNECT_FIELDS = ("training_job_id", "sampling_job_id", "log_prob_job_id")
# Clients in this process that created their jobs, as opposed to reattaching.
_created_clients: list[Any] = []


def _create_cortex_client(config: Any, server_state: Any = None, *, wait: bool = True) -> Any:
    """Stand-in for upstream's ``create_arctic_rl_client``.

    ``server_state`` exists only to match upstream's signature: Cortex holds the
    GPUs in its own sub-jobs, so there is no in-process Ray server to describe.
    """
    from arctic_platform.integrations._cortex_dispatch import create_cortex_client

    client = create_cortex_client(config, wait=wait)
    if all(getattr(config, field, None) is None for field in _RECONNECT_FIELDS):
        _created_clients.append(client)
    return client


def _exit_on_sigterm(signum: int, frame: Any) -> None:
    raise SystemExit(128 + signum)


@contextlib.contextmanager
def _releasing_created_jobs() -> Iterator[None]:
    """Cancel the jobs this driver created when the block exits, however it exits.

    SkyRL never calls ``shutdown``. Python's default SIGTERM action skips
    ``finally``, and Ray only installs a handler that raises ``SystemExit`` from
    ``ray.init`` on, which is after the placement wait. A process killed outright
    is left to the server's idle timeout.
    """
    from loguru import logger

    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        yield
    finally:
        while _created_clients:
            client = _created_clients.pop()
            try:
                client.shutdown()
            except Exception:
                logger.exception("Could not cancel a Cortex job on exit")
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


def _patch_upstream() -> Any:
    """Point upstream's client factory at Cortex in the calling process.

    Returns upstream's module. Rebinding on the module rather than passing a
    factory in, because upstream resolves the global at each call site and
    offers no injection point.
    """
    # Import by the name upstream is reached under: the recipe puts $SKYRL_HOME
    # on PYTHONPATH, and its entrypoint imports its own siblings relatively.
    from integrations.arctic_rl import entrypoint as upstream

    # Safe against upstream drift only while these names exist. Fail here rather
    # than later with a client that is silently the wrong one.
    for name in ("create_arctic_rl_client", "skyrl_entrypoint", "ArcticRLExp"):
        if not hasattr(upstream, name):
            raise RuntimeError(
                f"integrations.arctic_rl.entrypoint no longer exposes {name!r}; "
                "the Cortex entrypoint rebinds it to route the client through "
                "arctic_platform.client. Re-point it at whatever upstream now uses."
            )

    from arctic_platform.integrations.skyrl import install_cortex_driver_shims

    install_cortex_driver_shims()
    upstream.create_arctic_rl_client = _create_cortex_client
    return upstream


def patch_upstream() -> Any:
    """Rebind SkyRL's client factory to the Cortex client in this process."""
    return _patch_upstream()


def _with_placement_timeout(cfg: Any, rl_config: Any) -> Any:
    from arctic_platform.integrations._cortex_dispatch import placement_timeout

    arctic = getattr(cfg.trainer, "arctic_rl", None)
    startup = getattr(arctic, "startup_timeout", None) if arctic is not None else None
    timeout = placement_timeout(rl_config.job_ready_timeout, startup)
    if timeout == rl_config.job_ready_timeout:
        return rl_config
    return rl_config.model_copy(update=dict(job_ready_timeout=timeout))


def launch_driver(
    cfg: Any,
    remote_fn: Any,
    *,
    num_cpus: float = 0,
    env_prefixes: tuple[str, ...] = ("ARCTIC_", "WANDB_", "GIT_"),
    env_keys: tuple[str, ...] = ("PATH", "HF_HOME", "HF_HUB_CACHE"),
) -> None:
    """Provision Cortex, then run ``remote_fn(cfg, reconnect_config)`` under Ray.

    ``remote_fn`` has to call :func:`patch_upstream` itself. It runs in a fresh
    worker, and the driver's rebind does not reach that process. Use
    ``num_cpus=0`` when ``remote_fn`` submits more Ray tasks: a reserved CPU on
    the coordinator deadlocks them. The job is cancelled when this returns or
    raises, including on Ctrl-C and SIGTERM.
    """
    import os
    from pathlib import Path

    from integrations.arctic_rl.config import build_rl_config
    from loguru import logger
    from skyrl.train.utils.utils import prepare_runtime_environment

    upstream = patch_upstream()
    rl_config = _with_placement_timeout(cfg, build_rl_config(cfg))
    env_vars = prepare_runtime_environment(cfg)
    for key, value in os.environ.items():
        if key.startswith(env_prefixes):
            env_vars[key] = value
    for key in env_keys:
        value = os.environ.get(key)
        if value is not None and value != "":
            env_vars[key] = value
    skyrl_root = str(Path(upstream.__file__).resolve().parents[2])
    python_path = [skyrl_root, os.environ.get("PYTHONPATH", "")]
    env_vars["PYTHONPATH"] = os.pathsep.join(part for part in python_path if part != "")
    runtime_env = dict(env_vars=env_vars)
    remote = ray.remote(num_cpus=num_cpus)(remote_fn)

    with _releasing_created_jobs():
        logger.info("Provisioning Cortex training and sampling jobs")
        pre_client = _create_cortex_client(rl_config, wait=False)
        pre_client.wait_until_ready()
        logger.info(f"Cortex jobs ready: training={pre_client.training_job_id} sampling={pre_client.sampling_job_id}")
        ray.init(num_gpus=0, runtime_env=runtime_env, ignore_reinit_error=True)
        ray.get(remote.options(runtime_env=runtime_env).remote(cfg, pre_client.reconnect_config()))


@ray.remote(num_cpus=1)
def _cortex_skyrl_entrypoint(
    cfg: Any,
    reconnect_config: Optional[Any] = None,
    server_state: Optional[Any] = None,
) -> None:
    """Upstream's ``skyrl_entrypoint``, with the factory patched worker-side.

    The training loop runs in a Ray task, not in the driver, and that worker is
    a fresh process that imports upstream's module for itself -- so the driver's
    rebind does not reach it. Without this it would build the legacy Ray client
    and die on a server actor that Cortex never creates.
    """
    upstream = _patch_upstream()
    exp = upstream.ArcticRLExp(cfg, reconnect_config=reconnect_config, server_state=server_state)
    exp.run()


def main() -> None:
    upstream = _patch_upstream()
    # `main()` looks this up as a module global when it submits the task.
    upstream.skyrl_entrypoint = _cortex_skyrl_entrypoint
    with _releasing_created_jobs():
        upstream.main()


if __name__ == "__main__":
    main()
