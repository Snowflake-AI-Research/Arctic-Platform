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
"""

from __future__ import annotations

from typing import Any
from typing import Optional

import ray

__all__ = ["main"]


def _create_cortex_client(config: Any, server_state: Any = None) -> Any:
    """Stand-in for upstream's ``create_arctic_rl_client``.

    ``server_state`` exists only to match upstream's signature: Cortex holds the
    GPUs in its own sub-jobs, so there is no in-process Ray server to describe.
    """
    from arctic_platform.integrations._cortex_dispatch import create_cortex_client

    return create_cortex_client(config)


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
    upstream.main()


if __name__ == "__main__":
    main()
