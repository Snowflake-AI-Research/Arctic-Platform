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

"""ArcticRLClient -- a unified frontend client for HTTP and Ray clients for RL training.

Works identically against a remote dss-platform deployment or a local
``server.py`` instance -- the only differences are ``base_url`` and whether the
client launches the server.

All jobs (training, sampling, log-prob) are initialized automatically at
construction time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arctic_platform.rl.config import ArcticRLClientConfig

if TYPE_CHECKING:
    # Only referenced as a type hint on the function signature; the concrete
    # class lives under `arctic_platform.rl.server` which pulls ray in. Keeping
    # the runtime import guarded so a Cortex-only caller (no ray installed)
    # can still `from arctic_platform.rl import create_arctic_rl_client`.
    from arctic_platform.rl.server import ArcticRLServerState

logger = logging.getLogger(__name__)


def create_arctic_rl_client(config: ArcticRLClientConfig, arctic_rl_server_state: "ArcticRLServerState | None" = None):
    # `cortex` short-circuits the on-prem HTTP / Ray transports: dispatch to
    # `arctic_platform.client` (unified client + Cortex transport) via a thin
    # async facade that preserves the legacy public surface. Both merged
    # upstream integrations (SkyRL#1837, verl#6422's Arctic adapter) call this
    # factory unchanged; flipping `config.backend = "cortex"` in yaml is the
    # only knob they touch.
    if config.backend == "cortex":
        # Lazy import: `arctic_platform.client` pulls the Cortex transport
        # dependency chain (requests + pydantic only; no ray/vllm).
        from arctic_platform.rl._cortex_dispatch import build_cortex_client

        return build_cortex_client(config)

    # On-prem transports are lazy-loaded to keep the Cortex-only import
    # path free of ray / vllm / arctic_inference / uvicorn. The eager
    # module-level imports these files used to have would drag the on-prem
    # HTTP server code into every Cortex driver just to construct a client.
    if config.comm_protocol == "http":
        from arctic_platform.rl.http_client import ArcticRLHTTPClient

        return ArcticRLHTTPClient(config)
    elif config.comm_protocol == "ray":
        from arctic_platform.rl.ray_client import ArcticRLRayClient

        # assert arctic_rl_server_state is not None, "arctic_rl_server_state is required for comm_protocol: ray"
        return ArcticRLRayClient(config, arctic_rl_server_state)
    else:
        raise ValueError(f"Invalid communication protocol: {config.comm_protocol}")
