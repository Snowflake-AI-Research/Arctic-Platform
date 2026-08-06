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
import os

from arctic_platform.rl.config import ArcticRLClientConfig
from arctic_platform.rl.server import ArcticRLServerState

logger = logging.getLogger(__name__)


def create_arctic_rl_client(config: ArcticRLClientConfig, arctic_rl_server_state: ArcticRLServerState = None):
    # ``ARCTIC_BACKEND=cortex`` overrides ``config.backend`` so SkyRL adapters
    # that hard-code ``backend="local"`` can still route to Cortex without a
    # SkyRL-side patch. HTTP/Ray backends are imported lazily so a CPU-only
    # Cortex driver doesn't pay the ``arctic_inference.server`` import cost.
    if config.backend == "cortex" or os.environ.get("ARCTIC_BACKEND", "").lower() == "cortex":
        from arctic_platform.rl._cortex_dispatch import create_cortex_client

        if config.backend != "cortex":
            config = config.model_copy(update={"backend": "cortex"})
        return create_cortex_client(config)

    if config.comm_protocol == "http":
        from arctic_platform.rl.http_client import ArcticRLHTTPClient

        return ArcticRLHTTPClient(config)
    elif config.comm_protocol == "ray":
        from arctic_platform.rl.ray_client import ArcticRLRayClient

        return ArcticRLRayClient(config, arctic_rl_server_state)
    else:
        raise ValueError(f"Invalid communication protocol: {config.comm_protocol}")
