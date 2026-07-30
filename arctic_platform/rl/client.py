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
from typing import TYPE_CHECKING

from arctic_platform.rl.config import ArcticRLClientConfig

if TYPE_CHECKING:
    # Only referenced as a type hint on the function signature; the concrete
    # class lives under `arctic_platform.rl.server` which pulls ray in. Keeping
    # the runtime import guarded so a Cortex-only caller (no ray installed)
    # can still `from arctic_platform.rl import create_arctic_rl_client`.
    from arctic_platform.rl.server import ArcticRLServerState

logger = logging.getLogger(__name__)


# Env vars that override the incoming legacy config into a Cortex-backed one.
# The two merged upstream integrations (NovaSky-AI/SkyRL#1837 and verl-project/
# verl#6422's Arctic adapter at arctic_platform/integrations/verl/adapter.py)
# both hardcode `backend="local"` when they construct ArcticRLClientConfig.
# Editing either adapter would require a new upstream SkyRL PR + our own
# Arctic-Platform-side patch; that's an integration-side change we explicitly
# want to avoid. Instead the launcher exports these env vars and this factory
# rewrites the (hardcoded-local) config into a cortex one before dispatch.
_CORTEX_ENV_TOGGLE = "ARCTIC_RL_BACKEND"
_CORTEX_ENV_MAP: dict[str, str] = {
    "CORTEX_BASE_URL": "cortex_base_url",
    "CORTEX_HOST": "cortex_host",
    "CORTEX_DATABASE": "cortex_database",
    "CORTEX_SCHEMA": "cortex_schema",
    "CORTEX_ENDPOINT": "cortex_endpoint",
    "CORTEX_PAT_ENV_VAR": "cortex_pat_env_var",
    "CORTEX_MAX_SEQ_LEN": "max_seq_len",
}


def _maybe_override_from_env(config: ArcticRLClientConfig) -> ArcticRLClientConfig:
    """Apply Cortex overrides driven by launcher environment variables.

    When ``ARCTIC_RL_BACKEND=cortex`` is set, rewrite ``config.backend`` to
    ``"cortex"`` and populate ``cortex_*`` / ``max_seq_len`` from the env
    (``CORTEX_BASE_URL``, ``CORTEX_HOST``, ``CORTEX_DATABASE``, ``CORTEX_SCHEMA``,
    ``CORTEX_ENDPOINT``, ``CORTEX_PAT_ENV_VAR``, ``CORTEX_MAX_SEQ_LEN``).
    Explicit fields already on ``config`` win over env vars — the env is a
    fallback for launchers whose adapter code doesn't yet thread cortex knobs.

    When ``ARCTIC_RL_BACKEND`` is unset or has any other value, return ``config``
    untouched. When it's set to ``"cortex"`` on a config that already has
    ``backend="cortex"``, still merge the env fields in (idempotent).
    """
    requested = os.environ.get(_CORTEX_ENV_TOGGLE, "").strip().lower()
    if requested != "cortex":
        return config

    overrides: dict = {}
    if config.backend != "cortex":
        overrides["backend"] = "cortex"
    for env_key, field_name in _CORTEX_ENV_MAP.items():
        env_val = os.environ.get(env_key)
        if not env_val:
            continue
        # Preserve explicit config values — env vars only fill in gaps.
        if getattr(config, field_name, None) is not None:
            continue
        if field_name == "max_seq_len":
            try:
                overrides[field_name] = int(env_val)
            except ValueError:
                logger.warning("Ignoring non-integer %s=%r for max_seq_len", env_key, env_val)
            continue
        overrides[field_name] = env_val

    if not overrides:
        return config

    logger.info(
        "arctic_platform.rl: ARCTIC_RL_BACKEND=cortex active; overriding %s",
        sorted(overrides.keys()),
    )
    return config.model_copy(update=overrides)


def create_arctic_rl_client(config: ArcticRLClientConfig, arctic_rl_server_state: "ArcticRLServerState | None" = None):
    # Env-var override lets launchers force Cortex without touching either
    # integration's adapter code (both currently hardcode `backend="local"`).
    # See `_maybe_override_from_env` for the recognized env vars.
    config = _maybe_override_from_env(config)

    # `cortex` short-circuits the on-prem HTTP / Ray transports: dispatch to
    # `arctic_platform.client` (unified client + Cortex transport) via a thin
    # async facade that preserves the legacy public surface. Both merged
    # upstream integrations (SkyRL#1837, verl#6422's Arctic adapter) call this
    # factory unchanged; the env override (or `config.backend = "cortex"` set
    # directly) is the only knob they touch.
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
