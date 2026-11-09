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

"""The Cortex Training client the onboarding jobs are submitted through.

The suite drives the service end to end, so it needs the ``cortex_training``
SDK (``pip install cortex-training``). It is imported only when a client is
built, so the pure-logic modules and their tests run without it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Must equal cortex_training.client.DEBUG_OPTIONS_ENV: the client refuses a
# request with a `debug` block (the per-job image tag) unless it is set.
DEBUG_OPTIONS_ENV = "CORTEX_TRAINING_ENABLE_DEBUG_OPTIONS"


def enable_debug_options() -> None:
    os.environ[DEBUG_OPTIONS_ENV] = "1"


def load_connection(config_path: str) -> dict[str, Any]:
    """A connection file: ``host`` and ``pat``, plus optional database, schema,
    endpoint and polling settings, at the top level or under ``connection``."""
    parsed = json.loads(Path(config_path).expanduser().read_text(encoding="utf-8"))
    config = parsed.get("connection", parsed) if isinstance(parsed, dict) else None
    if not isinstance(config, dict):
        raise ValueError(f"connection config {config_path} must be a JSON object")
    return config


def make_client(config_path: str, **overrides: Any) -> Any:
    """A ``CortexTrainingClient`` authenticated with the file's PAT."""
    from cortex_training.client import CortexTrainingClient

    config = load_connection(config_path)
    host, pat = config.get("host"), config.get("pat")
    if host is None:
        raise ValueError("connection config needs `host` (Snowflake PAT auth)")
    if pat is None:
        raise ValueError("no PAT found: put `pat` in the connection config")
    kwargs: dict[str, Any] = dict(
        database=config.get("database", "CORTEX_TRAINING_DB"),
        schema=config.get("schema", "PUBLIC"),
        endpoint=config.get("endpoint", "cortex-training"),
        poll_interval=float(config.get("poll_interval", 0.5)),
        poll_timeout=float(config.get("poll_timeout", 1800.0)),
    )
    kwargs.update(overrides)
    return CortexTrainingClient.from_pat(host=host, pat=pat,
                                         verify_ssl=bool(config.get("verify_ssl", True)),
                                         **kwargs)
