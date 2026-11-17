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
"""Put the Harbor and OpenHands recipes on ``sys.path`` for this directory's tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
for _rel in ("recipes/rl/harbor", "recipes/rl/openhands/code_localization"):
    _p = str(_ROOT / _rel)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)


@pytest.fixture(autouse=True)
def _no_real_sandbox_sweep(monkeypatch):
    """Keep sandboxes that fake creates register away from the exit-time sweep, which would call the real service."""
    try:
        from arctic_platform.integrations.harbor.snowflake_sandbox import environment as sf_env
    except ImportError:
        return
    monkeypatch.setattr(sf_env, "_LIVE_SANDBOXES", {})
    monkeypatch.setattr(sf_env, "_sweep_at_exit", True)
