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
"""A recording fake for `CortexSession`, shared by the jobs / logs / CLI tests.

Everything under here is offline: nothing opens a socket, so the assertions are
about the URLs and bodies we would have sent.
"""

from __future__ import annotations

from typing import Any

import pytest

from arctic_platform.client.config import CortexConfig
from arctic_platform.client.cortex.jobs import CortexJobs

PREFIX = "https://acct.example.com/api/v2/databases/DB/schemas/PUBLIC/cortex-training"


class FakeSession:
    """Records every send and replies from a queued script."""

    def __init__(self, replies: list[Any] | None = None) -> None:
        self.base_url = "https://acct.example.com"
        self.prefix = PREFIX
        self.calls: list[dict] = []
        self.replies = list(replies or [])

    def send(self, method: str, url: str, *, retry_on=None, **kwargs: Any) -> dict:
        self.calls.append({"method": method, "url": url, "retry_on": retry_on, **kwargs})
        if not self.replies:
            return {}
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    @property
    def last(self) -> dict:
        return self.calls[-1]


@pytest.fixture
def prefix() -> str:
    """The endpoint URL every control-plane call hangs off."""
    return PREFIX


@pytest.fixture
def config() -> CortexConfig:
    return CortexConfig(host="acct.example.com", pat="tok", database="DB", schema="PUBLIC")


@pytest.fixture
def jobs(config):
    """`CortexJobs` with its HTTP layer swapped for the recorder."""

    def build(replies: list[Any] | None = None) -> CortexJobs:
        control = CortexJobs(config, poll_interval=0.0, poll_timeout=5.0)
        control.session = FakeSession(replies)  # type: ignore[assignment]
        return control

    return build
