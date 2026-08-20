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
"""The RL frontends, sync and async.

`ArcticRLClient` is the async one and `SyncArcticRLClient` the blocking one --
that naming is load-bearing for existing callers (the verl adapter and the skyrl
recipes await `ArcticRLClient`). The only op RL adds over the shared base is
`log_probs`, which needs a log-prob engine SFT never allocates.
"""

from __future__ import annotations

from typing import Any
from typing import Literal
from typing import overload

from arctic_platform.client.base import AsyncArcticClient
from arctic_platform.client.base import SyncArcticClient
from arctic_platform.client.config import ArcticClientConfig
from arctic_platform.client.requests import log_probs_request


class SyncArcticRLClient(SyncArcticClient):
    """RL frontend. Use `ArcticRLClient` for the async twin."""

    def log_probs(self, prompts: list, completions: list | None = None, top_k: int = 1) -> dict:
        return self._call(log_probs_request(self.jobs, prompts, completions, top_k))


class ArcticRLClient(AsyncArcticClient):
    """The async RL client. Use `SyncArcticRLClient` for blocking calls."""

    async def log_probs(self, prompts: list, completions: list | None = None, top_k: int = 1) -> dict:
        return await self._acall(log_probs_request(self.jobs, prompts, completions, top_k))


@overload
def create_arctic_rl_client(  # noqa: E704
    config: ArcticClientConfig, *, blocking_calls: Literal[False] = ..., server_state: Any = ...
) -> ArcticRLClient: ...


@overload
def create_arctic_rl_client(  # noqa: E704
    config: ArcticClientConfig, *, blocking_calls: Literal[True], server_state: Any = ...
) -> SyncArcticRLClient: ...


def create_arctic_rl_client(
    config: ArcticClientConfig, *, blocking_calls: bool = False, server_state: Any = None
) -> ArcticRLClient | SyncArcticRLClient:
    """Async client by default; pass blocking_calls=True for the sync client.

    Pass ``server_state`` together with a reconnect config (job ids set) to
    reattach to an already-running in-process Ray server from another process.
    """
    cls = SyncArcticRLClient if blocking_calls else ArcticRLClient
    return cls(config, server_state=server_state)
