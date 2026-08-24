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
"""The RL frontends, blocking and async.

Pass ``server_state`` together with a reconnect config (job ids set) to reattach
to an already-running in-process Ray server from another process.

The only op RL adds over the shared bases is `log_probs`, which needs a log-prob
engine SFT never allocates.
"""

from __future__ import annotations

from arctic_platform.client.base import ArcticClient
from arctic_platform.client.base import AsyncArcticClient
from arctic_platform.client.requests import log_probs_request


class ArcticRLClient(ArcticClient):
    """The blocking RL client. Use `AsyncArcticRLClient` to await calls instead."""

    def log_probs(self, prompts: list, completions: list | None = None, top_k: int = 1) -> dict:
        return self._call(log_probs_request(self.jobs, prompts, completions, top_k))


class AsyncArcticRLClient(AsyncArcticClient):
    """The async RL client; what the verl adapter and skyrl recipes await."""

    async def log_probs(self, prompts: list, completions: list | None = None, top_k: int = 1) -> dict:
        return await self._acall(log_probs_request(self.jobs, prompts, completions, top_k))
