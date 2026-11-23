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

"""A conversation's later turns sent as only their new token ids: the scheduler keeps each completed turn."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.inference.server import scheduler as scheduler_module
from arctic_platform.inference.server.scheduler import ConversationNotFound, Scheduler


class _Worker:
    """A worker handle whose ``generate`` records the prompt it got and samples two ids."""

    def __init__(self):
        self.prompts = []
        self.generate = SimpleNamespace(remote=self._generate)

    def _generate(self, prompt, sampling_params):
        self.prompts.append(torch.as_tensor(prompt).tolist())
        future = asyncio.get_running_loop().create_future()
        future.set_result({"token_ids": [100 + len(self.prompts), 200 + len(self.prompts)]})
        return SimpleNamespace(as_future=lambda: future)


def test_appended_turns_extend_the_kept_conversation():
    async def run():
        worker = _Worker()
        scheduler = Scheduler([worker], capacity=4)
        await scheduler.submit([1, 2, 3], {}, routing_key="c1", conversation="start")
        await scheduler.submit([4], {}, routing_key="c1", conversation="append")
        await scheduler.submit(torch.tensor([5, 6]), {}, routing_key="c1", conversation="append")
        with pytest.raises(ConversationNotFound):
            scheduler.submit([7], {}, routing_key="c2", conversation="append")
        return worker.prompts

    assert asyncio.run(run()) == [[1, 2, 3], [1, 2, 3, 101, 201, 4], [1, 2, 3, 101, 201, 4, 102, 202, 5, 6]]


def test_the_least_recently_used_conversation_is_evicted_first(monkeypatch):
    monkeypatch.setattr(scheduler_module, "_MAX_CONVERSATION_TOKENS", 12)

    async def run():
        scheduler = Scheduler([_Worker()], capacity=4)
        await scheduler.submit([1, 2, 3], {}, routing_key="old", conversation="start")
        await scheduler.submit([4, 5, 6], {}, routing_key="new", conversation="start")
        await scheduler.submit([7], {}, routing_key="new", conversation="append")
        await scheduler.submit([8], {}, routing_key="new", conversation="append")
        with pytest.raises(ConversationNotFound):
            scheduler.submit([9], {}, routing_key="old", conversation="append")

    asyncio.run(run())


def test_a_conversation_needs_a_routing_key_token_ids_and_one_sample():
    async def run():
        scheduler = Scheduler([_Worker()], capacity=4)
        for prompt, params, key in (([1], {}, None), ("text", {}, "c"), ([1], {"n": 2}, "c")):
            with pytest.raises(ValueError):
                scheduler.submit(prompt, params, routing_key=key, conversation="start")

    asyncio.run(run())
