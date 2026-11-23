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
import copy
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.inference.server import scheduler as scheduler_module
from arctic_platform.inference.server.scheduler import Scheduler


class _Worker:
    """A worker handle whose ``generate`` records the prompt it got and samples two ids."""

    def __init__(self):
        self.prompts = []
        self.defer = False
        self.generate = SimpleNamespace(remote=self._generate)

    def _generate(self, prompt, sampling_params):
        self.prompts.append(torch.as_tensor(prompt).tolist())
        future = asyncio.get_running_loop().create_future()
        self.last_call = future
        if not self.defer:
            future.set_result(self.result())
        return SimpleNamespace(as_future=lambda: future)

    def result(self, reason="stop"):
        return dict(
            token_ids=[100 + len(self.prompts), 200 + len(self.prompts)],
            finish_reason=reason,
            prompt_len=len(self.prompts[-1]),
        )


def test_appended_turns_extend_the_kept_conversation():
    async def run():
        worker = _Worker()
        scheduler = Scheduler([worker], capacity=4)
        prompt = torch.tensor([1, 2, 3], dtype=torch.int32)
        turn = scheduler.submit(prompt, {}, routing_key="c1", conversation="start")
        prompt[0] = 99
        await turn
        await scheduler.submit([4], {}, routing_key="c1", conversation="append")
        await scheduler.submit(torch.tensor([5, 6]), {}, routing_key="c1", conversation="append")
        with pytest.raises(scheduler_module.ConversationNotFound):
            scheduler.submit([7], {}, routing_key="c2", conversation="append")
        return worker.prompts

    assert asyncio.run(run()) == [[1, 2, 3], [1, 2, 3, 101, 201, 4], [1, 2, 3, 101, 201, 4, 102, 202, 5, 6]]


def test_the_least_recently_used_conversation_is_evicted_first(monkeypatch):
    monkeypatch.setattr(scheduler_module, "_MAX_CONVERSATION_TOKENS", 12, raising=False)

    async def run():
        scheduler = Scheduler([_Worker()], capacity=4)
        await scheduler.submit([1, 2, 3], {}, routing_key="old", conversation="start")
        await scheduler.submit([4, 5, 6], {}, routing_key="new", conversation="start")
        await scheduler.submit([7], {}, routing_key="old", conversation="append")
        await scheduler.submit([8], {}, routing_key="old", conversation="append")
        with pytest.raises(scheduler_module.ConversationNotFound):
            scheduler.submit([9], {}, routing_key="new", conversation="append")

    asyncio.run(run())


@pytest.mark.parametrize(("prompt", "params", "key"), [
    ([1], {}, None), ("text", {}, "c"), ([1], {"n": 2}, "c"), ([1], {"enable_thinking": True}, "c"),
    (torch.tensor([[1]]), {}, "c"), (torch.tensor([1.5]), {}, "c"), (torch.tensor([2**32]), {}, "c"),
    ([1.5], {}, "c"), ([2**32], {}, "c"),
])
def test_rejects_invalid_conversation_without_dispatch(prompt, params, key):
    async def run():
        worker = _Worker()
        scheduler = Scheduler([worker], capacity=4)
        await scheduler.submit([1], {}, "valid", conversation="start")
        with pytest.raises((ValueError, TypeError)):
            scheduler.submit(prompt, params, routing_key=key, conversation="start")
        assert worker.prompts == [[1]]

    asyncio.run(run())


def test_one_inflight_turn_per_key_and_no_actor_local_state_in_copies():
    async def run():
        worker = _Worker()
        scheduler = Scheduler([worker], capacity=4)
        await scheduler.submit([1], {}, "c", conversation="start")
        worker.defer = True
        turn = scheduler.submit([2], {}, "c", conversation="append")
        with pytest.raises(ValueError):
            scheduler.submit([3], {}, "c", conversation="append")
        replica = copy.copy(scheduler)
        with pytest.raises(scheduler_module.ConversationNotFound):
            replica.submit([4], {}, "c", conversation="append")
        worker.last_call.set_result(worker.result())
        await turn

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["abort", "error", "worker_exception", "caller_cancel", "weight_cancel"])
def test_incomplete_turn_leaves_previous_turn_available_for_retry(outcome):
    async def run():
        worker = _Worker()
        scheduler = Scheduler([worker], capacity=4)
        await scheduler.submit([1], {}, "c", conversation="start")
        worker.defer = True
        turn = scheduler.submit([2], {}, "c", conversation="append")
        if outcome == "weight_cancel":
            assert scheduler.cancel_worker_inflight(0) == 1
            with pytest.raises(RuntimeError):
                await turn
        elif outcome == "worker_exception":
            worker.last_call.set_exception(RuntimeError("worker failed"))
            with pytest.raises(RuntimeError, match="worker failed"):
                await turn
        elif outcome == "caller_cancel":
            assert turn.cancel()
            worker.last_call.set_result(worker.result())
            await asyncio.sleep(0)
        else:
            worker.last_call.set_result(worker.result(outcome))
            assert (await turn)["finish_reason"] == outcome
        worker.defer = False
        await scheduler.submit([3], {}, "c", conversation="append")
        assert worker.prompts[-1] == [1, 101, 201, 3]

    asyncio.run(run())


def test_rejected_conversation_batch_dispatches_no_turns():
    async def run():
        worker = _Worker()
        scheduler = Scheduler([worker], capacity=4)
        with pytest.raises(ValueError):
            await scheduler.submit_batch([[1], [2]], {}, ["c", "c"], conversation="start")
        assert not worker.prompts
        await scheduler.submit([1], {}, "c", conversation="start")
        with pytest.raises(scheduler_module.ConversationNotFound):
            await scheduler.submit_batch([[2], [3]], {}, ["c", "missing"], conversation="append")
        assert worker.prompts == [[1]]
        await scheduler.submit([4], {}, "d", conversation="start")
        await scheduler.submit_batch([[], torch.tensor([5])], {}, ["c", "d"], conversation="append")
        assert worker.prompts[-2:] == [[1, 101, 201], [4, 102, 202, 5]]

    asyncio.run(run())


def test_failed_cache_write_reaches_caller_and_dispatches_queued_request(monkeypatch):
    async def run():
        worker = _Worker()
        scheduler = Scheduler([worker], capacity=1)
        await scheduler.submit([1], {}, "c", conversation="start")
        worker.defer = True
        turn = scheduler.submit([2], {}, "c", conversation="append")
        completion = worker.last_call
        busy = scheduler.submit([3], {})
        busy_completion = worker.last_call
        queued = scheduler.submit([4], {})
        assert worker.prompts[-1] == [3]

        def fail(*args):
            raise RuntimeError("cache write failed")

        with monkeypatch.context() as patch:
            patch.setattr(scheduler, "_keep_conversation", fail)
            worker.defer = False
            completion.set_result(dict(finish_reason="stop", token_ids=[7], prompt_len=4))
            with pytest.raises(RuntimeError, match="cache write failed"):
                await asyncio.wait_for(turn, timeout=1)
            await asyncio.wait_for(queued, timeout=1)
        busy_completion.set_result(dict(finish_reason="stop", token_ids=[8], prompt_len=1))
        await busy
        await scheduler.submit([5], {}, "c", conversation="append")
        assert worker.prompts[-1] == [1, 101, 201, 5]

    asyncio.run(run())


def test_generate_api_and_pool_carry_the_conversation_to_the_scheduler(monkeypatch):
    from fastapi import HTTPException

    from arctic_platform.inference.server import api
    from arctic_platform.inference.server.replica_pool import ReplicaPool

    async def run():
        worker = _Worker()
        pool = ReplicaPool.__new__(ReplicaPool)
        pool._scheduler = Scheduler([worker], capacity=4)
        pool._sleeping = False
        pool._model_id = None
        monkeypatch.setattr(api, "backend", pool)

        async def generate(**fields):
            return await api.generate_endpoint(api.GenerateRequest(**fields))

        await generate(prompts=[[1, 2], [5]], routing_key=["a", "b"], conversation="start")
        reply = await generate(prompts=[[3], [6]], routing_key=["a", "b"], conversation="append")
        assert len(reply["results"]) == 2
        assert worker.prompts[-2:] == [[1, 2, 101, 201, 3], [5, 102, 202, 6]]
        with pytest.raises(HTTPException) as missing:
            await generate(prompts=[[4]], routing_key="never-started", conversation="append")
        assert missing.value.status_code == 404

    asyncio.run(run())
