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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The Ray state actor owns inference pools that cannot cross process boundaries."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from arctic_platform.common.ray_server import ArcticRLRayServer
from arctic_platform.common.ray_server import _prompt_token_logprobs


class _Remote:
    def __init__(self, value=None, *, fail=False, awaitable=False):
        self.value = value
        self.fail = fail
        self.awaitable = awaitable
        self.calls = []

    def remote(self, *args):
        self.calls.append(args)
        if self.fail:
            raise AssertionError("the non-picklable pool must stay in the state actor")
        if not self.awaitable:
            return self.value

        async def result():
            return self.value

        return result()


class _StateActor:
    def __init__(self):
        self.get_jobs = _Remote({1: {"job_type": "log_prob"}})
        self.get_training_workers = _Remote([])
        self.get_log_prob_workers = _Remote([])
        self.get_log_prob_tokenizer = _Remote(None)
        self.get_next_job_id = _Remote(2)
        self.get_weight_sync_ready = _Remote(False)
        self.get_weight_sync_bucket_size = _Remote(1)
        self.get_sampling_pool = _Remote(None)
        self.get_log_prob_pool = _Remote(fail=True)
        self.get_colocate = _Remote(False)
        self.log_probs = _Remote({"job_id": 1, "results": [0.25]}, awaitable=True)


def test_log_prob_pool_stays_in_state_actor():
    state = _StateActor()
    with patch("arctic_platform.common.ray_server.ray.get", side_effect=lambda value: value):
        server = ArcticRLRayServer(state)

    result = asyncio.run(server.log_probs(1, {"prompts": ["prompt"]}))

    assert result == {"job_id": 1, "results": [0.25]}
    assert state.get_log_prob_pool.calls == []
    assert state.log_probs.calls == [(1, {"prompts": ["prompt"]})]


def test_prompt_logprobs_are_aligned_to_the_scored_prompt_tokens():
    result = {
        "token_ids": [99],
        "prompt_logprobs": [
            None,
            {12: {"logprob": -1.25, "rank": 1}, 91: {"logprob": -3.0, "rank": 2}},
            {"13": {"logprob": -0.75, "rank": 1}},
        ],
    }

    converted = _prompt_token_logprobs([11, 12, 13], result)

    assert converted == {"token_ids": [12, 13], "logprobs": [-1.25, -0.75], "seq_len": 2}


def test_state_factory_waits_for_actor_readiness():
    from arctic_platform.common.ray_server import create_arctic_rl_ray_server_state

    ready = _Remote("ready-ref")

    class StateHandle:
        __ray_ready__ = ready

    state = StateHandle()

    class ActorBuilder:
        def __call__(self, actor_class):
            return self

        def remote(self, **kwargs):
            return state

    with (
        patch("arctic_platform.common.ray_server.placement_group", return_value=object()),
        patch("arctic_platform.common.ray_server.ray.remote", return_value=ActorBuilder()),
        patch("arctic_platform.common.ray_server.ray.get", return_value=None) as get,
    ):
        result = create_arctic_rl_ray_server_state(training_gpus=8)

    assert result is state
    assert ready.calls == [()]
    get.assert_called_once_with("ready-ref")
