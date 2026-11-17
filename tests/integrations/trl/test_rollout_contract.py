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

"""Server-side loss accumulation, sampler prompt ids, and NaN rewards."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("trl.experimental.async_grpo.async_rollout_worker")

from arctic_platform.integrations.trl.rollout import ArcticRolloutWorker  # noqa: E402
from arctic_platform.integrations.trl.trainer import _require_server_loss_gas  # noqa: E402


def test_server_side_loss_rejects_gradient_accumulation():
    client = SimpleNamespace(server_side_loss=True)
    with pytest.raises(ValueError, match="gradient_accumulation_steps=1"):
        _require_server_loss_gas(client, SimpleNamespace(gradient_accumulation_steps=2))
    _require_server_loss_gas(client, SimpleNamespace(gradient_accumulation_steps=1))
    _require_server_loss_gas(SimpleNamespace(server_side_loss=False), SimpleNamespace(gradient_accumulation_steps=4))


class _Tok:
    def __call__(self, text, add_special_tokens=True):
        body = [ord(char) for char in text]
        ids = [1, *body] if add_special_tokens else body
        return {"input_ids": ids}


def _completion(prompt_ids=None):
    row = {
        "token_ids": [9],
        "logprobs": [{9: {"logprob": -0.2}}],
        "text": "x",
    }
    if prompt_ids is not None:
        row["prompt_token_ids"] = list(prompt_ids)
    return row


def _worker(**extra):
    worker = object.__new__(ArcticRolloutWorker)
    worker.num_generations = 1
    worker.model_version = 0
    worker._sampling_params = {}
    worker.tokenizer = _Tok()
    worker.chat_template_kwargs = {}
    worker.old_logprobs_source = "sampler"
    worker.reward_func_names = ["accuracy"]
    for key, value in extra.items():
        setattr(worker, key, value)
    return worker


def test_generate_gets_local_prompt_ids_not_retokenized_text():
    seen = {}

    class _Client:
        def generate(self, prompts, sampling_params):
            seen["prompts"] = prompts
            return [_completion(prompt) for prompt in prompts]

    worker = _worker(client=_Client())
    worker._enqueue_score = lambda job: None
    worker._rollout_groups([{"prompt": "ab"}], 0)
    assert seen["prompts"] == [[ord("a"), ord("b")]]
    assert 1 not in seen["prompts"][0]


def test_mismatched_sampler_prompt_ids_raise():
    worker = _worker()
    with pytest.raises(RuntimeError, match="sampler prompt ids"):
        worker._emit_group({"prompt": "a"}, [97], [_completion([1, 97])], 0, 0)


def test_nan_reward_is_not_enqueued():
    queued = []

    def _reward(**kwargs):
        return [float("nan"), 1.0]

    worker = _worker(num_generations=2, reward_funcs=[_reward], _put=queued.append)
    worker._emit_group({"prompt": "a"}, [97], [_completion(), _completion()], 0, 0)
    assert len(queued) == 1
    assert queued[0].metrics["reward"] == 1.0
