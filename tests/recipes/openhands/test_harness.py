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
"""The OpenHands example closes its conversation and grades afterward."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from openhands_localization.conversation import run_and_close
from openhands_localization.rollout import finalize_rollout
from openhands_localization.rollout import grade_rollout


class _Conversation:
    def __init__(self, fail: str | None = None) -> None:
        self.closed = False
        self.fail = fail
        self.state = SimpleNamespace(events=[SimpleNamespace(model_dump=lambda: dict(kind="ok"))])

    def send_message(self, text: str) -> None:
        self.sent = text
        if self.fail == "send":
            raise RuntimeError("send failed")

    def run(self) -> None:
        if self.fail == "run":
            raise RuntimeError("run failed")

    def close(self) -> None:
        self.closed = True


class _Tokenizer:
    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        del ids, skip_special_tokens
        return "notes without a tool call"


def test_conversation_closes_after_a_successful_run():
    conversation = _Conversation()
    assert run_and_close(conversation, "localize this") == [dict(kind="ok")]
    assert conversation.closed
    assert conversation.sent == "localize this"


@pytest.mark.parametrize("fail", ["send", "run"])
def test_conversation_closes_when_the_run_raises(fail: str):
    conversation = _Conversation(fail=fail)
    with pytest.raises(RuntimeError):
        run_and_close(conversation, "localize this")
    assert conversation.closed


def test_malformed_finish_scores_zero_and_stays_in_the_loss():
    result = dict(
        token_messages=[dict(response_token_ids=[9])],
        locations=[dict(file="a.py")],
        reward=3.0,
        reward_detail=dict(reward=3.0),
        exhausted=False,
    )
    graded = finalize_rollout(result, _Tokenizer(), "Qwen/Qwen3.5-4B", 10)
    assert graded["reward"] == 0.0
    assert graded["locations"] is None
    assert graded["exhausted"] is False


def test_grader_does_not_require_a_conversation():
    reward, detail = grade_rollout(None, {})
    assert reward == 0.0
    assert detail["reward"] == 0.0
