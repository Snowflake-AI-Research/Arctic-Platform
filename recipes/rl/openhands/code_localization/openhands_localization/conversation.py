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
"""Run one OpenHands conversation and always release it.

Grading happens after this returns. The conversation does not know the reward.
"""

from __future__ import annotations

from typing import Any


def run_and_close(conversation: Any, user_message: str) -> list[dict]:
    """Send one task, run until the agent stops, and close the terminal.

    The terminal tool holds a tmux pty. Leaving it open fills ``/dev/pts``,
    and the next rollout dies in ``Conversation()`` before the model speaks.
    ``close`` runs when the conversation raises too.
    """
    try:
        conversation.send_message(user_message)
        conversation.run()
        return [event.model_dump() for event in conversation.state.events]
    finally:
        conversation.close()
