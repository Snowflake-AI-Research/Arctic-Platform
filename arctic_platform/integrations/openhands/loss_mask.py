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
"""Loss mask over a multi-turn response.

Adapted from the mask built in ``code_search_generator.py`` at
https://github.com/18jeffreyma/codescout/blob/abab719e08a55dde78c6da864cd24d84fd47bdf2/src/generator/code_search_generator.py

The response is the last turn's prompt with the first turn's prompt prefix
removed, then the last turn's sampled tokens. Chat scaffolding
(``<|im_start|>`` through the assistant role marker, plus a short buffer) is
masked. A rollout that used every turn without calling ``localization_finish``
is masked entirely, so it contributes no gradient.
"""

from __future__ import annotations


def assistant_loss_mask(
    response_ids: list[int],
    *,
    start_token_id: int,
    assistant_token_id: int,
    buffer_succeed: int = 5,
    buffer_precede: int = 1,
) -> list[int]:
    """1 on tokens the policy is trained on, 0 on chat scaffolding.

    ``buffer_precede`` tokens before each ``<|im_start|>`` are masked, and
    ``buffer_succeed`` tokens after the assistant role marker are masked.
    The role marker has to sit on the token immediately after ``<|im_start|>``.
    """
    mask: list[int] = []
    inside = False
    buffer = 0
    previous_was_start = False
    for token_id in response_ids:
        if token_id == start_token_id:
            inside = True
            # Mask the tokens just before this marker. If the response starts
            # on the marker there is nothing to mask, and the mask must stay
            # the same length as the response.
            popped = 0
            for _ in range(buffer_precede):
                if len(mask) == 0:
                    break
                mask.pop()
                popped += 1
            mask.extend([0] * popped)
            mask.append(0)
        elif token_id == assistant_token_id and previous_was_start:
            inside = False
            mask.append(0)
            buffer = buffer_succeed
        elif inside:
            mask.append(0)
        elif buffer > 0:
            mask.append(0)
            buffer -= 1
        else:
            mask.append(1)
        previous_was_start = token_id == start_token_id
    return mask


def mask_if_exhausted(mask: list[int], exhausted: bool) -> list[int]:
    """Zero the mask when the rollout never submitted inside the turn budget."""
    if not exhausted:
        return mask
    return [0] * len(mask)
