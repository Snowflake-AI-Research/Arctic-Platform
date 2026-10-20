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
"""Turn one training sequence into a SkyRL row. The mask comes from ``episodes.sampled_mask``."""

from __future__ import annotations

from typing import Any


def mask_if_exhausted(mask: list[int], exhausted: bool) -> list[int]:
    """Zero the mask when the rollout never submitted inside the turn budget."""
    if not exhausted:
        return list(mask)
    return [0] * len(mask)


def masked_row(
    prompt_ids: list[int],
    response_ids: list[int],
    loss_mask: list[int],
    reward: float,
    reward_detail: dict[str, Any],
    exhausted: bool,
) -> dict[str, Any]:
    """One SkyRL row. The reward is the grader's scalar for the whole rollout."""
    return dict(
        prompt_token_ids=list(prompt_ids),
        response_ids=list(response_ids),
        reward=float(reward),
        reward_detail=reward_detail,
        loss_mask=mask_if_exhausted(loss_mask, exhausted),
        stop_reason="complete",
    )
