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
"""Turn token events into one training row.

Each chat request re-renders the whole conversation, so the row keeps the
first turn's prompt and strips that same prefix from the last request. That
is one sequence for one conversation. A harness that compacts context or
spawns a subagent needs a row per model call instead.
"""

from __future__ import annotations

from typing import Any

from arctic_platform.integrations.harness.loss_mask import assistant_loss_mask
from arctic_platform.integrations.harness.loss_mask import mask_if_exhausted

_INSTRUCT_2507 = "Qwen3-4B-Instruct-2507"


def buffer_after_role(model_name: str) -> int:
    """Tokens after the assistant role marker that stay masked.

    Qwen3-4B-Instruct-2507 masks one token. Other Qwen3 templates mask five.
    """
    if _INSTRUCT_2507 in model_name:
        return 1
    return 5


def stitch_response(token_messages: list[dict]) -> tuple[list[int], list[int]] | None:
    """First-turn prompt ids, and the response ids trained on.

    The response is the last turn's full prompt plus its sampled tokens, with
    the first turn's prompt prefix removed. Earlier assistant turns are the
    re-tokenized chat, not the originally sampled ids.
    """
    if len(token_messages) == 0:
        return None
    first_prompt = list(token_messages[0]["prompt_token_ids"])
    last_prompt = list(token_messages[-1]["prompt_token_ids"])
    last_response = list(token_messages[-1]["response_token_ids"])
    combined = last_prompt + last_response
    return first_prompt, combined[len(first_prompt) :]


def training_row(result: dict[str, Any], tokenizer: Any, model_name: str) -> dict[str, Any]:
    """Mask a stitched rollout. The reward is whatever the task already set."""
    token_messages = result.get("token_messages") or []
    detail = result.get("reward_detail") or dict(reward=0.0)
    stitched = stitch_response(token_messages)
    if stitched is None:
        eos = tokenizer.eos_token_id or 0
        return dict(
            prompt_token_ids=[eos],
            response_ids=[eos],
            reward=0.0,
            reward_detail=detail,
            loss_mask=[0],
            stop_reason="error",
        )
    prompt_ids, response_ids = stitched
    start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_id = tokenizer.convert_tokens_to_ids("assistant")
    mask = assistant_loss_mask(
        response_ids,
        start_token_id=start_id,
        assistant_token_id=assistant_id,
        buffer_succeed=buffer_after_role(model_name),
    )
    mask = mask_if_exhausted(mask, bool(result.get("exhausted")))
    return dict(
        prompt_token_ids=prompt_ids,
        response_ids=response_ids,
        reward=float(result.get("reward") or 0.0),
        reward_detail=detail,
        loss_mask=mask,
        stop_reason="complete",
    )
