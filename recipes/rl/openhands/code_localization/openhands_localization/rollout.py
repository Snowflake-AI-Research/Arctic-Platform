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
"""Localization task: sandbox, agent, and reward.

The chat endpoint and the training row live in
``arctic_platform.integrations.harness``. This module is the OpenHands example:
it builds the agent, closes the conversation, and grades the submission.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

from openhands_localization.conversation import run_and_close
from openhands_localization.events import finish_locations
from openhands_localization.events import last_step_is_one_tool_call
from openhands_localization.prompt_text import render_system_prompt
from openhands_localization.prompt_text import render_user_prompt
from openhands_localization.reward import localization_reward
from openhands_localization.workspace import clone_instance

_FINISH_REGISTERED = False
_FINISH_LOCK = threading.Lock()


def _register_finish_tool(register_tool, tool_cls) -> None:
    """Register once per process. Parallel rollouts share this process."""
    global _FINISH_REGISTERED
    with _FINISH_LOCK:
        if _FINISH_REGISTERED:
            return
        register_tool("localization_finish", tool_cls)
        _FINISH_REGISTERED = True


def _token_events(messages: list[dict]) -> list[dict]:
    return [message for message in messages if message.get("kind") == "TokenEvent"]


def grade_rollout(locations: list[dict] | None, instance: dict) -> tuple[float, dict]:
    """Score a finished conversation. This does not run the agent."""
    return localization_reward(locations, instance)


def run_rollout(
    instance: dict,
    model_name: str,
    base_url: str,
    max_turns: int,
    workspace_root: str,
    training_phase: str,
    routing_key: str,
) -> dict[str, Any]:
    """Clone the repo, run the conversation, then grade the submission."""
    from openhands.sdk import LLM
    from openhands.sdk import Agent
    from openhands.sdk import Conversation
    from openhands.sdk import Tool
    from openhands.sdk.tool import register_tool
    from openhands.tools.terminal import TerminalTool
    from openhands_localization.finish import LocalizationFinishTool

    rollout_dir = Path(workspace_root) / uuid.uuid4().hex[:8]
    working_dir = clone_instance(
        instance["repo"],
        instance.get("base_commit"),
        instance["instance_id"],
        rollout_dir,
        instance.get("patch") if instance.get("use_patch") else None,
    )
    system_path = rollout_dir / "system_prompt.txt"
    system_path.write_text(render_system_prompt(max_turns), encoding="utf-8")

    _register_finish_tool(register_tool, LocalizationFinishTool)
    temperature = 0.6 if training_phase == "eval" else 1.0
    agent = Agent(
        llm=LLM(
            usage_id="agent",
            model=model_name,
            base_url=base_url,
            api_key="sk-xxx",
            temperature=temperature,
            litellm_extra_body=dict(
                return_token_ids=True,
                include_stop_str_in_output=False,
                chat_template_kwargs=dict(add_generation_prompt=True, enable_thinking=False),
            ),
        ),
        tools=[Tool(name=TerminalTool.name), Tool(name="localization_finish")],
        system_prompt_filename=str(system_path),
    )
    conversation = Conversation(
        agent=agent, max_iteration_per_run=max_turns, visualizer=None, workspace=str(working_dir)
    )
    messages = run_and_close(conversation, render_user_prompt(instance, str(working_dir)))
    locations = finish_locations(messages)
    token_messages = _token_events(messages)
    exhausted = locations is None and len(token_messages) >= max_turns
    reward, detail = grade_rollout(locations, instance)
    return dict(
        messages=messages,
        token_messages=token_messages,
        locations=locations,
        exhausted=exhausted,
        reward=reward,
        reward_detail=detail,
        routing_key=routing_key,
    )


def finalize_rollout(result: dict[str, Any], tokenizer: Any, _model_name: str, max_turns: int) -> dict[str, Any]:
    """Reject a malformed last tool call, and mark a rollout that never submitted.

    A rejected call scores 0 and keeps its loss mask. Spending the whole turn
    budget without a submission drops the rollout from the loss.
    """
    if result.get("error"):
        return result
    token_messages = result.get("token_messages") or []
    locations = result.get("locations")
    exhausted = bool(result.get("exhausted")) or (locations is None and len(token_messages) >= max_turns)
    if locations is not None and len(token_messages) > 0:
        last_ids = token_messages[-1].get("response_token_ids") or []
        last_text = tokenizer.decode(last_ids, skip_special_tokens=False)
        if not last_step_is_one_tool_call(last_text):
            _reward, detail = grade_rollout(None, {})
            return dict(result, locations=None, exhausted=False, reward=0.0, reward_detail=detail)
    reward = 0.0 if locations is None else result.get("reward")
    return dict(result, locations=locations, exhausted=exhausted, reward=reward)
