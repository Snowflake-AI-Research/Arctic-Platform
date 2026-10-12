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
"""Localization task: sandbox, agent, and grader.

The chat endpoint and the episodes live in
``arctic_platform.integrations.harness``. This module is the OpenHands example.
``submit`` starts the agent, ``collect_output`` returns the submission, and
``grade_localization`` scores it. The agent does not see the reward.
"""

from __future__ import annotations

import threading
import traceback
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

from arctic_platform.integrations.harness.generator import Grade

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


def run_conversation(
    instance: dict,
    model_name: str,
    base_url: str,
    api_key: str,
    max_turns: int,
    workspace_root: str,
    training_phase: str,
) -> dict[str, Any]:
    """Clone the repo and run the agent. Grading happens after this returns."""
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
            api_key=api_key,
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
    return dict(messages=messages, locations=finish_locations(messages))


class OpenHandsAdapter:
    """Runs one conversation per ``submit``. Status is the thread, not the reward."""

    def __init__(self, model_name: str, max_turns: int, workspace: str) -> None:
        self._model_name = model_name
        self._max_turns = max_turns
        self._workspace = workspace
        self._lock = threading.Lock()
        self._runs: dict[str, dict[str, Any]] = {}

    def submit(self, request: Any) -> str:
        rollout_id = request.routing_key
        with self._lock:
            self._runs[rollout_id] = dict(status="running", artifacts=None)

        def work() -> None:
            try:
                artifacts = run_conversation(
                    request.env_extra,
                    self._model_name,
                    request.base_url,
                    request.api_key,
                    self._max_turns,
                    self._workspace,
                    request.phase,
                )
            except Exception:
                artifacts = dict(error=traceback.format_exc(), locations=None, messages=[])
                self._store(rollout_id, "errored", artifacts)
                return
            self._store(rollout_id, "completed", artifacts)

        threading.Thread(target=work, name=f"openhands-{rollout_id}", daemon=True).start()
        return rollout_id

    def _store(self, rollout_id: str, status: str, artifacts: dict) -> None:
        with self._lock:
            self._runs[rollout_id] = dict(status=status, artifacts=artifacts)

    def get_status(self, rollout_id: str) -> str:
        with self._lock:
            return self._runs[rollout_id]["status"]

    def collect_output(self, rollout_id: str) -> dict:
        with self._lock:
            artifacts = self._runs[rollout_id]["artifacts"]
        return dict(artifacts or {})


def grade_localization(episodes: list[Any], artifacts: dict, instance: dict, tokenizer: Any, max_turns: int) -> Grade:
    """Score the submission. A malformed last tool call scores 0 and stays in the loss.

    Spending the whole turn budget without a submission drops the rollout from the loss.
    """
    if artifacts.get("error"):
        reward, detail = localization_reward(None, {})
        return Grade(reward=reward, reward_detail=detail, exhausted=False)
    locations = artifacts.get("locations")
    if locations is not None and len(episodes) > 0:
        last_ids = list(episodes[-1].response_token_ids)
        last_text = tokenizer.decode(last_ids, skip_special_tokens=False)
        if not last_step_is_one_tool_call(last_text):
            locations = None
    exhausted = locations is None and len(episodes) >= max_turns
    if locations is None:
        reward, detail = localization_reward(None, {})
    else:
        reward, detail = localization_reward(locations, instance)
    return Grade(reward=reward, reward_detail=detail, exhausted=exhausted)
