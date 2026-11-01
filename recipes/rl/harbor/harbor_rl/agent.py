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
"""Harbor ``BaseAgent`` for OpenHands SDK conversations on Cortex Training.

Supports both:
- Code localization Harbor tasks (using ``LocalizationFinishTool`` and writing
  ``/logs/agent/localization_submission.json`` for the task's ``tests/test.sh``
  F1 verifier), and
- General Harbor tasks (using ``TerminalTool`` + ``HarborFinishTool`` so the
  task's ``tests/test.sh`` verifier grades the workspace directly).
Also loads any OpenHands ``SKILL.md`` files passed via ``HARBOR_SKILL`` or
Harbor's ``--skill`` flag.

Training picks the localization mode from the dataset row's ground truth. With
``harbor run``, which passes no row, set it with ``--ak localization=true``.

The ``terminal`` tool runs every command through the trial's Harbor
environment (:mod:`harbor_rl.terminal`), so the agent works in the same
filesystem the verifier grades, whichever backend ``HARBOR_ENV`` selects.
``file_editor`` is not offered: it edits the training driver's filesystem
rather than the environment's.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import threading
import warnings
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext as HarborAgentContext

try:
    from harbor.agents.capabilities import AgentCapabilities
except ImportError:
    AgentCapabilities = None  # type: ignore[assignment,misc]

# The pinned OpenHands SDK has no lock around its pydantic classes. Defining a subclass
# marks every SDK model for a rebuild, and the next model built in any thread runs it;
# a conversation being built meanwhile in another thread can come out without its
# fields ("'ConversationState' object has no attribute 'security_analyzer'"). Two
# definitions of one class leave a duplicate that fails every later conversation in the
# process ("Duplicate kind detected"). So classes are defined, conversations are built,
# and the pending rebuild runs, all under this lock: once the first conversations are
# set up, the running ones never trigger a rebuild.
_SDK_SETUP_LOCK = threading.Lock()
_REGISTERED_TOOLS: set[str] = set()


def _settle_sdk_classes() -> None:
    """Run the rebuild that defining SDK classes left pending. Call with ``_SDK_SETUP_LOCK`` held."""
    from openhands.sdk.utils.models import _rebuild_if_required

    _rebuild_if_required()


def _ensure_tool_registered(name: str, build: Callable[[], Any]) -> None:
    """Register the tool class ``build()`` returns, building it only on the first call."""
    from openhands.sdk.tool import register_tool

    with _SDK_SETUP_LOCK:
        if name in _REGISTERED_TOOLS:
            return
        register_tool(name, build())
        _REGISTERED_TOOLS.add(name)
        _settle_sdk_classes()


def _load_skills(skill_paths: Sequence[str | Path]) -> list[Any]:
    from openhands.sdk.context import Skill

    loaded: list[Any] = []
    seen: set[str] = set()
    for raw in skill_paths:
        if not raw:
            continue
        p = Path(str(raw)).expanduser()
        candidates: list[Path] = []
        if p.is_file() and p.name == "SKILL.md":
            candidates.append(p)
        elif p.is_dir():
            if (p / "SKILL.md").is_file():
                candidates.append(p / "SKILL.md")
            else:
                candidates.extend(sorted(p.glob("*/SKILL.md")))
        for skill_file in candidates:
            name = skill_file.parent.name
            if name in seen:
                continue
            seen.add(name)
            loaded.append(
                Skill(
                    name=name,
                    content=skill_file.read_text(encoding="utf-8"),
                    source=str(skill_file),
                    trigger=None,
                )
            )
    return loaded


def _localization_finish_tool() -> Any:
    from openhands_localization.finish import LocalizationFinishTool

    return LocalizationFinishTool


@functools.cache
def _build_harbor_finish_tool() -> Any:
    """The ``harbor_finish`` tool class, built once: OpenHands rejects a second class with the same kind."""
    from openhands.sdk import Action
    from openhands.sdk import Observation
    from openhands.sdk import ToolDefinition
    from openhands.sdk.conversation.state import ConversationExecutionStatus
    from openhands.sdk.tool import ToolAnnotations
    from openhands.sdk.tool import ToolExecutor
    from pydantic import Field
    from rich.text import Text

    class HarborFinishAction(Action):
        summary: str = Field(default="done", description="Brief summary of what was completed.")

        @property
        def visualize(self) -> Text:
            t = Text()
            t.append(f"Finished task: {self.summary}\n")
            return t

    class HarborFinishObservation(Observation):
        @property
        def visualize(self) -> Text:
            return Text()

    class HarborFinishExecutor(ToolExecutor):
        def __call__(self, action: HarborFinishAction, conversation: Any = None) -> HarborFinishObservation:
            if conversation is not None:
                conversation.state.execution_status = ConversationExecutionStatus.FINISHED
            return HarborFinishObservation.from_text(text=action.summary)

    class HarborFinishTool(ToolDefinition[HarborFinishAction, HarborFinishObservation]):
        @classmethod
        def create(cls, conv_state: Any, **params: Any) -> Sequence[HarborFinishTool]:
            if params:
                raise ValueError("HarborFinishTool doesn't accept parameters")
            return [
                cls(
                    action_type=HarborFinishAction,
                    observation_type=HarborFinishObservation,
                    description=(
                        "Signal that you have completed the task and verified your changes "
                        "in the workspace. The agent terminates immediately after this call."
                    ),
                    executor=HarborFinishExecutor(),
                    annotations=ToolAnnotations(
                        title="harbor_finish",
                        readOnlyHint=True,
                        destructiveHint=False,
                        idempotentHint=True,
                        openWorldHint=False,
                    ),
                )
            ]

    return HarborFinishTool


def _build_openhands_tools(finish_tool_name: str, terminal: Any) -> list[Any]:
    from openhands.sdk import Tool
    from openhands.tools.terminal import TerminalTool

    raw_tools = os.environ.get("OPENHANDS_TOOLS", "").strip()
    if not raw_tools:
        return [terminal, Tool(name=finish_tool_name)]

    tool_map: dict[str, str] = {"terminal": TerminalTool.name}
    try:
        from openhands.tools.task_tracker import TaskTrackerTool

        tool_map["task_tracker"] = TaskTrackerTool.name
    except ImportError:
        pass

    tools: list[Any] = []
    seen: set[str] = set()
    for item in raw_tools.split(","):
        key = item.strip()
        if not key:
            continue
        if key in ("file_editor", "str_replace_editor"):
            raise ValueError(
                f"OPENHANDS_TOOLS={raw_tools!r}: {key} edits the training driver's filesystem, not the "
                "Harbor environment's. Use terminal (it runs in the environment) instead."
            )
        resolved = tool_map.get(key, key)
        if resolved not in seen:
            seen.add(resolved)
            tools.append(terminal if resolved == TerminalTool.name else Tool(name=resolved))
    if finish_tool_name and finish_tool_name not in seen:
        tools.append(Tool(name=finish_tool_name))
    return tools


GENERAL_HARBOR_SYSTEM_PROMPT = (
    "You are a skilled software engineer working in a Linux environment. "
    "You have access to the `terminal` tool to execute bash commands, inspect files, "
    "and modify the workspace. Complete the task described by the user within "
    "{max_turns} turns. Once you have written and verified the required output files, "
    "call the `harbor_finish` tool immediately to end the rollout so the verifier can "
    "grade your workspace."
)


class ConversationStop:
    """Lets a cancelled rollout pause the conversation running on its worker thread.

    ``stopped`` is also handed to the terminal executor, which refuses commands
    once it is set, so the step in progress ends without touching the environment.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conversation: Any = None
        self.stopped = threading.Event()

    def attach(self, conversation: Any) -> bool:
        """Register ``conversation``; False if the rollout was already stopped."""
        with self._lock:
            self._conversation = conversation
            return not self.stopped.is_set()

    def stop(self) -> None:
        """Ask the conversation to end after the step in progress. Never blocks.

        OpenHands holds the conversation's lock for a whole step, and ``pause()``
        waits for it, so the pause runs on its own thread: called from the event
        loop it would deadlock with a step waiting on that loop for a command.
        """
        with self._lock:
            self.stopped.set()
            conversation = self._conversation
        if conversation is not None:
            threading.Thread(target=conversation.pause, name="harbor-pause", daemon=True).start()


def _is_json_object(text: Any) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except (TypeError, ValueError):
        return False


@functools.cache
def _hosted_endpoint_llm() -> type:
    """The OpenHands ``LLM`` for ``harbor run``, which calls a hosted endpoint, not the training proxy.

    A Cortex model deployment accepts only standard chat-completions messages: string
    content, no ``name`` on tool results, and tool-call arguments that are a JSON
    object. The SDK replays a model's malformed arguments as they were sampled; the
    agent already saw the parse error for that call, so they are sent as ``{}``.
    """
    from openhands.sdk import LLM

    class HostedEndpointLLM(LLM):
        force_string_serializer: bool | None = True

        def format_messages_for_llm(self, messages: list[Any]) -> list[dict]:
            formatted = super().format_messages_for_llm(messages)
            for message in formatted:
                if message.get("role") == "tool":
                    message.pop("name", None)
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    if not _is_json_object(function.get("arguments")):
                        function["arguments"] = "{}"
            return formatted

    return HostedEndpointLLM


def _run_openhands_sync(*, terminal_executor: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], Any, int]:
    from harbor_rl.terminal import terminal_tool

    with terminal_tool(terminal_executor) as terminal:
        return _converse(terminal=terminal, **kwargs)


def _converse(
    *,
    instruction: str,
    working_dir: str,
    terminal: Any,
    logs_dir: Path,
    model_name: str,
    base_url: str,
    api_key: str,
    max_turns: int,
    training_phase: str | None,
    is_localization: bool,
    instance: dict[str, Any],
    skill_paths: list[str],
    stop: ConversationStop,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, int]:
    from openhands.sdk import LLM
    from openhands.sdk import Agent
    from openhands.sdk import AgentContext
    from openhands.sdk import Conversation
    from openhands.sdk.tool.tool import create_action_type_with_risk

    logs_dir.mkdir(parents=True, exist_ok=True)
    system_path = logs_dir / "system_prompt.txt"

    if is_localization:
        from openhands_localization.events import finish_locations
        from openhands_localization.prompt_text import render_system_prompt
        from openhands_localization.prompt_text import render_user_prompt

        system_path.write_text(render_system_prompt(max_turns), encoding="utf-8")
        _ensure_tool_registered("localization_finish", _localization_finish_tool)
        finish_tool_name = "localization_finish"
        problem = str(instance.get("problem_statement") or instruction)
        user_prompt = render_user_prompt({"problem_statement": problem}, working_dir)
    else:
        finish_locations = None
        system_path.write_text(GENERAL_HARBOR_SYSTEM_PROMPT.format(max_turns=max_turns), encoding="utf-8")
        _ensure_tool_registered("harbor_finish", _build_harbor_finish_tool)
        finish_tool_name = "harbor_finish"
        user_prompt = f"Your working directory is `{working_dir}`.\n\n{instruction}\n"

    custom_sys = os.environ.get("OPENHANDS_SYSTEM_PROMPT_FILE", "").strip()
    if custom_sys and Path(custom_sys).is_file():
        raw_sys = Path(custom_sys).read_text(encoding="utf-8")
        system_path.write_text(raw_sys.replace("{max_turns}", str(max_turns)), encoding="utf-8")

    skills = _load_skills(skill_paths)
    agent_context = AgentContext(skills=skills) if skills else None

    temperature = 1.0 if training_phase == "train" else 0.6
    llm_model = model_name if model_name.startswith("openai/") else f"openai/{model_name}"
    llm_cls: type = LLM
    llm_kwargs: dict[str, Any] = {}
    if training_phase is not None:
        llm_kwargs["litellm_extra_body"] = dict(
            chat_template_kwargs=dict(add_generation_prompt=True, enable_thinking=False),
        )
    else:
        llm_cls = _hosted_endpoint_llm()
    agent_kwargs: dict[str, Any] = dict(
        llm=llm_cls(
            usage_id="agent",
            model=llm_model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            **llm_kwargs,
        ),
        tools=_build_openhands_tools(finish_tool_name, terminal),
        system_prompt_filename=str(system_path),
    )
    if agent_context is not None:
        agent_kwargs["agent_context"] = agent_context

    # Local scratch space for the conversation's own state; commands run in the environment.
    scratch = logs_dir / "openhands"
    scratch.mkdir(parents=True, exist_ok=True)
    with _SDK_SETUP_LOCK:
        agent = Agent(**agent_kwargs)
        conversation = Conversation(
            agent=agent,
            max_iteration_per_run=max_turns,
            visualizer=None,
            workspace=str(scratch),
        )
        # The SDK would otherwise define these on the first LLM call, outside the lock.
        for tool in agent.tools_map.values():
            create_action_type_with_risk(tool.action_type)
        _settle_sdk_classes()
    try:
        if stop.attach(conversation):
            conversation.send_message(user_prompt)
            conversation.run()
        events = list(conversation.state.events)
    finally:
        try:
            conversation.close()
        except Exception as exc:
            warnings.warn(f"conversation.close() failed: {exc}", stacklevel=2)

    dumped = [e.model_dump() if hasattr(e, "model_dump") else e for e in events]
    locations = finish_locations(dumped) if (is_localization and finish_locations is not None) else None
    n_agent_turns = sum(
        1
        for e in dumped
        if isinstance(e, dict) and e.get("source") == "agent" and e.get("kind") in ("ActionEvent", "MessageEvent")
    )
    return dumped, locations, n_agent_turns


class OpenHandsHarborAgent(BaseAgent):
    """Harbor ``BaseAgent`` that runs an OpenHands ``Conversation`` in the trial's environment."""

    if AgentCapabilities is not None:
        capabilities = AgentCapabilities(base_url=True, skills=True)

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        max_turns: int = 10,
        training_phase: str | None = None,
        localization: bool | None = None,
        instance: dict[str, Any] | None = None,
        skill_paths: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
        terminal_timeout_sec: int = 180,
        **kwargs: Any,
    ) -> None:
        forward = {k: v for k, v in kwargs.items() if k in ("logger", "mcp_servers", "skills_dir")}
        super().__init__(logs_dir=Path(logs_dir), model_name=model_name, **forward)
        env_map = dict(extra_env or getattr(self, "_extra_env", None) or {})
        self._base_url = (
            base_url
            or env_map.get("OPENAI_BASE_URL")
            or env_map.get("LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("LLM_BASE_URL")
            or ""
        )
        self._api_key = (
            api_key
            or env_map.get("OPENAI_API_KEY")
            or env_map.get("LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or "harbor"
        )
        self._max_turns = int(max_turns)
        self._terminal_timeout_sec = int(terminal_timeout_sec)
        self._training_phase = training_phase
        self._localization = localization
        self._instance = dict(instance or {})
        env_skill = os.environ.get("HARBOR_SKILL", "")
        self._skill_paths = list(skill_paths or ([env_skill] if env_skill else []))
        if self.skills_dir:
            self._skill_paths.append(str(self.skills_dir))

    @staticmethod
    def name() -> str:
        return "openhands-harbor"

    def version(self) -> str | None:
        return "0.1"

    async def setup(self, environment: BaseEnvironment) -> None:  # noqa: ARG002
        return

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: HarborAgentContext,  # noqa: ARG002
    ) -> None:
        from harbor_rl.terminal import HarborTerminalExecutor

        pwd = await environment.exec("pwd")
        working_dir = (pwd.stdout or "").strip().splitlines()[-1] if pwd.return_code == 0 and pwd.stdout else ""
        if not working_dir:
            raise RuntimeError(f"Couldn't read the environment's working directory: {pwd.stderr or pwd.stdout!r}")
        stop = ConversationStop()
        executor = HarborTerminalExecutor(
            environment,
            asyncio.get_running_loop(),
            working_dir,
            default_timeout_s=self._terminal_timeout_sec,
            stopped=stop.stopped,
        )

        truth_exists = self._instance.get("file_changes") is not None or self._instance.get("target") is not None
        is_localization = truth_exists if self._localization is None else bool(self._localization)

        try:
            messages, locations, n_turns = await asyncio.to_thread(
                _run_openhands_sync,
                instruction=instruction,
                working_dir=working_dir,
                terminal_executor=executor,
                logs_dir=self.logs_dir,
                model_name=self.model_name or "Qwen/Qwen3.5-4B",
                base_url=self._base_url,
                api_key=self._api_key,
                max_turns=self._max_turns,
                training_phase=self._training_phase,
                is_localization=is_localization,
                instance=self._instance,
                skill_paths=self._skill_paths,
                stop=stop,
            )
        except asyncio.CancelledError:
            stop.stop()
            raise

        if locations is not None:
            sub_file = self.logs_dir / "localization_submission.json"
            sub_file.write_text(json.dumps(locations), encoding="utf-8")
            await environment.upload_file(sub_file, "/logs/agent/localization_submission.json")

        exhausted = (locations is None and n_turns >= self._max_turns) if truth_exists else False
        meta_file = self.logs_dir / "rollout_meta.json"
        meta_file.write_text(
            json.dumps(dict(exhausted=exhausted, locations=locations, messages=messages), default=str),
            encoding="utf-8",
        )
