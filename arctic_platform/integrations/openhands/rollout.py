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
"""One OpenHands localization rollout, and the SkyRL generator that runs a batch.

The conversation is ``send_message`` then ``run()``, bounded by
``generator.max_turns``. The registered tools are ``terminal`` and
``localization_finish``. There is no reminder and no second name for the
terminal.

Adapted from ``src/generator/code_search_generator.py`` in
https://github.com/18jeffreyma/codescout/tree/abab719e08a55dde78c6da864cd24d84fd47bdf2

OpenHands and Ray are imported inside the functions that need them.
"""

from __future__ import annotations

import asyncio
import json
import traceback
import uuid
from pathlib import Path
from typing import Any

from arctic_platform.integrations.openhands.events import finish_locations
from arctic_platform.integrations.openhands.events import last_step_is_one_tool_call
from arctic_platform.integrations.openhands.loss_mask import assistant_loss_mask
from arctic_platform.integrations.openhands.loss_mask import mask_if_exhausted
from arctic_platform.integrations.openhands.prompt_text import render_system_prompt
from arctic_platform.integrations.openhands.prompt_text import render_user_prompt
from arctic_platform.integrations.openhands.reward import localization_reward
from arctic_platform.integrations.openhands.workspace import clone_instance

_INSTRUCT_2507 = "Qwen3-4B-Instruct-2507"


def buffer_after_role(model_name: str) -> int:
    """Tokens after the assistant role marker that stay masked.

    Qwen3-4B-Instruct-2507 has no ``<think>`` block, so only the following
    newline is masked. Other Qwen3 templates keep the longer buffer from the
    cited generator.
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


def _token_events(messages: list[dict]) -> list[dict]:
    return [message for message in messages if message.get("kind") == "TokenEvent"]


def _public_messages(messages: list[dict]) -> list[dict]:
    hidden = {"prompt_token_ids", "response_token_ids"}
    return [{key: value for key, value in message.items() if key not in hidden} for message in messages]


_FINISH_REGISTERED = False


def _register_finish_tool(register_tool, tool_cls) -> None:
    """Register once per process. A reused Ray worker already has the name."""
    global _FINISH_REGISTERED
    if _FINISH_REGISTERED:
        return
    register_tool("localization_finish", tool_cls)
    _FINISH_REGISTERED = True


def run_rollout(
    instance: dict,
    *,
    model_name: str,
    base_url: str,
    max_turns: int,
    workspace_root: str,
    training_phase: str,
    routing_key: str,
) -> dict[str, Any]:
    """Clone the repo, run the conversation, and score the submission."""
    from openhands.sdk import LLM
    from openhands.sdk import Agent
    from openhands.sdk import Conversation
    from openhands.sdk import Tool
    from openhands.sdk.tool import register_tool
    from openhands.tools.terminal import TerminalTool

    from arctic_platform.integrations.openhands.finish import LocalizationFinishTool

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
    conversation.send_message(render_user_prompt(instance, str(working_dir)))
    conversation.run()
    messages = [event.model_dump() for event in conversation.state.events]
    locations = finish_locations(messages)
    token_messages = _token_events(messages)
    exhausted = locations is None and len(token_messages) >= max_turns
    reward, detail = localization_reward(locations, instance)
    return dict(
        messages=messages,
        token_messages=token_messages,
        locations=locations,
        exhausted=exhausted,
        reward=reward,
        reward_detail=detail,
        routing_key=routing_key,
    )


def score_rollout(result: dict[str, Any], tokenizer: Any, model_name: str, max_turns: int) -> dict[str, Any]:
    """Turn a rollout's token events into the training row."""
    token_messages = result["token_messages"]
    locations = result["locations"]
    exhausted = result["exhausted"] or (locations is None and len(token_messages) >= max_turns)
    if locations is not None and len(token_messages) > 0:
        last_ids = token_messages[-1].get("response_token_ids") or []
        last_text = tokenizer.decode(last_ids, skip_special_tokens=False)
        if not last_step_is_one_tool_call(last_text):
            locations = None
            exhausted = False
            result = dict(result, reward=0.0, reward_detail=localization_reward(None, {})[1])
    stitched = stitch_response(token_messages)
    if stitched is None:
        eos = tokenizer.eos_token_id or 0
        return dict(
            prompt_token_ids=[eos],
            response_ids=[eos],
            reward=0.0,
            reward_detail=result["reward_detail"],
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
    # A rejected last step keeps its mask and scores 0. Only a rollout that
    # spent the whole turn budget without a submission is masked out.
    mask = mask_if_exhausted(mask, exhausted and locations is None)
    return dict(
        prompt_token_ids=prompt_ids,
        response_ids=response_ids,
        reward=result["reward"] if locations is not None else 0.0,
        reward_detail=result["reward_detail"],
        loss_mask=mask,
        stop_reason="complete",
    )


def _ray_rollout(
    instance: dict,
    model_name: str,
    base_url: str,
    max_turns: int,
    workspace_root: str,
    training_phase: str,
    routing_key: str,
) -> dict[str, Any]:
    try:
        return run_rollout(
            instance,
            model_name=model_name,
            base_url=base_url,
            max_turns=max_turns,
            workspace_root=workspace_root,
            training_phase=training_phase,
            routing_key=routing_key,
        )
    except Exception:
        return dict(
            messages=[],
            token_messages=[],
            locations=None,
            exhausted=False,
            reward=0.0,
            reward_detail=dict(file_reward=0.0, module_reward=0.0, entity_reward=0.0, reward=0.0),
            error=traceback.format_exc(),
            routing_key=routing_key,
        )


_REMOTE = None


def _remote_rollout():
    """Ray task for one rollout. Decorated once; decorating again raises."""
    global _REMOTE
    if _REMOTE is None:
        import ray

        _REMOTE = ray.remote(num_cpus=0.01)(_ray_rollout)
    return _REMOTE


class OpenHandsGenerator:
    """SkyRL ``GeneratorInterface`` that runs one OpenHands conversation per row."""

    def __init__(
        self,
        generator_cfg: Any,
        tokenizer: Any,
        model_name: str,
        proxy: Any,
        workspace_root: str,
        traj_dir: str,
    ) -> None:
        self._cfg = generator_cfg
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._proxy = proxy
        self._workspace_root = workspace_root
        self._traj_dir = traj_dir
        self._litellm_model = "openai/" + model_name

    async def generate(self, input_batch: dict) -> dict:
        import ray
        from skyrl.train.generators.utils import get_rollout_metrics

        remote = _remote_rollout()
        metadata = input_batch["batch_metadata"]
        phase = metadata.training_phase
        if not isinstance(phase, str):
            phase = getattr(phase, "value", str(phase))
        max_turns = int(self._cfg.max_turns)
        refs = []
        for index, instance in enumerate(input_batch["env_extras"]):
            trajectory = input_batch["trajectory_ids"][index]
            routing_key = f"step{metadata.global_step}-{phase}-{trajectory.to_string()}"
            refs.append(
                remote.remote(
                    instance,
                    self._litellm_model,
                    self._proxy.url_for(routing_key),
                    max_turns,
                    self._workspace_root,
                    phase,
                    routing_key,
                )
            )
        results = await asyncio.gather(*[asyncio.to_thread(ray.get, ref) for ref in refs])
        rows = []
        reward_details = []
        for result, instance, trajectory in zip(results, input_batch["env_extras"], input_batch["trajectory_ids"]):
            scored = score_rollout(result, self._tokenizer, self._model_name, max_turns)
            rows.append(scored)
            reward_details.append(scored["reward_detail"])
            self._write_trajectory(metadata, phase, instance, trajectory, result, scored)
        metrics = get_rollout_metrics(
            [row["response_ids"] for row in rows],
            [row["reward"] for row in rows],
            loss_masks=[row["loss_mask"] for row in rows],
        )
        for key in ("file_reward", "module_reward", "entity_reward"):
            values = [detail[key] for detail in reward_details if key in detail]
            if len(values) > 0:
                metrics[f"reward/{key}"] = sum(values) / len(values)
        return dict(
            trajectory_ids=input_batch["trajectory_ids"],
            prompt_token_ids=[row["prompt_token_ids"] for row in rows],
            response_ids=[row["response_ids"] for row in rows],
            rewards=[row["reward"] for row in rows],
            loss_masks=[row["loss_mask"] for row in rows],
            stop_reasons=[row["stop_reason"] for row in rows],
            rollout_metrics=metrics,
            rollout_logprobs=None,
            is_last_step=None,
        )

    def _write_trajectory(
        self, metadata: Any, phase: str, instance: dict, trajectory: Any, result: dict, scored: dict
    ) -> None:
        if self._traj_dir == "":
            return
        directory = Path(self._traj_dir) / f"step_{metadata.global_step}" / phase
        directory.mkdir(parents=True, exist_ok=True)
        repetition = getattr(trajectory, "repetition_id", 0)
        instance_id = instance.get("instance_id", "unknown")
        if result.get("error"):
            path = directory / f"{instance_id}_{repetition}.error"
            path.write_text(result["error"], encoding="utf-8")
            return
        path = directory / f"{instance_id}_{repetition}.json"
        payload = dict(
            instance_id=instance_id,
            total_reward=scored["reward"],
            reward_dict=scored["reward_detail"],
            locations=result.get("locations"),
            messages=_public_messages(result.get("messages") or []),
        )
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")
