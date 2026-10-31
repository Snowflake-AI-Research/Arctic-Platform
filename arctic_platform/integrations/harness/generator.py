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
"""SkyRL generator for a harness that calls the chat endpoint.

The adapter owns the agent. The grader owns the reward. This class submits
one rollout per row, polls until it finishes, reads episodes from the proxy,
and grades the artifacts. It does not import a harness. Rollouts in one batch
run concurrently.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Protocol

from arctic_platform.integrations.harness.episodes import training_sequences
from arctic_platform.integrations.harness.tokens import masked_row

_HIDDEN = {"prompt_token_ids", "response_token_ids", "token_ids"}
_PRIVATE = {"messages", "error", "reward", "reward_detail"}


@dataclass(frozen=True)
class RolloutRequest:
    """One SkyRL row and the chat URL for that rollout.

    ``env_extra`` is the dataset row. ``base_url`` is
    ``ChatCompletionsProxy.url_for(routing_key)``. ``api_key`` is the bearer
    token minted for that URL. The adapter reads whatever else it needs from
    its own closure.
    """

    env_extra: dict[str, Any]
    trajectory_id: Any
    routing_key: str
    base_url: str
    api_key: str
    phase: str
    global_step: int


@dataclass(frozen=True)
class Grade:
    """Scalar reward for one rollout. ``exhausted`` zeros every episode's loss mask."""

    reward: float
    reward_detail: dict[str, Any]
    exhausted: bool = False


class HarnessAdapter(Protocol):
    """The three calls a harness makes around one rollout. The agent stays outside."""

    def submit(self, request: RolloutRequest) -> str:
        """Start the rollout and return its id. This does not block on the agent."""

    def get_status(self, rollout_id: str) -> str:
        """``running``, ``completed``, or ``errored``."""

    def collect_output(self, rollout_id: str) -> dict[str, Any]:
        """Artifacts for the grader. Token ids are not part of this payload."""


Grader = Callable[[list[Any], dict[str, Any], RolloutRequest], Grade]


def _public_messages(messages: list[dict]) -> list[dict]:
    return [{key: value for key, value in message.items() if key not in _HIDDEN} for message in messages]


def _detail_means(details: list[dict]) -> dict[str, float]:
    metrics = {}
    keys = {key for detail in details for key in detail if key != "reward"}
    for key in sorted(keys):
        values = [detail[key] for detail in details if isinstance(detail.get(key), (int, float))]
        if len(values) > 0:
            metrics[f"reward/{key}"] = sum(values) / len(values)
    return metrics


def rows_for_rollout(
    episodes: list[Any],
    grade: Grade,
    tokenizer: Any,
    step_wise: bool,
) -> list[dict[str, Any]]:
    """Training rows for one rollout. No episodes is one masked token and reward 0."""
    sequences = training_sequences(episodes, step_wise) if len(episodes) > 0 else []
    if len(sequences) == 0:
        eos = tokenizer.eos_token_id or 0
        row = masked_row([eos], [eos], [0], 0.0, grade.reward_detail, True)
        row["stop_reason"] = "error"
        return [row]
    return [
        masked_row(prompt_ids, response_ids, mask, grade.reward, grade.reward_detail, grade.exhausted)
        for prompt_ids, response_ids, mask in sequences
    ]


class HarnessGenerator:
    """SkyRL ``GeneratorInterface`` for any harness that speaks to the chat proxy."""

    def __init__(
        self,
        tokenizer: Any,
        proxy: Any,
        traj_dir: str,
        adapter: HarnessAdapter,
        grader: Grader,
        error_detail: dict[str, Any] | None = None,
        step_wise: bool = False,
    ) -> None:
        self._tokenizer = tokenizer
        self._proxy = proxy
        self._traj_dir = traj_dir
        self._adapter = adapter
        self._grader = grader
        self._error_detail = dict(error_detail or dict(reward=0.0))
        self._step_wise = step_wise

    async def generate(self, input_batch: dict) -> dict:
        from skyrl.train.generators.utils import get_rollout_metrics

        metadata = input_batch["batch_metadata"]
        phase = metadata.training_phase
        if not isinstance(phase, str):
            phase = getattr(phase, "value", str(phase))
        requests = []
        for env_extra, trajectory in zip(input_batch["env_extras"], input_batch["trajectory_ids"]):
            routing_key = f"step{metadata.global_step}-{phase}-{trajectory.to_string()}"
            api_key = self._proxy.issue(routing_key)
            requests.append(
                RolloutRequest(
                    env_extra=env_extra,
                    trajectory_id=trajectory,
                    routing_key=routing_key,
                    base_url=self._proxy.url_for(routing_key),
                    api_key=api_key,
                    phase=phase,
                    global_step=int(metadata.global_step),
                )
            )
        finished = await asyncio.gather(*[self._finish(request) for request in requests])
        for request, (_, artifacts, grade) in zip(requests, finished):
            self._write_trajectory(metadata, phase, request, artifacts, grade)
        errors = [artifacts.get("error") for _, artifacts, _ in finished]
        if len(errors) > 0 and all(errors):
            raise RuntimeError(
                f"Every {phase} rollout at step {metadata.global_step} errored, so there is nothing to train on. "
                f"First error:\n{errors[0]}"
            )
        rows = []
        trajectory_ids = []
        is_last_step = []
        reward_details = []
        for request, (rollout_rows, _, grade) in zip(requests, finished):
            reward_details.append(grade.reward_detail)
            for index, row in enumerate(rollout_rows):
                rows.append(row)
                trajectory_ids.append(request.trajectory_id)
                is_last_step.append(index == len(rollout_rows) - 1)
        if not self._step_wise and len(rows) != len(requests):
            raise RuntimeError(
                "a rollout produced more than one context, and generator.step_wise_trajectories is false"
            )
        metrics = get_rollout_metrics(
            [row["response_ids"] for row in rows],
            [row["reward"] for row in rows],
            loss_masks=[row["loss_mask"] for row in rows],
        )
        metrics.update(_detail_means(reward_details))
        output = dict(
            trajectory_ids=trajectory_ids if self._step_wise else input_batch["trajectory_ids"],
            prompt_token_ids=[row["prompt_token_ids"] for row in rows],
            response_ids=[row["response_ids"] for row in rows],
            rewards=[row["reward"] for row in rows],
            loss_masks=[row["loss_mask"] for row in rows],
            stop_reasons=[row["stop_reason"] for row in rows],
            rollout_metrics=metrics,
            rollout_logprobs=None,
            is_last_step=is_last_step if self._step_wise else None,
        )
        return output

    async def _finish(self, request: RolloutRequest) -> tuple[list[dict[str, Any]], dict[str, Any], Grade]:
        artifacts: dict[str, Any] = {}
        try:
            rollout_id = await asyncio.to_thread(self._adapter.submit, request)
            status = "running"
            while status == "running":
                status = await asyncio.to_thread(self._adapter.get_status, rollout_id)
                if status == "running":
                    await asyncio.sleep(0.1)
            artifacts = await asyncio.to_thread(self._adapter.collect_output, rollout_id) or {}
        except Exception:
            artifacts = dict(error=traceback.format_exc())
        episodes = self._proxy.episodes(request.routing_key)
        self._proxy.drop(request.routing_key)
        try:
            grade = self._grader(episodes, artifacts, request)
        except Exception:
            artifacts = dict(artifacts, error=traceback.format_exc())
            grade = Grade(reward=0.0, reward_detail=dict(self._error_detail), exhausted=False)
        if not isinstance(grade, Grade):
            grade = Grade(reward=0.0, reward_detail=dict(self._error_detail), exhausted=False)
        rows = rows_for_rollout(episodes, grade, self._tokenizer, self._step_wise)
        return rows, artifacts, grade

    def _write_trajectory(
        self, metadata: Any, phase: str, request: RolloutRequest, artifacts: dict, grade: Grade
    ) -> None:
        if self._traj_dir == "":
            return
        directory = Path(self._traj_dir) / f"step_{metadata.global_step}" / phase
        directory.mkdir(parents=True, exist_ok=True)
        repetition = getattr(request.trajectory_id, "repetition_id", 0)
        instance_id = request.env_extra.get("instance_id", "unknown")
        if artifacts.get("error"):
            path = directory / f"{instance_id}_{repetition}.error"
            path.write_text(str(artifacts["error"]), encoding="utf-8")
            return
        path = directory / f"{instance_id}_{repetition}.json"
        payload = trajectory_payload(instance_id, grade, artifacts)
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")


def trajectory_payload(instance_id: str, grade: Grade, artifacts: dict) -> dict[str, Any]:
    """JSON for one rollout. Extra keys from the adapter are kept as they are."""
    payload = dict(
        instance_id=instance_id,
        total_reward=grade.reward,
        reward_dict=grade.reward_detail,
        exhausted=grade.exhausted,
        messages=_public_messages(artifacts.get("messages") or []),
    )
    for key, value in artifacts.items():
        if key not in _PRIVATE and key not in payload:
            payload[key] = value
    return payload
