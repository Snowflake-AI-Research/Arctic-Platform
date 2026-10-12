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

The harness function owns the agent and the reward. This class calls
``rollout(request)`` for each row, stitches tokens, and writes trajectories.
It does not import a harness. Rollouts in one batch run on threads.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable

from arctic_platform.integrations.harness.tokens import training_row

_HIDDEN = {"prompt_token_ids", "response_token_ids"}
_PRIVATE = {
    "messages",
    "token_messages",
    "exhausted",
    "reward",
    "reward_detail",
    "error",
    "routing_key",
}


@dataclass(frozen=True)
class RolloutRequest:
    """One SkyRL row and the chat URL for that rollout.

    ``env_extra`` is the dataset row. ``base_url`` is
    ``ChatCompletionsProxy.url_for(routing_key)``. The harness reads whatever
    else it needs from its own closure.
    """

    env_extra: dict[str, Any]
    trajectory_id: Any
    routing_key: str
    base_url: str
    phase: str
    global_step: int


RolloutFn = Callable[[RolloutRequest], dict[str, Any]]
FinalizeFn = Callable[[dict[str, Any]], dict[str, Any]]


def _invoke(rollout: RolloutFn, error_detail: dict[str, Any], request: RolloutRequest) -> dict[str, Any]:
    """Call a task rollout. One failure becomes a zero-reward row, not a dead batch."""
    try:
        return rollout(request)
    except Exception:
        return dict(
            messages=[],
            token_messages=[],
            exhausted=False,
            reward=0.0,
            reward_detail=dict(error_detail),
            error=traceback.format_exc(),
            routing_key=request.routing_key,
        )


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


class HarnessGenerator:
    """SkyRL ``GeneratorInterface`` for any harness that speaks to the chat proxy."""

    def __init__(
        self,
        tokenizer: Any,
        model_name: str,
        proxy: Any,
        traj_dir: str,
        rollout: RolloutFn,
        finalize: FinalizeFn | None = None,
        error_detail: dict[str, Any] | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._proxy = proxy
        self._traj_dir = traj_dir
        self._rollout = rollout
        self._finalize = finalize
        self._error_detail = dict(error_detail or dict(reward=0.0))

    async def generate(self, input_batch: dict) -> dict:
        from skyrl.train.generators.utils import get_rollout_metrics

        metadata = input_batch["batch_metadata"]
        phase = metadata.training_phase
        if not isinstance(phase, str):
            phase = getattr(phase, "value", str(phase))
        requests = []
        for env_extra, trajectory in zip(input_batch["env_extras"], input_batch["trajectory_ids"]):
            routing_key = f"step{metadata.global_step}-{phase}-{trajectory.to_string()}"
            requests.append(
                RolloutRequest(
                    env_extra=env_extra,
                    trajectory_id=trajectory,
                    routing_key=routing_key,
                    base_url=self._proxy.url_for(routing_key),
                    phase=phase,
                    global_step=int(metadata.global_step),
                )
            )
        results = await asyncio.gather(
            *[asyncio.to_thread(_invoke, self._rollout, self._error_detail, request) for request in requests]
        )
        rows = []
        reward_details = []
        for result, env_extra, trajectory in zip(results, input_batch["env_extras"], input_batch["trajectory_ids"]):
            scored = self._score(result)
            rows.append(scored)
            reward_details.append(scored["reward_detail"])
            self._write_trajectory(metadata, phase, env_extra, trajectory, result, scored)
        metrics = get_rollout_metrics(
            [row["response_ids"] for row in rows],
            [row["reward"] for row in rows],
            loss_masks=[row["loss_mask"] for row in rows],
        )
        metrics.update(_detail_means(reward_details))
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

    def _score(self, result: dict) -> dict[str, Any]:
        if self._finalize is not None:
            result = self._finalize(result)
        return training_row(result, self._tokenizer, self._model_name)

    def _write_trajectory(
        self, metadata: Any, phase: str, env_extra: dict, trajectory: Any, result: dict, scored: dict
    ) -> None:
        if self._traj_dir == "":
            return
        directory = Path(self._traj_dir) / f"step_{metadata.global_step}" / phase
        directory.mkdir(parents=True, exist_ok=True)
        repetition = getattr(trajectory, "repetition_id", 0)
        instance_id = env_extra.get("instance_id", "unknown")
        if result.get("error"):
            path = directory / f"{instance_id}_{repetition}.error"
            path.write_text(result["error"], encoding="utf-8")
            return
        path = directory / f"{instance_id}_{repetition}.json"
        payload = trajectory_payload(instance_id, scored, result)
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")


def trajectory_payload(instance_id: str, scored: dict, result: dict) -> dict[str, Any]:
    """JSON for one rollout. Extra keys from the harness are kept as they are."""
    payload = dict(
        instance_id=instance_id,
        total_reward=scored["reward"],
        reward_dict=scored["reward_detail"],
        messages=_public_messages(result.get("messages") or []),
    )
    for key, value in result.items():
        if key not in _PRIVATE and key not in payload:
            payload[key] = value
    return payload
