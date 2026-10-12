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

The harness function owns the agent and the reward. This class batches it,
stitches tokens, and writes trajectories. It does not import a harness.
Ray is imported when a batch runs.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from functools import partial
from pathlib import Path
from typing import Any
from typing import Callable

from arctic_platform.integrations.harness.tokens import training_row

RolloutFn = Callable[..., dict[str, Any]]
FinalizeFn = Callable[[dict[str, Any], Any, str, int], dict[str, Any]]

_REMOTES: dict[int, Any] = {}
_HIDDEN = {"prompt_token_ids", "response_token_ids"}


def _invoke(rollout: RolloutFn, error_detail: dict[str, Any], *args: Any) -> dict[str, Any]:
    """Call a task rollout. One failure becomes a zero-reward row, not a dead batch."""
    try:
        return rollout(*args)
    except Exception:
        routing_key = args[6] if len(args) >= 7 else ""
        return dict(
            messages=[],
            token_messages=[],
            exhausted=False,
            reward=0.0,
            reward_detail=dict(error_detail),
            error=traceback.format_exc(),
            routing_key=routing_key,
        )


def _remote(rollout: RolloutFn, error_detail: dict[str, Any]) -> Any:
    """Ray task for this rollout function. Decorated once; decorating again raises."""
    key = id(rollout)
    if key not in _REMOTES:
        import ray

        _REMOTES[key] = ray.remote(num_cpus=0.01)(partial(_invoke, rollout, error_detail))
    return _REMOTES[key]


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
        generator_cfg: Any,
        tokenizer: Any,
        model_name: str,
        proxy: Any,
        workspace_root: str,
        traj_dir: str,
        rollout: RolloutFn,
        finalize: FinalizeFn | None = None,
        error_detail: dict[str, Any] | None = None,
    ) -> None:
        self._cfg = generator_cfg
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._proxy = proxy
        self._workspace_root = workspace_root
        self._traj_dir = traj_dir
        self._rollout = rollout
        self._finalize = finalize
        self._error_detail = dict(error_detail or dict(reward=0.0))
        self._litellm_model = "openai/" + model_name

    async def generate(self, input_batch: dict) -> dict:
        import ray
        from skyrl.train.generators.utils import get_rollout_metrics

        remote = _remote(self._rollout, self._error_detail)
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
            scored = self._score(result, max_turns)
            rows.append(scored)
            reward_details.append(scored["reward_detail"])
            self._write_trajectory(metadata, phase, instance, trajectory, result, scored)
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

    def _score(self, result: dict, max_turns: int) -> dict[str, Any]:
        if self._finalize is not None:
            result = self._finalize(result, self._tokenizer, self._model_name, max_turns)
        return training_row(result, self._tokenizer, self._model_name)

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
            messages=_public_messages(result.get("messages") or []),
        )
        if "locations" in result:
            payload["locations"] = result.get("locations")
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")
