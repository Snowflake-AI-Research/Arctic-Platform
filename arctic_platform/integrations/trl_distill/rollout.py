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

"""Student generate + teacher top-k score via ``ArcticOPDClient``.

Implements TRL ``RolloutWorkerProtocol`` (buffer, start/stop/version/health)
without importing TRL. Generation is remote; this process stays CPU-only.
"""

from __future__ import annotations

import queue
from typing import Any

from arctic_platform.integrations.trl_distill.types import RolloutSample
from arctic_platform.opd.scoring import score_teacher_topk


def _sampled_logprobs(output: dict[str, Any]) -> tuple[list[int], list[float]]:
    token_ids = list(output["token_ids"])
    positions = output.get("logprobs")
    if positions is None:
        raise RuntimeError("generate did not return logprobs; pass logprobs=0")
    values: list[float] = []
    for token_id, position in zip(token_ids, positions):
        if isinstance(position, dict):
            entry = position.get(token_id, position.get(str(token_id)))
            if entry is None:
                raise RuntimeError(f"generated token {token_id} missing from logprob keys")
            values.append(float(entry["logprob"] if isinstance(entry, dict) else entry))
        else:
            values.append(float(position))
    return token_ids, values


class ArcticOPDRolloutWorker:
    """TRL ``rollout_worker=`` backend: Arctic student generate + teacher score."""

    def __init__(
        self,
        client: Any,
        *,
        teacher_top_k: int = 8,
        temperature: float = 1.0,
        teacher_temperature: float = 1.0,
        add_tail_bucket: bool = True,
    ) -> None:
        self.client = client
        self.teacher_top_k = teacher_top_k
        self.temperature = temperature
        self.teacher_temperature = teacher_temperature
        self.add_tail_bucket = add_tail_bucket
        self.rollout_buffer: queue.Queue[RolloutSample] = queue.Queue()
        self.model_version = 0
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def update_model_version(self, model_version: int) -> None:
        self.model_version = int(model_version)

    def check_health(self, stale_after_s: float) -> None:
        del stale_after_s

    def generate_and_score(self, prompt_ids: list[list[int]], *, max_tokens: int) -> list[RolloutSample]:
        outputs = self.client.generate(
            prompt_ids,
            {
                "n": 1,
                "temperature": self.temperature,
                "max_tokens": max_tokens,
                "logprobs": 0,
                "top_k": -1,
            },
        )
        rollouts: list[dict[str, Any]] = []
        for prompt, output in zip(prompt_ids, outputs):
            token_ids, sampler_logprobs = _sampled_logprobs(output)
            if not token_ids:
                continue
            rollouts.append(
                {
                    "prompt_ids": prompt,
                    "completion_ids": token_ids,
                    "sampler_logprobs": sampler_logprobs,
                }
            )
        if not rollouts:
            raise RuntimeError("student generated empty completions for every prompt")
        scored = score_teacher_topk(
            self.client,
            rollouts,
            teacher_top_k=self.teacher_top_k,
            teacher_temperature=self.teacher_temperature,
            add_tail_bucket=self.add_tail_bucket,
        )
        samples = [
            RolloutSample(
                prompt_ids=list(row["prompt_ids"]),
                completion_ids=list(row["completion_ids"]),
                sampler_logprobs=list(row["sampler_logprobs"]),
                teacher_token_ids=list(row["teacher_token_ids"]),
                teacher_logprobs=list(row["teacher_logprobs"]),
                teacher_tail_logprob=row.get("teacher_tail_logprob"),
            )
            for row in scored
        ]
        for sample in samples:
            self.rollout_buffer.put(sample)
        return samples
