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

"""TRL-shaped rollout sample for async distillation, without importing TRL."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RolloutSample:
    prompt_ids: list[int]
    completion_ids: list[int]
    sampler_logprobs: list[float]
    teacher_token_ids: list[list[int]]
    teacher_logprobs: list[list[float]]
    teacher_tail_logprob: list[float] | None = None
    teacher_id: str = "default"
    prompt_id: int = 0
    model_version: int = 0
    enqueued_at: float = 0.0
    metrics: dict = field(default_factory=dict)

    @property
    def input_ids(self) -> list[int]:
        return list(self.prompt_ids) + list(self.completion_ids)

    @property
    def completion_mask(self) -> list[int]:
        return [0] * len(self.prompt_ids) + [1] * len(self.completion_ids)

    @property
    def teacher_topk_ids(self) -> list[list[int]]:
        return [[] for _ in self.prompt_ids] + list(self.teacher_token_ids)

    @property
    def teacher_topk_logprobs(self) -> list[list[float]]:
        return [[] for _ in self.prompt_ids] + list(self.teacher_logprobs)
