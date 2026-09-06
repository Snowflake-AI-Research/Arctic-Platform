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

"""Token-aligned teacher scoring for on-policy distillation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def _logprob_value(entry: Any) -> float:
    return float(entry["logprob"] if isinstance(entry, dict) else entry)


def _logprob_at(position: Any, token_id: int) -> float:
    if not isinstance(position, dict):
        raise RuntimeError(f"expected prompt_logprobs position dict, got {type(position).__name__}")
    entry = position.get(token_id)
    if entry is None:
        entry = position.get(str(token_id))
    if entry is None:
        raise RuntimeError(f"token {token_id} is absent from its prompt_logprobs position")
    return _logprob_value(entry)


def _position_candidates(position: Any) -> list[tuple[int, float]]:
    if not isinstance(position, dict):
        raise RuntimeError(f"expected prompt_logprobs position dict, got {type(position).__name__}")
    pairs: list[tuple[int, float]] = []
    for key, entry in position.items():
        pairs.append((int(key), _logprob_value(entry)))
    pairs.sort(key=lambda item: -item[1])
    return pairs


def _tail_logprob(logprobs: Sequence[float]) -> float:
    mass = sum(math.exp(value) for value in logprobs)
    remainder = max(1.0 - mass, 0.0)
    if remainder <= 0.0:
        return float("-inf")
    return math.log(remainder)


def _teacher_outputs(
    client: Any,
    rollouts: Sequence[dict[str, Any]],
    *,
    prompt_logprobs: int,
    temperature: float,
) -> tuple[list[list[int]], list[Any]]:
    full_batch = [list(r["prompt_ids"]) + list(r["completion_ids"]) for r in rollouts]
    outputs = client.generate_teacher(
        full_batch,
        {"max_tokens": 1, "temperature": temperature, "prompt_logprobs": prompt_logprobs},
    )
    if len(outputs) != len(full_batch):
        raise RuntimeError(f"teacher returned {len(outputs)} results for {len(full_batch)} prompts")
    return full_batch, outputs


def _require_prompt_logprobs(output: dict[str, Any], full_ids: list[int]) -> list[Any]:
    prompt_logprobs = output.get("prompt_logprobs")
    if prompt_logprobs is None:
        raise RuntimeError("teacher response has no prompt_logprobs")
    if len(prompt_logprobs) != len(full_ids):
        raise RuntimeError(f"prompt_logprobs length {len(prompt_logprobs)} != token length {len(full_ids)}")
    if prompt_logprobs[0] is not None:
        raise RuntimeError("prompt_logprobs[0] must be None")
    return prompt_logprobs


def score_teacher(
    client: Any,
    rollouts: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score exact student token IDs with the teacher, without retokenization."""
    full_batch, outputs = _teacher_outputs(client, rollouts, prompt_logprobs=0, temperature=1.0)

    scored: list[dict[str, Any]] = []
    for rollout, full_ids, output in zip(rollouts, full_batch, outputs):
        prompt_logprobs = _require_prompt_logprobs(output, full_ids)
        prompt_len = len(rollout["prompt_ids"])
        teacher_logprobs = [_logprob_at(prompt_logprobs[i], full_ids[i]) for i in range(prompt_len, len(full_ids))]
        if any(not math.isfinite(value) or value > 0.0 for value in teacher_logprobs):
            raise RuntimeError("teacher returned a non-finite or positive logprob")
        scored_rollout = dict(rollout)
        scored_rollout["teacher_logprobs"] = teacher_logprobs
        scored.append(scored_rollout)
    return scored


def score_teacher_topk(
    client: Any,
    rollouts: Sequence[dict[str, Any]],
    *,
    teacher_top_k: int = 8,
    teacher_temperature: float = 1.0,
    add_tail_bucket: bool = True,
) -> list[dict[str, Any]]:
    """Teacher-forced top-k support on the student's token ids.

    Requests ``prompt_logprobs=teacher_top_k``. vLLM reports the realized token
    even when it falls outside top-k. Optional tail is ``log(1 - sum exp)``.
    """
    if teacher_top_k < 1:
        raise ValueError("teacher_top_k must be >= 1")
    full_batch, outputs = _teacher_outputs(
        client, rollouts, prompt_logprobs=teacher_top_k, temperature=teacher_temperature
    )

    scored: list[dict[str, Any]] = []
    for rollout, full_ids, output in zip(rollouts, full_batch, outputs):
        prompt_logprobs = _require_prompt_logprobs(output, full_ids)
        prompt_len = len(rollout["prompt_ids"])
        token_ids: list[list[int]] = []
        token_logprobs: list[list[float]] = []
        tail: list[float] = []
        for index in range(prompt_len, len(full_ids)):
            realized = full_ids[index]
            pairs = _position_candidates(prompt_logprobs[index])
            if realized not in {token_id for token_id, _ in pairs}:
                raise RuntimeError(f"token {realized} is absent from its prompt_logprobs position")
            ids = [token_id for token_id, _ in pairs]
            lps = [logprob for _, logprob in pairs]
            if any(not math.isfinite(value) or value > 0.0 for value in lps):
                raise RuntimeError("teacher returned a non-finite or positive logprob")
            token_ids.append(ids)
            token_logprobs.append(lps)
            if add_tail_bucket:
                tail.append(_tail_logprob(lps))
        scored_rollout = dict(rollout)
        scored_rollout["teacher_token_ids"] = token_ids
        scored_rollout["teacher_logprobs"] = token_logprobs
        if add_tail_bucket:
            scored_rollout["teacher_tail_logprob"] = tail
        scored.append(scored_rollout)
    return scored
