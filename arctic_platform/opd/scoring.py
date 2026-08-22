# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Token-aligned teacher scoring for on-policy distillation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def _logprob_at(position: Any, token_id: int) -> float:
    if not isinstance(position, dict):
        raise RuntimeError(f"expected prompt_logprobs position dict, got {type(position).__name__}")
    entry = position.get(token_id)
    if entry is None:
        entry = position.get(str(token_id))
    if entry is None:
        raise RuntimeError(f"token {token_id} is absent from its prompt_logprobs position")
    return float(entry["logprob"] if isinstance(entry, dict) else entry)


def score_teacher(
    client: Any,
    rollouts: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score exact student token IDs with the teacher, without retokenization."""
    full_batch = [list(r["prompt_ids"]) + list(r["completion_ids"]) for r in rollouts]
    outputs = client.generate_teacher(
        full_batch,
        {"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": 0},
    )
    if len(outputs) != len(full_batch):
        raise RuntimeError(f"teacher returned {len(outputs)} results for {len(full_batch)} prompts")

    scored: list[dict[str, Any]] = []
    for rollout, full_ids, output in zip(rollouts, full_batch, outputs):
        prompt_logprobs = output.get("prompt_logprobs")
        if prompt_logprobs is None:
            raise RuntimeError("teacher response has no prompt_logprobs")
        if len(prompt_logprobs) != len(full_ids):
            raise RuntimeError(
                f"prompt_logprobs length {len(prompt_logprobs)} != token length {len(full_ids)}"
            )
        if prompt_logprobs[0] is not None:
            raise RuntimeError("prompt_logprobs[0] must be None")
        prompt_len = len(rollout["prompt_ids"])
        teacher_logprobs = [
            _logprob_at(prompt_logprobs[i], full_ids[i])
            for i in range(prompt_len, len(full_ids))
        ]
        if any(not math.isfinite(value) or value > 0.0 for value in teacher_logprobs):
            raise RuntimeError("teacher returned a non-finite or positive logprob")
        scored_rollout = dict(rollout)
        scored_rollout["teacher_logprobs"] = teacher_logprobs
        scored.append(scored_rollout)
    return scored
