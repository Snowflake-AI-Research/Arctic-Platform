# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Adapter: Harbor's on-disk trial output -> arctic ``RolloutDataset``.

Reads every ``{trial_dir}/result.json`` under a Harbor ``jobs/`` directory,
pulls ``agent_result.rollout_details`` (Harbor's own ``RolloutDetail`` shape:
``prompt_token_ids: list[list[int]]``, ``completion_token_ids: list[list[int]]``,
optional ``logprobs``) plus the verifier's ``reward``, and materializes them
as ``Rollout`` / ``RolloutDataset`` for the arctic GRPO backend.
"""

from __future__ import annotations

import json
from pathlib import Path

from arctic_platform.integrations.harbor.models import Rollout, RolloutDataset


def _flat_first_turn(per_turn: list[list[int]] | None) -> list[int]:
    """Harbor stores per-turn token IDs (list[list[int]]). Our GRPO batch is
    single-turn: take the first turn's IDs. A future multi-turn integration
    would concatenate or fold turn breaks into the loss mask."""
    if not per_turn:
        return []
    return list(per_turn[0])


def load_job_dir(
    job_dir: Path,
    dataset_id: str,
    model_name: str,
    tokenizer_name: str | None = None,
) -> RolloutDataset:
    """Load every trial under a Harbor job dir into a single RolloutDataset."""
    job_dir = Path(job_dir)
    rollouts: list[Rollout] = []

    for result_path in sorted(job_dir.glob("**/result.json")):
        try:
            data = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        agent_result = data.get("agent_result") or {}
        details = agent_result.get("rollout_details") or []
        if not details:
            continue

        # Group id = the task this rollout belongs to (rollouts of the same
        # task form a GRPO group). Harbor writes ``task_name`` at the top level
        # of every trial result.json — that's exactly one task per problem.
        group_id = data.get("task_name") or result_path.parent.name

        verifier_result = data.get("verifier_result") or {}
        rewards = verifier_result.get("rewards") or {}
        # A verifier can emit several named rewards. If the primary "reward"
        # key is present (canonical convention), use it; otherwise fall back
        # to the first value. Auxiliary metrics like ``any_correct`` /
        # ``terse_correct`` are kept as-is on ``metadata`` so the driver can
        # report multiple pass rates.
        if "reward" in rewards:
            reward = float(rewards["reward"])
        elif rewards:
            reward = float(next(iter(rewards.values())))
        else:
            reward = 0.0

        # Harbor may write more than one RolloutDetail if there are subagents.
        # The first entry is always the main agent by convention.
        d = details[0]
        prompt_ids = _flat_first_turn(d.get("prompt_token_ids"))
        completion_ids = _flat_first_turn(d.get("completion_token_ids"))
        if not prompt_ids or not completion_ids:
            continue

        rollouts.append(
            Rollout(
                prompt_token_ids=prompt_ids,
                completion_token_ids=completion_ids,
                reward=reward,
                group_id=group_id,
                metadata={
                    "trial_dir": str(result_path.parent),
                    "trial_name": result_path.parent.name,
                    # Preserve every reward field the verifier emitted so eval
                    # can compute additional pass rates (e.g. any_correct,
                    # terse_correct) without re-reading result.json.
                    "rewards": {k: float(v) for k, v in rewards.items()},
                },
            )
        )

    return RolloutDataset(
        rollouts=rollouts,
        dataset_id=dataset_id,
        model_name=model_name,
        tokenizer_name=tokenizer_name or model_name,
    )


def pass_at_1(ds: RolloutDataset) -> float:
    """Fraction of rollouts that scored the full training reward (1.0)."""
    if not ds.rollouts:
        return 0.0
    correct = sum(1 for r in ds.rollouts if r.reward >= 1.0)
    return correct / len(ds.rollouts)


def metric_at_1(ds: RolloutDataset, key: str) -> float:
    """Fraction of rollouts whose verifier emitted ``rewards[key] == 1``.

    Used for reporting metrics the verifier tracks but doesn't necessarily
    train on (e.g. ``any_correct`` — the raw arithmetic score — when the
    training reward folds in a second signal like brevity).
    """
    if not ds.rollouts:
        return 0.0
    n_hit = 0
    for r in ds.rollouts:
        rewards = (r.metadata or {}).get("rewards") or {}
        if rewards.get(key, 0.0) >= 1.0:
            n_hit += 1
    return n_hit / len(ds.rollouts)
