# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Adapter: Harbor's on-disk trial output -> arctic ``RolloutDataset``.

Reads every ``{trial_dir}/result.json`` under a Harbor ``jobs/`` directory,
pulls ``agent_result.rollout_details`` (Harbor's ``RolloutDetail`` shape:
``prompt_token_ids: list[list[int]]``, ``completion_token_ids: list[list[int]]``,
optional ``logprobs``) plus the verifier's ``reward``, and materializes them
as ``Rollout`` / ``RolloutDataset`` for the arctic GRPO backend.

Multi-turn agents produce N (prompt, completion) pairs. The adapter
flattens each trial into a single (prompt, completion, loss_mask) tuple:
prompt = context into the final turn, completion = the final response,
loss_mask = 1.0 on every model-produced position across all turns.
"""

from __future__ import annotations

import json
from pathlib import Path

from arctic_platform.integrations.harbor.models import Rollout, RolloutDataset


def _flatten_multi_turn(
    prompt_per_turn: list[list[int]] | None,
    completion_per_turn: list[list[int]] | None,
) -> tuple[list[int], list[int], list[float] | None]:
    """Flatten per-turn tokens into (prompt, completion, loss_mask).

    Assumes Harbor's invariant that ``prompt[i+1]`` starts with
    ``prompt[i] + completion[i]``. When it holds, the flat sequence is
    ``prompt[-1] + completion[-1]`` and every earlier completion occupies
    a known slice inside ``prompt[-1]``; the returned ``loss_mask`` is 1.0
    on those slices and on the final completion, 0.0 elsewhere.

    When the invariant fails (e.g. a chat template re-tokenizes between
    turns), ``loss_mask`` is returned as ``None`` and the caller trains
    only on the final completion.
    """
    if not prompt_per_turn or not completion_per_turn:
        return [], [], None

    n = min(len(prompt_per_turn), len(completion_per_turn))
    prompt_last = list(prompt_per_turn[-1])
    completion_last = list(completion_per_turn[-1])
    flat_len = len(prompt_last) + len(completion_last)

    # Final turn's completion is always trainable.
    mask = [0.0] * len(prompt_last) + [1.0] * len(completion_last)

    # Walk each earlier turn and mark its completion positions inside
    # prompt_last. If the invariant fails for any turn, drop the mask
    # entirely; final-turn-only training is still valid.
    invariant_holds = True
    for i in range(n - 1):
        p_i = prompt_per_turn[i]
        c_i = completion_per_turn[i]
        # The prior turn's completion should sit at offset len(p_i) in
        # prompt_last (or in prompt_per_turn[i+1], which prompt_last extends).
        start = len(p_i)
        end = start + len(c_i)
        if end > len(prompt_last):
            invariant_holds = False
            break
        # Cheap consistency check: the tokens at [start:end] in prompt_last
        # should equal c_i. If they don't, the agent's chat template
        # rewrote/re-tokenized — abandon the mask.
        if prompt_last[start:end] != list(c_i):
            invariant_holds = False
            break
        for pos in range(start, end):
            mask[pos] = 1.0

    if not invariant_holds:
        mask = None

    return prompt_last, completion_last, mask


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
        prompt_ids, completion_ids, loss_mask = _flatten_multi_turn(
            d.get("prompt_token_ids"),
            d.get("completion_token_ids"),
        )
        if not prompt_ids or not completion_ids:
            continue

        rollouts.append(
            Rollout(
                prompt_token_ids=prompt_ids,
                completion_token_ids=completion_ids,
                loss_mask=loss_mask,
                reward=reward,
                group_id=group_id,
                metadata={
                    "trial_dir": str(result_path.parent),
                    "trial_name": result_path.parent.name,
                    "n_turns": len(d.get("completion_token_ids") or []),
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
