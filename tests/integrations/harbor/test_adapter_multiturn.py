# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-turn flattening for Harbor rollouts.

Covers ``_flatten_multi_turn`` — the piece that turns Harbor's per-turn
``RolloutDetail`` (``list[list[int]]`` for prompt and completion) into the
flat ``(prompt, completion, loss_mask)`` an arctic GRPO batch expects.

Cases:

* Single-turn (Harbor sends N=1) still works — loss mask matches the current
  turn's completion.
* Multi-turn with the standard Harbor invariant (each turn's prompt starts
  with the prior turn's prompt+completion) — every model-produced token
  across all turns is marked trainable.
* Invariant-break (agent's chat template re-tokenizes) — the flatten falls
  back to ``loss_mask=None``, i.e. train only on the final turn's completion.
* Empty / missing inputs — degrade to empty outputs instead of raising.
"""

from __future__ import annotations

import json
from pathlib import Path

from arctic_platform.integrations.harbor.adapter import (
    _flatten_multi_turn,
    load_job_dir,
)


def test_single_turn_matches_final_completion():
    prompt = [[1, 2, 3, 4]]
    completion = [[10, 11]]
    p, c, mask = _flatten_multi_turn(prompt, completion)
    assert p == [1, 2, 3, 4]
    assert c == [10, 11]
    # 4 prompt tokens (mask 0) + 2 completion tokens (mask 1).
    assert mask == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]


def test_two_turns_invariant_holds_marks_all_model_tokens():
    # Turn 0: prompt = [1,2], completion = [10,11]
    # Turn 1: prompt = [1,2,10,11,20], completion = [30,31]
    prompt = [[1, 2], [1, 2, 10, 11, 20]]
    completion = [[10, 11], [30, 31]]
    p, c, mask = _flatten_multi_turn(prompt, completion)
    assert p == [1, 2, 10, 11, 20]
    assert c == [30, 31]
    # Positions 2..4 inside p correspond to turn 0's completion (trainable);
    # positions 5..6 (inside c) are turn 1's completion (trainable).
    #                                  1    2    10   11   20   30   31
    assert mask ==                    [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]


def test_three_turns_invariant_holds():
    # T0: prompt=[1], completion=[9]
    # T1: prompt=[1,9,2], completion=[8]        (added "2" as user reply)
    # T2: prompt=[1,9,2,8,3], completion=[7]    (added "3" as user reply)
    prompt = [[1], [1, 9, 2], [1, 9, 2, 8, 3]]
    completion = [[9], [8], [7]]
    p, c, mask = _flatten_multi_turn(prompt, completion)
    assert p == [1, 9, 2, 8, 3]
    assert c == [7]
    #             1    9    2    8    3    7
    assert mask == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]


def test_invariant_break_falls_back_to_final_only():
    # Turn 1's prompt doesn't match turn 0's prompt+completion — as if the
    # chat template rewrote tokens between turns.
    prompt = [[1, 2], [99, 99, 99, 20]]
    completion = [[10, 11], [30, 31]]
    p, c, mask = _flatten_multi_turn(prompt, completion)
    assert p == [99, 99, 99, 20]
    assert c == [30, 31]
    # loss_mask dropped; downstream trainer masks with a "final completion
    # only" default (the safe, always-correct subset).
    assert mask is None


def test_empty_inputs_return_empty():
    assert _flatten_multi_turn(None, None) == ([], [], None)
    assert _flatten_multi_turn([], []) == ([], [], None)
    assert _flatten_multi_turn([[1, 2]], []) == ([], [], None)


def _write_trial(
    trial_dir: Path,
    *,
    task_name: str,
    prompt_per_turn: list[list[int]],
    completion_per_turn: list[list[int]],
    reward: float,
) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": task_name,
                "agent_result": {
                    "rollout_details": [
                        {
                            "prompt_token_ids": prompt_per_turn,
                            "completion_token_ids": completion_per_turn,
                        }
                    ],
                },
                "verifier_result": {"rewards": {"reward": reward}},
            }
        )
    )


def test_load_job_dir_populates_loss_mask_for_multi_turn(tmp_path):
    _write_trial(
        tmp_path / "trial-0",
        task_name="mul-1",
        prompt_per_turn=[[1, 2], [1, 2, 10, 11, 20]],
        completion_per_turn=[[10, 11], [30, 31]],
        reward=1.0,
    )
    _write_trial(
        tmp_path / "trial-1",
        task_name="mul-1",
        prompt_per_turn=[[5, 6]],
        completion_per_turn=[[7]],
        reward=0.0,
    )
    ds = load_job_dir(tmp_path, dataset_id="test", model_name="dummy/model")
    assert len(ds.rollouts) == 2

    multi = next(r for r in ds.rollouts if r.group_id == "mul-1" and r.reward == 1.0)
    assert multi.prompt_token_ids == [1, 2, 10, 11, 20]
    assert multi.completion_token_ids == [30, 31]
    assert multi.loss_mask == [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    assert multi.metadata["n_turns"] == 2

    single = next(r for r in ds.rollouts if r.reward == 0.0)
    assert single.loss_mask == [0.0, 0.0, 1.0]
    assert single.metadata["n_turns"] == 1
