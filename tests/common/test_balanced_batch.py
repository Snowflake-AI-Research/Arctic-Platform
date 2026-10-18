# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the shared rank assignment."""

import torch

from arctic_platform.common.utils.balanced_batch import assign_row_lpt
from arctic_platform.common.utils.balanced_batch import assign_zorro_groups
from arctic_platform.common.utils.balanced_batch import linear_dedup_costs
from arctic_platform.common.utils.balanced_batch import prompt_groups


def test_row_lpt_fills_floor_then_sorts():
    # Costs 100, 90, 10, 10. Longest pairs with a short row, then indices sort.
    assignment = assign_row_lpt([100 * 100, 90 * 90, 10 * 10, 10 * 10], 2)
    assert assignment == [[0, 3], [1, 2]]


def test_row_lpt_sorts_indices_that_were_inserted_out_of_order():
    # Rank 1 receives row 2 before row 0. The final sort is what makes [0, 2].
    assignment = assign_row_lpt([1, 100, 50, 1], 2)
    assert assignment == [[1, 3], [0, 2]]


def test_row_lpt_uses_floor_then_ceil_when_the_batch_does_not_divide():
    assignment = assign_row_lpt([5, 4, 3, 2, 1], 2)
    assert assignment == [[0, 3, 4], [1, 2]]


def test_zorro_groups_stay_intact_and_adjacent():
    groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    costs = [10.0, 40.0, 30.0, 20.0]
    assignment = assign_zorro_groups(groups, costs, num_shards=2, num_rows=8)
    # Highest cost first. Equal bin cost prefers the lower rank.
    assert assignment == [[2, 3, 0, 1], [4, 5, 6, 7]]


def test_zorro_groups_raise_when_the_batch_does_not_divide():
    try:
        assign_zorro_groups([[0, 1], [2]], [1.0, 1.0], num_shards=2, num_rows=3)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected a non-divisible batch to raise")


def test_zorro_groups_raise_when_a_group_exceeds_the_rank_cap():
    try:
        assign_zorro_groups([[0, 1, 2, 3], [4, 5]], [1.0, 1.0], num_shards=2, num_rows=6)
    except ValueError as exc:
        assert "does not fit" in str(exc)
    else:
        raise AssertionError("expected an oversized group to raise")


def test_small_batch_keeps_arrival_order():
    assert assign_zorro_groups([[0], [1]], [1.0, 1.0], num_shards=2, num_rows=2) is None


def test_prompt_ids_outrank_identical_prompt_columns():
    input_ids = torch.tensor([[7, 1], [7, 2], [7, 3], [7, 4]])
    groups = prompt_groups(
        input_ids,
        response_length=1,
        max_token_len=100,
        max_group_length=2,
        prompt_group_ids=torch.tensor([0, 0, 1, 1]),
    )
    assert groups == [[0, 1], [2, 3]]
    mask = torch.ones_like(input_ids)
    assert linear_dedup_costs(mask, groups, prompt_len=1) == [3.0, 3.0]


def test_unusable_prompt_ids_fall_back_to_prompt_columns():
    input_ids = torch.tensor([[7, 1], [7, 2], [8, 3], [8, 4]])
    groups = prompt_groups(
        input_ids,
        response_length=1,
        max_token_len=100,
        max_group_length=4,
        prompt_group_ids=torch.tensor([[0, 0], [1, 1]]),
    )
    assert groups == [[0, 1], [2, 3]]
