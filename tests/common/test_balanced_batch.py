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

"""CPU tests for the shared rank assignment and the on-prem splitter."""

import pytest
import torch

from arctic_platform.common.utils.balanced_batch import assign_row_lpt
from arctic_platform.common.utils.balanced_batch import assign_zorro_groups
from arctic_platform.common.utils.balanced_batch import linear_dedup_costs
from arctic_platform.common.utils.balanced_batch import prompt_groups
from arctic_platform.common.utils.batch import _split_batch


def _flat(assignment):
    return [row for rows in assignment for row in rows]


def test_row_lpt_fills_floor_then_sorts():
    assignment = assign_row_lpt([100 * 100, 90 * 90, 10 * 10, 10 * 10], 2)
    assert assignment == [[0, 3], [1, 2]]


def test_row_lpt_sorts_indices_that_were_inserted_out_of_order():
    assignment = assign_row_lpt([1, 100, 50, 1], 2)
    assert assignment == [[1, 3], [0, 2]]


def test_row_lpt_uses_floor_then_ceil_when_the_batch_does_not_divide():
    assignment = assign_row_lpt([5, 4, 3, 2, 1], 2)
    assert assignment == [[0, 3, 4], [1, 2]]


def test_zorro_groups_stay_intact_and_adjacent():
    groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    costs = [10.0, 40.0, 30.0, 20.0]
    assignment = assign_zorro_groups(groups, costs, num_shards=2, num_rows=8)
    assert assignment == [[2, 3, 0, 1], [4, 5, 6, 7]]


def test_zorro_groups_backtrack_when_greedy_dead_ends():
    groups = [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9], [10, 11]]
    assignment = assign_zorro_groups(groups, [1.0] * 5, num_shards=2, num_rows=12)
    assert {frozenset(rows) for rows in assignment} == {frozenset(range(6)), frozenset(range(6, 12))}


def test_zorro_thousands_of_single_row_groups_assign():
    groups = [[row] for row in range(2048)]
    costs = [float(row % 7) for row in range(2048)]
    for num_shards in (2, 8):
        assignment = assign_zorro_groups(groups, costs, num_shards=num_shards, num_rows=2048)
        assert sorted(_flat(assignment)) == list(range(2048))
        assert {len(rank) for rank in assignment} == {2048 // num_shards}


def test_zorro_infeasible_hundreds_of_groups_raise():
    # Each rank holds 100 groups of 3 and has one spare row, so the 301st group
    # cannot land. Equal sizes prove that without running the search.
    groups = [[3 * index, 3 * index + 1, 3 * index + 2] for index in range(301)]
    with pytest.raises(ValueError, match="do not fit") as raised:
        assign_zorro_groups(groups, [1.0] * 301, num_shards=3, num_rows=903)
    assert "budget" not in str(raised.value)


def test_zorro_proven_infeasible_groups_do_not_fit():
    # Three pairs cannot pack into two ranks of three rows, and the search is small enough to finish.
    groups = [[0, 1], [2, 3], [4, 5]]
    with pytest.raises(ValueError, match="do not fit") as raised:
        assign_zorro_groups(groups, [1.0, 1.0, 1.0], num_shards=2, num_rows=6)
    assert "budget" not in str(raised.value)


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


def test_onprem_sample_split_matches_assign_row_lpt():
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]],
        dtype=torch.long,
    )
    rows = torch.arange(32).view(4, 8)
    envelope = {
        "batch": {"input_ids": rows.clone(), "attention_mask": mask},
        "meta": {},
        "processing": {"loss_fn": "verl_grpo"},
    }
    shards, reorder = _split_batch(envelope, num_workers=2)
    lengths = mask.sum(dim=-1).tolist()
    expected = _flat(assign_row_lpt([length * length for length in lengths], 2))
    assert reorder == expected
    restored = {"input_ids": torch.cat([shard["batch"]["input_ids"] for shard in shards], dim=0)}
    from arctic_platform.common.utils.batch import restore_batch_order

    restore_batch_order(restored, reorder)
    assert torch.equal(restored["input_ids"], rows)


def test_onprem_zorro_split_matches_assign_zorro_groups():
    # Two prompts, two responses each. Response width is 1.
    input_ids = torch.tensor([[1, 1, 9], [1, 1, 8], [2, 2, 7], [2, 2, 6]])
    attention_mask = torch.ones_like(input_ids)
    original = input_ids.clone()
    envelope = {
        "batch": {"input_ids": input_ids.clone(), "attention_mask": attention_mask.clone()},
        "meta": {
            "zorro_train_enable": True,
            "max_response_len": 1,
            "max_token_len_per_gpu": 100,
            "rollout_n": 2,
        },
        "processing": {"loss_fn": "verl_grpo"},
    }
    shards, reorder = _split_batch(envelope, num_workers=2)
    groups = prompt_groups(input_ids, response_length=1, max_token_len=100, max_group_length=2)
    costs = linear_dedup_costs(attention_mask, groups, prompt_len=2)
    expected = assign_zorro_groups(groups, costs, 2, 4)
    assert reorder == _flat(expected)
    # Each original prompt pair stays on one rank.
    placed = {tuple(sorted(rank)) for rank in expected}
    assert placed == {(0, 1), (2, 3)}
    restored = {"input_ids": torch.cat([shard["batch"]["input_ids"] for shard in shards], dim=0)}
    from arctic_platform.common.utils.batch import restore_batch_order

    restore_batch_order(restored, reorder)
    assert torch.equal(restored["input_ids"], original)


def test_per_row_lists_follow_the_tensor_assignment():
    rows = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]])
    shards, _order = _split_batch(
        {
            "batch": {"input_ids": rows.clone(), "attention_mask": rows.clone(), "uids": ["a", "b", "c", "d"]},
            "meta": {"balance_batch": True},
            "processing": {"loss_fn": "grpo"},
        },
        num_workers=2,
    )
    by_uid = {"a": rows[0], "b": rows[1], "c": rows[2], "d": rows[3]}
    seen = []
    for shard in shards:
        ids = shard["batch"]["input_ids"]
        uids = shard["batch"]["uids"]
        assert len(uids) == int(ids.shape[0])
        for index, uid in enumerate(uids):
            assert torch.equal(ids[index], by_uid[uid])
            seen.append(uid)
    assert sorted(seen) == ["a", "b", "c", "d"]


def _rank_sets(indices, num_workers):
    width = len(indices) // num_workers
    return [set(indices[rank * width : (rank + 1) * width]) for rank in range(num_workers)]


def _left_padded_prompt_batch():
    """Four rows, two logical prompts. Prompt columns differ inside each prompt."""
    input_ids = torch.arange(40).view(4, 10)
    attention_mask = torch.zeros(4, 10, dtype=torch.long)
    attention_mask[0, :10] = 1
    attention_mask[1, :3] = 1
    attention_mask[2, :9] = 1
    attention_mask[3, :1] = 1
    return input_ids, attention_mask


def test_prompt_ids_keep_a_left_padded_group_together():
    input_ids, attention_mask = _left_padded_prompt_batch()

    _, reorder = _split_batch(
        {
            "batch": {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
                "prompt_group_ids": torch.tensor([0, 0, 1, 1]),
            },
            "meta": {
                "zorro_train_enable": True,
                "max_response_len": 1,
                "max_token_len_per_gpu": 100,
                "rollout_n": 2,
            },
            "processing": {"loss_fn": "verl_grpo"},
        },
        num_workers=2,
    )
    new_ranks = _rank_sets(reorder, 2)
    assert {0, 1} in new_ranks
    assert {2, 3} in new_ranks


def test_non_divisible_zorro_batch_names_the_failure():
    input_ids = torch.arange(6).view(3, 2)
    attention_mask = torch.ones(3, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="divisible"):
        _split_batch(
            {
                "batch": {"input_ids": input_ids.clone(), "attention_mask": attention_mask.clone()},
                "meta": {
                    "zorro_train_enable": True,
                    "max_response_len": 1,
                    "max_token_len_per_gpu": 16,
                    "rollout_n": 2,
                },
                "processing": {"loss_fn": "verl_grpo"},
            },
            num_workers=2,
        )


def test_splitter_keeps_a_non_row_value():
    input_ids, attention_mask = _left_padded_prompt_batch()

    shards, reorder = _split_batch(
        {
            "batch": {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
                "prompt_group_ids": torch.tensor([0, 0, 1, 1]),
                "pad_token_id": 7,
            },
            "meta": {
                "zorro_train_enable": True,
                "max_response_len": 1,
                "max_token_len_per_gpu": 100,
                "rollout_n": 2,
            },
            "processing": {"loss_fn": "verl_grpo"},
        },
        num_workers=2,
    )
    assert {0, 1} in _rank_sets(reorder, 2)
    assert {2, 3} in _rank_sets(reorder, 2)
    assert [shard["batch"]["pad_token_id"] for shard in shards] == [7, 7]


def test_stamp_and_apply_keep_explicit_true():
    from arctic_platform.common.balance_metadata import apply_balance_metadata
    from arctic_platform.common.balance_metadata import stamp_balance_meta

    prompts = torch.tensor([[1, 1], [1, 1], [2, 2], [2, 2]])
    meta = {"load_balancer": True, "zorro_train_enable": True, "rollout_n": 2}
    stamp_balance_meta(meta, balance_batch=True, zorro_train_enable=True, prompts=prompts, rollout_n=2)
    assert meta["balance_batch"] is True
    assert "load_balancer" not in meta
    assert meta["prompt_group_ids"].tolist() == [0, 0, 1, 1]

    context = {}
    apply_balance_metadata(context, meta, prompts, prompts)
    assert context["balance_batch"] is True
    assert context["prompt_group_ids"].tolist() == [0, 0, 1, 1]

    unstamped = {}
    apply_balance_metadata(
        unstamped,
        {"zorro_train_enable": True, "rollout_n": 2, "balance_batch": True},
        prompts,
        prompts,
    )
    assert "prompt_group_ids" not in unstamped

    off = {"prompt_group_ids": meta["prompt_group_ids"]}
    stamp_balance_meta(off, balance_batch=False, zorro_train_enable=True, prompts=None, rollout_n=2)
    assert off["balance_batch"] is False
    assert "prompt_group_ids" not in off
    applied = {"prompt_group_ids": meta["prompt_group_ids"]}
    apply_balance_metadata(applied, off, prompts, prompts)
    assert applied["balance_batch"] is False
    assert "prompt_group_ids" not in applied
    kept = {}
    apply_balance_metadata(
        kept,
        {"balance_batch": False, "prompt_group_ids": meta["prompt_group_ids"]},
        prompts,
        prompts,
    )
    assert kept["balance_batch"] is False
    assert kept["prompt_group_ids"].tolist() == [0, 0, 1, 1]


def test_string_zorro_flag_is_not_group_balance():
    from arctic_platform.common.utils.balanced_batch import assign_balanced_batch

    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]])
    assignment = assign_balanced_batch(2, {"zorro_train_enable": "false"}, attention_mask=mask)
    assert assignment is not None


def test_legacy_balancer_keys_are_ignored():
    from arctic_platform.common.utils.balanced_batch import assign_balanced_batch

    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]])
    for key in ("load_balancer", "zorro_load_balancer", "zorro_train_load_balancer"):
        assert assign_balanced_batch(2, {key: False}, attention_mask=mask) is not None
    assert assign_balanced_batch(2, {"balance_batch": False, "load_balancer": True}, attention_mask=mask) is None


def test_reordered_prompt_blocks_omit_group_ids():
    from arctic_platform.common.balance_metadata import stamp_balance_meta

    prompts = torch.tensor([[1, 1], [2, 2], [1, 1], [2, 2]])
    meta = {}
    stamp_balance_meta(meta, balance_batch=True, zorro_train_enable=True, prompts=prompts, rollout_n=2)
    assert meta["balance_batch"] is True
    assert "prompt_group_ids" not in meta


def test_client_balance_uses_dense_prompts_and_falls_back_for_jagged_rows():
    from arctic_platform.common.balance_metadata import stamp_client_balance

    dense = torch.tensor([[1, 1], [1, 1], [2, 2], [2, 2]])
    meta = {"zorro_train_enable": True, "rollout_n": 2}
    stamp_client_balance(
        meta,
        balance_batch=True,
        zorro_train_enable=True,
        rollout_n=2,
        prompts=dense,
        input_ids=torch.zeros((4, 3), dtype=torch.long),
        max_prompt_len=1,
    )
    assert meta["balance_batch"] is True
    assert meta["prompt_group_ids"].tolist() == [0, 0, 1, 1]

    jagged = torch.nested.nested_tensor(
        [torch.tensor([1, 1]), torch.tensor([1, 1]), torch.tensor([9, 9]), torch.tensor([9, 9])],
        layout=torch.jagged,
    )
    input_ids = torch.tensor([[1, 1, 5], [1, 1, 6], [2, 2, 7], [2, 2, 8]])
    jagged_meta = {"zorro_train_enable": True}
    stamp_client_balance(
        jagged_meta,
        balance_batch=True,
        zorro_train_enable=True,
        rollout_n=2,
        prompts=jagged,
        input_ids=input_ids,
        max_prompt_len=2,
    )
    assert jagged_meta["prompt_group_ids"].tolist() == [0, 0, 1, 1]

    positional = {"prompt_group_ids": dense.new_zeros(4)}
    stamp_client_balance(positional, balance_batch=False, zorro_train_enable=True, rollout_n=2, prompts=dense)
    assert positional["balance_batch"] is False
    assert "prompt_group_ids" not in positional


def test_stamped_group_ids_keep_left_padded_rows_together_onprem():
    """Ids come from the client stamp on meta. On-prem promotes them onto the batch."""
    from arctic_platform.common.balance_metadata import stamp_balance_meta

    prompts = torch.tensor([[1, 1], [1, 1], [2, 2], [2, 2]])
    meta = {
        "zorro_train_enable": True,
        "max_response_len": 1,
        "max_token_len_per_gpu": 100,
        "rollout_n": 2,
    }
    stamp_balance_meta(meta, balance_batch=True, zorro_train_enable=True, prompts=prompts, rollout_n=2)
    input_ids = torch.tensor([[1, 1, 9], [8, 1, 7], [2, 2, 6], [5, 2, 4]])
    _, reorder = _split_batch(
        {
            "batch": {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)},
            "meta": meta,
            "processing": {"loss_fn": "verl_grpo"},
        },
        num_workers=2,
    )
    assert {0, 1} in _rank_sets(reorder, 2)
    assert {2, 3} in _rank_sets(reorder, 2)


_TWO_PROMPT_LENGTHS = [90, 85, 80, 75, 20, 15, 10, 5]
_TWO_PROMPT_IDS = [0, 0, 0, 0, 1, 1, 1, 1]


def test_sample_balance_keeps_each_prompt_group_on_one_rank():
    from arctic_platform.common.utils.balanced_batch import assign_balanced_batch

    ids = torch.tensor(_TWO_PROMPT_IDS)
    from_batch = assign_balanced_batch(2, {}, lengths=_TWO_PROMPT_LENGTHS, prompt_group_ids=ids)
    from_meta = assign_balanced_batch(2, {"prompt_group_ids": ids}, lengths=_TWO_PROMPT_LENGTHS)
    assert from_batch == from_meta == [[0, 1, 2, 3], [4, 5, 6, 7]]
    for rows in from_batch:
        assert len({_TWO_PROMPT_IDS[row] for row in rows}) == 1


def test_prompt_mean_loss_matches_one_rank_after_the_split():
    from arctic_platform.rl.processors.functional import agg_loss

    lengths = _TWO_PROMPT_LENGTHS
    width = max(lengths)
    attention_mask = (torch.arange(width).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)).long()
    loss_mat = torch.linspace(-1.0, 1.0, len(lengths) * width).view(len(lengths), width)
    ids = torch.tensor(_TWO_PROMPT_IDS)
    shards, reorder = _split_batch(
        {
            "batch": {
                "input_ids": torch.arange(len(lengths) * width).view(len(lengths), width),
                "attention_mask": attention_mask.clone(),
                "loss_mat": loss_mat.clone(),
                "prompt_group_ids": ids.clone(),
            },
            "meta": {},
            "processing": {"loss_fn": "grpo"},
        },
        num_workers=2,
    )
    assert reorder is not None
    shard_ids = [shard["batch"]["prompt_group_ids"] for shard in shards]
    assert sum(int(torch.unique(rank_ids).numel()) for rank_ids in shard_ids) == 2

    expected = agg_loss(loss_mat, attention_mask, "prompt-mean", global_batch_size=2, prompt_group_ids=ids)
    # DeepSpeed averages the per-rank losses, and each rank scales by dp_size.
    per_rank = [
        agg_loss(
            shard["batch"]["loss_mat"],
            shard["batch"]["attention_mask"],
            "prompt-mean",
            dp_size=2,
            global_batch_size=2,
            prompt_group_ids=shard["batch"]["prompt_group_ids"],
        )
        for shard in shards
    ]
    torch.testing.assert_close(sum(per_rank) / 2, expected, rtol=0, atol=1e-6)


def test_prompt_groups_balance_whole_groups_by_cost():
    from arctic_platform.common.utils.balanced_batch import assign_balanced_batch

    lengths = [10, 10, 9, 9, 1, 1, 2, 2]
    ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    assert assign_balanced_batch(2, {}, lengths=lengths, prompt_group_ids=ids) == [[0, 1, 4, 5], [2, 3, 6, 7]]


def test_interleaved_prompt_group_rows_stay_together():
    from arctic_platform.common.utils.balanced_batch import assign_balanced_batch

    ids = [0, 1, 0, 1, 2, 3, 2, 3]
    assignment = assign_balanced_batch(2, {}, lengths=[9, 1, 8, 2, 7, 3, 6, 4], prompt_group_ids=torch.tensor(ids))
    assert sorted(_flat(assignment)) == list(range(8))
    for rows in assignment:
        assert len(rows) == 4
        for group_id in {ids[row] for row in rows}:
            assert {row for row, other in enumerate(ids) if other == group_id} <= set(rows)


def test_prompt_groups_that_cannot_fill_equal_ranks_keep_arrival_order():
    from arctic_platform.common.utils.balanced_batch import assign_balanced_batch

    three_pairs = torch.tensor([0, 0, 1, 1, 2, 2])
    assert assign_balanced_batch(2, {}, lengths=[6, 5, 4, 3, 2, 1], prompt_group_ids=three_pairs) is None
    odd_rows = torch.tensor([0, 0, 1, 1, 2])
    assert assign_balanced_batch(2, {}, lengths=[5, 4, 3, 2, 1], prompt_group_ids=odd_rows) is None


def test_unusable_prompt_group_ids_keep_arrival_order():
    from arctic_platform.common.utils.balanced_batch import assign_balanced_batch

    lengths = [4, 3, 2, 1]
    assert assign_balanced_batch(2, {}, lengths=lengths, prompt_group_ids=torch.tensor([0, 0, 1])) is None
    assert assign_balanced_batch(2, {}, lengths=lengths, prompt_group_ids=torch.tensor([[0, 0], [1, 1]])) is None
