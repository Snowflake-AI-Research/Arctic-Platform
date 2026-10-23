# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the shared rank assignment and the on-prem splitter."""

import sys
from pathlib import Path

import pytest
import torch

_DSS_ROOT = Path(__file__).resolve().parents[3] / "dss-monorepo" / "cortex" / "neutrino" / "dss-backend"
if _DSS_ROOT.is_dir() and str(_DSS_ROOT) not in sys.path:
    sys.path.insert(0, str(_DSS_ROOT))

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
    groups = prompt_groups(
        input_ids, response_length=1, max_token_len=100, max_group_length=2
    )
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


def _load_cortex_balanced_batch():
    """Import Cortex batching without ``gpu/__init__``, which pulls the sampling stack."""
    import types

    name = "dss.ray_dss.jobs.gpu"
    parts = name.split(".")
    for index in range(len(parts)):
        package = ".".join(parts[: index + 1])
        if package in sys.modules:
            continue
        module = types.ModuleType(package)
        module.__path__ = [str(_DSS_ROOT.joinpath(*package.split(".")))]
        module.__package__ = package
        sys.modules[package] = module
    from dss.ray_dss.jobs.gpu.batching import BalancedBatch

    return BalancedBatch


def test_cortex_assignment_matches_onprem():
    BalancedBatch = _load_cortex_balanced_batch()
    import dss.ray_dss.jobs.gpu.batching as batching

    assign = batching._assign_balanced_batch()
    assert assign.__code__.co_filename.endswith(
        "Arctic-Platform/arctic_platform/common/utils/balanced_batch.py"
    )
    assert "dss-monorepo/Arctic-Platform" not in assign.__code__.co_filename

    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]],
        dtype=torch.long,
    )
    rows = torch.arange(32).view(4, 8)
    envelope = {
        "batch": {"input_ids": rows.clone(), "attention_mask": mask.clone()},
        "meta": {},
        "processing": {"loss_fn": "verl_grpo"},
    }
    _, onprem = _split_batch(envelope, num_workers=2)
    cortex = BalancedBatch({"input_ids": rows, "attention_mask": mask}, 2)
    assert onprem == _flat(cortex._assignment)

    input_ids = torch.tensor([[1, 1, 9], [1, 1, 8], [2, 2, 7], [2, 2, 6]])
    attention_mask = torch.ones_like(input_ids)
    meta = {
        "zorro_train_enable": True,
        "max_response_len": 1,
        "max_token_len_per_gpu": 100,
        "rollout_n": 2,
    }
    zorro_rows = input_ids.clone()
    _, onprem_zorro = _split_batch(
        {
            "batch": {"input_ids": zorro_rows, "attention_mask": attention_mask.clone()},
            "meta": dict(meta),
            "processing": {"loss_fn": "verl_grpo"},
        },
        num_workers=2,
    )
    cortex_zorro = BalancedBatch(
        {"input_ids": input_ids, "attention_mask": attention_mask, "context": meta},
        2,
    )
    assert onprem_zorro == _flat(cortex_zorro._assignment)
    per_rank = [input_ids[rank] for rank in cortex_zorro._assignment]
    restored = cortex_zorro.restore_tensor(per_rank)
    assert torch.equal(restored, input_ids)


def _reorg_global_batch():
    """Load ``reorg_global_batch`` without importing the RL package.

    That package init requires the ``[sft]`` extra, and this module imports
    ``tensordict`` only for a different function. The dev interpreter has neither.
    """
    import importlib.util
    import types

    import arctic_platform

    root = Path(arctic_platform.__file__).resolve().parent
    created = []

    def install(name, module):
        if name not in sys.modules:
            sys.modules[name] = module
            created.append(name)

    tensordict = types.ModuleType("tensordict")
    tensordict.TensorDict = dict
    install("tensordict", tensordict)

    rl = types.ModuleType("arctic_platform.rl")
    rl.__path__ = [str(root / "rl")]
    rl.__package__ = "arctic_platform.rl"
    install("arctic_platform.rl", rl)

    utils = types.ModuleType("arctic_platform.rl.utils")
    utils.__path__ = [str(root / "rl" / "utils")]
    utils.__package__ = "arctic_platform.rl.utils"
    install("arctic_platform.rl.utils", utils)

    debug = types.ModuleType("arctic_platform.rl.utils.debug")

    def _silent(*_args, **_kwargs):
        return None

    class _Timer:
        def __init__(self, *_args, **_kwargs):
            pass

    debug.pr = _silent
    debug.pr0 = _silent
    debug.SynchronizedWallClockTimerSimple = _Timer
    debug.SynchronizedWallClockTimerSimpleDummy = _Timer
    install("arctic_platform.rl.utils.debug", debug)

    path = root / "rl" / "zorro_train" / "seqlen_balancing.py"
    spec = importlib.util.spec_from_file_location("reorg_global_batch_under_test", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name in created:
            sys.modules.pop(name, None)
    return module.reorg_global_batch


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


def test_prompt_ids_keep_a_left_padded_group_that_reorg_splits():
    reorg_global_batch = _reorg_global_batch()
    input_ids, attention_mask = _left_padded_prompt_batch()
    position_ids = torch.arange(10).expand(4, 10).contiguous()
    _, old_indices = reorg_global_batch(
        {
            "input_ids": input_ids.clone(),
            "attention_mask": attention_mask.clone(),
            "position_ids": position_ids.clone(),
        },
        response_length=1,
        world_size=2,
        max_token_len=100,
        max_group_length_threshold=2,
    )
    old_ranks = _rank_sets(old_indices, 2)
    assert {0, 1} not in old_ranks
    assert {2, 3} not in old_ranks

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
    reorg_global_batch = _reorg_global_batch()
    input_ids = torch.arange(6).view(3, 2)
    attention_mask = torch.ones(3, 2, dtype=torch.long)
    position_ids = torch.arange(2).expand(3, 2).contiguous()
    with pytest.raises(ValueError, match="shouldn't reach here"):
        reorg_global_batch(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
                "position_ids": position_ids.clone(),
            },
            response_length=1,
            world_size=2,
            max_token_len=16,
            max_group_length_threshold=2,
        )

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


def test_reorg_indexes_a_non_row_value_and_the_splitter_keeps_it():
    reorg_global_batch = _reorg_global_batch()
    input_ids, attention_mask = _left_padded_prompt_batch()
    position_ids = torch.arange(10).expand(4, 10).contiguous()
    with pytest.raises(TypeError, match="not subscriptable"):
        reorg_global_batch(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
                "position_ids": position_ids.clone(),
                "pad_token_id": 7,
            },
            response_length=1,
            world_size=2,
            max_token_len=100,
            max_group_length_threshold=2,
        )

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


def test_cortex_forwards_explicit_balance_batch_true():
    _load_cortex_balanced_batch()
    import dss.ray_dss.jobs.gpu.batching as batching

    seen = {}
    real = batching._assign_balanced_batch()
    original = batching._assign_balanced_batch

    def capture(num_shards, meta, **kwargs):
        seen["balance_batch"] = meta.get("balance_batch", "missing")
        return real(num_shards, meta, **kwargs)

    batching._assign_balanced_batch = lambda: capture
    try:
        mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.long)
        balanced = batching.BalancedBatch(
            {"attention_mask": mask, "context": {"balance_batch": True}},
            2,
        )
    finally:
        batching._assign_balanced_batch = original

    assert seen["balance_batch"] is True
    assert balanced._assignment is not None


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
