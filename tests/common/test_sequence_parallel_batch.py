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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Logical SP batch preparation only; no native models, ranks or token partition."""

from copy import deepcopy
from types import ModuleType

import pytest
import torch

from arctic_platform.common.utils.batch import _split_batch
from arctic_platform.common.utils.batch import restore_batch_order
from arctic_platform.common.utils.batch import sequence_parallel_worker_groups
from arctic_platform.common.utils.server_models import JobConfig
from arctic_platform.common.utils.server_models import sp_size_from_job_config
from arctic_platform.testing_utils import torch_assert_equal


def _membership(groups=((0, 2), (1, 3)), order=(0, 1, 2, 3)):
    by_rank = {}
    for group in groups:
        for local_rank, rank in enumerate(group):
            by_rank[rank] = {
                "rank": rank,
                "world_size": len(order),
                "sp_size": len(group),
                "sp_rank": local_rank,
                "sp_ranks": list(group),
            }
    return [by_rank[rank] for rank in order]


def _envelope(rows=5):
    return {
        "batch": {
            "input_ids": torch.arange(rows * 4).reshape(rows, 4),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]).repeat(rows, 1),
            "labels": torch.arange(rows * 4).reshape(rows, 4),
        },
        "meta": {"drop_position_ids": True},
        "processing": {"loss_fn": "sft_ce"},
    }


def test_sp_preparation_resolves_public_worker_alias_without_wire_mutation():
    request = {"ds_worker_config": {"sp_size": 8}, "ds_config": {"sequence_parallel_size": 8}}
    original = deepcopy(request)
    assert sp_size_from_job_config(request) == 8
    config = JobConfig(model_name="synthetic", **request)
    assert config.sp_size == 8
    assert sp_size_from_job_config(config.model_dump()) == 8
    assert config.model_dump()["ds_worker_config"] == original["ds_worker_config"]
    assert request == original


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "8"])
def test_sp_preparation_rejects_invalid_public_worker_alias(value):
    with pytest.raises(ValueError):
        JobConfig(model_name="synthetic", ds_worker_config={"sp_size": value})


@pytest.mark.parametrize(
    "other",
    [
        {"ds_worker_config": {"sp_size": 8, "sequence_parallel": 4}},
        {"ds_worker_config": {"sp_size": 8, "sequence_parallel_size": 4}},
        {"ds_worker_config": {"sp_size": 8, "parallelism": {"sequence_parallel": 4}}},
        {"ds_worker_config": {"sp_size": 8}, "training_config": {"sp_size": 4}},
        {"ds_worker_config": {"sp_size": 8}, "ds_config": {"sequence_parallel_size": 4}},
    ],
)
def test_sp_preparation_rejects_conflicting_public_worker_alias(other):
    with pytest.raises(ValueError, match="conflicting"):
        JobConfig(model_name="synthetic", **other)


def test_sp_preparation_group_mapping_uses_ordered_runtime_ranks_not_worker_positions():
    info = _membership(groups=((2, 0), (3, 1)), order=(1, 2, 3, 0))
    assert sequence_parallel_worker_groups(info, 4, 2) == [[2, 0], [1, 3]]
    source = _envelope()
    shards, reorder = _split_batch(source, 4, 2, worker_sp_info=info)
    assert reorder is None
    for index in (2, 0):
        torch_assert_equal(shards[index]["batch"]["input_ids"], source["batch"]["input_ids"][:3])
    for index in (1, 3):
        torch_assert_equal(shards[index]["batch"]["input_ids"], source["batch"]["input_ids"][3:])
    assert all(shard["meta"]["dp_size"] == 2 for shard in shards)
    assert "dp_size" not in source["meta"]
    assert "position_ids" not in source["batch"]
    torch_assert_equal(shards[0]["batch"]["position_ids"], torch.tensor([[0, 1, 2, 0]]).repeat(3, 1))
    assert shards[0]["batch"] is not shards[2]["batch"]


def test_sp_preparation_one_logical_row_can_feed_one_complete_sp_group():
    info = _membership(groups=((0, 1, 2, 3),))
    shards, _ = _split_batch(_envelope(rows=1), 4, 4, worker_sp_info=info)
    assert all(shard["meta"]["dp_size"] == 1 for shard in shards)
    assert all(shard["batch"]["input_ids"].shape == (1, 4) for shard in shards)


def test_sp_preparation_gas_preserves_per_microbatch_labels_and_reconstructs_positions():
    source = _envelope()
    second = _envelope(rows=3)["batch"]
    second["input_ids"] += 100
    source["batch"] = [source["batch"], second]
    shards, reorder = _split_batch(source, 4, 2, worker_sp_info=_membership())
    assert reorder is None
    assert all(len(shard["batch"]) == 2 for shard in shards)
    for group, row_slice in (((0, 2), slice(0, 3)), ((1, 3), slice(3, None))):
        for rank in group:
            torch_assert_equal(shards[rank]["batch"][0]["labels"], source["batch"][0]["labels"][row_slice])
    torch_assert_equal(shards[0]["batch"][1]["input_ids"], second["input_ids"][:2])
    torch_assert_equal(shards[1]["batch"][1]["input_ids"], second["input_ids"][2:])
    assert all("position_ids" not in mb for mb in source["batch"])
    assert all("position_ids" in mb for shard in shards for mb in shard["batch"])


def test_sp_preparation_balances_logical_dp_rows_and_preserves_inverse_reorder(monkeypatch):
    import sys

    module = ModuleType("arctic_platform.rl.zorro_train.seqlen_balancing")
    source = _envelope(rows=4)
    order = [2, 0, 3, 1]
    observed = []

    def balance(batch, **kwargs):
        observed.append(kwargs)
        assert "position_ids" in batch
        return {name: value[order] for name, value in batch.items()}, order

    module.reorg_global_batch = balance
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source["meta"].update(load_balancer=True, rollout_n=1, max_response_len=4, max_token_len_per_gpu=8)
    shards, reorder = _split_batch(source, 4, 2, worker_sp_info=_membership())
    assert observed[0]["world_size"] == 2
    assert reorder == order
    logical = {"input_ids": torch.cat([shards[0]["batch"]["input_ids"], shards[1]["batch"]["input_ids"]])}
    torch_assert_equal(restore_batch_order(logical, reorder)["input_ids"], source["batch"]["input_ids"])


@pytest.mark.parametrize(
    "fault",
    ["count", "world", "size", "local", "duplicate_rank", "duplicate_member", "order", "overlap", "bool"],
)
def test_sp_preparation_rejects_incomplete_or_contradictory_runtime_membership(fault):
    info = _membership()
    if fault == "count":
        info.pop()
    elif fault == "world":
        info[0]["world_size"] = 8
    elif fault == "size":
        info[0]["sp_size"] = 1
    elif fault == "local":
        info[0]["sp_rank"] = 1
    elif fault == "duplicate_rank":
        info[1] = deepcopy(info[0])
    elif fault == "duplicate_member":
        info[0]["sp_ranks"] = [0, 0]
    elif fault == "order":
        info[2]["sp_ranks"] = [2, 0]
        info[2]["sp_rank"] = 0
    elif fault == "overlap":
        info[1]["sp_ranks"] = [1, 2]
    else:
        info[0]["rank"] = False
    with pytest.raises(ValueError):
        _split_batch(_envelope(), 4, 2, worker_sp_info=info)


def test_sp_preparation_unbound_native_caller_still_fails_before_input_processing():
    with pytest.raises(ValueError, match="actual worker SP membership"):
        _split_batch({}, 4, 2)


def test_sp_preparation_rejects_microbatch_smaller_than_logical_dp():
    with pytest.raises(ValueError, match="Batch dimension"):
        _split_batch(_envelope(rows=1), 4, 2, worker_sp_info=_membership())


def test_sp_preparation_dense_default_keeps_disjoint_rows_and_world_scaling():
    source = _envelope(rows=5)
    shards, reorder = _split_batch(source, 4)
    assert reorder is None
    assert all(shard["meta"]["dp_size"] == 4 for shard in shards)
    torch_assert_equal(torch.cat([shard["batch"]["input_ids"] for shard in shards]), source["batch"]["input_ids"])
