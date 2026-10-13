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

"""SP logprob reconstruction preserves request rows within every DP replica."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.common.utils.batch import flatten_sequence_rows
from arctic_platform.common.utils.batch import merge_sp_dict_shards
from arctic_platform.common.utils.batch import ray_split_batch
from arctic_platform.common.utils.batch import unpack_batch


def forward():
    # Execute the real server method without starting Ray or CUDA on a CPU test host.
    path = Path(__file__).resolve().parents[2] / "arctic_platform/common/ray_server.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ArcticRLRayServer")
    definition = next(node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "forward")
    definition.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {
        "torch": torch,
        "ray": SimpleNamespace(get=lambda refs: refs),
        "prepare_request_loss": lambda batch: None,
        "unpack_batch": unpack_batch,
        "ray_split_batch": ray_split_batch,
        "flatten_sequence_rows": flatten_sequence_rows,
        "merge_sp_dict_shards": merge_sp_dict_shards,
        "restore_batch_order": lambda values, order: values,
    }
    module = ast.fix_missing_locations(ast.Module(body=[future, definition], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["forward"]


@pytest.mark.parametrize(
    "dp,sp,rows,width",
    [(1, 1, 1, 3), (2, 1, 5, 3), (1, 2, 1, 3), (1, 2, 3, 5), (2, 2, 5, 3), (2, 8, 33, 7), (4, 2, 9, 5), (2, 8, 2, 3)],
)
@pytest.mark.parametrize("envelope", ["ap", "cortex", "drop_positions"])
@pytest.mark.parametrize("job_type", ["training", "log_prob"])
def test_logprobs_restore_every_request_row(dp, sp, rows, width, envelope, job_type):
    calls = []

    class Remote:
        def __init__(self, rank):
            self.rank = rank

        def remote(self, batch):
            calls.append((self.rank, batch))
            if sp > 1 or rows == 1:
                assert batch["batch"]["input_ids"].shape[0] == 1
            assert batch["meta"]["dp_size"] == dp
            return batch["batch"]["labels"].float()

    server = SimpleNamespace(
        jobs={1: {"job_type": job_type, "sp_size": sp}},
        training_workers=[SimpleNamespace(compute_log_probs=Remote(rank)) for rank in range(dp * sp)],
        log_prob_workers=[SimpleNamespace(compute_log_probs=Remote(rank)) for rank in range(dp * sp)],
        _verify_job=lambda *args: None,
    )
    labels = torch.arange(rows * width).reshape(rows, width) + 10
    positions = torch.arange(width).expand(rows, -1).clone()
    request = {
        "batch": {
            "input_ids": labels.clone(),
            "position_ids": positions,
            "attention_mask": torch.ones_like(labels),
            "labels": labels.clone(),
        },
        "meta": {"labels_are_shifted": True},
        "processing": {"loss_fn": "sft", "return_logprobs": True},
    }
    if envelope == "cortex":
        request = {"kwargs": request["batch"], "context": request["meta"], "processing": request["processing"]}
    elif envelope == "drop_positions":
        del request["batch"]["position_ids"]
        request["meta"]["drop_position_ids"] = True

    response = asyncio.run(forward()(server, 1, request))

    torch.testing.assert_close(response["logprobs"], labels.float(), rtol=0, atol=0)
    assert response["job_id"] == 1
    assert [rank for rank, _ in calls] == list(range(dp * sp))
    for replica, row_shard in enumerate(torch.tensor_split(labels, dp, dim=0)):
        shards = [batch["batch"] for _, batch in calls[replica * sp : (replica + 1) * sp]]
        if sp == 1:
            torch.testing.assert_close(shards[0]["labels"], row_shard, rtol=0, atol=0)
            continue
        length = row_shard.numel()
        torch.testing.assert_close(
            torch.cat([batch["labels"] for batch in shards], dim=1)[:, :length],
            row_shard.reshape(1, -1),
            rtol=0,
            atol=0,
        )
        expected_positions = torch.arange(width).expand(row_shard.shape[0], -1).reshape(1, -1)
        torch.testing.assert_close(
            torch.cat([batch["position_ids"] for batch in shards], dim=1)[:, :length],
            expected_positions,
            rtol=0,
            atol=0,
        )
    original = request.get("batch", request.get("kwargs"))
    assert torch.equal(original["labels"], labels)
    assert ("position_ids" in original) is (envelope != "drop_positions")


def test_too_few_rows_rejected_before_worker_calls():
    server = SimpleNamespace(
        jobs={1: {"job_type": "training", "sp_size": 2}},
        training_workers=[SimpleNamespace() for _ in range(8)],
        _verify_job=lambda *args: None,
    )
    request = {
        "batch": {"input_ids": torch.ones(2, 3, dtype=torch.long)},
        "meta": {},
        "processing": {"loss_fn": "sft", "return_logprobs": True},
    }
    with pytest.raises(ValueError, match="at least one sample per DP worker"):
        asyncio.run(forward()(server, 1, request))
