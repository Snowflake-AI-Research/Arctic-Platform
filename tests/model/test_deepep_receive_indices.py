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

"""The receive indices ``permute_tokens`` reads hold what DeepEP writes into them: ``-1`` and local expert ids.

DeepEP allocates the receive indices with ``torch::empty`` and fills a receive row only by copying it out of a
sender's channel buffer, where the sender has already replaced each global expert id with a local id or ``-1``
(DeepEP 1.2.1+73b6ea4: ``csrc/deep_ep.cpp:553``, ``csrc/kernels/intranode.cu:369-373`` and ``:496``). A value below
``-1`` therefore came from no sender of that dispatch: the row was never written, or was overwritten afterwards, and
``torch.bincount`` is the first operation that rejects it. The row's hidden state comes from the same receive row, so
masking the index to ``-1`` would not recover a token; it would pass whatever that memory held through the experts
and the combine.

A context-parallel shard can deliver no rows at all, or rows whose every slot belongs to another rank. Neither is an
error; both count zero tokens for every local expert.
"""

import re

import pytest
import torch

from arctic_platform.model.implementations.moe.distributed.token_permute import permute_tokens
from arctic_platform.testing_utils import torch_assert_equal

NUM_LOCAL_EXPERTS = 4
NUM_TOPK = 2
HIDDEN = 8
BINCOUNT_ERROR = "bincount only supports 1-d non-negative integral inputs."


@pytest.mark.parametrize("unwritten", [-2, torch.iinfo(torch.int64).min], ids=["minus-two", "int64-min"])
def test_receive_index_below_minus_one_raises_the_bincount_error(unwritten):
    """One slot below ``-1`` among valid rows raises the error the expert-parallel forward stops with."""
    dispatched_indices = torch.tensor([[0, -1], [3, 1], [unwritten, 2]], dtype=torch.int64)
    hidden_states = torch.zeros(dispatched_indices.shape[0], HIDDEN)
    dispatched_scores = torch.ones(dispatched_indices.shape)

    with pytest.raises(RuntimeError, match=re.escape(BINCOUNT_ERROR)):
        permute_tokens(hidden_states, dispatched_indices, dispatched_scores, NUM_LOCAL_EXPERTS)


@pytest.mark.parametrize("rows", [0, 3], ids=["no-rows", "all-minus-one"])
def test_receive_buffer_without_local_slots_counts_zero_for_every_expert(rows):
    """No delivered rows, or rows whose every slot belongs to another rank, permute to nothing and count zero."""
    dispatched_indices = torch.full((rows, NUM_TOPK), -1, dtype=torch.int64)

    permuted, permuted_scores, permuted_indices, num_tokens_per_expert = permute_tokens(
        torch.zeros(rows, HIDDEN), dispatched_indices, torch.zeros(rows, NUM_TOPK), NUM_LOCAL_EXPERTS
    )

    torch_assert_equal(num_tokens_per_expert, torch.zeros(NUM_LOCAL_EXPERTS, dtype=torch.int64))
    assert permuted.shape == (0, HIDDEN)
    assert permuted_scores.numel() == 0 and permuted_indices.numel() == 0
