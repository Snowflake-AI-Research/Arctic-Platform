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

"""The expert offsets handed to a grouped matmul may not reach past the rows the dispatch delivered.

``torch._grouped_mm`` reads rows ``offsets[i-1]:offsets[i]`` of its operand for expert ``i``, so the final
offset asserts a length. In the expert-parallel forward the operand is the permuted receive buffer and the
offsets come from per-expert counts, and if those two are produced by different code they can disagree: the
receive buffer carries routing slots marked ``-1`` that the permutation drops, while the dispatch counts
still include them. The kernel then reads memory past the tensor and folds it into that expert's output.

That read is corruption first and a crash second. On a four-rank fixture at hidden size 512 the offsets
reached row 128 of a 125-row operand on every run, and the device assert that names it fired on about one
run in ten; the other nine produced expert outputs built partly from adjacent allocations, silently. So the
property is asserted here on the counts themselves rather than by observing that a run did not abort.
"""

from __future__ import annotations

import pytest
import torch

from arctic_platform.model.implementations.moe.layers.moe import _run_experts_grouped_mm_impl
from arctic_platform.model.implementations.qwen35.distributed.token_permute import permute_tokens

NUM_LOCAL_EXPERTS = 4
HIDDEN = 8


def test_permutation_reports_counts_that_sum_to_the_rows_it_produced():
    """The counts must be derived from the same mask that selected the rows, not taken from the dispatch.

    Three of the twelve routing slots below are ``-1``: entries the receive buffer has room for and no token
    behind. The permutation keeps nine rows. Any count vector that sums to more than nine describes an
    operand this tensor is not, and the grouped matmul indexes it as though it were.
    """
    dispatched_indices = torch.tensor(
        [[0, -1], [1, 2], [-1, -1], [3, 0], [2, -1], [1, 3], [0, 2]],
        dtype=torch.int64,
    )
    dispatched_scores = torch.arange(dispatched_indices.numel(), dtype=torch.float32).reshape(dispatched_indices.shape)
    hidden_states = torch.arange(dispatched_indices.shape[0] * HIDDEN, dtype=torch.float32).reshape(
        dispatched_indices.shape[0], HIDDEN
    )

    permuted, _scores, permuted_indices, num_tokens_per_expert = permute_tokens(
        hidden_states, dispatched_indices, dispatched_scores, NUM_LOCAL_EXPERTS
    )

    delivered = int((dispatched_indices != -1).sum())
    assert (
        permuted.shape[0] == delivered
    ), f"the permutation produced {permuted.shape[0]} rows for {delivered} delivered routing slots"
    assert permuted_indices.shape[0] == delivered
    assert num_tokens_per_expert.numel() == NUM_LOCAL_EXPERTS, (
        f"one count per local expert is what becomes the offsets; got {num_tokens_per_expert.numel()} for "
        f"{NUM_LOCAL_EXPERTS} experts"
    )
    assert int(num_tokens_per_expert.sum()) == permuted.shape[0], (
        f"the per-expert counts sum to {int(num_tokens_per_expert.sum())} while the permuted operand has "
        f"{permuted.shape[0]} rows. The grouped matmul reads rows offsets[i-1]:offsets[i] of that operand, so "
        f"a larger sum reads past its end; counts={num_tokens_per_expert.tolist()}"
    )

    # Segment boundaries must also line up with the sort, not merely add up: a count vector with the right
    # total and the wrong split sends an expert its neighbour's rows, which no total can detect.
    expert_of_row = dispatched_indices[dispatched_indices != -1].sort(stable=True).values
    for expert, count in enumerate(num_tokens_per_expert.tolist()):
        assert int((expert_of_row == expert).sum()) == count, (
            f"expert {expert} holds {int((expert_of_row == expert).sum())} of the sorted rows but its count "
            f"says {count}"
        )


def test_the_grouped_matmul_refuses_offsets_that_reach_past_its_operand():
    """The guard at the call site is the last thing between a wrong count vector and a read past the tensor.

    It is asserted separately from the derivation above because the two protect against different failures:
    the derivation keeps this rank's counts honest, and the guard catches any future caller whose counts come
    from somewhere else.
    """
    rows = 5
    x = torch.zeros((rows, HIDDEN))
    w1 = torch.zeros((2, HIDDEN, HIDDEN))
    w2 = torch.zeros((2, HIDDEN, HIDDEN))
    w3 = torch.zeros((2, HIDDEN, HIDDEN))
    overrunning = torch.tensor([2, 4], dtype=torch.int64)

    with pytest.raises(AssertionError) as excinfo:
        _run_experts_grouped_mm_impl(w1, w2, w3, x, overrunning)

    message = str(excinfo.value)
    assert "6" in message and str(rows) in message, (
        "the error must name how far the offsets reach and how many rows exist, so the caller that produced "
        f"the counts can be identified; got: {message}"
    )
    assert overrunning.tolist().__str__() in message, f"the error must carry the count vector itself; got: {message}"
