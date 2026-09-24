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
"""Row bookkeeping for the expert-parallel receive buffer.

These functions are pure tensor code and live apart from ``deepep.py`` because that module imports the
``deep_ep`` extension at module scope. The invariant they carry -- that the reported per-expert counts describe
exactly the rows the permutation produced -- is what keeps the grouped matmul's final offset inside its
operand, and it has to be assertable wherever tests run, including hosts without the extension built.
"""

import torch

from ..token_combine import sum_rows_by_token


def permute_tokens(
    hidden_states: torch.Tensor,
    dispatched_indices: torch.Tensor,
    dispatched_scores: torch.Tensor,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort the delivered rows by expert, and report how many rows each expert got.

    The expert segmentation has to be derived from the rows that survive the ``-1`` mask, not taken from the
    counts the dispatch reported. A receive buffer carries slots for routing entries this rank did not receive
    a token for; those are marked ``-1`` and dropped here, while the dispatch counts still include them. The
    two then disagree by the number of dropped slots, and since the counts become the ``offs`` argument of a
    grouped matrix multiply whose operand is this permuted tensor, the last offset points past the end of it.
    Deriving both from one mask makes the disagreement unrepresentable.
    """
    mask = dispatched_indices != -1
    valid_expert_ids = dispatched_indices[mask]
    valid_scores = dispatched_scores[mask]

    sort_order = torch.argsort(valid_expert_ids, stable=True)
    permuted_indices = torch.arange(len(hidden_states), device=hidden_states.device).repeat_interleave(
        mask.sum(dim=1)
    )[sort_order]
    permuted_hidden_states = hidden_states.index_select(0, permuted_indices)
    permuted_scores = valid_scores[sort_order]

    num_tokens_per_expert = torch.bincount(valid_expert_ids, minlength=num_local_experts)
    if num_tokens_per_expert.numel() != num_local_experts:
        # bincount widens to hold the largest id, so a longer result means the recv-side indices are not the
        # local, zero-based expert ids the sort above already assumes them to be.
        raise ValueError(
            f"dispatch returned expert ids up to {int(valid_expert_ids.max())} for a rank holding "
            f"{num_local_experts} experts; the permutation and the grouped matmul both read these as local "
            "indices, so a wider range means the recv-side index space changed"
        )
    return permuted_hidden_states, permuted_scores, permuted_indices, num_tokens_per_expert


def unpermute_tokens(
    permuted_hidden_states: torch.Tensor,
    permuted_indices: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    return sum_rows_by_token(permuted_hidden_states, permuted_indices, num_tokens)
