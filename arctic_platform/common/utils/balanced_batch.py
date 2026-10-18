# Copyright 2026 Snowflake Inc.
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

"""Rank assignment shared by the on-prem server and Cortex.

The functions return row indices. They do not mutate the batch. Callers measure
lengths and pass costs in, so this module does not import the RL package.
"""

from __future__ import annotations

from typing import Optional, Sequence


def assign_row_lpt(costs: Sequence[int], num_shards: int) -> list[list[int]]:
    """Cardinality-constrained LPT. Each rank gets floor or ceil rows, then rows are sorted.

    ``costs`` is one non-negative cost per row. Ties break toward the lower rank.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    n = len(costs)
    if n < num_shards:
        raise ValueError(f"batch_size={n} is smaller than num_shards={num_shards}")

    order = sorted(range(n), key=lambda idx: -costs[idx])
    min_rows = n // num_shards
    max_rows = (n + num_shards - 1) // num_shards
    assignment: list[list[int]] = [[] for _ in range(num_shards)]
    shard_costs = [0] * num_shards
    for idx in order:
        if any(len(indices) < min_rows for indices in assignment):
            candidates = [rank for rank, indices in enumerate(assignment) if len(indices) < min_rows]
        else:
            candidates = [rank for rank, indices in enumerate(assignment) if len(indices) < max_rows]
        if not candidates:
            raise RuntimeError(
                f"Unable to assign row {idx}: batch_size={n}, num_shards={num_shards}, max_rows={max_rows}"
            )
        rank = min(candidates, key=lambda candidate: (shard_costs[candidate], candidate))
        assignment[rank].append(idx)
        shard_costs[rank] += costs[idx]
    for rank_indices in assignment:
        rank_indices.sort()
    return assignment


def assign_zorro_groups(
    groups: Sequence[Sequence[int]],
    costs: Sequence[float],
    num_shards: int,
    num_rows: int,
) -> Optional[list[list[int]]]:
    """Pack whole prompt groups into equal-sized rank bins.

    Returns ``None`` when ``num_rows <= num_shards``: there is nothing to balance,
    and the caller keeps arrival order. Raises when the batch is not divisible by
    ``num_shards`` or a group does not fit in one rank. Groups are not split here.
    Within a rank, groups stay in the order they were placed, and each group's
    rows stay in the order they were given.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if len(groups) != len(costs):
        raise ValueError(f"groups and costs differ in length: {len(groups)} vs {len(costs)}")
    if num_rows <= num_shards:
        return None
    if num_rows % num_shards != 0:
        raise ValueError(
            f"ZoRRo group assignment needs batch_size divisible by num_shards; "
            f"got batch_size={num_rows} num_shards={num_shards}"
        )
    covered = [index for group in groups for index in group]
    if sorted(covered) != list(range(num_rows)):
        raise ValueError(
            f"prompt groups must be a permutation of {num_rows} rows, got {len(covered)} indices"
        )

    cap = num_rows // num_shards
    order = sorted(range(len(groups)), key=lambda i: (len(groups[i]), costs[i]), reverse=True)
    bins: list[list[int]] = [[] for _ in range(num_shards)]
    bin_rows = [0] * num_shards
    bin_costs = [0.0] * num_shards
    for index in order:
        group = list(groups[index])
        size = len(group)
        if size > cap:
            raise ValueError(
                f"prompt group of {size} rows does not fit in a rank cap of {cap} "
                f"(batch_size={num_rows}, num_shards={num_shards})"
            )
        ranked = sorted(range(num_shards), key=lambda rank: bin_costs[rank])
        chosen = next((rank for rank in ranked if bin_rows[rank] + size <= cap), None)
        if chosen is None:
            raise ValueError(
                f"prompt group of {size} rows does not fit in any rank "
                f"(batch_size={num_rows}, num_shards={num_shards}, cap={cap})"
            )
        bins[chosen].extend(group)
        bin_rows[chosen] += size
        bin_costs[chosen] += float(costs[index])
    for rank, count in enumerate(bin_rows):
        if count != cap:
            raise ValueError(
                f"rank {rank} has {count} rows after group assignment; expected {cap}"
            )
    return bins


def prompt_groups(
    input_ids,
    *,
    response_length: int,
    max_token_len: int,
    max_group_length: Optional[int],
    prompt_group_ids=None,
) -> list[list[int]]:
    """Group rows by ``prompt_group_ids`` or by identical prompt columns.

    A positive ``max_group_length`` splits a group into contiguous chunks of at
    most that many rows, in the group's existing row order. The token-budget
    check matches ``create_prompt_groups``.
    """
    import torch

    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise ValueError("prompt_groups requires 2D input_ids")
    batch_size, seq_len = input_ids.shape
    if response_length < 0 or response_length > seq_len:
        raise ValueError(
            f"response_length={response_length} is outside sequence width {seq_len}"
        )
    prompt_len = seq_len - response_length
    ids_ok = (
        torch.is_tensor(prompt_group_ids)
        and prompt_group_ids.ndim == 1
        and int(prompt_group_ids.shape[0]) == batch_size
    )
    if ids_ok:
        grouped: dict[int, list[int]] = {}
        for row, group_id in enumerate(prompt_group_ids.reshape(-1).tolist()):
            grouped.setdefault(int(group_id), []).append(row)
        groups = list(grouped.values())
    else:
        prompts = input_ids[:, :prompt_len]
        grouped_prompts: dict[tuple, list[int]] = {}
        for row in range(batch_size):
            key = tuple(prompts[row].tolist())
            grouped_prompts.setdefault(key, []).append(row)
        groups = list(grouped_prompts.values())
    return _split_groups_to_max_length(
        groups,
        prompt_len=prompt_len,
        response_length=response_length,
        max_token_len=max_token_len,
        max_group_length=max_group_length,
    )


def linear_dedup_costs(attention_mask, groups: Sequence[Sequence[int]], prompt_len: int) -> list[float]:
    """Prompt tokens once, plus every response token. ``quadratic_coeff`` is 0."""
    import torch

    if not torch.is_tensor(attention_mask) or attention_mask.ndim != 2:
        raise ValueError("linear_dedup_costs requires a 2D attention_mask")
    costs = []
    width = int(attention_mask.shape[1])
    if prompt_len < 0 or prompt_len > width:
        raise ValueError(f"prompt_len={prompt_len} is outside attention width {width}")
    for group in groups:
        rows = list(group)
        prompt_tokens = attention_mask[rows[0], :prompt_len].sum().item()
        response_tokens = attention_mask[rows, prompt_len:].sum().item()
        costs.append(float(prompt_tokens + response_tokens))
    return costs


def _split_groups_to_max_length(
    groups: list[list[int]],
    *,
    prompt_len: int,
    response_length: int,
    max_token_len: int,
    max_group_length: Optional[int],
) -> list[list[int]]:
    if not groups or max_group_length is None or max_group_length <= 0:
        return groups
    max_group_length_threshold = max_group_length
    max_token_len_effective = prompt_len + response_length * max_group_length_threshold
    if max_token_len < max_token_len_effective:
        raise ValueError(
            f"""
            {max_token_len=} is smaller than {max_token_len_effective}.
            Where max_token_len_effective = {prompt_len=} + {response_length=} * {max_group_length_threshold=}.
            You can either raise actor_rollout_ref.actor.ppo_max_token_len_per_gpu to a higher value or reduce the number of rollouts (`actor_rollout_ref.rollout.n`) or to split the prompt groups by setting `arctic_rl.train.zorro_train.max_rollouts` to a smaller value than `actor_rollout_ref.rollout.n`, so that the math above checks out.
            Reducing prompt and/or response sizes is another way if it doesn't break the training needs."""
        )
    if max(len(group) for group in groups) <= max_group_length:
        return groups
    split: list[list[int]] = []
    for group in groups:
        if len(group) > max_group_length:
            for start in range(0, len(group), max_group_length):
                split.append(group[start : start + max_group_length])
        else:
            split.append(group)
    return split
