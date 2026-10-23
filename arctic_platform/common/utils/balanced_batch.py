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

``assign_balanced_batch`` is the splitter entry point. It keeps arrival order,
balances sample lengths, or balances whole ZoRRo prompt groups. The functions
return row indices and do not mutate the batch. This module does not import
the RL package.
"""

from __future__ import annotations

from typing import Optional, Sequence


def assign_balanced_batch(
    num_shards: int,
    meta: dict,
    *,
    input_ids=None,
    attention_mask=None,
    lengths: Optional[Sequence[int]] = None,
    prompt_group_ids=None,
) -> Optional[list[list[int]]]:
    """Assign rows for one request.

    ``balance_batch`` false keeps arrival order (``None``). ``True`` and a missing
    key both balance. An explicit legacy ``load_balancer`` / ``zorro_load_balancer``
    / ``zorro_train_load_balancer`` false also keeps arrival order when
    ``balance_batch`` itself is absent. ZoRRo balances whole prompt groups. Every
    other request balances individual sample lengths, using ``lengths`` when the
    caller measured them and ``attention_mask`` otherwise.
    """
    if _balance_disabled(meta):
        return None
    if meta.get("zorro_train_enable"):
        return _assign_prompt_groups(
            num_shards,
            meta,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_group_ids=prompt_group_ids if prompt_group_ids is not None else meta.get("prompt_group_ids"),
        )
    measured = lengths
    if measured is None:
        measured = _mask_lengths(attention_mask)
    if measured is None or len(measured) < num_shards:
        return None
    costs = [int(length) * int(length) for length in measured]
    return assign_row_lpt(costs, num_shards)


def _balance_disabled(meta: dict) -> bool:
    """True when this request opted out of balancing.

    ``balance_batch`` wins. When it is absent, an explicit false on a legacy
    balancer key still opts out. A missing key balances.
    """
    if meta.get("balance_batch") is False:
        return True
    if "balance_batch" in meta:
        return False
    return any(
        meta.get(key) is False
        for key in ("load_balancer", "zorro_load_balancer", "zorro_train_load_balancer")
    )


def _mask_lengths(attention_mask) -> Optional[list[int]]:
    import torch

    if not torch.is_tensor(attention_mask) or attention_mask.ndim != 2:
        return None
    return [int(length) for length in attention_mask.to(torch.long).sum(dim=-1).tolist()]


def _assign_prompt_groups(
    num_shards: int,
    meta: dict,
    *,
    input_ids,
    attention_mask,
    prompt_group_ids,
) -> Optional[list[list[int]]]:
    import torch

    response_length = meta.get("max_response_len")
    max_token_len = meta.get("max_token_len_per_gpu")
    max_group = meta.get("zorro_train_max_rollouts")
    if max_group is None:
        max_group = meta.get("rollout_n")
    if response_length is None or max_token_len is None or max_group is None:
        raise ValueError(
            "ZoRRo balancing requires max_response_len, max_token_len_per_gpu, "
            "and rollout_n or zorro_train_max_rollouts"
        )
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2 or not torch.is_tensor(attention_mask):
        raise ValueError("ZoRRo balancing requires input_ids and attention_mask")
    groups = prompt_groups(
        input_ids,
        response_length=int(response_length),
        max_token_len=int(max_token_len),
        max_group_length=int(max_group),
        prompt_group_ids=prompt_group_ids,
    )
    prompt_len = int(input_ids.shape[1]) - int(response_length)
    costs = linear_dedup_costs(attention_mask, groups, prompt_len)
    return assign_zorro_groups(groups, costs, num_shards, int(input_ids.shape[0]))


def assign_row_lpt(costs: Sequence[int], num_shards: int) -> list[list[int]]:
    """Cardinality-constrained LPT, then rows are sorted within each rank.

    Rank cardinalities follow ``torch.tensor_split``: the first ``n % num_shards``
    ranks get ceil rows and the rest get floor, so the bins line up with the DP cut
    that ``split_dict`` makes later. ``costs`` is one non-negative cost per row.
    Ties break toward the lower rank.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    n = len(costs)
    if n < num_shards:
        raise ValueError(f"batch_size={n} is smaller than num_shards={num_shards}")

    order = sorted(range(n), key=lambda idx: -costs[idx])
    min_rows = n // num_shards
    remainder = n % num_shards
    # Match torch.tensor_split: the first ``remainder`` ranks receive one extra row.
    # Pinning each rank to its final cardinality up front keeps the greedy pass from
    # filling every rank to ``min_rows`` first, which strands the large rows together.
    caps = [min_rows + (1 if rank < remainder else 0) for rank in range(num_shards)]
    assignment: list[list[int]] = [[] for _ in range(num_shards)]
    shard_costs = [0] * num_shards
    for idx in order:
        candidates = [rank for rank, indices in enumerate(assignment) if len(indices) < caps[rank]]
        if not candidates:
            raise RuntimeError(
                f"Unable to assign row {idx}: batch_size={n}, num_shards={num_shards}, caps={caps}"
            )
        rank = min(candidates, key=lambda candidate: (shard_costs[candidate], candidate))
        assignment[rank].append(idx)
        shard_costs[rank] += costs[idx]
    for rank_indices in assignment:
        rank_indices.sort()
    return assignment


def _pack_groups(sizes, costs, order, num_shards, cap):
    """Place groups in ``order``, cheapest rank first, backtracking on a dead end.

    The first pass is the greedy min-cost placement, so the result matches the
    greedy pass whenever that pass fits every group. Backtracking only runs after a
    dead end. Ranks with the same row count and cost are interchangeable, so a
    dead end skips ranks that match one already tried. Returns ``{group: rank}``
    or ``None``.
    """
    rows = [0] * num_shards
    rank_costs = [0.0] * num_shards
    placed: dict[int, int] = {}

    def place(pos: int) -> bool:
        if pos == len(order):
            return True
        index = order[pos]
        tried = set()
        for rank in sorted(range(num_shards), key=lambda candidate: (rank_costs[candidate], candidate)):
            state = (rows[rank], rank_costs[rank])
            if rows[rank] + sizes[index] > cap or state in tried:
                continue
            tried.add(state)
            saved_rows, saved_cost = rows[rank], rank_costs[rank]
            rows[rank] += sizes[index]
            rank_costs[rank] += float(costs[index])
            placed[index] = rank
            if place(pos + 1):
                return True
            rows[rank], rank_costs[rank] = saved_rows, saved_cost
            del placed[index]
        return False

    return placed if place(0) else None


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
    for index in order:
        size = len(groups[index])
        if size > cap:
            raise ValueError(
                f"prompt group of {size} rows does not fit in a rank cap of {cap} "
                f"(batch_size={num_rows}, num_shards={num_shards})"
            )
    placed = _pack_groups([len(group) for group in groups], costs, order, num_shards, cap)
    if placed is None:
        raise ValueError(
            f"prompt groups do not fit into {num_shards} ranks of {cap} rows "
            f"(batch_size={num_rows}, num_shards={num_shards})"
        )
    bins: list[list[int]] = [[] for _ in range(num_shards)]
    for index in order:
        bins[placed[index]].extend(list(groups[index]))
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
