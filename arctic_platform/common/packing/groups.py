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

"""Which rows share a model call, and how to select them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import torch

from .walk import map_batch_values


@dataclass(frozen=True)
class MicrobatchSplit:
    batches: List[Dict]
    restore_order: Optional[List[int]]


def batch_tensor(batch: Dict[str, Any], key: str, *, min_dims: int) -> Optional[torch.Tensor]:
    """A batch's ``key`` tensor, whether it rides at the top level or inside an RL ``context`` sub-dict.

    RL requests nest their per-token tensors, SFT requests do not, and both reach the same planning code, so
    every lookup of a batch's shape has to consider both placements.
    """
    context = batch.get("context")
    candidates = [batch.get(key)]
    if isinstance(context, dict):
        candidates.append(context.get(key))
    for value in candidates:
        if value is not None and torch.is_tensor(value) and value.ndim >= min_dims:
            return value
    return None


def batch_position_ids(batch: Dict[str, Any]) -> Optional[torch.Tensor]:
    """The 2D ``position_ids`` of a batch."""
    position_ids = batch_tensor(batch, "position_ids", min_dims=2)
    if position_ids is not None and position_ids.ndim == 2:
        return position_ids
    return None


def batch_input_ids(batch: Dict[str, Any]) -> Optional[torch.Tensor]:
    """The ``[B, S]`` ``input_ids`` of a batch or shard."""
    return batch_tensor(batch, "input_ids", min_dims=2)


def batch_num_rows(batch: Dict[str, Any]) -> Optional[int]:
    input_ids = batch_input_ids(batch)
    return int(input_ids.shape[0]) if input_ids is not None else None


def batch_seqlen(batch: Dict[str, Any]) -> Optional[int]:
    input_ids = batch_input_ids(batch)
    return int(input_ids.shape[1]) if input_ids is not None else None


def token_validity_mask(batch: Dict[str, Any]) -> torch.Tensor:
    """A 2D ``[B, S]`` mask of the real (non-pad) tokens of every row.

    Requests usually ship ``position_ids`` rather than an ``attention_mask``, since a mask large enough to
    describe a packed batch costs more to move than the positions it encodes. Positions carry the signal
    anyway: a row's real tokens are the leading run whose positions advance by one, so pad tokens (which repeat
    or reset) fall out of the comparison.

    The run has to stop at the first break rather than accept every column that matches: padding holds whatever
    the producer left in it, and a pad value that coincides with ``p0 + i`` would otherwise read as real, which
    bills the row a length its real prefix does not have.
    """
    attention_mask = batch_tensor(batch, "attention_mask", min_dims=2)
    if attention_mask is not None and attention_mask.ndim == 2:
        return attention_mask

    position_ids = batch_position_ids(batch)
    if position_ids is not None and torch.is_tensor(position_ids):
        seq_positions = torch.arange(position_ids.shape[1], device=position_ids.device).unsqueeze(0)
        expected_positions = position_ids[:, :1] + seq_positions
        advances_by_one = position_ids.eq(expected_positions).to(dtype=torch.long)
        return advances_by_one.cumprod(dim=1)

    raise ValueError("token-budget grouping requires a 2D attention_mask or 2D position_ids to find real tokens")


def select_rows(batch: Dict[str, Any], index: torch.Tensor, n_rows: int) -> Dict[str, Any]:
    """Take the rows named by ``index``, in that order, from every row-aligned leaf of ``batch``.

    A leaf counts as row-aligned when its first dimension is ``n_rows``; anything else -- a scalar, a config
    value, a tensor of another shape -- is carried through untouched.
    """

    def selected(_key: str, value: Any) -> Any:
        if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) == n_rows:
            return value.index_select(0, index.to(value.device))
        return value

    return map_batch_values(batch, selected)


def singleton_microbatch_groups(n_rows: int) -> List[List[int]]:
    """One row per microbatch, preserving row order."""
    if n_rows < 0:
        raise ValueError(f"n_rows must be non-negative, got {n_rows}")
    return [[row] for row in range(n_rows)]


def token_budget_groups(valid_lengths: Sequence[int], max_tokens_per_mb: int) -> List[List[int]]:
    """Group row indices so each group's packed token count stays within ``max_tokens_per_mb``.

    First-fit-decreasing: the longest rows are placed first so a long row never ends up stranded behind short
    ones, and a row longer than the budget simply gets a group of its own. Groups, and the rows inside them,
    come back in ascending order, which keeps the plan a pure function of the row lengths -- every rank derives
    the identical plan from the identical shard without communicating.
    """
    if max_tokens_per_mb <= 0:
        raise ValueError(f"max_tokens_per_mb must be positive, got {max_tokens_per_mb}")

    lengths = [int(length) for length in valid_lengths]
    groups: List[List[int]] = []
    group_loads: List[int] = []
    for row in sorted(range(len(lengths)), key=lambda index: lengths[index], reverse=True):
        for group_index, load in enumerate(group_loads):
            if load + lengths[row] <= max_tokens_per_mb:
                groups[group_index].append(row)
                group_loads[group_index] += lengths[row]
                break
        else:
            groups.append([row])
            group_loads.append(lengths[row])
    return sorted(sorted(group) for group in groups)


def shard_valid_lengths(batch: Dict[str, Any]) -> List[int]:
    """How many real tokens each row of ``batch`` holds, which is what a token budget is spent on."""
    validity_mask = token_validity_mask(batch)
    if int(validity_mask.shape[0]) == 0:
        return []
    return [int(length) for length in validity_mask.sum(dim=1).to(torch.long).cpu().tolist()]


def split_groups_to_count(
    groups: Sequence[Sequence[int]],
    valid_lengths: Sequence[int],
    target_count: int,
) -> List[List[int]]:
    """Split row groups until there are ``target_count`` of them, or until no group holds more than one row.

    Every rank of a request runs the same number of model calls, so a shard whose rows fit in fewer needs more
    calls than its own budget asked for. Splitting is the cheap way to get them: the tokens are unchanged, so
    each call still does real work, where repeating a call would spend a forward and a backward on a result
    that is discarded. The group holding the most tokens gives up its longest row, which keeps the widest call
    from being the one that stays widest. Splitting can only lower a group's token count, so the budget still
    holds. A shard with fewer rows than ``target_count`` comes back short, having nothing left to split, and
    its caller pads the remainder.
    """
    if target_count < len(groups):
        raise ValueError(f"cannot reduce {len(groups)} group(s) to {target_count}")

    result = [list(group) for group in groups]
    while len(result) < target_count:
        splittable = [index for index, group in enumerate(result) if len(group) > 1]
        if not splittable:
            break
        source = max(splittable, key=lambda index: sum(valid_lengths[row] for row in result[index]))
        longest_row = max(result[source], key=lambda row: valid_lengths[row])
        result[source].remove(longest_row)
        result.append([longest_row])
    return sorted(sorted(group) for group in result)


def token_budget_microbatch_groups(
    batch: Dict[str, Any],
    max_tokens_per_mb: int,
    *,
    min_groups: int = 0,
) -> List[List[int]]:
    """Token-budget row groups for one shard, in the order they will be executed.

    ``min_groups`` raises the group count to a request-wide schedule, splitting groups rather than adding calls
    that do no work; a shard with too few rows to reach it comes back short.
    """
    valid_lengths = shard_valid_lengths(batch)
    n_rows = len(valid_lengths)
    if n_rows == 0:
        return []

    groups = token_budget_groups(valid_lengths, max_tokens_per_mb)
    if min_groups > len(groups):
        groups = split_groups_to_count(groups, valid_lengths, min_groups)

    covered = sorted(int(row) for group in groups for row in group)
    if covered != list(range(n_rows)):
        raise ValueError(f"token-budget groups must partition rows [0, {n_rows}), got {groups}")
    return groups


def split_microbatches(
    tensor_dict: Dict,
    groups: Sequence[Sequence[int]],
    n_rows: int,
) -> MicrobatchSplit:
    """Apply a shared row plan and return the inverse row order."""
    forward_order = [int(row) for group in groups for row in group]
    if sorted(forward_order) != list(range(n_rows)):
        raise ValueError(f"microbatch groups must partition rows [0, {n_rows}), got {groups}")

    mbs: List[Dict] = []
    for group in groups:
        index = torch.as_tensor([int(row) for row in group], dtype=torch.long)
        mbs.append(select_rows(tensor_dict, index, n_rows))

    restore_order = None
    if forward_order != list(range(n_rows)):
        restore_order = [0] * n_rows
        for position, row in enumerate(forward_order):
            restore_order[row] = position
    return MicrobatchSplit(mbs, restore_order)
