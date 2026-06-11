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

"""Micro-batch splitting utilities for RL training."""

from __future__ import annotations

import bisect
import itertools
from dataclasses import dataclass as _dataclass
from dataclasses import field as _field

import numpy as np
import torch
import torch.distributed as dist

DEFAULT_MAX_TOKENS_PER_MB = 10240


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _ffd_allocate_inner(values, capacity: int, min_groups: int, n_groups_divisor: int = 1):
    value_indices = np.argsort(-values)
    group_indices: list = []
    group_values: list = []
    group_cnt = 0
    for idx in value_indices:
        if len(group_values) < min_groups or group_values[0][0] + values[idx] > capacity:
            bisect.insort(group_values, (float(values[idx]), group_cnt))
            group_indices.append([idx])
            group_cnt += 1
        else:
            i = bisect.bisect_right(group_values, (capacity - values[idx], len(values)))
            candidates = [group_values[j][1] for j in range(i)]
            lens = [len(group_indices[g]) for g in candidates]
            j = np.argmin(lens)
            v, group_idx = group_values.pop(j)
            bisect.insort(group_values, (float(values[idx] + v), group_idx))
            group_indices[group_idx].append(idx)
    return group_indices


def _ffd_allocate(values, capacity: int, min_groups: int, n_groups_divisor: int = 1):
    if min_groups is None or min_groups < n_groups_divisor:
        min_groups = n_groups_divisor
    if any(v > capacity for v in values):
        raise RuntimeError(f"Values exceed capacity {capacity}")
    if len(values) < min_groups:
        raise RuntimeError(f"Too few values for min_groups={min_groups}")
    while True:
        res = _ffd_allocate_inner(np.array(values), capacity, min_groups)
        min_groups += n_groups_divisor - min_groups % n_groups_divisor
        if len(res) % n_groups_divisor == 0:
            break
        if len(values) < min_groups:
            raise RuntimeError("Cannot satisfy n_groups_divisor constraint")
    return res


@_dataclass
class MicroBatchSpec:
    """Specification for splitting micro-batches during training."""

    n_mbs: int | None = _field(default=1, metadata={"help": "Number of micro-batches."})
    granularity: int = _field(default=1, metadata={"help": "Granularity per micro-batch."})
    max_tokens_per_mb: int | None = _field(default=None, metadata={"help": "Max tokens per micro-batch."})
    n_mbs_divisor: int = _field(default=1, metadata={"help": "Divisor for number of micro-batches."})

    @classmethod
    def new(cls, mb_spec: "MicroBatchSpec", **kwargs):
        fields = dict(
            n_mbs=mb_spec.n_mbs,
            granularity=mb_spec.granularity,
            max_tokens_per_mb=mb_spec.max_tokens_per_mb,
            n_mbs_divisor=mb_spec.n_mbs_divisor,
        )
        fields.update(kwargs)
        return cls(**fields)


@_dataclass
class MicroBatchList:
    data: dict
    mb_spec: MicroBatchSpec
    mbs: list
    group_lens: list
    forward_indices: list | None = None
    backward_indices: list | None = None

    def __len__(self) -> int:
        return len(self.mbs)


def _flat2d_mb(arr):
    return list(itertools.chain(*arr))


def _reorder_list(xs, indices):
    return [xs[i] for i in indices]


def _is_multi_modal_key(key: str) -> bool:
    return key.startswith("multi_modal_input")


def _dict_of_list2list_of_dict(dict_of_lists):
    if not dict_of_lists:
        return []
    keys = list(dict_of_lists.keys())
    length = len(dict_of_lists[keys[0]])
    return [{key: dict_of_lists[key][i] for key in keys} for i in range(length)]


def _allocate_balanced_mbs(mb_spec, lens):
    group_indices = _ffd_allocate(
        lens, mb_spec.max_tokens_per_mb, min_groups=mb_spec.n_mbs or 1, n_groups_divisor=mb_spec.n_mbs_divisor
    )
    return sorted([sorted(g) for g in group_indices])


def split_padded_tensor_dict_into_mb_list(data: dict, mb_spec: MicroBatchSpec, group=None) -> MicroBatchList:
    """Split a padded dict of tensors into micro-batches."""
    if "attention_mask" not in data:
        raise ValueError("Input data must contain 'attention_mask'.")
    if mb_spec.max_tokens_per_mb is None:
        mb_spec = MicroBatchSpec.new(mb_spec, max_tokens_per_mb=DEFAULT_MAX_TOKENS_PER_MB)
    granularity = mb_spec.granularity
    bs = data["attention_mask"].shape[0]
    max_seqlen = data["attention_mask"].shape[1]
    seq_lens = data["attention_mask"].sum(1).long().cpu().numpy().tolist()
    input_lens = data["attention_mask"].view(bs // granularity, granularity, -1).sum(dim=(1, 2)).long().cpu().numpy()

    multimodal_keys = {key for key in data if _is_multi_modal_key(key)}
    to_split, not_to_split = {}, {}
    for key, value in data.items():
        if key in multimodal_keys:
            continue
        if key == "position_ids" or (torch.is_tensor(value) and value.numel() == bs * max_seqlen):
            to_split[key] = value
        else:
            not_to_split[key] = value

    group_indices = _allocate_balanced_mbs(mb_spec, input_lens)
    if dist.is_initialized():
        # Sync microbatch count across all ranks so every rank runs the same
        # number of microbatches. Without this, different sequence length
        # distributions on each rank (after the DP batch split) produce
        # different microbatch counts -> mismatched ZeRO-2 all-reduce boundaries
        # -> NCCL deadlock.
        sync_group = group  # may be None -> uses default world group
        world_size = dist.get_world_size(sync_group)
        all_n_mbs = [None] * world_size
        dist.all_gather_object(all_n_mbs, len(group_indices), group=sync_group)
        max_n_mbs = max(all_n_mbs)
        if max_n_mbs != len(group_indices):
            group_indices = _allocate_balanced_mbs(MicroBatchSpec.new(mb_spec, n_mbs=max_n_mbs), input_lens)

    group_indices = [
        _flat2d_mb([list(range(i * granularity, (i + 1) * granularity)) for i in gi]) for gi in group_indices
    ]
    splitted_lens = [[seq_lens[i] for i in gi] for gi in group_indices]
    group_n_seqs = [len(x) for x in splitted_lens]
    group_lens = [sum(x) for x in splitted_lens]
    forward_indices = _flat2d_mb(group_indices)
    import numpy as _np

    backward_indices = _np.zeros(bs, dtype=_np.int64)
    backward_indices[forward_indices] = _np.arange(bs)

    def _split(tensor):
        unpacked = [tensor[i] for i in range(bs)]
        reordered = torch.stack(_reorder_list(unpacked, forward_indices))
        result, offset = [], 0
        for n in group_n_seqs:
            result.append(reordered[offset : offset + n])
            offset += n
        return result

    to_split = {k: _split(v) for k, v in to_split.items()}
    for key in multimodal_keys:
        to_split[key] = [[data[key][i] for i in gi] for gi in group_indices]

    mbs = _dict_of_list2list_of_dict(to_split)
    results = [{**mb, **not_to_split} for mb in mbs]

    return MicroBatchList(
        data=data,
        mb_spec=mb_spec,
        mbs=results,
        forward_indices=forward_indices,
        backward_indices=backward_indices.tolist(),
        group_lens=group_lens,
    )
