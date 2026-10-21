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

"""Splitting one packed model call across sequence shards, and mapping a shard's tokens back to rows."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List

import torch

from .pack import PackMetadata
from .walk import map_batch_values


def token_window(padded_tokens: int, shard_count: int, shard_index: int) -> tuple[int, int]:
    """The ``[start, end)`` packed-token window owned by ``shard_index``."""
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive, got {shard_count}")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}), got {shard_index}")
    if padded_tokens % shard_count != 0:
        raise ValueError(
            f"packed tokens ({padded_tokens}) must divide by shard_count ({shard_count}); "
            "pad the call with pad_packed_microbatch first"
        )
    window_tokens = padded_tokens // shard_count
    start = shard_index * window_tokens
    return start, start + window_tokens


def token_shard(
    packed: Dict[str, Any],
    metadata: PackMetadata,
    shard_count: int,
    shard_index: int,
) -> Dict[str, Any]:
    """Slice a packed ``[1, T, ...]`` call down to one shard's contiguous token window.

    Concatenating the shards' windows in index order reproduces the packed sequence exactly, which is what the
    provider adapters rely on when they all-gather ``position_ids`` to rebuild global varlen boundaries.
    """
    start, end = token_window(metadata.padded_tokens, shard_count, shard_index)
    unpadded_tokens = int(metadata.cu_seqlens[-1])

    def sliced(key: str, value):
        if torch.is_tensor(value) and value.ndim >= 2:
            token_count = int(value.shape[1])
            if token_count == metadata.padded_tokens:
                return value[:, start:end].contiguous()
            if token_count == unpadded_tokens != metadata.padded_tokens:
                raise ValueError(
                    f"{key!r} still has the unpadded token width ({token_count}, packed width is "
                    f"{metadata.padded_tokens}), so it would reach the worker whole while the rest of the "
                    "call arrives sharded; pass the whole call through pad_packed_microbatch"
                )
        return value

    return map_batch_values(packed, sliced)


def window_cu_seqlens(
    metadata: PackMetadata,
    shard_count: int,
    shard_index: int,
) -> torch.Tensor:
    """Varlen boundaries for one shard's window: each row's tokens as they appear inside that window.

    Loss code reads these boundaries to reduce per-row (per-rollout) sums out of a packed window, so the
    contract is one segment per packed row -- boundary count is ``batch_size + 1`` no matter how the window
    falls. A row the shard does not reach at all becomes an empty segment, and a row it only partly holds
    becomes that part; summing a row across the shards recovers the whole row.

    Any tail padding (added so the token axis divides by ``shard_count``) is folded into the last row's
    segment rather than becoming a segment of its own: pad tokens carry ``IGNORE_INDEX`` labels and zero loss
    weight, so they add nothing to that row's sums, and the boundaries still cover the window exactly, which
    is what packed loss reductions check.
    """
    window_start, window_end = token_window(metadata.padded_tokens, shard_count, shard_index)
    boundaries = metadata.cu_seqlens.clamp(min=window_start, max=window_end) - window_start
    boundaries = boundaries.to(dtype=torch.int32)
    boundaries[-1] = window_end - window_start
    return boundaries


def cu_seqlens_from_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    """Varlen boundaries of a packed sequence, read back out of its ``position_ids``.

    Packing writes each row's positions starting from 0, so a 0 anywhere but the first token marks the start of
    the next row. That makes the positions a self-describing layout: whoever holds them can rebuild the segment
    boundaries without being told the row lengths, which is what lets a rank recover the *global* boundaries
    from all-gathered positions after its own window has lost sight of where its rows began.

    Padding appended by ``pad_packed_microbatch`` carries position 0, so each pad token reads as its own
    single-token segment rather than extending the row it follows.
    """
    if not torch.is_tensor(position_ids) or position_ids.ndim < 1:
        raise ValueError(f"position_ids must be a tensor with at least one dimension, got {position_ids!r}")
    flat = position_ids[0] if position_ids.ndim == 3 else position_ids
    flat = flat.reshape(-1)
    if flat.numel() == 0:
        return torch.zeros(1, dtype=torch.int32, device=position_ids.device)
    segment_lengths = torch.cat([flat[0:1], flat[:-1][(flat == 0)[1:]] + 1, flat[-1:] + 1])
    return segment_lengths.cumsum(dim=0, dtype=torch.int32)


def window_row_pieces(
    window_values: torch.Tensor,
    metadata: PackMetadata,
    shard_count: int,
    shard_index: int,
) -> List[torch.Tensor]:
    """Split one shard's packed per-token window back into per-row pieces, in packed row order.

    A row's tokens are contiguous inside the packed call, so each shard holds a prefix, a suffix, a middle
    slice, or nothing of any given row. Concatenating the shards' pieces for a row in shard order rebuilds that
    row's tokens, which is the layout the head-side restore expects.

    ``window_values`` must carry the window's tokens on its leading axis. A value with no token axis -- a
    scalar model output such as the loss a forward returns when it is given labels -- has no tokens to cut at
    the row boundaries, and naming that here keeps it from reading as a malformed window further down.
    """
    if not torch.is_tensor(window_values) or window_values.ndim < 1:
        raise ValueError(f"window_values must be a tensor with a token axis, got {window_values!r}")
    if window_values.ndim >= 2 and window_values.shape[0] == 1:
        window_values = window_values.squeeze(0)
    window_start, window_end = token_window(metadata.padded_tokens, shard_count, shard_index)
    if int(window_values.shape[0]) != window_end - window_start:
        raise ValueError(
            f"window has {int(window_values.shape[0])} tokens but shard {shard_index} owns {window_end - window_start}"
        )

    pieces: List[torch.Tensor] = []
    boundaries = metadata.cu_seqlens.tolist()
    for row in range(metadata.batch_size):
        row_start = int(boundaries[row])
        row_end = int(boundaries[row + 1])
        overlap_start = max(row_start, window_start)
        overlap_end = min(row_end, window_end)
        if overlap_end <= overlap_start:
            pieces.append(window_values[:0])
            continue
        pieces.append(window_values[overlap_start - window_start : overlap_end - window_start])
    return pieces
