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

"""Packing a row group into one varlen model call, and putting the results back into rows."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any
from typing import Dict

import torch
import torch.nn.functional as F

from .groups import batch_position_ids
from .groups import batch_tensor
from .walk import DROP
from .walk import map_batch_values

# The loss ignore index every provider in this stack uses; also the pad value for packed ``labels``.
IGNORE_INDEX = -100


@dataclass(frozen=True)
class PackMetadata:
    """Where each row of a packed model call lives.

    ``batch_size`` / ``sequence_length`` are the pre-pack ``[B, S]`` shape, so results can be scattered back
    into that layout. ``cu_seqlens`` are the packed row boundaries and ``padded_tokens`` is the packed width
    after any divisibility padding, which is what a token-axis split divides up.
    """

    batch_size: int
    sequence_length: int
    valid_lengths: torch.Tensor
    cu_seqlens: torch.Tensor
    padded_tokens: int


def sequence_pad_value(key: str) -> float:
    if key == "labels":
        return IGNORE_INDEX
    if key == "temperature":
        # Temperature is used as a divisor by the LM head. Padding with zero
        # creates infinities even though the corresponding loss mask is false.
        return 1.0
    return 0.0


def packed_token_index(
    valid_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    """Position of every packed token inside a flattened ``[B * S, ...]`` batch, in packed order.

    Row ``r``'s tokens are its leading ``valid_lengths[r]`` positions, which land at ``r * S + 0 ..``. Because
    the mapping depends only on the row lengths, one index serves every tensor in the call and both directions:
    packing is an ``index_select`` and unpacking an ``index_copy_``, whatever the trailing shape.
    """
    device = cu_seqlens.device
    lengths = valid_lengths.to(dtype=torch.long, device=device)
    packed_tokens = int(lengths.sum())
    row_ids = torch.repeat_interleave(torch.arange(lengths.numel(), device=device), lengths)
    row_starts = torch.repeat_interleave(cu_seqlens[:-1].to(dtype=torch.long, device=device), lengths)
    within_row = torch.arange(packed_tokens, device=device) - row_starts
    return row_ids * sequence_length + within_row


def pack_microbatch(data: Dict[str, Any]) -> tuple[Dict[str, Any], PackMetadata]:
    """Pack a row group's valid tokens into a single varlen ``[1, T, ...]`` model call.

    The returned dict holds model inputs only, so a caller can splat it into the engine. The packed varlen
    boundaries ride on the returned ``PackMetadata`` instead, because they describe this call's rows while a
    sharded model call needs the *global* boundaries, which the provider adapters rebuild from all-gathered
    ``position_ids``.
    """
    attention_mask = batch_tensor(data, "attention_mask", min_dims=2)
    position_ids = batch_position_ids(data)
    if attention_mask is None or attention_mask.ndim != 2:
        raise ValueError("packing requires a 2D attention_mask")
    if position_ids is None or not torch.is_tensor(position_ids):
        raise ValueError(
            "packing requires position_ids with the same first two dimensions as attention_mask, "
            f"got position_ids={getattr(position_ids, 'shape', None)} attention_mask={attention_mask.shape}"
        )
    if tuple(position_ids.shape[:2]) != tuple(attention_mask.shape):
        raise ValueError(
            "packing requires position_ids with the same first two dimensions as attention_mask, "
            f"got position_ids={tuple(position_ids.shape)} attention_mask={attention_mask.shape}"
        )

    batch_size, sequence_length = attention_mask.shape
    valid_sequence_lengths = attention_mask.sum(dim=1, dtype=torch.int32)
    # Each row is billed its token count and then read from its first column, so validity has to sit at the
    # front. A left-padded row would pack pad tokens and drop as many real ones, and nothing downstream can
    # tell: the call has the right width and the wrong contents.
    prefix_mask = torch.arange(sequence_length, device=attention_mask.device).unsqueeze(
        0
    ) < valid_sequence_lengths.unsqueeze(1)
    if not torch.equal(attention_mask.to(dtype=torch.bool), prefix_mask):
        raise ValueError(
            "packing requires left-aligned rows: every row's real tokens must be its leading columns, with "
            "padding at the tail"
        )
    cu_seqlens = F.pad(
        torch.cumsum(valid_sequence_lengths, dim=0, dtype=torch.int32),
        (1, 0),
        value=0,
    )
    packed_tokens = int(cu_seqlens[-1])
    token_index = packed_token_index(valid_sequence_lengths, cu_seqlens, sequence_length)

    def packed_value(key: str, value: Any) -> Any:
        # ``attention_mask`` is dropped: the packed layout carries token validity, and a [1, T] mask of all
        # ones would only mislead a model into treating pad-free packing as a single sequence.
        if key == "attention_mask":
            return DROP
        if torch.is_tensor(value) and value.ndim >= 2 and tuple(value.shape[:2]) == (batch_size, sequence_length):
            rows = value.reshape(batch_size * sequence_length, *value.shape[2:])
            return rows.index_select(0, token_index.to(rows.device)).unsqueeze(0)
        return value

    packed = map_batch_values(data, packed_value)

    return packed, PackMetadata(
        batch_size,
        sequence_length,
        valid_sequence_lengths,
        cu_seqlens,
        packed_tokens,
    )


def pad_packed_microbatch(
    packed: Dict[str, Any],
    metadata: PackMetadata,
    multiple: int,
) -> tuple[Dict[str, Any], PackMetadata]:
    """Right-pad a packed ``[1, T, ...]`` call so ``T`` is a multiple of ``multiple``.

    A token-axis split hands every shard the same width, so ``T`` has to divide by the shard count. Padding
    goes at the tail with per-key neutral values: ``labels`` get ``IGNORE_INDEX`` so pad tokens contribute no
    loss, and ``position_ids`` get 0 so each pad token reads as its own single-token sequence rather than
    extending the last real one.

    Nested dicts are padded too, and by their own leaf key, because RL ships its per-token tensors inside a
    ``context`` sub-dict. A leaf left at the unpadded width would no longer match the width ``token_shard``
    slices, so it would reach the worker whole while its siblings arrived sharded.
    """
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    current_tokens = metadata.padded_tokens
    padded_tokens = ((current_tokens + multiple - 1) // multiple) * multiple
    if padded_tokens == current_tokens:
        return packed, metadata

    pad_tokens = padded_tokens - current_tokens

    def padded_value(key: str, value: Any) -> Any:
        if torch.is_tensor(value) and value.ndim >= 2 and int(value.shape[1]) == current_tokens:
            pad_widths = [0, 0] * (value.ndim - 2) + [0, pad_tokens]
            return F.pad(value, pad_widths, value=sequence_pad_value(key))
        return value

    return map_batch_values(packed, padded_value), replace(metadata, padded_tokens=padded_tokens)


def unpack_output(
    packed_tensor: torch.Tensor,
    metadata: PackMetadata,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Restore a packed ``[1, T, ...]`` tensor to padded ``[B, S, ...]``."""
    batch_size = metadata.batch_size
    sequence_length = metadata.sequence_length
    valid_lengths = metadata.valid_lengths
    if not torch.is_tensor(valid_lengths) or valid_lengths.numel() != batch_size:
        raise ValueError("pack metadata must contain one tensor length per row")
    if packed_tensor.ndim >= 2 and packed_tensor.shape[0] == 1:
        packed_tensor = packed_tensor.squeeze(0)

    trailing_shape = tuple(packed_tensor.shape[1:])
    output = torch.full(
        (batch_size * sequence_length, *trailing_shape),
        pad_value,
        dtype=packed_tensor.dtype,
        device=packed_tensor.device,
    )
    if batch_size:
        lengths = valid_lengths.to(dtype=torch.long, device=packed_tensor.device)
        shortest, longest = torch.stack((lengths.min(), lengths.max())).tolist()
        if shortest < 0 or longest > sequence_length:
            raise ValueError(f"valid sequence lengths must be in [0, {sequence_length}], got {lengths.tolist()}")
        cu_seqlens = metadata.cu_seqlens.to(packed_tensor.device)
        if not torch.equal(torch.diff(cu_seqlens).to(dtype=torch.long), lengths):
            raise ValueError(
                "pack metadata disagrees with itself: the cu_seqlens segment widths "
                f"({torch.diff(cu_seqlens).tolist()}) must equal valid_lengths ({lengths.tolist()})"
            )
        token_index = packed_token_index(lengths, cu_seqlens, sequence_length)
        # A padded call carries tail tokens past the last row; they belong to no row and are dropped here.
        output.index_copy_(0, token_index, packed_tensor[: token_index.numel()])
    return output.view(batch_size, sequence_length, *trailing_shape)
