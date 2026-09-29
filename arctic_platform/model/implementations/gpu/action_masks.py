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
from __future__ import annotations

import math
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from typing import cast

import torch

ActionMaskEntry = tuple[int, bool, tuple[int, ...]]
ActionMasks = dict[str, Any]


@dataclass(frozen=True)
class LmHeadActionMasks:
    source_positions: torch.Tensor
    set_indices: torch.Tensor
    set_modes_allow: torch.Tensor
    set_offsets: torch.Tensor
    token_ids: torch.Tensor
    vocab_size: int
    allow_source_positions: torch.Tensor
    allow_pair_source_positions: torch.Tensor
    allow_pair_token_ids: torch.Tensor
    deny_pair_source_positions: torch.Tensor
    deny_pair_token_ids: torch.Tensor
    localized: bool = False

    @property
    def has_constraints(self) -> bool:
        return bool(self.source_positions.numel())


def _require_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be int, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    raise TypeError(f"{name} must be int, got {type(value).__name__}")


def _require_1d_array(value: object | None, *, name: str) -> list[object]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)

    ndim = getattr(value, "ndim", None)
    tolist = getattr(value, "tolist", None)
    if ndim is None or not callable(tolist):
        raise TypeError(f"{name} must be a list, tuple, or 1-D tensor/array, got {type(value).__name__}")
    if ndim != 1:
        raise ValueError(f"{name} must be 1-D, got {ndim}-D")

    values = tolist()
    if not isinstance(values, list):
        raise TypeError(f"{name}.tolist() must return a list")
    return values


def normalize_action_masks(raw: object | None, *, context: str = "action_masks") -> ActionMasks | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError(f"{context} must be a mapping, got {type(raw).__name__}")
    seq_len = _require_int(raw.get("seq_len"), name=f"{context}.seq_len")
    vocab_size = _require_int(raw.get("vocab_size"), name=f"{context}.vocab_size")
    positions = [
        _require_int(value, name=f"{context}.positions[]")
        for value in _require_1d_array(raw.get("positions"), name=f"{context}.positions")
    ]
    set_indices = [
        _require_int(value, name=f"{context}.set_indices[]")
        for value in _require_1d_array(raw.get("set_indices"), name=f"{context}.set_indices")
    ]
    set_modes_allow = [
        bool(value)
        for value in _require_1d_array(
            raw.get("set_modes_allow"),
            name=f"{context}.set_modes_allow",
        )
    ]
    set_offsets = [
        _require_int(value, name=f"{context}.set_offsets[]")
        for value in _require_1d_array(raw.get("set_offsets"), name=f"{context}.set_offsets")
    ]
    token_ids = [
        _require_int(value, name=f"{context}.token_ids[]")
        for value in _require_1d_array(raw.get("token_ids"), name=f"{context}.token_ids")
    ]
    if len(positions) != len(set_indices):
        raise ValueError(f"{context}.positions and set_indices length mismatch")
    if len(set_offsets) != len(set_modes_allow) + 1:
        raise ValueError(f"{context}.set_offsets must have len(set_modes_allow) + 1")
    if set_offsets and set_offsets[0] != 0:
        raise ValueError(f"{context}.set_offsets must start at 0")
    if set_offsets and set_offsets[-1] != len(token_ids):
        raise ValueError(f"{context}.set_offsets last value must equal len(token_ids)")
    if positions != sorted(set(positions)):
        raise ValueError(f"{context}.positions must be sorted and unique")
    if positions and positions[-1] >= seq_len:
        raise ValueError(f"{context}.positions exceed seq_len")
    if token_ids and (min(token_ids) < 0 or max(token_ids) >= vocab_size):
        raise ValueError(f"{context}.token_ids exceed vocab_size")
    return {
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "positions": positions,
        "set_indices": set_indices,
        "set_modes_allow": set_modes_allow,
        "set_offsets": set_offsets,
        "token_ids": token_ids,
    }


def _interned_token_id_sets(masks: ActionMasks) -> list[tuple[int, ...]]:
    token_ids = masks["token_ids"]
    offsets = masks["set_offsets"]
    return [tuple(token_ids[start:end]) for start, end in zip(offsets[:-1], offsets[1:], strict=True)]


def action_mask_entries(action_masks: ActionMasks | None) -> list[ActionMaskEntry]:
    masks = normalize_action_masks(action_masks)
    if masks is None:
        return []
    token_id_sets = _interned_token_id_sets(masks)
    entries: list[ActionMaskEntry] = []
    for position, set_index in zip(masks["positions"], masks["set_indices"], strict=True):
        entries.append((position, bool(masks["set_modes_allow"][set_index]), token_id_sets[set_index]))
    return entries


def action_masks_from_entries(
    *, seq_len: int, vocab_size: int, entries: Iterable[ActionMaskEntry]
) -> ActionMasks | None:
    positions: list[int] = []
    set_indices: list[int] = []
    set_keys: dict[tuple[bool, tuple[int, ...]], int] = {}
    set_modes_allow: list[bool] = []
    set_offsets: list[int] = [0]
    token_ids: list[int] = []
    # Reuse a tuple already interned by action_mask_entries so packing a long row does not
    # re-sort and re-hash thousands of token ids at every constrained position.
    object_set_indices: dict[tuple[int, bool], int] = {}
    for position, mode_allow, raw_ids in sorted(entries, key=lambda item: item[0]):
        object_key = (id(raw_ids), bool(mode_allow))
        set_index = object_set_indices.get(object_key)
        if set_index is not None:
            positions.append(int(position))
            set_indices.append(set_index)
            continue
        ids = tuple(sorted(int(token_id) for token_id in raw_ids))
        if not ids:
            continue
        key = (bool(mode_allow), ids)
        set_index = set_keys.get(key)
        if set_index is None:
            set_index = len(set_modes_allow)
            set_keys[key] = set_index
            set_modes_allow.append(bool(mode_allow))
            token_ids.extend(ids)
            set_offsets.append(len(token_ids))
        object_set_indices[object_key] = set_index
        positions.append(int(position))
        set_indices.append(set_index)
    if not positions:
        return None
    return normalize_action_masks(
        {
            "seq_len": int(seq_len),
            "vocab_size": int(vocab_size),
            "positions": positions,
            "set_indices": set_indices,
            "set_modes_allow": set_modes_allow,
            "set_offsets": set_offsets,
            "token_ids": token_ids,
        }
    )


def pack_row_action_masks(row_masks: Sequence[ActionMasks | None], lengths: Sequence[int]) -> ActionMasks | None:
    if len(row_masks) != len(lengths):
        raise ValueError("row_masks and lengths must have the same length")
    offset = 0
    vocab_size: int | None = None
    entries: list[ActionMaskEntry] = []
    for masks, length in zip(row_masks, lengths, strict=True):
        normalized = normalize_action_masks(masks)
        if normalized is not None:
            if vocab_size is None:
                vocab_size = normalized["vocab_size"]
            elif vocab_size != normalized["vocab_size"]:
                raise ValueError("Cannot pack action masks with different vocab_size")
            entries.extend(
                (position + offset, mode, ids)
                for position, mode, ids in action_mask_entries(normalized)
                if position < length
            )
        offset += int(length)
    if vocab_size is None:
        return None
    return action_masks_from_entries(seq_len=offset, vocab_size=vocab_size, entries=entries)


def pack_sp_row_action_masks(
    row_masks: Sequence[ActionMasks | None],
    position_ids: torch.Tensor,
) -> ActionMasks | None:
    """Project full-sequence action masks onto packed SP source positions."""
    if not torch.is_tensor(position_ids) or position_ids.ndim != 2:
        raise ValueError("SP action masks require 2D position_ids")
    if len(row_masks) != int(position_ids.shape[0]):
        raise ValueError("row_masks and position_ids rows must have the same length")
    return pack_window_row_action_masks(row_masks, list(position_ids))


def pack_window_row_action_masks(
    row_masks: Sequence[ActionMasks | None],
    position_pieces: Sequence[torch.Tensor],
) -> ActionMasks | None:
    """Project full-sequence action masks onto one packed window, given each row's position piece.

    ``position_pieces[i]`` holds the positions of row ``i`` that live in this window, in packed order. Packed SP
    windows hold uneven pieces -- a whole row, the head or tail of a row split mid-sequence, or nothing -- so
    each row contributes only its own piece width to the packed offset. An empty piece contributes no entries.
    """
    packed_offset = 0
    vocab_size: int | None = None
    entries: list[ActionMaskEntry] = []
    for masks, row_positions in zip(row_masks, position_pieces, strict=True):
        row_width = int(row_positions.numel())
        normalized = normalize_action_masks(masks)
        if normalized is not None:
            if vocab_size is None:
                vocab_size = normalized["vocab_size"]
            elif vocab_size != normalized["vocab_size"]:
                raise ValueError("Cannot pack action masks with different vocab_size")

            if row_width:
                global_start, global_last = (int(bound) for bound in row_positions[[0, -1]].tolist())
                global_end = global_last + 1
                if global_end - global_start != row_width:
                    raise ValueError("SP action-mask projection requires contiguous position_ids")
                for target_position, mode, ids in action_mask_entries(normalized):
                    local_source = target_position - 1 - global_start
                    if 0 <= local_source < row_width:
                        entries.append((packed_offset + local_source + 1, mode, ids))
        packed_offset += row_width

    if vocab_size is None:
        return None
    # A constraint on the token after the shard's final source position maps
    # to target position packed_offset, so the target-coordinate extent is T+1.
    return action_masks_from_entries(
        seq_len=packed_offset + 1,
        vocab_size=vocab_size,
        entries=entries,
    )


def _expand_lm_head_action_mask_pairs(
    source_positions: torch.Tensor,
    set_indices: torch.Tensor,
    set_modes_allow: torch.Tensor,
    set_offsets: torch.Tensor,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand interned sets into batched ``(source position, token id)`` pairs.

    The expansion happens once per LM-head invocation. Applying a vocab tile can
    then use a handful of batched indexing operations instead of one Python loop
    and several tiny accelerator kernels per constrained position.
    """
    if not source_positions.numel():
        empty = torch.empty(0, dtype=torch.long, device=source_positions.device)
        return empty, empty, empty, empty, empty

    position_modes_allow = set_modes_allow.index_select(0, set_indices)
    allow_source_positions = source_positions[position_modes_allow]
    set_starts = set_offsets.index_select(0, set_indices)
    set_ends = set_offsets.index_select(0, set_indices + 1)
    set_lengths = set_ends - set_starts

    pair_source_positions = torch.repeat_interleave(source_positions, set_lengths)
    pair_set_starts = torch.repeat_interleave(set_starts, set_lengths)
    pair_group_starts = torch.repeat_interleave(set_lengths.cumsum(0) - set_lengths, set_lengths)
    pair_token_offsets = (
        torch.arange(pair_source_positions.numel(), dtype=torch.long, device=source_positions.device)
        - pair_group_starts
    )
    pair_token_ids = token_ids.index_select(0, pair_set_starts + pair_token_offsets)
    pair_modes_allow = torch.repeat_interleave(position_modes_allow, set_lengths)

    return (
        allow_source_positions,
        pair_source_positions[pair_modes_allow],
        pair_token_ids[pair_modes_allow],
        pair_source_positions[~pair_modes_allow],
        pair_token_ids[~pair_modes_allow],
    )


def action_masks_to_lm_head(action_masks: ActionMasks | None, *, device: torch.device) -> LmHeadActionMasks | None:
    masks = normalize_action_masks(action_masks)
    if masks is None:
        return None
    positions = torch.tensor(masks["positions"], dtype=torch.long, device=device)
    source_positions = positions - 1
    kept = source_positions >= 0
    source_positions = source_positions[kept]
    set_indices = torch.tensor(masks["set_indices"], dtype=torch.long, device=device)[kept]
    set_modes_allow = torch.tensor(masks["set_modes_allow"], dtype=torch.bool, device=device)
    set_offsets = torch.tensor(masks["set_offsets"], dtype=torch.long, device=device)
    token_ids = torch.tensor(masks["token_ids"], dtype=torch.long, device=device)
    (
        allow_source_positions,
        allow_pair_source_positions,
        allow_pair_token_ids,
        deny_pair_source_positions,
        deny_pair_token_ids,
    ) = _expand_lm_head_action_mask_pairs(
        source_positions,
        set_indices,
        set_modes_allow,
        set_offsets,
        token_ids,
    )
    return LmHeadActionMasks(
        source_positions=source_positions,
        set_indices=set_indices,
        set_modes_allow=set_modes_allow,
        set_offsets=set_offsets,
        token_ids=token_ids,
        vocab_size=masks["vocab_size"],
        allow_source_positions=allow_source_positions,
        allow_pair_source_positions=allow_pair_source_positions,
        allow_pair_token_ids=allow_pair_token_ids,
        deny_pair_source_positions=deny_pair_source_positions,
        deny_pair_token_ids=deny_pair_token_ids,
    )


def _constrained_slice(source_positions: torch.Tensor, token_start: int, token_end: int) -> tuple[int, int]:
    """Half-open range of ``source_positions`` entries that fall in ``[token_start, token_end)``.

    Both bounds are looked up in a single search and read back in a single device transfer, because this runs
    once per lm-head token chunk and the positions live on the accelerator.
    """
    bounds = torch.searchsorted(
        source_positions,
        torch.tensor([token_start, token_end], dtype=torch.long, device=source_positions.device),
        right=False,
    ).tolist()
    return int(bounds[0]), int(bounds[1])


def _empty_localized_lm_head_masks(action_masks: LmHeadActionMasks) -> LmHeadActionMasks:
    empty = torch.empty(0, dtype=torch.long, device=action_masks.source_positions.device)
    return LmHeadActionMasks(
        source_positions=empty,
        set_indices=empty,
        set_modes_allow=action_masks.set_modes_allow,
        set_offsets=action_masks.set_offsets,
        token_ids=action_masks.token_ids,
        vocab_size=action_masks.vocab_size,
        allow_source_positions=empty,
        allow_pair_source_positions=empty,
        allow_pair_token_ids=empty,
        deny_pair_source_positions=empty,
        deny_pair_token_ids=empty,
        localized=True,
    )


def slice_lm_head_action_masks(
    action_masks: LmHeadActionMasks | None,
    *,
    token_start: int,
    token_end: int,
) -> LmHeadActionMasks | None:
    """Restrict interned pairs to one token tile and shift positions into that tile.

    Pair arrays are sorted by source position, so this is a handful of binary searches rather than a
    scan of the packed window. The LM-head vocab loop then only filters this slice by token id.
    """
    if action_masks is None:
        return None
    if action_masks.localized or not action_masks.has_constraints:
        return action_masks
    first, last = _constrained_slice(action_masks.source_positions, token_start, token_end)
    if first == last:
        return _empty_localized_lm_head_masks(action_masks)

    def _localize_pairs(positions: torch.Tensor, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pair_first, pair_last = _constrained_slice(positions, token_start, token_end)
        return positions[pair_first:pair_last] - token_start, token_ids[pair_first:pair_last]

    allow_first, allow_last = _constrained_slice(action_masks.allow_source_positions, token_start, token_end)
    allow_pair_source_positions, allow_pair_token_ids = _localize_pairs(
        action_masks.allow_pair_source_positions, action_masks.allow_pair_token_ids
    )
    deny_pair_source_positions, deny_pair_token_ids = _localize_pairs(
        action_masks.deny_pair_source_positions, action_masks.deny_pair_token_ids
    )
    return LmHeadActionMasks(
        source_positions=action_masks.source_positions[first:last] - token_start,
        set_indices=action_masks.set_indices[first:last],
        set_modes_allow=action_masks.set_modes_allow,
        set_offsets=action_masks.set_offsets,
        token_ids=action_masks.token_ids,
        vocab_size=action_masks.vocab_size,
        allow_source_positions=action_masks.allow_source_positions[allow_first:allow_last] - token_start,
        allow_pair_source_positions=allow_pair_source_positions,
        allow_pair_token_ids=allow_pair_token_ids,
        deny_pair_source_positions=deny_pair_source_positions,
        deny_pair_token_ids=deny_pair_token_ids,
        localized=True,
    )


def apply_lm_head_action_masks_(
    scaled_logits: torch.Tensor,
    action_masks: LmHeadActionMasks | None,
    *,
    token_start: int,
    vocab_start: int,
    vocab_end: int,
) -> None:
    if action_masks is None or not action_masks.has_constraints:
        return
    if not action_masks.localized:
        action_masks = slice_lm_head_action_masks(
            action_masks,
            token_start=token_start,
            token_end=token_start + scaled_logits.shape[0],
        )
        if action_masks is None or not action_masks.has_constraints:
            return
    constrained_rows = action_masks.source_positions
    invalid_vocab_start = max(vocab_start, action_masks.vocab_size)
    if invalid_vocab_start < vocab_end:
        scaled_logits[constrained_rows, invalid_vocab_start - vocab_start :] = float("-inf")

    allow_rows = action_masks.allow_source_positions
    allow_pair_in_vocab = (action_masks.allow_pair_token_ids >= vocab_start) & (
        action_masks.allow_pair_token_ids < vocab_end
    )
    allow_pair_rows = action_masks.allow_pair_source_positions[allow_pair_in_vocab]
    allow_pair_columns = action_masks.allow_pair_token_ids[allow_pair_in_vocab] - vocab_start
    allowed_values = scaled_logits[allow_pair_rows, allow_pair_columns].clone()
    scaled_logits[allow_rows] = float("-inf")
    scaled_logits[allow_pair_rows, allow_pair_columns] = allowed_values

    deny_pair_in_vocab = (action_masks.deny_pair_token_ids >= vocab_start) & (
        action_masks.deny_pair_token_ids < vocab_end
    )
    deny_pair_rows = action_masks.deny_pair_source_positions[deny_pair_in_vocab]
    deny_pair_columns = action_masks.deny_pair_token_ids[deny_pair_in_vocab] - vocab_start
    scaled_logits[deny_pair_rows, deny_pair_columns] = float("-inf")


def validate_action_mask_targets(
    labels: torch.Tensor, action_masks: LmHeadActionMasks | None, *, token_start: int, target_logits: torch.Tensor
) -> None:
    if action_masks is None or not action_masks.has_constraints:
        return
    if action_masks.localized:
        rows = action_masks.source_positions
    else:
        token_end = token_start + target_logits.shape[0]
        first, last = _constrained_slice(action_masks.source_positions, token_start, token_end)
        if first == last:
            return
        rows = action_masks.source_positions[first:last] - token_start
    bad = ~torch.isfinite(target_logits.index_select(0, rows))
    if bool(bad.any()):
        bad_row = int(rows[bad][0].item())
        raise ValueError(
            f"action mask rejected sampled token id {int(labels[bad_row].item())} at source position"
            f" {token_start + bad_row}"
        )


def filter_lm_head_action_masks(
    action_masks: LmHeadActionMasks | None,
    keep_source_positions: torch.Tensor,
) -> LmHeadActionMasks | None:
    """Drop constraints whose source position is ``False`` in ``keep_source_positions``.

    ``keep_source_positions`` holds one boolean per flattened source position, in the coordinate space of
    ``LmHeadActionMasks.source_positions``, so the masks passed here must not yet be localized to a token tile.
    A position whose label is the ignore index is scored at a substitute vocabulary entry rather than at a
    target of its own, so a constraint left in place there can drive the substitute's logit to ``-inf``, which
    makes :func:`validate_action_mask_targets` reject a position the caller asked to ignore.

    Every array keyed by source position is filtered: the interned set constraints, the rows an allow-set
    blanks before its pairs are restored, and both pair encodings. Dropping a row from one and not another
    would leave :func:`apply_lm_head_action_masks_` blanking a row whose allowed tokens are gone. The set
    tables themselves (``set_modes_allow``, ``set_offsets``, ``token_ids``) are keyed by set rather than by
    position and are shared, so they are carried through untouched.

    Selecting with a boolean mask preserves the ascending order of each array that
    :func:`slice_lm_head_action_masks` requires for its ``searchsorted`` lookups.
    """
    if action_masks is None or not action_masks.has_constraints:
        return action_masks
    keep = keep_source_positions.reshape(-1).to(device=action_masks.source_positions.device, dtype=torch.bool)
    position_arrays = (
        action_masks.source_positions,
        action_masks.allow_source_positions,
        action_masks.allow_pair_source_positions,
        action_masks.deny_pair_source_positions,
    )
    widest = max(int(positions.max().item()) for positions in position_arrays if positions.numel())
    # An out-of-range index_select is a device-side assert on CUDA with no context, so name the mismatch here.
    if widest >= int(keep.numel()):
        raise ValueError("action mask source positions exceed label width")

    def _kept(positions: torch.Tensor) -> torch.Tensor:
        return keep.index_select(0, positions)

    kept = _kept(action_masks.source_positions)
    allow_kept = _kept(action_masks.allow_source_positions)
    allow_pair_kept = _kept(action_masks.allow_pair_source_positions)
    deny_pair_kept = _kept(action_masks.deny_pair_source_positions)
    return LmHeadActionMasks(
        source_positions=action_masks.source_positions[kept],
        set_indices=action_masks.set_indices[kept],
        set_modes_allow=action_masks.set_modes_allow,
        set_offsets=action_masks.set_offsets,
        token_ids=action_masks.token_ids,
        vocab_size=action_masks.vocab_size,
        allow_source_positions=action_masks.allow_source_positions[allow_kept],
        allow_pair_source_positions=action_masks.allow_pair_source_positions[allow_pair_kept],
        allow_pair_token_ids=action_masks.allow_pair_token_ids[allow_pair_kept],
        deny_pair_source_positions=action_masks.deny_pair_source_positions[deny_pair_kept],
        deny_pair_token_ids=action_masks.deny_pair_token_ids[deny_pair_kept],
        localized=action_masks.localized,
    )


def slice_action_masks_for_logits_to_keep(
    action_masks: ActionMasks | None,
    *,
    batch_size: int,
    original_seq_len: int,
    logits_to_keep: object,
) -> ActionMasks | None:
    """Remap full-sequence masks into the coordinate space selected by ``logits_to_keep``.

    Positions in ``action_masks`` are target positions over the full sequence, but a caller that passes
    ``logits_to_keep`` scores only a subset of columns. A constraint on a dropped column has no target left and is
    discarded; a constraint on a kept column moves to that column's index within the kept subset.
    """
    masks = normalize_action_masks(action_masks)
    if masks is None:
        return None
    selected_columns = _logits_to_keep_columns(logits_to_keep, original_seq_len)
    if selected_columns is None or selected_columns == list(range(original_seq_len)):
        return masks

    kept_seq_len = len(selected_columns)
    selected_by_column: dict[int, list[int]] = {}
    for local_column, source_column in enumerate(selected_columns):
        selected_by_column.setdefault(source_column, []).append(local_column)
    source_limit = batch_size * original_seq_len
    entries: list[ActionMaskEntry] = []
    for target_position, mode, ids in action_mask_entries(masks):
        source_position = target_position - 1
        if source_position < 0:
            continue
        if source_position >= source_limit:
            raise ValueError("action mask source positions exceed label width")
        row = source_position // original_seq_len
        col = source_position % original_seq_len
        for local_col in selected_by_column.get(col, ()):
            local_source_position = row * kept_seq_len + local_col
            entries.append((local_source_position + 1, mode, ids))

    # A constraint on the token after a row's final kept source position maps to target position
    # ``batch_size * kept_seq_len``, so the target-coordinate extent is one past the kept token count.
    return action_masks_from_entries(
        seq_len=batch_size * kept_seq_len + 1,
        vocab_size=masks["vocab_size"],
        entries=entries,
    )


def _logits_to_keep_columns(logits_to_keep: object, original_seq_len: int) -> list[int] | None:
    """Columns ``logits_to_keep`` selects, or ``None`` when it selects the whole sequence.

    An int is the Hugging Face "score only the last k tokens" convention; a 1D integer tensor names the kept
    columns directly and may repeat or reorder them. Negative column indices count from the end of the sequence.
    """
    if isinstance(logits_to_keep, bool):
        raise TypeError("logits_to_keep must be int or a 1D integer tensor")
    if isinstance(logits_to_keep, int):
        if logits_to_keep <= 0 or logits_to_keep >= original_seq_len:
            return None
        return list(range(original_seq_len - int(logits_to_keep), original_seq_len))
    if torch.is_tensor(logits_to_keep):
        columns_tensor = cast(torch.Tensor, logits_to_keep)
        if columns_tensor.ndim != 1:
            raise ValueError("logits_to_keep tensor must be 1D")
        raw_columns = columns_tensor.detach().cpu().tolist()
    elif isinstance(logits_to_keep, Sequence) and not isinstance(logits_to_keep, (str, bytes)):
        raw_columns = list(logits_to_keep)
    else:
        raise TypeError("logits_to_keep must be int or a 1D integer tensor")

    columns: list[int] = []
    for raw_column in raw_columns:
        column = _require_int(raw_column, name="logits_to_keep[]")
        if column < 0:
            column += original_seq_len
        if column < 0 or column >= original_seq_len:
            raise ValueError("logits_to_keep indices exceed sequence length")
        columns.append(column)
    return columns
