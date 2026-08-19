from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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


def normalize_action_masks(raw: object | None, *, context: str = "action_masks") -> ActionMasks | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError(f"{context} must be a mapping, got {type(raw).__name__}")
    seq_len = _require_int(raw.get("seq_len"), name=f"{context}.seq_len")
    vocab_size = _require_int(raw.get("vocab_size"), name=f"{context}.vocab_size")
    positions = [_require_int(value, name=f"{context}.positions[]") for value in raw.get("positions") or []]
    set_indices = [_require_int(value, name=f"{context}.set_indices[]") for value in raw.get("set_indices") or []]
    set_modes_allow = [bool(value) for value in raw.get("set_modes_allow") or []]
    set_offsets = [_require_int(value, name=f"{context}.set_offsets[]") for value in raw.get("set_offsets") or []]
    token_ids = [_require_int(value, name=f"{context}.token_ids[]") for value in raw.get("token_ids") or []]
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


def action_mask_entries(action_masks: ActionMasks | None) -> list[ActionMaskEntry]:
    masks = normalize_action_masks(action_masks)
    if masks is None:
        return []
    entries: list[ActionMaskEntry] = []
    for position, set_index in zip(masks["positions"], masks["set_indices"], strict=True):
        start = masks["set_offsets"][set_index]
        end = masks["set_offsets"][set_index + 1]
        entries.append((position, bool(masks["set_modes_allow"][set_index]), tuple(masks["token_ids"][start:end])))
    return entries


def action_masks_from_entries(*, seq_len: int, vocab_size: int, entries: Iterable[ActionMaskEntry]) -> ActionMasks | None:
    positions: list[int] = []
    set_indices: list[int] = []
    set_keys: dict[tuple[bool, tuple[int, ...]], int] = {}
    set_modes_allow: list[bool] = []
    set_offsets: list[int] = [0]
    token_ids: list[int] = []
    for position, mode_allow, raw_ids in sorted(entries, key=lambda item: item[0]):
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
            entries.extend((position + offset, mode, ids) for position, mode, ids in action_mask_entries(normalized) if position < length)
        offset += int(length)
    if vocab_size is None:
        return None
    return action_masks_from_entries(seq_len=offset, vocab_size=vocab_size, entries=entries)


def pack_sp_row_action_masks(
    row_masks: Sequence[ActionMasks | None],
    position_ids: torch.Tensor,
) -> ActionMasks | None:
    """Project full-sequence action masks onto packed Ulysses SP source positions."""
    if not torch.is_tensor(position_ids) or position_ids.ndim != 2:
        raise ValueError("SP action masks require 2D position_ids")
    if len(row_masks) != int(position_ids.shape[0]):
        raise ValueError("row_masks and position_ids rows must have the same length")

    row_width = int(position_ids.shape[1])
    packed_offset = 0
    vocab_size: int | None = None
    entries: list[ActionMaskEntry] = []
    for masks, row_positions in zip(row_masks, position_ids, strict=True):
        normalized = normalize_action_masks(masks)
        if normalized is not None:
            if vocab_size is None:
                vocab_size = normalized["vocab_size"]
            elif vocab_size != normalized["vocab_size"]:
                raise ValueError("Cannot pack action masks with different vocab_size")

            if row_width:
                global_start = int(row_positions[0].item())
                global_end = int(row_positions[-1].item()) + 1
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


def action_masks_to_lm_head(action_masks: ActionMasks | None, *, device: torch.device) -> LmHeadActionMasks | None:
    masks = normalize_action_masks(action_masks)
    if masks is None:
        return None
    positions = torch.tensor(masks["positions"], dtype=torch.long, device=device)
    source_positions = positions - 1
    kept = source_positions >= 0
    if not bool(kept.any()):
        return LmHeadActionMasks(
            source_positions=torch.empty(0, dtype=torch.long, device=device),
            set_indices=torch.empty(0, dtype=torch.long, device=device),
            set_modes_allow=torch.tensor(masks["set_modes_allow"], dtype=torch.bool, device=device),
            set_offsets=torch.tensor(masks["set_offsets"], dtype=torch.long, device=device),
            token_ids=torch.tensor(masks["token_ids"], dtype=torch.long, device=device),
            vocab_size=masks["vocab_size"],
        )
    return LmHeadActionMasks(
        source_positions=source_positions[kept],
        set_indices=torch.tensor(masks["set_indices"], dtype=torch.long, device=device)[kept],
        set_modes_allow=torch.tensor(masks["set_modes_allow"], dtype=torch.bool, device=device),
        set_offsets=torch.tensor(masks["set_offsets"], dtype=torch.long, device=device),
        token_ids=torch.tensor(masks["token_ids"], dtype=torch.long, device=device),
        vocab_size=masks["vocab_size"],
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
    token_end = token_start + scaled_logits.shape[0]
    first = int(torch.searchsorted(action_masks.source_positions, torch.tensor(token_start, dtype=torch.long, device=action_masks.source_positions.device), right=False).item())
    last = int(torch.searchsorted(action_masks.source_positions, torch.tensor(token_end, dtype=torch.long, device=action_masks.source_positions.device), right=False).item())
    if first == last:
        return
    constrained_positions = action_masks.source_positions[first:last]
    constrained_set_indices = action_masks.set_indices[first:last]
    invalid_vocab_start = max(vocab_start, action_masks.vocab_size)
    for source_position, set_index in zip(constrained_positions.tolist(), constrained_set_indices.tolist(), strict=True):
        row = scaled_logits[source_position - token_start]
        if invalid_vocab_start < vocab_end:
            row[invalid_vocab_start - vocab_start :] = float("-inf")
        token_ids_start = int(action_masks.set_offsets[set_index].item())
        token_ids_end = int(action_masks.set_offsets[set_index + 1].item())
        token_ids = action_masks.token_ids[token_ids_start:token_ids_end]
        in_chunk = token_ids[(token_ids >= vocab_start) & (token_ids < vocab_end)] - vocab_start
        if bool(action_masks.set_modes_allow[set_index].item()):
            if in_chunk.numel():
                allowed_values = row.index_select(0, in_chunk).clone()
                row.fill_(float("-inf"))
                row[in_chunk] = allowed_values
            else:
                row.fill_(float("-inf"))
        elif in_chunk.numel():
            row[in_chunk] = float("-inf")


def validate_action_mask_targets(labels: torch.Tensor, action_masks: LmHeadActionMasks | None, *, token_start: int, target_logits: torch.Tensor) -> None:
    if action_masks is None or not action_masks.has_constraints:
        return
    token_end = token_start + target_logits.shape[0]
    first = int(torch.searchsorted(action_masks.source_positions, torch.tensor(token_start, dtype=torch.long, device=action_masks.source_positions.device), right=False).item())
    last = int(torch.searchsorted(action_masks.source_positions, torch.tensor(token_end, dtype=torch.long, device=action_masks.source_positions.device), right=False).item())
    if first == last:
        return
    rows = action_masks.source_positions[first:last] - token_start
    bad = ~torch.isfinite(target_logits.index_select(0, rows))
    if bool(bad.any()):
        bad_row = int(rows[bad][0].item())
        raise ValueError(f"action mask rejected sampled token id {int(labels[bad_row].item())} at source position {token_start + bad_row}")
