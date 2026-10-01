from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import Any

import numpy as np

ActionMaskEntry = tuple[int, bool, tuple[int, ...]]
ActionMasks = dict[str, Any]


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


def action_masks_from_entries(*, seq_len: int, vocab_size: int, entries: Iterable[ActionMaskEntry]) -> ActionMasks | None:
    positions: list[int] = []
    set_indices: list[int] = []
    set_keys: dict[tuple[bool, tuple[int, ...]], int] = {}
    set_modes_allow: list[bool] = []
    set_offsets: list[int] = [0]
    token_ids: list[int] = []
    for position, mode_allow, raw_ids in sorted(entries, key=lambda item: item[0]):
        if position >= seq_len:
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


def _token_ids_from_words(words: np.ndarray, *, vocab_size: int, want_allowed: bool) -> tuple[int, ...]:
    bit_values = np.unpackbits(words.view(np.uint8), bitorder="little")[:vocab_size]
    selected = bit_values if want_allowed else np.logical_not(bit_values)
    return tuple(int(token_id) for token_id in np.flatnonzero(selected).tolist())


@lru_cache(maxsize=8192)
def _mask_set_from_words(words_bytes: bytes, *, vocab_size: int) -> tuple[bool, tuple[int, ...]]:
    words = np.frombuffer(words_bytes, dtype=np.uint32)
    allowed_count = int(np.unpackbits(words.view(np.uint8), bitorder="little")[:vocab_size].sum().item())
    if allowed_count <= 0:
        raise ValueError("xgrammar returned no allowed token ids")
    denied_count = vocab_size - allowed_count
    if denied_count == 0:
        return False, ()
    use_allow = allowed_count <= denied_count
    return use_allow, _token_ids_from_words(words, vocab_size=vocab_size, want_allowed=use_allow)


def action_mask_entry_from_bitmask_words(
    words: np.ndarray,
    *,
    position: int,
    vocab_size: int,
) -> ActionMaskEntry | None:
    normalized_words = words.astype(np.uint32, copy=True)
    valid_bits = vocab_size % 32
    if valid_bits:
        normalized_words[-1] &= np.uint32((1 << valid_bits) - 1)
    mode_allow, token_ids = _mask_set_from_words(normalized_words.tobytes(), vocab_size=vocab_size)
    if not token_ids:
        return None
    return int(position), mode_allow, token_ids
