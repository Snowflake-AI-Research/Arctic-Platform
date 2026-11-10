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

"""Row-owned media (images, video) through planning, packing and token windows.

A batch carries each media kind as three top-level leaves, in the Hugging Face naming:

* ``pixel_values`` -- every image's patches, ``[patches, patch_dim]``, concatenated in image order;
* ``image_grid_thw`` -- one ``[t, h, w]`` patch grid per image, ``[images, 3]``;
* ``image_row_ids`` -- the row each image belongs to, ``[images]``, non-decreasing.

and the same for video (``pixel_values_videos``, ``video_grid_thw``, ``video_row_ids``). Image ``i`` owns
``prod(grid_i)`` consecutive patches, and ``prod(grid_i) / merge**2`` tokens of its row, in order.

None of these leaves is row- or token-shaped: their leading dimension counts patches or images. The generic
shape rules elsewhere in this package (a leaf whose first dimension equals the row count is a row leaf; a leaf
whose second dimension is the token count is a token leaf) would mis-slice them whenever a count happens to
coincide, so media leaves are recognized by key and handled here instead.

An image's tokens are one contiguous run. A video's are ``t`` frame runs of ``h * w / merge**2`` tokens each,
in order, with only text between them: the Qwen3-VL processor puts each frame's timestamp text before it. An
item's span runs from its first token to its last, text between frames included.

``token_shard`` gives each window the media whose span sits inside it, whole: a vision tower encodes complete
images and videos, and the window holds all of the item's placeholder tokens to scatter them into. An item whose
span crosses a window boundary is refused, because each window would get all of its pixels but only part of its
placeholder tokens. A caller avoids that by moving the boundary off the item: pad or reorder the text before it
so that the item starts in the window it ends in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch

from .mrope import IMAGE_TOKEN_TYPE
from .mrope import TEXT_TOKEN_TYPE
from .mrope import VIDEO_TOKEN_TYPE


@dataclass(frozen=True)
class MediaKind:
    name: str
    values_key: str
    grid_key: str
    row_ids_key: str
    token_type: int

    @property
    def keys(self) -> Tuple[str, str, str]:
        return (self.values_key, self.grid_key, self.row_ids_key)


IMAGE_MEDIA = MediaKind("image", "pixel_values", "image_grid_thw", "image_row_ids", IMAGE_TOKEN_TYPE)
VIDEO_MEDIA = MediaKind("video", "pixel_values_videos", "video_grid_thw", "video_row_ids", VIDEO_TOKEN_TYPE)
MEDIA_KINDS = (IMAGE_MEDIA, VIDEO_MEDIA)
MEDIA_KEYS = frozenset(key for kind in MEDIA_KINDS for key in kind.keys)


class MediaBatchError(ValueError):
    """The caller's media batch is malformed: its media leaves disagree with each other, its rows or its tokens.

    A ValueError, so callers that catch ValueError still do; a caller that must tell a bad batch from an
    internal error catches this instead of matching the message.
    """


class MediaStraddleError(MediaBatchError):
    """A media item's tokens straddle a window boundary, so no window can hold the item whole.

    A MediaBatchError; a caller that must tell this refusal from other malformed media (to change its layout
    and retry, say) catches this instead of matching the message.
    ``kind`` is the media kind's name (``"image"`` or ``"video"``), ``index`` the item's position among that
    kind's items in the call, ``span`` its ``(start, end)`` token range and ``boundary`` the token it crosses.
    """

    def __init__(self, kind: str, index: int, span: Tuple[int, int], boundary: int) -> None:
        start, end = span
        super().__init__(
            f"{kind} {index}'s tokens [{start}, {end}) straddle the window boundary at token {boundary}; "
            "each window would get all its pixels but only part of its tokens. Move the boundary off "
            "the item: pad or reorder the text before it so it sits inside one window"
        )
        self.kind = kind
        self.index = index
        self.span = (start, end)
        self.boundary = boundary

    def __reduce__(self):
        # Rebuild from the attributes, not the message, so the error survives a trip across processes.
        return type(self), (self.kind, self.index, self.span, self.boundary)


def split_media(batch: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``batch`` without its media leaves, and the media leaves alone.

    Media rides at the top level only. A media key inside a sub-dict (RL's ``context``) is refused rather than
    walked, because the generic rules would otherwise slice it by shape.
    """
    rest: Dict[str, Any] = {}
    media: Dict[str, Any] = {}
    for key, value in batch.items():
        if key in MEDIA_KEYS:
            media[key] = value
        else:
            if isinstance(value, dict):
                nested = sorted(MEDIA_KEYS.intersection(_all_keys(value)))
                if nested:
                    raise MediaBatchError(f"media leaves {nested} must be top-level batch keys, not inside {key!r}")
            rest[key] = value
    return rest, media


def _all_keys(batch: Mapping[str, Any]) -> List[str]:
    keys = []
    for key, value in batch.items():
        keys.append(key)
        if isinstance(value, dict):
            keys.extend(_all_keys(value))
    return keys


def _present_kinds(media: Mapping[str, Any]) -> List[MediaKind]:
    kinds = []
    for kind in MEDIA_KINDS:
        present = [key for key in kind.keys if media.get(key) is not None]
        if not present:
            continue
        if len(present) != len(kind.keys):
            missing = sorted(set(kind.keys) - set(present))
            raise MediaBatchError(f"{kind.name} media needs {list(kind.keys)} together; missing {missing}")
        kinds.append(kind)
    return kinds


def _patch_counts(grid_thw: torch.Tensor) -> torch.Tensor:
    return grid_thw.to(torch.long).prod(dim=1)


def check_media(media: Mapping[str, Any], n_rows: int) -> None:
    """Refuse media whose leaves disagree with each other or with the batch's ``n_rows`` rows."""
    for kind in _present_kinds(media):
        values, grid, row_ids = (media[key] for key in kind.keys)
        if grid.ndim != 2 or grid.shape[1] != 3:
            raise MediaBatchError(f"{kind.grid_key} must be [{kind.name}s, 3], got {tuple(grid.shape)}")
        if row_ids.ndim != 1 or row_ids.shape[0] != grid.shape[0]:
            raise MediaBatchError(
                f"{kind.row_ids_key} must hold one row id per {kind.name} ({grid.shape[0]}), "
                f"got shape {tuple(row_ids.shape)}"
            )
        if row_ids.numel():
            if int(row_ids.min()) < 0 or int(row_ids.max()) >= n_rows:
                raise MediaBatchError(f"{kind.row_ids_key} must be in [0, {n_rows}), got {row_ids.tolist()}")
            if (row_ids[1:] < row_ids[:-1]).any():
                raise MediaBatchError(
                    f"{kind.row_ids_key} must be non-decreasing (media in row order), got {row_ids.tolist()}"
                )
        patches = int(_patch_counts(grid).sum())
        if values.ndim < 1 or int(values.shape[0]) != patches:
            raise MediaBatchError(
                f"{kind.values_key} has {int(values.shape[0]) if values.ndim else 0} patches but "
                f"{kind.grid_key} describes {patches}"
            )


def select_media_rows(media: Mapping[str, Any], index: torch.Tensor, n_rows: int) -> Dict[str, Any]:
    """The media of the rows named by ``index``, in that order, with row ids renumbered to positions in ``index``."""
    check_media(media, n_rows)
    selected: Dict[str, Any] = dict(media)
    rows = [int(row) for row in index.tolist()]
    for kind in _present_kinds(media):
        values, grid, row_ids = (media[key] for key in kind.keys)
        patch_counts = _patch_counts(grid)
        patch_starts = torch.cumsum(patch_counts, dim=0) - patch_counts
        items: List[int] = []
        new_row_ids: List[int] = []
        for position, row in enumerate(rows):
            owned = torch.nonzero(row_ids == row).flatten().tolist()
            items.extend(owned)
            new_row_ids.extend([position] * len(owned))
        item_index = torch.as_tensor(items, dtype=torch.long)
        patch_index = (
            torch.cat([torch.arange(int(patch_starts[i]), int(patch_starts[i] + patch_counts[i])) for i in items])
            if items
            else torch.zeros(0, dtype=torch.long)
        )
        selected[kind.values_key] = values.index_select(0, patch_index.to(values.device))
        selected[kind.grid_key] = grid.index_select(0, item_index.to(grid.device))
        selected[kind.row_ids_key] = torch.as_tensor(new_row_ids, dtype=row_ids.dtype, device=row_ids.device)
    return selected


def media_token_spans(
    media: Mapping[str, Any],
    kind: MediaKind,
    mm_token_type_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    """``[items, 2]`` packed ``[start, end)`` token span of every item of ``kind``, first token to last.

    Items take the call's ``kind`` tokens in packed order. An image's tokens have to be one contiguous run; a
    video's have to be ``t`` contiguous frame runs of ``h * w / merge**2`` tokens with only text between them.
    Each item's span has to sit inside the row its row id names, and the call's ``kind`` token count has to
    equal what the grids expand to. Token types cannot tell one video from the next, so a video's frames are
    the next ``t`` frame runs in packed order, as ``mrope_position_ids`` reads them too.
    """
    grid, row_ids = media[kind.grid_key], media[kind.row_ids_key]
    token_counts = _patch_counts(grid) // (spatial_merge_size**2)
    token_types = mm_token_type_ids.reshape(-1)
    positions = torch.nonzero(token_types == kind.token_type).flatten()
    if int(positions.numel()) != int(token_counts.sum()):
        raise MediaBatchError(
            f"the packed call has {int(positions.numel())} {kind.name} tokens but {kind.grid_key} expands to "
            f"{int(token_counts.sum())} at spatial_merge_size {spatial_merge_size}"
        )
    spans = torch.zeros(int(grid.shape[0]), 2, dtype=torch.long)
    offset = 0
    boundaries = cu_seqlens.tolist()
    for item, count in enumerate(token_counts.tolist()):
        item_positions = positions[offset : offset + count]
        offset += count
        if count == 0:
            raise MediaBatchError(f"{kind.name} {item} expands to no tokens")
        if kind.token_type == VIDEO_TOKEN_TYPE:
            _check_video_frames(item, grid[item], item_positions, token_types, spatial_merge_size)
        elif int(item_positions[-1]) - int(item_positions[0]) + 1 != count:
            raise MediaBatchError(f"{kind.name} {item}'s {count} tokens are not contiguous in the packed call")
        start, last = int(item_positions[0]), int(item_positions[-1])
        row = int(row_ids[item])
        if not (boundaries[row] <= start and last < boundaries[row + 1]):
            raise MediaBatchError(
                f"{kind.name} {item} belongs to row {row} ([{boundaries[row]}, {boundaries[row + 1]})) but its "
                f"tokens are at [{start}, {last + 1})"
            )
        spans[item, 0], spans[item, 1] = start, last + 1
    return spans


def _check_video_frames(
    item: int,
    grid: torch.Tensor,
    positions: torch.Tensor,
    token_types: torch.Tensor,
    spatial_merge_size: int,
) -> None:
    """Refuse a video whose ``positions`` are not its frames' contiguous runs with only text between them.

    Frames may also sit back to back, with no text between, so a run is cut into frames by the grid alone.
    """
    frames, height, width = (int(value) for value in grid.tolist())
    frame_tokens = height * width // (spatial_merge_size**2)
    if frame_tokens * frames != int(positions.numel()):
        raise MediaBatchError(
            f"video {item}'s grid {[frames, height, width]} does not split into {frames} frames of whole merged "
            f"tokens at spatial_merge_size {spatial_merge_size}"
        )
    previous_last = None
    for frame in range(frames):
        frame_positions = positions[frame * frame_tokens : (frame + 1) * frame_tokens]
        start, last = int(frame_positions[0]), int(frame_positions[-1])
        if last - start + 1 != frame_tokens:
            raise MediaBatchError(
                f"video {item}'s frame {frame} ({frame_tokens} tokens) is not contiguous in the packed call"
            )
        if previous_last is not None:
            between = token_types[previous_last + 1 : start]
            if (between != TEXT_TOKEN_TYPE).any():
                raise MediaBatchError(
                    f"video {item} has non-text tokens (types {sorted(set(between.tolist()) - {TEXT_TOKEN_TYPE})}) "
                    f"between its frames {frame - 1} and {frame}, at [{previous_last + 1}, {start})"
                )
        previous_last = last


def window_media(
    media: Mapping[str, Any],
    mm_token_type_ids: Optional[torch.Tensor],
    cu_seqlens: torch.Tensor,
    window: Tuple[int, int],
    *,
    spatial_merge_size: Optional[int],
    boundaries: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """The media one ``[start, end)`` token window needs: every item whose span sits inside it, whole.

    An item whose span straddles a window boundary -- some of its tokens before it and some at or after it -- is
    refused with a ``MediaStraddleError`` naming the item, its span and the boundary. ``boundaries`` are the token
    offsets where the call's windows meet, so that every window of one call refuses the same straddle; they
    default to this window's own ``start`` and ``end``. To avoid a straddle, move the boundary off the item: pad or
    reorder the text before the item so it lands wholly in one window.

    A kind with no item in the window keeps its keys, with zero items, so every window of a call carries the
    same set of leaves.
    """
    kinds = _present_kinds(media)
    if not kinds:
        return dict(media)
    if spatial_merge_size is None:
        raise ValueError("splitting a call that carries media needs spatial_merge_size to locate media tokens")
    if mm_token_type_ids is None:
        raise MediaBatchError("splitting a call that carries media needs its packed mm_token_type_ids")
    window_start, window_end = window
    cuts = sorted({window_start, window_end} if boundaries is None else {int(cut) for cut in boundaries})
    result: Dict[str, Any] = dict(media)
    for kind in kinds:
        values, grid, row_ids = (media[key] for key in kind.keys)
        spans = media_token_spans(media, kind, mm_token_type_ids, cu_seqlens, spatial_merge_size)
        for item, (start, end) in enumerate(spans.tolist()):
            for cut in cuts:
                if start < cut < end:
                    raise MediaStraddleError(kind.name, item, (start, end), cut)
        inside = torch.nonzero((spans[:, 0] >= window_start) & (spans[:, 1] <= window_end)).flatten()
        patch_counts = _patch_counts(grid)
        patch_starts = torch.cumsum(patch_counts, dim=0) - patch_counts
        patch_index = (
            torch.cat([torch.arange(int(patch_starts[i]), int(patch_starts[i] + patch_counts[i])) for i in inside])
            if inside.numel()
            else torch.zeros(0, dtype=torch.long)
        )
        result[kind.values_key] = values.index_select(0, patch_index.to(values.device))
        result[kind.grid_key] = grid.index_select(0, inside.to(grid.device))
        result[kind.row_ids_key] = row_ids.index_select(0, inside.to(row_ids.device))
    return result
