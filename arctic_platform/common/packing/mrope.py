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

"""Qwen-style multimodal rotary (M-RoPE) position ids for a packed call.

Qwen vision-language models rotate each token by three positions instead of one: temporal, height and width. A
text token advances all three together. A run of image tokens lays out its merged patch grid from the current
position -- temporal index on the first axis, row on the second, column on the third -- and the text after it
resumes at the current position plus the grid's largest side.

The returned layout has four channels, ``[4, 1, T]``:

* channel 0 is the plain text position, ``0 .. len - 1`` within every packed row. It is what
  ``cu_seqlens_from_position_ids`` reads (a 3-D tensor is read from its channel 0), so varlen boundaries still
  derive from the positions, and what packed-attention code reads to find where rows start;
* channels 1-3 are the temporal, height and width positions the rotary embedding consumes.

Every row restarts at 0 on every channel, so a packed row gets exactly the positions it would get alone. Tail
padding past the last row (``pad_packed_microbatch``) gets 0 on every channel, matching the 2-D convention.

Token types follow the Hugging Face ``mm_token_type_ids`` convention: 0 text, 1 image, 2 video. Grids are
``[t, h, w]`` in patches before the spatial merge, one row per image (or video) in packed order across rows.
A video is laid out as ``t`` frames of ``[1, h, w]``, each placed like an image, which is the Qwen3-VL layout;
consecutive frames with no text between them advance the position frame by frame.
"""

from __future__ import annotations

import itertools
from typing import Iterator
from typing import List
from typing import Optional

import torch

TEXT_TOKEN_TYPE = 0
IMAGE_TOKEN_TYPE = 1
VIDEO_TOKEN_TYPE = 2

MROPE_CHANNELS = 4


def vision_position_ids(
    start_position: int,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """``[3, t * h * w / merge**2]`` temporal/height/width positions of one merged patch grid.

    Tokens run in row-major order over ``(t, h, w)``, which is the order the vision tower emits merged patches.
    """
    grid_t, grid_h, grid_w = (int(value) for value in grid_thw.tolist())
    if grid_h % spatial_merge_size or grid_w % spatial_merge_size:
        raise ValueError(f"grid {[grid_t, grid_h, grid_w]} does not divide by spatial_merge_size {spatial_merge_size}")
    grid_h //= spatial_merge_size
    grid_w //= spatial_merge_size
    temporal = torch.arange(grid_t, device=device) + start_position
    height = torch.arange(grid_h, device=device) + start_position
    width = torch.arange(grid_w, device=device) + start_position
    return torch.stack(
        [
            temporal.repeat_interleave(grid_h * grid_w),
            height.repeat_interleave(grid_w).repeat(grid_t),
            width.repeat(grid_h * grid_t),
        ]
    )


def _grid_extent(grid_thw: torch.Tensor, spatial_merge_size: int) -> int:
    """How far one grid advances the position: its largest side, after the merge."""
    grid_t, grid_h, grid_w = (int(value) for value in grid_thw.tolist())
    return max(grid_t, grid_h // spatial_merge_size, grid_w // spatial_merge_size)


def _video_frames(video_grid_thw: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Each video grid ``[t, h, w]`` as ``t`` frame grids ``[1, h, w]``."""
    if video_grid_thw is None:
        return None
    frames = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0).clone()
    frames[:, 0] = 1
    return frames


def _media_positions(
    grids: Optional[Iterator[torch.Tensor]],
    run_length: int,
    current_position: int,
    spatial_merge_size: int,
    device: torch.device,
    what: str,
) -> tuple[List[torch.Tensor], int]:
    """Positions for one run of ``run_length`` media tokens, consuming as many grids as the run holds."""
    if grids is None:
        raise ValueError(f"{what}_grid_thw is required for {what} tokens")
    pieces: List[torch.Tensor] = []
    remaining = run_length
    while remaining:
        grid = next(grids, None)
        if grid is None:
            raise ValueError(f"not enough {what}_grid_thw rows for the {what} tokens")
        positions = vision_position_ids(current_position, grid, spatial_merge_size, device=device)
        if positions.shape[1] > remaining:
            raise ValueError(
                f"{what} token run of {run_length} does not match {what}_grid_thw: a grid of "
                f"{positions.shape[1]} tokens needs more than the {remaining} left in the run"
            )
        pieces.append(positions)
        current_position += _grid_extent(grid, spatial_merge_size)
        remaining -= positions.shape[1]
    return pieces, current_position


def mrope_position_ids(
    mm_token_type_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    spatial_merge_size: int,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``[4, 1, T]`` M-RoPE position ids for a packed call (see the module docstring for the layout).

    ``mm_token_type_ids`` is the packed call's ``[1, T]`` (or ``[T]``) token types and ``cu_seqlens`` its row
    boundaries, as ``PackMetadata.cu_seqlens`` records them. Tokens past ``cu_seqlens[-1]`` are tail padding and
    must be text. A media run may not cross a row boundary, and every grid must be used exactly once.
    """
    token_types = mm_token_type_ids.reshape(-1)
    device = token_types.device
    total_tokens = int(token_types.numel())
    boundaries = [int(value) for value in cu_seqlens.tolist()]
    if not boundaries or boundaries[0] != 0 or boundaries[-1] > total_tokens:
        raise ValueError(f"cu_seqlens {boundaries} do not describe rows inside {total_tokens} packed tokens")
    if (token_types[boundaries[-1] :] != TEXT_TOKEN_TYPE).any():
        raise ValueError("tail padding past the last packed row must have token type 0 (text)")

    images = iter(image_grid_thw) if image_grid_thw is not None else None
    frames_grid = _video_frames(video_grid_thw)
    frames = iter(frames_grid) if frames_grid is not None else None

    position_ids = torch.zeros(MROPE_CHANNELS, 1, total_tokens, dtype=torch.long, device=device)
    for row_start, row_end in zip(boundaries[:-1], boundaries[1:]):
        row_length = row_end - row_start
        if row_length < 0:
            raise ValueError(f"cu_seqlens must be non-decreasing, got {boundaries}")
        current_position = 0
        pieces: List[torch.Tensor] = []
        row_types = token_types[row_start:row_end].tolist()
        for token_type, run in itertools.groupby(row_types):
            run_length = sum(1 for _ in run)
            if token_type == TEXT_TOKEN_TYPE:
                text = torch.arange(current_position, current_position + run_length, device=device)
                pieces.append(text.unsqueeze(0).expand(3, -1))
                current_position += run_length
            elif token_type == IMAGE_TOKEN_TYPE:
                media, current_position = _media_positions(
                    images, run_length, current_position, spatial_merge_size, device, "image"
                )
                pieces.extend(media)
            elif token_type == VIDEO_TOKEN_TYPE:
                media, current_position = _media_positions(
                    frames, run_length, current_position, spatial_merge_size, device, "video"
                )
                pieces.extend(media)
            else:
                raise ValueError(f"unsupported multimodal token type {token_type}")
        if pieces:
            position_ids[1:, 0, row_start:row_end] = torch.cat(pieces, dim=1)
        position_ids[0, 0, row_start:row_end] = torch.arange(row_length, device=device)

    for leftover, what in ((images, "image"), (frames, "video")):
        if leftover is not None and next(leftover, None) is not None:
            raise ValueError(f"{what}_grid_thw has rows no {what} tokens use")
    return position_ids
