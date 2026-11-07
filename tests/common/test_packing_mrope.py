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

"""M-RoPE position ids for packed rows: pure tensor plumbing, no model, no GPU.

The image expectations are the hand-written tensors of prime-rl's packed M-RoPE tests for the same model family
(channels 1-3 here are its three channels); channel 0, video, and padding are this layout's additions.
"""

import pytest
import torch

from arctic_platform.common.packing import cu_seqlens_from_position_ids
from arctic_platform.common.packing import mrope_position_ids
from arctic_platform.common.packing import vision_position_ids


def _cu(*lengths):
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)


def _types(*values):
    return torch.tensor([values])


def test_text_only_row_is_arange_on_every_channel():
    position_ids = mrope_position_ids(_types(0, 0, 0, 0), _cu(4), spatial_merge_size=2)

    assert position_ids.shape == (4, 1, 4)
    assert position_ids.dtype == torch.long
    torch.testing.assert_close(position_ids, torch.arange(4).view(1, 1, -1).expand(4, 1, -1))


def test_single_image_with_surrounding_text():
    position_ids = mrope_position_ids(
        _types(0, 0, 1, 1, 1, 1, 0),
        _cu(7),
        spatial_merge_size=2,
        image_grid_thw=torch.tensor([[1, 4, 4]]),
    )

    expected = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 2, 2, 2, 4],
            [0, 1, 2, 2, 3, 3, 4],
            [0, 1, 2, 3, 2, 3, 4],
        ]
    )
    torch.testing.assert_close(position_ids[:, 0], expected)


def test_adjacent_images_consume_one_grid_each():
    position_ids = mrope_position_ids(
        _types(0, *[1] * 8, 0),
        _cu(10),
        spatial_merge_size=2,
        image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 4]]),
    )

    expected_rope = torch.tensor(
        [
            [0, 1, 1, 1, 1, 3, 3, 3, 3, 5],
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
            [0, 1, 2, 1, 2, 3, 4, 3, 4, 5],
        ]
    )
    torch.testing.assert_close(position_ids[1:, 0], expected_rope)
    torch.testing.assert_close(position_ids[0, 0], torch.arange(10))


def test_non_square_grid_advances_by_its_longest_side():
    # 2 x 6 patches merge to 1 x 3 tokens; the text after resumes 3 past the image's start.
    position_ids = mrope_position_ids(
        _types(0, 1, 1, 1, 0),
        _cu(5),
        spatial_merge_size=2,
        image_grid_thw=torch.tensor([[1, 2, 6]]),
    )

    torch.testing.assert_close(
        position_ids[1:, 0],
        torch.tensor(
            [
                [0, 1, 1, 1, 4],
                [0, 1, 1, 1, 4],
                [0, 1, 2, 3, 4],
            ]
        ),
    )


def test_packed_rows_restart_independently():
    position_ids = mrope_position_ids(
        _types(0, 1, 1, 1, 1, 0, 1, 1, 1, 1),
        _cu(5, 5),
        spatial_merge_size=2,
        image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 4]]),
    )

    expected_row = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 2, 2],
            [0, 1, 2, 1, 2],
        ]
    )
    torch.testing.assert_close(position_ids[:, 0, :5], expected_row)
    torch.testing.assert_close(position_ids[:, 0, 5:], expected_row)


def test_packed_rows_match_the_rows_built_alone():
    rows = [
        (_types(0, 1, 1, 1, 1, 0), torch.tensor([[1, 4, 4]]), None),
        (_types(0, 0, 0), None, None),
        (_types(0, *[2] * 8, 0, 0), None, torch.tensor([[2, 4, 4]])),
    ]
    alone = [
        mrope_position_ids(types, _cu(types.shape[1]), spatial_merge_size=2, image_grid_thw=img, video_grid_thw=vid)
        for types, img, vid in rows
    ]

    packed = mrope_position_ids(
        torch.cat([types for types, _, _ in rows], dim=1),
        _cu(*[types.shape[1] for types, _, _ in rows]),
        spatial_merge_size=2,
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        video_grid_thw=torch.tensor([[2, 4, 4]]),
    )

    torch.testing.assert_close(packed, torch.cat(alone, dim=2))


def test_cu_seqlens_round_trip_through_the_text_channel():
    cu_seqlens = _cu(6, 3, 4)
    types = _types(0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0)  # rows of 6, 3, 4 + 3 tail pad
    position_ids = mrope_position_ids(
        types,
        cu_seqlens,
        spatial_merge_size=2,
        image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 4]]),
    )

    rebuilt = cu_seqlens_from_position_ids(position_ids)
    # Each pad token reads as its own one-token segment, as with 2-D positions.
    assert rebuilt.tolist() == [0, 6, 9, 13, 14, 15, 16]


def test_tail_padding_is_zero_on_every_channel():
    position_ids = mrope_position_ids(_types(0, 0, 0, 0, 0), _cu(3), spatial_merge_size=2)

    torch.testing.assert_close(position_ids[:, 0, 3:], torch.zeros(4, 2, dtype=torch.long))


def test_video_frames_are_placed_like_consecutive_images():
    # A [2, 4, 4] video is two [1, 4, 4] frames of 2 x 2 merged tokens each.
    position_ids = mrope_position_ids(
        _types(0, *[2] * 8, 0),
        _cu(10),
        spatial_merge_size=2,
        video_grid_thw=torch.tensor([[2, 4, 4]]),
    )

    torch.testing.assert_close(
        position_ids[1:, 0],
        torch.tensor(
            [
                [0, 1, 1, 1, 1, 3, 3, 3, 3, 5],
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                [0, 1, 2, 1, 2, 3, 4, 3, 4, 5],
            ]
        ),
    )


def test_video_frames_split_by_text_continue_the_same_video():
    # Text between frames (a timestamp, say) sits between them on every axis.
    position_ids = mrope_position_ids(
        _types(*[2] * 4, 0, *[2] * 4),
        _cu(9),
        spatial_merge_size=2,
        video_grid_thw=torch.tensor([[2, 4, 4]]),
    )

    torch.testing.assert_close(position_ids[1, 0], torch.tensor([0, 0, 0, 0, 2, 3, 3, 3, 3]))


def test_vision_position_ids_are_row_major_over_t_h_w():
    positions = vision_position_ids(5, torch.tensor([2, 2, 4]), spatial_merge_size=2)

    torch.testing.assert_close(
        positions,
        torch.tensor(
            [
                [5, 5, 6, 6],
                [5, 5, 5, 5],
                [5, 6, 5, 6],
            ]
        ),
    )


@pytest.mark.parametrize(
    "types, cu_seqlens, grids, match",
    [
        (_types(0, 1, 1), _cu(3), torch.tensor([[1, 4, 4]]), "does not match image_grid_thw"),
        (_types(0, 1, 1, 1, 1, 1, 1, 1, 1), _cu(9), torch.tensor([[1, 4, 4]]), "not enough image_grid_thw"),
        (_types(0, 1, 1, 1, 1), _cu(5), torch.tensor([[1, 4, 4], [1, 4, 4]]), "rows no image tokens use"),
        (_types(0, 1, 1, 1, 1), _cu(5), None, "image_grid_thw is required"),
        (_types(0, 3), _cu(2), None, "unsupported multimodal token type 3"),
        (_types(1, 1, 1, 1, 0), _cu(2, 3), torch.tensor([[1, 4, 4]]), "does not match image_grid_thw"),
        (_types(0, 0, 1), _cu(2), None, "tail padding"),
        (_types(0, 1, 1), _cu(3), torch.tensor([[1, 3, 3]]), "does not divide by spatial_merge_size"),
    ],
    ids=[
        "run-shorter-than-grid",
        "too-few-grids",
        "unused-grid",
        "missing-grids",
        "unknown-type",
        "image-crosses-row-boundary",
        "media-in-tail-padding",
        "grid-not-mergeable",
    ],
)
def test_inconsistent_inputs_are_rejected(types, cu_seqlens, grids, match):
    with pytest.raises(ValueError, match=match):
        mrope_position_ids(types, cu_seqlens, spatial_merge_size=2, image_grid_thw=grids)


def test_partly_used_video_is_rejected():
    with pytest.raises(ValueError, match="rows no video tokens use"):
        mrope_position_ids(
            _types(0, 2, 2, 2, 2), _cu(5), spatial_merge_size=2, video_grid_thw=torch.tensor([[2, 4, 4]])
        )
