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

"""Row-owned media through row selection, packing, padding and token windows: pure tensors, no model."""

import pytest
import torch

from arctic_platform.common.packing import cu_seqlens_from_position_ids
from arctic_platform.common.packing import media_token_spans
from arctic_platform.common.packing import mrope_position_ids
from arctic_platform.common.packing import pack_microbatch
from arctic_platform.common.packing import pad_packed_microbatch
from arctic_platform.common.packing import select_rows
from arctic_platform.common.packing import token_shard
from arctic_platform.common.packing.media import VIDEO_MEDIA

MERGE = 2

# Rows of the reference batch, as (token types, image grids). Image tokens: prod(grid) / MERGE**2.
#   row 0: text, image A (4 tokens), text, text            -> 7 tokens
#   row 1: text only                                       -> 3 tokens
#   row 2: text, image B (2 tokens), text, image C (2)     -> 6 tokens
ROWS = [
    ([0, 1, 1, 1, 1, 0, 0], [[1, 4, 4]]),
    ([0, 0, 0], []),
    ([0, 1, 1, 0, 1, 1], [[1, 2, 4], [1, 2, 4]]),
]


def _patches(item: int, count: int, dim: int) -> torch.Tensor:
    """Patches that name their owner: column 0 is the item, column 1 the patch index within it."""
    values = torch.zeros(count, dim)
    values[:, 0] = item
    values[:, 1] = torch.arange(count)
    return values


def _batch(rows=ROWS, patch_dim=4, kind="image"):
    width = max(len(types) for types, _ in rows)
    n = len(rows)
    batch = {
        "input_ids": torch.zeros(n, width, dtype=torch.long),
        "position_ids": torch.zeros(n, width, dtype=torch.long),
        "attention_mask": torch.zeros(n, width, dtype=torch.long),
        "mm_token_type_ids": torch.zeros(n, width, dtype=torch.long),
        "labels": torch.full((n, width), -100, dtype=torch.long),
    }
    grids, row_ids, patches = [], [], []
    token_type = 1 if kind == "image" else 2
    for row, (types, row_grids) in enumerate(rows):
        length = len(types)
        batch["attention_mask"][row, :length] = 1
        batch["position_ids"][row, :length] = torch.arange(length)
        batch["mm_token_type_ids"][row, :length] = torch.tensor(types) * token_type
        batch["input_ids"][row, :length] = 100 * row + torch.arange(length)
        for grid in row_grids:
            patches.append(_patches(len(grids), int(torch.tensor(grid).prod()), patch_dim))
            grids.append(grid)
            row_ids.append(row)
    keys = (
        ("pixel_values", "image_grid_thw", "image_row_ids")
        if kind == "image"
        else ("pixel_values_videos", "video_grid_thw", "video_row_ids")
    )
    batch[keys[0]] = torch.cat(patches) if patches else torch.zeros(0, patch_dim)
    batch[keys[1]] = torch.tensor(grids, dtype=torch.long).reshape(-1, 3)
    batch[keys[2]] = torch.tensor(row_ids, dtype=torch.long)
    return batch


def _owners(pixel_values):
    return sorted(set(int(v) for v in pixel_values[:, 0].tolist()))


# ─── Row selection ──────────────────────────────────────────────────────


def test_select_rows_moves_media_with_its_rows_and_renumbers_row_ids():
    batch = _batch()
    selected = select_rows(batch, torch.tensor([2, 0]), n_rows=3)

    # Row 2's images B and C come first now, then row 0's image A.
    assert selected["image_row_ids"].tolist() == [0, 0, 1]
    assert selected["image_grid_thw"].tolist() == [[1, 2, 4], [1, 2, 4], [1, 4, 4]]
    owners = selected["pixel_values"][:, 0].tolist()
    assert owners == [1.0] * 8 + [2.0] * 8 + [0.0] * 16
    torch.testing.assert_close(selected["pixel_values"][:8], batch["pixel_values"][16:24])
    assert selected["input_ids"][0, :6].tolist() == batch["input_ids"][2, :6].tolist()


def test_select_rows_does_not_mistake_patches_for_rows_when_the_counts_coincide():
    # Two rows and one image of exactly two patches: the generic rule would treat the patches as rows.
    batch = _batch(rows=[([0, 0], []), ([0, 0], [])])
    batch["pixel_values"] = _patches(0, 2, 4)
    batch["image_grid_thw"] = torch.tensor([[1, 1, 2]])
    batch["image_row_ids"] = torch.tensor([0])

    selected = select_rows(batch, torch.tensor([1, 0]), n_rows=2)

    torch.testing.assert_close(selected["pixel_values"], batch["pixel_values"])
    assert selected["image_row_ids"].tolist() == [1]


def test_select_rows_drops_media_of_rows_left_out():
    selected = select_rows(_batch(), torch.tensor([1]), n_rows=3)

    assert selected["pixel_values"].shape == (0, 4)
    assert selected["image_grid_thw"].shape == (0, 3)
    assert selected["image_row_ids"].tolist() == []


# ─── Packing and padding ────────────────────────────────────────────────


def test_pack_passes_media_through_unchanged():
    batch = _batch()
    packed, metadata = pack_microbatch(batch)

    for key in ("pixel_values", "image_grid_thw", "image_row_ids"):
        assert packed[key] is batch[key]
    assert packed["mm_token_type_ids"].shape == (1, 16)
    assert metadata.cu_seqlens.tolist() == [0, 7, 10, 16]


def test_pad_leaves_media_alone_even_when_a_dimension_matches_the_token_count():
    # 16 packed tokens and a patch_dim of 16: the generic rule would pad pixel_values on its second axis.
    batch = _batch(patch_dim=16)
    packed, metadata = pack_microbatch(batch)
    padded, padded_metadata = pad_packed_microbatch(packed, metadata, multiple=6)

    assert padded_metadata.padded_tokens == 18
    assert padded["pixel_values"].shape == (32, 16)
    assert padded["mm_token_type_ids"].shape == (1, 18)
    assert padded["mm_token_type_ids"][0, 16:].tolist() == [0, 0]


def test_pad_extends_multimodal_positions_with_zeros():
    packed, metadata = pack_microbatch(_batch())
    packed["position_ids"] = _mrope(packed, metadata)

    padded, _ = pad_packed_microbatch(packed, metadata, multiple=6)

    assert padded["position_ids"].shape == (4, 1, 18)
    assert padded["position_ids"][:, 0, 16:].eq(0).all()
    torch.testing.assert_close(padded["position_ids"][:, :, :16], packed["position_ids"])


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda b: b.update(image_row_ids=torch.tensor([0, 3, 3])), r"must be in \[0, 3\)"),
        (lambda b: b.update(image_row_ids=torch.tensor([2, 2, 0])), "non-decreasing"),
        (lambda b: b.update(pixel_values=b["pixel_values"][:-1]), "31 patches but image_grid_thw describes 32"),
        (lambda b: b.update(image_row_ids=torch.tensor([0, 2])), "one row id per image"),
        (lambda b: b.pop("image_row_ids"), r"missing \['image_row_ids'\]"),
        (lambda b: b.update(context={"pixel_values": b.pop("pixel_values")}), "must be top-level"),
    ],
    ids=["row-out-of-range", "rows-out-of-order", "patch-count", "row-id-count", "partial-keys", "nested"],
)
def test_pack_rejects_inconsistent_media(mutate, match):
    batch = _batch()
    mutate(batch)
    with pytest.raises(ValueError, match=match):
        pack_microbatch(batch)


# ─── Token windows ──────────────────────────────────────────────────────


def _mrope(packed, metadata):
    return mrope_position_ids(
        packed["mm_token_type_ids"],
        metadata.cu_seqlens,
        spatial_merge_size=MERGE,
        image_grid_thw=packed.get("image_grid_thw"),
        video_grid_thw=packed.get("video_grid_thw"),
    )


def _windows(batch, shard_count, multiple=None):
    packed, metadata = pack_microbatch(batch)
    packed, metadata = pad_packed_microbatch(packed, metadata, multiple or shard_count)
    packed["position_ids"] = _mrope(packed, metadata)
    windows = [token_shard(packed, metadata, shard_count, i, spatial_merge_size=MERGE) for i in range(shard_count)]
    return packed, metadata, windows


def test_each_window_gets_the_whole_images_whose_tokens_it_holds():
    # Packed layout: row 0 [0, 7) with image A at 1..4, row 1 [7, 10), row 2 [10, 16) with B at 11..12 and
    # C at 14..15. Four windows of four tokens: A straddles windows 0 and 1, B straddles 2 and 3.
    _, _, windows = _windows(_batch(), shard_count=4)

    assert [_owners(w["pixel_values"]) for w in windows] == [[0], [0], [1], [1, 2]]
    assert [w["image_row_ids"].tolist() for w in windows] == [[0], [0], [2], [2, 2]]
    assert [w["image_window_token_ranges"].tolist() for w in windows] == [
        [[0, 3]],
        [[3, 4]],
        [[0, 1]],
        [[1, 2], [0, 2]],
    ]
    # A window carries whole images: every one of A's 16 patches, in order, in both windows that touch it.
    for window in windows[:2]:
        torch.testing.assert_close(window["pixel_values"][:, 1], torch.arange(16, dtype=torch.float))


def test_window_ranges_select_exactly_the_windows_image_tokens():
    batch = _batch()
    packed, _, windows = _windows(batch, shard_count=4)
    merged_tokens = {0: 4, 1: 2, 2: 2}

    covered = {item: [] for item in merged_tokens}
    for window in windows:
        image_slots = int((window["mm_token_type_ids"] == 1).sum())
        ranges = window["image_window_token_ranges"]
        # The ranges fill this window's image-token slots exactly, in order.
        assert int((ranges[:, 1] - ranges[:, 0]).sum()) == image_slots
        owners = []
        for patch_owner in window["pixel_values"][:, 0].tolist():
            if not owners or owners[-1] != int(patch_owner):
                owners.append(int(patch_owner))
        for owner, (first, end) in zip(owners, ranges.tolist()):
            covered[owner].extend(range(first, end))
    # Across the windows, every merged embedding of every image is placed once.
    assert covered == {item: list(range(count)) for item, count in merged_tokens.items()}


def test_window_without_media_keeps_empty_media_leaves():
    # Three windows of four tokens: the first holds only row 0, which is text.
    rows = [([0, 0, 0, 0], []), ([0, 1, 1, 1, 1, 0, 0, 0], [[1, 4, 4]])]
    _, _, windows = _windows(_batch(rows=rows), shard_count=3)

    empty = windows[0]
    assert empty["pixel_values"].shape == (0, 4)
    assert empty["image_grid_thw"].shape == (0, 3)
    assert empty["image_row_ids"].shape == (0,)
    assert empty["image_window_token_ranges"].shape == (0, 2)


def test_multimodal_positions_are_cut_on_the_token_axis():
    packed, metadata, windows = _windows(_batch(), shard_count=4)

    assert all(w["position_ids"].shape == (4, 1, 4) for w in windows)
    gathered = torch.cat([w["position_ids"] for w in windows], dim=2)
    torch.testing.assert_close(gathered, packed["position_ids"])
    assert cu_seqlens_from_position_ids(gathered).tolist() == metadata.cu_seqlens.tolist()


def test_patch_dim_equal_to_the_padded_width_is_not_sliced():
    _, _, windows = _windows(_batch(patch_dim=16), shard_count=4)

    assert [w["pixel_values"].shape for w in windows] == [(16, 16), (16, 16), (8, 16), (16, 16)]


def test_video_frames_follow_the_same_rules():
    # A [2, 4, 4] video is 8 merged tokens; two windows of 6 split it 5 / 3.
    rows = [([0, *[1] * 8, 0, 0, 0], [[2, 4, 4]])]
    packed, _, windows = _windows(_batch(rows=rows, kind="video"), shard_count=2)

    assert [w["video_window_token_ranges"].tolist() for w in windows] == [[[0, 5]], [[5, 8]]]
    assert all(w["pixel_values_videos"].shape == (32, 4) for w in windows)
    assert packed["position_ids"][1, 0].tolist() == [0, 1, 1, 1, 1, 3, 3, 3, 3, 5, 6, 7]


def test_selected_rows_pack_and_window_like_a_batch_built_in_that_order():
    batch = _batch()
    _, _, from_selection = _windows(select_rows(batch, torch.tensor([2, 0]), n_rows=3), shard_count=2, multiple=2)
    _, _, built = _windows(_batch(rows=[ROWS[2], ROWS[0]]), shard_count=2, multiple=2)

    for got, want in zip(from_selection, built):
        assert got["image_row_ids"].tolist() == want["image_row_ids"].tolist()
        assert got["image_window_token_ranges"].tolist() == want["image_window_token_ranges"].tolist()
        torch.testing.assert_close(got["position_ids"], want["position_ids"])
        torch.testing.assert_close(got["pixel_values"][:, 1], want["pixel_values"][:, 1])


def test_splitting_media_needs_the_merge_size():
    packed, metadata = pack_microbatch(_batch())
    with pytest.raises(ValueError, match="needs spatial_merge_size"):
        token_shard(packed, metadata, 1, 0)


def test_splitting_media_needs_token_types():
    batch = _batch()
    batch.pop("mm_token_type_ids")
    packed, metadata = pack_microbatch(batch)
    with pytest.raises(ValueError, match="needs its packed mm_token_type_ids"):
        token_shard(packed, metadata, 1, 0, spatial_merge_size=MERGE)


def test_grids_that_disagree_with_the_image_tokens_are_rejected():
    batch = _batch()
    batch["image_grid_thw"] = torch.tensor([[1, 4, 4], [1, 2, 4], [1, 2, 2]])
    batch["pixel_values"] = batch["pixel_values"][:28]
    packed, metadata = pack_microbatch(batch)
    with pytest.raises(ValueError, match="has 8 image tokens but image_grid_thw expands to 7"):
        token_shard(packed, metadata, 1, 0, spatial_merge_size=MERGE)


def test_image_whose_tokens_sit_in_another_row_is_rejected():
    # Row ids say A belongs to row 1, but its tokens are in row 0.
    batch = _batch()
    batch["image_row_ids"] = torch.tensor([1, 2, 2])
    packed, metadata = pack_microbatch(batch)
    with pytest.raises(ValueError, match="image 0 belongs to row 1"):
        token_shard(packed, metadata, 1, 0, spatial_merge_size=MERGE)


# ─── Videos laid out frame by frame ─────────────────────────────────────

# The Qwen3-VL processor writes a video as one "<t seconds><vision_start>frame<vision_end>" per frame, so text
# sits between a video's frames. Token types here: 0 text, 1 image, 2 video. A [t, 2, 4] video is t frames of
# 2 merged tokens; a [1, 2, 4] image is 2 tokens.
T, I, V = 0, 1, 2


def _mixed_batch(rows, patch_dim=4):
    """A batch of rows given as (token types, image grids, video grids), patches naming their owner per kind."""
    batch = _batch(rows=[(types, []) for types, _, _ in rows], patch_dim=patch_dim)
    for row, (types, _, _) in enumerate(rows):
        batch["mm_token_type_ids"][row, : len(types)] = torch.tensor(types)
    for kind, column in (("image", 1), ("video", 2)):
        prefix = "image" if kind == "image" else "video"
        grids, row_ids, patches = [], [], []
        for row, entry in enumerate(rows):
            for grid in entry[column]:
                patches.append(_patches(len(grids), int(torch.tensor(grid).prod()), patch_dim))
                grids.append(grid)
                row_ids.append(row)
        batch["pixel_values" if kind == "image" else "pixel_values_videos"] = (
            torch.cat(patches) if patches else torch.zeros(0, patch_dim)
        )
        batch[f"{prefix}_grid_thw"] = torch.tensor(grids, dtype=torch.long).reshape(-1, 3)
        batch[f"{prefix}_row_ids"] = torch.tensor(row_ids, dtype=torch.long)
    return batch


def _video_spans(batch):
    packed, metadata = pack_microbatch(batch)
    return media_token_spans(packed, VIDEO_MEDIA, packed["mm_token_type_ids"], metadata.cu_seqlens, MERGE)


# Row 0: one 3-frame video, each frame after its timestamp text  -> 12 tokens, video at [1, 11)
# Row 1: an image, then two videos of 2 frames                   -> 16 tokens
VIDEO_ROWS = [
    ([T, V, V, T, T, V, V, T, T, V, V, T], [], [[3, 2, 4]]),
    ([T, I, I, T, V, V, T, V, V, T, T, V, V, T, V, V], [[1, 2, 4]], [[2, 2, 4], [2, 2, 4]]),
]


def test_a_video_spans_its_frames_and_the_text_between_them():
    spans = _video_spans(_mixed_batch(VIDEO_ROWS[:1]))

    assert spans.tolist() == [[1, 11]]


def test_several_videos_in_packed_rows_with_images_each_span_their_own_frames():
    # Row 1 starts at packed token 12: its videos are at 16..20 and 23..27. Row 2 starts at 28.
    rows = [*VIDEO_ROWS, ([T, V, V, V, V, T], [], [[2, 2, 4]])]
    spans = _video_spans(_mixed_batch(rows))

    # The last row's two frames sit back to back, with no text between.
    assert spans.tolist() == [[1, 11], [16, 21], [23, 28], [29, 33]]


def test_window_ranges_count_a_videos_embeddings_not_the_text_between_its_frames():
    # Packed: row 0 [0, 12) with the video's frames at 1..2, 5..6, 9..10. Three windows of four tokens.
    packed, _, windows = _windows(_mixed_batch(VIDEO_ROWS[:1]), shard_count=3)

    assert [w["video_window_token_ranges"].tolist() for w in windows] == [[[0, 2]], [[2, 4]], [[4, 6]]]
    # Each window carries the whole video: all three frames' patches.
    assert all(w["pixel_values_videos"].shape == (24, 4) for w in windows)
    assert packed["position_ids"].shape == (4, 1, 12)


def test_window_holding_only_text_between_frames_still_gets_the_video_with_an_empty_range():
    rows = [([T, V, V, T, T, T, T, V, V, T], [], [[2, 2, 4]])]
    _, _, windows = _windows(_mixed_batch(rows), shard_count=5)

    assert [w["video_window_token_ranges"].tolist() for w in windows] == [
        [[0, 1]],
        [[1, 2]],
        [[2, 2]],
        [[2, 3]],
        [[3, 4]],
    ]


def test_window_ranges_fill_each_windows_video_slots_and_place_every_embedding_once():
    batch = _mixed_batch(VIDEO_ROWS)
    for shard_count in (1, 3, 9):
        _, _, windows = _windows(batch, shard_count=shard_count)
        covered = {0: [], 1: [], 2: []}
        for window in windows:
            ranges = window["video_window_token_ranges"]
            assert int((ranges[:, 1] - ranges[:, 0]).sum()) == int((window["mm_token_type_ids"] == V).sum())
            image_ranges = window["image_window_token_ranges"]
            assert int((image_ranges[:, 1] - image_ranges[:, 0]).sum()) == int(
                (window["mm_token_type_ids"] == I).sum()
            )
            owners = []
            for patch_owner in window["pixel_values_videos"][:, 0].tolist():
                if not owners or owners[-1] != int(patch_owner):
                    owners.append(int(patch_owner))
            for owner, (first, end) in zip(owners, ranges.tolist()):
                covered[owner].extend(range(first, end))
        assert covered == {0: list(range(6)), 1: list(range(4)), 2: list(range(4))}


@pytest.mark.parametrize(
    "rows, match",
    [
        # Frames of 2 tokens laid out as runs of 1 and 3.
        ([([T, V, T, V, V, V, T], [], [[2, 2, 4]])], "video 0's frame 0 .2 tokens. is not contiguous"),
        # An image between the video's two frames.
        (
            [([T, V, V, T, I, I, T, V, V, T], [[1, 2, 4]], [[2, 2, 4]])],
            r"video 0 has non-text tokens \(types \[1\]\) between its frames 0 and 1, at \[3, 7\)",
        ),
        # The video's second frame is in the next row.
        ([([T, V, V, T], [], [[2, 2, 4]]), ([T, V, V, T], [], [])], "video 0 belongs to row 0"),
        # A grid whose frames are not whole merged tokens: 12 patches are 3 tokens, frames of 6 patches 1 each.
        ([([T, V, V, V, T], [], [[2, 2, 3]])], r"does not split into 2 frames of whole merged tokens"),
    ],
    ids=["frame-size", "image-between-frames", "frame-outside-row", "fractional-frame"],
)
def test_videos_that_are_not_text_separated_frame_runs_are_rejected(rows, match):
    batch = _mixed_batch(rows)
    packed, metadata = pack_microbatch(batch)
    with pytest.raises(ValueError, match=match):
        token_shard(packed, metadata, 1, 0, spatial_merge_size=MERGE)


def test_an_image_split_by_text_is_still_rejected():
    rows = [([T, I, I, T, I, I, T], [[1, 4, 4]], [])]
    packed, metadata = pack_microbatch(_mixed_batch(rows))
    with pytest.raises(ValueError, match="image 0's 4 tokens are not contiguous"):
        token_shard(packed, metadata, 1, 0, spatial_merge_size=MERGE)
