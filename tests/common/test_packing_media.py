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
from arctic_platform.common.packing import window_media
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
    # C at 14..15. Two windows of eight tokens: A sits in window 0, B and C in window 1.
    packed, _, windows = _windows(_batch(), shard_count=2)

    assert [_owners(w["pixel_values"]) for w in windows] == [[0], [1, 2]]
    assert [w["image_row_ids"].tolist() for w in windows] == [[0], [2, 2]]
    assert [w["image_grid_thw"].tolist() for w in windows] == [[[1, 4, 4]], [[1, 2, 4], [1, 2, 4]]]
    # A window carries whole images: every one of A's 16 patches, in order, and only the leaves the call has.
    torch.testing.assert_close(windows[0]["pixel_values"][:, 1], torch.arange(16, dtype=torch.float))
    assert all(w.keys() == packed.keys() for w in windows)
    # The patches are the batch's own, untouched: A's in window 0, B's then C's in window 1.
    torch.testing.assert_close(torch.cat([w["pixel_values"] for w in windows]), packed["pixel_values"])


def test_each_image_lands_in_exactly_one_window():
    _, _, windows = _windows(_batch(), shard_count=2)

    owners = [owner for window in windows for owner in _owners(window["pixel_values"])]
    assert owners == [0, 1, 2]
    # The images a window gets hold exactly its image-token slots.
    for window in windows:
        tokens = int((window["image_grid_thw"].prod(dim=1) // MERGE**2).sum())
        assert tokens == int((window["mm_token_type_ids"] == 1).sum())


@pytest.mark.parametrize("shard_index", range(4))
def test_an_image_straddling_a_window_boundary_is_refused_on_every_window(shard_index):
    # Four windows of four tokens: A at [1, 5) straddles the boundary at 4. Every window refuses the call, the
    # ones A never touches too, so no rank goes on alone.
    packed, metadata = pad_packed_microbatch(*pack_microbatch(_batch()), 4)
    with pytest.raises(ValueError, match=r"image 0's tokens \[1, 5\) straddle the window boundary at token 4"):
        token_shard(packed, metadata, 4, shard_index, spatial_merge_size=MERGE)


def test_window_media_alone_checks_its_own_boundaries_unless_given_the_calls():
    packed, metadata = pack_microbatch(_batch())
    media = {key: packed[key] for key in ("pixel_values", "image_grid_thw", "image_row_ids")}
    types = packed["mm_token_type_ids"]

    def window(span, **kwargs):
        return window_media(media, types, metadata.cu_seqlens, span, spatial_merge_size=MERGE, **kwargs)

    # A at [1, 5) crosses this window's own start.
    with pytest.raises(ValueError, match=r"image 0's tokens \[1, 5\) straddle the window boundary at token 4"):
        window((4, 8))
    # [8, 16) does not touch A, and its own boundaries cut no image: B and C, whole.
    assert window((8, 16))["image_grid_thw"].tolist() == [[1, 2, 4], [1, 2, 4]]
    # Given the call's boundaries, it refuses the straddle elsewhere in the call too.
    with pytest.raises(ValueError, match="at token 4"):
        window((8, 16), boundaries=[0, 4, 8, 12, 16])


def test_window_without_media_keeps_empty_media_leaves():
    # Three windows of four tokens: the first holds only row 0, which is text; the image fills the second exactly.
    rows = [([0, 0, 0, 0], []), ([1, 1, 1, 1, 0, 0, 0, 0], [[1, 4, 4]])]
    _, _, windows = _windows(_batch(rows=rows), shard_count=3)

    empty = windows[0]
    assert empty["pixel_values"].shape == (0, 4)
    assert empty["image_grid_thw"].shape == (0, 3)
    assert empty["image_row_ids"].shape == (0,)
    assert [w["image_grid_thw"].shape[0] for w in windows] == [0, 1, 0]


def test_multimodal_positions_are_cut_on_the_token_axis():
    packed, metadata, windows = _windows(_batch(), shard_count=2)

    assert all(w["position_ids"].shape == (4, 1, 8) for w in windows)
    gathered = torch.cat([w["position_ids"] for w in windows], dim=2)
    torch.testing.assert_close(gathered, packed["position_ids"])
    assert cu_seqlens_from_position_ids(gathered).tolist() == metadata.cu_seqlens.tolist()


def test_patch_dim_equal_to_the_padded_width_is_not_sliced():
    _, _, windows = _windows(_batch(patch_dim=16), shard_count=2)

    assert [w["pixel_values"].shape for w in windows] == [(16, 16), (16, 16)]


def test_video_frames_follow_the_same_rules():
    # A [2, 4, 4] video is 8 merged tokens at [1, 9); with the trailing text, one window holds it whole.
    rows = [([0, *[1] * 8, 0, 0, 0], [[2, 4, 4]])]
    packed, _, windows = _windows(_batch(rows=rows, kind="video"), shard_count=1)

    assert windows[0]["video_grid_thw"].tolist() == [[2, 4, 4]]
    assert windows[0]["pixel_values_videos"].shape == (32, 4)
    assert "video_window_token_ranges" not in windows[0]
    assert packed["position_ids"][1, 0].tolist() == [0, 1, 1, 1, 1, 3, 3, 3, 3, 5, 6, 7]


def test_selected_rows_pack_and_window_like_a_batch_built_in_that_order():
    batch = _batch()
    _, _, from_selection = _windows(select_rows(batch, torch.tensor([2, 0]), n_rows=3), shard_count=2, multiple=2)
    _, _, built = _windows(_batch(rows=[ROWS[2], ROWS[0]]), shard_count=2, multiple=2)

    for got, want in zip(from_selection, built):
        assert got["image_row_ids"].tolist() == want["image_row_ids"].tolist()
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


def _video_owners(window):
    return _owners(window["pixel_values_videos"])


def test_each_window_gets_the_whole_videos_and_images_inside_it():
    # Row 0 [0, 12): the 3-frame video at [1, 11). Row 1 [12, 24): an image at [13, 15), a video at [16, 21).
    rows = [VIDEO_ROWS[0], ([T, I, I, T, V, V, T, V, V, T, T, T], [[1, 2, 4]], [[2, 2, 4]])]
    packed, _, windows = _windows(_mixed_batch(rows), shard_count=2)

    assert [_video_owners(w) for w in windows] == [[0], [1]]
    assert [_owners(w["pixel_values"]) for w in windows] == [[], [0]]
    assert [w["video_row_ids"].tolist() for w in windows] == [[0], [1]]
    # Each window carries a whole video: all three frames' patches.
    assert [w["pixel_values_videos"].shape for w in windows] == [(24, 4), (16, 4)]
    assert all(w.keys() == packed.keys() for w in windows)


@pytest.mark.parametrize(
    "rows, shard_count, match",
    [
        # Three windows of ten (28 tokens padded to 30): the first video, [1, 11), crosses 10.
        (VIDEO_ROWS, 3, r"video 0's tokens \[1, 11\) straddle the window boundary at token 10"),
        # The boundary falls on the text between the video's two frames.
        ([([T, V, V, T, T, T, T, V, V, T], [], [[2, 2, 4]])], 2, r"video 0's tokens \[1, 9\) straddle .* token 5"),
        # In the second packed row: the video at [17, 21) crosses 18.
        (
            [([T] * 12, [], []), ([T, T, T, T, T, V, V, V, V, T, T, T], [], [[2, 2, 4]])],
            4,
            r"video 0's tokens \[17, 21\) straddle the window boundary at token 18",
        ),
        # Two windows of 14: the image at [13, 15) in the second row crosses 14.
        (VIDEO_ROWS, 2, r"image 0's tokens \[13, 15\) straddle the window boundary at token 14"),
    ],
    ids=["video-first-row", "boundary-between-frames", "video-second-row", "image-second-row"],
)
def test_media_straddling_a_window_boundary_in_packed_rows_is_refused(rows, shard_count, match):
    packed, metadata = pad_packed_microbatch(*pack_microbatch(_mixed_batch(rows)), shard_count)
    for shard_index in range(shard_count):
        with pytest.raises(ValueError, match=match):
            token_shard(packed, metadata, shard_count, shard_index, spatial_merge_size=MERGE)


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
