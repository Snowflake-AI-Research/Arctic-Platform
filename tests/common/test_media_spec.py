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

"""The per-model media spec: the Qwen-VL entry reproduces the packing constants, and the registry takes new entries."""

import dataclasses

import pytest
import torch

from arctic_platform.common.packing import IMAGE_MEDIA
from arctic_platform.common.packing import QWEN_VL_MEDIA_SPEC
from arctic_platform.common.packing import VIDEO_MEDIA
from arctic_platform.common.packing import MediaKind
from arctic_platform.common.packing import MediaSpec
from arctic_platform.common.packing import media_spec as media_spec_module
from arctic_platform.common.packing import media_spec_for
from arctic_platform.common.packing import media_token_spans
from arctic_platform.common.packing import register_media_spec

MERGE = 2


@pytest.fixture
def registry(monkeypatch):
    """A copy of the registry, so a test's entries do not outlive it."""
    specs = dict(media_spec_module._MEDIA_SPECS)
    monkeypatch.setattr(media_spec_module, "_MEDIA_SPECS", specs)
    return specs


@pytest.mark.parametrize("model_type", ["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe", "qwen4_exp"])
def test_qwen_vl_model_types_get_the_qwen_vl_spec(model_type):
    assert media_spec_for(model_type) is QWEN_VL_MEDIA_SPEC


def test_unknown_and_text_only_model_types_have_no_spec():
    assert media_spec_for("not_a_model") is None
    assert media_spec_for("llama") is None
    assert media_spec_for("") is None


def test_qwen_vl_placeholder_ids_from_a_real_config():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3VLConfig(
        text_config={"vocab_size": 64, "hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2},
        vision_config={"depth": 1, "hidden_size": 16, "out_hidden_size": 16, "num_heads": 2},
        image_token_id=29,
        video_token_id=30,
        vision_start_token_id=27,
        vision_end_token_id=28,
    )
    assert QWEN_VL_MEDIA_SPEC.placeholder_ids(config) == {"image": 29, "video": 30}
    # The same config as config.json gives the same ids.
    assert QWEN_VL_MEDIA_SPEC.placeholder_ids(config.to_dict()) == {"image": 29, "video": 30}


@pytest.mark.parametrize("model_type", media_spec_module.QWEN_VL_MODEL_TYPES)
def test_qwen_vl_placeholder_ids_from_each_family_default_config(model_type):
    transformers = pytest.importorskip("transformers")
    try:
        config = transformers.AutoConfig.for_model(model_type)
    except ValueError:
        pytest.skip(f"this transformers has no {model_type} config")
    ids = media_spec_for(model_type).placeholder_ids(config)
    assert ids == {"image": config.image_token_id, "video": config.video_token_id}


def test_qwen_vl_placeholder_ids_need_both_ids():
    with pytest.raises(ValueError, match="video_token_id"):
        QWEN_VL_MEDIA_SPEC.placeholder_ids({"image_token_id": 29})


def test_qwen_vl_leaf_names_are_the_ones_media_uses():
    assert QWEN_VL_MEDIA_SPEC.modalities == ("image", "video")
    assert QWEN_VL_MEDIA_SPEC.kind("image") is IMAGE_MEDIA
    assert QWEN_VL_MEDIA_SPEC.kind("video") is VIDEO_MEDIA
    assert QWEN_VL_MEDIA_SPEC.kind("image").keys == ("pixel_values", "image_grid_thw", "image_row_ids")
    assert QWEN_VL_MEDIA_SPEC.kind("video").keys == ("pixel_values_videos", "video_grid_thw", "video_row_ids")
    assert QWEN_VL_MEDIA_SPEC.positions == "mrope"
    assert QWEN_VL_MEDIA_SPEC.item_prefix_tokens == 0
    with pytest.raises(KeyError, match="audio"):
        QWEN_VL_MEDIA_SPEC.kind("audio")


def _kind_tokens_in_spans(media, kind, token_types, cu_seqlens):
    spans = media_token_spans(media, kind, token_types, cu_seqlens, MERGE)
    flat = token_types.reshape(-1)
    return [int((flat[start:end] == kind.token_type).sum()) for start, end in spans.tolist()]


def test_qwen_vl_tokens_per_item_matches_media_token_spans():
    # Row 0: text, image (1x4x4 -> 4 tokens), text, image (1x2x4 -> 2 tokens), text.
    # Row 1: text, a 3-frame video (3x2x4 -> 2 tokens a frame) with timestamp text between frames, text.
    row0 = [0, 1, 1, 1, 1, 0, 1, 1, 0]
    row1 = [0, 2, 2, 0, 0, 2, 2, 0, 2, 2, 0]
    token_types = torch.tensor([row0 + row1])
    cu_seqlens = torch.tensor([0, len(row0), len(row0) + len(row1)])
    image_grid = torch.tensor([[1, 4, 4], [1, 2, 4]])
    video_grid = torch.tensor([[3, 2, 4]])
    media = {
        "image_grid_thw": image_grid,
        "image_row_ids": torch.tensor([0, 0]),
        "video_grid_thw": video_grid,
        "video_row_ids": torch.tensor([1]),
    }
    for kind, grid in ((IMAGE_MEDIA, image_grid), (VIDEO_MEDIA, video_grid)):
        expected = _kind_tokens_in_spans(media, kind, token_types, cu_seqlens)
        assert [QWEN_VL_MEDIA_SPEC.tokens_per_item(row, MERGE) for row in grid] == expected
        assert [QWEN_VL_MEDIA_SPEC.tokens_per_item(row.tolist(), MERGE) for row in grid] == expected
    assert QWEN_VL_MEDIA_SPEC.tokens_per_item([3, 2, 4], MERGE) == 6


def test_register_a_second_spec(registry):
    image = MediaKind("image", "images", "image_sizes", "image_rows", "image_ranges", 1)
    spec = MediaSpec(
        kinds=(image,),
        placeholder_ids=lambda config: {"image": int(config["image_token_index"])},
        tokens_per_item=lambda grid_row, merge: int(grid_row[1]) * int(grid_row[2]) // merge**2 + 1,
        positions="1d",
        item_prefix_tokens=1,
    )
    register_media_spec("example_vlm", spec)
    assert media_spec_for("example_vlm") is spec
    assert media_spec_for("qwen4_exp") is QWEN_VL_MEDIA_SPEC
    assert spec.modalities == ("image",)
    assert spec.placeholder_ids({"image_token_index": 7}) == {"image": 7}
    assert spec.tokens_per_item([1, 4, 4], MERGE) == 5


def test_a_duplicate_model_type_is_refused(registry):
    with pytest.raises(ValueError, match="already has a media spec"):
        register_media_spec("qwen4_exp", QWEN_VL_MEDIA_SPEC)
    assert media_spec_for("qwen4_exp") is QWEN_VL_MEDIA_SPEC


def test_registering_needs_a_model_type_and_a_spec(registry):
    with pytest.raises(ValueError, match="non-empty model_type"):
        register_media_spec("", QWEN_VL_MEDIA_SPEC)
    with pytest.raises(TypeError, match="MediaSpec"):
        register_media_spec("example_vlm", {"positions": "1d"})
    assert media_spec_for("example_vlm") is None


def test_a_spec_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        QWEN_VL_MEDIA_SPEC.positions = "1d"


def test_a_malformed_spec_is_refused():
    with pytest.raises(ValueError, match="positions"):
        MediaSpec((IMAGE_MEDIA,), QWEN_VL_MEDIA_SPEC.placeholder_ids, QWEN_VL_MEDIA_SPEC.tokens_per_item, "rope3d")
    with pytest.raises(ValueError, match="distinct"):
        MediaSpec((IMAGE_MEDIA, IMAGE_MEDIA), QWEN_VL_MEDIA_SPEC.placeholder_ids, len, "mrope")
    with pytest.raises(ValueError, match="item_prefix_tokens"):
        MediaSpec((IMAGE_MEDIA,), QWEN_VL_MEDIA_SPEC.placeholder_ids, len, "1d", item_prefix_tokens=-1)
