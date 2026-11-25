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

from arctic_platform.common.packing import IMAGE_MEDIA
from arctic_platform.common.packing import QWEN_VL_MEDIA_SPEC
from arctic_platform.common.packing import VIDEO_MEDIA
from arctic_platform.common.packing import MediaKind
from arctic_platform.common.packing import MediaSpec
from arctic_platform.common.packing import media_spec as media_spec_module
from arctic_platform.common.packing import media_spec_for
from arctic_platform.common.packing import register_media_spec


@pytest.fixture
def registry(monkeypatch):
    """A copy of the registry, so a test's entries do not outlive it."""
    specs = dict(media_spec_module._MEDIA_SPECS)
    monkeypatch.setattr(media_spec_module, "_MEDIA_SPECS", specs)
    return specs


def test_qwen4_exp_gets_the_qwen_vl_spec():
    assert media_spec_for("qwen4_exp") is QWEN_VL_MEDIA_SPEC


@pytest.mark.parametrize("model_type", ["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"])
def test_qwen_vl_model_types_without_an_mrope_media_forward_have_no_spec(model_type):
    # Their Arctic modeling does not read [4, 1, T] positions and video (qwen3_5_moe flattens position_ids).
    assert media_spec_for(model_type) is None


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


def test_qwen_vl_placeholder_ids_leave_out_a_kind_with_no_id():
    assert QWEN_VL_MEDIA_SPEC.placeholder_ids({"image_token_id": 29}) == {"image": 29}
    assert QWEN_VL_MEDIA_SPEC.placeholder_ids({"video_token_id": 30, "text_config": {}}) == {"video": 30}
    assert QWEN_VL_MEDIA_SPEC.placeholder_ids({"vocab_size": 64}) == {}


def test_qwen_vl_placeholder_ids_are_the_shared_config_reading():
    # The inference worker reads a model type with no spec the same way.
    assert QWEN_VL_MEDIA_SPEC.placeholder_ids is media_spec_module.config_placeholder_ids


def test_qwen_vl_placeholder_ids_fall_back_to_text_config():
    config = {"image_token_id": 29, "text_config": {"image_token_id": 99, "video_token_id": 30}}
    # Top-level first: the image id is the config's own, the video id comes from text_config.
    assert QWEN_VL_MEDIA_SPEC.placeholder_ids(config) == {"image": 29, "video": 30}
    transformers = pytest.importorskip("transformers")
    nested = transformers.PretrainedConfig(text_config=transformers.PretrainedConfig(image_token_id=29))
    assert QWEN_VL_MEDIA_SPEC.placeholder_ids(nested) == {"image": 29}


def test_qwen_vl_spatial_merge_size_from_the_vision_config():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3VLConfig(vision_config={"depth": 1, "hidden_size": 16, "spatial_merge_size": 3})
    assert QWEN_VL_MEDIA_SPEC.spatial_merge_size(config) == 3
    assert QWEN_VL_MEDIA_SPEC.spatial_merge_size(config.to_dict()) == 3
    for missing in ({}, {"vision_config": {}}):
        with pytest.raises(ValueError, match="vision_config.spatial_merge_size"):
            QWEN_VL_MEDIA_SPEC.spatial_merge_size(missing)


def test_qwen_vl_leaf_names_are_the_ones_media_uses():
    assert QWEN_VL_MEDIA_SPEC.modalities == ("image", "video")
    assert QWEN_VL_MEDIA_SPEC.kind("image") is IMAGE_MEDIA
    assert QWEN_VL_MEDIA_SPEC.kind("video") is VIDEO_MEDIA
    assert QWEN_VL_MEDIA_SPEC.kind("image").keys == ("pixel_values", "image_grid_thw", "image_row_ids")
    assert QWEN_VL_MEDIA_SPEC.kind("video").keys == ("pixel_values_videos", "video_grid_thw", "video_row_ids")
    assert QWEN_VL_MEDIA_SPEC.positions == "mrope"
    with pytest.raises(KeyError, match="audio"):
        QWEN_VL_MEDIA_SPEC.kind("audio")


def test_register_a_second_spec(registry):
    image = MediaKind("image", "images", "image_sizes", "image_rows", 1)
    spec = MediaSpec(
        kinds=(image,),
        placeholder_ids=lambda config: {"image": int(config["image_token_index"])},
        spatial_merge_size=lambda config: int(config["merge"]),
        positions="1d",
    )
    register_media_spec("example_vlm", spec)
    assert media_spec_for("example_vlm") is spec
    assert media_spec_for("qwen4_exp") is QWEN_VL_MEDIA_SPEC
    assert spec.modalities == ("image",)
    assert spec.placeholder_ids({"image_token_index": 7}) == {"image": 7}
    assert spec.spatial_merge_size({"merge": 2}) == 2


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
    ids, merge = QWEN_VL_MEDIA_SPEC.placeholder_ids, QWEN_VL_MEDIA_SPEC.spatial_merge_size
    with pytest.raises(ValueError, match="positions"):
        MediaSpec((IMAGE_MEDIA,), ids, merge, "rope3d")
    with pytest.raises(ValueError, match="distinct"):
        MediaSpec((IMAGE_MEDIA, IMAGE_MEDIA), ids, merge, "mrope")
