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

"""Packed M-RoPE positions match Hugging Face's ``get_rope_index`` for Qwen3.5 and qwen4_exp.

``get_rope_index`` reads only ``config.vision_config.spatial_merge_size`` and ``get_vision_position_ids``, so it
runs on a tiny config with no weights. Each packed row is compared with Hugging Face's positions for the same row
alone: channels 1-3 of ``mrope_position_ids`` are its three channels.

Inputs follow the layout the Qwen processors emit: every image, and every video frame, is a run of media tokens
with text (at least ``<vision_start>`` / ``<vision_end>``, and a timestamp between frames) on both sides. Hugging
Face takes one grid per media run, so two media items with no text between them are outside what it accepts;
``mrope_position_ids`` places them one after the other, which ``test_packing_mrope.py`` covers.
"""

import functools
import importlib
import random
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.common.packing import mrope_position_ids

MODELS = [
    pytest.param("qwen3_5", "Qwen3_5Model", "Qwen3_5Config", id="qwen3_5"),
    pytest.param("qwen4_exp", "Qwen4ExpModel", "Qwen4ExpConfig", id="qwen4_exp"),
]


def _hf_rope_index(module: str, model_class: str, config_class: str, spatial_merge_size: int):
    """The model class's ``get_rope_index``, bound to a stand-in holding only a tiny config."""
    modeling = pytest.importorskip(f"transformers.models.{module}.modeling_{module}")
    configuration = importlib.import_module(f"transformers.models.{module}.configuration_{module}")
    config = getattr(configuration, config_class)(vision_config={"spatial_merge_size": spatial_merge_size})
    model = getattr(modeling, model_class)
    stand_in = SimpleNamespace(config=config)
    stand_in.get_vision_position_ids = functools.partial(model.get_vision_position_ids, stand_in)
    return functools.partial(model.get_rope_index, stand_in)


def _hf_row_positions(rope_index, row):
    """Hugging Face's ``[3, len]`` positions for one row alone."""
    types = torch.tensor([row["types"]], dtype=torch.int32)
    position_ids, _ = rope_index(
        input_ids=torch.zeros_like(types, dtype=torch.long),
        mm_token_type_ids=types,
        image_grid_thw=torch.tensor(row["images"]) if row["images"] else None,
        video_grid_thw=torch.tensor(row["videos"]) if row["videos"] else None,
    )
    return position_ids[:, 0, :]


def _media_tokens(grid, spatial_merge_size):
    t, h, w = grid
    return t * (h // spatial_merge_size) * (w // spatial_merge_size)


def _row(pieces, spatial_merge_size):
    """A row from ``("text", n)``, ``("image", [1, h, w])`` and ``("video", [t, h, w])`` pieces.

    A video's frames are separated by one text token (its timestamp), as the processor lays them out.
    """
    types, images, videos = [], [], []
    for kind, value in pieces:
        if kind == "text":
            types += [0] * value
        elif kind == "image":
            images.append(value)
            types += [1] * _media_tokens(value, spatial_merge_size)
        else:
            videos.append(value)
            t, h, w = value
            for frame in range(t):
                if frame:
                    types.append(0)
                types += [2] * _media_tokens([1, h, w], spatial_merge_size)
    return {"types": types, "images": images, "videos": videos}


def _assert_packed_matches_hf(rope_index, rows, spatial_merge_size, padding=0):
    types = [token for row in rows for token in row["types"]] + [0] * padding
    images = [grid for row in rows for grid in row["images"]]
    videos = [grid for row in rows for grid in row["videos"]]
    lengths = [len(row["types"]) for row in rows]
    cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)

    packed = mrope_position_ids(
        torch.tensor([types]),
        cu_seqlens,
        spatial_merge_size=spatial_merge_size,
        image_grid_thw=torch.tensor(images) if images else None,
        video_grid_thw=torch.tensor(videos) if videos else None,
    )

    for row, start, end in zip(rows, cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        expected = _hf_row_positions(rope_index, row)
        torch.testing.assert_close(packed[1:, 0, start:end], expected.to(packed.dtype), rtol=0, atol=0)
        torch.testing.assert_close(packed[0, 0, start:end], torch.arange(end - start), rtol=0, atol=0)


@pytest.mark.parametrize("module, model_class, config_class", MODELS)
def test_text_image_and_video_rows_match_hf(module, model_class, config_class):
    merge = 2
    rope_index = _hf_rope_index(module, model_class, config_class, merge)
    rows = [
        _row([("text", 3), ("image", [1, 4, 4]), ("text", 2)], merge),
        _row([("text", 1), ("image", [1, 4, 6]), ("text", 4), ("image", [1, 2, 8]), ("text", 1)], merge),
        _row([("text", 2), ("video", [3, 4, 6]), ("text", 3)], merge),
        _row([("text", 5)], merge),
        _row([("text", 1), ("image", [1, 6, 2]), ("text", 2), ("video", [2, 2, 2]), ("text", 1)], merge),
    ]
    _assert_packed_matches_hf(rope_index, rows, merge, padding=3)


@pytest.mark.parametrize("module, model_class, config_class", MODELS)
@pytest.mark.parametrize("merge", [1, 2])
def test_random_packed_rows_match_hf(module, model_class, config_class, merge):
    rope_index = _hf_rope_index(module, model_class, config_class, merge)
    rng = random.Random(f"{module}-{merge}")
    for _ in range(25):
        rows = []
        for _ in range(rng.randint(1, 4)):
            pieces = [("text", rng.randint(1, 4))]
            for _ in range(rng.randint(0, 3)):
                h, w = (merge * rng.randint(1, 4) for _ in range(2))
                if rng.random() < 0.5:
                    pieces.append(("image", [1, h, w]))
                else:
                    pieces.append(("video", [rng.randint(1, 3), h, w]))
                pieces.append(("text", rng.randint(1, 3)))
            rows.append(_row(pieces, merge))
        _assert_packed_matches_hf(rope_index, rows, merge, padding=rng.randint(0, 2))
