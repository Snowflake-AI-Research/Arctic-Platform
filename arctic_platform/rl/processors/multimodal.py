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

"""Per-sequence image inputs for the training pipeline.

A batch carries images as ``batch["multi_modal_inputs"]``: one entry per sequence, either ``None`` for a text-only
sequence or a dict with that sequence's ``pixel_values`` (``[patches, channels * temporal_patch * patch**2]``) and
``image_grid_thw`` (``[images, 3]``). The microbatch splitter keeps each entry with its sequence, and the forward
concatenates the entries of the sequences it runs, in row order, into the ``pixel_values`` and ``image_grid_thw``
the model takes. Batch-level ``pixel_values`` have no sequence dimension, so a split cannot follow them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

MULTI_MODAL_INPUTS_KEY = "multi_modal_inputs"
IMAGE_INPUT_KEYS = ("pixel_values", "image_grid_thw")


def collate_multi_modal_inputs(items: Sequence[dict[str, Any] | None] | None) -> dict[str, torch.Tensor]:
    """Concatenate per-sequence image inputs in sequence order. Returns ``{}`` when no sequence has an image."""
    if items is None:
        return {}
    pixel_values: list[torch.Tensor] = []
    grids: list[torch.Tensor] = []
    for index, item in enumerate(items):
        if not item:
            continue
        unknown = sorted(set(item) - set(IMAGE_INPUT_KEYS))
        if unknown:
            raise ValueError(f"{MULTI_MODAL_INPUTS_KEY}[{index}] has unsupported keys {unknown}")
        missing = [key for key in IMAGE_INPUT_KEYS if key not in item]
        if missing:
            raise ValueError(f"{MULTI_MODAL_INPUTS_KEY}[{index}] is missing {missing}")
        grid = torch.as_tensor(item["image_grid_thw"]).reshape(-1, 3)
        values = torch.as_tensor(item["pixel_values"])
        patches = int(grid.prod(dim=-1).sum())
        if values.ndim != 2 or values.shape[0] != patches:
            raise ValueError(
                f"{MULTI_MODAL_INPUTS_KEY}[{index}] pixel_values {tuple(values.shape)} do not hold the {patches} "
                f"patches of image_grid_thw {grid.tolist()}"
            )
        pixel_values.append(values)
        grids.append(grid)
    if not grids:
        return {}
    return {"pixel_values": torch.cat(pixel_values), "image_grid_thw": torch.cat(grids)}
