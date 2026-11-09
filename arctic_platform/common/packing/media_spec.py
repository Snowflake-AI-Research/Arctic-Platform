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

"""How a model lays out its media, looked up by ``model_type``.

The rest of this package is written for the Qwen-VL layout: placeholder token ids read from the config's
``image_token_id`` / ``video_token_id``, batch leaves named as in ``media.py``, ``t * h * w / merge**2`` tokens
per item, and ``[4, 1, T]`` M-RoPE positions. A ``MediaSpec`` states each of those for one model family, so a
caller asks ``media_spec_for(model_type)`` instead of assuming them, and a family with another layout
(other placeholder rules, other leaf names, a token count on another grid, 1-D positions, a marker token
before each item that belongs to it) registers its own entry with ``register_media_spec``.

The Qwen-VL entry reproduces the constants the existing code uses; nothing here changes that code yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch

from .media import IMAGE_MEDIA
from .media import VIDEO_MEDIA
from .media import MediaKind

POSITION_LAYOUTS = ("mrope", "1d")


@dataclass(frozen=True)
class MediaSpec:
    """One model family's media layout.

    * ``kinds`` -- the media kinds the model takes, each with its batch leaf names (values, grid, row ids) and
      its ``mm_token_type_ids`` value; ``modalities`` are their names, in order.
    * ``placeholder_ids(config)`` -- ``{modality: token id}`` of the placeholder token each item's tokens use,
      read from the model's config (a ``PretrainedConfig`` or its ``config.json`` dict).
    * ``tokens_per_item(grid_row, spatial_merge_size)`` -- how many placeholder tokens one item of grid
      ``grid_row`` expands to in its row.
    * ``positions`` -- ``"mrope"`` for ``[4, 1, T]`` multimodal positions (``mrope_position_ids``), ``"1d"`` for
      plain text positions.
    * ``item_prefix_tokens`` -- how many text tokens just before an item's first placeholder token belong to the
      item (a begin marker, say), so a window that cuts between them and the item would split the item.
    """

    kinds: Tuple[MediaKind, ...]
    placeholder_ids: Callable[[Any], Dict[str, int]]
    tokens_per_item: Callable[[Sequence[int], int], int]
    positions: str
    item_prefix_tokens: int = 0

    def __post_init__(self) -> None:
        names = [kind.name for kind in self.kinds]
        if not names or len(set(names)) != len(names):
            raise ValueError(f"a media spec needs one or more distinct media kinds, got {names}")
        if self.positions not in POSITION_LAYOUTS:
            raise ValueError(f"positions must be one of {POSITION_LAYOUTS}, got {self.positions!r}")
        if self.item_prefix_tokens < 0:
            raise ValueError(f"item_prefix_tokens must be >= 0, got {self.item_prefix_tokens}")

    @property
    def modalities(self) -> Tuple[str, ...]:
        return tuple(kind.name for kind in self.kinds)

    def kind(self, modality: str) -> MediaKind:
        """The ``MediaKind`` (leaf names, token type) of ``modality``."""
        for kind in self.kinds:
            if kind.name == modality:
                return kind
        raise KeyError(f"no {modality!r} media in this spec; it has {list(self.modalities)}")


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def _qwen_vl_placeholder_ids(config: Any) -> Dict[str, int]:
    ids = {}
    for modality, key in (("image", "image_token_id"), ("video", "video_token_id")):
        value = _config_value(config, key)
        if value is None:
            raise ValueError(f"a Qwen-VL config needs {key} to place {modality} tokens")
        ids[modality] = int(value)
    return ids


def _qwen_vl_tokens_per_item(grid_row: Sequence[int], spatial_merge_size: int) -> int:
    # A video's frames are h * w / merge**2 tokens each; t of them make the same total as an image's grid.
    if isinstance(grid_row, torch.Tensor):
        grid_row = grid_row.tolist()
    grid_t, grid_h, grid_w = (int(value) for value in grid_row)
    return grid_t * grid_h * grid_w // (spatial_merge_size**2)


QWEN_VL_MEDIA_SPEC = MediaSpec(
    kinds=(IMAGE_MEDIA, VIDEO_MEDIA),
    placeholder_ids=_qwen_vl_placeholder_ids,
    tokens_per_item=_qwen_vl_tokens_per_item,
    positions="mrope",
    item_prefix_tokens=0,
)

# Qwen vision-language families whose media follows media.py's layout, Qwen3-VL video frames included.
QWEN_VL_MODEL_TYPES = ("qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe", "qwen4_exp")

_MEDIA_SPECS: Dict[str, MediaSpec] = {}


def register_media_spec(model_type: str, spec: MediaSpec) -> None:
    """Make ``spec`` the media layout of ``model_type``; a model type already registered is refused."""
    if not model_type:
        raise ValueError("register_media_spec needs a non-empty model_type")
    if not isinstance(spec, MediaSpec):
        raise TypeError(f"register_media_spec needs a MediaSpec, got {type(spec).__name__}")
    if model_type in _MEDIA_SPECS:
        raise ValueError(f"model_type {model_type!r} already has a media spec")
    _MEDIA_SPECS[model_type] = spec


def media_spec_for(model_type: str) -> Optional[MediaSpec]:
    """The media layout registered for ``model_type``, or None for a model with no registered media."""
    return _MEDIA_SPECS.get(model_type)


for _model_type in QWEN_VL_MODEL_TYPES:
    register_media_spec(_model_type, QWEN_VL_MEDIA_SPEC)
del _model_type
