# Copyright 2026 Snowflake Inc.
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

"""Allocate the KV cache of a pipeline stage that holds no layer of some group.

vLLM 0.33 groups the KV cache layers of the whole model, then projects every
group onto each pipeline stage (``_project_kv_cache_groups_to_worker``). A
stage with none of a group's layers keeps the group, with no layer names, and
with the group's whole-model ``UniformTypeKVCacheSpecs``, so that every stage
has the same group count, block sizes and block table layout. But
``get_kv_cache_config_from_groups`` builds a uniform-type group's tensors from
that spec's layers, not from the group's layer names. The stage then receives
tensors for another stage's layers, and ``allocate_kv_cache`` fails with
``StopIteration`` looking for their group. DeepSeek-V4.1 at PP2 hits this on
stage 2, which has none of the ratio-2 compressed layers.

The fix drops, after placement, the tensors whose layers the stage does not
hold. They come only from such empty groups, whose share of the block size is
already zero (it is summed over the group's layer names), so nothing else
moves. PP1 keeps vLLM's placement as is.

Remove once the pinned vLLM builds uniform-type tensors from the layer names.
"""

from __future__ import annotations

import dataclasses
from typing import Any

_PATCH_ATTR = "_arctic_pp_kv_cache_groups"


def drop_unheld_kv_cache_tensors(kv_cache_config: Any) -> Any:
    """``kv_cache_config`` without the tensors of layers none of its groups hold."""
    held = {name for group in kv_cache_config.kv_cache_groups for name in group.layer_names}
    kept = []
    for tensor in kv_cache_config.kv_cache_tensors:
        layers = set(tensor.layers)
        if layers <= held:
            kept.append(tensor)
        elif layers & held:
            raise ValueError(
                "a KV cache tensor mixes layers this pipeline stage holds and does not "
                f"hold: {sorted(layers - held)} of {tensor.layers}"
            )
    if len(kept) == len(kv_cache_config.kv_cache_tensors):
        return kv_cache_config
    return dataclasses.replace(kv_cache_config, kv_cache_tensors=kept)


def patch_get_kv_cache_config_from_groups(planning: Any) -> None:
    original = planning.get_kv_cache_config_from_groups
    if getattr(original, _PATCH_ATTR, False):
        return

    def get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory):
        kv_cache_config = original(vllm_config, kv_cache_groups, available_memory)
        if vllm_config.parallel_config.pipeline_parallel_size <= 1:
            return kv_cache_config
        return drop_unheld_kv_cache_tensors(kv_cache_config)

    setattr(get_kv_cache_config_from_groups, _PATCH_ATTR, True)
    get_kv_cache_config_from_groups._orig = original
    planning.get_kv_cache_config_from_groups = get_kv_cache_config_from_groups


def ensure_pp_kv_cache_groups_fix() -> None:
    """Patch vLLM's KV cache placement for pipeline stages missing a group's layers."""
    from arctic_platform.inference.utils import require_supported_vllm_version

    require_supported_vllm_version("KV cache groups with pipeline parallelism")

    from vllm.v1.core import kv_cache_planning

    patch_get_kv_cache_config_from_groups(kv_cache_planning)
