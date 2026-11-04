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

"""Set a constant hybrid KV cache group size.

A hybrid model's layers share one pool of equal-sized blocks. vLLM splits
each attention type into groups with the same layer count (the group size)
and sizes every block for that many layers. A group with fewer layers still
takes the whole block, so the unused layer slots are padding.

Set the size through vLLM's ``additional_config``:

    vllm serve ... --additional-config '{"arctic_kv_group_size": 8}'
    AsyncEngineArgs(..., additional_config={"arctic_kv_group_size": 8})

For the Arctic inference server, put ``additional_config`` in
``ModelConfig.extra_engine_kwargs``.

"""

from __future__ import annotations

from collections import defaultdict
from contextvars import ContextVar
from typing import Any, Callable

# Key in ``VllmConfig.additional_config``. That dict is shared with vLLM and
# every plugin, so the key carries the owner's prefix.
KV_GROUP_SIZE_KEY = "arctic_kv_group_size"

# get_kv_cache_groups knows the VllmConfig; the uniform-page-size helper it
# calls does not. The wrapper publishes the config for the duration of the call.
_VLLM_CONFIG: ContextVar[Any] = ContextVar("arctic_kv_group_size_vllm_config", default=None)


def configured_group_size(vllm_config: Any) -> int | None:
    """The forced group size in ``additional_config``, or ``None`` when unset."""
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        return None
    value = additional_config.get(KV_GROUP_SIZE_KEY)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"additional_config[{KV_GROUP_SIZE_KEY!r}] must be a positive integer, got {value!r}")
    return value


def layer_buckets(kv_cache_spec: dict[str, Any]) -> list[list[str]]:
    """Layers whose specs merge share a bucket, as in vLLM 0.31."""
    same_type_layers: dict[Any, list[str]] = defaultdict(list)
    for layer_name, layer_spec in kv_cache_spec.items():
        same_type_layers[layer_spec].append(layer_name)

    buckets: list[list[str]] = []
    spec_buckets: list[list[Any]] = []
    for layer_spec, layer_names in same_type_layers.items():
        for names, specs in zip(buckets, spec_buckets):
            try:
                type(specs[0]).merge([*specs, layer_spec])
            except (AssertionError, ValueError):
                continue
            names.extend(layer_names)
            specs.append(layer_spec)
            break
        else:
            buckets.append(list(layer_names))
            spec_buckets.append([layer_spec])
    return buckets


def split_buckets(buckets: list[list[str]], group_size: int) -> list[list[str]]:
    """Split each bucket into ceil(n / group_size) groups, round-robin as vLLM
    does so pipeline stages stay balanced."""
    groups = []
    for layers in buckets:
        num_groups = -(-len(layers) // group_size) # ceiling devision
        for i in range(num_groups):
            groups.append(layers[i::num_groups])
    return groups


def uniform_page_size_groups(
    vllm_config: Any,
    kv_cache_spec: dict[str, Any],
    original: Callable[[dict[str, Any]], list[Any]],
) -> list[Any]:
    """Replacement body for vLLM's ``_get_kv_cache_groups_uniform_page_size``."""
    group_size = configured_group_size(vllm_config)
    if group_size is None:
        return original(kv_cache_spec)

    from vllm.v1.core import kv_cache_utils

    grouped = split_buckets(layer_buckets(kv_cache_spec), group_size)
    kv_cache_utils.logger.info(
        "additional_config[%r]=%d: %d hybrid KV cache groups of up to %d layers",
        KV_GROUP_SIZE_KEY, group_size, len(grouped), max(map(len, grouped)))
    return kv_cache_utils.create_kv_cache_group_specs(kv_cache_spec, grouped)


def ensure_kv_group_size_patch() -> None:
    """Route vLLM's hybrid KV grouping through ``uniform_page_size_groups``."""
    from arctic_platform.inference.utils import require_supported_vllm_version

    require_supported_vllm_version("hybrid KV cache group size")

    from vllm.v1.core import kv_cache_utils

    if getattr(kv_cache_utils.get_kv_cache_groups, "_arctic_kv_group_size", False):
        return

    original_get_kv_cache_groups = kv_cache_utils.get_kv_cache_groups
    original_uniform = kv_cache_utils._get_kv_cache_groups_uniform_page_size

    def get_kv_cache_groups(vllm_config, kv_cache_spec):
        token = _VLLM_CONFIG.set(vllm_config)
        try:
            return original_get_kv_cache_groups(vllm_config, kv_cache_spec)
        finally:
            _VLLM_CONFIG.reset(token)

    def _get_kv_cache_groups_uniform_page_size(kv_cache_spec):
        return uniform_page_size_groups(_VLLM_CONFIG.get(), kv_cache_spec, original_uniform)

    get_kv_cache_groups._arctic_kv_group_size = True
    get_kv_cache_groups._orig = original_get_kv_cache_groups
    _get_kv_cache_groups_uniform_page_size._orig = original_uniform
    kv_cache_utils.get_kv_cache_groups = get_kv_cache_groups
    kv_cache_utils._get_kv_cache_groups_uniform_page_size = _get_kv_cache_groups_uniform_page_size
