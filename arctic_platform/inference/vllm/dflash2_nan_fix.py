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

"""Compatibility fixes for DFlash2 with GDN-backed hybrid models.

Remove these patches after both upstream fixes are included in the pinned
vLLM release:

* https://github.com/vllm-project/vllm/pull/51508
* https://github.com/vllm-project/vllm/pull/56524
"""

from itertools import product as iprod
from typing import Any, Iterable

import torch
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.single_type_kv_cache_manager import (
    MambaManager,
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer

from arctic_platform.inference.patching import ArcticPatch


def _sanitize_stale_gdn_metadata(metadata):
    accepted = metadata.num_accepted_tokens
    state_indices = metadata.spec_state_indices_tensor
    if accepted is None or state_indices is None:
        return metadata

    stale_rows = accepted == 0
    state_indices.masked_fill_(stale_rows.unsqueeze(-1), NULL_BLOCK_ID)
    accepted.clamp_(min=1)
    return metadata


class GDNAttentionMetadataBuilderPatch(
        ArcticPatch[GDNAttentionMetadataBuilder]):

    _orig_build = GDNAttentionMetadataBuilder.build

    def build(self, *args, **kwargs):
        metadata = self._orig_build(*args, **kwargs)
        return _sanitize_stale_gdn_metadata(metadata)


class SingleTypeKVCacheManagerPatch(
        ArcticPatch[SingleTypeKVCacheManager]):

    _orig_init = SingleTypeKVCacheManager.__init__

    def __init__(
        self,
        kv_cache_spec,
        block_pool,
        enable_caching,
        kv_cache_group_id,
        scheduler_block_size,
        dcp_world_size=1,
        pcp_world_size=1,
        needs_kv_cache_zeroing=False,
        max_admission_blocks_per_request=None,
    ):
        self._orig_init(
            kv_cache_spec,
            block_pool,
            enable_caching,
            kv_cache_group_id,
            scheduler_block_size,
            dcp_world_size,
            pcp_world_size,
            needs_kv_cache_zeroing,
            max_admission_blocks_per_request,
        )
        if isinstance(kv_cache_spec, MambaSpec):
            self._record_new_block_ids = needs_kv_cache_zeroing


class MambaManagerPatch(ArcticPatch[MambaManager]):

    _orig_allocate_new_blocks = MambaManager.allocate_new_blocks

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
    ):
        if (self.mamba_cache_mode != "align"
                or not self._record_new_block_ids):
            return self._orig_allocate_new_blocks(
                request_id, num_tokens, num_tokens_main_model)

        pool = self.block_pool
        pool_vars = vars(pool)
        had_override = "get_new_blocks" in pool_vars
        previous_override = pool_vars.get("get_new_blocks")
        original_get_new_blocks = pool.get_new_blocks
        allocated_blocks = []

        def tracked_get_new_blocks(num_blocks):
            blocks = original_get_new_blocks(num_blocks)
            allocated_blocks.extend(blocks)
            return blocks

        # The align path bypasses the base manager's allocation recorder.
        # Capture only physical allocations, excluding relocated state blocks
        # that can also appear in the method's return value.
        pool.get_new_blocks = tracked_get_new_blocks
        try:
            result = self._orig_allocate_new_blocks(
                request_id, num_tokens, num_tokens_main_model)
        finally:
            if had_override:
                pool.get_new_blocks = previous_override
            else:
                delattr(pool, "get_new_blocks")

        self.new_block_ids.extend(block.block_id
                                  for block in allocated_blocks)
        return result


class KVBlockZeroerPatch(ArcticPatch[KVBlockZeroer]):

    def __init__(
        self,
        device: torch.device,
        attn_groups_iter: Iterable[AttentionGroup],
        kernel_block_sizes: list[int],
        static_forward_context: dict[str, Any],
        num_blocks: int,
        runner_only_attn_layers: set[str] | None = None,
    ) -> None:
        self.device = device
        self._meta: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]
            | None) = None

        if runner_only_attn_layers is None:
            runner_only_attn_layers = set()
        seen_ptrs: dict[int, int] = {}
        seg_addrs: list[int] = []
        seg_block_strides: list[int] = []
        seg_page_sizes: list[int] = []

        for group in attn_groups_iter:
            spec = group.kv_cache_spec
            if not isinstance(spec, (AttentionSpec, MambaSpec)):
                continue
            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue
            kernel_bs = kernel_block_sizes[group.kv_cache_group_id]
            assert spec.block_size % kernel_bs == 0
            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                layer_cache = static_forward_context[layer_name].kv_cache
                kvs = (list(layer_cache)
                       if isinstance(layer_cache, (list, tuple)) else
                       [layer_cache])
                for kv in kvs:
                    if not isinstance(kv, torch.Tensor):
                        continue
                    if kv.device.type != self.device.type:
                        continue
                    data_ptr = kv.data_ptr()

                    assert kv.shape[0] % num_blocks == 0, (
                        f"{layer_name}: {kv.shape[0]} kernel blocks is not a "
                        f"multiple of {num_blocks} logical blocks")
                    ratio = kv.shape[0] // num_blocks

                    element_size = kv.element_size()
                    block_stride_bytes = kv.stride(0) * element_size
                    assert block_stride_bytes % 4 == 0
                    assert kv.shape[0] % ratio == 0
                    outer_dims = [
                        dim for dim in range(1, kv.ndim)
                        if kv.stride(dim) * element_size > block_stride_bytes
                    ]
                    outer_strides = [
                        kv.stride(dim) * element_size for dim in outer_dims
                    ]
                    inner_dims = [
                        dim for dim in range(1, kv.ndim)
                        if dim not in outer_dims
                    ]
                    kernel_page_bytes = element_size + sum(
                        (kv.shape[dim] - 1) * kv.stride(dim) * element_size
                        for dim in inner_dims)
                    assert kernel_page_bytes % 4 == 0
                    logical_block_stride_bytes = block_stride_bytes * ratio
                    ranges = (range(kv.shape[dim]) for dim in outer_dims)
                    for outer in iprod(*ranges):
                        offset_bytes = sum(
                            index * stride
                            for index, stride in zip(outer, outer_strides))
                        assert (data_ptr + offset_bytes) % 4 == 0
                        for virtual_index in range(ratio):
                            address = (data_ptr + offset_bytes
                                       + virtual_index * block_stride_bytes)
                            if (index := seen_ptrs.get(address)) is not None:
                                assert seg_block_strides[index] == (
                                    logical_block_stride_bytes // 4)
                                seg_page_sizes[index] = max(
                                    seg_page_sizes[index],
                                    kernel_page_bytes // 4)
                                continue
                            seen_ptrs[address] = len(seg_addrs)
                            seg_addrs.append(address)
                            seg_block_strides.append(
                                logical_block_stride_bytes // 4)
                            seg_page_sizes.append(kernel_page_bytes // 4)

        if not seg_addrs:
            self._meta = None
            return

        max_page_size_el = max(seg_page_sizes)
        block_size = min(1 << (max_page_size_el - 1).bit_length(),
                         1024)
        self._meta = (
            torch.tensor(seg_addrs, dtype=torch.uint64,
                         device=self.device),
            torch.tensor(seg_block_strides,
                         dtype=torch.int64,
                         device=self.device),
            torch.tensor(seg_page_sizes,
                         dtype=torch.int64,
                         device=self.device),
            (max_page_size_el + block_size - 1) // block_size,
            block_size,
            len(seg_addrs),
        )


_PATCHED = False


def apply_dflash2_nan_fixes() -> None:
    global _PATCHED
    if _PATCHED:
        return

    GDNAttentionMetadataBuilderPatch.apply_patch()
    SingleTypeKVCacheManagerPatch.apply_patch()
    MambaManagerPatch.apply_patch()
    KVBlockZeroerPatch.apply_patch()
    _PATCHED = True
