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

"""Keep a finished request's decoded tokens reachable at prefix-hash granularity.

When a full-attention cache block spans several hash blocks (decode context
parallelism multiplies the block by the DCP size), a prefix hit inside the
block exists only as a partial prefix-cache entry. vLLM registers one at the
prompt's last hash boundary, so a follow-up request that resends a finished
request's prompt and output re-prefills every decoded token past the last
full block. When the request is freed, its last computed hash boundary is
registered too, so the follow-up hits there, as it does when the block equals
the hash block.
"""

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

from arctic_platform.inference.patching import ArcticPatch


class KVCacheManagerDecodedTailPatch(ArcticPatch[KVCacheManager]):

    _orig_free = KVCacheManager.free

    def free(self, request) -> None:
        if self.enable_caching:
            for manager in self.coordinator.single_type_managers:
                if type(manager) is FullAttentionManager and manager.kv_cache_spec.prefix_cacheable:
                    _cache_computed_tail(manager, request)
        self._orig_free(request)


def _cache_computed_tail(manager: FullAttentionManager, request) -> None:
    hash_block_size = manager.block_pool.hash_block_size
    boundary = min(request.num_computed_tokens, len(request.block_hashes) * hash_block_size)
    boundary -= boundary % hash_block_size
    blocks = manager.req_to_blocks.get(request.request_id, ())
    # A boundary on a block edge is already cached as that full block.
    if boundary == 0 or boundary % manager.block_size == 0 or boundary // manager.block_size >= len(blocks):
        return
    manager.block_pool.cache_partial_block(
        request=request,
        block=blocks[boundary // manager.block_size],
        num_tokens=boundary,
        kv_cache_group_id=manager.kv_cache_group_id,
        block_size=manager.block_size,
    )


_PATCHED = False


def ensure_decoded_tail_prefix_hit() -> None:
    global _PATCHED
    if not _PATCHED:
        KVCacheManagerDecodedTailPatch.apply_patch()
        _PATCHED = True
