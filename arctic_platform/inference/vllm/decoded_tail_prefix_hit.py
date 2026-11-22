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

"""Expose the final computed attention prefix to hybrid cache lookups.

Decode context parallelism can make an attention block span several prefix
hashes. At generation completion, register the last reusable interior hash so
a continuation can share the block when its other cache groups also hit.
"""

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.request import Request

from arctic_platform.inference.patching import ArcticPatch


class SchedulerDecodedTailPatch(ArcticPatch[Scheduler]):
    _orig_handle_stopped_request = Scheduler._handle_stopped_request

    def _handle_stopped_request(self, request: Request) -> bool:
        # Resumable stop handling can rewind the counters before returning.
        settled = min(request.num_computed_tokens - request.num_in_flight_tokens, request.num_tokens - 1)
        finished = self._orig_handle_stopped_request(request)
        coordinator = self.kv_cache_manager.coordinator
        if finished and coordinator.enable_partial_hash_hits:
            for manager in coordinator.single_type_managers:
                if (
                    type(manager) is FullAttentionManager
                    and manager.enable_caching
                    and manager.kv_cache_spec.prefix_cacheable
                ):
                    cacheable = coordinator.get_num_cacheable_tokens(settled, manager.kv_cache_group_id)
                    _cache_computed_tail(manager, request, cacheable)
        return finished


def _cache_computed_tail(manager: FullAttentionManager, request: Request, cacheable: int) -> None:
    hash_size = manager.block_pool.hash_block_size
    boundary = min(cacheable // hash_size, len(request.block_hashes)) * hash_size
    if boundary < request.num_prompt_tokens or boundary % manager.block_size == 0:
        return
    block = manager.req_to_blocks[request.request_id][boundary // manager.block_size]
    # Promotion replaces the shorter primary. Reinsert it as an alias for prompt resends.
    previous = block.block_hash_num_tokens
    boundaries = (boundary, previous) if previous is not None and previous < boundary else (boundary,)
    for length in boundaries:
        manager.block_pool.cache_partial_block(
            request=request,
            block=block,
            num_tokens=length,
            kv_cache_group_id=manager.kv_cache_group_id,
            block_size=manager.block_size,
        )


_PATCHED = False


def ensure_decoded_tail_prefix_hit() -> None:
    global _PATCHED
    if not _PATCHED:
        SchedulerDecodedTailPatch.apply_patch()
        _PATCHED = True
