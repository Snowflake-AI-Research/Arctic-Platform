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

from functools import cache

from vllm.distributed.aux_output_connector.connector import AuxOutputSchedulerConnector
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash
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
                # R-SWA recycles decode gaps and HiSparse publishes only after host writeback.
                if type(manager) is FullAttentionManager and manager.kv_cache_spec.prefix_cacheable:
                    cacheable = coordinator.get_num_cacheable_tokens(settled, manager.kv_cache_group_id)
                    _cache_computed_tail(manager, request, cacheable)
                    routes = self.aux_output_connector
                    if routes is not None and routes._manager is manager:
                        _retain_partial_routes(routes, request, cacheable)
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


def _retain_partial_routes(routes: AuxOutputSchedulerConnector, request: Request, cacheable: int) -> None:
    grain = routes._hash_block_size
    per_block = routes._block_size // grain
    prompt_limit = request.num_prompt_tokens - int(not routes._manager.use_eagle)
    for limit in (min(prompt_limit, cacheable), cacheable):
        end = min(limit // grain, len(request.block_hashes))
        count = end % per_block
        if end <= 0 or not count:
            continue
        last = request.block_hashes[end - 1]
        key = maybe_convert_block_hash(last)
        if key not in routes._kv_cached and routes._gpu_cached(last):
            # Native eviction and reset release the references stored under this endpoint.
            hashes = request.block_hashes[end - count:end]
            routes._kv_cached[key] = hashes
            routes._retained_hashes.extend(hashes)


class AuxDecodedTailPatch(ArcticPatch[AuxOutputSchedulerConnector]):
    _orig_max_cache_hit = AuxOutputSchedulerConnector.max_cache_hit

    def max_cache_hit(self, request: Request) -> int:
        hit = self._orig_max_cache_hit(request)
        # Only the first missing physical block can extend the contiguous route prefix.
        grain = self._hash_block_size
        next_block = hit // self._block_size + 1
        end = min(len(request.block_hashes), next_block * (self._block_size // grain) - 1)
        for index in range(end - 1, hit // grain - 1, -1):
            key = maybe_convert_block_hash(request.block_hashes[index])
            if key in self._kv_cached:
                self._kv_cached.move_to_end(key)
                return min((index + 1) * grain, request.num_tokens - 1)
        return hit


@cache
def ensure_decoded_tail_prefix_hit() -> None:
    SchedulerDecodedTailPatch.apply_patch()
    AuxDecodedTailPatch.apply_patch()
