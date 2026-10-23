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

"""End Mamba prefill chunks on the EAGLE replay boundaries (DSS issue 0b15d1bb).

With DFlash2 on Qwen3.8-27B the KV groups use 416- and 832-token blocks. A prefix hit resumes at
B = floor((N - 1) / 832) * 832 - 832 and needs Mamba state saved at exactly B, but vLLM 0.31.0 ends
prefill chunks on the 416 grid, so that state is never saved: prompts with odd floor((N - 1) / 416)
never hit the prefix cache and the rest miss twice. This also ends a chunk at each replay boundary
(one extra scheduler step per prompt) and does nothing without EAGLE groups. Remove once the pinned
vLLM stops prefill at ``KVCacheCoordinator.get_replay_boundaries`` itself.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

_MARKER = "_arctic_replay_boundary_stop"


def ensure_mamba_replay_boundary_stop() -> None:
    """Patch vLLM so the Mamba prefill split also ends chunks on replay boundaries."""
    from arctic_platform.inference.utils import require_supported_vllm_version

    require_supported_vllm_version("Mamba replay-boundary prefill stop")

    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler._mamba_block_aligned_split, _MARKER, False):
        return

    original = Scheduler._mamba_block_aligned_split

    @wraps(original)
    def _mamba_block_aligned_split(
        self: Any,
        request: Any,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        num_new_tokens = original(
            self, request, num_new_tokens, num_new_local_computed_tokens, num_external_computed_tokens
        )
        start = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
        coordinator = self.kv_cache_manager.coordinator
        # Every replay boundary is below the prompt end, so only a prefill chunk can contain one.
        if start >= request.num_prompt_tokens or len(coordinator.eagle_group_ids) == 0:
            return num_new_tokens
        end = start + num_new_tokens
        # Same rule as the stock stops: end at the earliest one strictly inside the chunk.
        end = min((stop for stop in coordinator.get_replay_boundaries(request) if start < stop < end), default=end)
        return end - start

    setattr(_mamba_block_aligned_split, _MARKER, True)
    Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split
