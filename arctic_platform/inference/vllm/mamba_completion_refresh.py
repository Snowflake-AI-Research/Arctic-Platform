# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refresh retired Mamba checkpoints through the normal, fenced completion free.

Candidate for vLLM 0.30.0; CPU ordering tested, not GPU validated. This only
reorders still-resident checkpoints; retention policy and allocation are unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock

from arctic_platform.inference.patching import ArcticPatch

_APPLIED = False


def ensure_mamba_completion_refresh() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from arctic_platform.inference.utils import require_supported_vllm_version

    require_supported_vllm_version("Mamba completion checkpoint refresh")

    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    original_init = MambaManager.__init__
    original_remove = MambaManager._remove_blocks_in_range
    original_skipped = MambaManager.remove_skipped_blocks
    original_pop = MambaManager.pop_blocks_for_free

    class MambaCompletionRefreshPatch(ArcticPatch[MambaManager]):
        def __init__(self, *args, **kwargs) -> None:
            original_init(self, *args, **kwargs)
            self._retired_checkpoint_hashes: dict[str, dict[int, BlockHashWithGroupId]] = defaultdict(dict)

        def _remove_blocks_in_range(self, request_id: str, first_block: int, last_block: int) -> None:
            if self.mamba_cache_mode == "align":
                blocks = self.req_to_blocks.get(request_id, [])
                first_block = max(first_block, self._num_retired_blocks.get(request_id, 0))
                for idx in range(first_block, min(last_block, len(blocks))):
                    block = blocks[idx]
                    if not block.is_null and block.block_hash is not None:
                        self._retired_checkpoint_hashes[request_id][idx] = block.block_hash
            return original_remove(self, request_id, first_block, last_block)

        def remove_skipped_blocks(
            self, request_id: str, processed_computed_tokens: int, num_prompt_tokens: int | None = None
        ) -> None:
            if self.mamba_cache_mode == "align":
                idx = self.last_state_block_idx.get(request_id)
                if idx is not None and idx < (processed_computed_tokens - 1) // self.block_size:
                    block = self.req_to_blocks[request_id][idx]
                    if not block.is_null and block.block_hash is not None:
                        self._retired_checkpoint_hashes[request_id][idx] = block.block_hash
            return original_skipped(self, request_id, processed_computed_tokens, num_prompt_tokens)

        def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
            if self.mamba_cache_mode == "align":
                blocks = self.req_to_blocks.get(request_id, [])
                retired = self._retired_checkpoint_hashes.pop(request_id, {})
                for idx, block_hash in sorted(retired.items(), reverse=True):
                    block = self.block_pool.cached_block_hash_to_block.get_one_block(block_hash)
                    if block is None or block.ref_cnt != 0 or not blocks[idx].is_null:
                        continue
                    # Rejoin the normal tail-first release, including its async fence.
                    self.block_pool.free_block_queue.remove(block)
                    block.ref_cnt += 1
                    blocks[idx] = block
            return original_pop(self, request_id)

    MambaCompletionRefreshPatch.apply_patch()
    _APPLIED = True
