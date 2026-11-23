# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Keep a running request's reusable prefix-cache boundary until it completes.

A hybrid hit at a boundary needs the attention blocks (held until completion),
the Mamba state there and every sliding-window group's tail ending there; the
last two retire into the free queue during the run. Each request now keeps
referencing the newest such blocks per group and returns them through the
normal, fenced, tail-first completion free; older retired Mamba checkpoints
are still refreshed there when resident. CPU tested, not GPU validated.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock

from arctic_platform.inference.patching import ArcticPatch

_APPLIED = False


def _hold_newest(manager, request_id: str, retiring: list[tuple[int, KVCacheBlock]], keep: int) -> None:
    """Keep referencing the newest ``keep`` hashed blocks being retired; release older holds."""
    retiring = [(idx, block) for idx, block in retiring if not block.is_null and block.block_hash is not None]
    if not retiring:
        return
    held = manager._held_boundary_blocks[request_id]
    for idx, block in retiring:
        if idx not in held:
            block.ref_cnt += 1
            held[idx] = block
    stale = sorted(held)[:-keep]
    if stale:
        manager.block_pool.free_blocks([held.pop(idx) for idx in reversed(stale)])


def ensure_mamba_completion_refresh() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from arctic_platform.inference.utils import require_supported_vllm_version

    require_supported_vllm_version("Mamba completion checkpoint refresh")

    from vllm.v1.core.single_type_kv_cache_manager import MambaManager, SlidingWindowManager
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    original_init = MambaManager.__init__
    original_remove = MambaManager._remove_blocks_in_range
    original_skipped = MambaManager.remove_skipped_blocks
    original_pop = MambaManager.pop_blocks_for_free

    class MambaCompletionRefreshPatch(ArcticPatch[MambaManager]):
        def __init__(self, *args, **kwargs) -> None:
            original_init(self, *args, **kwargs)
            self._retired_checkpoint_hashes: dict[str, dict[int, BlockHashWithGroupId]] = defaultdict(dict)
            self._held_boundary_blocks: dict[str, dict[int, KVCacheBlock]] = defaultdict(dict)

        def _remove_blocks_in_range(self, request_id: str, first_block: int, last_block: int) -> None:
            if self.mamba_cache_mode == "align":
                blocks = self.req_to_blocks.get(request_id, [])
                first_block = max(first_block, self._num_retired_blocks.get(request_id, 0))
                retiring = [(idx, blocks[idx]) for idx in range(first_block, min(last_block, len(blocks)))]
                for idx, block in retiring:
                    if not block.is_null and block.block_hash is not None:
                        self._retired_checkpoint_hashes[request_id][idx] = block.block_hash
                _hold_newest(self, request_id, retiring, 1)
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
                        _hold_newest(self, request_id, [(idx, block)], 1)
            return original_skipped(self, request_id, processed_computed_tokens, num_prompt_tokens)

        def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
            if self.mamba_cache_mode == "align":
                blocks = self.req_to_blocks.get(request_id, [])
                for idx, block in self._held_boundary_blocks.pop(request_id, {}).items():
                    blocks[idx] = block  # the hold becomes the table's reference
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

    original_window_init = SlidingWindowManager.__init__
    original_window_remove = SlidingWindowManager._remove_blocks_in_range
    original_window_pop = SlidingWindowManager.pop_blocks_for_free
    original_window_cap = SlidingWindowSpec.max_admission_blocks_per_request

    class SlidingWindowCompletionHoldPatch(ArcticPatch[SlidingWindowManager]):
        def __init__(self, *args, **kwargs) -> None:
            original_window_init(self, *args, **kwargs)
            self._held_boundary_blocks: dict[str, dict[int, KVCacheBlock]] = defaultdict(dict)

        def _remove_blocks_in_range(self, request_id: str, first_block: int, last_block: int) -> None:
            blocks = self.req_to_blocks.get(request_id, [])
            retiring = []
            for idx in range(min(last_block, len(blocks)) - 1, first_block - 1, -1):
                if blocks[idx].is_null:
                    break
                retiring.append((idx, blocks[idx]))
            keep = self._contiguous_blocks_for_hit(self.sliding_window, self.block_size, self.use_eagle)
            _hold_newest(self, request_id, retiring, keep)
            return original_window_remove(self, request_id, first_block, last_block)

        def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
            blocks = self.req_to_blocks.get(request_id, [])
            for idx, block in self._held_boundary_blocks.pop(request_id, {}).items():
                blocks[idx] = block
            return original_window_pop(self, request_id)

    class SlidingWindowAdmissionPatch(ArcticPatch[SlidingWindowSpec]):
        def max_admission_blocks_per_request(self, max_in_flight_tokens: int, max_model_len: int) -> int:
            # Real-held blocks now include the held tail (EAGLE's widest).
            held = SlidingWindowManager._contiguous_blocks_for_hit(self.sliding_window, self.block_size, True)
            return original_window_cap(self, max_in_flight_tokens, max_model_len) + held

    MambaCompletionRefreshPatch.apply_patch()
    SlidingWindowCompletionHoldPatch.apply_patch()
    SlidingWindowAdmissionPatch.apply_patch()
    _APPLIED = True
