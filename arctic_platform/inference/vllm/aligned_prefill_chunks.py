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

"""Opt-in: split chunked prefills only at fixed positions, so a trainer can reproduce them.

A GatedDeltaNet layer prefills each scheduled chunk as its own chunked-scan call that resumes from the state the
previous chunk left, so where a prompt is cut changes the result's rounding. vLLM cuts wherever the step's token
budget runs out. With ``ARCTIC_PREFILL_CHUNK_ALIGN=C`` every chunk of a prompt except its last ends at a multiple of
``C`` tokens from the sequence start; a chunk that cannot reach one waits for the next step. The vLLM-aligned trainer
(``vllm_aligned_prefill_chunk=C``) splits its prefill at the same positions.

Only the plain cache mode is supported: prefix caching ("align" mode) adds stops that depend on the cache contents.
"""

from __future__ import annotations

import os

ENV_VAR = "ARCTIC_PREFILL_CHUNK_ALIGN"

_APPLIED = False


def prefill_chunk_alignment() -> int:
    value = os.environ.get(ENV_VAR, "0")
    alignment = int(value)
    if alignment < 0:
        raise ValueError(f"{ENV_VAR}={value!r} must be a non-negative chunk size")
    return alignment


def aligned_chunk_tokens(start: int, num_new_tokens: int, prefill_end: int, alignment: int) -> int:
    """The tokens of a prefill chunk from ``start`` once its end is floored to a multiple of ``alignment``.

    The prompt's last chunk (reaching ``prefill_end``) is unchanged; ``0`` means the request waits this step.
    """
    if alignment <= 0 or num_new_tokens <= 0 or start >= prefill_end:
        return num_new_tokens
    end = start + num_new_tokens
    if end >= prefill_end:
        return num_new_tokens
    aligned_end = end // alignment * alignment
    return max(aligned_end - start, 0)


def ensure_aligned_prefill_chunks() -> None:
    global _APPLIED
    alignment = prefill_chunk_alignment()
    if _APPLIED or alignment == 0:
        return
    from vllm.v1.core.sched.scheduler import Scheduler

    original_init = Scheduler.__init__
    original_split = Scheduler._mamba_block_aligned_split

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.cache_config.mamba_cache_mode == "align":
            raise ValueError(f"{ENV_VAR} does not support prefix caching (mamba_cache_mode='align')")
        self._arctic_plain_mamba_split = not self.need_mamba_block_aligned_split
        # The scheduler calls the split hook only when this is set.
        self.need_mamba_block_aligned_split = True

    def _mamba_block_aligned_split(
        self, request, num_new_tokens, num_new_local_computed_tokens=0, num_external_computed_tokens=0
    ):
        if not getattr(self, "_arctic_plain_mamba_split", False):
            num_new_tokens = original_split(
                self, request, num_new_tokens, num_new_local_computed_tokens, num_external_computed_tokens
            )
        start = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        return aligned_chunk_tokens(start, num_new_tokens, prefill_end, alignment)

    Scheduler.__init__ = __init__
    Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split
    _APPLIED = True
