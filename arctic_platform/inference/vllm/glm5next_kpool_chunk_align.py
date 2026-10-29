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

"""Refuse GLM-5.3-Flash's pooled indexer cache unless prefill chunks are pool-aligned.

vLLM 0.31's ``_kpool_compress_insert`` (models/glm5next/nvidia/sparse_indexer.py)
builds each pool's index key from the last ``index_kpool`` rows of the batch, so a
chunk that starts mid-pool leaves that pool unwritten or pools it from the previous
request's keys. Only ``mamba_cache_mode == "align"``, which vLLM sets when prefix
caching is on, ends chunks on block boundaries; with ``"none"`` a chunk ends where
the token budget runs out.
"""

from arctic_platform.inference.patching import ArcticPatch

_PATCHED = False


def check_kpool_chunk_alignment(cache_config, index_kpool: int) -> None:
    if index_kpool > 1 and cache_config.mamba_cache_mode != "align":
        raise ValueError(
            f"GLM-5.3-Flash's pooled indexer (index_kpool={index_kpool}) needs pool-aligned prefill chunks, which "
            f"vLLM schedules only with mamba_cache_mode='align' (got {cache_config.mamba_cache_mode!r}): "
            "set enable_prefix_caching=True."
        )


def apply_glm5next_kpool_chunk_align_check() -> None:
    global _PATCHED
    if _PATCHED:
        return

    from vllm.models.glm5next.common.attention import Glm5NextIndexerCache

    original_init = Glm5NextIndexerCache.__init__

    class Glm5NextIndexerCacheAlignPatch(ArcticPatch[Glm5NextIndexerCache]):
        def __init__(self, *args, cache_config, index_kpool, **kwargs):
            check_kpool_chunk_alignment(cache_config, index_kpool)
            original_init(self, *args, cache_config=cache_config, index_kpool=index_kpool, **kwargs)

    Glm5NextIndexerCacheAlignPatch.apply_patch()
    _PATCHED = True
