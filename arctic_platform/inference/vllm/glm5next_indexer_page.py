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

"""Keep GLM-5.3-Flash indexer pages legal for DeepGEMM.

vLLM's hybrid aligner grows the shared KV block so it covers the Mamba page.
For GLM-5.3-Flash that block is 640 tokens. The indexer stores one state per
``index_kpool`` tokens (4), so a 640-token block contains 160 states. DeepGEMM's
paged-MQA kernel only accepts 32 or 64 states per page on Hopper, and the
indexer metadata builder passes that 160-state count through. CUDA-graph
capture then dies in ``get_paged_mqa_logits_metadata``.

The kernel page vLLM already computes for this indexer is
``index_kpool * {32, 64}`` tokens. On a 640-token block the common size is
128 tokens, which is 32 states and divides the Mamba page. This patch makes
the GLM indexer advertise those pages, then builds indexer metadata against
that kernel page instead of recompressing it back to the 160-state block.

Remove this once the pinned vLLM contains
https://github.com/vllm-project/vllm/pull/57169.
"""

from dataclasses import replace

from arctic_platform.inference.patching import ArcticPatch

# DeepGEMM ``get_paged_mqa_logits_metadata`` accepts these state counts.
# SM120 drops the 32-state page; every other CUDA arch used here keeps both.
_HOPPER_PAGE_STATES = (32, 64)
_SM120_PAGE_STATES = (64,)


def paged_mqa_page_sizes(sm120: bool) -> tuple[int, ...]:
    return _SM120_PAGE_STATES if sm120 else _HOPPER_PAGE_STATES


def indexer_kernel_block_sizes(index_kpool: int, page_states: tuple[int, ...]) -> list[int]:
    """Token counts of one DeepGEMM page, for this model's state packing."""
    return [index_kpool * states for states in page_states]


def metadata_block_size(
    manager_block_size: int,
    kernel_block_size: int | None,
    tokens_per_state: int,
    page_states: tuple[int, ...],
) -> int | None:
    """Block size the indexer metadata builder should see.

    Returns the kernel page when it is a legal DeepGEMM page and the worker
    has already expanded the manager block into those pages. Returns ``None``
    when the stock builder is already safe, including DeepSeek-V3.2 (one token
    per state) and a kernel page DeepGEMM would reject.
    """
    if (
        not isinstance(tokens_per_state, int)
        or tokens_per_state <= 1
        or not kernel_block_size
        or kernel_block_size % tokens_per_state != 0
        or manager_block_size % kernel_block_size != 0
    ):
        return None
    if kernel_block_size // tokens_per_state not in page_states:
        return None
    if manager_block_size == kernel_block_size:
        return None
    return kernel_block_size


def build_with_deepgemm_page(builder, original_build, page_states, args, kwargs):
    """Run ``original_build`` against a legal DeepGEMM page, then restore."""
    spec = builder.kv_cache_spec
    block_size = metadata_block_size(
        spec.block_size,
        getattr(builder, "kernel_block_size", None),
        getattr(spec, "tokens_per_state", 1),
        page_states,
    )
    if block_size is None:
        return original_build(builder, *args, **kwargs)
    # ``num_states`` and the block-table recompress both read
    # ``spec.block_size``. Showing the kernel page makes the metadata and
    # slot mapping use 32 or 64 states, and leaves the worker's
    # already-expanded block ids alone.
    builder.kv_cache_spec = replace(spec, block_size=block_size)
    try:
        return original_build(builder, *args, **kwargs)
    finally:
        builder.kv_cache_spec = spec


_PATCHED = False
_BACKENDS: dict[tuple[int, tuple[int, ...]], type] = {}


def _indexer_backend(index_kpool: int, page_states: tuple[int, ...]):
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend

    key = (index_kpool, page_states)
    backend = _BACKENDS.get(key)
    if backend is not None:
        return backend

    sizes = indexer_kernel_block_sizes(index_kpool, page_states)

    class Glm5NextIndexerBackend(DeepseekV32IndexerBackend):
        @staticmethod
        def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int]:
            return list(sizes)

    _BACKENDS[key] = Glm5NextIndexerBackend
    return Glm5NextIndexerBackend


def apply_glm5next_indexer_page_fix() -> None:
    global _PATCHED
    if _PATCHED:
        return

    from vllm.models.glm5next.common.attention import Glm5NextIndexerCache
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder

    page_states = paged_mqa_page_sizes(_is_sm120(current_platform))
    original_backend = Glm5NextIndexerCache.get_attn_backend
    original_build = DeepseekV32IndexerMetadataBuilder.build

    class Glm5NextIndexerCachePatch(ArcticPatch[Glm5NextIndexerCache]):
        def get_attn_backend(self):
            kpool = int(getattr(self, "_index_kpool", 1) or 1)
            # DeepSeek-V3.2 packs one token per state and keeps its own
            # 64-token kernel page. GLM is the cache that packs several
            # tokens into each state.
            if kpool <= 1:
                return original_backend(self)
            return _indexer_backend(kpool, page_states)

    class DeepseekV32IndexerMetadataBuilderPatch(ArcticPatch[DeepseekV32IndexerMetadataBuilder]):

        def build(self, *args, **kwargs):
            return build_with_deepgemm_page(self, original_build, page_states, args, kwargs)

    Glm5NextIndexerCachePatch.apply_patch()
    DeepseekV32IndexerMetadataBuilderPatch.apply_patch()
    _PATCHED = True


def _is_sm120(platform) -> bool:
    checker = getattr(platform, "is_device_capability_family", None)
    if checker is None:
        return False
    try:
        return bool(checker(120))
    except Exception:
        return False
