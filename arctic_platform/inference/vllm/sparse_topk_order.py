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

"""Give the sparse-attention key lists of GLM-5.3-Flash and Qwen3.8-Flash-Next a fixed order.

vLLM's indexer top-k kernels (``top_k_per_row_prefill``, ``cooperative_topk``,
``persistent_topk``) place each selected id through an atomic slot counter, so
the same set comes out in a different order on every run. Nothing sorts it
before the sparse attention kernels, which accumulate keys in list order
(online softmax), so the same input gives different output bits, and the next
FP8 indexer can then select a different set. Sorting each row ascending, with
the -1 padding last, keeps the set and makes the order a function of it.
"""

import torch

from arctic_platform.inference.patching import ArcticPatch

_PATCHED = False


def sort_topk_ids(ids: torch.Tensor) -> torch.Tensor:
    """Each row's ids ascending, with the -1 padding last."""
    pad = torch.iinfo(ids.dtype).max
    ordered = torch.where(ids < 0, pad, ids).sort(dim=-1).values
    return torch.where(ordered == pad, -1, ordered)


def apply_sparse_topk_order_fix() -> None:
    global _PATCHED
    if _PATCHED:
        return

    from vllm.models.glm5next.nvidia.ops import kpool_compress
    from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer

    expand_pools_and_append_tail = kpool_compress.expand_pools_and_append_tail
    topk = qsa_indexer._topk

    class KpoolCompressPatch(ArcticPatch[kpool_compress]):
        # GLM-5.3-Flash prefill and decode both expand their selected pools here. The
        # tail tokens sit above every complete pool, so sorted pools give a sorted list.
        @staticmethod
        def expand_pools_and_append_tail(pool_ids, seq_lens, pool_size):
            return expand_pools_and_append_tail(sort_topk_ids(pool_ids), seq_lens, pool_size)

    class QsaIndexerPatch(ArcticPatch[qsa_indexer]):
        # Qwen3.8-Flash-Next decode and prefill selection both write block ids here.
        # expand_qsa_block_indices reads the first min(visible, block_topk) entries as valid.
        @staticmethod
        def _topk(logits, visible_blocks, token_topk, compress_ratio, block_indices, topk_workspace):
            topk(logits, visible_blocks, token_topk, compress_ratio, block_indices, topk_workspace)
            block_indices.copy_(sort_topk_ids(block_indices))

    KpoolCompressPatch.apply_patch()
    QsaIndexerPatch.apply_patch()
    _PATCHED = True
