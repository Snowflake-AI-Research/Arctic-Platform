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

"""Give the sparse-attention key lists of GLM-5.3, GLM-5.3-Flash and Qwen3.8-Flash-Next a fixed order.

vLLM's indexer top-k kernels (``top_k_per_row_prefill``, ``cooperative_topk``,
``persistent_topk``) hand out output slots with ``atomicAdd``, so the same set
can come back in a different order on each run, and the sparse attention
kernels sum keys in list order. Sorting each row ascending, -1 padding last,
keeps the set and makes its order a function of it. It does not change which
ids top-k selects.
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

    from vllm.models.deepseek_v32 import attention as dsa_attention
    from vllm.models.glm5next.nvidia.ops import kpool_compress
    from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer

    expand_pools_and_append_tail = kpool_compress.expand_pools_and_append_tail
    topk = qsa_indexer._topk
    sparse_attn_indexer = dsa_attention.sparse_attn_indexer

    class KpoolCompressPatch(ArcticPatch[kpool_compress]):
        # GLM-5.3-Flash prefill and decode both expand their selected pools here; the tail stays after any padding.
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

    class DsaAttentionPatch(ArcticPatch[dsa_attention]):
        # GLM-5.3's DSA indexer: the sparse MLA kernels read the token ids from this shared buffer afterwards.
        @staticmethod
        def sparse_attn_indexer(hidden_states, *args, **kwargs):
            ids = sparse_attn_indexer(hidden_states, *args, **kwargs)
            rows = ids[: hidden_states.shape[0]]
            rows.copy_(sort_topk_ids(rows))
            return ids

    KpoolCompressPatch.apply_patch()
    QsaIndexerPatch.apply_patch()
    DsaAttentionPatch.apply_patch()
    _PATCHED = True
