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
"""The GLM-5.2/5.3 DSA indexer (glm52 fp8_indexer): its DeepGEMM path picks its portable path's keys.

The DeepGEMM path scores with the MQA-logits kernel of vLLM's bundled DeepGEMM on SM90, and selects with FlashInfer's
top-k when it is installed. The portable path scores with the Triton kernel and selects with torch.topk. vLLM 0.33's
kernel takes max_seqlen_k instead of clean_logits and returns compressed logits; called with the old arguments, it
raised TypeError.

    python -m pytest -q tests/model/test_glm52_fp8_indexer.py
"""

from __future__ import annotations

import pytest
import torch

from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import set_seed


@require_torch_gpu
def test_fast_indexer_path_selects_the_portable_paths_keys(monkeypatch):
    """DeepGEMM scores + FlashInfer top-k (when installed) pick the keys of Triton scores + torch.topk, over two packed
    sequences and an unaligned query range (as under context parallelism)."""
    from arctic_platform.model.implementations.glm52.models.kernels import fp8_indexer as indexer

    if indexer._deep_gemm_mqa_logits(torch.cuda.get_device_capability()[0]) is None:
        pytest.skip("the indexer scores with DeepGEMM on SM90, where vLLM bundles it")

    set_seed(2)
    S, H, D, topk = 5000, 32, 128, 2048
    q = torch.randn(S, H, D, device="cuda").bfloat16()
    k = torch.randn(S, D, device="cuda").bfloat16()
    w = torch.randn(S, H, device="cuda").bfloat16()
    pos = torch.cat([torch.arange(1700), torch.arange(S - 1700)]).cuda()
    ks = (torch.arange(S, device="cuda") - pos).to(torch.int32)
    ke = torch.arange(1, S + 1, device="cuda", dtype=torch.int32)
    shard = slice(1234, S)  # an unaligned query range, as under context parallelism
    fast = indexer.fp8_indexer(q[shard], k, w[shard], ks[shard], ke[shard], topk)
    monkeypatch.setattr(indexer, "_deep_gemm_mqa_logits", lambda capability_major: None)
    monkeypatch.setattr(indexer, "_flashinfer_top_k", None)
    portable = indexer.fp8_indexer(q[shard], k, w[shard], ks[shard], ke[shard], topk)

    assert torch.equal((fast == S).sum(-1), (portable == S).sum(-1))
    # Scores agree to ~1e-7, so only near-ties at the top-k boundary may swap.
    rows_equal = (fast.sort(-1).values == portable.sort(-1).values).all(-1).float().mean().item()
    assert rows_equal > 0.995
