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
"""The fused DeepSeek-V4.1 indexer scores against the FP32 reference contraction."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused index scores run on CUDA")


@pytest.mark.parametrize("key_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("queries,entries", [(1, 7), (130, 1000), (257, 4099)])
def test_fused_index_scores_match_the_fp32_contraction(queries, entries, key_dtype):
    from arctic_platform.model.implementations.deepseek_v41.attention import _index_scores
    from arctic_platform.model.implementations.deepseek_v41.attention import _topk_indices
    from arctic_platform.model.implementations.kernels.deepseek_v4.index_scores import index_scores
    from arctic_platform.model.implementations.kernels.deepseek_v4.index_scores import split_keys

    torch.manual_seed(0)
    heads, dim = 32, 128
    query = torch.randn(1, queries, heads, dim, device="cuda").bfloat16()
    keys = torch.randn(1, entries, dim, device="cuda").to(key_dtype)
    weights = (torch.randn(1, queries, heads, device="cuda") * heads**-0.5).bfloat16()

    reference = _index_scores(query, keys.float(), weights)[0]
    fused = index_scores(query[0], split_keys(keys[0]), weights[0])
    # FP32 sums in another order; with the key residual the keys themselves are kept to 2^-17.
    torch.testing.assert_close(fused, reference, rtol=2e-5, atol=2e-5 * reference.abs().max().item())

    visible = torch.ones(queries, entries, dtype=torch.bool, device="cuda")
    k = min(64, entries)
    same = (_topk_indices(fused, visible, k).sort(-1).values == _topk_indices(reference, visible, k).sort(-1).values)
    assert same.float().mean().item() > 0.999
