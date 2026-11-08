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

from arctic_platform.inference.vllm.glm5_batch_invariance import _single_split


def test_sparse_mla_attention_runs_one_split():
    seen = {}

    def flash_attn_varlen_func(*args, **kwargs):
        seen.update(kwargs)
        return "out"

    call = _single_split(flash_attn_varlen_func)
    assert call(q=1, num_splits=0) == "out"
    assert seen == {"q": 1, "num_splits": 1}
    assert call.__wrapped__ is flash_attn_varlen_func


def test_grouped_topk_is_row_invariant_and_matches_the_reference_math():
    import torch

    from arctic_platform.inference.vllm.glm5_batch_invariance import grouped_topk

    torch.manual_seed(0)
    logits = torch.randn(7, 288)
    bias = torch.randn(288) * 0.1
    kwargs = dict(topk=8, renormalize=True, num_expert_group=1, topk_group=1, scoring_func="sigmoid",
                  routed_scaling_factor=2.5, e_score_correction_bias=bias)  # fmt: skip
    weights, ids = grouped_topk(None, logits, **kwargs)
    scores = logits.sigmoid()
    want_ids = torch.topk(scores + bias, k=8, dim=-1, sorted=True)[1]
    want = scores.gather(1, want_ids)
    want = want / want.sum(-1, keepdim=True) * 2.5
    assert torch.equal(ids, want_ids.int())
    torch.testing.assert_close(weights, want)
    for row in range(7):
        alone_w, alone_ids = grouped_topk(None, logits[row : row + 1], **kwargs)
        assert torch.equal(alone_w[0], weights[row]) and torch.equal(alone_ids[0], ids[row])


def test_deterministic_topk_breaks_ties_by_column_and_orders_descending():
    import torch

    from arctic_platform.inference.vllm.glm5_batch_invariance import deterministic_topk

    logits = torch.tensor(
        [
            [0.5, 2.0, 2.0, -1.0, 2.0, 0.0],  # ties at columns 1, 2, 4; window [0, 6)
            [9.0, -3.0, -0.0, 0.0, -2.0, 7.0],  # window [1, 5): columns 1..4
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # window [4, 6): fewer valid columns than k
        ]
    )
    starts, ends = torch.tensor([0, 1, 4]), torch.tensor([6, 5, 6])
    out = torch.empty(3, 3, dtype=torch.int32)
    deterministic_topk(logits, starts, ends, 3, out)
    assert out.tolist() == [[4, 2, 1], [3, 2, 1], [1, 0, -1]]
    torch.manual_seed(0)
    big = torch.randn(5, 300).round(decimals=1)  # many ties
    zeros, ends = torch.zeros(5, dtype=torch.long), torch.full((5,), 300)
    a = deterministic_topk(big, zeros, ends, 64, torch.empty(5, 64, dtype=torch.int32))
    b = deterministic_topk(big[2:3], zeros[:1], ends[:1], 64, torch.empty(1, 64, dtype=torch.int32))
    assert torch.equal(a[2:3], b)
    assert torch.equal(a.sort(dim=-1, descending=True).values, a)
