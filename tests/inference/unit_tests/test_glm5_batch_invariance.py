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
