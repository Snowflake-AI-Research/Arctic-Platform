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


def test_decode_indexer_logits_are_scored_per_request(monkeypatch):
    import sys
    import types

    import torch

    from arctic_platform.inference.vllm import glm5_batch_invariance

    def logits(q, kv_cache, weights, context_lens, block_tables, schedule_metadata, max_model_len, clean_logits,
               indices=None):  # fmt: skip
        values, _ = q
        # Batch-dependent on purpose: every row is shifted by the batch's total context.
        rows = values.float().sum((1, 2, 3)) + weights.sum(-1) + context_lens.sum()
        return rows[:, None] + schedule_metadata[0, 0] + torch.arange(max_model_len)

    def metadata(context_lens, block_size, num_sms, indices=None):
        return torch.full((num_sms + 1, 2), float(context_lens.sum()))

    deep_gemm = types.SimpleNamespace(fp8_fp4_paged_mqa_logits=logits, get_paged_mqa_logits_metadata=metadata)
    utils = types.ModuleType("vllm.utils")
    utils.deep_gemm = deep_gemm
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.utils", utils)
    monkeypatch.setitem(sys.modules, "vllm.utils.deep_gemm", deep_gemm)
    glm5_batch_invariance._per_request_decode_indexer_logits()

    values = torch.randn(3, 1, 2, 4)
    weights = torch.randn(3, 2)
    lens = torch.tensor([[5], [9], [7]])
    kv_cache = torch.zeros(4, 64, 1, 8)
    tables = torch.arange(6).view(3, 2)
    batched = deep_gemm.fp8_fp4_paged_mqa_logits((values, None), kv_cache, weights, lens, tables,
                                                 metadata(lens, 64, 4), 16, False)  # fmt: skip
    for b in range(3):
        alone = deep_gemm.fp8_fp4_paged_mqa_logits((values[b : b + 1], None), kv_cache, weights[b : b + 1],
                                                   lens[b : b + 1], tables[b : b + 1], metadata(lens[b : b + 1], 64, 4),
                                                   16, False)  # fmt: skip
        assert torch.equal(batched[b : b + 1], alone)
    rows_idx = torch.arange(3)
    varlen = deep_gemm.fp8_fp4_paged_mqa_logits((values, None), kv_cache, weights, lens, tables,
                                                metadata(lens, 64, 4), 16, False, indices=rows_idx)  # fmt: skip
    assert torch.equal(varlen, batched)


def test_indexer_head_weights_use_the_batch_invariant_matmul(monkeypatch):
    import sys
    import types

    import torch

    from arctic_platform.inference.vllm import batch_invariant_fp32_logits
    from arctic_platform.inference.vllm import glm5_batch_invariance

    calls = []

    def matmul_fp32_out(a, b):
        calls.append((a.dtype, b.dtype))
        return a.float() @ b.float()

    attention = types.ModuleType("vllm.models.glm5next.common.attention")
    attention.torch = torch
    common = types.ModuleType("vllm.models.glm5next.common")
    common.attention = attention
    for name, module in (("vllm", types.ModuleType("vllm")), ("vllm.models", types.ModuleType("vllm.models")),
                         ("vllm.models.glm5next", types.ModuleType("vllm.models.glm5next")),
                         ("vllm.models.glm5next.common", common),
                         ("vllm.models.glm5next.common.attention", attention)):  # fmt: skip
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(batch_invariant_fp32_logits, "matmul_fp32_out", matmul_fp32_out)
    glm5_batch_invariance._batch_invariant_indexer_head_weights()

    hidden = torch.randn(5, 8, dtype=torch.bfloat16)
    weight = torch.randn(8, 3, dtype=torch.bfloat16).float()
    out = attention.torch.mm(hidden.float(), weight)
    assert calls == [(torch.bfloat16, torch.bfloat16)]
    assert torch.equal(out, hidden.float() @ weight)
    assert attention.torch.mm(hidden, hidden.t()).dtype == torch.bfloat16 and len(calls) == 1
    assert attention.torch.float32 is torch.float32 and attention.torch.Tensor is torch.Tensor
