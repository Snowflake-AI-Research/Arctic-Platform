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


def test_sparse_mla_prefill_always_takes_the_mqa_path(monkeypatch):
    import sys
    import types

    from arctic_platform.inference.vllm import glm5_batch_invariance

    class MLAAttention:
        def _use_sparse_mha(self, attn_metadata):
            return True

    mla_attention = types.ModuleType("vllm.model_executor.layers.attention.mla_attention")
    mla_attention.MLAAttention = MLAAttention
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.attention"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, mla_attention.__name__, mla_attention)
    glm5_batch_invariance._sparse_mla_always_mqa()

    assert MLAAttention()._use_sparse_mha(object()) is False


def test_tokens_that_fit_in_index_topk_get_causal_indices_beside_longer_sequences(monkeypatch):
    import sys
    import types

    import torch

    from arctic_platform.inference.vllm import glm5_batch_invariance

    def _fill_causal_indices(rows, positions):
        causal_range = torch.arange(rows.shape[1], dtype=torch.int32)
        rows[:] = causal_range[None, :]
        rows[causal_range[None, :] > positions[:, None].to(torch.int32)] = -1

    def sparse_attn_indexer_kpool(hidden_states, topk_tokens, topk_indices_buffer, positions=None):
        topk_indices_buffer[: hidden_states.shape[0]] = 7  # stands in for scored selections
        return topk_indices_buffer

    common = types.ModuleType("vllm.models.glm5next.common.sparse_indexer")
    common._fill_causal_indices = _fill_causal_indices
    nvidia = types.ModuleType("vllm.models.glm5next.nvidia.sparse_indexer")
    nvidia.sparse_attn_indexer_kpool = sparse_attn_indexer_kpool
    for name in ("vllm", "vllm.models", "vllm.models.glm5next", "vllm.models.glm5next.common",
                 "vllm.models.glm5next.nvidia"):  # fmt: skip
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, common.__name__, common)
    monkeypatch.setitem(sys.modules, nvidia.__name__, nvidia)
    sys.modules["vllm.models.glm5next.nvidia"].sparse_indexer = nvidia
    glm5_batch_invariance._causal_indices_per_token()

    buffer = torch.full((4, 4), -2, dtype=torch.int32)
    positions = torch.tensor([0, 2, 3, 9])
    out = nvidia.sparse_attn_indexer_kpool(torch.zeros(4, 1), 4, buffer, positions=positions)

    assert out is buffer
    assert buffer[:3].tolist() == [[0, -1, -1, -1], [0, 1, 2, -1], [0, 1, 2, 3]]
    assert buffer[3].tolist() == [7, 7, 7, 7]


def test_causal_rows_fill_without_a_host_sync_matches_the_row_selection():
    import torch

    from arctic_platform.inference.vllm.glm5_batch_invariance import fill_causal_rows

    def by_selection(buffer, positions, topk_tokens):
        positions = positions[: buffer.shape[0]]
        rows = torch.nonzero(positions < topk_tokens).flatten()
        col = torch.arange(buffer.shape[1], dtype=torch.int32)
        causal = col[None, :].repeat(rows.numel(), 1)
        causal[col[None, :] > positions[rows, None].to(torch.int32)] = -1
        buffer[rows] = causal

    torch.manual_seed(0)
    for rows, n_positions in ((6, 6), (6, 4), (4, 6)):
        buffer = torch.randint(-1, 50, (rows, 16), dtype=torch.int32)
        positions = torch.randint(0, 24, (n_positions,))
        expected = buffer.clone()
        by_selection(expected[:n_positions], positions, 12)
        fill_causal_rows(buffer, positions, 12)
        assert torch.equal(buffer, expected)


def test_indexer_key_norm_runs_eagerly_with_the_reference_math(monkeypatch):
    import sys
    import types

    import torch
    import torch.nn.functional as F

    from arctic_platform.inference.vllm import glm5_batch_invariance

    attention = types.ModuleType("vllm.models.glm5next.common.attention")
    attention._fused_indexer_k_norm = "compiled"
    common = types.ModuleType("vllm.models.glm5next.common")
    common.attention = attention
    for name, module in (("vllm", types.ModuleType("vllm")), ("vllm.models", types.ModuleType("vllm.models")),
                         ("vllm.models.glm5next", types.ModuleType("vllm.models.glm5next")),
                         ("vllm.models.glm5next.common", common),
                         ("vllm.models.glm5next.common.attention", attention)):  # fmt: skip
        monkeypatch.setitem(sys.modules, name, module)
    glm5_batch_invariance._eager_indexer_k_norm()

    x = torch.randn(16, 128, dtype=torch.bfloat16)
    weight, bias = torch.randn(128), torch.randn(128)
    out = attention._fused_indexer_k_norm(x, weight, bias, 128, 1e-6)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, F.layer_norm(x.float(), (128,), weight, bias, 1e-6).to(torch.bfloat16))
    assert torch.equal(attention._fused_indexer_k_norm(x[3:4], weight, bias, 128, 1e-6), out[3:4])
