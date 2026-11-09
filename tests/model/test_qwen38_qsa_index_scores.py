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
"""The fused QSA index scorer against the PyTorch formula, and the selections it leads to."""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused QSA index scorer runs on CUDA")


def test_fused_index_scores_match_the_fp32_formula():
    from arctic_platform.model.implementations.qwen38.qsa_index_scores import qsa_index_scores

    torch.manual_seed(0)
    queries = torch.randn(300, 4, 128, device="cuda").bfloat16()
    keys = torch.randn(1000, 128, device="cuda").bfloat16()
    visible = torch.randint(0, 1001, (300,), device="cuda")
    scores = qsa_index_scores(queries, keys, visible)

    expected = torch.relu(torch.einsum("qhd,pd->qhp", queries.float(), keys.float())).sum(1) / math.sqrt(128)
    expected.masked_fill_(torch.arange(1000, device="cuda")[None, :] >= visible[:, None], -torch.inf)
    finite = torch.isfinite(expected)
    assert torch.equal(torch.isfinite(scores), finite)
    assert torch.allclose(scores[finite], expected[finite], rtol=1e-5, atol=1e-5)


def test_selections_from_the_fused_scorer_match_the_pytorch_path(monkeypatch):
    from arctic_platform.model.implementations.qwen38 import qsa_index_scores as fused_module
    from arctic_platform.model.implementations.qwen38.qsa_flex import select_qsa_token_ids

    torch.manual_seed(1)
    length, rows, offset = 16384, 4096, 16384 - 4096
    queries = torch.randn(1, rows, 4, 128, device="cuda").bfloat16()
    keys = torch.randn(1, length // 4, 1, 128, device="cuda").bfloat16()
    kwargs = dict(token_budget=512, compress_ratio=4, query_offset=offset)
    fused = select_qsa_token_ids(queries, keys, torch.tensor([length], device="cuda"), **kwargs)
    monkeypatch.setattr(fused_module, "fused_index_scores_supported", lambda _queries: False)
    reference = select_qsa_token_ids(queries, keys, torch.tensor([length], device="cuda"), **kwargs)

    assert fused.shape == reference.shape
    same = (fused.sort(-1).values == reference.sort(-1).values).all(-1)
    # FP32 accumulation order can flip an exact near-tie at the k-th block; nothing else may differ.
    assert same.float().mean() > 0.999, same.float().mean()
