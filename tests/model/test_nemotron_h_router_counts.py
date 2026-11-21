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
"""Nemotron-H's router counts the tokens per expert with an integer kernel, with histc's counts and dtype."""

import torch

from arctic_platform.model.implementations.moe.layers.moe import NemotronHRouter


def test_nemotron_h_router_counts_tokens_per_expert_with_integers(monkeypatch):
    histc = torch.histc

    def deterministic_histc(input, *args, **kwargs):
        # What CUDA does under torch.use_deterministic_algorithms(True).
        if input.is_floating_point():
            raise RuntimeError("_histc_cuda with floating point input does not have a deterministic implementation")
        return histc(input, *args, **kwargs)

    torch.manual_seed(0)
    router = NemotronHRouter(dim=16, num_experts=8, top_k=2, n_group=2, topk_group=1, norm_topk_prob=True)
    router.init_weights(0.02)
    x = torch.randn(64, 16)
    monkeypatch.setattr(torch, "histc", deterministic_histc)

    top_scores, selected, counts = router(x)

    expected = histc(selected.reshape(-1).float(), bins=8, min=0, max=8)
    assert counts.dtype == expected.dtype == torch.float32
    assert torch.equal(counts, expected) and int(counts.sum()) == 64 * 2
    # The last expert's id equals histc's upper edge, which histc counts in the last bin.
    assert torch.equal(counts[-1], (selected == 7).sum().float())
