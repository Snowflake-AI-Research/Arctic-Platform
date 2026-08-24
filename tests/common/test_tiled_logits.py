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
"""Tests for the shared memory-efficient logits/CE primitives.

* ``TestLogitsChunkRows`` — pure sizing math; runs anywhere (CPU client ok).
* ``TestLmHeadTemperatureLayout`` — ``temperature != 1`` keeps the incoming logits rank.
* ``TestPerTokenLogprobParityGPU`` — real GPU kernel path (``none`` / ``compute`` /
  ``memory``). Skips without CUDA via ``@require_torch_gpu``; run via autorun on the
  GPU box. Does **not** force the torch fallback — whatever CE kernel is installed
  on the server is what production uses.
"""

from __future__ import annotations

from functools import partial

import torch

from arctic_platform.common.utils.tiled_logits import TiledLogProbEntropy
from arctic_platform.common.utils.tiled_logits import chunked_logprobs_entropy_from_hidden
from arctic_platform.common.utils.tiled_logits import lm_head_logits
from arctic_platform.common.utils.tiled_logits import logits_chunk_rows
from arctic_platform.common.utils.tiled_logits import tiled_logprobs_entropy_from_hidden
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import make_tied_lm_head_model
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close


class TestLogitsChunkRows(TestCasePlus):
    def test_basic_sizing(self):
        # 1 GiB budget, fp32 (4 bytes), vocab 1000 -> floor(2**30 / (1000*4)).
        self.assertEqual(logits_chunk_rows(1000, 1), (2**30) // (1000 * 4))

    def test_at_least_one_row(self):
        self.assertEqual(logits_chunk_rows(10**9, 0.0), 1)

    def test_scales_with_budget(self):
        self.assertEqual(logits_chunk_rows(1000, 2), 2 * logits_chunk_rows(1000, 1))


class TestLmHeadTemperatureLayout(TestCasePlus):
    """``temperature != 1`` must not change logits rank."""

    def _model(self, hidden_size, vocab_size, device="cpu"):
        return make_tied_lm_head_model(hidden_size, vocab_size, device=device)

    def test_temperature_keeps_incoming_shape(self):
        set_seed(0)
        b, s, hidden_size, vocab_size = 2, 4, 8, 16
        model = self._model(hidden_size, vocab_size)
        hidden = torch.randn(b, s, hidden_size)
        logits = lm_head_logits(model, hidden, temperature=0.7)
        self.assertEqual(tuple(logits.shape), (b, s, vocab_size))

    def test_n1_row_temperature_keeps_shape(self):
        set_seed(0)
        hidden_size, vocab_size = 8, 16
        model = self._model(hidden_size, vocab_size)
        hidden = torch.randn(1, hidden_size)
        labels = torch.randint(0, vocab_size, (1,))
        logprobs, _ = tiled_logprobs_entropy_from_hidden(
            model, hidden, labels, temperature=0.7, calculate_entropy=False
        )
        self.assertEqual(tuple(logprobs.shape), tuple(labels.shape))

    def test_b1_keeps_batch_axis(self):
        set_seed(0)
        s, hidden_size, vocab_size = 4, 8, 16
        model = self._model(hidden_size, vocab_size)
        hidden1 = torch.randn(1, s, hidden_size)
        labels1 = torch.randint(0, vocab_size, (1, s))
        hidden2 = torch.randn(2, s, hidden_size)
        labels2 = torch.randint(0, vocab_size, (2, s))
        lp1, _ = tiled_logprobs_entropy_from_hidden(model, hidden1, labels1, temperature=1.5, calculate_entropy=False)
        lp2, _ = tiled_logprobs_entropy_from_hidden(model, hidden2, labels2, temperature=1.5, calculate_entropy=False)
        self.assertEqual(tuple(lp1.shape), tuple(labels1.shape))
        self.assertEqual(tuple(lp2.shape), tuple(labels2.shape))

    def test_one_row_memory_shard_temperature(self):
        set_seed(0)
        n, hidden_size, vocab_size = 3, 8, 16
        model = self._model(hidden_size, vocab_size)
        hidden = torch.randn(n, hidden_size)
        labels = torch.randint(0, vocab_size, (n,))
        logprobs, _ = TiledLogProbEntropy.apply(
            tiled_logprobs_entropy_from_hidden,
            model,
            hidden,
            labels,
            0.7,
            False,
            n,
            [model.lm_head.weight],
        )
        self.assertEqual(tuple(logprobs.shape), tuple(labels.shape))


@require_torch_gpu
class TestPerTokenLogprobParityGPU(TestCasePlus):
    """``none`` / ``compute`` / ``memory`` agree on per-token logprobs on CUDA."""

    def test_modes_agree(self):
        set_seed(0)
        N, H, V = 40, 8, 32
        model = make_tied_lm_head_model(H, V)
        hidden = torch.randn(N, H, device="cuda")
        labels = torch.randint(0, V, (N,), device="cuda")

        lp_none, _ = tiled_logprobs_entropy_from_hidden(
            model, hidden, labels, calculate_entropy=False, logits_compute_in_fp32=True
        )
        lp_compute, _ = chunked_logprobs_entropy_from_hidden(
            model,
            hidden,
            labels,
            calculate_entropy=False,
            peak_mem_gib=1e-6,  # force multiple chunks
            logits_compute_in_fp32=True,
        )
        chunk_rows = max(1, logits_chunk_rows(V, 1e-6))
        num_shards = max(1, -(N // -chunk_rows))
        lp_mem, _ = TiledLogProbEntropy.apply(
            partial(tiled_logprobs_entropy_from_hidden, logits_compute_in_fp32=True),
            model,
            hidden,
            labels,
            1.0,
            False,
            num_shards,
            [model.lm_head.weight],
        )

        # Same fp32 CE math across modes → float noise only (rtol=0). Widening
        # atol here would hide a broken chunk / tile boundary.
        torch_assert_close(lp_none, lp_compute, rtol=0, atol=1e-5)
        torch_assert_close(lp_none, lp_mem, rtol=0, atol=1e-5)
