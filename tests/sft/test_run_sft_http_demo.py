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
"""CPU wire-format tests for the HTTP SFT demo (``run_sft_http_demo``).

These are the discriminator behind the GPU e2e ``TestSftCeHttpE2EModesGPU``:
that test only proves the three ``logits_optimization`` modes *agree*, so it
cannot tell "all modes correct" from "the mode flag was silently ignored".
The distinguishing logic is ``_build_batch`` mapping the CLI flags onto the
``processing`` envelope — pin that here on CPU (no tokenizer download, no server).
"""

from __future__ import annotations

import torch

from arctic_platform.sft.examples.run_sft_http_demo import _build_batch
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import torch_assert_equal


class _FakeTokenizer:
    """Return fixed ``input_ids`` / ``attention_mask`` without any download.

    Row 0 carries one right-pad (attention_mask 0) so the padding→-100 masking
    is exercised; row 1 is fully real.
    """

    def __call__(self, texts, **kwargs):
        input_ids = torch.arange(1, 11).view(2, 5)
        attention_mask = torch.ones(2, 5, dtype=torch.long)
        attention_mask[0, -1] = 0
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TestBuildBatchProcessingWire(TestCasePlus):
    """The CLI → ``processing`` mapping the GPU e2e depends on."""

    def _build(self, **kw):
        return _build_batch(_FakeTokenizer(), ["a", "b"], [2, 1], pad_token_id=0, **kw)

    def test_sft_ce_memory_emits_distinct_config(self):
        env = self._build(
            loss_fn="sft_ce",
            logits_optimization="memory",
            logits_optimization_peak_mem_size_in_gib=3,
        )
        # If this drifted to omitting config, the GPU e2e would silently run
        # `none` for every mode and still "pass".
        self.assertEqual(
            env["processing"],
            {
                "loss_fn": "sft_ce",
                "config": {
                    "logits_optimization": "memory",
                    "logits_optimization_peak_mem_size_in_gib": 3,
                },
            },
        )

    def test_sft_ce_compute_emits_distinct_config(self):
        env = self._build(
            loss_fn="sft_ce",
            logits_optimization="compute",
            logits_optimization_peak_mem_size_in_gib=7,
        )
        self.assertEqual(env["processing"]["config"]["logits_optimization"], "compute")
        self.assertEqual(
            env["processing"]["config"]["logits_optimization_peak_mem_size_in_gib"], 7
        )

    def test_sft_ce_none_omits_config(self):
        env = self._build(loss_fn="sft_ce", logits_optimization="none")
        # `none` is the classic full-logits path: no config block on the wire.
        self.assertEqual(env["processing"], {"loss_fn": "sft_ce"})

    def test_plain_sft_ignores_optimization_flag(self):
        # Documented contract: `logits_optimization` is a no-op for plain `sft`.
        env = self._build(loss_fn="sft", logits_optimization="memory")
        self.assertEqual(env["processing"], {"loss_fn": "sft"})

    def test_labels_mask_prompt_and_padding(self):
        env = self._build(loss_fn="sft")
        labels = env["batch"]["labels"]
        # row0: prompt_len=2 → first two masked; col4 is padding → masked.
        torch_assert_equal(labels[0], torch.tensor([-100, -100, 3, 4, -100]))
        # row1: prompt_len=1 → first masked; no padding.
        torch_assert_equal(labels[1], torch.tensor([-100, 7, 8, 9, 10]))
        self.assertEqual(env["meta"], {"pad_token_id": 0})
