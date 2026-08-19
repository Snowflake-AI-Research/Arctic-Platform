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
"""``finalize_fwd_bwd_metrics`` must make ``avg_loss`` match the global token-mean."""

from __future__ import annotations

from arctic_platform.common.utils.batch import finalize_fwd_bwd_metrics
from arctic_platform.testing_utils import TestCasePlus


class TestFinalizeFwdBwdMetrics(TestCasePlus):
    def test_avg_loss_matches_token_mean_across_unequal_shards(self):
        # Rank0: sum=10 over 2 tokens (mean 5); Rank1: sum=30 over 6 tokens (mean 5).
        # Global token-mean = 40/8 = 5. Summing per-rank avg_loss would wrongly give 10.
        ranks = [
            {"avg_loss": 5.0, "metrics": {"loss.sum": 10.0, "loss.tokens": 2.0}},
            {"avg_loss": 5.0, "metrics": {"loss.sum": 30.0, "loss.tokens": 6.0}},
        ]
        metrics, avg_loss = finalize_fwd_bwd_metrics(ranks)
        self.assertAlmostEqual(metrics["loss"], 5.0)
        self.assertAlmostEqual(avg_loss, 5.0)

    def test_avg_loss_matches_when_rank_means_differ(self):
        # Unequal per-rank means: Σsum/Σtokens, not mean(avg_loss) or sum(avg_loss).
        ranks = [
            {"avg_loss": 2.0, "metrics": {"loss.sum": 4.0, "loss.tokens": 2.0}},
            {"avg_loss": 8.0, "metrics": {"loss.sum": 8.0, "loss.tokens": 1.0}},
        ]
        metrics, avg_loss = finalize_fwd_bwd_metrics(ranks)
        self.assertAlmostEqual(metrics["loss"], 12.0 / 3.0)
        self.assertAlmostEqual(avg_loss, metrics["loss"])

    def test_legacy_fallback_sums_avg_loss_without_paired_metrics(self):
        ranks = [{"avg_loss": 1.5, "metrics": {}}, {"avg_loss": 2.5, "metrics": {}}]
        metrics, avg_loss = finalize_fwd_bwd_metrics(ranks)
        self.assertNotIn("loss", metrics)
        self.assertAlmostEqual(avg_loss, 4.0)
