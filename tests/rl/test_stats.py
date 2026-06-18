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

"""CPU unit tests for the metrics tracker (``processors/stats_tracker.py``) and the DP shard-stats helpers
(``utils/batch.py``).

These are pure reductions/diagnostics that the GPU path only touches incidentally, so they are exercised directly on
CPU with tiny tensors (no GPU, no dist: ``export`` is called with ``reduce_group=None`` so no collective runs)::

    pytest tests/rl/test_stats.py
"""

from __future__ import annotations

import os

import torch

from arctic_platform.rl.processors import stats_tracker
from arctic_platform.rl.processors.stats_tracker import DistributedStatsTracker
from arctic_platform.rl.utils.batch import log_dp_shard_tokens
from arctic_platform.rl.utils.batch import shard_token_stats
from arctic_platform.testing_utils import TestCasePlus


class TestDistributedStatsTracker(TestCasePlus):
    def test_scalar_keeps_last_value(self):
        tracker = DistributedStatsTracker()
        tracker.scalar(lr=0.1)
        tracker.scalar(lr=0.2)
        self.assertAlmostEqual(tracker.export()["lr"], 0.2, places=6)

    def test_scalar_unwraps_tensor(self):
        tracker = DistributedStatsTracker()
        tracker.scalar(grad_norm=torch.tensor(3.5))
        self.assertAlmostEqual(tracker.export()["grad_norm"], 3.5, places=6)

    def test_stat_averages_over_denominator_mask(self):
        tracker = DistributedStatsTracker()
        # The denominator mask marks valid tokens; stat() averages only over the entries where the mask is truthy.
        tracker.denominator(n_valid_tokens=torch.tensor([1, 1, 0, 1]))
        tracker.stat(loss=torch.tensor([1.0, 2.0, 99.0, 3.0]))
        out = tracker.export()
        self.assertAlmostEqual(out["loss"], 2.0, places=6)  # mean(1, 2, 3); the masked-out 99 is excluded
        self.assertAlmostEqual(out["n_valid_tokens"], 1.0, places=6)  # self-referential mask -> mean of the 1s

    def test_stat_without_recorded_denominator_averages_all(self):
        tracker = DistributedStatsTracker()
        tracker.stat(denominator="missing", x=torch.tensor([2.0, 4.0]))
        self.assertAlmostEqual(tracker.export()["x"], 3.0, places=6)

    def test_scope_prefixes_keys(self):
        tracker = DistributedStatsTracker()
        with tracker.scope("actor"):
            tracker.scalar(lr=0.1)
            with tracker.scope("clip"):
                tracker.scalar(ratio=0.9)
        out = tracker.export()
        self.assertIn("actor/lr", out)
        self.assertIn("actor/clip/ratio", out)

    def test_named_tracker_uses_name_as_root_scope(self):
        tracker = DistributedStatsTracker("train")
        tracker.scalar(step=5)
        self.assertIn("train/step", tracker.export())

    def test_disable_scope_temporarily_clears_prefix(self):
        tracker = DistributedStatsTracker("train")
        with tracker.scope("actor"):
            with tracker.disable_scope():
                tracker.scalar(global_step=7)
            tracker.scalar(lr=0.1)
        out = tracker.export()
        self.assertIn("global_step", out)  # recorded with the scope stack cleared
        self.assertIn("train/actor/lr", out)

    def test_record_timing_emits_nonnegative_scalar(self):
        tracker = DistributedStatsTracker()
        with tracker.record_timing("gen_time"):
            pass
        out = tracker.export()
        self.assertIn("gen_time", out)
        self.assertGreaterEqual(out["gen_time"], 0.0)

    def test_export_reset_semantics(self):
        tracker = DistributedStatsTracker()
        tracker.scalar(a=1.0)
        self.assertIn("a", tracker.export(reset=False))
        self.assertIn("a", tracker.export(reset=True))  # still present this call
        self.assertEqual(tracker.export(), {})  # cleared after the reset

    def test_empty_export_is_empty(self):
        self.assertEqual(DistributedStatsTracker().export(), {})

    def test_export_all_method_mirrors_export(self):
        tracker = DistributedStatsTracker()
        tracker.scalar(a=1.0)
        self.assertEqual(tracker.export_all(), {"a": 1.0})


class TestModuleLevelTrackers(TestCasePlus):
    def setUp(self):
        super().setUp()
        self._drain()

    def tearDown(self):
        self._drain()
        super().tearDown()

    @staticmethod
    def _drain():
        stats_tracker.export_all(reset=True)
        stats_tracker.TRACKERS.clear()

    def test_default_tracker_helpers(self):
        with stats_tracker.scope("actor"):
            stats_tracker.scalar(lr=0.3)
        self.assertAlmostEqual(stats_tracker.export()["actor/lr"], 0.3, places=6)

    def test_get_caches_named_trackers(self):
        first = stats_tracker.get("rollout")
        self.assertIs(first, stats_tracker.get("rollout"))

    def test_export_all_merges_default_and_named(self):
        stats_tracker.scalar(global_step=2)
        stats_tracker.get("rollout").scalar(reward=1.5)
        out = stats_tracker.export_all()
        self.assertIn("global_step", out)
        self.assertIn("rollout/reward", out)

    def test_module_record_timing(self):
        with stats_tracker.record_timing("phase_time"):
            pass
        self.assertGreaterEqual(stats_tracker.export()["phase_time"], 0.0)

    def test_module_denominator_and_stat(self):
        stats_tracker.denominator(n_valid_tokens=torch.tensor([1, 0, 1]))
        stats_tracker.stat(loss=torch.tensor([2.0, 99.0, 4.0]))
        self.assertAlmostEqual(stats_tracker.export()["loss"], 3.0, places=6)


class TestShardTokenStats(TestCasePlus):
    def test_padded_batch_token_stats(self):
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        stats = shard_token_stats({"attention_mask": attention_mask})
        self.assertEqual(stats["valid_tokens"], 5)
        self.assertEqual(stats["batch_size"], 2)
        self.assertEqual(stats["seq_len_min"], 2)
        self.assertEqual(stats["seq_len_max"], 3)

    def test_packed_and_cu_seqlens_and_loss_mask(self):
        stats = shard_token_stats(
            {
                "input_ids": torch.arange(7),  # 1D packed
                "cu_seqlens": torch.tensor([0, 3, 7], dtype=torch.int32),
                "loss_mask": torch.tensor([1, 0, 1, 1, 0, 1, 1]),
            }
        )
        self.assertEqual(stats["packed_tokens"], 7)
        self.assertEqual(stats["cu_seqlens_packed"], 7)
        self.assertEqual(stats["loss_mask_tokens"], 5)

    def test_packed_2d_single_row(self):
        stats = shard_token_stats({"input_ids": torch.arange(10).unsqueeze(0)})  # [1, T]
        self.assertEqual(stats["packed_tokens"], 10)

    def test_meta_batch_num_tokens(self):
        stats = shard_token_stats({"attention_mask": torch.ones(1, 4)}, {"batch_num_tokens": 128})
        self.assertEqual(stats["meta_batch_num_tokens"], 128)

    def test_empty_batch_returns_empty(self):
        self.assertEqual(shard_token_stats({}), {})


class TestLogDpShardTokens(TestCasePlus):
    def test_noop_without_env_flag(self):
        os.environ.pop("ARL_LOG_DP_SHARD_TOKENS", None)
        log_dp_shard_tokens(0, "tag", {"attention_mask": torch.ones(1, 4)})  # must not raise

    def test_logs_when_env_flag_set(self):
        os.environ["ARL_LOG_DP_SHARD_TOKENS"] = "1"
        try:
            log_dp_shard_tokens(0, "tag", {"attention_mask": torch.ones(2, 3)})  # prints; must not raise
        finally:
            os.environ.pop("ARL_LOG_DP_SHARD_TOKENS", None)
