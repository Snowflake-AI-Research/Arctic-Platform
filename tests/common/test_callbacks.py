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
"""``CallbackRunner``/``merge_metrics`` dispatch callbacks and reduce metrics correctly."""

from __future__ import annotations

import unittest
from typing import Any

from arctic_platform.common.callbacks import Callback
from arctic_platform.common.callbacks import CallbackRunner
from arctic_platform.common.callbacks import MetricsCallback
from arctic_platform.common.callbacks import Reduce
from arctic_platform.common.callbacks import merge_metrics


class _Trainer:
    def __init__(self, rank: int) -> None:
        self.rank = rank


class _OrderRecorder(Callback):
    calls: list = []

    @classmethod
    def enabled(cls, training_config):
        return True

    def post_init(self, trainer):
        self.calls.append((type(self).__name__, "post_init"))


class _First(_OrderRecorder):
    calls: list = []


class _Second(_OrderRecorder):
    calls: list = []


class _Disabled(Callback):
    @classmethod
    def enabled(cls, training_config):
        return False

    def __init__(self, training_config):
        raise AssertionError("disabled callback must not be constructed")


class _RankZeroOnly(Callback):
    rank_zero_only = True

    @classmethod
    def enabled(cls, training_config):
        return True


class _Metrics(MetricsCallback):
    @classmethod
    def enabled(cls, training_config):
        return True


class _OtherMetrics(_Metrics):
    pass


class _RankZeroMetrics(_Metrics):
    rank_zero_only = True


class _ErroringOnError(Callback):
    @classmethod
    def enabled(cls, training_config):
        return True

    def on_error(self, trainer, stage, exc):
        raise RuntimeError("boom")


class TestCallbackRunner(unittest.TestCase):
    def test_only_enabled_callbacks_are_instantiated(self):
        runner = CallbackRunner([_Disabled, _Metrics], {}, rank=0)
        self.assertEqual([type(cb) for cb in runner.callbacks], [_Metrics])

    def test_rank_zero_only_callback_skipped_on_other_ranks(self):
        self.assertEqual(len(CallbackRunner([_RankZeroOnly], {}, rank=0).callbacks), 1)
        self.assertEqual(len(CallbackRunner([_RankZeroOnly], {}, rank=1).callbacks), 0)

    def test_run_dispatches_in_registry_order(self):
        _First.calls = []
        _Second.calls = []
        trainer = _Trainer(rank=0)
        runner = CallbackRunner([_First, _Second], {}, rank=0)
        runner.run("post_init", trainer)
        self.assertEqual(
            [call for cb in runner.callbacks for call in cb.calls],
            [("_First", "post_init"), ("_Second", "post_init")],
        )

    def test_log_twice_in_one_step_raises(self):
        cb = _Metrics({})
        cb.log("x", 1)
        with self.assertRaises(ValueError):
            cb.log("x", 2)

    def test_rank_zero_only_rejects_non_rank0_reduce(self):
        cb = _RankZeroMetrics({})
        with self.assertRaises(ValueError):
            cb.log("x", 1, reduce=Reduce.SUM)
        self.assertEqual(cb.pending, {})

    def test_flush_name_clash_clears_pending(self):
        runner = CallbackRunner([_Metrics, _OtherMetrics], {}, rank=0)
        runner.callbacks[0].log("x", 1)
        runner.callbacks[1].log("x", 2)
        with self.assertRaises(ValueError):
            runner.flush()
        self.assertEqual(runner.callbacks[0].pending, {})
        self.assertEqual(runner.callbacks[1].pending, {})

    def test_flush_clears_pending_for_next_step(self):
        runner = CallbackRunner([_Metrics], {}, rank=0)
        metrics_cb = runner.callbacks[0]
        metrics_cb.log("x", 1)
        flushed = runner.flush()
        self.assertEqual(flushed, {"x": (Reduce.RANK0, 1)})
        self.assertEqual(metrics_cb.pending, {})

    def test_on_error_clears_pending_and_calls_every_callback(self):
        runner = CallbackRunner([_Metrics, _ErroringOnError], {}, rank=0)
        runner.callbacks[0].log("x", 1)
        exc = RuntimeError("original")
        # Must not raise even though _ErroringOnError.on_error itself raises.
        runner.on_error(_Trainer(rank=0), "step", exc)
        self.assertEqual(runner.callbacks[0].pending, {})


class TestMergeMetrics(unittest.TestCase):
    def test_rank0_keeps_rank_zeros_value(self):
        per_rank: list[dict[str, Any]] = [
            {"x": (Reduce.RANK0, "r0-value")},
            {"x": (Reduce.RANK0, "r1-value")},
        ]
        self.assertEqual(merge_metrics(per_rank), {"x": "r0-value"})

    def test_sum_max_min_combine_scalars(self):
        per_rank = [
            {"s": (Reduce.SUM, 1), "mx": (Reduce.MAX, 1), "mn": (Reduce.MIN, 5)},
            {"s": (Reduce.SUM, 2), "mx": (Reduce.MAX, 7), "mn": (Reduce.MIN, 2)},
        ]
        self.assertEqual(merge_metrics(per_rank), {"s": 3, "mx": 7, "mn": 2})

    def test_mean_combines_scalars(self):
        per_rank = [{"m": (Reduce.MEAN, 2)}, {"m": (Reduce.MEAN, 4)}]
        self.assertEqual(merge_metrics(per_rank), {"m": 3.0})

    def test_per_rank_returns_rank_ordered_list(self):
        per_rank = [{"p": (Reduce.PER_RANK, "a")}, {"p": (Reduce.PER_RANK, "b")}]
        self.assertEqual(merge_metrics(per_rank), {"p": ["a", "b"]})

    def test_sum_combines_nested_dicts_and_lists_elementwise(self):
        per_rank = [
            {"d": (Reduce.SUM, {"a": [1, 2], "b": 10})},
            {"d": (Reduce.SUM, {"a": [3, 4], "b": 20})},
        ]
        self.assertEqual(merge_metrics(per_rank), {"d": {"a": [4, 6], "b": 30}})

    def test_mismatched_reductions_across_ranks_raises(self):
        per_rank = [{"x": (Reduce.SUM, 1)}, {"x": (Reduce.MAX, 1)}]
        with self.assertRaises(ValueError):
            merge_metrics(per_rank)

    def test_mismatched_metric_names_across_ranks_raises(self):
        per_rank = [{"x": (Reduce.SUM, 1)}, {"y": (Reduce.SUM, 1)}]
        with self.assertRaises(ValueError):
            merge_metrics(per_rank)

    def test_mean_combines_nested_dicts_and_lists(self):
        per_rank = [
            {"d": (Reduce.MEAN, {"a": [2, 4]})},
            {"d": (Reduce.MEAN, {"a": [4, 8]})},
        ]
        self.assertEqual(merge_metrics(per_rank), {"d": {"a": [3.0, 6.0]}})

    def test_empty_per_rank_raises(self):
        with self.assertRaises(ValueError):
            merge_metrics([])

    def test_list_length_mismatch_raises(self):
        per_rank = [{"a": (Reduce.SUM, [1, 2])}, {"a": (Reduce.SUM, [1])}]
        with self.assertRaises(ValueError):
            merge_metrics(per_rank)

    def test_mixed_dict_and_scalar_raises(self):
        per_rank = [{"a": (Reduce.SUM, {"x": 1})}, {"a": (Reduce.SUM, 1)}]
        with self.assertRaises(ValueError):
            merge_metrics(per_rank)

    def test_mismatched_dict_keys_across_ranks_raises(self):
        per_rank = [
            {"d": (Reduce.SUM, {"a": 1})},
            {"d": (Reduce.SUM, {"b": 1})},
        ]
        with self.assertRaises(ValueError):
            merge_metrics(per_rank)


if __name__ == "__main__":
    unittest.main()
