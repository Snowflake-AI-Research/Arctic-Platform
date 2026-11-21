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

import json
import pickle
import unittest
from typing import Any

from arctic_platform.common.callbacks import Callback
from arctic_platform.common.callbacks import CallbackRunner
from arctic_platform.common.callbacks import InvalidMetric
from arctic_platform.common.callbacks import MergeResult
from arctic_platform.common.callbacks import MetricsCallback
from arctic_platform.common.callbacks import Reduce
from arctic_platform.common.callbacks import merge_metrics

_QUARANTINED = InvalidMetric.QUARANTINED


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
    metric_specs = {"x": Reduce.RANK0}

    @classmethod
    def enabled(cls, training_config):
        return True


class _OtherMetrics(_Metrics):
    metric_specs = {"x": Reduce.SUM}


class _RankZeroMetrics(_Metrics):
    rank_zero_only = True


class _RankZeroSumMetrics(_Metrics):
    rank_zero_only = True
    metric_specs = {"x": Reduce.SUM}


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

    def test_run_dispatches_post_fwd_only_microbatch(self):
        class _Recorder(Callback):
            @classmethod
            def enabled(cls, training_config):
                return True

            def __init__(self, training_config):
                self.calls = []

            def post_fwd_only_microbatch(self, trainer, is_padding):
                self.calls.append(is_padding)

        trainer = _Trainer(rank=0)
        runner = CallbackRunner([_Recorder], {}, rank=0)
        runner.run("post_fwd_only_microbatch", trainer, True)
        runner.run("post_fwd_only_microbatch", trainer, False)
        self.assertEqual(runner.callbacks[0].calls, [True, False])

    def test_log_twice_in_one_step_raises(self):
        cb = _Metrics({})
        cb.log("x", 1)
        with self.assertRaises(ValueError):
            cb.log("x", 2)

    def test_log_undeclared_name_raises(self):
        cb = _Metrics({})
        with self.assertRaisesRegex(ValueError, "metric_specs"):
            cb.log("y", 1)
        self.assertEqual(cb.pending, {})

    def test_log_uses_declared_reduce(self):
        cb = _OtherMetrics({})
        cb.log("x", 1)
        self.assertEqual(cb.pending, {"x": (Reduce.SUM, 1)})

    def test_rank_zero_only_non_rank0_spec_rejected_at_init_on_every_rank(self):
        for rank in (0, 1):
            with self.assertRaisesRegex(ValueError, "rank_zero_only"):
                CallbackRunner([_RankZeroSumMetrics], {}, rank=rank)

    def test_overlapping_specs_rejected_at_init(self):
        with self.assertRaisesRegex(ValueError, "declared by both"):
            CallbackRunner([_Metrics, _OtherMetrics], {}, rank=0)

    def test_overlap_with_rank_zero_only_rejected_on_other_ranks_too(self):
        with self.assertRaisesRegex(ValueError, "declared by both"):
            CallbackRunner([_Metrics, _RankZeroMetrics], {}, rank=1)

    def test_non_reduce_spec_rejected_at_init(self):
        class _Typo(_Metrics):
            metric_specs = {"x": "sum"}

        with self.assertRaises(TypeError):
            CallbackRunner([_Typo], {}, rank=0)

    def test_flush_clears_pending_for_next_step(self):
        runner = CallbackRunner([_Metrics], {}, rank=0)
        metrics_cb = runner.callbacks[0]
        metrics_cb.log("x", 1)
        flushed = runner.flush()
        self.assertEqual(flushed, {"x": (Reduce.RANK0, 1)})
        self.assertEqual(metrics_cb.pending, {})

    def test_flush_marks_declared_but_unlogged_metric_invalid(self):
        runner = CallbackRunner([_OtherMetrics], {}, rank=0)
        with self.assertLogs("arctic_platform.common.callbacks", level="ERROR"):
            flushed = runner.flush()
        self.assertEqual(flushed, {"x": (Reduce.SUM, _QUARANTINED)})

    def test_on_error_clears_pending_and_calls_every_callback(self):
        runner = CallbackRunner([_Metrics, _ErroringOnError], {}, rank=0)
        runner.callbacks[0].log("x", 1)
        exc = RuntimeError("original")
        # Must not raise even though _ErroringOnError.on_error itself raises.
        runner.on_error(_Trainer(rank=0), "step", exc)
        self.assertEqual(runner.callbacks[0].pending, {})
        # on_error failure quarantines that callback; flush still clears quarantine.
        self.assertIn(runner.callbacks[1], runner._failed)
        runner.flush()
        self.assertEqual(runner._failed, set())

    def test_run_quarantines_a_failing_callback_and_keeps_siblings(self):
        class _Boom(MetricsCallback):
            metric_specs = {"boom": Reduce.MAX}

            @classmethod
            def enabled(cls, training_config):
                return True

            def post_step(self, trainer):
                self.log("boom", 1)
                raise RuntimeError("post-step boom")

        class _Ok(MetricsCallback):
            metric_specs = {"ok": Reduce.SUM}

            @classmethod
            def enabled(cls, training_config):
                return True

            def post_step(self, trainer):
                self.log("ok", 1)

        runner = CallbackRunner([_Boom, _Ok], {}, rank=0)
        runner.run("post_step", _Trainer(rank=0))
        # The quarantined callback keeps its declared reduce but not its partial value.
        self.assertEqual(runner.flush(), {"boom": (Reduce.MAX, _QUARANTINED), "ok": (Reduce.SUM, 1)})
        self.assertEqual(runner.callbacks[0].pending, {})
        # Quarantine clears at flush so the next step can retry the callback.
        self.assertEqual(runner._failed, set())

    def test_quarantined_callback_is_skipped_on_later_hooks_until_flush(self):
        class _Boom(Callback):
            def __init__(self, training_config):
                self.calls = []

            @classmethod
            def enabled(cls, training_config):
                return True

            def pre_step(self, trainer):
                self.calls.append("pre_step")
                raise RuntimeError("pre-step boom")

            def post_step(self, trainer):
                self.calls.append("post_step")

        runner = CallbackRunner([_Boom], {}, rank=0)
        runner.run("pre_step", _Trainer(rank=0))
        runner.run("post_step", _Trainer(rank=0))
        self.assertEqual(runner.callbacks[0].calls, ["pre_step"])
        runner.flush()
        runner.run("post_step", _Trainer(rank=0))
        self.assertEqual(runner.callbacks[0].calls, ["pre_step", "post_step"])

    def test_post_init_raise_propagates_without_quarantine(self):
        class _Boom(Callback):
            @classmethod
            def enabled(cls, training_config):
                return True

            def post_init(self, trainer):
                raise ValueError("bad config")

        runner = CallbackRunner([_Boom], {}, rank=0)
        with self.assertRaisesRegex(ValueError, "bad config"):
            runner.run("post_init", _Trainer(rank=0))
        self.assertEqual(runner._failed, set())

    def test_on_quarantine_is_invoked_once_and_on_error_keeps_quarantine(self):
        class _Boom(Callback):
            def __init__(self, training_config):
                self.calls = []
                self.quarantines = 0

            @classmethod
            def enabled(cls, training_config):
                return True

            def pre_step(self, trainer):
                self.calls.append("pre_step")
                raise RuntimeError("pre-step boom")

            def post_step(self, trainer):
                self.calls.append("post_step")

            def on_quarantine(self, trainer):
                self.quarantines += 1

            def on_error(self, trainer, stage, exc):
                self.calls.append("on_error")

        runner = CallbackRunner([_Boom], {}, rank=0)
        trainer = _Trainer(rank=0)
        runner.run("pre_step", trainer)
        self.assertEqual(runner.callbacks[0].quarantines, 1)
        runner.on_error(trainer, "step", RuntimeError("peer"))
        # Quarantined callbacks are skipped on on_error; quarantine survives.
        self.assertEqual(runner.callbacks[0].calls, ["pre_step"])
        self.assertIn(runner.callbacks[0], runner._failed)
        runner.flush()
        self.assertEqual(runner._failed, set())

    def test_asymmetric_quarantine_drops_only_that_metric_on_merge(self):
        class _Stats(MetricsCallback):
            metric_specs = {"router_stats": Reduce.SUM}

            @classmethod
            def enabled(cls, training_config):
                return True

            def post_step(self, trainer):
                self.log("router_stats", {"router": {"sampled_tokens": 5.0}})

        class _BoomStats(_Stats):
            def pre_fwd_bwd(self, trainer):
                raise RuntimeError("snapshot boom")

        class _Count(MetricsCallback):
            metric_specs = {"count": Reduce.SUM}

            @classmethod
            def enabled(cls, training_config):
                return True

            def post_step(self, trainer):
                self.log("count", 1)

        healthy = CallbackRunner([_Stats, _Count], {}, rank=0)
        failed = CallbackRunner([_BoomStats, _Count], {}, rank=1)
        trainer = _Trainer(rank=0)
        healthy.run("post_step", trainer)
        failed.run("pre_fwd_bwd", trainer)
        failed.run("post_step", trainer)
        healthy_metrics = healthy.flush()
        failed_metrics = failed.flush()
        self.assertEqual(healthy_metrics.keys(), failed_metrics.keys())
        self.assertEqual(failed_metrics["router_stats"], (Reduce.SUM, _QUARANTINED))
        self.assertEqual(
            merge_metrics([healthy_metrics, failed_metrics]),
            MergeResult(values={"count": 2}, dropped=frozenset({"router_stats"})),
        )


class TestMergeMetrics(unittest.TestCase):
    def test_rank0_keeps_rank_zeros_value(self):
        per_rank: list[dict[str, Any]] = [
            {"x": (Reduce.RANK0, "r0-value")},
            {"x": (Reduce.RANK0, "r1-value")},
        ]
        self.assertEqual(merge_metrics(per_rank).values, {"x": "r0-value"})

    def test_sum_max_min_combine_scalars(self):
        per_rank = [
            {"s": (Reduce.SUM, 1), "mx": (Reduce.MAX, 1), "mn": (Reduce.MIN, 5)},
            {"s": (Reduce.SUM, 2), "mx": (Reduce.MAX, 7), "mn": (Reduce.MIN, 2)},
        ]
        self.assertEqual(merge_metrics(per_rank).values, {"s": 3, "mx": 7, "mn": 2})

    def test_mean_combines_scalars(self):
        per_rank = [{"m": (Reduce.MEAN, 2)}, {"m": (Reduce.MEAN, 4)}]
        self.assertEqual(merge_metrics(per_rank).values, {"m": 3.0})

    def test_per_rank_returns_rank_ordered_list(self):
        per_rank = [{"p": (Reduce.PER_RANK, "a")}, {"p": (Reduce.PER_RANK, "b")}]
        self.assertEqual(merge_metrics(per_rank).values, {"p": ["a", "b"]})

    def test_sum_combines_nested_dicts_and_lists_elementwise(self):
        per_rank = [
            {"d": (Reduce.SUM, {"a": [1, 2], "b": 10})},
            {"d": (Reduce.SUM, {"a": [3, 4], "b": 20})},
        ]
        self.assertEqual(merge_metrics(per_rank).values, {"d": {"a": [4, 6], "b": 30}})

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
        self.assertEqual(merge_metrics(per_rank).values, {"d": {"a": [3.0, 6.0]}})

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

    def test_invalid_on_any_rank_drops_only_that_metric(self):
        per_rank = [
            {"keep": (Reduce.SUM, 1), "router_stats": (Reduce.SUM, {"sampled_tokens": 5.0})},
            {"keep": (Reduce.SUM, 2), "router_stats": (Reduce.SUM, _QUARANTINED)},
        ]
        self.assertEqual(
            merge_metrics(per_rank),
            MergeResult(values={"keep": 3}, dropped=frozenset({"router_stats"})),
        )

    def test_invalid_per_rank_metric_is_dropped_not_listed(self):
        per_rank = [{"p": (Reduce.PER_RANK, "a")}, {"p": (Reduce.PER_RANK, _QUARANTINED)}]
        self.assertEqual(merge_metrics(per_rank), MergeResult(values={}, dropped=frozenset({"p"})))

    def test_rank0_metric_validity_comes_from_rank_zero_only(self):
        invalid_on_rank0 = [{"x": (Reduce.RANK0, _QUARANTINED)}, {"x": (Reduce.RANK0, "r1")}]
        self.assertEqual(merge_metrics(invalid_on_rank0), MergeResult(values={}, dropped=frozenset({"x"})))
        invalid_on_rank1 = [{"x": (Reduce.RANK0, "r0")}, {"x": (Reduce.RANK0, _QUARANTINED)}]
        self.assertEqual(merge_metrics(invalid_on_rank1), MergeResult(values={"x": "r0"}, dropped=frozenset()))

    def test_rank_zero_only_metric_absent_on_other_ranks_merges(self):
        per_rank = [{"x": (Reduce.RANK0, "r0"), "s": (Reduce.SUM, 1)}, {"s": (Reduce.SUM, 2)}]
        self.assertEqual(merge_metrics(per_rank), MergeResult(values={"x": "r0", "s": 3}, dropped=frozenset()))

    def test_invalid_entry_with_mismatched_reduce_still_raises(self):
        per_rank = [{"x": (Reduce.SUM, 1)}, {"x": (Reduce.MAX, _QUARANTINED)}]
        with self.assertRaises(ValueError):
            merge_metrics(per_rank)


class TestInvalidMetric(unittest.TestCase):
    def test_pickle_round_trip_keeps_identity(self):
        entry = (Reduce.SUM, InvalidMetric.QUARANTINED)
        op, value = pickle.loads(pickle.dumps(entry))
        self.assertIs(op, Reduce.SUM)
        self.assertIs(value, InvalidMetric.QUARANTINED)

    def test_ray_serialization_round_trip_keeps_identity(self):
        try:
            from ray import cloudpickle
        except ImportError:
            self.skipTest("ray is not installed")
        self.assertIs(cloudpickle.loads(cloudpickle.dumps(InvalidMetric.QUARANTINED)), InvalidMetric.QUARANTINED)

    def test_is_not_json_serializable(self):
        with self.assertRaises(TypeError):
            json.dumps({"router_stats": InvalidMetric.QUARANTINED})


if __name__ == "__main__":
    unittest.main()
