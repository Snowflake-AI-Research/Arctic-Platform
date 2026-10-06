# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Step metrics must describe the whole step, not one micro-batch."""

from __future__ import annotations

import pytest

from arctic_platform.integrations.harbor.backend import _merge_fwd_bwd_metrics


def test_empty_input_yields_empty_dict():
    assert _merge_fwd_bwd_metrics([]) == {}
    assert _merge_fwd_bwd_metrics([{}, {}]) == {}


def test_counts_sum_across_micro_batches():
    merged = _merge_fwd_bwd_metrics(
        [
            {"trainable_logprob_count_all": 1000.0},
            {"trainable_logprob_count_all": 214.0},
        ]
    )
    assert merged["trainable_logprob_count_all"] == 1214.0


def test_rl_model_calls_sums():
    merged = _merge_fwd_bwd_metrics([{"rl_model_calls": 1}, {"rl_model_calls": 1}])
    assert merged["rl_model_calls"] == 2


def test_means_are_token_weighted_not_arithmetic():
    """The big slice should dominate; a plain mean would read 0.55."""
    merged = _merge_fwd_bwd_metrics(
        [
            {"trainable_logprob_count_all": 9000.0, "approx_kl": 0.1},
            {"trainable_logprob_count_all": 1000.0, "approx_kl": 1.0},
        ]
    )
    assert merged["approx_kl"] == pytest.approx((9000 * 0.1 + 1000 * 1.0) / 10000)
    assert merged["approx_kl"] == pytest.approx(0.19)


def test_extremes_take_the_extreme():
    merged = _merge_fwd_bwd_metrics(
        [
            {
                "trainable_logprob_count_all": 10.0,
                "trainable_logprob_abs_delta_max_all": 23.25,
                "trainable_logprob_lowvar_all/max": 19.0,
                "trainable_logprob_lowvar_all/min": 0.5,
            },
            {
                "trainable_logprob_count_all": 10.0,
                "trainable_logprob_abs_delta_max_all": 2.0,
                "trainable_logprob_lowvar_all/max": 3.0,
                "trainable_logprob_lowvar_all/min": -1.0,
            },
        ]
    )
    assert merged["trainable_logprob_abs_delta_max_all"] == 23.25
    assert merged["trainable_logprob_lowvar_all/max"] == 19.0
    assert merged["trainable_logprob_lowvar_all/min"] == -1.0


def test_falls_back_to_plain_mean_when_no_weights_reported():
    merged = _merge_fwd_bwd_metrics([{"loss": 0.2}, {"loss": 0.4}])
    assert merged["loss"] == pytest.approx(0.3)


def test_zero_weights_do_not_divide_by_zero():
    merged = _merge_fwd_bwd_metrics(
        [
            {"trainable_logprob_count_all": 0.0, "loss": 0.2},
            {"trainable_logprob_count_all": 0.0, "loss": 0.4},
        ]
    )
    assert merged["trainable_logprob_count_all"] == 0.0
    assert merged["loss"] == pytest.approx(0.3)


def test_rank_is_not_averaged():
    merged = _merge_fwd_bwd_metrics([{"rank": 0}, {"rank": 0}])
    assert merged["rank"] == 0


def test_non_numeric_values_pass_through():
    merged = _merge_fwd_bwd_metrics([{"note": "a"}, {"note": "b"}])
    assert merged["note"] in {"a", "b"}


def test_key_missing_from_some_micro_batches_uses_only_reporters():
    """A key only the big slice reports must not be diluted by slices that omit it."""
    merged = _merge_fwd_bwd_metrics(
        [
            {"trainable_logprob_count_all": 9000.0, "entropy": 0.5},
            {"trainable_logprob_count_all": 1000.0},
        ]
    )
    assert merged["entropy"] == pytest.approx(0.5)
