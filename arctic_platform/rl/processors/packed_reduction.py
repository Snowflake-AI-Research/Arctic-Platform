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

"""Packed-microbatch loss reduction: resolve, apply, and combine metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from arctic_platform.common.registry import LOSS_FNS
from arctic_platform.common.registry import PACKED_LOSS_REDUCTION_ATTR
from arctic_platform.common.registry import resolve_fn

_SUMMED_METRIC_PREFIXES = ("loss_term_",)
_SUMMED_METRIC_SUFFIXES = ("_sum", "_count")
_GLOBAL_SCALE_KEYS = ("dp_size", "batch_num_tokens", "global_batch_size")


@dataclass(frozen=True)
class PackedLossReduction:
    """Objective-owned rules for combining packed microbatch losses.

    ``loss_scales`` are applied before backward. ``reporting_weights`` combine
    per-microbatch means and rate-like metrics. Additive losses are already
    normalized against a batch-global denominator and are summed for reporting;
    local means are reported with their objective-owned weights.
    """

    loss_scales: tuple[float, ...]
    reporting_weights: tuple[float, ...]
    loss_is_additive: bool


def metric_is_summed(key: str) -> bool:
    """Whether a scalar metric is additive across packed microbatches.

    Additive metrics — objective-term contributions (``loss_term_*``) and
    token/sequence counts (``*_count``, ``*_sum``) — must be SUMMED when
    microbatch results are combined. Averaging them would misreport totals
    whenever a call splits into multiple microbatches.

    SFT pairing uses a different convention (``{name}.sum`` / ``{name}.tokens``)
    via ``combine_metric_microbatches`` and is not this helper.
    """
    return key.startswith(_SUMMED_METRIC_PREFIXES) or key.endswith(_SUMMED_METRIC_SUFFIXES)


def _validated_packed_weights(weights: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(weight) for weight in weights)
    if not values:
        raise ValueError("packed loss reduction requires at least one microbatch")
    if any(not math.isfinite(weight) or weight < 0 for weight in values):
        raise ValueError("packed loss reduction weights must be finite and non-negative")
    return values


def local_mean_packed_loss_reduction(weights: Sequence[float]) -> PackedLossReduction:
    """Combine microbatch-local means into one weighted local mean."""
    values = _validated_packed_weights(weights)
    if sum(values) == 0:
        values = (1.0,) * len(values)
    total = sum(values)
    return PackedLossReduction(
        loss_scales=tuple(weight / total for weight in values),
        reporting_weights=values,
        loss_is_additive=False,
    )


def additive_packed_loss_reduction(reporting_weights: Sequence[float]) -> PackedLossReduction:
    """Sum microbatch contributions already normalized by a global denominator."""
    weights = _validated_packed_weights(reporting_weights)
    if sum(weights) == 0:
        weights = (1.0,) * len(weights)
    return PackedLossReduction(
        loss_scales=(1.0,) * len(weights),
        reporting_weights=weights,
        loss_is_additive=True,
    )


def _scale_value_present(bag: dict | None, key: str) -> bool:
    return bag is not None and key in bag and bag[key] is not None


def _scale_values_equal(key: str, left, right) -> bool:
    if key == "batch_num_tokens":
        try:
            return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-9)
        except (TypeError, ValueError):
            return left == right
    return left == right


def assert_aligned_global_loss_scales(microbatches: Sequence[dict]) -> None:
    """Require every packed microbatch to carry the same step-global trio."""
    if len(microbatches) <= 1:
        return
    first = microbatches[0]
    for index, microbatch in enumerate(microbatches[1:], start=1):
        for key in _GLOBAL_SCALE_KEYS:
            have_first = _scale_value_present(first, key)
            have_other = _scale_value_present(microbatch, key)
            if not have_first and not have_other:
                continue
            if (
                not have_first
                or not have_other
                or not _scale_values_equal(key, first[key], microbatch[key])
            ):
                raise ValueError(
                    f"packed microbatch {index} {key}={microbatch.get(key)!r} disagrees "
                    f"with microbatch 0 {key}={first.get(key)!r}"
                )


def _config_with_microbatch_scales(processing: dict | None, microbatches: Sequence[dict]) -> dict:
    """Fill missing trio keys from the first microbatch; config wins when set."""
    config = dict((processing or {}).get("config") or {})
    first = microbatches[0] if microbatches else {}
    for key in _GLOBAL_SCALE_KEYS:
        if config.get(key) is None and first.get(key) is not None:
            config[key] = first[key]
    return config


def resolve_packed_loss_reduction(
    processing: dict | None,
    microbatches: Sequence[dict],
) -> PackedLossReduction:
    """Resolve an objective's packed-microbatch reduction and preflight it."""
    n_mbs = len(microbatches)
    if n_mbs == 0:
        raise ValueError("packed loss reduction requires at least one microbatch")

    assert_aligned_global_loss_scales(microbatches)

    loss_fn_name = (processing or {}).get("loss_fn", "grpo")
    if loss_fn_name is None:
        return local_mean_packed_loss_reduction((1.0,) * n_mbs)

    fn = resolve_fn(LOSS_FNS, loss_fn_name)
    resolver = getattr(fn, PACKED_LOSS_REDUCTION_ATTR, None)
    if resolver is None:
        if n_mbs > 1:
            raise ValueError(
                f"loss_fn {loss_fn_name!r} does not declare packed-microbatch "
                "reduction metadata; use one microbatch or register the loss with "
                "packed_loss_reduction="
            )
        return local_mean_packed_loss_reduction((1.0,))

    reduction = resolver(
        microbatches,
        _config_with_microbatch_scales(processing, microbatches),
        loss_fn_name,
    )
    if not isinstance(reduction, PackedLossReduction):
        raise TypeError(
            f"loss_fn {loss_fn_name!r} packed reduction resolver must return PackedLossReduction"
        )
    if not (len(reduction.loss_scales) == len(reduction.reporting_weights) == n_mbs):
        raise ValueError(
            f"loss_fn {loss_fn_name!r} packed reduction metadata must contain {n_mbs} entries"
        )
    _validated_packed_weights(reduction.loss_scales)
    reporting_weights = _validated_packed_weights(reduction.reporting_weights)
    if sum(reporting_weights) == 0:
        raise ValueError(f"loss_fn {loss_fn_name!r} packed reporting weights must have a positive sum")
    return reduction


def apply_packed_loss_reduction(
    engine,
    loss_tensor: torch.Tensor,
    scale: float,
    *,
    backward: bool | str,
) -> torch.Tensor | None:
    """Scale a ``loss_only`` tensor and optionally run ``engine.backward``.

    DeepSpeed must not divide by GAS again (``scale_wrt_gas=False``).
    """
    scaled = loss_tensor * scale
    if backward is True:
        engine.backward(scaled, scale_wrt_gas=False)
        return None
    return scaled


def combine_packed_losses(losses: Sequence[float], reduction: PackedLossReduction) -> float:
    """Combine per-microbatch reported losses with the objective's reduction."""
    if reduction.loss_is_additive:
        return float(sum(losses))
    total = sum(reduction.reporting_weights)
    return float(sum(loss * weight for loss, weight in zip(losses, reduction.reporting_weights)) / total)


def combine_packed_metrics(
    microbatch_metrics: Sequence[dict],
    reporting_weights: Sequence[float],
) -> dict[str, float]:
    """Sum additive metrics; weighted-mean the rest. No ``{name}`` vs ``{name}.sum`` mix."""
    n_metrics = len(microbatch_metrics)
    weights = tuple(float(weight) for weight in reporting_weights)
    total_weight = sum(weights)
    if total_weight == 0:
        total_weight = float(n_metrics or 1)
        weights = (1.0,) * n_metrics
    all_keys = {key for metrics in microbatch_metrics for key in (metrics or {})}
    colliding = sorted(key for key in all_keys if f"{key}.sum" in all_keys)
    if colliding:
        raise ValueError(
            "packed metrics mix a rate key with its sum pair: "
            + ", ".join(f"{key!r} and {key + '.sum'!r}" for key in colliding)
        )
    summed: dict[str, float] = {}
    averaged: dict[str, float] = {}
    for metrics, weight in zip(microbatch_metrics, weights):
        for key, value in (metrics or {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if metric_is_summed(key):
                summed[key] = summed.get(key, 0.0) + number
            else:
                averaged[key] = averaged.get(key, 0.0) + number * weight
    out = {key: value / total_weight for key, value in averaged.items()}
    out.update(summed)
    return out
