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

"""Adam moment and moment-independent update agreement after one forward-backward step."""

from __future__ import annotations

from pathlib import Path
from typing import List

from ..harness.registry import Mismatch
from ..harness.registry import TestOutcome
from ..harness.registry import TestResult
from ..harness.registry import correctness_test
from ..harness.runner import OPTIMIZER_LEARNING_RATE
from ..harness.spec import tolerance_text
from .optimizer_state import compare_optimizer_artifacts


@correctness_test(
    "single-step-optimizer",
    title="Adam optimizer-state and update-residual correctness for one step",
    criterion="full-tensor moment and update-residual deltas stay within the config-specific gate",
)
def run(ctx) -> List[TestResult]:
    return [_run_arm(ctx, arm) for arm in ctx.arms]


def _run_arm(ctx, arm) -> TestResult:
    """Read the optimizer artifacts of the same execution the gradient check judged."""
    return _compare_arm(ctx, arm, ctx.reference_for(arm), ctx.target_for(arm))


def _compare_arm(ctx, arm, reference: dict, target) -> TestResult:
    if not reference.get("optimizer_state_manifest") or not target.optimizer_state_manifest:
        raise ValueError("optimizer step did not return both optimizer-state manifests")
    comparison = compare_optimizer_artifacts(
        Path(target.optimizer_state_manifest),
        Path(reference["optimizer_state_manifest"]),
        optimizer_config=ctx.cfg.training["optimizer"],
        learning_rate=OPTIMIZER_LEARNING_RATE,
    )
    compared = len(comparison.deltas)
    expected = compared + 3 * len(comparison.only_reference)
    if comparison.only_dss or comparison.only_reference:
        return TestResult(
            test_id="single-step-optimizer",
            config_id=ctx.config_id,
            arm=arm.name,
            outcome=TestOutcome.FAIL,
            summary=f"only {compared}/{expected} optimizer quantities matched by parameter name",
            reason=(
                f"Unmatched Arctic Platform parameters: {', '.join(comparison.only_dss[:4]) or 'none'}. "
                f"Unmatched reference parameters: {', '.join(comparison.only_reference[:4]) or 'none'}."
            ),
            metrics={
                "optimizer_values_compared": float(compared),
                "unmatched_dss": float(len(comparison.only_dss)),
                "unmatched_reference": float(len(comparison.only_reference)),
            },
        )

    gate = ctx.spec.tolerance_for("single-step-optimizer")
    values = [
        Mismatch(name=f"{item.name}::{item.state}", target=item.delta_norm, reference=0.0, difference=item.delta_norm)
        for item in comparison.deltas
    ]
    failures = sorted((value for value in values if value.target > gate), key=lambda value: value.target, reverse=True)
    worst = max(values, key=lambda value: value.target, default=None)
    return TestResult(
        test_id="single-step-optimizer",
        config_id=ctx.config_id,
        arm=arm.name,
        outcome=TestOutcome.PASS if not failures else TestOutcome.FAIL,
        summary=(
            f"{len(failures)}/{compared} optimizer moment and update-residual tensors failed to match "
            f"{tolerance_text(gate)} tolerance"
        ),
        reason=(
            "The values are full-tensor Arctic Platform-minus-reference L2 norms. The update residual removes the "
            "Adam update implied by each engine's own moments before comparison."
        ),
        mismatches=failures,
        worst_name=worst.name if worst else None,
        metrics={
            "reference_loss": float(reference["loss"]),
            "dss_loss": target.avg_loss,
            "loss_delta": abs(target.avg_loss - float(reference["loss"])),
            "optimizer_values_compared": float(compared),
            "optimizer_values_over_criterion": float(len(failures)),
            "max_optimizer_delta_norm": worst.target if worst else 0.0,
            "stated_criterion_abs": gate,
            "reference_microbatches": float(reference["microbatches"]),
            **({"dss_model_calls_observed": float(target.model_calls)} if target.model_calls is not None else {}),
        },
    )


run.per_arm = _run_arm
