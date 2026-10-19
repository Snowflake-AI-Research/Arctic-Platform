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

import json
import math
from pathlib import Path
from typing import List

from ..harness.registry import Mismatch
from ..harness.registry import TestOutcome
from ..harness.registry import TestResult
from ..harness.registry import correctness_test
from ..harness.runner import OPTIMIZER_LEARNING_RATE
from ..harness.spec import tolerance_text
from .optimizer_state import OPTIMIZER_NORM_CRITERION
from .optimizer_state import compare_optimizer_norms
from .optimizer_state import summarize_optimizer_artifacts


@correctness_test(
    "single-step-optimizer",
    title="Adam optimizer-state and update-residual norm correctness for one step",
    criterion=(
        f"{OPTIMIZER_NORM_CRITERION}: scalar norm differences stay within the explicitly reviewed reference gate"
    ),
)
def run(ctx) -> List[TestResult]:
    return [_run_arm(ctx, arm) for arm in ctx.arms]


def _run_arm(ctx, arm) -> TestResult:
    """Read the optimizer artifacts of the same execution the gradient check judged."""
    return _compare_arm(ctx, arm, ctx.reference_for(arm), ctx.target_for(arm))


def _compare_arm(ctx, arm, reference: dict, target) -> TestResult:
    if not reference.get("optimizer_norms"):
        raise ValueError("optimizer comparison requires stored optimizer_norms; prepare a compatible reference")
    gate_record = reference.get("optimizer_norm_gate")
    if (
        not isinstance(gate_record, dict)
        or gate_record.get("criterion") != OPTIMIZER_NORM_CRITERION
        or gate_record.get("status") != "reviewed"
    ):
        raise ValueError(
            f"optimizer comparison requires a reviewed optimizer_norm_gate for {OPTIMIZER_NORM_CRITERION}"
        )
    gate = gate_record.get("absolute")
    if type(gate) not in {int, float} or not math.isfinite(gate) or gate < 0:
        raise ValueError("optimizer_norm_gate.absolute must be finite and nonnegative")
    provenance = gate_record.get("provenance")
    if (
        not isinstance(provenance, (str, dict))
        or not provenance
        or (isinstance(provenance, str) and not provenance.strip())
    ):
        raise ValueError("optimizer_norm_gate requires nonempty provenance")
    provenance_text = json.dumps(provenance, sort_keys=True, allow_nan=False)
    if not target.optimizer_state_manifest:
        raise ValueError("optimizer step did not return the target optimizer-state manifest")
    dss_summary = summarize_optimizer_artifacts(
        Path(target.optimizer_state_manifest),
        optimizer_config=ctx.cfg.training["optimizer"],
        learning_rate=OPTIMIZER_LEARNING_RATE,
        backend="dss",
    )
    reference_summary = reference["optimizer_norms"]
    comparison = compare_optimizer_norms(dss_summary, reference_summary)
    compared = len(comparison.deltas)
    expected = reference_summary["optimizer_values_count"]
    if comparison.only_dss or comparison.only_reference:
        return TestResult(
            test_id="single-step-optimizer",
            config_id=ctx.config_id,
            arm=arm.name,
            outcome=TestOutcome.FAIL,
            summary=f"only {compared}/{expected} optimizer quantities matched by parameter name",
            reason=(
                f"Unmatched Arctic Platform parameters: {', '.join(comparison.only_dss[:4]) or 'none'}. "
                f"Unmatched reference parameters: {', '.join(comparison.only_reference[:4]) or 'none'}. "
                f"Criterion: {OPTIMIZER_NORM_CRITERION}; reviewed gate provenance: {provenance_text}."
            ),
            metrics={
                "optimizer_values_compared": float(compared),
                "unmatched_dss": float(len(comparison.only_dss)),
                "unmatched_reference": float(len(comparison.only_reference)),
            },
        )

    values = [
        Mismatch(
            name=f"{item.name}::{item.state}",
            target=dss_summary["parameters"][item.name][item.state],
            reference=reference_summary["parameters"][item.name][item.state],
            difference=item.delta_norm,
        )
        for item in comparison.deltas
    ]
    failures = sorted(
        (value for value in values if value.abs_diff > gate), key=lambda value: value.abs_diff, reverse=True
    )
    worst = max(values, key=lambda value: value.abs_diff, default=None)
    return TestResult(
        test_id="single-step-optimizer",
        config_id=ctx.config_id,
        arm=arm.name,
        outcome=TestOutcome.PASS if not failures else TestOutcome.FAIL,
        summary=(
            f"{len(failures)}/{compared} optimizer moment and update-residual norms failed to match "
            f"{tolerance_text(gate)} tolerance"
        ),
        reason=(
            f"Criterion: {OPTIMIZER_NORM_CRITERION}. Differences are absolute differences of float32 L2 norms, "
            "including zero and near-zero norms without exemptions. The update residual removes the Adam update "
            "implied by each engine's own moments before reduction. Equal norms can conceal sign, permutation, "
            f"or directional errors. Reviewed gate provenance: {provenance_text}."
        ),
        mismatches=failures,
        worst_name=worst.name if worst else None,
        metrics={
            "reference_loss": float(reference["loss"]),
            "dss_loss": target.avg_loss,
            "loss_delta": abs(target.avg_loss - float(reference["loss"])),
            "optimizer_values_compared": float(compared),
            "optimizer_values_over_criterion": float(len(failures)),
            "max_optimizer_norm_delta": worst.abs_diff if worst else 0.0,
            "stated_criterion_abs": gate,
            "reference_microbatches": float(reference["microbatches"]),
            **({"dss_model_calls_observed": float(target.model_calls)} if target.model_calls is not None else {}),
        },
    )


run.per_arm = _run_arm
