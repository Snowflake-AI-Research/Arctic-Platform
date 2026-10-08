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

"""Test 1: per-tensor gradient-norm agreement after one forward-backward.

The test is not really asking whether a matmul is right. It is asking whether Arctic Platform gets two *scalars* right,
neither of which a single-sequence smoke test can expose.

The loss denominator. Arctic Platform optimizes a token-weighted mean over the whole request. Padding makes rows
contribute unequal token counts, and sequence parallelism gives each rank a different active-token count,
so a denominator computed per rank, per row, or per microbatch produces a gradient wrong by one constant.

Gradient accumulation. A global batch too large for one step is split by token budget. Dividing by the
microbatch count on top of the token weighting, or failing to, scales the whole gradient.

Both faults present the same way: every tensor off by the *same* ratio. That is why the ratio column
matters more than the absolute difference, and why the arms deliberately carry padding, unequal row
lengths, and a microbatch count above one.
"""

from __future__ import annotations

from typing import List

from ..harness.names import align
from ..harness.registry import Mismatch
from ..harness.registry import TestOutcome
from ..harness.registry import TestResult
from ..harness.registry import correctness_test
from ..harness.spec import tolerance_text


@correctness_test(
    "single-step-grads",
    title="Forward-backward gradient correctness for a single step",
    criterion="per-tensor gradient L2 norms stay within the config-specific reference-repeatability gate",
)
def run(ctx) -> List[TestResult]:
    results: List[TestResult] = []
    for arm in ctx.arms:
        results.append(_run_arm(ctx, arm))
    return results


def _run_arm(ctx, arm) -> TestResult:
    reference = ctx.reference_for(arm)
    target = ctx.target_for(arm)

    loss_delta = abs(target.avg_loss - reference["loss"])

    pairs, only_dss, only_reference = align(target.grad_norms, reference["grad_norms"])

    # Every parameter pairs, or the comparison is not the one being claimed: an unmatched name is a tensor
    # nobody checked, and the tensors that did pair would report a pass for coverage the case never had.
    expected = len(pairs) + len(only_reference)
    if only_dss or only_reference:
        return TestResult(
            test_id="single-step-grads",
            config_id=ctx.config_id,
            arm=arm.name,
            outcome=TestOutcome.FAIL,
            summary=f"only {len(pairs)}/{expected} semantic grad norm tensors matched by name",
            reason=(
                "Parameter-name alignment is incomplete, so the comparison "
                "does not cover every parameter. Unmatched Arctic Platform names: "
                f"{', '.join(only_dss[:4])}{' ...' if len(only_dss) > 4 else ''}. "
                "Unmatched reference names: "
                f"{', '.join(only_reference[:4])}{' ...' if len(only_reference) > 4 else ''}."
            ),
            metrics={
                "tensors_compared": float(len(pairs)),
                "dss_tensors": float(expected),
                "reference_tensors": float(len(reference["grad_norms"])),
            },
        )

    gate = ctx.spec.tolerance_for("single-step-grads")
    mismatches = [Mismatch(name=name, target=a, reference=b) for name, a, b in pairs]
    over_stated = [m for m in mismatches if m.abs_diff > gate]
    mismatches.sort(key=lambda m: m.abs_diff, reverse=True)
    worst_name = mismatches[0].name if mismatches else None
    failures = sorted(over_stated, key=lambda m: m.abs_diff, reverse=True)

    ratios = sorted(m.ratio for m in mismatches)
    median_ratio = ratios[len(ratios) // 2]
    uniform = len(over_stated) > len(pairs) // 2

    outcome = TestOutcome.PASS if not over_stated else TestOutcome.FAIL
    reason = ""
    if over_stated and uniform:
        reason = (
            f"Every parameter's gradient norm is off by roughly the same factor ({median_ratio:.6f}), "
            "which points at a scalar "
            "rather than at a module: the loss denominator or the gradient-accumulation scaling. "
            f"This arm packs {arm.global_batch_size} sequences into {arm.dss_microbatches} microbatches with "
            f"{arm.pad_fraction:.1%} padding."
        )
    elif over_stated:
        reason = (
            f"The median ratio is {median_ratio:.6f}, so the disagreement is localized rather "
            "than a global scale error."
        )

    return TestResult(
        test_id="single-step-grads",
        config_id=ctx.config_id,
        arm=arm.name,
        outcome=outcome,
        summary=f"{len(over_stated)}/{len(pairs)} grad norm tensors failed to match {tolerance_text(gate)} tolerance",
        reason=reason,
        mismatches=failures,
        worst_name=worst_name,
        metrics={
            "dss_loss": target.avg_loss,
            "reference_loss": reference["loss"],
            "loss_delta": loss_delta,
            "median_ratio": median_ratio,
            "max_abs_diff": max(m.abs_diff for m in mismatches),
            "tensors_compared": float(len(pairs)),
            "tensors_over_stated_criterion": float(len(over_stated)),
            "stated_criterion_abs": gate,
            "unmatched_dss": float(len(only_dss)),
            "unmatched_reference": float(len(only_reference)),
            "reference_microbatches": float(reference["microbatches"]),
            "dss_microbatches_predicted": float(arm.dss_microbatches),
            **({"dss_model_calls_observed": float(target.model_calls)} if target.model_calls is not None else {}),
        },
    )


# Driving the arms from the runner lets each arm be compared the moment its reference lands, instead of
# every arm waiting for the slowest one.
run.per_arm = _run_arm
