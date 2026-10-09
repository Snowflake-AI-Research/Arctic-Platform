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

"""Test 20: a run interrupted by a checkpoint reproduces the loss of a run that was not.

Both sides are Arctic Platform on identical inputs, so the single-GPU Hugging Face reference takes no part. What
separates them is a job boundary: the reference run trains twenty iterations in one job, while the target
run trains ten, writes a resumable checkpoint, and runs the last ten in a fresh job initialized from that
checkpoint. Weights, optimizer moments and LR-scheduler state therefore arrive from disk for the target's
second half and from memory for the reference's.

Iteration 10 is the control. Up to it the two runs are the same computation in the same engine on the same
batches, so a disagreement there is a harness defect rather than a finding, and it would leave the
iteration-20 comparison with nothing to stand on. Iteration 20 is the resume verdict: it is reached through
ten optimizer steps that read the restored state, so a moment or a scheduler position that did not survive
the round trip moves the loss.

Every iteration consumes its own batch, seeded by iteration, and both runs consume the same batches in the
same order -- the resumed job continues the data order at batch 11 rather than replaying it.
"""

from __future__ import annotations

import traceback
from decimal import ROUND_CEILING
from decimal import Decimal
from typing import Dict
from typing import List

from ..harness.batches import Batch
from ..harness.batches import build_batch
from ..harness.dss_driver import GatewayTransport
from ..harness.dss_driver import pack
from ..harness.registry import Mismatch
from ..harness.registry import TestOutcome
from ..harness.registry import TestResult
from ..harness.registry import correctness_test
from ..harness.seeds import SEED
from ..harness.spec import TestTolerance
from ..harness.spec import tolerance_text
from ..harness.transport import TrainingJob
from ..harness.transport import Transport

TEST_ID = "checkpoint-resume-loss"

ITERATIONS = 20
CHECKPOINT_ITERATION = 10
# The rate both runs step at, held here rather than read from the config, so the two runs differ
# only by the checkpoint boundary whatever config they run under.
LEARNING_RATE = 1e-3
# Onboarding runs this many identical short trajectories with determinism off. The widest per-iteration
# loss range across them, times two and raised to the next 0.001, is the gate the regression reads.
CALIBRATION_RUNS = 3
CALIBRATION_STEPS = 10
LOSS_RANGE_MULTIPLIER = 2.0
GATE_QUANTUM = Decimal("0.001")
GATE_MINIMUM = Decimal("0.001")

RECORDED_ITERATIONS = (CHECKPOINT_ITERATION, ITERATIONS)


@correctness_test(
    TEST_ID,
    title="Loss agreement between an uninterrupted run and a checkpoint-and-resume run",
    criterion=(
        "the loss at the checkpoint iteration and at the final iteration agree between the two runs "
        "within the absolute gate onboarding measured for this config"
    ),
    compares_to_reference=False,
    # TODO: switch back to ``requires_hosted_control_plane=True`` once the hosted worker image carries
    # ``debug.full_determinism_must_comply``. That image raises on a refused deterministic attention backward, so
    # every head-dimension-256 config dies at worker init there; a gateway resumes through the runtime load route.
    requires_hosted_control_plane=False,
)
def run(ctx) -> List[TestResult]:
    """One transport for the whole check; every case runs its three jobs through it, one at a time.

    The transport decides where those jobs run -- a gateway on this node, or the hosted control plane --
    and nothing below this line depends on which it is.
    """
    results: List[TestResult] = []
    workdir = ctx.workdir / TEST_ID
    workdir.mkdir(parents=True, exist_ok=True)
    with ctx.transport_for(workdir) as transport:
        for arm in ctx.arms:
            try:
                results.append(_run_arm(ctx, transport, arm))
            except Exception as exc:  # noqa: BLE001 - one failing case must not hide the others
                traceback.print_exc()
                results.append(
                    TestResult(
                        test_id=TEST_ID,
                        config_id=ctx.config_id,
                        arm=arm.name,
                        outcome=TestOutcome.FAIL,
                        summary=f"{type(exc).__name__}: {exc}",
                    )
                )
    return results


def _run_arm(ctx, transport: Transport, arm) -> TestResult:
    bodies = _iteration_bodies(ctx, arm)
    reference = _uninterrupted(transport, bodies)
    target = _checkpoint_and_resume(transport, bodies)
    return _verdict(ctx, arm, reference, target)


def iteration_batches(ctx, arm) -> List[Batch]:
    """One batch per iteration, all twenty built before either run starts.

    The seed moves with the iteration, so consecutive iterations train on different data and a run that
    replayed a batch instead of advancing would not reach the same loss. Each batch carries the padding
    and unequal row lengths every generated case carries.
    """
    return [
        build_batch(arm.name, arm.global_batch_size, arm.max_seq_len, ctx.vocab_size, seed=SEED + iteration)
        for iteration in range(1, ITERATIONS + 1)
    ]


def _iteration_bodies(ctx, arm) -> List[bytes]:
    """The iteration batches as request bytes, so both runs send the same bytes in the same order."""
    provider = str(ctx.cfg.training.get("model_provider", "huggingface"))
    return [pack(batch, model_provider=provider) for batch in iteration_batches(ctx, arm)]


def _trajectory(job: TrainingJob, bodies: List[bytes], *, first_iteration: int) -> Dict[int, float]:
    """Run ``bodies`` in order from ``first_iteration``, keeping the losses at the recorded iterations."""
    losses: Dict[int, float] = {}
    for iteration, body in enumerate(bodies, start=first_iteration):
        loss = job.fwd_bwd_step(body, learning_rate=LEARNING_RATE)
        if iteration in RECORDED_ITERATIONS:
            losses[iteration] = loss
    return losses


def _uninterrupted(transport: Transport, bodies: List[bytes]) -> Dict[int, float]:
    """Twenty iterations in one job, no checkpoint written."""
    with transport.job() as job:
        return _trajectory(job, bodies, first_iteration=1)


def _checkpoint_and_resume(transport: Transport, bodies: List[bytes]) -> Dict[int, float]:
    """Ten iterations, a resumable checkpoint, then iterations 11 to 20 in a job initialized from it."""
    with transport.job() as first_job:
        losses = _trajectory(first_job, bodies[:CHECKPOINT_ITERATION], first_iteration=1)
        checkpoint = first_job.save_resumable()

    with transport.job(resume_from=checkpoint) as resumed_job:
        losses.update(
            _trajectory(resumed_job, bodies[CHECKPOINT_ITERATION:], first_iteration=CHECKPOINT_ITERATION + 1)
        )
    return losses


def compare(reference: Dict[int, float], target: Dict[int, float]) -> List[Mismatch]:
    """The recorded iterations, as compared quantities ordered by iteration."""
    return [
        Mismatch(name=f"loss at iteration {iteration}", target=target[iteration], reference=reference[iteration])
        for iteration in RECORDED_ITERATIONS
    ]


def _verdict(ctx, arm, reference: Dict[int, float], target: Dict[int, float]) -> TestResult:
    tolerance = ctx.spec.tolerance_for(TEST_ID)
    compared = compare(reference, target)
    failures = sorted(
        (value for value in compared if value.abs_diff > tolerance), key=lambda value: value.abs_diff, reverse=True
    )
    worst = max(compared, key=lambda value: value.abs_diff)
    metrics: Dict[str, float] = {"max_abs_diff": worst.abs_diff, "stated_criterion_abs": tolerance}
    for iteration in RECORDED_ITERATIONS:
        metrics[f"reference_loss_iteration_{iteration}"] = reference[iteration]
        metrics[f"target_loss_iteration_{iteration}"] = target[iteration]
    reason = ""
    if any(value.name == compared[0].name for value in failures):
        reason = (
            f"The two runs already disagree at iteration {CHECKPOINT_ITERATION}, before the checkpoint "
            f"is written, so the iteration-{ITERATIONS} comparison has nothing to stand on. Up to the "
            "checkpoint the two runs are the same computation on the same batches, which leaves two "
            "causes and excludes a resume defect: either they were given different work, or the jobs are "
            "not reduction-order pinned and have drifted apart under their own optimizer steps. The "
            "first iteration's losses separate them -- identical there means the inputs agree and the "
            "drift is the engine's."
        )
    return TestResult(
        test_id=TEST_ID,
        config_id=ctx.config_id,
        arm=arm.name,
        outcome=TestOutcome.PASS if not failures else TestOutcome.FAIL,
        summary=(
            f"{len(failures)}/{len(compared)} recorded losses failed to match {tolerance_text(tolerance)} tolerance"
        ),
        reason=reason,
        mismatches=failures,
        worst_name=worst.name,
        metrics=metrics,
    )


def loss_range_gate(raw_range: float, worst: str) -> TestTolerance:
    """Twice the measured loss range, raised to the next 0.001, and no smaller than 0.001.

    ``TestTolerance`` accepts a computed gate only when ``absolute`` is that upward-rounded value, so the
    number written into the spec is the one the validator will load.
    """
    computed_value = float(Decimal(str(raw_range)) * Decimal(str(LOSS_RANGE_MULTIPLIER)))
    computed = Decimal(str(computed_value))
    absolute = max(
        GATE_MINIMUM,
        (computed / GATE_QUANTUM).to_integral_value(rounding=ROUND_CEILING) * GATE_QUANTUM,
    )
    return TestTolerance(
        absolute=float(absolute),
        calibration_runs=CALIBRATION_RUNS,
        raw_max_same_tensor_range=float(raw_range),
        multiplier=LOSS_RANGE_MULTIPLIER,
        computed_gate=computed_value,
        rounding_quantum=float(GATE_QUANTUM),
        minimum_gate=float(GATE_MINIMUM),
        worst_tensor=worst,
        status="calibrated",
    )


def measure_undeterministic_loss_range(
    training_config: dict, model_path: str, *, slots: int, vocab_size: int, arms, workdir, attn_implementation: str
):
    """Three 10-step runs per case, determinism off. Returns the widest per-iteration loss range.

    Every run of a case consumes the same ten batches, so the range is the engine's spread and not a
    change of data. The regression that follows keeps best-effort determinism and reads the gate this
    range selects.
    """
    provider = str(training_config.get("model_provider", "huggingface"))
    widest = -1.0
    where = ""
    workdir.mkdir(parents=True, exist_ok=True)
    with GatewayTransport(
        workdir,
        training_config,
        model_path,
        SEED,
        slots=slots,
        attn_implementation=attn_implementation,
        determinism="off",
    ) as transport:
        for arm in arms:
            batches = [
                build_batch(arm.name, arm.global_batch_size, arm.max_seq_len, vocab_size, seed=SEED + iteration)
                for iteration in range(1, CALIBRATION_STEPS + 1)
            ]
            bodies = [pack(batch, model_provider=provider) for batch in batches]
            runs = []
            for run_index in range(CALIBRATION_RUNS):
                with transport.job() as job:
                    losses = {
                        iteration: job.fwd_bwd_step(body, learning_rate=LEARNING_RATE)
                        for iteration, body in enumerate(bodies, start=1)
                    }
                runs.append(losses)
                print(f"calibration {arm.name} run {run_index + 1}/{CALIBRATION_RUNS}: {losses}", flush=True)
            for iteration in range(1, CALIBRATION_STEPS + 1):
                values = [run[iteration] for run in runs]
                span = max(values) - min(values)
                label = f"{arm.name} loss at iteration {iteration}"
                print(f"calibration range {label}: {span}", flush=True)
                if span > widest:
                    widest = span
                    where = label
    if widest < 0:
        raise RuntimeError("checkpoint-resume calibration recorded no loss")
    return widest, where
