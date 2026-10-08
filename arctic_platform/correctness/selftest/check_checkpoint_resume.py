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

"""The checkpoint-and-resume check's verdict, resume payload, and per-iteration batches, without a GPU."""

from __future__ import annotations

import contextlib
import zlib
from types import SimpleNamespace

from arctic_platform.correctness.checks import checkpoint_resume
from arctic_platform.correctness.harness import dss_driver
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.transport import Checkpoint

CONTROL = checkpoint_resume.CHECKPOINT_ITERATION
FINAL = checkpoint_resume.ITERATIONS


class _Gate:
    def tolerance_for(self, test_id: str) -> float:
        assert test_id == checkpoint_resume.TEST_ID
        return 1e-3


def _verdict(reference: dict, target: dict):
    return checkpoint_resume._verdict(
        SimpleNamespace(config_id="example", spec=_Gate()), SimpleNamespace(name="gas1"), reference, target
    )


def test_identical_losses_pass() -> None:
    result = _verdict({CONTROL: 2.5, FINAL: 1.75}, {CONTROL: 2.5, FINAL: 1.75})

    assert result.outcome is Outcome.PASS
    assert result.mismatches == []
    assert result.metrics["max_abs_diff"] == 0.0
    assert result.metrics["stated_criterion_abs"] == 1e-3


def test_a_final_loss_inside_the_tolerance_passes() -> None:
    result = _verdict({CONTROL: 2.5, FINAL: 1.75}, {CONTROL: 2.5, FINAL: 1.7505})

    assert result.outcome is Outcome.PASS
    assert result.metrics[f"target_loss_iteration_{FINAL}"] == 1.7505


def test_a_final_loss_outside_the_tolerance_fails() -> None:
    """The resume verdict: ten steps run from restored weights, optimizer moments and scheduler state."""
    result = _verdict({CONTROL: 2.5, FINAL: 1.75}, {CONTROL: 2.5, FINAL: 1.753})

    assert result.outcome is Outcome.FAIL
    assert [mismatch.name for mismatch in result.mismatches] == [f"loss at iteration {FINAL}"]
    assert result.worst_name == f"loss at iteration {FINAL}"
    assert result.metrics[f"reference_loss_iteration_{FINAL}"] == 1.75


def test_a_control_disagreement_is_explained_rather_than_blamed_on_the_resume() -> None:
    """Before the checkpoint the two runs are one computation, so a difference there does not measure the resume."""
    result = _verdict({CONTROL: 2.5, FINAL: 1.75}, {CONTROL: 2.6, FINAL: 1.75})

    assert result.outcome is Outcome.FAIL
    assert [mismatch.name for mismatch in result.mismatches] == [f"loss at iteration {CONTROL}"]
    # The explanation belongs to a control failure and to nothing else: a run that agrees at the control
    # and diverges afterwards is measuring the resume, and needs no caveat.
    assert str(CONTROL) in result.reason
    assert _verdict({CONTROL: 2.5, FINAL: 1.75}, {CONTROL: 2.5, FINAL: 1.9}).reason == ""


def test_a_resumed_arctic_job_loads_the_saved_path_after_creation(tmp_path, monkeypatch) -> None:
    calls = []

    class Client:
        def load_checkpoint(self, *, path):
            calls.append(("load", path))

    @contextlib.contextmanager
    def running_job(session, payload):
        calls.append(("create", payload["model_name"]))
        yield SimpleNamespace(client=Client())

    monkeypatch.setattr(dss_driver, "running_job", running_job)
    transport = dss_driver.GatewayTransport(tmp_path, {"n_gpus": 1}, "/models/reduced", 1234, slots=1)

    checkpoint = Checkpoint(reference="/checkpoints/global_step10", source_job="/checkpoints/global_step10")
    with transport.job(resume_from=checkpoint):
        pass

    assert calls == [("create", "/models/reduced"), ("load", "/checkpoints/global_step10")]


def test_arctic_checkpoint_reference_is_the_saved_path(tmp_path) -> None:
    class Client:
        def save_checkpoint(self):
            return {"path": "/checkpoints/global_step10", "global_step": 10}

    wrapped = dss_driver.GatewayTrainingJob(
        dss_driver.GatewaySession(tmp_path, 1),
        SimpleNamespace(client=Client()),
    )

    checkpoint = wrapped.save_resumable()

    assert checkpoint.reference == "/checkpoints/global_step10"
    assert checkpoint.source_job == "/checkpoints/global_step10"


def test_every_iteration_gets_its_own_batch() -> None:
    """A run that replayed one batch instead of advancing would reach iteration 20 on the wrong data."""
    batches = checkpoint_resume.iteration_batches(
        SimpleNamespace(vocab_size=128),
        SimpleNamespace(name="gas1", global_batch_size=2, max_seq_len=64),
    )

    assert len(batches) == FINAL
    rows = {tuple(batch.input_ids.flatten().tolist()) for batch in batches}
    assert len(rows) == FINAL


class _Job:
    """A job that answers every request with a loss decided by the request's own bytes.

    Two jobs sent the same body therefore report the same loss, so a run whose halves consumed the
    intended batches agrees at both recorded iterations and one that replayed or skipped a batch does not.
    """

    def __init__(self, resume_from) -> None:
        self.resume_from = resume_from
        self.bodies: list = []
        self.rates: list = []
        self.saved = None

    def fwd_bwd_step(self, body: bytes, *, learning_rate: float) -> float:
        self.bodies.append(body)
        self.rates.append(learning_rate)
        tokens = body["batch"]["input_ids"].detach().cpu().numpy().tobytes()
        return float(zlib.crc32(tokens))

    def save_resumable(self) -> Checkpoint:
        self.saved = Checkpoint(reference="cp_saved", source_job="job-1")
        return self.saved


class _Transport:
    """Records every job a check creates, in the order it created them."""

    def __init__(self) -> None:
        self.jobs: list = []

    def __enter__(self) -> "_Transport":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    @contextlib.contextmanager
    def job(self, *, resume_from=None):
        created = _Job(resume_from)
        self.jobs.append(created)
        yield created


def _context(transport, tmp_path):
    return SimpleNamespace(
        config_id="example",
        workdir=tmp_path,
        vocab_size=128,
        cfg=SimpleNamespace(training={}),
        spec=_Gate(),
        arms=[SimpleNamespace(name="gas1", global_batch_size=2, max_seq_len=64)],
        transport_for=lambda workdir: transport,
    )


def test_the_resumed_job_continues_the_data_order_at_batch_11(tmp_path) -> None:
    """Three jobs: twenty iterations uninterrupted, then ten, then iterations 11 to 20 from the checkpoint."""
    transport = _Transport()
    ctx = _context(transport, tmp_path)

    results = checkpoint_resume.run(ctx)

    bodies = checkpoint_resume._iteration_bodies(ctx, ctx.arms[0])
    uninterrupted, first_half, resumed = transport.jobs

    def body_tokens(values):
        return [body["batch"]["input_ids"].tolist() for body in values]

    assert body_tokens(uninterrupted.bodies) == body_tokens(bodies)
    assert uninterrupted.resume_from is None
    assert body_tokens(first_half.bodies) == body_tokens(bodies[:CONTROL])
    assert body_tokens(resumed.bodies) == body_tokens(bodies[CONTROL:])
    assert len(resumed.bodies) == FINAL - CONTROL
    assert resumed.resume_from == first_half.saved
    assert [result.outcome for result in results] == [Outcome.PASS]


def test_every_iteration_steps_at_the_check_s_own_learning_rate(tmp_path) -> None:
    transport = _Transport()

    checkpoint_resume.run(_context(transport, tmp_path))

    rates = {rate for job in transport.jobs for rate in job.rates}
    assert rates == {checkpoint_resume.LEARNING_RATE}


def test_a_case_that_raises_fails_that_case_alone(tmp_path) -> None:
    class Refusing(_Transport):
        @contextlib.contextmanager
        def job(self, *, resume_from=None):
            raise RuntimeError("no capacity")
            yield  # pragma: no cover - unreachable, present so this stays a generator

    results = checkpoint_resume.run(_context(Refusing(), tmp_path))

    assert [result.outcome for result in results] == [Outcome.FAIL]
    assert results[0].summary == "RuntimeError: no capacity"


def test_loss_range_gate_is_twice_the_range_rounded_up() -> None:
    gate = checkpoint_resume.loss_range_gate(0.1001, "gas1 loss at iteration 10")

    assert gate.calibration_runs == 3
    assert gate.multiplier == 2.0
    assert gate.computed_gate == 0.2002
    assert gate.absolute == 0.201
    assert gate.worst_tensor == "gas1 loss at iteration 10"


def test_a_zero_loss_range_still_meets_the_minimum_gate() -> None:
    gate = checkpoint_resume.loss_range_gate(0.0, "gas1 loss at iteration 1")

    assert gate.absolute == 0.001
