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

from __future__ import annotations

import io
import time

from arctic_platform.correctness.harness.console import activity
from arctic_platform.correctness.harness.console import arm_verdict
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.registry import TestResult as Result


class TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_activity_rotates_and_erases_the_temporary_line() -> None:
    stream = TTYBuffer()
    with activity("Running reference gas1", stream=stream, interval=0.001):
        deadline = time.monotonic() + 0.2
        while "\r  Running reference gas1 /" not in stream.getvalue() and time.monotonic() < deadline:
            time.sleep(0.001)

    output = stream.getvalue()
    assert "\r  Running reference gas1 |" in output
    assert "\r  Running reference gas1 /" in output
    assert output.endswith("\r")
    assert " " * len("  Running reference gas1 |") in output


def test_activity_prints_one_progress_line_when_output_is_not_interactive() -> None:
    stream = io.StringIO()
    with activity("Running Arctic Platform gas1", stream=stream):
        pass

    assert stream.getvalue() == "  Running Arctic Platform gas1 ...\n"


def test_passing_gate_reports_passed_tensors() -> None:
    result = Result(
        test_id="single-step-grads",
        config_id="example",
        outcome=Outcome.PASS,
        metrics={
            "tensors_compared": 47.0,
            "tensors_over_stated_criterion": 0.0,
            "stated_criterion_abs": 0.013,
        },
    )

    rendered = arm_verdict(result, None)

    assert "PASSED 1.3e-02 tolerance" in rendered
    assert "47/47" in rendered
    assert "FAILED 1.3e-02 tolerance" not in rendered


def test_failing_gate_reports_failed_tensors() -> None:
    result = Result(
        test_id="single-step-grads",
        config_id="example",
        outcome=Outcome.FAIL,
        summary="two mismatches",
        metrics={
            "tensors_compared": 47.0,
            "tensors_over_stated_criterion": 2.0,
            "stated_criterion_abs": 0.013,
        },
    )

    rendered = arm_verdict(result, None)

    assert "FAILED 1.3e-02 tolerance" in rendered
    assert "2/47" in rendered


def test_optimizer_gate_names_residual_delta_and_reports_passed_values() -> None:
    result = Result(
        test_id="single-step-optimizer",
        config_id="example",
        outcome=Outcome.PASS,
        worst_name="layer.weight::exp_avg",
        metrics={
            "optimizer_values_compared": 141.0,
            "optimizer_values_over_criterion": 0.0,
            "max_optimizer_delta_norm": 0.004,
            "stated_criterion_abs": 0.012,
        },
    )

    rendered = arm_verdict(result, None)

    assert "optimizer moment and update-residual delta L2 norms" in rendered
    assert "PASSED 1.2e-02 tolerance" in rendered
    assert "141/141" in rendered
    assert "layer.weight::exp_avg" in rendered
