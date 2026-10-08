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

"""Test registration and the three outcomes a test-by-config cell can take."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional


class TestOutcome(enum.Enum):
    """Pass, fail, and the case where a test does not apply to a config at all.

    Folding INAPPLICABLE into PASS would make a green matrix claim coverage it does not have. Anything that
    prevents a comparison from completing is a failure, not a third kind of not-passing.
    """

    PASS = "pass"
    FAIL = "fail"
    INAPPLICABLE = "inapplicable"


@dataclass
class Mismatch:
    """One compared quantity, worst-first in the report.

    ``ratio`` localizes the defect: uniform across every tensor means a normalization or accumulation
    error, while a handful of outliers points at a single module.
    """

    name: str
    target: float
    reference: float
    difference: Optional[float] = None

    @property
    def abs_diff(self) -> float:
        return self.difference if self.difference is not None else abs(self.target - self.reference)

    @property
    def ratio(self) -> float:
        return self.target / self.reference if self.reference != 0 else float("inf")


@dataclass
class TestResult:
    test_id: str
    config_id: str
    outcome: TestOutcome
    arm: Optional[str] = None
    # How many GPUs Arctic Platform was placed on. A config is measured at the width it declares and again at a
    # multiple of it, so two results for one test and case differ only in this field.
    gpus: Optional[int] = None
    summary: str = ""
    reason: str = ""
    # Only the comparisons that failed. The widest-disagreeing parameter is carried separately because it
    # is worth reporting on a pass too, and listing passing parameters to surface it buries the failures.
    mismatches: List[Mismatch] = field(default_factory=list)
    worst_name: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TestSpecEntry:
    test_id: str
    title: str
    criterion: str
    fn: Callable
    # A test that compares Arctic Platform against the reduced single-GPU Hugging Face reference. The reference
    # executes one training step and nothing else, so such a test has no baseline for a job whose
    # training step consumes rollouts the job generates itself.
    compares_to_reference: bool = True
    # A test whose jobs only the hosted control plane can create, so no gateway a client drives itself
    # can run it. Onboarding refuses such a test rather than reporting a verdict it did not measure.
    requires_hosted_control_plane: bool = False

    @property
    def per_arm(self) -> Optional[Callable]:
        """The single-case entry point, for tests that can judge one case without seeing the others.

        Resolved on access rather than stored at registration. The decorator runs partway down the test
        module, before the attribute at the bottom of that module has been assigned, so a value captured
        at registration is always None.
        """
        return getattr(self.fn, "per_arm", None)


_REGISTRY: Dict[str, TestSpecEntry] = {}


def correctness_test(
    test_id: str,
    *,
    title: str,
    criterion: str,
    compares_to_reference: bool = True,
    requires_hosted_control_plane: bool = False,
) -> Callable:
    """Register a correctness test under a stable id used by ``run --test <id>``."""

    def wrap(fn: Callable) -> Callable:
        if test_id in _REGISTRY:
            raise ValueError(f"duplicate correctness test id: {test_id}")
        _REGISTRY[test_id] = TestSpecEntry(
            test_id=test_id,
            title=title,
            criterion=criterion,
            fn=fn,
            compares_to_reference=compares_to_reference,
            requires_hosted_control_plane=requires_hosted_control_plane,
        )
        return fn

    return wrap


def registered_tests() -> Dict[str, TestSpecEntry]:
    from .. import checks  # noqa: F401  -- import for side-effect registration

    return dict(_REGISTRY)


RL_INAPPLICABLE_REASON = (
    "reinforcement-learning config: the single-GPU Hugging Face reference runs one supervised training "
    "step and cannot generate rollouts, so this test has no baseline to compare against"
)


def reference_tests() -> Dict[str, TestSpecEntry]:
    """The registered tests that need the single-GPU reference, and so do not apply to an RL config."""
    return {test_id: test for test_id, test in registered_tests().items() if test.compares_to_reference}
