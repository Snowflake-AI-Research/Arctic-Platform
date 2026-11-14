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
from dataclasses import replace
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
    error_type: Optional[str] = None


@dataclass
class TestSpecEntry:
    test_id: str
    title: str
    criterion: str
    fn: Optional[Callable] = None
    # A test that compares Arctic Platform against the reduced single-GPU Hugging Face reference. The reference
    # executes one training step and nothing else, so such a test has no baseline for a job whose
    # training step consumes rollouts the job generates itself.
    compares_to_reference: bool = True
    # A test whose jobs only the hosted control plane can create, so no gateway a client drives itself
    # can run it. Onboarding refuses such a test rather than reporting a verdict it did not measure.
    requires_hosted_control_plane: bool = False
    workload_modes: tuple[str, ...] = ("sft", "rl")

    @property
    def per_arm(self) -> Optional[Callable]:
        """The single-case entry point, for tests that can judge one case without seeing the others.

        Resolved on access rather than stored at registration. The decorator runs partway down the test
        module, before the attribute at the bottom of that module has been assigned, so a value captured
        at registration is always None.
        """
        return getattr(self.fn, "per_arm", None)


_TEST_DEFINITIONS = {
    test.test_id: test
    for test in (
        TestSpecEntry(
            "single-step-grads",
            "Forward-backward gradient correctness for a single step",
            "per-tensor gradient L2 norms stay within the config-specific reference-repeatability gate",
            workload_modes=("sft",),
        ),
        TestSpecEntry(
            "single-step-optimizer",
            "Adam optimizer-state and update-residual norm correctness for one step",
            "optimizer-state-l2-norm-v1: scalar norm differences stay within the explicitly reviewed reference gate",
            workload_modes=("sft",),
        ),
        TestSpecEntry(
            "checkpoint-resume-loss",
            "Loss agreement between an uninterrupted run and a checkpoint-and-resume run",
            "the loss at the checkpoint iteration and at the final iteration agree between the two runs "
            "within the absolute gate onboarding measured for this config",
            compares_to_reference=False,
        ),
        TestSpecEntry(
            "inference-checkpoint-loss",
            "Validation loss agreement between the training zone and a sampling zone on its checkpoint",
            "the answer-token cross entropy of a held-out GSM8K batch agrees between the training zone that "
            "trained the weights and a sampling zone serving the checkpoint it saved, within 3e-3 on "
            "B200 and 1e-3 on other GPU families",
            compares_to_reference=False,
        ),
        TestSpecEntry(
            "e2e-sft-train-validate",
            "Training and validation loss agreement over a 100-step SFT run replayed in Arctic Platform",
            "the final training loss and the held-out validation loss of a 100-step SFT run agree between Arctic"
            " Platform and the single-GPU Hugging Face reference on the same replayed batches; the whole per-step"
            " trajectory is recorded and reported as a diagnostic",
            workload_modes=("sft",),
        ),
        TestSpecEntry(
            "rl-weight-sync",
            "Weight sync delivers every trained element to the sampling zone unchanged",
            "after one RL step, every tensor the training zone sends reaches every sampling rank with identical "
            "bytes and element count, the trainer sends every element it holds, and every sampler parameter is "
            "written, element for element, with what it received, or, for a frozen table the trainer does not send, "
            "matches the trainer's copy in every checkpoint shard",
            compares_to_reference=False,
            workload_modes=("rl",),
        ),
        TestSpecEntry(
            "rl-router-replay",
            "Router replay: the training zone routes exactly as the sampling zone did",
            "for every generated sample, the expert indices the sampling zone returns, the indices the training "
            "zone receives for it, and the indices every MoE router uses on the forward pass and on the "
            "activation-checkpoint recompute are identical, element for element",
            compares_to_reference=False,
            workload_modes=("rl",),
        ),
    )
}
_REGISTRY: Dict[str, TestSpecEntry] = {}


def test_definitions() -> Dict[str, TestSpecEntry]:
    """Discover test metadata without importing executable checks or their backends."""
    return {identifier: replace(test, fn=None) for identifier, test in {**_TEST_DEFINITIONS, **_REGISTRY}.items()}


def correctness_test(
    test_id: str,
    *,
    title: str | None = None,
    criterion: str | None = None,
    compares_to_reference: bool = True,
    requires_hosted_control_plane: bool = False,
    workload_modes: tuple[str, ...] | None = None,
) -> Callable:
    """Register a correctness test under a stable id used by ``run --test <id>``."""

    def wrap(fn: Callable) -> Callable:
        if test_id in _REGISTRY:
            raise ValueError(f"duplicate correctness test id: {test_id}")
        if title is None and criterion is None:
            _REGISTRY[test_id] = replace(_TEST_DEFINITIONS[test_id], fn=fn)
        else:
            if title is None or criterion is None:
                raise ValueError("custom correctness tests require both title and criterion")
            _REGISTRY[test_id] = TestSpecEntry(
                test_id=test_id,
                title=title,
                criterion=criterion,
                fn=fn,
                compares_to_reference=compares_to_reference,
                requires_hosted_control_plane=requires_hosted_control_plane,
                workload_modes=workload_modes or (("sft",) if compares_to_reference else ("sft", "rl")),
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


def workload_inapplicable_reason(test: TestSpecEntry, config) -> str | None:
    """Select workload-compatible checks before model or backend initialization."""
    if not getattr(config, "has_training", True):
        return "correctness checks require a training sub-job; this config is inference-only"
    if config.is_rl and test.compares_to_reference:
        return RL_INAPPLICABLE_REASON
    mode = "rl" if config.is_rl else "sft"
    if mode not in test.workload_modes:
        return f"{test.test_id} requires a {'reinforcement-learning' if mode == 'sft' else 'supervised'} config"
    return None


def onboarding_inapplicable_reason(test: TestSpecEntry, config) -> str | None:
    reason = workload_inapplicable_reason(test, config)
    if reason:
        return reason
    if test.requires_hosted_control_plane:
        return "requires the hosted control plane; run the hosted command instead of local onboarding"
    return None
