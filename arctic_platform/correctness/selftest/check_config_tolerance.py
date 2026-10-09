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

"""Config-specific correctness gates."""

from types import SimpleNamespace

import pytest

from arctic_platform.correctness.checks.fwd_bwd import _run_arm
from arctic_platform.correctness.harness.spec import ArmSpec
from arctic_platform.correctness.harness.spec import ModelSpec
from arctic_platform.correctness.harness.spec import TestSpec as ConfigSpec
from arctic_platform.correctness.harness.spec import TestTolerance as Tolerance


def _context(*, gate: float, dss_norm: float):
    arm = ArmSpec(
        name="fixed",
        global_batch_size=1,
        max_seq_len=8,
        total_tokens=8,
        active_tokens=7,
        dss_microbatches=1,
        pad_fraction=0.125,
    )
    spec = ConfigSpec(
        config_id="unit",
        config_path="unit.config",
        model=ModelSpec(
            source_checkpoint="model",
            num_hidden_layers=1,
            vision_depth=0,
            seed=0,
            param_count=1,
            cache_path="model",
            content_hash="hash",
            sized_against_gib=1.0,
        ),
        arms=[arm],
        attn_implementations=["flash_attention_3"],
        applicable_tests=["single-step-grads"],
        test_tolerances={"single-step-grads": Tolerance(absolute=gate)},
    )
    reference = {"loss": 1.0, "grad_norms": {"weight": 1.0}, "microbatches": 1}
    dss = SimpleNamespace(avg_loss=1.0, grad_norms={"weight": dss_norm}, model_calls=1)
    context = SimpleNamespace(
        config_id="unit",
        spec=spec,
        reference_for=lambda _arm: reference,
        target_for=lambda _arm: dss,
    )
    return context, arm


def test_gradient_norm_verdict_uses_the_frozen_config_gate():
    passing_context, arm = _context(gate=1e-2, dss_norm=1.005)
    failing_context, _ = _context(gate=1e-3, dss_norm=1.005)

    assert _run_arm(passing_context, arm).outcome.value == "pass"
    assert _run_arm(failing_context, arm).outcome.value == "fail"


def test_calibrated_gate_rounds_up_and_has_a_one_millithreshold_floor():
    floor = Tolerance(absolute=1e-3, computed_gate=0.4e-3)
    rounded = Tolerance(absolute=2e-3, computed_gate=1.2e-3)

    assert floor.absolute == 1e-3
    assert rounded.absolute == 2e-3
    with pytest.raises(ValueError, match="does not equal upward-rounded"):
        Tolerance(absolute=1e-3, computed_gate=1.2e-3)
