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
"""One optimizer step of the training sub-job, read back through the client's exports.

The step's reply gives the global gradient norm the engine clipped by and every parameter's gradient norm. The
resumable export gives the fp32 master weights and Adam moments: after one step the first moment is
``(1 - beta1)`` times the clipped gradient the optimizer consumed. The AdamW check runs at the profile's ``audit_lr``
so the update is visible in the launch dtype, and recomputes the stored weights-only update from that consumed
gradient. The landing check runs at the launch learning rate and measures how much of the update survives in the
master weights, which accumulate across steps.
"""

from __future__ import annotations

import audit_lib
import pytest
from conftest import require_ok

pytestmark = [pytest.mark.integration]

DIRECTION = audit_lib.register_check(
    "optimizer.update_is_adamw",
    "every stored weight update is one AdamW step on the gradient the optimizer consumed, element for element",
)
CLIP_NORM = audit_lib.register_check(
    "optimizer.clip_norm_is_reduced_norm",
    "the global gradient norm the engine clips by is the norm of the gradient it reduced",
)
CLIP_SCALE = audit_lib.register_check(
    "optimizer.clip_scale_exact",
    "the optimizer consumes the gradient scaled by the exact clip coefficient, not one rounded to a low dtype",
)
LANDS = audit_lib.register_check(
    "optimizer.update_lands_at_launch_lr", "at the launch learning rate the step changes the weights it should"
)
FROZEN = audit_lib.register_check("optimizer.frozen_unchanged", "frozen parameters are not modified by a step")
POLICY = audit_lib.register_check(
    "optimizer.implementation_policy", "the engine runs the optimizer implementation the profile expects"
)


@pytest.mark.audit_check(DIRECTION)
def test_update_is_adamw(profile, baseline_step, record):
    audit_lib.assert_update_is_adamw(profile, require_ok(baseline_step, "training job step"), record)


@pytest.mark.audit_check(CLIP_NORM)
def test_clip_norm_is_reduced_norm(profile, baseline_step, record):
    step = require_ok(baseline_step, "training job step")
    if step["reduced_grad_norm"] is None:
        pytest.skip("coverage gap: the step reported no per-parameter gradient norms")
    audit_lib.assert_clip_norm_is_reduced_norm(profile, step, record)


@pytest.mark.audit_check(CLIP_SCALE)
def test_clip_scale_exact(request, profile, baseline_step, record):
    step = require_ok(baseline_step, "training job step")
    if not step["gradient_clipping"]:
        pytest.skip("the launch does not clip gradients")
    if step["clip_coefficients"]["exact"] >= 1.0:
        step = require_ok(request.getfixturevalue("forced_clip_step"), "training job step with forced clipping")
    coefficients, measured = step["clip_coefficients"], step["clip_applied_measured"]
    if measured is None:
        pytest.skip("coverage gap: the step reported no per-parameter gradient norms")
    rel = {kind: abs(measured / c - 1) for kind, c in coefficients.items()}
    record(gradient_clipping=step["gradient_clipping"], coefficients=coefficients, measured=measured, rel=rel)
    assert coefficients["exact"] < 1.0, f"clipping did not engage: {coefficients}"
    assert rel["exact"] <= profile.tolerances["grad_norm_rel"], (
        f"the optimizer consumed the gradient scaled by {measured:.6g}, not the clip coefficient"
        f" {coefficients['exact']:.6g} (the dtype-rounded one is {coefficients['grad_dtype']:.6g})"
    )


@pytest.mark.audit_check(LANDS)
def test_update_lands_at_launch_lr(profile, step_launch_lr, record):
    step = require_ok(step_launch_lr, "training job step at launch lr")
    allowed = profile.expect.get("no_grad_params", [])
    moved = expected = 0
    per_tensor = []
    for name, (should, did) in step["landing"].items():
        if audit_lib.matches(allowed, name):
            continue
        expected += should
        moved += did
        if should:
            per_tensor.append((did / should, name))
    per_tensor.sort()
    lost = 1.0 - moved / max(expected, 1)
    record(lr=step["lr"], lost_fraction=lost, least_moved=[(n, round(f, 4)) for f, n in per_tensor[:15]])
    assert (
        lost <= profile.tolerances["max_lost_update_fraction"]
    ), f"at lr={step['lr']} the step leaves {lost:.1%} of master weights with a nonzero gradient unchanged"


@pytest.mark.audit_check(FROZEN)
def test_frozen_unchanged(profile, baseline_step, record):
    step = require_ok(baseline_step, "training job step")
    record(frozen=len(step["frozen_params"]), changed=step["frozen_changed"][:20])
    assert not step["frozen_changed"], f"frozen parameters modified by the step: {step['frozen_changed'][:20]}"


@pytest.mark.audit_check(POLICY)
def test_implementation_policy(profile):
    pytest.skip("coverage gap: the service does not report which optimizer implementation it runs")
