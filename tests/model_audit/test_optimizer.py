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
"""One optimizer step through the production engine (``initialize_training_engine``) on the baseline layout.

The AdamW check runs at the profile's ``audit_lr`` so the update is visible in the launch dtype, and recomputes the
step from the engine-reduced gradient (whose correctness ``trainer.gradients_match_reference`` checks), clipped by
whichever of the exact or the gradient-dtype clip coefficient the engine matches (``optimizer.clip_scale_exact``
reports which). The landing
check runs at the launch learning rate and measures how much of the update survives in the optimizer's master
weights, which accumulate across steps.
"""

from __future__ import annotations

import audit_lib
import pytest
from conftest import require_ok

pytestmark = [pytest.mark.integration]

DIRECTION = audit_lib.register_check(
    "optimizer.update_is_adamw",
    "every stored weight update is one AdamW step on the gradient the engine reduced, element for element",
)
CLIP_NORM = audit_lib.register_check(
    "optimizer.clip_norm_is_reduced_norm",
    "the global gradient norm the engine clips by is the norm of the gradient it reduced",
)
_FORCED_CLIP = 1e-3
CLIP_SCALE = audit_lib.register_check(
    "optimizer.clip_scale_exact",
    "the stored update follows the exact clip coefficient, not one rounded to the gradient's dtype",
)
LANDS = audit_lib.register_check(
    "optimizer.update_lands_at_launch_lr", "at the launch learning rate the step changes the weights it should"
)
FROZEN = audit_lib.register_check("optimizer.frozen_unchanged", "frozen parameters are not modified by a step")
POLICY = audit_lib.register_check(
    "optimizer.implementation_policy", "the engine runs the optimizer implementation the profile expects"
)


@pytest.fixture(scope="session")
def step_launch_lr(probes, profile, tiny_ckpt):
    lr = profile.optimizer()["lr"]
    return probes.run(
        "trainer",
        profile,
        tiny_ckpt,
        gpus=profile.baseline_gpus,
        mode="step",
        parallelism=profile.tiny_parallelism(),
        lr=lr,
    )


@pytest.fixture(scope="session")
def clipped_step(probes, profile, tiny_ckpt):
    """The baseline step with clipping far below any gradient norm, so the clip coefficient shapes every update."""
    return probes.run(
        "trainer",
        profile,
        tiny_ckpt,
        gpus=profile.baseline_gpus,
        mode="step",
        parallelism=profile.tiny_parallelism(),
        engine_overrides={"gradient_clipping": _FORCED_CLIP},
    )


@pytest.mark.audit_check(DIRECTION)
def test_update_is_adamw(profile, baseline_step, record):
    audit_lib.assert_update_is_adamw(profile, require_ok(baseline_step, "production step"), record)


@pytest.mark.audit_check(CLIP_NORM)
def test_clip_norm_is_reduced_norm(profile, baseline_step, record):
    audit_lib.assert_clip_norm_is_reduced_norm(profile, require_ok(baseline_step, "production step"), record)


@pytest.mark.audit_check(CLIP_SCALE)
def test_clip_scale_exact(request, profile, baseline_step, record):
    launch_clip = require_ok(baseline_step, "production step")["gradient_clipping"]
    if not launch_clip:
        pytest.skip("the launch does not clip gradients")
    step = require_ok(request.getfixturevalue("clipped_step"), "production step with forced clipping")
    coefficients, applied = step["clip_coefficients"], step["clip_applied"]
    record(launch_clipping=launch_clip, forced_clipping=_FORCED_CLIP, coefficients=coefficients, applied=applied)
    assert coefficients["exact"] < 1.0, f"gradient_clipping={_FORCED_CLIP} did not engage: {coefficients}"
    # Whichever coefficient the probe picked has to explain the stored update, or the comparison says nothing.
    audit_lib.assert_update_is_adamw(profile, step, record)
    if coefficients["exact"] != coefficients["grad_dtype"]:
        rel = abs(coefficients["grad_dtype"] / coefficients["exact"] - 1)
        assert applied == "exact", f"the engine scaled the gradient by the clip coefficient rounded by {rel:.3%}"


@pytest.mark.audit_check(LANDS)
def test_update_lands_at_launch_lr(profile, reference, step_launch_lr, record):
    ref, step = require_ok(reference, "reference"), require_ok(step_launch_lr, "production step at launch lr")
    allowed = profile.expect.get("no_grad_params", [])
    update = step.get("master_update") or step["update"]
    moved = expected = 0
    per_tensor = []
    for name, grad in ref["fp32_grads"].items():
        if audit_lib.matches(allowed, name) or name not in update:
            continue
        should = grad != 0
        did = update[name] != 0
        expected += int(should.sum())
        moved += int((did & should).sum())
        if int(should.sum()):
            per_tensor.append((float((did & should).sum()) / float(should.sum()), name))
    per_tensor.sort()
    lost = 1.0 - moved / max(expected, 1)
    record(lr=step["lr"], lost_fraction=lost, least_moved=[(n, round(f, 4)) for f, n in per_tensor[:15]])
    assert (
        lost <= profile.tolerances["max_lost_update_fraction"]
    ), f"at lr={step['lr']} the step leaves {lost:.1%} of master weights with a nonzero gradient unchanged"


@pytest.mark.audit_check(FROZEN)
def test_frozen_unchanged(profile, baseline_step, record):
    step = require_ok(baseline_step, "production step")
    record(frozen=len(step["frozen_params"]), changed=step["frozen_changed"][:20])
    assert not step["frozen_changed"], f"frozen parameters modified by the step: {step['frozen_changed'][:20]}"


@pytest.mark.audit_check(POLICY)
def test_implementation_policy(profile, baseline_step, record):
    step = require_ok(baseline_step, "production step")
    choice = step.get("optimizer_choice") or {}
    record(**choice)
    expected = profile.expect.get("optimizer_implementation")
    if expected is None:
        pytest.skip("profile does not declare expect.optimizer_implementation")
    assert choice.get("implementation") == expected, f"engine chose {choice}, profile expects {expected!r}"
