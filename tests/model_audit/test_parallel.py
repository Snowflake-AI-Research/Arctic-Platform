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
"""Expert, sequence and data parallelism: same rows, same weights, one optimizer step per layout.

Every data-parallel replica gets the same packed row, so correctly reduced gradients equal the reference gradients.
Layouts come from the profile's ``tiny.parallel`` list, mirroring the launch layout's kinds at a smaller degree.
"""

from __future__ import annotations

import audit_lib
import pytest
from conftest import require_ok

pytestmark = [pytest.mark.integration]

LOGPROBS = audit_lib.register_check(
    "parallel.logprobs_match_baseline", "EP/SP log-probs match the baseline layout's log-probs at every position"
)
GRADS = audit_lib.register_check(
    "parallel.gradients_match_reference",
    "after the engine's EP/SP/DP reductions every gradient matches the fp32 reference (reduction scale included)",
)
UPDATE = audit_lib.register_check(
    "parallel.update_is_adamw", "under EP/SP/DP every stored update is one AdamW step on the reduced gradient"
)
CLIP_NORM = audit_lib.register_check(
    "parallel.clip_norm_is_reduced_norm",
    "under EP/SP/DP the global gradient norm the engine clips by is the norm of the reduced gradient",
)


def pytest_generate_tests(metafunc):
    if "layout" in metafunc.fixturenames:
        layouts = []
        for profile in audit_lib.all_profiles():
            layouts += [layout for layout in profile.tiny.get("parallel", []) if layout not in layouts]
        ids = [audit_lib.layout_id(lay) for lay in layouts]
        metafunc.parametrize("layout", layouts or [None], ids=ids or ["none"])


@pytest.fixture
def parallel_step(probes, profile, tiny_ckpt, layout):
    if layout is None or layout not in profile.tiny.get("parallel", []):
        pytest.skip(f"{profile.name} does not declare this layout in tiny.parallel")
    parallelism = {k: v for k, v in layout.items() if k in ("expert_parallel", "sequence_parallel")}
    return probes.run(
        "trainer", profile, tiny_ckpt, gpus=audit_lib.layout_world(layout), mode="step", parallelism=parallelism
    )


@pytest.mark.audit_check(LOGPROBS)
def test_logprobs_match_baseline(profile, reference, baseline_step, parallel_step, record):
    ref = require_ok(reference, "reference")
    one, many = require_ok(baseline_step, "baseline step"), require_ok(parallel_step, "parallel step")
    n = ref["labels"].shape[1]
    actual, expected = many["packed_logprobs"][:, :n], one["packed_logprobs"][:, :n]
    cuts = [t["position"] for t in audit_lib.thresholds(profile, audit_lib.tiny_config(profile))]
    got = audit_lib.bucketed_errors(actual, expected, ref["pos"], ref["labels"], cuts)
    floor = audit_lib.bucketed_errors(ref["lowp_logprobs"], ref["fp32_logprobs"], ref["pos"], ref["labels"], cuts)
    record(buckets=got, noise_floor=floor)
    bad = {b: got[b] for b in got if not audit_lib.within_noise(got[b]["max_abs"], floor[b]["max_abs"], profile)}
    assert not bad, f"parallel log-probs beyond the noise floor: {bad}"


@pytest.mark.audit_check(GRADS)
def test_gradients_match_reference(profile, reference, parallel_step, record):
    ref, step = require_ok(reference, "reference"), require_ok(parallel_step, "parallel step")
    audit_lib.assert_gradients_match(profile, step["reduced_grads"], ref, record)


@pytest.mark.audit_check(UPDATE)
def test_update_is_adamw(profile, parallel_step, record):
    audit_lib.assert_update_is_adamw(profile, require_ok(parallel_step, "parallel step"), record)


@pytest.mark.audit_check(CLIP_NORM)
def test_clip_norm_is_reduced_norm(profile, parallel_step, record):
    audit_lib.assert_clip_norm_is_reduced_norm(profile, require_ok(parallel_step, "parallel step"), record)
