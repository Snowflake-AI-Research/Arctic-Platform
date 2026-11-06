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
"""Sampling sub-job against the training sub-job and the local reference, through ``generate`` prompt log-probs.

Trainer/sampler agreement is what RL's importance ratio assumes. Weight sync is checked end to end: after a training
step the service syncs the trainer's weights into the sampler, which must then match the fp32 reference run on the
trainer's own export of those weights as closely as it matched the reference on the initial weights.
"""

from __future__ import annotations

from pathlib import Path

import audit_lib
import pytest
import torch
from conftest import ISOLATION_GAP
from conftest import launch_tp
from conftest import require_ok

pytestmark = [pytest.mark.integration, pytest.mark.vllm]

MATCHES_TRAINER = audit_lib.register_check(
    "sampler.matches_trainer", "sampler log-probs match trainer log-probs at every position (RL ratio is 1 at step 0)"
)
MATCHES_REFERENCE = audit_lib.register_check(
    "sampler.matches_reference", "sampler log-probs match the fp32 reference at every position"
)
REPEATABLE = audit_lib.register_check("sampler.repeatable", "the same request scored twice gives identical log-probs")
BATCH = audit_lib.register_check(
    "sampler.batch_invariant", "a request's log-probs do not depend on the other requests in its batch"
)
SYNC = audit_lib.register_check(
    "sampler.weight_sync_matches_reference",
    "after a weight sync the sampler scores like the fp32 reference on the trainer's export of the synced weights",
)
TP = audit_lib.register_check(
    "sampler.tensor_parallel_matches_reference",
    "sampler log-probs at TP degrees below launch match the fp32 reference",
)


def _packed(rows, per_row) -> torch.Tensor:
    return torch.cat([lp for lp in per_row]).unsqueeze(0)[:, : sum(len(r) for r in rows)]


def _compare(profile, ref, actual_per_row, expected_packed) -> tuple[dict, dict, dict]:
    """Errors and noise floors per position bucket and per row, and the entries beyond the floor."""
    rows = ref["rows"]
    n = sum(len(r) for r in rows)
    actual = _packed(rows, actual_per_row)
    pos, labels = ref["pos"][:, :n], ref["labels"][:, :n]
    cuts = audit_lib.position_cuts(profile)
    got = audit_lib.bucketed_errors(actual, expected_packed[:, :n], pos, labels, cuts)
    floor = audit_lib.bucketed_errors(ref["lowp_logprobs"][:, :n], ref["fp32_logprobs"][:, :n], pos, labels, cuts)
    # Each row is its own request, so a request-level fault in a short row must not be diluted by long rows.
    for i, sl in enumerate(audit_lib.row_slices(rows)):
        valid = labels[:, sl] != -100
        key = f"row{i}(n={len(rows[i])})"
        got[key] = audit_lib.logprob_errors(actual[:, sl], expected_packed[:, sl], valid)
        floor[key] = audit_lib.logprob_errors(ref["lowp_logprobs"][:, sl], ref["fp32_logprobs"][:, sl], valid)
        got[key]["k3"] = None
    bad = {
        b: (got[b]["mean_abs"], floor[b]["mean_abs"], got[b]["k3"])
        for b in got
        if not audit_lib.within_noise(got[b]["mean_abs"], floor[b]["mean_abs"], profile)
    }
    return got, floor, bad


def _against(profile, ref, actual_per_row, expected_packed, record, label):
    got, floor, bad = _compare(profile, ref, actual_per_row, expected_packed)
    record(**{f"{label}_buckets": got, "noise_floor_buckets": floor})
    assert not bad, f"{label}: buckets beyond the noise floor (mean, floor mean, k3): {bad}"


@pytest.mark.audit_check(MATCHES_TRAINER)
def test_matches_trainer(profile, reference, trainer_forward, sampler_run, record):
    ref, fwd = require_ok(reference, "local reference"), require_ok(trainer_forward, "training job")
    sampler = require_ok(sampler_run, "sampler")
    assert (
        "isolated_error" not in fwd
    ), f"a forward of one row alone failed: {fwd.get('isolated_error', '').strip().splitlines()[-1:]}"
    if "isolated_logprobs" not in fwd:
        pytest.skip(ISOLATION_GAP)
    trainer_packed = torch.cat(fwd["isolated_logprobs"]).unsqueeze(0)
    _against(profile, ref, sampler["alone"], trainer_packed, record, "sampler_vs_trainer")


@pytest.mark.audit_check(MATCHES_REFERENCE)
def test_matches_reference(profile, reference, sampler_run, record):
    ref, sampler = require_ok(reference, "local reference"), require_ok(sampler_run, "sampler")
    _against(profile, ref, sampler["alone"], ref["fp32_logprobs"], record, "sampler_vs_fp32")


def _row_errors(a_rows, b_rows) -> list[dict]:
    return [audit_lib.logprob_errors(a, b) for a, b in zip(a_rows, b_rows)]


@pytest.mark.audit_check(REPEATABLE)
def test_repeatable(profile, sampler_run, record):
    sampler = require_ok(sampler_run, "sampler")
    errors = _row_errors(sampler["alone_repeat"], sampler["alone"])
    record(per_row=errors)
    atol = float(profile.tolerances.get("sampler_repeat_atol", 0.0))
    assert all(e["max_abs"] <= atol for e in errors), f"repeat differs: {errors}"


@pytest.mark.audit_check(BATCH)
def test_batch_invariant(profile, reference, sampler_run, record):
    ref, sampler = require_ok(reference, "local reference"), require_ok(sampler_run, "sampler")
    errors = _row_errors(sampler["batched"], sampler["alone"])
    floors = []
    for sl in audit_lib.row_slices(ref["rows"]):
        floors.append(audit_lib.logprob_errors(ref["lowp_logprobs"][0, sl][:-1], ref["fp32_logprobs"][0, sl][:-1]))
    record(per_row=errors, noise_floor_per_row=floors)
    bad = [(e, f) for e, f in zip(errors, floors) if not audit_lib.within_noise(e["max_abs"], f["max_abs"], profile)]
    assert not bad, f"batched scores beyond the noise floor (error, floor): {bad}"


def pytest_generate_tests(metafunc):
    if "sync_format" in metafunc.fixturenames:
        formats = []
        for profile in audit_lib.all_profiles():
            formats += [f for f in profile.sync_formats if f not in formats]
        metafunc.parametrize("sync_format", formats)


@pytest.mark.audit_check(SYNC)
def test_weight_sync_matches_reference(profile, probes, sampler_run, sync_format, record):
    if sync_format not in profile.sync_formats:
        pytest.skip(f"{profile.name} does not sync the {sync_format!r} format")
    sampler = require_ok(sampler_run, "sampler")
    assert f"sync_{sync_format}_error" not in sampler, sampler[f"sync_{sync_format}_error"]
    if f"after_sync_{sync_format}" not in sampler:
        pytest.skip("the job has no training sub-job to sync from")
    state = sampler["exports"][sampler[f"sync_{sync_format}_state"]]
    synced = require_ok(probes.reference(profile, Path(state)), "local reference on the synced weights")
    record(rpc=sampler.get(f"sync_{sync_format}_rpc"))
    order = profile.sync_formats
    previous = order[order.index(sync_format) - 1] if order.index(sync_format) else None
    before = sampler[f"after_sync_{previous}"] if previous else sampler["alone"]
    unsynced, _, moved = _compare(profile, synced, before, synced["fp32_logprobs"])
    record(before_sync_buckets=unsynced)
    assert moved, "the synced weights score like the sampler's previous ones, so a sync that does nothing would pass"
    _against(profile, synced, sampler[f"after_sync_{sync_format}"], synced["fp32_logprobs"], record, "synced_vs_fp32")


@pytest.mark.audit_check(TP)
@pytest.mark.parametrize("tp", [1, 2, 4], ids=lambda tp: f"tp{tp}")
def test_tensor_parallel_matches_reference(profile, client, reference, tp, record):
    """Smaller TP degrees than launch, each against the fp32 reference so a failure names the broken degree."""
    launch = launch_tp(profile)
    if tp >= launch:
        pytest.skip(f"TP={tp} is not below the launch TP={launch} (sampler.matches_reference covers launch)")
    ref = require_ok(reference, "local reference")
    run = require_ok(client.get(f"sampler-tp{tp}"), f"sampling job at TP={tp}")
    _against(profile, ref, run["sampler"]["alone"], ref["fp32_logprobs"], record, f"tp{tp}_vs_reference")
