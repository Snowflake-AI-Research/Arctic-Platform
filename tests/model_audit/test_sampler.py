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
"""Sampler (vLLM with the platform patches and weight-sync extension) against the trainer and the reference.

Trainer/sampler agreement is what RL's importance ratio assumes. Weight sync is checked end to end: checkpoint B's
production export is synced into an engine started on checkpoint A, and must score like an engine started on B.
"""

from __future__ import annotations

import audit_lib
import pytest
import torch
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
    "sampler.weight_sync_matches_fresh_load",
    "after a production weight sync the sampler scores exactly like a sampler started on the synced weights",
)
TP = audit_lib.register_check(
    "sampler.tensor_parallel_matches_reference",
    "sampler log-probs at TP degrees below launch match the fp32 reference",
)


def _packed(rows, per_row) -> torch.Tensor:
    return torch.cat([lp for lp in per_row]).unsqueeze(0)[:, : sum(len(r) for r in rows)]


def _against(profile, ref, actual_per_row, expected_packed, record, label):
    rows = ref["rows"]
    n = sum(len(r) for r in rows)
    actual = _packed(rows, actual_per_row)
    pos, labels = ref["pos"][:, :n], ref["labels"][:, :n]
    cuts = [t["position"] for t in audit_lib.thresholds(profile, audit_lib.tiny_config(profile))]
    got = audit_lib.bucketed_errors(actual, expected_packed[:, :n], pos, labels, cuts)
    floor = audit_lib.bucketed_errors(ref["lowp_logprobs"][:, :n], ref["fp32_logprobs"][:, :n], pos, labels, cuts)
    # Each row is its own request, so a request-level fault in a short row must not be diluted by long rows.
    for i, sl in enumerate(audit_lib.row_slices(rows)):
        valid = labels[:, sl] != -100
        key = f"row{i}(n={len(rows[i])})"
        got[key] = audit_lib.logprob_errors(actual[:, sl], expected_packed[:, sl], valid)
        floor[key] = audit_lib.logprob_errors(ref["lowp_logprobs"][:, sl], ref["fp32_logprobs"][:, sl], valid)
        got[key]["k3"] = None
    record(**{f"{label}_buckets": got, "noise_floor_buckets": floor})
    bad = {
        b: (got[b]["mean_abs"], floor[b]["mean_abs"], got[b]["k3"])
        for b in got
        if not audit_lib.within_noise(got[b]["mean_abs"], floor[b]["mean_abs"], profile)
    }
    assert not bad, f"{label}: buckets beyond the noise floor (mean, floor mean, k3): {bad}"


@pytest.mark.audit_check(MATCHES_TRAINER)
def test_matches_trainer(profile, reference, trainer_forward, sampler_run, record):
    ref, fwd = require_ok(reference, "reference"), require_ok(trainer_forward, "production forward")
    sampler = require_ok(sampler_run, "sampler")
    trainer_packed = torch.cat(fwd["isolated_logprobs"]).unsqueeze(0)
    _against(profile, ref, sampler["alone"], trainer_packed, record, "sampler_vs_trainer")


@pytest.mark.audit_check(MATCHES_REFERENCE)
def test_matches_reference(profile, reference, sampler_run, record):
    ref, sampler = require_ok(reference, "reference"), require_ok(sampler_run, "sampler")
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
    ref, sampler = require_ok(reference, "reference"), require_ok(sampler_run, "sampler")
    errors = _row_errors(sampler["batched"], sampler["alone"])
    floors = []
    for sl in audit_lib.row_slices(ref["rows"]):
        floors.append(audit_lib.logprob_errors(ref["lowp_logprobs"][0, sl][:-1], ref["fp32_logprobs"][0, sl][:-1]))
    record(per_row=errors, noise_floor_per_row=floors)
    bad = [(e, f) for e, f in zip(errors, floors) if not audit_lib.within_noise(e["max_abs"], f["max_abs"], profile)]
    assert not bad, f"batched scores beyond the noise floor (error, floor): {bad}"


def pytest_generate_tests(metafunc):
    if "sync_format" in metafunc.fixturenames:
        metafunc.parametrize("sync_format", ["hf", "vllm"])
    if "sync_transport" in metafunc.fixturenames:
        metafunc.parametrize("sync_transport", ["file", "stream"])


@pytest.mark.audit_check(SYNC)
def test_weight_sync_matches_fresh_load(
    probes, profile, tiny_ckpt, export_b, sampler_fresh_b, sync_format, sync_transport, record
):
    exported = require_ok(export_b, "export of B")
    if sync_format not in exported["paths"]:
        pytest.skip(f"the production model does not export the {sync_format!r} weight format")
    # One sampler per format: a second sync into the same engine would start from B and hide a no-op.
    synced = require_ok(
        probes.run(
            "sampler",
            profile,
            tiny_ckpt,
            gpus=profile.sampler_gpus,
            sync=[f"{sync_format}={exported['paths'][sync_format]}"],
            sync_transport=sync_transport,
        ),
        "sampler synced to B",
    )
    if f"sync_{sync_format}_unsupported" in synced:
        pytest.skip(f"coverage gap: {synced[f'sync_{sync_format}_unsupported']}")
    fresh = require_ok(sampler_fresh_b, "sampler started on B")
    assert f"sync_{sync_format}_error" not in synced, synced[f"sync_{sync_format}_error"]
    errors = _row_errors(synced[f"after_sync_{sync_format}"], fresh["alone"])
    repeat_noise = _row_errors(fresh["alone_repeat"], fresh["alone"])
    unsynced = _row_errors(synced["alone"], fresh["alone"])
    record(per_row=errors, repeat_noise=repeat_noise, before_sync=unsynced, rpc=synced.get(f"sync_{sync_format}_rpc"))
    bad = [
        (e, r)
        for e, r in zip(errors, repeat_noise)
        if e["max_abs"] > 2 * r["max_abs"] + profile.tolerances["noise_slack"]
    ]
    assert not bad, f"synced sampler differs from a fresh load beyond its own repeat noise (error, noise): {bad}"


@pytest.mark.audit_check(TP)
@pytest.mark.parametrize("tp", [1, 2, 4], ids=lambda tp: f"tp{tp}")
def test_tensor_parallel_matches_reference(profile, probes, tiny_ckpt, reference, tp, record):
    """Smaller TP degrees than launch, each against the fp32 reference so a failure names the broken degree."""
    launch = int(profile.sampler_args(tiny=False).get("tensor_parallel_size", 1))
    if tp >= launch:
        pytest.skip(f"TP={tp} is not below the launch TP={launch} (sampler.matches_reference covers launch)")
    if tp > torch.cuda.device_count():
        pytest.skip(f"TP={tp} needs {tp} GPUs")
    ref = require_ok(reference, "reference")
    run = require_ok(
        probes.run("sampler", profile, tiny_ckpt, gpus=tp, engine_args={"tensor_parallel_size": tp}),
        f"sampler TP={tp}",
    )
    _against(profile, ref, run["alone"], ref["fp32_logprobs"], record, f"tp{tp}_vs_reference")
