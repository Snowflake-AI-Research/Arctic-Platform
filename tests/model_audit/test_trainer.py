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
"""Trainer on the baseline layout: the production model against the transformers fp32 reference on the same tiny
checkpoint.

Gates are relative to the reference's own launch-dtype run (the noise floor): the production model may differ from
fp32 by at most ``noise_factor`` times what plain dtype rounding does to the same math, per position bucket.
"""

from __future__ import annotations

import audit_lib
import pytest
import torch
from conftest import require_ok

pytestmark = [pytest.mark.integration]

BUILD = audit_lib.register_check("trainer.build", "the production loader builds the tiny model at the launch settings")
LOGPROBS = audit_lib.register_check(
    "trainer.logprobs_match_reference", "trainer per-token log-probs match the fp32 reference at every position"
)
DETERMINISTIC = audit_lib.register_check("trainer.deterministic", "the same forward twice gives identical log-probs")
PACKED = audit_lib.register_check(
    "trainer.packed_rows_isolated", "a row's log-probs do not depend on the other rows packed with it"
)
ORDER = audit_lib.register_check("trainer.row_order_invariant", "a row's log-probs do not depend on its pack offset")
SHORT = audit_lib.register_check(
    "trainer.short_rows_supported", "a pack ending in a sample of any length from 1 token up runs and stays finite"
)
GRAD_COMPLETE = audit_lib.register_check(
    "trainer.gradients_complete", "every trainable parameter gets a finite, nonzero gradient unless declared"
)
GRAD_MATCH = audit_lib.register_check(
    "trainer.gradients_match_reference",
    "every HF-named gradient the optimizer receives, after the engine's reductions, matches the fp32 reference",
)
HEAD_GRAD = audit_lib.register_check(
    "trainer.head_gradient_matches_unfused",
    "the output head's input gradient, contracted with its input over tokens, is as close to fp64 as an unfused head "
    "in the launch dtype",
)
EXPORT = audit_lib.register_check(
    "trainer.export_round_trip", "the HF weight export reproduces the loaded checkpoint, every tensor, bit for bit"
)
CONFIG_RESAVE = audit_lib.register_check(
    "trainer.config_survives_transformers_resave",
    "the config class the production loader resolves reads a config re-saved by the pinned transformers (as any HF "
    "tool writing a checkpoint of this model does) the same as the shipped one",
)


def _cuts(profile, tiny_ckpt) -> list[int]:
    config = audit_lib.probe_inputs(profile, str(tiny_ckpt))[0]
    return [t["position"] for t in audit_lib.thresholds(profile, config)]


def _assert_buckets(profile, actual, expected, floor, pos, labels, cuts, record, label):
    got = audit_lib.bucketed_errors(actual, expected, pos, labels, cuts)
    ref = audit_lib.bucketed_errors(floor, expected, pos, labels, cuts)
    record(**{f"{label}_buckets": got, "noise_floor_buckets": ref})
    bad = {
        b: (got[b]["mean_abs"], ref[b]["mean_abs"], got[b]["max_abs"], ref[b]["max_abs"])
        for b in got
        if not audit_lib.within_noise(got[b]["mean_abs"], ref[b]["mean_abs"], profile)
        or not audit_lib.within_noise(got[b]["max_abs"], ref[b]["max_abs"], profile)
    }
    assert not bad, f"{label}: buckets beyond the noise floor (mean, floor mean, max, floor max): {bad}"


@pytest.mark.audit_check(BUILD)
def test_build(profile, trainer_forward, record):
    result = require_ok(trainer_forward, "production model build/forward")
    record(loader=result["loader"], spec=result["spec"])
    assert result["loader"] == profile.expect["loader"]


@pytest.mark.audit_check(LOGPROBS)
def test_logprobs_match_reference(profile, tiny_ckpt, reference, trainer_forward, record):
    ref = require_ok(reference, "reference")
    fwd = require_ok(trainer_forward, "production forward")
    _assert_buckets(
        profile,
        fwd["packed_logprobs"],
        ref["fp32_logprobs"],
        ref["lowp_logprobs"],
        ref["pos"],
        ref["labels"],
        _cuts(profile, tiny_ckpt),
        record,
        "trainer",
    )


@pytest.mark.audit_check(DETERMINISTIC)
def test_deterministic(profile, reference, trainer_forward, record):
    ref, fwd = require_ok(reference, "reference"), require_ok(trainer_forward, "production forward")
    mask = ref["labels"] != -100
    err = audit_lib.logprob_errors(fwd["repeat_logprobs"], fwd["packed_logprobs"], mask)
    record(**err)
    assert err["max_abs"] <= float(profile.tolerances.get("determinism_atol", 0.0)), err


def _per_row(profile, rows, packed, other, floor, fp32, record, label):
    errors, floors = [], []
    for sl, row_other in zip(audit_lib.row_slices(rows), other):
        n = sl.stop - sl.start - 1
        errors.append(audit_lib.logprob_errors(packed[0, sl][:n], row_other[:n]))
        floors.append(audit_lib.logprob_errors(floor[0, sl][:n], fp32[0, sl][:n]))
    record(**{f"{label}_per_row": errors, "noise_floor_per_row": floors})
    bad = [
        (len(r), e, f)
        for r, e, f in zip(rows, errors, floors)
        if not audit_lib.within_noise(e["max_abs"], f["max_abs"], profile)
    ]
    assert not bad, f"{label}: rows beyond the noise floor (row length, error, floor): {bad}"


@pytest.mark.audit_check(PACKED)
def test_packed_rows_isolated(profile, reference, trainer_forward, record):
    ref, fwd = require_ok(reference, "reference"), require_ok(trainer_forward, "production forward")
    _per_row(
        profile,
        ref["rows"],
        fwd["packed_logprobs"],
        fwd["isolated_logprobs"],
        ref["lowp_logprobs"],
        ref["fp32_logprobs"],
        record,
        "packed_vs_isolated",
    )


@pytest.mark.audit_check(ORDER)
def test_row_order_invariant(profile, reference, trainer_forward, record):
    ref, fwd = require_ok(reference, "reference"), require_ok(trainer_forward, "production forward")
    rows = ref["rows"]
    reversed_rows = list(reversed(rows))
    reversed_slices = dict(zip(range(len(rows) - 1, -1, -1), audit_lib.row_slices(reversed_rows)))
    other = [fwd["reversed_logprobs"][0, reversed_slices[i]] for i in range(len(rows))]
    _per_row(
        profile,
        rows,
        fwd["packed_logprobs"],
        other,
        ref["lowp_logprobs"],
        ref["fp32_logprobs"],
        record,
        "original_vs_reversed",
    )


@pytest.mark.audit_check(SHORT)
def test_short_rows_supported(profile, trainer_forward, record):
    fwd = require_ok(trainer_forward, "production forward")
    record(short_rows=fwd["short_rows"])
    bad = {length: status for length, status in fwd["short_rows"].items() if status != "ok"}
    assert not bad, f"packs ending in a sample of these lengths fail: {bad}"


@pytest.mark.audit_check(GRAD_COMPLETE)
def test_gradients_complete(profile, trainer_forward, record):
    fwd = require_ok(trainer_forward, "production forward")
    allowed = profile.expect.get("no_grad_params", [])
    undeclared = [n for n in fwd["no_grad_params"] if audit_lib.matches(allowed, n) is None]
    record(no_grad=fwd["no_grad_params"][:50], nonfinite=fwd["nonfinite_grad_params"][:50], undeclared=undeclared[:50])
    assert not fwd["nonfinite_grad_params"], f"non-finite gradients: {fwd['nonfinite_grad_params'][:20]}"
    assert not undeclared, f"trainable parameters without a gradient: {undeclared[:20]}"


@pytest.mark.audit_check(GRAD_MATCH)
def test_gradients_match_reference(profile, reference, baseline_step, record):
    ref, step = require_ok(reference, "reference"), require_ok(baseline_step, "production step")
    audit_lib.assert_gradients_match(profile, step["reduced_grads"], ref, record)


@pytest.mark.audit_check(HEAD_GRAD)
def test_head_gradient_matches_unfused(profile, trainer_forward, record):
    head = require_ok(trainer_forward, "production forward")["head_contraction"]
    err = audit_lib.relative_l2(head["production"], head["fp64"])
    floor = audit_lib.relative_l2(head["lowp"], head["fp64"])
    record(rel_l2=err, floor=floor)
    assert audit_lib.within_noise(err, floor, profile), f"rel_l2={err:.4g} vs unfused {floor:.4g}"


@pytest.mark.audit_check(EXPORT)
def test_export_round_trip(profile, tiny_ckpt, trainer_forward, record):
    fwd = require_ok(trainer_forward, "production forward")
    exported = fwd["exported"]
    saved = {k: v.float() for k, v in audit_lib.read_checkpoint_tensors(tiny_ckpt).items()}
    allowed = profile.expect.get("no_grad_params", []) + profile.expect.get("export_ignore", [])
    missing = sorted(n for n in saved if n not in exported and audit_lib.matches(allowed, n) is None)
    extra = sorted(n for n in exported if n not in saved)
    changed = []
    for name in sorted(set(saved) & set(exported)):
        a, e = exported[name], saved[name]
        if a.shape != e.shape:
            changed.append((name, f"shape {tuple(a.shape)} != {tuple(e.shape)}"))
        elif not torch.equal(a, e):
            changed.append((name, f"max_abs={float((a - e).abs().max()):.4g}"))
    record(missing=missing[:30], extra=extra[:30], changed=changed[:30], tensors=len(saved))
    assert not missing, f"checkpoint tensors the export drops: {missing[:20]}"
    assert not extra, f"exported tensors absent from the checkpoint: {extra[:20]}"
    assert not changed, f"exported tensors that differ from the checkpoint: {changed[:20]}"


@pytest.mark.audit_check(CONFIG_RESAVE)
def test_config_survives_transformers_resave(profile, trainer_forward, record):
    resave = require_ok(trainer_forward, "production forward")["config_resave"]
    record(**resave)
    if "skipped" in resave:
        pytest.skip(resave["skipped"])
    assert not resave[
        "changed"
    ], f"{resave['resolved_class']} reads a re-saved config differently: {resave['changed']}"
