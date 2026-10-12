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

"""A projection's rows must not depend on the row count of the call, and the pad must only claim what it does.

Padding a short call up to a row count where cuBLASLt's kernel selection is already pinned, then slicing the
wanted rows back, is exact -- zero rows contribute nothing to a matmul. It removes a row-count dependence only
where that dependence comes from the split-K reduction over the output, which the output size bounds. Whether
it does is decided by the alignment of the contraction, and this file is where that is held:

* an aligned contraction at or above the contraction floor is exposed below its row target and bit-identical
  to a tall call once padded to it,
* an aligned contraction below the floor is already bit-identical at every row count and is left alone,
* an unaligned contraction is exposed at row counts padding cannot reach, so the module refuses it rather than
  padding it and reporting coverage.

Each of the three fails for its own reason. The first fails if the defect is absent, which would make the rest
vacuous. The second fails if the pad perturbs a call that was already exact. The third fails if a row pad
would in fact have covered an unaligned width, which would make the refusal gratuitous.

``CUBLAS_WORKSPACE_CONFIG`` is read when cuBLAS creates its handle, so every measurement runs in a child
process that started with the environment under test; setting it in-process would report the default's
behaviour while claiming to test the configured one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from arctic_platform.model.implementations.debug.row_invariant_projection import CONTRACTION_ALIGNMENT
from arctic_platform.model.implementations.debug.row_invariant_projection import DRIVER_DEFAULT_WORKSPACE
from arctic_platform.model.implementations.debug.row_invariant_projection import MIN_DEPENDENT_CONTRACTION
from arctic_platform.model.implementations.debug.row_invariant_projection import MIN_PADDABLE_OUTPUT_WIDTH
from arctic_platform.model.implementations.debug.row_invariant_projection import WORKSPACE_BYTES_PER_OUTPUT_ELEMENT
from arctic_platform.model.implementations.debug.row_invariant_projection import RowInvariancePadCannotCover
from arctic_platform.model.implementations.debug.row_invariant_projection import RowInvarianceUnverified
from arctic_platform.model.implementations.debug.row_invariant_projection import apply_row_invariant_projections
from arctic_platform.model.implementations.debug.row_invariant_projection import contraction_is_paddable
from arctic_platform.model.implementations.debug.row_invariant_projection import effective_workspace_config
from arctic_platform.model.implementations.debug.row_invariant_projection import invariant_row_target
from arctic_platform.model.implementations.debug.row_invariant_projection import is_row_count_dependent
from arctic_platform.model.implementations.debug.row_invariant_projection import maybe_apply_row_invariant_projections
from arctic_platform.model.implementations.debug.row_invariant_projection import row_invariant_linear
from arctic_platform.model.implementations.debug.row_invariant_projection import row_invariant_projections_requested
from arctic_platform.model.implementations.debug.row_invariant_projection import verify_targets_cover_the_runtime
from arctic_platform.model.implementations.debug.row_invariant_projection import workspace_bytes

# The workspace the tables below were measured at. Stated here rather than read from a configuration so that
# changing a configured value fails a test instead of silently moving what these assert.
MEASURED_WORKSPACE = ":16:8"

# The call every other row count is compared against: a token's own row inside a call tall enough that its
# kernel is pinned at every width here.
REFERENCE_ROWS = 4096

# Swept per shape. Below a shape's target the unpadded path is exposed and the padded path must not be; from
# the target up the two must agree bit for bit.
ROW_COUNTS = (1, 2, 3, 4, 8, 16, 32, 256)

# ``(K, N)``, aligned contractions at or above the floor. Three output widths spanning three row targets --
# 16 rows, 4 rows and 16 rows again at a longer contraction -- so a target right for one width and wrong for
# another cannot pass.
ALIGNED_DEPENDENT_SHAPES = ((4096, 1024), (4096, 4096), (12288, 1024))

# An aligned contraction below the floor. Invariant at every row count with no pad at all, which is the half
# of the predicate that says what must *not* be padded.
ALIGNED_INVARIANT_SHAPES = ((2560, 4096),)

# Unaligned contractions in the band where the contraction floor claims there is nothing to do. Both follow
# their row count there, and neither is reached by padding rows.
UNALIGNED_SHAPES = ((1028, 1024), (2500, 1024))

# Measured on H200 under CUDA 13 in bfloat16 at ``MEASURED_WORKSPACE``: the smallest row count whose result
# matches a 4096-row call, by output width, identical for aligned K 2568 through 12288.
MEASURED_ROW_TARGETS = {256: 64, 1024: 16, 2048: 8, 4096: 4}

# The boundary before the power-of-two margin is applied: ``ceil(workspace_bytes / 8 / N)``. Asserted
# separately from the targets so the rule is pinned rather than only its rounding -- at N 1536 and 3072 the
# two differ, which is where a target that happened to match by rounding would hide a wrong rule.
EXPECTED_BOUNDARY_ROWS = {256: 64, 1024: 16, 1536: 11, 2048: 8, 3072: 6, 4096: 4, 8192: 2}

# Aligned contraction widths measured invariant at every row count from 1 to 1024 at ``N`` 1024, and the
# smallest measured to move. The floor has to fall between them, so a floor above 2568 leaves a width exposed
# and a floor at or below 2560 pads widths that need nothing.
LARGEST_INVARIANT_ALIGNED_CONTRACTION = 2560
SMALLEST_DEPENDENT_ALIGNED_CONTRACTION = 2568

_CHILD = textwrap.dedent(
    """
    import json
    import sys

    import torch
    import torch.nn.functional as F

    from arctic_platform.model.implementations.debug.row_invariant_projection import (
        RowInvariancePadCannotCover,
        invariant_row_target,
        row_invariant_linear,
    )

    groups = json.loads(sys.argv[1])
    reference_rows = int(sys.argv[2])
    row_counts = json.loads(sys.argv[3])

    report = []
    for group, shapes in groups.items():
        for width, out_features in shapes:
            generator = torch.Generator(device="cuda").manual_seed(31_000 + width * 1000 + out_features)
            x = torch.randn(reference_rows, width, generator=generator, device="cuda", dtype=torch.bfloat16)
            # The model's initializer scale, so the kernel sees the magnitudes the layer gives it.
            weight = torch.randn(
                out_features, width, generator=generator, device="cuda", dtype=torch.float32
            ).mul_(0.02).to(torch.bfloat16)
            target = invariant_row_target(out_features)
            with torch.no_grad():
                reference = F.linear(x, weight)
                floor = float((reference - F.linear(x, weight)).abs().max())
                scale = float(reference.abs().max())
                for rows in row_counts:
                    if rows > reference_rows:
                        continue
                    # Production calls a projection on a ``[1, tokens, K]`` activation; the leading batch
                    # dimension is rows of the product to cuBLAS, so the call under test carries it.
                    short = x[:rows].contiguous().unsqueeze(0)
                    plain = F.linear(short, weight).squeeze(0)
                    entry = {
                        "group": group,
                        "width": width,
                        "out_features": out_features,
                        "rows": rows,
                        "target": target,
                        "floor": floor,
                        "scale": scale,
                        "plain_delta": float((plain - reference[:rows]).abs().max()),
                        "refused": False,
                        "padded_delta": None,
                        "padded_vs_plain": None,
                    }
                    try:
                        padded = row_invariant_linear(short, weight, None).squeeze(0)
                        entry["padded_delta"] = float((padded - reference[:rows]).abs().max())
                        entry["padded_vs_plain"] = float((padded - plain).abs().max())
                    except RowInvariancePadCannotCover:
                        entry["refused"] = True
                    # The row pad applied by hand, whatever the module decided. For a contraction the module
                    # refuses, this is what padding would have delivered, and it is the measurement the
                    # refusal has to be justified against rather than asserted.
                    flat = short.reshape(rows, width)
                    if rows < target:
                        buffer = torch.cat([flat, flat.new_zeros((target - rows, width))], dim=0)
                        by_hand = F.linear(buffer, weight)[:rows]
                    else:
                        by_hand = F.linear(flat, weight)
                    entry["by_hand_delta"] = float((by_hand - reference[:rows]).abs().max())
                    report.append(entry)
    print(json.dumps({"device": torch.cuda.get_device_name(0), "cases": report}))
    """
)


def _run_child(environment: dict[str, str]) -> dict:
    groups = {
        "aligned_dependent": ALIGNED_DEPENDENT_SHAPES,
        "aligned_invariant": ALIGNED_INVARIANT_SHAPES,
        "unaligned": UNALIGNED_SHAPES,
    }
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(groups), str(REFERENCE_ROWS), json.dumps(ROW_COUNTS)],
        env={**os.environ, **environment},
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert (
        completed.returncode == 0
    ), f"the measurement child failed ({completed.returncode}):\n{completed.stdout}\n{completed.stderr}"
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_the_predicate_selects_a_projection_by_alignment_and_not_only_by_size():
    """What the pad covers is decided by the contraction's alignment first and its length second.

    A contraction that is not a multiple of ``CONTRACTION_ALIGNMENT`` is row-count dependent at lengths far
    below the floor, and padding rows does not reach it, so it must not be selected as covered. A contraction
    that is aligned and below the floor needs nothing, so it must not be padded either. Both directions are
    asserted here, without a GPU, so a change to either constant fails on the arithmetic before it fails on
    the hardware.
    """
    for in_features in (1028, 1540, 2052, 2500):
        assert not contraction_is_paddable(in_features), (
            f"K={in_features} is not a multiple of {CONTRACTION_ALIGNMENT}, so padding rows leaves its "
            "row-count dependence in place and it must not be treated as coverable"
        )
        with torch.device("meta"):
            module = torch.nn.Linear(in_features, 4096, bias=False)
        assert not is_row_count_dependent(module), (
            f"K={in_features} N=4096 is selected for padding, which would spend the pad and remove none of "
            "the deviation; the predicate has to refuse it instead"
        )

    for in_features in (1024, 1032, 1088, 2048, LARGEST_INVARIANT_ALIGNED_CONTRACTION):
        assert contraction_is_paddable(in_features)
        with torch.device("meta"):
            module = torch.nn.Linear(in_features, 4096, bias=False)
        assert not is_row_count_dependent(module), (
            f"K={in_features} is invariant at every row count measured, so padding it adds arithmetic where "
            "there is no dependence to remove"
        )

    for in_features in (SMALLEST_DEPENDENT_ALIGNED_CONTRACTION, 2592, 4096, 12288):
        assert contraction_is_paddable(in_features)
        with torch.device("meta"):
            module = torch.nn.Linear(in_features, 4096, bias=False)
        assert is_row_count_dependent(module), (
            f"K={in_features} follows its row count below its target and padding rows removes it exactly, so "
            "the predicate has to select it"
        )


def test_the_contraction_floor_sits_between_the_two_widths_that_bracket_it():
    """The floor has to fall in the gap the measurement leaves, and there is only one such gap.

    Every multiple of ``CONTRACTION_ALIGNMENT`` to 2560 is bit-identical to a 4096-row call at every row count
    swept; every one from 2568 up moves. A floor above 2568 therefore leaves 2568, 2576 and 2584 exposed with
    nothing patching them, and a floor at or below 2560 pads widths with no dependence to remove. Asserted
    apart from the alignment condition so the two defects cannot mask each other -- a run that fails both
    reports both.
    """
    assert LARGEST_INVARIANT_ALIGNED_CONTRACTION < MIN_DEPENDENT_CONTRACTION, (
        f"the floor is {MIN_DEPENDENT_CONTRACTION}, at or below the largest aligned contraction measured "
        f"invariant at every row count ({LARGEST_INVARIANT_ALIGNED_CONTRACTION}), so it pads widths that "
        "need nothing"
    )
    assert MIN_DEPENDENT_CONTRACTION <= SMALLEST_DEPENDENT_ALIGNED_CONTRACTION, (
        f"the floor is {MIN_DEPENDENT_CONTRACTION}, above the smallest aligned contraction measured to "
        f"follow its row count ({SMALLEST_DEPENDENT_ALIGNED_CONTRACTION}), so every aligned width from "
        f"{SMALLEST_DEPENDENT_ALIGNED_CONTRACTION} up to it is exposed with nothing patching it"
    )
    assert MIN_DEPENDENT_CONTRACTION % CONTRACTION_ALIGNMENT == 0, (
        "the floor is only ever compared against contractions the alignment condition has already admitted, "
        "so a floor that is not itself aligned describes a width that can never reach it"
    )
    for in_features in range(SMALLEST_DEPENDENT_ALIGNED_CONTRACTION, 2592 + 1, CONTRACTION_ALIGNMENT):
        with torch.device("meta"):
            module = torch.nn.Linear(in_features, 1024, bias=False)
        assert is_row_count_dependent(module), (
            f"K={in_features} N=1024 follows its row count -- measured up to 1.562e-02 against a 0.000e+00 "
            "floor -- and the predicate does not select it"
        )


def test_a_model_with_an_unaligned_contraction_is_refused_rather_than_half_covered():
    """A projection padding cannot cover must stop the model being built, not be quietly skipped.

    Skipping is the failure with no symptom: the patch count and the log would both describe a covered model
    while one projection kept a row-count dependence larger than the ones removed around it. The refusal names
    the module and the shape, because the remedy is to align the width rather than to widen a threshold.
    """
    with torch.device("meta"):
        model = torch.nn.Sequential(
            torch.nn.Linear(4096, 4096, bias=False),
            torch.nn.Linear(2500, 4096, bias=False),
        )
    with pytest.raises(RowInvariancePadCannotCover) as raised:
        apply_row_invariant_projections(model, verify=False)
    message = str(raised.value)
    assert (
        "2500" in message and str(CONTRACTION_ALIGNMENT) in message
    ), f"the refusal has to name the offending contraction and the quantum it misses; got: {message}"
    assert (
        "forward" not in model[0].__dict__
    ), "the refusal has to come before anything is patched, so a model is either wholly covered or wholly untouched"

    with torch.device("meta"):
        aligned = torch.nn.Sequential(torch.nn.Linear(4096, 4096, bias=False))
    assert apply_row_invariant_projections(aligned, verify=False) == 1

    # The same refusal reaches a direct call, so a caller that never goes through the model walk cannot
    # silently receive a pad that does nothing.
    with pytest.raises(RowInvariancePadCannotCover):
        row_invariant_linear(torch.zeros(1, 2500), torch.zeros(8, 2500))


def test_the_row_target_rule_reproduces_every_measured_width():
    """The rule has to derive the measured table, not approximate it.

    Stated as a table of measurements rather than as the rule restated, so an off-by-one in the rule or a
    changed constant fails here without a GPU. Both halves are asserted: the boundary the condition gives, and
    the power-of-two target the margin gives above it.
    """
    elements = workspace_bytes(MEASURED_WORKSPACE) // WORKSPACE_BYTES_PER_OUTPUT_ELEMENT
    for out_features, expected_boundary in EXPECTED_BOUNDARY_ROWS.items():
        boundary = -(-elements // out_features)
        assert boundary == expected_boundary, (
            f"N={out_features}: the condition rows*N < {elements} puts the boundary at {boundary} rows where "
            f"{expected_boundary} was measured and tested as a prediction"
        )
    for out_features, expected in MEASURED_ROW_TARGETS.items():
        target = invariant_row_target(out_features, MEASURED_WORKSPACE)
        assert target == expected, (
            f"N={out_features}: the rule gives {target} rows where {expected} is the smallest row count "
            f"measured to match a {REFERENCE_ROWS}-row call"
        )


def test_the_target_is_derived_from_the_workspace_and_moves_with_it(monkeypatch):
    """The target must follow the workspace the process has, not a table that happens to match today's value.

    Padding to a target measured against one workspace while a different one is in force pads to the wrong row
    count, so the derivation is asserted to be live -- a workspace twice the size doubles the target -- and
    both branches of the resolution are covered: a configured value, and an absent variable, which is the
    larger workspace and therefore the larger target.

    The environment is established here rather than inherited, because nothing in this repository exports
    ``CUBLAS_WORKSPACE_CONFIG`` for a pytest process, and reading it without setting it would assert on
    whichever shell launched the run.
    """
    assert workspace_bytes(":16:8") == 16 * 1024 * 8
    assert workspace_bytes(":16:8:32:4") == 16 * 1024 * 8 + 32 * 1024 * 4
    for bad in ("", ":16", ":16:8:32"):
        with pytest.raises(ValueError):
            workspace_bytes(bad)

    assert invariant_row_target(1024, ":8:8") == 8
    assert invariant_row_target(1024, ":16:8") == 16
    assert invariant_row_target(1024, ":32:8") == 32
    assert invariant_row_target(4096, ":32:8") == 8

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", MEASURED_WORKSPACE)
    assert effective_workspace_config() == MEASURED_WORKSPACE
    for out_features, expected in MEASURED_ROW_TARGETS.items():
        assert invariant_row_target(out_features) == expected

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    assert effective_workspace_config() == DRIVER_DEFAULT_WORKSPACE, (
        "with the variable absent the resolution must give the driver's own workspace, which is the larger "
        f"one; got {effective_workspace_config()!r} against {DRIVER_DEFAULT_WORKSPACE!r}"
    )
    assert effective_workspace_config({}) == DRIVER_DEFAULT_WORKSPACE
    assert effective_workspace_config({"CUBLAS_WORKSPACE_CONFIG": ""}) == DRIVER_DEFAULT_WORKSPACE
    assert workspace_bytes(DRIVER_DEFAULT_WORKSPACE) > workspace_bytes(MEASURED_WORKSPACE), (
        "the driver default must be the larger workspace; if it is not, the reasoning that an absent variable "
        "widens the exposed band no longer holds and every target here has to be re-measured"
    )
    for out_features in MEASURED_ROW_TARGETS:
        absent = invariant_row_target(out_features)
        assert absent > MEASURED_ROW_TARGETS[out_features], (
            f"N={out_features}: an absent workspace gives a target of {absent} rows against "
            f"{MEASURED_ROW_TARGETS[out_features]} at {MEASURED_WORKSPACE}; it has to be larger, because the "
            "exposed band is larger. A target that shrinks here under-pads every process that does not set "
            "the variable, which is the failure mode with no symptom"
        )


def test_a_runtime_that_disagrees_with_the_environment_is_caught():
    """A string in the environment is a claim about cuBLAS, not proof, so the claim gets checked.

    cuBLAS reads ``CUBLAS_WORKSPACE_CONFIG`` once when it creates its handle. Setting it afterwards, or in a
    parent that does not pass it on, leaves the variable saying one thing and the kernels doing another -- and
    the pad then pads to a row count that is itself still row-count dependent, with no symptom at all. The
    check is one-sided: a device demonstrating *less* workspace than assumed only means the targets are
    generous, while more means they are too small, and only the former may pass silently.
    """
    from arctic_platform.model.implementations.debug import row_invariant_projection

    original = row_invariant_projection.measure_workspace_bytes
    try:
        row_invariant_projection.measure_workspace_bytes = lambda device, dtype=torch.bfloat16: (
            workspace_bytes(DRIVER_DEFAULT_WORKSPACE)
        )
        with pytest.raises(RowInvarianceUnverified) as raised:
            verify_targets_cover_the_runtime(MEASURED_WORKSPACE, torch.device("cpu"))
        message = str(raised.value)
        assert MEASURED_WORKSPACE in message and "CUBLAS_WORKSPACE_CONFIG" in message, (
            "the failure has to name the workspace assumed and the variable to set, since the fix is to place "
            f"it in the spawning environment; got: {message}"
        )

        row_invariant_projection.measure_workspace_bytes = lambda device, dtype=torch.bfloat16: 1024
        assert verify_targets_cover_the_runtime(MEASURED_WORKSPACE, torch.device("cpu")) == 1024
    finally:
        row_invariant_projection.measure_workspace_bytes = original


def test_a_linear_subclass_that_overrides_forward_is_not_patched():
    """The pad reimplements ``torch.nn.Linear.forward``, so it may only replace that.

    The output heads subclass :class:`torch.nn.Linear` and override ``forward`` to take labels, a temperature
    and an action mask, and to return a loss rather than logits. They match every shape condition the pad
    tests, so nothing but the override check keeps the pad off them, and replacing one substitutes a plain
    projection for a chunked log-probability head -- loud on a path that passes extra arguments, and silent on
    one that does not.
    """
    from arctic_platform.model.implementations.qwen35.models.layers.lm_head import FusedOutputLinear

    with torch.device("meta"):
        head = FusedOutputLinear(4096, 4096, chunk_size=1024)
        plain = torch.nn.Linear(4096, 4096, bias=False)

    assert not is_row_count_dependent(head), (
        f"{type(head).__name__} overrides forward and would be replaced by a plain projection; it matches on "
        f"shape (K={head.in_features}, N={head.out_features}), so only the override check excludes it"
    )
    assert is_row_count_dependent(plain), (
        "the override check must not exclude an ordinary projection of the same shape, or the pad stops "
        "covering the modules it exists for"
    )

    model = torch.nn.Sequential(head, plain)
    assert apply_row_invariant_projections(model, verify=False) == 1
    # Patching installs ``forward`` in the instance dictionary, so its presence is what "patched" means;
    # ``module.forward`` itself builds a fresh bound method on every access and compares equal to nothing.
    assert "forward" not in head.__dict__ and "forward" in plain.__dict__


def test_a_narrow_output_is_left_exposed_rather_than_padded_to_an_unreachable_row_count():
    """A width whose target no call can reach is excluded by width, and the exclusion is on the width.

    The target rises as the output narrows, so at one column it reaches the whole workspace bound. A ceiling
    on the row count instead would only hold while the workspace is small: with the variable absent, targets
    that are affordable today grow past any fixed ceiling, and the ceiling would then exclude a projection
    whose scores decide a categorical choice.
    """
    with torch.device("meta"):
        one_column = torch.nn.Linear(4096, 1, bias=False)
        narrowest = torch.nn.Linear(4096, MIN_PADDABLE_OUTPUT_WIDTH, bias=False)
    assert not is_row_count_dependent(one_column), (
        f"a single-column output needs {invariant_row_target(1, MEASURED_WORKSPACE)} rows at "
        f"{MEASURED_WORKSPACE} and {invariant_row_target(1, DRIVER_DEFAULT_WORKSPACE)} with the variable "
        "absent, neither of which a call can carry"
    )
    assert is_row_count_dependent(narrowest)

    with torch.device("meta"):
        model = torch.nn.Sequential(torch.nn.Linear(4096, 1, bias=False))
    assert apply_row_invariant_projections(model, verify=False) == 0


@pytest.mark.integration
def test_a_short_projection_returns_the_tall_calls_own_rows():
    """One, two and three rows must return what a 4096-row call returns for them, wherever padding can reach.

    Four assertions, each of which fails for its own reason. The reproduction fails if an aligned projection
    below its target is no longer exposed -- the defect having gone away, which would make the rest vacuous.
    The fix fails if a padded call disagrees with the tall call. The no-op fails if the pad perturbs a row
    count that was already exact. And the refusal evidence fails if a row pad would in fact have covered an
    unaligned contraction, which is what would make refusing one gratuitous rather than accurate.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device: this is a property of the GEMM kernel the device selects")

    measured = _run_child({"CUBLAS_WORKSPACE_CONFIG": MEASURED_WORKSPACE})
    expected = len(ROW_COUNTS) * (
        len(ALIGNED_DEPENDENT_SHAPES) + len(ALIGNED_INVARIANT_SHAPES) + len(UNALIGNED_SHAPES)
    )
    assert len(measured["cases"]) == expected

    exposed_unpadded: list[str] = []
    unaligned_unhelped: list[str] = []
    failures: list[str] = []
    for case in measured["cases"]:
        where = f"K={case['width']} N={case['out_features']} rows={case['rows']}"
        if case["floor"] != 0.0:
            failures.append(
                f"{where}: the reference call repeated disagreed with itself by {case['floor']:.3e}, so this "
                "shape cannot measure a row-count effect"
            )

        if case["group"] == "aligned_dependent":
            if case["refused"]:
                failures.append(f"{where}: the pad refused an aligned contraction it is able to cover")
                continue
            if case["padded_delta"] != 0.0:
                failures.append(
                    f"{where}: the padded call disagrees with the same rows inside a {REFERENCE_ROWS}-row "
                    f"call by {case['padded_delta']:.3e} (target {case['target']} rows, floor "
                    f"{case['floor']:.3e}, output scale {case['scale']:.3e})"
                )
            if case["rows"] >= case["target"] and case["padded_vs_plain"] != 0.0:
                failures.append(
                    f"{where}: at or above its target of {case['target']} rows the pad must be inert, but it "
                    f"moved the result by {case['padded_vs_plain']:.3e} against the unpadded call"
                )
            if case["rows"] < case["target"] and case["plain_delta"] != 0.0:
                exposed_unpadded.append(f"{where} d={case['plain_delta']:.3e}")

        if case["group"] == "aligned_invariant":
            if case["refused"]:
                failures.append(
                    f"{where}: the pad refused an aligned contraction, which it is supposed to leave "
                    "untouched rather than reject"
                )
            elif case["padded_vs_plain"] != 0.0:
                failures.append(
                    f"{where}: an aligned contraction below the floor must be left alone, but the padded "
                    f"path differs from the unpadded one by {case['padded_vs_plain']:.3e}"
                )
            if case["plain_delta"] != 0.0:
                failures.append(
                    f"{where}: an aligned contraction below {MIN_DEPENDENT_CONTRACTION} is expected to be "
                    f"invariant unpadded, but moved {case['plain_delta']:.3e}; the contraction floor no "
                    "longer describes this device"
                )

        if case["group"] == "unaligned":
            if not case["refused"]:
                failures.append(
                    f"{where}: the pad covered an unaligned contraction instead of refusing it, reporting a "
                    "coverage that padding rows does not deliver"
                )
            if case["rows"] < case["target"] and case["by_hand_delta"] != 0.0:
                unaligned_unhelped.append(
                    f"{where} unpadded={case['plain_delta']:.3e} padded_to_{case['target']}"
                    f"={case['by_hand_delta']:.3e}"
                )

    print(
        f"MEASURED {measured['device']} CUBLAS_WORKSPACE_CONFIG={MEASURED_WORKSPACE} "
        + "; ".join(
            f"{case['group']} K={case['width']} N={case['out_features']} r={case['rows']} "
            f"plain={case['plain_delta']:.3e} by_hand={case['by_hand_delta']:.3e} "
            f"padded={'refused' if case['refused'] else format(case['padded_delta'], '.3e')}"
            for case in measured["cases"]
        )
    )
    assert exposed_unpadded, (
        "no unpadded call below its row target disagreed with the tall call, so this device does not show the "
        "dependence the padding removes and the rest of this test proves nothing; the row targets or the "
        f"contraction floor ({MIN_DEPENDENT_CONTRACTION}) no longer describe it"
    )
    assert unaligned_unhelped, (
        "padding rows made every unaligned contraction agree with the tall call, so refusing them costs "
        "coverage the pad could have delivered and the alignment condition should be removed rather than "
        f"kept. Contractions checked: {[shape[0] for shape in UNALIGNED_SHAPES]}"
    )
    assert (
        not failures
    ), "a projection's answer for a token still depends on how many tokens shared the call:\n  " + "\n  ".join(
        failures
    )


# Seeds ordered by how few one-row scorings each needs to reach a selection flip on H200 at hidden 4096 with
# 256 experts: 199, 1460 and 2789 respectively. The order is a budget rather than a requirement -- any seed
# that flips satisfies the assertion -- so a device whose kernel selection differs spends more of the budget
# instead of failing.
ROUTER_SEEDS = (9, 4, 8)

# The width at which the router's contraction reaches the pad's floor and the exposure exists at all. The
# configuration default of 2048 is below it, and a 2048-wide contraction into 256 columns is invariant at
# every row count by measurement, so it cannot exercise this property.
ROUTER_HIDDEN = 4096

# Tokens scored per seed, and the packed reference's height. Far above the gate's 64-row target, so the
# reference is itself invariant.
ROUTER_TOKENS = 4096

_ROUTER_CHILD = textwrap.dedent(
    """
    import json
    import sys

    import torch

    from arctic_platform.model.implementations.debug.row_invariant_projection import (
        apply_row_invariant_projections,
    )
    from arctic_platform.model.implementations.qwen35.models.layers.moe import TokenChoiceTopKRouter
    from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeConfig,
    )

    case = json.loads(sys.argv[1])
    hidden, tokens = case["hidden"], case["tokens"]
    config = Qwen3_5MoeConfig(hidden_size=hidden)
    router = TokenChoiceTopKRouter(
        dim=hidden,
        num_experts=config.num_experts,
        top_k=config.num_experts_per_tok,
        score_func="sigmoid",
        route_norm=True,
        route_scale=1.0,
    ).to(device="cuda", dtype=torch.bfloat16)
    patched = apply_row_invariant_projections(router, verify=False) if case["pad"] else 0

    report, stop = [], False
    for seed in case["seeds"]:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        weight = (
            torch.randn(hidden, config.num_experts, generator=generator, device="cuda",
                        dtype=torch.float32)
            * config.initializer_range
        ).to(torch.bfloat16)
        with torch.no_grad():
            router.gate.weight.copy_(weight.t())
        window = torch.randn(tokens, hidden, generator=generator, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            packed = router.gate(window)
            floor = float((packed.double() - router.gate(window).double()).abs().max())
            _, packed_indices, _ = router(window)
        packed_sets = [frozenset(row) for row in packed_indices.tolist()]
        for rows in case["row_counts"]:
            flips, deviation, scored, first = 0, 0.0, 0, None
            for start in range(0, tokens, rows):
                block = window[start : start + rows]
                with torch.no_grad():
                    logits = router.gate(block)
                    _, indices, _ = router(block)
                deviation = max(deviation, float(
                    (logits.double() - packed[start : start + block.shape[0]].double()).abs().max()
                ))
                for offset, row in enumerate(indices.tolist()):
                    scored += 1
                    if frozenset(row) != packed_sets[start + offset]:
                        flips += 1
                        if first is None:
                            first = scored
                if case["stop_at_first_flip"] and flips:
                    stop = True
                    break
            report.append({"seed": seed, "rows": rows, "flips": flips, "scored": scored,
                           "first": first, "deviation": deviation, "floor": floor,
                           "scale": float(packed.abs().max())})
            if stop:
                break
        if stop:
            break
    print(json.dumps({"device": torch.cuda.get_device_name(0), "patched": patched,
                      "experts": config.num_experts, "cases": report}))
    """
)


def _run_router_child(*, pad: bool, row_counts, stop_at_first_flip: bool) -> dict:
    case = {
        "hidden": ROUTER_HIDDEN,
        "tokens": ROUTER_TOKENS,
        "seeds": list(ROUTER_SEEDS),
        "row_counts": list(row_counts),
        "pad": pad,
        "stop_at_first_flip": stop_at_first_flip,
    }
    completed = subprocess.run(
        [sys.executable, "-c", _ROUTER_CHILD, json.dumps(case)],
        env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": MEASURED_WORKSPACE},
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert (
        completed.returncode == 0
    ), f"the measurement child failed ({completed.returncode}):\n{completed.stdout}\n{completed.stderr}"
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.integration
def test_an_unpadded_router_sends_a_token_to_different_experts_below_the_boundary():
    """Without the pad, a one-row router call routes some token to a different expert set than a packed call.

    This is the consequence that no tolerance can express. Every other effect of the row-count dependence is a
    small change in a value, bounded by the rounding of the type; this one is categorical, because ``topk``
    over the gate scores turns a one-bit move into a different set of weights, and the difference in that
    token's output is then bounded by the experts that were swapped.

    It is also the half of the pair that fails if the defect is absent. A device on which a one-row call
    agreed with a packed one would make the invariance test below pass while asserting nothing, and this is
    what says so. It searches rather than constructs, because a flip needs a near-tie at the ``top_k``
    boundary, and reports the scorings spent so a changed rate shows up as a widening budget before it shows
    up as a flake.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device: this is a property of the GEMM kernel the device selects")

    measured = _run_router_child(pad=False, row_counts=(1,), stop_at_first_flip=True)
    spent = sum(case["scored"] for case in measured["cases"])
    found = [case for case in measured["cases"] if case["flips"]]
    print(
        f"UNPADDED {measured['device']} CUBLAS_WORKSPACE_CONFIG={MEASURED_WORKSPACE} "
        f"experts={measured['experts']} target={invariant_row_target(measured['experts'], MEASURED_WORKSPACE)}"
        f" rows, scorings spent {spent} "
        + "; ".join(
            f"seed {case['seed']} flips={case['flips']}/{case['scored']} first_at={case['first']} "
            f"logit_dev={case['deviation']:.3e} floor={case['floor']:.3e}"
            for case in measured["cases"]
        )
    )
    for case in measured["cases"]:
        assert case["floor"] == 0.0, (
            f"seed {case['seed']}: the packed call repeated disagreed with itself by {case['floor']:.3e}, so "
            "nothing measured against it can be read as a row-count effect"
        )
    assert found, (
        f"no one-row call selected a different expert set than the packed call in {spent} scorings across "
        f"{len(ROUTER_SEEDS)} seeds. Either the dependence this module removes is gone on this device -- "
        "which would make the invariance test beside this one vacuous -- or the flip rate has fallen below "
        "what this budget can see. Either way the pad can no longer be shown to be doing work."
    )


@pytest.mark.integration
def test_a_padded_router_sends_every_token_to_the_same_experts_at_every_row_count():
    """With the pad installed, the expert set a token is routed to does not depend on the call's row count.

    The assertion is an integer comparison against the same token's selection inside a packed call, at row
    counts below the gate's target and at it. A tolerance cannot express "the same experts", and a
    tolerance-shaped assertion on the scores is exactly what lets a flip through: the score moves by less than
    any tolerance a reviewer would object to and the token still goes somewhere else. The logits are asserted
    bit-identical beside the selection, so a shape whose scores move without changing the argmax still fails.

    The patch count is asserted too. The pad reaches the gate through the contraction floor rather than by
    naming the router, so a floor raised above this hidden size would leave the gate exposed with nothing in a
    log to say so -- a silent reopening of the exposure rather than a visible removal of the remedy.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device: this is a property of the GEMM kernel the device selects")

    row_counts = (1, 2, 4, 8, 16, 32, 64)
    measured = _run_router_child(pad=True, row_counts=row_counts, stop_at_first_flip=False)
    assert measured["patched"] == 1, (
        f"the pad patched {measured['patched']} modules in a router built at hidden {ROUTER_HIDDEN}, where "
        "the gate alone is expected. The predicate selects a projection by its contraction reaching "
        f"{MIN_DEPENDENT_CONTRACTION} and being a multiple of {CONTRACTION_ALIGNMENT}, so a floor raised "
        f"above {ROUTER_HIDDEN} leaves the gate exposed silently"
    )

    failures: list[str] = []
    for case in measured["cases"]:
        head = f"seed {case['seed']} rows={case['rows']}"
        if case["floor"] != 0.0:
            failures.append(
                f"{head}: the packed call repeated disagreed with itself by {case['floor']:.3e}, so nothing "
                "measured against it can be read as a row-count effect"
            )
        if case["flips"]:
            failures.append(
                f"{head}: {case['flips']} of {case['scored']} tokens were routed to a different expert set "
                f"than the same token in a {ROUTER_TOKENS}-row call, first at scoring {case['first']}. Those "
                "tokens are processed by different weights, so the difference in their output is bounded by "
                "the experts that were swapped rather than by any tolerance"
            )
        if case["deviation"] != 0.0:
            failures.append(
                f"{head}: router logits disagree with a {ROUTER_TOKENS}-row call by {case['deviation']:.3e} "
                f"(floor {case['floor']:.3e}, scale {case['scale']:.3e}), so the selection is one near-tie "
                "away from moving even where it has not moved yet"
            )
    print(
        f"PADDED {measured['device']} patched={measured['patched']} experts={measured['experts']} "
        + "; ".join(
            f"seed {case['seed']} rows={case['rows']} flips={case['flips']}/{case['scored']} "
            f"logit_dev={case['deviation']:.3e}"
            for case in measured["cases"]
        )
    )
    assert len(measured["cases"]) == len(ROUTER_SEEDS) * len(row_counts)
    assert not failures, "a token's expert set depends on how many tokens shared the router call:\n  " + "\n  ".join(
        failures
    )


def test_the_pad_is_requested_only_by_a_determinism_run():
    """A production recipe keeps the projection it had; the pad follows ``debug.full_determinism``.

    The pad costs a taller call on every covered projection, so which runs pay for it is a decision worth
    pinning here rather than leaving to a branch in the model loader.
    """
    assert row_invariant_projections_requested({}) is False
    assert row_invariant_projections_requested({"debug": {}}) is False
    assert row_invariant_projections_requested({"debug": {"full_determinism": False}}) is False
    assert row_invariant_projections_requested({"debug": {"full_determinism": True}}) is True
    assert row_invariant_projections_requested({"prime_rl": {"debug": {"full_determinism": True}}}) is True


def test_the_entry_point_patches_only_when_the_run_asked():
    """The loader's single call is the whole gate: a run that did not ask comes back with nothing patched.

    A loader calls this and reads neither the flag nor the reason, so the no-op path is the one worth pinning.
    A regression that patches unconditionally is silent -- the arithmetic stays exact -- and every production
    step pays a taller call for a property it never reads.
    """
    with torch.device("meta"):
        unasked = torch.nn.Sequential(torch.nn.Linear(4096, 4096, bias=False))
        asked = torch.nn.Sequential(torch.nn.Linear(4096, 4096, bias=False))

    assert maybe_apply_row_invariant_projections(unasked, {}, verify=False) == 0
    assert "forward" not in unasked[0].__dict__
    assert maybe_apply_row_invariant_projections(asked, {"debug": {"full_determinism": True}}, verify=False) == 1
    assert "forward" in asked[0].__dict__
