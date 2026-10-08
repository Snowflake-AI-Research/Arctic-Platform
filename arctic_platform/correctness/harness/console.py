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

"""Render a run's progress and per-arm verdicts to a terminal.

Separate from ``report``, which writes the Markdown and JSON a run leaves behind. This module writes only
while a run is in progress, where the reader needs each arm's conclusion at the moment it is reached, with
the evidence beside it, rather than a summary line that sends them to a file.
"""

from __future__ import annotations

import sys
import threading
from typing import List
from typing import Optional
from typing import TextIO

from .registry import TestOutcome
from .registry import TestResult
from .spec import ArmSpec
from .spec import tolerance_text

# Wide enough for the failure table, which is the widest line any verdict prints. Blank lines on both
# sides: a rule with numbers packed against it separates nothing.
RULE = "  " + "-" * 99
SEPARATOR = ["", RULE, ""]
SPINNER_FRAMES = "|/-\\"


class Activity:
    """Show a short rotating work line and erase it before the caller prints the result."""

    def __init__(self, message: str, *, stream: TextIO | None = None, interval: float = 0.1) -> None:
        self.message = message
        self.stream = stream if stream is not None else sys.stdout
        self.interval = interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._frame = 0
        self._width = 0
        self._interactive = bool(getattr(self.stream, "isatty", lambda: False)())

    def _render(self) -> None:
        with self._lock:
            frame = SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)]
            self._frame += 1
            line = f"  {self.message} {frame}"
            self._width = max(self._width, len(line))
            self.stream.write("\r" + line.ljust(self._width))
            self.stream.flush()

    def _spin(self) -> None:
        while not self._stop.wait(self.interval):
            self._render()

    def __enter__(self) -> "Activity":
        if not self._interactive:
            print(f"  {self.message} ...", file=self.stream, flush=True)
            return self
        self._render()
        self._thread = threading.Thread(target=self._spin, name="correctness-console-spinner", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._interactive:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            self.stream.write("\r" + " " * self._width + "\r")
            self.stream.flush()


def activity(message: str, *, stream: TextIO | None = None, interval: float = 0.1) -> Activity:
    """Return a temporary work line for a blocking operation."""
    return Activity(message, stream=stream, interval=interval)


def separator() -> str:
    """The same three-line break the per-case verdicts use, for callers that print their own blocks."""
    return "\n".join(SEPARATOR)


def stopped_early(skipped) -> str:
    joined = ", ".join(skipped)
    return (
        f"  stopped after the first failing arm; not measured: {joined}\n"
        "  re-run without --fail-fast to measure every arm."
    )


def duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}m{rest:04.1f}s"


def runtime_table(timing, wall_seconds: float) -> str:
    """Per-case time for each engine, so a case that dominates the run is visible without a profiler."""
    names = list(timing)
    if not names:
        return "\n".join(SEPARATOR + [f"  runtime: {duration(wall_seconds)} wall clock"])
    width = max(9, max(len(n) for n in names) + 2)
    header = "  runtime    " + "".join(n.rjust(width) for n in names) + "total".rjust(width)
    rows = list(SEPARATOR) + [header]
    engines = [("reference", "reference"), ("Arctic Platform", "dss")]
    if any("optimizer_reference" in timing[name] for name in names):
        engines.append(("opt ref", "optimizer_reference"))
    if any("optimizer_dss" in timing[name] for name in names):
        engines.append(("opt Arctic Platform", "optimizer_dss"))
    for engine, key in engines:
        cells = [timing[n].get(key) for n in names]
        total = sum(v for v in cells if v is not None)
        rows.append(
            f"    {engine:<9}"
            + "".join((duration(v) if v is not None else "-").rjust(width) for v in cells)
            + duration(total).rjust(width)
        )
    rows.append(f"  wall clock for this config, including gateway startup and model load: {duration(wall_seconds)}")
    return "\n".join(rows)


def _fmt(value: float, width: int = 0) -> str:
    return f"{value:.6g}".rjust(width)


SHOWN = 10


def _mismatch_table(mismatches) -> List[str]:
    """Columns sized to the rows that are printed, not to the longest name in the whole failure set.

    Only the first ``SHOWN`` rows reach the terminal, so widening every column to accommodate names that
    stay in the report leaves a gutter down the middle of the table.
    """
    shown = list(mismatches[:SHOWN])
    header = ("parameter grad norm", "target", "ref", "abs diff", "ratio")
    cells = [
        (
            mm.name,
            _fmt(mm.target),
            _fmt(mm.reference),
            f"{abs(mm.target - mm.reference):.3e}",
            f"{(mm.target / mm.reference if mm.reference else float('nan')):.6f}",
        )
        for mm in shown
    ]
    widths = [max(len(header[i]), max(len(row[i]) for row in cells)) for i in range(len(header))]

    def line(row, first_left=True):
        out = row[0].ljust(widths[0]) if first_left else row[0].rjust(widths[0])
        return "      " + out + "".join("  " + row[i].rjust(widths[i]) for i in range(1, len(row)))

    lines = [line(header)]
    lines.extend(line(row) for row in cells)
    if len(mismatches) > SHOWN:
        lines.append(f"      ... {len(mismatches) - SHOWN} more failing, all in the report")
    return lines


def _optimizer_mismatch_table(mismatches) -> List[str]:
    shown = list(mismatches[:SHOWN])
    header = ("parameter and optimizer value", "target-reference delta L2 norm")
    cells = [(mismatch.name, f"{mismatch.target:.3e}") for mismatch in shown]
    widths = [max(len(header[i]), max(len(row[i]) for row in cells)) for i in range(2)]
    lines = ["      " + header[0].ljust(widths[0]) + "  " + header[1].rjust(widths[1])]
    lines.extend("      " + row[0].ljust(widths[0]) + "  " + row[1].rjust(widths[1]) for row in cells)
    if len(mismatches) > SHOWN:
        lines.append(f"      ... {len(mismatches) - SHOWN} more failing, all in the report")
    return lines


def arm_verdict(result: TestResult, arm: Optional[ArmSpec], timing: Optional[dict] = None) -> str:
    """One self-contained block: what was compared, what came out, and the verdict."""
    m = result.metrics
    # A rule above each case keeps consecutive verdicts from reading as one wall of numbers.
    lines: List[str] = list(SEPARATOR)
    if arm is not None:
        # Under sequence parallelism every sequence occupies the whole group, so the sequence count is
        # also the model-call count. Saying so here stops the reader from reading it as a per-GPU batch.
        lines.append(
            f"  case {arm.name} -- {arm.global_batch_size} sequence(s) of {arm.max_seq_len:,} "
            "tokens, run one after another through the whole GPU group"
        )
        lines.append(
            f"        {arm.total_tokens:,} token slots, {arm.active_tokens:,} real tokens "
            f"({arm.pad_fraction:.1%} padding)"
        )
    else:
        lines.append(f"  {result.test_id}")

    if result.outcome is TestOutcome.INAPPLICABLE:
        lines.append(f"    INAPPLICABLE  {result.reason}")
        return "\n".join(lines)

    if "reference_loss" in m and "dss_loss" in m:
        lines.append(
            f"    loss          reference {m['reference_loss']:<12.6f} Arctic Platform {m['dss_loss']:<12.6f} "
            f"delta {m.get('loss_delta', 0.0):.3e}"
        )
    elif "loss_delta" in m:
        lines.append(f"    loss          delta {m['loss_delta']:.3e}")

    observed = m.get("dss_model_calls_observed")
    ref_calls = m.get("reference_microbatches")
    if observed is not None or ref_calls is not None:
        predicted = m.get("dss_microbatches_predicted")
        note = (
            ""
            if observed is None or predicted is None or int(predicted) == int(observed)
            else f" (predicted {int(predicted)})"
        )
        left = f"{int(ref_calls):<12d}" if ref_calls is not None else f"{'-':<12}"
        right = f"{int(observed)}" if observed is not None else "-"
        lines.append(f"    model calls   reference {left} Arctic Platform {right}{note}")

    compared = int(m.get("tensors_compared", 0))
    if compared:
        unmatched = int(m.get("unmatched_dss", 0)) + int(m.get("unmatched_reference", 0))
        over_stated = int(m.get("tensors_over_stated_criterion", 0))
        label = 43  # one column for every sub-row, so the values below form a single readable column
        tail = f", {unmatched} unmatched" if unmatched else ""
        lines.append(f"    gradient L2 norm per parameter ({compared} parameters compared{tail})")
        gate = m.get("stated_criterion_abs")
        if gate is not None:
            if over_stated:
                gate_result = f"FAILED {tolerance_text(gate)} tolerance"
                gate_count = f"{over_stated}/{compared}"
            else:
                gate_result = f"PASSED {tolerance_text(gate)} tolerance"
                gate_count = f"{compared}/{compared}"
            lines.append("        " + gate_result.ljust(label) + gate_count)
        if "median_ratio" in m:
            lines.append(
                "        " + "median Arctic Platform/reference ratio".ljust(label) + f"{m['median_ratio']:.6f}"
            )
        if "max_abs_diff" in m:
            worst = result.worst_name or "-"
            lines.append("        " + "largest disagreement".ljust(label) + f"{m['max_abs_diff']:.3e}  {worst}")

    optimizer_compared = int(m.get("optimizer_values_compared", 0))
    if optimizer_compared:
        over_gate = int(m.get("optimizer_values_over_criterion", 0))
        gate = m.get("stated_criterion_abs")
        lines.append(f"    optimizer moment and update-residual delta L2 norms ({optimizer_compared} compared)")
        if gate is not None:
            if over_gate:
                gate_result = f"FAILED {tolerance_text(gate)} tolerance"
                gate_count = f"{over_gate}/{optimizer_compared}"
            else:
                gate_result = f"PASSED {tolerance_text(gate)} tolerance"
                gate_count = f"{optimizer_compared}/{optimizer_compared}"
            lines.append("        " + gate_result.ljust(43) + gate_count)
        worst = result.worst_name or "-"
        lines.append(
            "        " + "largest disagreement".ljust(43) + f"{m.get('max_optimizer_delta_norm', 0.0):.3e}  {worst}"
        )

    if result.outcome is TestOutcome.PASS:
        lines.append("")
        lines.append("    PASS")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"    FAIL  {result.summary}")
    if result.reason:
        lines.append(f"          {result.reason}")
    if result.mismatches:
        table = (
            _optimizer_mismatch_table(result.mismatches) if optimizer_compared else _mismatch_table(result.mismatches)
        )
        lines.extend(table)
    return "\n".join(lines)
