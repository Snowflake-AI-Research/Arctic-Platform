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


"Run environment, progress rendering, and durable result reports."

from __future__ import annotations

import json
import math
import platform
import socket
import subprocess
import sys
import threading
from dataclasses import asdict
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import TextIO

from .config import ArmSpec
from .config import TestSpec
from .config import tolerance_text
from .registry import TestOutcome
from .registry import TestResult

_PACKAGES = ("torch", "transformers", "deepspeed", "accelerate", "flash_attn", "flash_attn_3")


def _version(name: str) -> Optional[str]:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version

    try:
        return version(name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _git(repo: Path) -> Dict[str, str]:
    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    commit = run("rev-parse", "--short", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    dirty = run("status", "--porcelain")
    info = {}
    if commit:
        info["commit"] = commit + (" (uncommitted changes present)" if dirty else "")
    if branch:
        info["branch"] = branch
    return info


def _gpus() -> Dict[str, str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return {"gpus": f"{count} x {name} ({total:.0f} GiB each)", "cuda": torch.version.cuda or "unknown"}
    except Exception:
        return {}


def collect(repo: Optional[Path] = None) -> Dict[str, str]:
    """Everything needed to reproduce a run, on a best-effort basis: a missing fact is omitted, never fatal."""
    repo = repo or Path(__file__).resolve().parents[2]
    info: Dict[str, str] = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
    }
    info.update(_gpus())
    info.update(_git(repo))
    versions = {name: v for name in _PACKAGES if (v := _version(name))}
    if versions:
        info["packages"] = ", ".join(f"{k} {v}" for k, v in versions.items())
    return info


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


def total_duration(seconds: float) -> str:
    """Final wall-clock time, rounded upward to a whole second."""
    hours, remainder = divmod(math.ceil(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


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
        maximum = m.get("max_optimizer_norm_delta", m.get("max_optimizer_delta_norm"))
        if maximum is not None:
            worst = result.worst_name or "-"
            lines.append("        " + "largest disagreement".ljust(43) + f"{maximum:.3e}  {worst}")

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


def _counts(results: List[TestResult]) -> Dict[str, int]:
    out = {o.value: 0 for o in TestOutcome}
    for r in results:
        out[r.outcome.value] += 1
    return out


def _table(headers: List[str], rows: List[List[str]], right: Optional[set] = None) -> List[str]:
    """A Markdown table whose source lines are readable without a renderer, columns padded to one width."""
    right = right or set()
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: List[str]) -> str:
        out = [c.rjust(widths[i]) if i in right else c.ljust(widths[i]) for i, c in enumerate(cells)]
        return "| " + " | ".join(out) + " |"

    sep = ["-" * (widths[i] - 1) + ":" if i in right else "-" * widths[i] for i in range(len(headers))]
    return [fmt(headers), "| " + " | ".join(sep) + " |"] + [fmt(r) for r in rows]


def _header(specs: Dict[str, TestSpec], execution: Dict[str, dict]) -> List[str]:
    """What was measured, on what, with which software."""
    lines = [
        "# Arctic Platform correctness report",
        "",
        (
            "Each case runs through Arctic Platform on the topology its config describes and through a single-GPU "
            "HuggingFace reference holding identical weights and data. Forward-backward correctness compares "
            "each parameter's gradient L2 norm before clipping. Delta-update correctness applies one AdamW "
            "step and compares full-tensor parameter updates and optimizer moments. Each test uses the frozen "
            "absolute gate recorded in that config's reviewed spec."
        ),
        "",
        (
            "The reference is the golden truth: one GPU, no parallelism, no offload, gradients accumulated in "
            "float32. The model is the source checkpoint's architecture reduced to a layer count one GPU can "
            "hold, carrying that checkpoint's own weights in the layers it keeps, materialized once and loaded "
            "by both engines. Batches are generated from a fixed seed, contain rows of unequal length, and "
            "always contain padding, so token-weighted loss and uneven sharding are exercised rather than "
            "assumed."
        ),
        "",
    ]
    if any(item.get("backend") == "cortex" for item in execution.values()):
        lines[2:5] = [
            (
                "Cortex checks use the requested catalog model or an explicitly published reduced model. A publication"
                " binding records local model/tokenizer SHA-256 digests and the separate managed model ID; the HF"
                " reference runs outside Cortex. Publication alone does not verify remote worker bytes. Checkpoint"
                " checks compare Arctic Platform against Arctic Platform. Checks requiring local fixtures without a"
                " publication binding or worker instrumentation fail before remote execution. The frozen specs are"
                " retained as reviewed contracts, not evidence of remote model identity."
            ),
            "",
            "Environment fields below describe the client process, not the remote worker image or GPUs.",
        ]
    if any(spec.model.param_count is None for spec in specs.values()):
        lines[2:5] = [
            (
                "Reference-free checks use the original model and their reviewed comparison criteria, without a"
                " supervised Hugging Face reference. Original-model metadata is not a generated reference fixture"
                " or evidence that the target loaded the model. Other checks retain their reviewed reference and"
                " fixture contracts."
            ),
            "",
            "Environment fields below describe the client process, not independently verified target workers.",
        ]
    env = collect()
    labels = {
        "generated": "generated",
        "host": "host",
        "gpus": "GPUs",
        "cuda": "CUDA",
        "python": "Python",
        "commit": "repository commit",
        "branch": "branch",
        "packages": "packages",
    }
    rows = [[labels.get(k, k), v] for k, v in env.items()]
    lines += _table(["field", "value"], rows)
    lines.append("")
    return lines


def to_markdown(
    results: List[TestResult], specs: Dict[str, TestSpec], *, execution: Dict[str, dict] | None = None
) -> str:
    execution = execution or {}
    counts = _counts(results)
    lines: List[str] = _header(specs, execution)
    lines.append(
        f"**Overall: {counts['pass']} passed, {counts['fail']} failed, {counts['inapplicable']} inapplicable**"
    )
    lines.append("")

    for config_id in sorted(set(specs) | {result.config_id for result in results}):
        lines.append(f"## {config_id}")
        lines.append("")
        spec = specs.get(config_id)
        rows = [r for r in results if r.config_id == config_id]
        if spec is None:
            lines.append("No reviewed specification was loaded. No numerical comparison is established.")
            lines.append("")
            lines += _table(
                ["test", "case", "outcome", "summary"],
                [[f"`{r.test_id}`", r.arm or "-", f"**{r.outcome.value}**", r.summary] for r in rows],
            )
            for result in rows:
                lines.extend(["", f"### {result.test_id} / {result.arm or '-'}", "", result.reason or result.summary])
            lines.append("")
            continue
        metadata = execution.get(config_id, {})
        remote = metadata.get("backend") == "cortex"
        bound_fixture = remote and metadata.get("fixture_sha256") is not None
        local_fixture = bound_fixture and metadata.get("fixture_distribution") == "local-gateway"
        published = bound_fixture and not local_fixture
        inference_model = metadata.get("inference_model_name") if remote else None
        if metadata:
            lines.append(f"- execution backend: `{metadata.get('backend') or 'unknown'}`")
        if remote:
            label = "requested published reduced model" if published else "requested catalog model"
            if local_fixture:
                label = "requested local reduced model"
            if bound_fixture and inference_model is not None:
                label += " (fixture-based checks only)"
            lines.append(f"- {label}: `{metadata['requested_model_name']}`")
            if inference_model is not None:
                lines.append(f"- `inference-checkpoint-loss` requested full model: `{inference_model}`")
            if bound_fixture:
                lines.append(f"- local model/tokenizer SHA-256 binding: `{metadata['fixture_sha256']}`")
                if local_fixture:
                    lines.append(f"- local gateway: `{metadata['gateway_url']}`")
                else:
                    lines.append(f"- publication stage: `{metadata['fixture_stage_path']}`")
            lines.append("- remote model weights and worker environment: not independently verified")
        model_label = "model"
        if remote and not bound_fixture:
            model_label = "reviewed reference fixture (not loaded on Cortex)"
        elif bound_fixture and inference_model is not None:
            model_label = "reviewed reference fixture (fixture-based checks only; not used by inference)"
        parameter_count = f"{spec.model.param_count:,} parameters" if spec.model.param_count is not None else None
        if parameter_count is None:
            model_label = "original model metadata (not a generated reference fixture)"
            parameter_count = "parameter count unknown"
        lines.append(
            f"- {model_label}: {parameter_count}, {spec.model.num_hidden_layers} layers "
            f"of `{spec.model.source_checkpoint}`, hash `{spec.model.content_hash}`"
        )
        lines.append(f"- attention implementations: {', '.join(spec.attn_implementations)}")
        for test_id, tolerance in sorted(spec.test_tolerances.items()):
            lines.append(f"- `{test_id}` absolute gate: `{tolerance_text(tolerance.absolute)}` ({tolerance.status})")
        lines.append("")
        lines += _table(
            ["case", "sequences", "seq len", "token slots", "real tokens", "padding", "Arctic Platform model calls"],
            [
                [
                    f"`{a.name}`",
                    f"{a.global_batch_size:,}",
                    f"{a.max_seq_len:,}",
                    f"{a.total_tokens:,}",
                    f"{a.active_tokens:,}",
                    f"{a.pad_fraction:.1%}",
                    f"{a.dss_microbatches}",
                ]
                for a in spec.arms
            ],
            right={1, 2, 3, 4, 5, 6},
        )
        lines.append("")

        rows = [r for r in results if r.config_id == config_id]
        lines += _table(
            ["test", "case", "outcome", "summary"],
            [[f"`{r.test_id}`", r.arm or "-", f"**{r.outcome.value}**", r.summary] for r in rows],
        )
        lines.append("")

        for r in rows:
            if r.outcome is not TestOutcome.FAIL:
                continue
            lines.append(f"### {r.test_id} / {r.arm or '-'} failed")
            lines.append("")
            if r.reason:
                lines.append(r.reason)
                lines.append("")
            if r.metrics:
                lines += _table(
                    ["metric", "value"], [[k, f"{v:.6g}"] for k, v in sorted(r.metrics.items())], right={1}
                )
                lines.append("")
            if r.mismatches:
                if r.test_id == "single-step-optimizer":
                    lines += _table(
                        ["parameter and optimizer value", "target-reference delta L2 norm"],
                        [[f"`{m.name}`", f"{m.abs_diff:.3e}"] for m in r.mismatches],
                        right={1},
                    )
                else:
                    lines += _table(
                        ["parameter", "target norm", "reference norm", "abs diff", "ratio"],
                        [
                            [
                                f"`{m.name}`",
                                f"{m.target:.6g}",
                                f"{m.reference:.6g}",
                                f"{m.abs_diff:.3e}",
                                f"{m.ratio:.6f}",
                            ]
                            for m in r.mismatches
                        ],
                        right={1, 2, 3, 4},
                    )
                lines.append("")
    return "\n".join(lines) + "\n"


def to_json(results: List[TestResult], specs: Dict[str, TestSpec], *, execution: Dict[str, dict] | None = None) -> str:
    payload = {
        "totals": _counts(results),
        "results": [
            {
                "test_id": r.test_id,
                "config_id": r.config_id,
                "arm": r.arm,
                "gpus": r.gpus,
                "outcome": r.outcome.value,
                "summary": r.summary,
                "reason": r.reason,
                **({"error_type": r.error_type} if r.error_type is not None else {}),
                "metrics": r.metrics,
                "mismatches": [
                    {
                        "name": m.name,
                        "target": m.target,
                        "reference": m.reference,
                        "abs_diff": m.abs_diff,
                        "ratio": m.ratio if math.isfinite(m.ratio) else None,
                    }
                    for m in r.mismatches
                ],
            }
            for r in results
        ],
        "specs": {k: asdict(v) for k, v in specs.items()},
        "execution": execution or {},
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _point_latest(link: Path, target: Path) -> None:
    temporary = link.with_name(f".{link.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target.name)
    temporary.replace(link)


def write(
    results: List[TestResult],
    specs: Dict[str, TestSpec],
    out_dir: Path,
    *,
    timestamp: str | None = None,
    execution: Dict[str, dict] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now().strftime("%Y-%m-%d-%H-%M")
    markdown = out_dir / f"report-{stamp}.md"
    machine = out_dir / f"report-{stamp}.json"
    markdown.write_text(to_markdown(results, specs, execution=execution))
    machine.write_text(to_json(results, specs, execution=execution))
    _point_latest(out_dir / "report.md", markdown)
    _point_latest(out_dir / "report.json", machine)
