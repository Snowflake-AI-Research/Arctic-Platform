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

"""Render results as Markdown for a human and JSON for a machine."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

from .environment import collect
from .registry import TestOutcome
from .registry import TestResult
from .spec import TestSpec
from .spec import tolerance_text


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


def _header(specs: Dict[str, TestSpec]) -> List[str]:
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


def to_markdown(results: List[TestResult], specs: Dict[str, TestSpec]) -> str:
    counts = _counts(results)
    lines: List[str] = _header(specs)
    lines.append(
        f"**Overall: {counts['pass']} passed, {counts['fail']} failed, {counts['inapplicable']} inapplicable**"
    )
    lines.append("")

    for config_id, spec in sorted(specs.items()):
        lines.append(f"## {config_id}")
        lines.append("")
        lines.append(
            f"- model: {spec.model.param_count:,} parameters, {spec.model.num_hidden_layers} layers "
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


def to_json(results: List[TestResult], specs: Dict[str, TestSpec]) -> str:
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
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now().strftime("%Y-%m-%d-%H-%M")
    markdown = out_dir / f"report-{stamp}.md"
    machine = out_dir / f"report-{stamp}.json"
    markdown.write_text(to_markdown(results, specs))
    machine.write_text(to_json(results, specs))
    _point_latest(out_dir / "report.md", markdown)
    _point_latest(out_dir / "report.json", machine)
