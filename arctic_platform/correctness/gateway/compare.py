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

"""Gate gateway targets against repeated HF references, and compare targets built from different trees."""

from __future__ import annotations

import json
import statistics
from decimal import ROUND_CEILING
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from ..harness.lora import canonical_name
from ..harness.names import align

_GATE_QUANTUM = Decimal("0.001")


def load_record(path: str | Path) -> dict:
    return canonical_lora_norms(json.loads(Path(path).read_text()))


def canonical_lora_norms(record: dict) -> dict:
    """Keep only LoRA tensors under one ``<module>.lora_{A,B}.weight`` name on either engine.

    Records without LoRA tensors are returned unchanged.
    """
    norms = record["grad_norms"]
    if not any(".lora_" in name for name in norms):
        return record
    mapped = {}
    for name, value in norms.items():
        if ".lora_" not in name:
            continue
        key = canonical_name(name)
        for prefix in ("module.", "base_model.model."):
            key = key.removeprefix(prefix)
        if key in mapped:
            raise ValueError(f"two LoRA tensors map to {key}")
        mapped[key] = value
    return {**record, "grad_norms": mapped}


def gate(references: list[dict]) -> tuple[float, float]:
    """Max per-tensor spread across references and the absolute gate: 4x that, rounded up to 1e-3, at least 1e-3."""
    if len(references) < 2:
        return 0.0, float(_GATE_QUANTUM)
    spread = 0.0
    for name in sorted(references[0]["grad_norms"]):
        values = [r["grad_norms"][name] for r in references]
        spread = max(spread, max(values) - min(values))
    rounded = (Decimal(str(4.0 * spread)) / _GATE_QUANTUM).to_integral_value(rounding=ROUND_CEILING) * _GATE_QUANTUM
    return spread, max(float(_GATE_QUANTUM), float(rounded))


def verdict(fixture: str | Path, references: list[dict], target: dict, label: str) -> dict:
    """The ``single-step-grads`` result for one target, plus relative-difference statistics."""
    from ..checks.fwd_bwd import _run_arm
    from .target import arm_batch

    spread, threshold = gate(references)
    reference = references[0]
    spec, batch = arm_batch(fixture, target["arm"], target["n_gpus"], target["max_seq_len"])
    ctx = SimpleNamespace(
        config_id=label,
        arms=[spec],
        spec=SimpleNamespace(tolerance_for=lambda _test: threshold),
        reference_for=lambda _arm: reference,
        target_for=lambda _arm: SimpleNamespace(
            avg_loss=target["loss"], grad_norms=target["grad_norms"], model_calls=target.get("model_calls")
        ),
    )
    arm = SimpleNamespace(
        name=spec.name,
        global_batch_size=spec.global_batch_size,
        dss_microbatches=spec.microbatches(target["max_tokens_per_mb"]),
        pad_fraction=batch.pad_fraction,
    )
    result = _run_arm(ctx, arm)
    pairs, only_target, only_ref = align(target["grad_norms"], reference["grad_norms"])
    rows = {name: (a, b) for name, a, b in pairs}
    rel = sorted(abs(a - b) / abs(b) for a, b in rows.values() if b)
    over = sorted(((n, a, b) for n, (a, b) in rows.items() if abs(a - b) > threshold), key=lambda x: -abs(x[1] - x[2]))
    return {
        "label": label,
        "arm": target["arm"],
        "job_id": target.get("job_id"),
        "n_gpus": target["n_gpus"],
        "max_tokens_per_mb": target["max_tokens_per_mb"],
        "model_calls": target.get("model_calls"),
        "outcome": str(getattr(result.outcome, "value", result.outcome)),
        "summary": result.summary,
        "reason": result.reason,
        "gate": threshold,
        "reference_runs": len(references),
        "reference_spread": spread,
        "reference_loss": reference["loss"],
        "loss": target["loss"],
        "dloss": target["loss"] - reference["loss"],
        "tensors": len(rows),
        "rel_median": rel[len(rel) // 2] if rel else None,
        "rel_max": rel[-1] if rel else None,
        "ratio_median": statistics.median(a / b for a, b in rows.values() if b) if rows else None,
        "over_gate": [{"name": n, "target": a, "reference": b} for n, a, b in over],
        "only_target": only_target,
        "only_target_nonzero": [n for n in only_target if target["grad_norms"][n] != 0.0],
        "only_reference": only_ref,
        "norms": {n: list(v) for n, v in rows.items()},
        "adapter_export": target.get("adapter_export"),
    }


def head_to_head(a: dict, b: dict, repeat: dict | None = None) -> dict:
    """Two verdicts on the same references: relative target-vs-target change and which sits closer to the reference.

    ``repeat`` is a second run of ``a``'s tree and sets the noise floor for the change.
    """
    common = sorted(n for n in set(a["norms"]) & set(b["norms"]) if b["norms"][n][0])
    diffs = sorted(((a["norms"][n][0] - b["norms"][n][0]) / b["norms"][n][0], n) for n in common)
    absrel = sorted(abs(d) for d, _ in diffs)
    out = {
        "tensors": len(common),
        "median": absrel[len(absrel) // 2] if absrel else None,
        "max": max(diffs, key=lambda x: abs(x[0])) if diffs else None,
        "a_further_from_ref": sum(
            abs(a["norms"][n][0] - a["norms"][n][1]) > abs(b["norms"][n][0] - b["norms"][n][1]) for n in common
        ),
    }
    if repeat is not None:
        rr = sorted(
            abs(a["norms"][n][0] - repeat["norms"][n][0]) / abs(repeat["norms"][n][0])
            for n in common
            if n in repeat["norms"] and repeat["norms"][n][0]
        )
        if rr:
            out["repeat_median"], out["repeat_max"] = rr[len(rr) // 2], rr[-1]
    return out


def reference_paths(results: Path, config: str, label: str, arm: str) -> list[Path]:
    """Per-label references (LoRA, whose adapter comes from that label's job) or else shared ones."""
    per_label = sorted(results.glob(f"ref-{config}-{label}-{arm}-r*.json"))
    return per_label or sorted(results.glob(f"ref-{config}-{arm}-r*.json"))


def summarize(
    results: str | Path,
    fixtures: dict[str, str],
    labels: list[str],
    *,
    arms: tuple[str, ...] = ("gas1", "gas4"),
    pair: tuple[str, str] | None = None,
    repeat: str | None = None,
    min_references: int = 3,
) -> list[dict]:
    """Verdicts for ``target-{config}-{label}-{arm}.json`` against ``ref-{config}[-{label}]-{arm}-r*.json``.

    ``fixtures`` maps each config name to the reference fixture its arms are built from.
    """
    results = Path(results)
    rows = []
    for config, fixture in fixtures.items():
        for arm in arms:
            found = {}
            for label in labels:
                path = results / f"target-{config}-{label}-{arm}.json"
                refs = [load_record(p) for p in reference_paths(results, config, label, arm)]
                if not path.exists() or len(refs) < min_references:
                    continue
                row = {"config": config, **verdict(fixture, refs, load_record(path), label)}
                found[label] = row
                rows.append(row)
            if pair and all(p in found for p in pair):
                rows.append(
                    {
                        "config": config,
                        "arm": arm,
                        "head_to_head": list(pair),
                        **head_to_head(found[pair[0]], found[pair[1]], found.get(repeat)),
                    }
                )
    return rows
