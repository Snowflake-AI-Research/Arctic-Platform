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

"""Measure reference repeatability without consulting a target run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from decimal import ROUND_CEILING
from decimal import Decimal
from pathlib import Path


def calibrate(paths: list[Path]) -> dict:
    if len(paths) < 16:
        raise ValueError("at least 16 independent reference runs are required")
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("each reference run must have a distinct file")
    records = [json.loads(path.read_bytes()) for path in paths]
    identity = (
        "schema",
        "model",
        "recipe_sha256",
        "inputs_sha256",
        "model_files_sha256",
        "model_class",
        "reference_source_sha256",
        "reference_attention",
        "dtype",
        "optimizer",
        "gradient_clipping",
        "torch_version",
        "transformers_version",
        "initial_parameters",
    )
    baseline = records[0]
    if baseline.get("schema") != "catalog-hf-reference-v2":
        raise ValueError("unsupported reference schema")
    if any(any(record.get(key) != baseline.get(key) for key in identity) for record in records[1:]):
        raise ValueError("reference runs do not share the same workload and implementation")
    spread = {"gradient": (0.0, ""), "optimizer": (0.0, "")}
    for arm in ("gas1", "gas4"):
        cases = [record if arm == "gas1" else record["gas4"] for record in records]
        if arm == "gas4" and any(
            case.get("gradient_accumulation_dtype") != cases[0].get("gradient_accumulation_dtype") for case in cases
        ):
            raise ValueError("reference accumulation dtypes differ")
        names = set(baseline["initial_parameters"])
        for case in cases:
            if set(case.get("gradient_norms", {})) != names or set(case.get("optimizer_step", {})) != names:
                raise ValueError("reference parameter coverage differs")
        for name in names:
            for category, key in (
                ("gradient", "gradient_norms"),
                ("optimizer", "parameter_update_residual_norm"),
                ("optimizer", "exp_avg_norm"),
                ("optimizer", "exp_avg_sq_norm"),
            ):
                values = [
                    case[key][name] if category == "gradient" else case["optimizer_step"][name][key] for case in cases
                ]
                if any(
                    not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values
                ):
                    raise ValueError("reference measurements must be finite nonnegative norms")
                variation = max(values) - min(values)
                if variation > spread[category][0]:
                    spread[category] = (variation, f"{arm}:{name}:{key}")
    quantum = Decimal("0.001")
    result = {
        "policy": "reference-repeatability-4x-upward-1e-3-v1",
        "runs": len(paths),
        "reference_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
        "workload": {
            key: baseline[key]
            for key in ("model", "recipe_sha256", "inputs_sha256", "model_files_sha256", "reference_source_sha256")
        },
        "gates": {},
    }
    for category, (variation, name) in spread.items():
        computed = Decimal(str(variation)) * 4
        selected = max(quantum, (computed / quantum).to_integral_value(rounding=ROUND_CEILING) * quantum)
        result["gates"][category] = {
            "absolute_tolerance": float(selected),
            "max_reference_range": variation,
            "worst_measurement": name,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("references", nargs="+", type=Path)
    args = parser.parse_args()
    result = calibrate(args.references)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
