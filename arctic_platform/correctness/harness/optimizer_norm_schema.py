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

"""Validate compact optimizer norms without importing measurement or model dependencies."""

from __future__ import annotations

import math

from .names import normalize

STATE_KEYS = ("parameter_update", "exp_avg", "exp_avg_sq")
COMPARISON_STATE_KEYS = ("parameter_update_residual", "exp_avg", "exp_avg_sq")
OPTIMIZER_NORM_CRITERION = "optimizer-state-l2-norm-v1"


def validate_optimizer_norm_summary(summary: dict) -> dict:
    """Return the canonical parameter mapping after validating the complete step-one norm schema."""
    if not isinstance(summary, dict) or summary.get("criterion") != OPTIMIZER_NORM_CRITERION:
        raise ValueError(f"optimizer norms require criterion {OPTIMIZER_NORM_CRITERION}")
    if type(summary.get("step")) is not int or summary["step"] != 1:
        raise ValueError("optimizer norms must describe step 1")
    if summary.get("state_keys") != list(COMPARISON_STATE_KEYS):
        raise ValueError("optimizer norms have incomplete or incompatible state coverage")
    parameters = summary.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("optimizer norms have no parameter coverage")
    counts = {
        "parameter_count": len(parameters),
        "optimizer_values_count": len(parameters) * len(COMPARISON_STATE_KEYS),
    }
    for key, expected in counts.items():
        if type(summary.get(key)) is not int or summary[key] != expected:
            raise ValueError(f"optimizer norms have inconsistent {key}")
    for name, values in parameters.items():
        if not isinstance(name, str) or not name or normalize(name) != name:
            raise ValueError("optimizer norms require nonempty canonical parameter names")
        if not isinstance(values, dict) or set(values) != {"shape", *COMPARISON_STATE_KEYS}:
            raise ValueError(f"optimizer norms for {name} have incomplete or incompatible state coverage")
        shape = values["shape"]
        if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
            raise ValueError(f"optimizer norms for {name} have an invalid shape")
        for state in COMPARISON_STATE_KEYS:
            value = values[state]
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise ValueError(f"optimizer norm for {name}::{state} must be finite and nonnegative")
    return parameters
