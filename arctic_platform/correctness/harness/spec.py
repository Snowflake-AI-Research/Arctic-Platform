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

"""The frozen artifact onboarding writes and regression consumes.

Onboarding may be expensive; regression re-derives nothing. The spec records the materialized model,
deterministic cases, and selected attention implementations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from decimal import ROUND_CEILING
from decimal import Decimal
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional


@dataclass
class ArmSpec:
    """One global batch size, with the shape onboarding measured for it."""

    name: str
    global_batch_size: int
    max_seq_len: int
    total_tokens: int
    active_tokens: int
    dss_microbatches: int
    pad_fraction: float

    @property
    def exercises_accumulation(self) -> bool:
        return self.dss_microbatches > 1


@dataclass
class ModelSpec:
    """The synthetic model recipe. Regenerable from the seed, so the path is a cache and not an input."""

    source_checkpoint: str
    num_hidden_layers: int
    vision_depth: int
    seed: int
    param_count: int
    cache_path: str
    content_hash: str
    sized_against_gib: float
    measured_reference_peak_gib: Optional[float] = None


@dataclass
class TestTolerance:
    """A frozen regression gate and the reference measurement that selected it."""

    absolute: float
    calibration_runs: Optional[int] = None
    raw_max_same_tensor_range: Optional[float] = None
    multiplier: Optional[float] = None
    computed_gate: Optional[float] = None
    rounding_quantum: float = 1e-3
    minimum_gate: float = 1e-3
    worst_tensor: Optional[str] = None
    status: str = "uncalibrated"

    def __post_init__(self) -> None:
        if self.absolute <= 0:
            raise ValueError(f"test tolerance must be positive, got {self.absolute}")
        if self.rounding_quantum <= 0 or self.minimum_gate <= 0:
            raise ValueError("test tolerance rounding quantum and minimum gate must be positive")
        if self.computed_gate is None:
            if self.absolute < self.minimum_gate:
                raise ValueError(f"selected tolerance {self.absolute} is below minimum gate {self.minimum_gate}")
            return
        quantum = Decimal(str(self.rounding_quantum))
        computed = Decimal(str(self.computed_gate))
        minimum = Decimal(str(self.minimum_gate))
        expected = max(
            minimum,
            (computed / quantum).to_integral_value(rounding=ROUND_CEILING) * quantum,
        )
        if Decimal(str(self.absolute)) != expected:
            raise ValueError(
                f"selected tolerance {self.absolute} does not equal upward-rounded calibrated gate {expected}"
            )


@dataclass
class TestSpec:
    config_id: str
    config_path: str
    model: ModelSpec
    arms: List[ArmSpec]
    attn_implementations: List[str]
    applicable_tests: List[str]
    config_checksum: str = ""
    test_tolerances: Dict[str, TestTolerance] = field(default_factory=dict)
    test_settings: Dict[str, dict] = field(default_factory=dict)
    inapplicable: Dict[str, str] = field(default_factory=dict)
    param_name_map: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def tolerance_for(self, test_id: str) -> float:
        try:
            return self.test_tolerances[test_id].absolute
        except KeyError as exc:
            raise ValueError(
                f"test {test_id!r} has no frozen tolerance in config spec {self.config_id!r}; "
                "run config onboarding first"
            ) from exc

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @staticmethod
    def read(path: Path) -> "TestSpec":
        raw = json.loads(Path(path).read_text())
        return TestSpec(
            config_id=raw["config_id"],
            config_path=raw["config_path"],
            model=ModelSpec(**raw["model"]),
            arms=[ArmSpec(**a) for a in raw["arms"]],
            attn_implementations=raw["attn_implementations"],
            applicable_tests=raw["applicable_tests"],
            config_checksum=raw.get("config_checksum", ""),
            test_tolerances={
                test_id: TestTolerance(**tolerance) for test_id, tolerance in raw.get("test_tolerances", {}).items()
            },
            test_settings=raw.get("test_settings", {}),
            inapplicable=raw.get("inapplicable", {}),
            param_name_map=raw.get("param_name_map", {}),
            notes=raw.get("notes", []),
        )


NON_MODEL_PROCESSOR_SIDECARS = frozenset({"preprocessor_config.json", "video_preprocessor_config.json"})


def hash_directory(path: Path, suffixes=(".safetensors", ".json")) -> str:
    """Hash model and tokenizer assets while ignoring processor metadata that does not affect text execution."""
    h = hashlib.sha256()
    for f in sorted(Path(path).rglob("*")):
        if f.is_file() and f.suffix in suffixes and f.name not in NON_MODEL_PROCESSOR_SIDECARS:
            h.update(f.name.encode())
            h.update(str(f.stat().st_size).encode())
    return h.hexdigest()[:16]


# Scratch probes retain the historical gate. Product tests read their frozen gate from TestSpec.
STATED_CRITERION_ABS = 1e-3


def tolerance_text(value: float) -> str:
    """``1e-03`` rather than ``1.000e-03``, without rounding a value that needs its digits."""
    mantissa, exponent = f"{value:.3e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{exponent}"
