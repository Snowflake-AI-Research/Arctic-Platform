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

"""Compare named Adam moments and moment-independent update residuals from one training step."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import torch

from ..harness.names import CanonicalTensorGroup
from ..harness.names import canonical_dss_tensor_groups
from ..harness.names import normalize

STATE_KEYS = ("parameter_update", "exp_avg", "exp_avg_sq")
COMPARISON_STATE_KEYS = ("parameter_update_residual", "exp_avg", "exp_avg_sq")


@dataclass(frozen=True)
class StateDelta:
    name: str
    state: str
    delta_norm: float


@dataclass(frozen=True)
class OptimizerComparison:
    deltas: List[StateDelta]
    only_dss: List[str]
    only_reference: List[str]

    @property
    def worst_delta(self) -> float:
        return max((item.delta_norm for item in self.deltas), default=0.0)


def _load_manifest(path: Path) -> Tuple[int, Dict[str, Path]]:
    manifest = json.loads(path.read_text())
    step = int(manifest.get("step", 0))
    entries: Dict[str, Path] = {}
    for item in manifest["parameters"]:
        name = normalize(item["name"])
        if name in entries:
            raise ValueError(f"optimizer artifact has duplicate normalized parameter name: {name}")
        entries[name] = path.parent / item["file"]
    return step, entries


def _load_tensor_group(
    entries: Dict[str, Path],
    group: CanonicalTensorGroup,
    device,
) -> Dict[str, "torch.Tensor"]:
    import torch

    artifacts = [torch.load(entries[name], map_location="cpu", weights_only=True) for name in group.originals]
    tensors = {}
    for state in STATE_KEYS:
        if any(state not in artifact for artifact in artifacts):
            raise ValueError(f"optimizer artifact for {group.originals[0]} is missing {state}")
        values = [artifact[state].to(device=device, dtype=torch.float32) for artifact in artifacts]
        tensors[state] = values[0] if group.concatenate_dim is None else torch.cat(values, dim=group.concatenate_dim)
    return tensors


def _comparison_tensors(
    tensors: Dict[str, "torch.Tensor"], *, step: int, optimizer_config: dict, learning_rate: float, backend: str
) -> Dict[str, "torch.Tensor"]:
    """Separate the Adam update from the moments that determine it.

    A first Adam step is almost a sign operation: small accepted gradient differences near zero can change the raw
    parameter update by one learning-rate unit while both moment tensors remain within their gate. Compare those
    moments directly, then compare only the part of each update they do not explain. DeepSpeed's FusedAdam kernel
    receives its scalar arguments as float32 and computes bias corrections from those rounded values; the independent
    PyTorch reference computes them from Python floats. Reproducing each backend's scalar arithmetic prevents a
    one-ULP-per-element backend difference from accumulating into a false full-tensor residual.
    """
    import torch

    beta1, beta2 = (float(value) for value in optimizer_config.get("betas", [0.9, 0.999]))
    epsilon = float(optimizer_config.get("eps", 1e-8))
    learning_rate = float(learning_rate)
    if backend == "dss":
        beta1 = float(torch.tensor(beta1, dtype=torch.float32))
        beta2 = float(torch.tensor(beta2, dtype=torch.float32))
        beta1_correction = float(torch.tensor(1.0 - math.pow(beta1, step), dtype=torch.float32))
        beta2_correction = float(torch.tensor(1.0 - math.pow(beta2, step), dtype=torch.float32))
        epsilon = float(torch.tensor(epsilon, dtype=torch.float32))
        learning_rate = float(torch.tensor(learning_rate, dtype=torch.float32))
    elif backend == "reference":
        beta1_correction = 1.0 - beta1**step
        beta2_correction = 1.0 - beta2**step
    else:
        raise ValueError(f"unknown optimizer comparison backend: {backend!r}")

    exp_avg = tensors["exp_avg"].float()
    exp_avg_sq = tensors["exp_avg_sq"].float()
    unbiased_avg = exp_avg / beta1_correction
    unbiased_avg_sq = exp_avg_sq / beta2_correction
    denominator = unbiased_avg_sq.sqrt() + epsilon
    if backend == "dss":
        moment_update = -(unbiased_avg / denominator) * learning_rate
    else:
        moment_update = -learning_rate * unbiased_avg / denominator
    return {
        "parameter_update_residual": tensors["parameter_update"].float() - moment_update,
        "exp_avg": exp_avg,
        "exp_avg_sq": exp_avg_sq,
    }


def _is_unused_zero_group(entries: Dict[str, Path], group: CanonicalTensorGroup) -> bool:
    import torch

    if not all(name.startswith("visual.") for name in group.originals):
        return False
    tensors = _load_tensor_group(entries, group, torch.device("cpu"))
    return all(not bool(torch.count_nonzero(value)) for value in tensors.values())


def compare_optimizer_artifacts(
    dss_manifest: Path, reference_manifest: Path, *, optimizer_config: dict, learning_rate: float
) -> OptimizerComparison:
    """Return moment deltas and the update delta left after each engine's own moments are accounted for."""
    import torch

    dss_step, dss = _load_manifest(Path(dss_manifest))
    reference_step, reference = _load_manifest(Path(reference_manifest))
    if dss_step != reference_step or dss_step != 1:
        raise ValueError(
            f"optimizer artifact step mismatch: Arctic Platform={dss_step}, reference={reference_step}, expected=1"
        )

    dss_groups = canonical_dss_tensor_groups(dss)
    only_dss_keys = set(dss_groups) - set(reference)
    only_dss_keys = {key for key in only_dss_keys if not _is_unused_zero_group(dss, dss_groups[key])}
    shared = sorted(set(dss_groups) & set(reference))
    deltas = []
    comparison_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in shared:
        dss_tensors = _load_tensor_group(dss, dss_groups[name], comparison_device)
        reference_tensors = torch.load(reference[name], map_location="cpu", weights_only=True)
        reference_values = {}
        for state in STATE_KEYS:
            if state not in reference_tensors:
                raise ValueError(f"optimizer artifact for {name} is missing {state}")
            reference_values[state] = reference_tensors[state].to(device=comparison_device, dtype=torch.float32)
            if dss_tensors[state].shape != reference_values[state].shape:
                raise ValueError(
                    f"optimizer artifact shape mismatch for {name}::{state}: "
                    f"{tuple(dss_tensors[state].shape)} != {tuple(reference_values[state].shape)}"
                )
        dss_values = _comparison_tensors(
            dss_tensors,
            step=dss_step,
            optimizer_config=optimizer_config,
            learning_rate=learning_rate,
            backend="dss",
        )
        reference_values = _comparison_tensors(
            reference_values,
            step=reference_step,
            optimizer_config=optimizer_config,
            learning_rate=learning_rate,
            backend="reference",
        )
        for state in COMPARISON_STATE_KEYS:
            delta_norm = torch.linalg.vector_norm(dss_values[state] - reference_values[state])
            deltas.append(StateDelta(name=name, state=state, delta_norm=float(delta_norm.cpu())))
        del dss_tensors, dss_values, reference_tensors, reference_values
    return OptimizerComparison(
        deltas=deltas,
        only_dss=sorted(original for key in only_dss_keys for original in dss_groups[key].originals),
        only_reference=sorted(set(reference) - set(dss_groups)),
    )


def max_pairwise_optimizer_delta(
    manifests: List[Path], *, optimizer_config: dict, learning_rate: float, progress=None
) -> StateDelta | None:
    """Return the largest comparison-tensor delta among every pair of reference artifacts."""
    import torch

    loaded = [_load_manifest(Path(manifest)) for manifest in manifests]
    if not loaded:
        return None
    if any(step != 1 for step, _ in loaded):
        raise ValueError("optimizer calibration artifacts must all describe step 1")
    names = set(loaded[0][1])
    if any(set(entries) != names for _, entries in loaded[1:]):
        raise ValueError("optimizer calibration runs returned different parameter sets")
    comparison_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    worst = None
    for name in sorted(names):
        paths = [entries[name] for _, entries in loaded]
        artifacts = []
        expected_shapes = {}
        for artifact_path in paths:
            tensors = torch.load(artifact_path, map_location="cpu", weights_only=True)
            for state in STATE_KEYS:
                if state not in tensors:
                    raise ValueError(f"optimizer artifact for {name} is missing {state}")
                if state not in expected_shapes:
                    expected_shapes[state] = tensors[state].shape
                elif tensors[state].shape != expected_shapes[state]:
                    raise ValueError(f"optimizer calibration shape mismatch for {name}::{state}")
            artifacts.append(
                _comparison_tensors(
                    {state: tensors[state].to(device=comparison_device, dtype=torch.float32) for state in STATE_KEYS},
                    step=1,
                    optimizer_config=optimizer_config,
                    learning_rate=learning_rate,
                    backend="reference",
                )
            )

        for state in COMPARISON_STATE_KEYS:
            values = [artifact[state] for artifact in artifacts]
            for left_index, left in enumerate(values[:-1]):
                for right in values[left_index + 1 :]:
                    delta = float(torch.linalg.vector_norm(left - right).cpu())
                    if worst is None or delta > worst.delta_norm:
                        worst = StateDelta(name=name, state=state, delta_norm=delta)
            del values
        del artifacts
        if progress is not None:
            progress.update()
    return worst
