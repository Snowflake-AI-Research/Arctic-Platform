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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Selftests for the Qwen3.6 routed expert gradient-scaling diagnostic helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _stub_module(name: str, **attrs: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_diagnostic(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    def _unused(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("selftest should not call runtime dependencies")

    def _normalize(name: str) -> str:
        return ".".join(part for part in name.split(".") if part not in {"module", "model"})

    stubs = {
        "arctic_platform.correctness.harness.arms": _stub_module(
            "arctic_platform.correctness.harness.arms",
            correctness_microbatch_tokens=lambda value: min(int(value), 65536),
        ),
        "arctic_platform.correctness.harness.batches": _stub_module(
            "arctic_platform.correctness.harness.batches",
            build_batch=_unused,
            load=_unused,
            save=_unused,
        ),
        "arctic_platform.correctness.harness.config": _stub_module(
            "arctic_platform.correctness.harness.config",
            load_config=_unused,
        ),
        "arctic_platform.correctness.harness.dss_driver": _stub_module(
            "arctic_platform.correctness.harness.dss_driver",
            _globalize_expert_norms=lambda per_param, per_expert: {
                **per_param,
                **{name: sum(value * value for value in values) ** 0.5 for name, values in per_expert.items()},
            },
            _rank_owned_mapping=lambda value: value if isinstance(value, dict) else {},
            _rank_owned_path=lambda value: value,
            _scalar=float,
            build_payload=_unused,
            gateway=_unused,
            pack=_unused,
            running_job=_unused,
        ),
        "arctic_platform.correctness.harness.names": _stub_module(
            "arctic_platform.correctness.harness.names",
            canonical_dss_tensor_groups=lambda names: {},
            normalize=_normalize,
        ),
        "arctic_platform.correctness.harness.runner": _stub_module(
            "arctic_platform.correctness.harness.runner",
            OPTIMIZER_LEARNING_RATE=0.01,
            run_reference=_unused,
        ),
        "arctic_platform.correctness.harness.seeds": _stub_module(
            "arctic_platform.correctness.harness.seeds",
            SEED=1234,
        ),
        "arctic_platform.correctness.harness.spec": _stub_module(
            "arctic_platform.correctness.harness.spec",
            ArmSpec=object,
            TestSpec=object,
        ),
        "arctic_platform.correctness.harness.workdir": _stub_module(
            "arctic_platform.correctness.harness.workdir",
            correctness_workdir=_unused,
        ),
        "arctic_platform.correctness.reference.model_features": _stub_module(
            "arctic_platform.correctness.reference.model_features",
            uses_mixer_packing=lambda _model_path: True,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    telemetry_path = Path(__file__).resolve().parents[1] / "diagnostics" / "qwen36_expert_grad_telemetry.py"
    telemetry_spec = importlib.util.spec_from_file_location(
        "arctic_platform.correctness.diagnostics.qwen36_expert_grad_telemetry",
        telemetry_path,
    )
    assert telemetry_spec is not None
    assert telemetry_spec.loader is not None
    telemetry = importlib.util.module_from_spec(telemetry_spec)
    monkeypatch.setitem(sys.modules, telemetry_spec.name, telemetry)
    telemetry_spec.loader.exec_module(telemetry)

    path = Path(__file__).resolve().parents[1] / "diagnostics" / "qwen36_expert_grad_scaling.py"
    spec = importlib.util.spec_from_file_location("_qwen36_expert_grad_scaling_selftest", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_scale_candidates_include_ep_expert_dp_world_and_router_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    diagnostic = _load_diagnostic(monkeypatch)

    candidates = diagnostic.scale_candidates(n_gpus=8, ep_size=4, router_top_k=8)

    assert [(candidate.name, candidate.factor) for candidate in candidates] == [
        ("no scaling", 1.0),
        ("expert_parallel", 4.0),
        ("expert_data_parallel", 2.0),
        ("world_size", 8.0),
    ]


def test_classify_pre_step_high_artifact_matches_and_moments_consume_same_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic = _load_diagnostic(monkeypatch)
    candidates = diagnostic.scale_candidates(n_gpus=8, ep_size=4, router_top_k=None)

    result = diagnostic.classify_scaling(
        pre_to_ref=4.05,
        artifact_to_ref=4.08,
        artifact_to_pre=1.01,
        exp_avg_effective_to_ref=4.02,
        candidates=candidates,
    )

    assert result == "pre-step high near expert_parallel; optimizer moments consume same scale"


def test_classify_artifact_globalization_added_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    diagnostic = _load_diagnostic(monkeypatch)
    candidates = diagnostic.scale_candidates(n_gpus=8, ep_size=4, router_top_k=None)

    result = diagnostic.classify_scaling(
        pre_to_ref=1.01,
        artifact_to_ref=4.1,
        artifact_to_pre=4.05,
        exp_avg_effective_to_ref=4.1,
        candidates=candidates,
    )

    assert result == "artifact/globalization adds scale near expert_parallel"


def test_summarize_scaling_uses_canonical_routed_names(monkeypatch: pytest.MonkeyPatch) -> None:
    diagnostic = _load_diagnostic(monkeypatch)
    candidates = diagnostic.scale_candidates(n_gpus=8, ep_size=4, router_top_k=None)

    rows = diagnostic.summarize_scaling(
        {"layers.2.mlp.experts.gate_up_proj": 2.0},
        {"module.model.layers.2.mlp.experts.gate_up_proj.weight": 8.0},
        {"layers.2.mlp.experts.gate_up_proj": 8.1},
        {"layers.2.mlp.experts.gate_up_proj": 8.0},
        {"model.layers.2.mlp.experts.gate_up_proj.weight": [1.0, 2.0, 3.0, 4.0]},
        candidates=candidates,
    )

    assert len(rows) == 1
    assert rows[0].name == "layers.2.mlp.experts.gate_up_proj"
    assert rows[0].pre_to_ref == pytest.approx(4.0)
    assert rows[0].artifact_to_pre == pytest.approx(1.0125)
    assert rows[0].expert_count == 4
