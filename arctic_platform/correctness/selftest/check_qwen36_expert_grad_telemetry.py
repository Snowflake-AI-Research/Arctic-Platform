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

"""Selftests for the Qwen3.6 expert-gradient telemetry diagnostic helpers."""

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

    path = Path(__file__).resolve().parents[1] / "diagnostics" / "qwen36_expert_grad_telemetry.py"
    spec = importlib.util.spec_from_file_location("_qwen36_expert_grad_telemetry_selftest", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_routed_expert_name_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    diagnostic = _load_diagnostic(monkeypatch)

    assert diagnostic._is_routed_expert_name("module.model.layers.2.mlp.experts.gate_up_proj")
    assert diagnostic._is_routed_expert_name("module.model.layers.2.mlp.experts.gate_up_proj.weight")
    assert diagnostic._is_routed_expert_name("layers.2.mlp.experts.down_proj")
    assert (
        diagnostic._expert_compare_name("module.model.layers.2.mlp.experts.gate_up_proj.weight")
        == "layers.2.mlp.experts.gate_up_proj"
    )
    assert diagnostic._is_routed_expert_name("layers.2.mlp.experts.w1")
    assert not diagnostic._is_routed_expert_name("layers.2.mlp.shared_expert.gate_proj.weight")
    assert not diagnostic._is_routed_expert_name("layers.2.self_attn.q_proj.weight")


def test_summarize_experts_uses_canonical_names_and_sorts_by_hottest_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic = _load_diagnostic(monkeypatch)

    rows = diagnostic.summarize_experts(
        {
            "layers.2.mlp.experts.gate_up_proj": 2.0,
            "layers.3.mlp.experts.down_proj": 4.0,
        },
        {
            "module.model.layers.2.mlp.experts.gate_up_proj.weight": 8.0,
            "layers.3.mlp.experts.down_proj": 4.2,
        },
        {
            "layers.2.mlp.experts.gate_up_proj": 2.01,
            "layers.3.mlp.experts.down_proj": 5.0,
        },
        {
            "model.layers.2.mlp.experts.gate_up_proj.weight": [1.0, 2.0, 3.0, 4.0],
        },
    )

    assert [row.name for row in rows] == [
        "layers.2.mlp.experts.gate_up_proj",
        "layers.3.mlp.experts.down_proj",
    ]
    assert rows[0].telemetry_ratio == pytest.approx(4.0)
    assert rows[0].artifact_ratio == pytest.approx(1.005)
    assert rows[0].expert_count == 4
    assert rows[1].telemetry_ratio == pytest.approx(1.05)
    assert rows[1].artifact_ratio == pytest.approx(1.25)


def test_interpretation_detects_high_telemetry_matching_artifact(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    diagnostic = _load_diagnostic(monkeypatch)
    row = diagnostic.ExpertRow(
        name="layers.2.mlp.experts.gate_up_proj",
        reference=2.0,
        telemetry=8.0,
        artifact=2.01,
        telemetry_ratio=4.0,
        artifact_ratio=1.005,
        telemetry_delta=6.0,
        artifact_delta=0.01,
        expert_count=4,
    )

    diagnostic.print_interpretation([row])

    assert "Telemetry reconstruction is the next patch target" in capsys.readouterr().out
