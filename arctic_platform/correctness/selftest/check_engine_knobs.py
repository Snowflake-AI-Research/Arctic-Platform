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

"""Selftests for diagnostic-only engine knob overrides."""

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


def _load_engine_knobs(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    def _unused(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("selftest should not call diagnostic runtime dependencies")

    stubs = {
        "arctic_platform.correctness.harness.batches": _stub_module(
            "arctic_platform.correctness.harness.batches",
            build_batch=_unused,
            save=_unused,
        ),
        "arctic_platform.correctness.harness.config": _stub_module(
            "arctic_platform.correctness.harness.config",
            load_config=_unused,
        ),
        "arctic_platform.correctness.harness.dss_driver": _stub_module(
            "arctic_platform.correctness.harness.dss_driver",
            build_payload=_unused,
            fwd_bwd_step=_unused,
            gateway=_unused,
            pack=_unused,
            running_job=_unused,
        ),
        "arctic_platform.correctness.harness.names": _stub_module(
            "arctic_platform.correctness.harness.names",
            align=_unused,
        ),
        "arctic_platform.correctness.harness.runner": _stub_module(
            "arctic_platform.correctness.harness.runner",
            run_reference=_unused,
        ),
        "arctic_platform.correctness.harness.seeds": _stub_module(
            "arctic_platform.correctness.harness.seeds",
            SEED=1234,
        ),
        "arctic_platform.correctness.harness.spec": _stub_module(
            "arctic_platform.correctness.harness.spec",
            STATED_CRITERION_ABS=1e-3,
            TestSpec=object,
        ),
        "arctic_platform.correctness.onboarding.synth_model": _stub_module(
            "arctic_platform.correctness.onboarding.synth_model",
            materialize_pretrained=_unused,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).resolve().parents[1] / "diagnostics" / "engine_knobs.py"
    spec = importlib.util.spec_from_file_location("_engine_knobs_selftest", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_engine_override_deep_merges_ep_size(monkeypatch: pytest.MonkeyPatch) -> None:
    engine_knobs = _load_engine_knobs(monkeypatch)
    training = {
        "ep_size": 4,
        "sp_size": 8,
        "n_gpus": 8,
        "ds_worker_config": {
            "ep_size": 4,
            "unchanged": "kept",
        },
        "ds_config": {
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu"},
            },
        },
    }
    monkeypatch.setenv(
        "PROBE_ENGINE_OVERRIDE",
        '{"ds_worker_config":{"ep_size":1},"ep_size":1,"ds_config":{"zero_optimization":{"stage":0}}}',
    )

    assert engine_knobs.apply_probe_engine_override(training) is training

    assert training["ep_size"] == 1
    assert training["sp_size"] == 8
    assert training["n_gpus"] == 8
    assert training["ds_worker_config"] == {
        "ep_size": 1,
        "unchanged": "kept",
    }
    assert training["ds_config"]["zero_optimization"] == {
        "stage": 0,
        "offload_optimizer": {"device": "cpu"},
    }
