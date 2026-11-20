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
"""The MoE runtime options (moe/runtime_options.py): the MoE loaders' options steer the runtime switches; an unset
option leaves each switch's environment variable in charge. CPU only."""

from __future__ import annotations

import importlib

import pytest
from pydantic import ValidationError

from arctic_platform.model.implementations.moe.runtime_options import MoeRuntimeOptions
from arctic_platform.model.implementations.moe.runtime_options import apply_moe_runtime_options
from arctic_platform.model.loaders.generic_moe import GenericMoeOptions
from arctic_platform.model.loaders.glm_moe_dsa import GlmMoeDsaOptions
from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions

# field -> (module, reader, environment variable, value when set, value when off)
SWITCHES = {
    "w4a8_experts": ("kernels.w4a8", "w4a8_experts_requested", "ARL_W4A8_EXPERTS", True, False),
    "w8a8_experts": ("kernels.fp8_grouped_w8a8", "w8a8_experts_requested", "ARL_W8A8_EXPERTS", True, False),
}


def _module(name):
    return importlib.import_module(f"arctic_platform.model.implementations.{name}")


def _env_value(value) -> str:
    return "1" if value is True else str(value)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for _, _, env, _, _ in SWITCHES.values():
        monkeypatch.delenv(env, raising=False)
    yield
    apply_moe_runtime_options(MoeRuntimeOptions())  # leave every switch to its environment variable again


def _switches(switches=SWITCHES):
    return {field: getattr(_module(spec[0]), spec[1])() for field, spec in switches.items()}


@pytest.mark.parametrize("options_cls", [GlmMoeDsaOptions, GenericMoeOptions, Qwen3_5MoeOptions])
def test_loader_options_carry_the_runtime_fields(options_cls):
    switches = SWITCHES
    options = options_cls.model_validate({field: spec[3] for field, spec in switches.items()})
    apply_moe_runtime_options(options)
    assert _switches(switches) == {field: spec[3] for field, spec in switches.items()}


def test_options_override_the_environment(monkeypatch):
    for spec in SWITCHES.values():
        monkeypatch.setenv(spec[2], _env_value(spec[3]))
    apply_moe_runtime_options(GlmMoeDsaOptions(**{field: spec[4] for field, spec in SWITCHES.items()}))
    assert _switches() == {field: spec[4] for field, spec in SWITCHES.items()}


def test_unset_options_leave_the_environment_in_charge(monkeypatch):
    apply_moe_runtime_options(MoeRuntimeOptions())
    assert _switches() == {field: spec[4] for field, spec in SWITCHES.items()}
    for spec in SWITCHES.values():
        monkeypatch.setenv(spec[2], _env_value(spec[3]))
    assert _switches() == {field: spec[3] for field, spec in SWITCHES.items()}


def test_unknown_runtime_fields_are_refused():
    with pytest.raises(ValidationError):
        GlmMoeDsaOptions.model_validate({"frozen_expert_ofload": True})


def test_w8a8_experts_reach_the_experts_built_after_the_load():
    """GroupedExperts read the switch when they are built, which the loader does after applying the options."""
    from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts

    apply_moe_runtime_options(GlmMoeDsaOptions(w8a8_experts=True))
    assert GroupedExperts(256, 128, 2, use_grouped_mm=True, fp8_block_size=128).w8a8
    # A checkpoint without block-FP8 experts (BF16, or packed 4-bit) is refused rather than run W8A16.
    with pytest.raises(ValueError, match="w8a8_experts runs block-FP8 routed experts"):
        GroupedExperts(256, 128, 2, use_grouped_mm=True)
    with pytest.raises(ValueError, match="w8a8_experts runs block-FP8 routed experts"):
        GroupedExperts(256, 128, 2, use_grouped_mm=True, fp8_block_size=128, packed4="int4")
