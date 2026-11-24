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
option turns its switch off. CPU only."""

from __future__ import annotations

import importlib

import pytest

from arctic_platform.model.implementations.moe.runtime_options import MoeRuntimeOptions
from arctic_platform.model.implementations.moe.runtime_options import apply_moe_runtime_options
from arctic_platform.model.loaders.generic_moe import GenericMoeOptions
from arctic_platform.model.loaders.glm_moe_dsa import GlmMoeDsaOptions
from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions

# field -> (module, reader, value when set, value when off)
SWITCHES = {
    "w8a8_experts": ("kernels.fp8_grouped_w8a8", "w8a8_experts_requested", True, False),
}


def _module(name):
    return importlib.import_module(f"arctic_platform.model.implementations.{name}")


@pytest.fixture(autouse=True)
def _clean():
    yield
    apply_moe_runtime_options(MoeRuntimeOptions())  # every switch off again


def _switches(switches=SWITCHES):
    return {field: getattr(_module(spec[0]), spec[1])() for field, spec in switches.items()}


@pytest.mark.parametrize("options_cls", [GlmMoeDsaOptions, GenericMoeOptions, Qwen3_5MoeOptions])
def test_loader_options_carry_the_runtime_fields(options_cls):
    switches = SWITCHES
    options = options_cls.model_validate({field: spec[2] for field, spec in switches.items()})
    apply_moe_runtime_options(options)
    assert _switches(switches) == {field: spec[2] for field, spec in switches.items()}
