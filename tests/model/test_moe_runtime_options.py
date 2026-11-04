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
"""The MoE runtime options (moe/runtime_options.py): loader options steer the offload, sharding, W4A8 and UltraEP
switches; an unset option leaves each switch's environment variable in charge. CPU only. A switch whose module this
branch does not have is checked to be refused when set."""

from __future__ import annotations

import importlib
import os

import pytest

from arctic_platform.model.implementations.moe.runtime_options import MoeRuntimeOptions
from arctic_platform.model.implementations.moe.runtime_options import apply_moe_runtime_options
from arctic_platform.model.loaders.generic_moe import GenericMoeOptions
from arctic_platform.model.loaders.glm_moe_dsa import GlmMoeDsaOptions
from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions

# field -> (module, reader, environment variable, value when set, value when off)
SWITCHES = {
    "frozen_expert_offload": (
        "moe.frozen_offload",
        "frozen_expert_offload_requested",
        "ARL_OFFLOAD_FROZEN_EXPERTS",
        True,
        False,
    ),
    "shard_frozen_dense": (
        "moe.frozen_shard",
        "frozen_dense_sharding_requested",
        "ARL_SHARD_FROZEN_DENSE",
        True,
        False,
    ),
    "w4a8_experts": ("kernels.w4a8", "w4a8_experts_requested", "ARL_W4A8_EXPERTS", True, False),
    "ultraep_spares": ("moe.ultraep_packed4", "ultraep_spares", "ARL_ULTRAEP_SPARES", 2, 0),
}


def _module(name):
    try:
        return importlib.import_module(f"arctic_platform.model.implementations.{name}")
    except ImportError:
        return None


PRESENT = {field: spec for field, spec in SWITCHES.items() if _module(spec[0]) is not None}
ABSENT = sorted(set(SWITCHES) - set(PRESENT))


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for _, _, env, _, _ in SWITCHES.values():
        monkeypatch.delenv(env, raising=False)
    monkeypatch.delenv("ULTRA_EP_IPC_SYMM", raising=False)
    yield
    apply_moe_runtime_options(MoeRuntimeOptions())  # leave every switch to its environment variable again


def _switches():
    return {field: getattr(_module(spec[0]), spec[1])() for field, spec in PRESENT.items()}


@pytest.mark.parametrize("options_cls", [GlmMoeDsaOptions, GenericMoeOptions, Qwen3_5MoeOptions])
def test_loader_options_carry_the_runtime_fields(options_cls):
    options = options_cls.model_validate(
        {field: spec[3] for field, spec in PRESENT.items()} | {"ultraep_ipc_symm": True}
    )
    apply_moe_runtime_options(options)
    assert _switches() == {field: spec[3] for field, spec in PRESENT.items()}


def test_options_override_the_environment(monkeypatch):
    for field, spec in PRESENT.items():
        monkeypatch.setenv(spec[2], str(spec[3]) if not isinstance(spec[3], bool) else "1")
    apply_moe_runtime_options(MoeRuntimeOptions(**{field: spec[4] for field, spec in PRESENT.items()}))
    assert _switches() == {field: spec[4] for field, spec in PRESENT.items()}


def test_unset_options_leave_the_environment_in_charge(monkeypatch):
    apply_moe_runtime_options(MoeRuntimeOptions())
    assert _switches() == {field: spec[4] for field, spec in PRESENT.items()}
    for field, spec in PRESENT.items():
        monkeypatch.setenv(spec[2], str(spec[3]) if not isinstance(spec[3], bool) else "1")
    assert _switches() == {field: spec[3] for field, spec in PRESENT.items()}


@pytest.mark.parametrize("field", ABSENT or ["none"])
def test_a_switch_this_branch_lacks_is_refused_when_set(field):
    if field == "none":
        pytest.skip("every switch's module is on this branch")
    apply_moe_runtime_options(MoeRuntimeOptions(**{field: SWITCHES[field][4]}))  # off: accepted
    with pytest.raises(ValueError, match="this branch has no"):
        apply_moe_runtime_options(MoeRuntimeOptions(**{field: SWITCHES[field][3]}))


def test_ipc_symm_is_written_for_ultraep():
    apply_moe_runtime_options(MoeRuntimeOptions(ultraep_ipc_symm=True))
    assert os.environ["ULTRA_EP_IPC_SYMM"] == "1"
    apply_moe_runtime_options(MoeRuntimeOptions(ultraep_ipc_symm=False))
    assert os.environ["ULTRA_EP_IPC_SYMM"] == "0"


def test_unknown_runtime_fields_are_refused():
    with pytest.raises(ValueError):
        GlmMoeDsaOptions.model_validate({"frozen_expert_ofload": True})
    with pytest.raises(ValueError):
        MoeRuntimeOptions(ultraep_spares=-1)
