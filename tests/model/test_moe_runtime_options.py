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
"""The MoE runtime options (moe/runtime_options.py): glm_moe_dsa loader options steer the runtime switches; an unset
option leaves each switch's environment variable in charge. CPU only."""

from __future__ import annotations

import importlib
import os
import sys
import types

import pytest
import torch
from pydantic import ValidationError

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
    "ultraep_spares": ("moe.ultraep_replicas", "ultraep_spares", "ARL_ULTRAEP_SPARES", 2, 0),
    # expert_replicas is set apart (test_online_replicas_*): it cannot be on together with ultraep_spares.
    "expert_replicas_ema": ("moe.expert_replicas", "replica_ema", "ARL_EXPERT_REPLICAS_EMA", 0.25, 0.5),
}
REPLICAS_ENV = "ARL_EXPERT_REPLICAS"
PLACEMENT_ENV = "ARL_EXPERT_PLACEMENT"


def _module(name):
    return importlib.import_module(f"arctic_platform.model.implementations.{name}")


def _env_value(value) -> str:
    return "1" if value is True else str(value)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for _, _, env, _, _ in SWITCHES.values():
        monkeypatch.delenv(env, raising=False)
    monkeypatch.delenv(REPLICAS_ENV, raising=False)
    monkeypatch.delenv(PLACEMENT_ENV, raising=False)
    monkeypatch.delenv("ULTRA_EP_IPC_SYMM", raising=False)
    yield
    apply_moe_runtime_options(MoeRuntimeOptions())  # leave every switch to its environment variable again


def _switches():
    return {field: getattr(_module(spec[0]), spec[1])() for field, spec in SWITCHES.items()}


def test_glm_loader_options_carry_the_runtime_fields():
    options = GlmMoeDsaOptions.model_validate(
        {field: spec[3] for field, spec in SWITCHES.items()} | {"ultraep_ipc_symm": True}
    )
    apply_moe_runtime_options(options)
    assert _switches() == {field: spec[3] for field, spec in SWITCHES.items()}
    assert os.environ["ULTRA_EP_IPC_SYMM"] == "1"


def test_options_override_the_environment(monkeypatch):
    for spec in SWITCHES.values():
        monkeypatch.setenv(spec[2], _env_value(spec[3]))
    apply_moe_runtime_options(MoeRuntimeOptions(**{field: spec[4] for field, spec in SWITCHES.items()}))
    assert _switches() == {field: spec[4] for field, spec in SWITCHES.items()}


def test_unset_options_leave_the_environment_in_charge(monkeypatch):
    apply_moe_runtime_options(MoeRuntimeOptions())
    assert _switches() == {field: spec[4] for field, spec in SWITCHES.items()}
    for spec in SWITCHES.values():
        monkeypatch.setenv(spec[2], _env_value(spec[3]))
    assert _switches() == {field: spec[3] for field, spec in SWITCHES.items()}


@pytest.mark.parametrize("value", ["", "0", "false", "False"])
def test_offload_environment_off_values(monkeypatch, value):
    monkeypatch.setenv("ARL_OFFLOAD_FROZEN_EXPERTS", value)
    apply_moe_runtime_options(MoeRuntimeOptions())
    assert not _module("moe.frozen_offload").frozen_expert_offload_requested()


@pytest.mark.parametrize("options_cls", [GenericMoeOptions, Qwen3_5MoeOptions])
def test_other_moe_loaders_refuse_the_runtime_fields(options_cls, monkeypatch):
    """Only glm_moe_dsa installs these modules; the other MoE loaders refuse the fields instead of ignoring them."""
    fields = {field: spec[3] for field, spec in SWITCHES.items()} | {
        "ultraep_ipc_symm": True,
        "expert_replicas": 2,
        "expert_placement": "/nodes/placement.json",
    }
    for field, value in fields.items():
        with pytest.raises(ValidationError):
            options_cls.model_validate({field: value})
    # Their loads leave every switch to its environment variable.
    for spec in SWITCHES.values():
        monkeypatch.setenv(spec[2], _env_value(spec[3]))
    apply_moe_runtime_options(options_cls())
    assert _switches() == {field: spec[3] for field, spec in SWITCHES.items()}


def test_ipc_symm_is_written_for_ultraep():
    apply_moe_runtime_options(MoeRuntimeOptions(ultraep_ipc_symm=True))
    assert os.environ["ULTRA_EP_IPC_SYMM"] == "1"
    apply_moe_runtime_options(MoeRuntimeOptions(ultraep_ipc_symm=False))
    assert os.environ["ULTRA_EP_IPC_SYMM"] == "0"
    apply_moe_runtime_options(MoeRuntimeOptions())  # unset: UltraEP's own variable stays as it is
    assert os.environ["ULTRA_EP_IPC_SYMM"] == "0"


def test_unknown_runtime_fields_are_refused():
    with pytest.raises(ValidationError):
        GlmMoeDsaOptions.model_validate({"frozen_expert_ofload": True})
    with pytest.raises(ValidationError):
        MoeRuntimeOptions(ultraep_spares=-1)
    for fields in ({"expert_replicas": -1}, {"expert_replicas_ema": 0.0}, {"expert_replicas_ema": 1.5}):
        with pytest.raises(ValidationError):
            GlmMoeDsaOptions.model_validate(fields)
    # The W4AFP8 ladder's other switches are not on this branch.
    for field in ("shard_frozen_dense", "w4a8_experts"):
        with pytest.raises(ValidationError):
            GlmMoeDsaOptions.model_validate({field: True})


def test_the_spare_planners_are_exclusive():
    """UltraEP's spares and the online replicas both fill the spare expert slots."""
    with pytest.raises(ValidationError, match="at most one"):
        GlmMoeDsaOptions(ultraep_spares=2, expert_replicas=2)
    assert GlmMoeDsaOptions(ultraep_spares=2, expert_replicas=0).ultraep_spares == 2
    assert GlmMoeDsaOptions(ultraep_spares=0, expert_replicas=2).expert_replicas == 2


def test_online_replicas_take_the_option_then_the_environment(monkeypatch):
    replicas = _module("moe.expert_replicas")
    apply_moe_runtime_options(GlmMoeDsaOptions())
    assert (replicas.replica_spares(), replicas.replica_ema()) == (0, 0.5)
    monkeypatch.setenv(REPLICAS_ENV, "1")
    assert replicas.replica_spares() == 1
    apply_moe_runtime_options(GlmMoeDsaOptions(expert_replicas=4, expert_replicas_ema=0.75))
    assert (replicas.replica_spares(), replicas.replica_ema()) == (4, 0.75)
    apply_moe_runtime_options(GlmMoeDsaOptions())  # a later load that sets nothing goes back to the environment
    assert replicas.replica_spares() == 1


def test_both_ultraep_paths_read_one_spare_count(monkeypatch):
    """The trainable experts' UltraEP (moe/ultraep_replicas.py) and the frozen block-FP8 one (moe/ultraep_frozen.py)
    take the same loader option."""
    frozen = _module("moe.ultraep_frozen")
    trainable = _module("moe.ultraep_replicas")
    apply_moe_runtime_options(GlmMoeDsaOptions(ultraep_spares=3))
    assert frozen.ultraep_spares() == trainable.ultraep_spares() == 3
    monkeypatch.setenv("ARL_ULTRAEP_SPARES", "1")
    apply_moe_runtime_options(GlmMoeDsaOptions())
    assert frozen.ultraep_spares() == trainable.ultraep_spares() == 1


@pytest.mark.parametrize("configured, inherited, seen", [(True, None, "1"), (False, "1", "0"), (None, "1", "1")])
def test_ipc_symm_is_in_the_environment_when_ultraep_allocates(monkeypatch, configured, inherited, seen):
    """UltraEP reads ULTRA_EP_IPC_SYMM inside ``Manager``; the configured mode must be there by then, and an unset
    one must leave the inherited value alone."""
    trainable = _module("moe.ultraep_replicas")
    if inherited is not None:
        monkeypatch.setenv("ULTRA_EP_IPC_SYMM", inherited)
    observed = []

    class Manager:
        def __init__(self, **kwargs):
            observed.append(os.environ.get("ULTRA_EP_IPC_SYMM"))

    monkeypatch.setitem(sys.modules, "ultra_ep", types.SimpleNamespace(Manager=Manager))
    monkeypatch.setattr(trainable.dist, "get_world_size", lambda group=None: 2)
    apply_moe_runtime_options(GlmMoeDsaOptions(ultraep_spares=2, ultraep_ipc_symm=configured))
    state = trainable.UltraEPState(object(), 3, trainable.ultraep_spares(), torch.zeros(3, 4, 8), torch.zeros(3, 8, 4))
    assert state.spares == 2
    assert observed == [seen, seen]


@pytest.mark.parametrize(
    "value",
    ["/nodes/placement.json", '{"ep_size": 2, "layers": {"0": [1, 0]}}', {"ep_size": 2, "layers": {"0": [1, 0]}}],
)
def test_expert_placement_takes_a_path_the_json_or_the_object(value):
    assert GlmMoeDsaOptions(expert_placement=value).expert_placement == value


@pytest.mark.parametrize("value", ["", "  ", "{not json", '{"ep_size": 2}', {"layers": [1, 0]}, {"ep_size": 2}])
def test_malformed_expert_placements_are_refused(value):
    with pytest.raises(ValidationError):
        GlmMoeDsaOptions(expert_placement=value)


def test_the_placement_option_wins_and_the_environment_is_the_fallback(monkeypatch):
    placement = _module("moe.expert_placement")
    configured = {"ep_size": 2, "layers": {"3": [1, 0]}}
    monkeypatch.setenv(PLACEMENT_ENV, "/nodes/placement.json")
    apply_moe_runtime_options(GlmMoeDsaOptions(expert_placement=configured))
    assert placement.placement_value() == configured
    apply_moe_runtime_options(GlmMoeDsaOptions())
    assert placement.placement_value() == "/nodes/placement.json"
    monkeypatch.delenv(PLACEMENT_ENV)
    assert placement.placement_value() is None


def test_deepep_v2_settings_take_the_option_then_the_environment_then_the_default(monkeypatch):
    runtime_options = _module("moe.runtime_options")
    monkeypatch.delenv("ARL_DEEPEP_V2_NUM_SMS", raising=False)
    monkeypatch.delenv("ARL_DEEPEP_V2_MAX_TOKENS", raising=False)
    apply_moe_runtime_options(GlmMoeDsaOptions())
    assert (runtime_options.deepep_v2_num_sms(32), runtime_options.deepep_v2_max_tokens(8192)) == (32, 8192)
    monkeypatch.setenv("ARL_DEEPEP_V2_NUM_SMS", "0")
    monkeypatch.setenv("ARL_DEEPEP_V2_MAX_TOKENS", "16384")
    assert (runtime_options.deepep_v2_num_sms(32), runtime_options.deepep_v2_max_tokens(8192)) == (0, 16384)
    apply_moe_runtime_options(GlmMoeDsaOptions(deepep_v2_num_sms=24, deepep_v2_max_tokens=20000))
    assert (runtime_options.deepep_v2_num_sms(32), runtime_options.deepep_v2_max_tokens(8192)) == (24, 20000)
    apply_moe_runtime_options(GlmMoeDsaOptions(deepep_v2_num_sms=0))
    assert runtime_options.deepep_v2_num_sms(32) == 0
    apply_moe_runtime_options(GenericMoeOptions())  # a loader without the fields leaves the environment in charge
    assert (runtime_options.deepep_v2_num_sms(32), runtime_options.deepep_v2_max_tokens(8192)) == (0, 16384)


@pytest.mark.parametrize("fields", [{"deepep_v2_num_sms": -1}, {"deepep_v2_max_tokens": 0}])
def test_bad_deepep_v2_settings_are_refused(fields):
    with pytest.raises(ValidationError):
        GlmMoeDsaOptions(**fields)
