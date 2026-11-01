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

"""The MoE runtime options (moe/runtime_options.py) reach their modules, and their environment variables stay the
fallback. CPU only."""

import json
import sys
import types

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from arctic_platform.model import ModelSpec
from arctic_platform.model import ParallelismConfig
from arctic_platform.model import build_model
from arctic_platform.model import loader as loader_mod
from arctic_platform.model.implementations.moe import expert_replicas
from arctic_platform.model.implementations.moe import ultraep_replicas
from arctic_platform.model.implementations.moe.runtime_options import MoeRuntimeOptions
from arctic_platform.model.implementations.moe.runtime_options import apply_moe_runtime_options
from arctic_platform.model.loaders.generic_moe import GenericMoeOptions

# The loaders that reject expert parallelism, and so have no MoE runtime.
_NO_EXPERT_PARALLEL = {"huggingface", "dflash2"}
LOADER_OPTIONS = sorted(
    {entry.options for name, entry in loader_mod._LOADERS.items() if name not in _NO_EXPERT_PARALLEL},
    key=lambda options: options.__name__,
)
RUNTIME_ENV = (
    "ARL_ULTRAEP_SPARES",
    "ULTRA_EP_IPC_SYMM",
    "ARL_EXPERT_REPLICAS",
    "ARL_EXPERT_REPLICAS_EMA",
    "ARL_EXPERT_PLACEMENT",
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in RUNTIME_ENV:
        monkeypatch.delenv(name, raising=False)
    yield
    apply_moe_runtime_options(MoeRuntimeOptions())


@pytest.mark.parametrize("options_cls", LOADER_OPTIONS)
def test_every_moe_loader_takes_the_fields_unset_by_default(options_cls):
    assert issubclass(options_cls, MoeRuntimeOptions)
    options = options_cls()
    assert (options.ultraep_spares, options.ultraep_ipc_symm) == (None, None)
    assert (options.expert_replicas, options.expert_replicas_ema, options.expert_placement) == (None, None, None)
    options = options_cls(ultraep_spares=2, ultraep_ipc_symm=True)
    assert (options.ultraep_spares, options.ultraep_ipc_symm) == (2, True)
    assert options_cls(expert_replicas=2, expert_replicas_ema=1.0).expert_replicas == 2


@pytest.mark.parametrize(
    "fields",
    [
        {"ultraep_spares": -1},
        {"expert_replicas": -2},
        {"expert_replicas_ema": 0.0},
        {"expert_replicas_ema": 1.5},
        {"ultraep_spares": 2, "expert_replicas": 2},
    ],
)
def test_bad_values_are_refused(fields):
    with pytest.raises(ValidationError):
        GenericMoeOptions(**fields)


@pytest.mark.parametrize(
    "value",
    ["/nodes/placement.json", '{"ep_size": 2, "layers": {"0": [1, 0]}}', {"ep_size": 2, "layers": {"0": [1, 0]}}],
)
def test_expert_placement_takes_a_path_the_json_or_the_object(value):
    assert GenericMoeOptions(expert_placement=value).expert_placement == value


@pytest.mark.parametrize("value", ["", "  ", "{not json", '{"ep_size": 2}', {"layers": [1, 0]}, {"ep_size": 2}])
def test_malformed_expert_placements_are_refused(value):
    with pytest.raises(ValidationError):
        GenericMoeOptions(expert_placement=value)


def test_one_planner_may_be_off_while_the_other_is_on():
    assert GenericMoeOptions(ultraep_spares=2, expert_replicas=0).ultraep_spares == 2


def test_configured_values_win_and_unset_ones_fall_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("ARL_ULTRAEP_SPARES", "3")
    monkeypatch.setenv("ARL_EXPERT_REPLICAS", "1")
    monkeypatch.setenv("ARL_EXPERT_REPLICAS_EMA", "0.25")
    apply_moe_runtime_options(GenericMoeOptions())
    assert (ultraep_replicas.ultraep_spares(), expert_replicas.replica_spares(), expert_replicas.replica_ema()) == (
        3,
        1,
        0.25,
    )
    apply_moe_runtime_options(GenericMoeOptions(ultraep_spares=0, expert_replicas=4, expert_replicas_ema=0.75))
    assert (ultraep_replicas.ultraep_spares(), expert_replicas.replica_spares(), expert_replicas.replica_ema()) == (
        0,
        4,
        0.75,
    )
    # A later load that sets nothing goes back to the environment.
    apply_moe_runtime_options(GenericMoeOptions())
    assert ultraep_replicas.ultraep_spares() == 3


def test_defaults_without_config_or_environment():
    apply_moe_runtime_options(GenericMoeOptions())
    assert (ultraep_replicas.ultraep_spares(), expert_replicas.replica_spares(), expert_replicas.replica_ema()) == (
        0,
        0,
        0.5,
    )


def test_an_options_model_without_the_fields_counts_as_unset(monkeypatch):
    monkeypatch.setenv("ARL_ULTRAEP_SPARES", "2")
    apply_moe_runtime_options(types.SimpleNamespace(ep_comm_backend="uccl"))
    assert ultraep_replicas.ultraep_spares() == 2


@pytest.mark.parametrize("configured, inherited, seen", [(True, None, "1"), (False, "1", "0"), (None, "1", "1")])
def test_ipc_symm_is_in_the_environment_when_ultraep_allocates(monkeypatch, configured, inherited, seen):
    """UltraEP reads ULTRA_EP_IPC_SYMM inside ``Manager``; the configured mode must be there by then, and an unset
    one must leave the inherited value alone."""
    import os

    if inherited is not None:
        monkeypatch.setenv("ULTRA_EP_IPC_SYMM", inherited)
    observed = []

    class Manager:
        def __init__(self, **kwargs):
            observed.append(os.environ.get("ULTRA_EP_IPC_SYMM"))

    monkeypatch.setitem(sys.modules, "ultra_ep", types.SimpleNamespace(Manager=Manager))
    monkeypatch.setattr(ultraep_replicas.dist, "get_world_size", lambda group=None: 2)
    apply_moe_runtime_options(GenericMoeOptions(ultraep_spares=2, ultraep_ipc_symm=configured))
    w1 = torch.zeros(3, 4, 8)
    w2 = torch.zeros(3, 8, 4)
    state = ultraep_replicas.UltraEPState(object(), 3, ultraep_replicas.ultraep_spares(), w1, w2)
    assert state.spares == 2
    assert observed == [seen, seen]


def _checkpoint(tmp_path, model_type: str) -> str:
    path = tmp_path / model_type
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(path)


def test_the_spec_hands_the_fields_to_the_loader(monkeypatch, tmp_path):
    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as integration

    seen = {}

    def load(**kwargs):
        seen.update(kwargs)
        return nn.Identity()

    monkeypatch.setattr(integration, "load_generic_moe_model", load)
    spec = ModelSpec(
        model_path_or_name=_checkpoint(tmp_path, "qwen3_moe"),
        parallelism=ParallelismConfig(expert_parallel=2),
        loader_options={"ultraep_spares": 2, "ultraep_ipc_symm": True, "expert_replicas_ema": 0.5},
    )
    assert spec.loader_options["ultraep_spares"] == 2
    build_model(spec, parallel_groups={"ep_group": object()})
    options = seen["options"]
    assert (options.ultraep_spares, options.ultraep_ipc_symm, options.expert_replicas) == (2, True, None)


def test_load_moe_model_applies_the_options_before_it_builds_the_model(monkeypatch):
    from arctic_platform.model.implementations.moe import deepspeed_integration as common

    order = []
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

    class Mesh:
        @staticmethod
        def from_group(**kwargs):
            return object()

    def build_model_config(*args):
        order.append(("config", ultraep_replicas.ultraep_spares()))
        return types.SimpleNamespace()

    def family_loader(*args, **kwargs):
        order.append(("load", ultraep_replicas.ultraep_spares()))
        return nn.Identity()

    adapter = types.SimpleNamespace(build_model_config=build_model_config)
    common.load_moe_model(
        adapter,
        family_loader,
        model_name="m",
        optimization_dtype="bfloat16",
        attn_implementation="eager",
        ep_size=2,
        ep_group=object(),
        options=GenericMoeOptions(ep_comm_backend="uccl", ultraep_spares=2),
        patch_moe_detection=lambda: None,
        device_mesh_type=Mesh,
    )
    assert order == [("config", 2), ("load", 2)]


def test_deepep_v2_settings_take_the_option_then_the_environment_then_the_default(monkeypatch):
    from arctic_platform.model.implementations.moe import runtime_options

    monkeypatch.delenv("ARL_DEEPEP_V2_NUM_SMS", raising=False)
    monkeypatch.delenv("ARL_DEEPEP_V2_MAX_TOKENS", raising=False)
    apply_moe_runtime_options(GenericMoeOptions())
    assert (runtime_options.deepep_v2_num_sms(32), runtime_options.deepep_v2_max_tokens(8192)) == (32, 8192)
    monkeypatch.setenv("ARL_DEEPEP_V2_NUM_SMS", "0")
    monkeypatch.setenv("ARL_DEEPEP_V2_MAX_TOKENS", "16384")
    assert (runtime_options.deepep_v2_num_sms(32), runtime_options.deepep_v2_max_tokens(8192)) == (0, 16384)
    apply_moe_runtime_options(GenericMoeOptions(deepep_v2_num_sms=24, deepep_v2_max_tokens=20000))
    assert (runtime_options.deepep_v2_num_sms(32), runtime_options.deepep_v2_max_tokens(8192)) == (24, 20000)
    apply_moe_runtime_options(GenericMoeOptions(deepep_v2_num_sms=0))
    assert runtime_options.deepep_v2_num_sms(32) == 0


@pytest.mark.parametrize("fields", [{"deepep_v2_num_sms": -1}, {"deepep_v2_max_tokens": 0}])
def test_bad_deepep_v2_settings_are_refused(fields):
    with pytest.raises(ValidationError):
        GenericMoeOptions(**fields)


@pytest.mark.parametrize("options_cls", LOADER_OPTIONS)
def test_every_moe_loader_takes_the_deepep_v2_settings(options_cls):
    assert options_cls(deepep_v2_num_sms=0, deepep_v2_max_tokens=4096).deepep_v2_num_sms == 0
