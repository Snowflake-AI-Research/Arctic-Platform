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
"""The MoE runtime options (moe/runtime_options.py): the MoE loaders' options steer the runtime switches (offload,
sharding, W4A8, UltraEP, the online replicas, the expert placement, deepep_v2); an unset option leaves each switch's
environment variable in charge. CPU only."""

from __future__ import annotations

import importlib
import json
import os
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
    "w8a8_experts": ("kernels.fp8_grouped_w8a8", "w8a8_experts_requested", "ARL_W8A8_EXPERTS", True, False),
    "ultraep_spares": ("moe.ultraep_replicas", "ultraep_spares", "ARL_ULTRAEP_SPARES", 2, 0),
    # expert_replicas is set apart (test_online_replicas_*): it cannot be on together with ultraep_spares.
    "expert_replicas_ema": ("moe.expert_replicas", "replica_ema", "ARL_EXPERT_REPLICAS_EMA", 0.25, 0.5),
}
# glm_moe_dsa's own runtime fields (GlmMoeDsaOptions), which the other loaders' options refuse.
GLM_SWITCHES = {
    "shard_dense": ("moe.dense_shard", "dense_sharding_requested", "ARL_SHARD_DENSE", True, False),
}
ALL_SWITCHES = SWITCHES | GLM_SWITCHES
REPLICAS_ENV = "ARL_EXPERT_REPLICAS"
PLACEMENT_ENV = "ARL_EXPERT_PLACEMENT"


def _module(name):
    return importlib.import_module(f"arctic_platform.model.implementations.{name}")


def _env_value(value) -> str:
    return "1" if value is True else str(value)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for _, _, env, _, _ in ALL_SWITCHES.values():
        monkeypatch.delenv(env, raising=False)
    monkeypatch.delenv(REPLICAS_ENV, raising=False)
    monkeypatch.delenv(PLACEMENT_ENV, raising=False)
    monkeypatch.delenv("ULTRA_EP_IPC_SYMM", raising=False)
    yield
    apply_moe_runtime_options(MoeRuntimeOptions())  # leave every switch to its environment variable again


def _switches(switches=ALL_SWITCHES):
    return {field: getattr(_module(spec[0]), spec[1])() for field, spec in switches.items()}


@pytest.mark.parametrize("options_cls", [GlmMoeDsaOptions, GenericMoeOptions, Qwen3_5MoeOptions])
def test_loader_options_carry_the_runtime_fields(options_cls):
    switches = ALL_SWITCHES if options_cls is GlmMoeDsaOptions else SWITCHES
    options = options_cls.model_validate(
        {field: spec[3] for field, spec in switches.items()} | {"ultraep_ipc_symm": True}
    )
    apply_moe_runtime_options(options)
    assert _switches(switches) == {field: spec[3] for field, spec in switches.items()}
    assert os.environ["ULTRA_EP_IPC_SYMM"] == "1"


@pytest.mark.parametrize("options_cls", [MoeRuntimeOptions, GenericMoeOptions, Qwen3_5MoeOptions])
@pytest.mark.parametrize("field", sorted(GLM_SWITCHES) + ["shard_dense_group_size"])
def test_only_the_glm_moe_dsa_loader_takes_the_dense_shard_fields(options_cls, field):
    with pytest.raises(ValidationError):
        options_cls.model_validate({field: 2 if field == "shard_dense_group_size" else True})


def test_options_override_the_environment(monkeypatch):
    for spec in ALL_SWITCHES.values():
        monkeypatch.setenv(spec[2], _env_value(spec[3]))
    apply_moe_runtime_options(GlmMoeDsaOptions(**{field: spec[4] for field, spec in ALL_SWITCHES.items()}))
    assert _switches() == {field: spec[4] for field, spec in ALL_SWITCHES.items()}


def test_unset_options_leave_the_environment_in_charge(monkeypatch):
    apply_moe_runtime_options(MoeRuntimeOptions())
    assert _switches() == {field: spec[4] for field, spec in ALL_SWITCHES.items()}
    for spec in ALL_SWITCHES.values():
        monkeypatch.setenv(spec[2], _env_value(spec[3]))
    assert _switches() == {field: spec[3] for field, spec in ALL_SWITCHES.items()}


@pytest.mark.parametrize("value", ["", "0", "false", "False"])
def test_offload_environment_off_values(monkeypatch, value):
    monkeypatch.setenv("ARL_OFFLOAD_FROZEN_EXPERTS", value)
    apply_moe_runtime_options(MoeRuntimeOptions())
    assert not _module("moe.frozen_offload").frozen_expert_offload_requested()


def test_the_dense_shard_group_size_takes_the_option_then_the_environment(monkeypatch):
    dense_shard = _module("moe.dense_shard")
    monkeypatch.delenv(dense_shard.SHARD_DENSE_GROUP_SIZE_ENV, raising=False)
    apply_moe_runtime_options(MoeRuntimeOptions())
    assert dense_shard.configured_group_size() is None  # the ranks of one host
    monkeypatch.setenv(dense_shard.SHARD_DENSE_GROUP_SIZE_ENV, "4")
    assert dense_shard.configured_group_size() == 4
    apply_moe_runtime_options(GlmMoeDsaOptions(shard_dense=True, shard_dense_group_size=2))
    assert dense_shard.configured_group_size() == 2 and dense_shard.dense_sharding_requested()
    with pytest.raises(ValidationError):
        GlmMoeDsaOptions(shard_dense_group_size=1)


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
    """The trainable experts' UltraEP (moe/ultraep_replicas.py) and the frozen quantized one (moe/ultraep_frozen.py:
    block FP8, packed 4-bit) take the same loader option."""
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
    apply_moe_runtime_options(GenericMoeOptions())  # a load that sets neither leaves the environment in charge
    assert (runtime_options.deepep_v2_num_sms(32), runtime_options.deepep_v2_max_tokens(8192)) == (0, 16384)


@pytest.mark.parametrize("fields", [{"deepep_v2_num_sms": -1}, {"deepep_v2_max_tokens": 0}])
def test_bad_deepep_v2_settings_are_refused(fields):
    with pytest.raises(ValidationError):
        GlmMoeDsaOptions(**fields)


# The loaders that reject expert parallelism, and so have no MoE runtime.
_NO_EXPERT_PARALLEL = {"huggingface", "dflash2"}
LOADER_OPTIONS = sorted(
    {entry.options for name, entry in loader_mod._LOADERS.items() if name not in _NO_EXPERT_PARALLEL},
    key=lambda options: options.__name__,
)


@pytest.mark.parametrize("options_cls", LOADER_OPTIONS)
def test_every_moe_loader_takes_the_fields_unset_by_default(options_cls):
    assert issubclass(options_cls, MoeRuntimeOptions)
    options = options_cls()
    assert (options.ultraep_spares, options.ultraep_ipc_symm) == (None, None)
    assert (options.expert_replicas, options.expert_replicas_ema, options.expert_placement) == (None, None, None)
    assert (options.frozen_expert_offload, options.shard_frozen_dense, options.w4a8_experts) == (None, None, None)
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


@pytest.mark.parametrize("options_cls", LOADER_OPTIONS)
def test_every_moe_loader_takes_the_deepep_v2_settings(options_cls):
    assert options_cls(deepep_v2_num_sms=0, deepep_v2_max_tokens=4096).deepep_v2_num_sms == 0


@pytest.mark.parametrize("options_cls", LOADER_OPTIONS)
def test_every_moe_loader_refuses_deepep_v2_recompute_without_the_replay(options_cls):
    """A whole-block recompute dispatches through DeepEP V2 with its forward's handle, which the combine replay
    carries; every MoE loader refuses the backend under full checkpointing without it."""
    with pytest.raises(ValidationError, match="combine_replay"):
        options_cls(ep_comm_backend="deepep_v2", ac_config={"mode": "full"})
    assert options_cls(ep_comm_backend="deepep_v2", ac_config={"mode": "full", "combine_replay": True}).ac_config
    assert options_cls(ep_comm_backend="deepep_v2").ep_comm_backend == "deepep_v2"
    assert options_cls(ep_comm_backend="uccl", ac_config={"mode": "full"}).ac_config.combine_replay is False


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
