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

"""The conversion half of the MoE loaders runs without a process group and fills the cache the load then reads."""

from pathlib import Path
from types import SimpleNamespace

import pytest


class _FakePrimeModel:
    def state_dict(self):
        return {"model.layers.0.mlp.experts.w1": None}

    def is_hf_state_dict(self, state_dict):
        return "model.layers.0.mlp.experts.0.gate_proj.weight" in state_dict

    def is_prime_state_dict(self, state_dict):
        return "model.layers.0.mlp.experts.w1" in state_dict

    def convert_to_prime(self, state_dict):
        return state_dict

    def convert_to_hf(self, state_dict):
        return state_dict


@pytest.mark.parametrize("builder_name", ["qwen35", "glm52"])
@pytest.mark.parametrize("scope", ["node-local", "shared"])
@pytest.mark.parametrize("rank", [0, 8])
def test_prepare_weight_conversion_writes_with_the_load_writer_rule(monkeypatch, tmp_path, builder_name, scope, rank):
    import importlib

    builder = importlib.import_module(f"arctic_platform.model.implementations.{builder_name}.model_builder")
    source = tmp_path / "source"
    cache = tmp_path / "cache" / "prime"
    calls = []
    monkeypatch.setattr(builder, "PreTrainedModelPrimeRL", _FakePrimeModel)
    hf_keys = ["model.layers.0.mlp.experts.0.gate_proj.weight"]
    monkeypatch.setattr(builder, "load_state_dict_keys", lambda path: hf_keys)
    monkeypatch.setattr(builder, "resolve_conversion_cache_path", lambda config, path, fmt: cache)
    monkeypatch.setattr(builder, "conversion_cache_is_node_local", lambda path: scope == "node-local")
    monkeypatch.setattr(builder, "conversion_cache_ready", lambda path: False)
    monkeypatch.setattr(builder, "ensure_node_local_conversion_cache", lambda *a, **k: calls.append(("node", k)))
    monkeypatch.setattr(builder, "_write_conversion_cache", lambda *a, **k: calls.append(("shared", k)))
    config = SimpleNamespace(dequantize_checkpoint=False)

    result = builder.prepare_weight_conversion(_FakePrimeModel(), config, source, rank=rank, local_rank=0)

    assert result == cache
    if scope == "node-local":
        assert [(kind, k["rank"], k["local_rank"]) for kind, k in calls] == [("node", rank, 0)]
    else:
        assert [kind for kind, _ in calls] == (["shared"] if rank == 0 else [])


@pytest.mark.parametrize("builder_name", ["qwen35", "glm52"])
def test_prepare_weight_conversion_skips_a_checkpoint_in_the_model_format(monkeypatch, tmp_path, builder_name):
    import importlib

    builder = importlib.import_module(f"arctic_platform.model.implementations.{builder_name}.model_builder")
    monkeypatch.setattr(builder, "PreTrainedModelPrimeRL", _FakePrimeModel)
    monkeypatch.setattr(builder, "load_state_dict_keys", lambda path: ["model.layers.0.mlp.experts.w1"])
    unexpected = lambda *a, **k: pytest.fail("no conversion applies")  # noqa: E731
    monkeypatch.setattr(builder, "resolve_conversion_cache_path", unexpected)

    for model in (_FakePrimeModel(), object()):
        assert builder.prepare_weight_conversion(model, SimpleNamespace(), tmp_path, rank=0, local_rank=0) is None


def _adapter(events, *, random_init=False):
    from arctic_platform.model.implementations.moe.deepspeed_integration import MoEDeepSpeedAdapter

    def build_model_config(model_name, ep_size, dp_replicate, optimization_dtype, attn, options):
        events.append(("config", model_name, ep_size, dp_replicate, optimization_dtype, attn, options))
        return SimpleNamespace(debug=SimpleNamespace(random_init=random_init), optimization_dtype=optimization_dtype)

    def get_model(config, *, device, dtype):
        events.append(("model", device.type, dtype))
        return "meta-model"

    def prepare(model, config, source, *, rank, local_rank):
        events.append(("prepare", model, source, rank, local_rank))
        return Path("/cache/prime")

    unused = lambda *a, **k: pytest.fail("the load path is not part of preparation")  # noqa: E731
    return MoEDeepSpeedAdapter(
        dtype_map={"bfloat16": "bf16-dtype"},
        get_model=get_model,
        configure_moe_ep_backend=unused,
        configure_family_backend=unused,
        inject_lm_head=unused,
        apply_sequence_parallelism=unused,
        apply_ac=unused,
        load_dcp_from_hf=unused,
        reset_runtime_moe_buffers=unused,
        shared_expert_type=object,
        shared_expert_forward=unused,
        build_model_config=build_model_config,
        resolve_weight_source=lambda config: Path("/source"),
        prepare_weight_conversion=prepare,
    )


def _prepare(adapter, **overrides):
    from arctic_platform.model.implementations.moe.deepspeed_integration import prepare_moe_weights

    arguments = dict(
        model_name="org/model",
        optimization_dtype="bfloat16",
        attn_implementation="sdpa",
        ep_size=16,
        world_size=32,
        options="options",
        rank=8,
        local_rank=0,
    )
    return prepare_moe_weights(adapter, **{**arguments, **overrides})


def test_prepare_moe_weights_converts_from_a_meta_model_with_the_load_config():
    events = []

    assert _prepare(_adapter(events)) == Path("/cache/prime")
    assert events == [
        ("config", "org/model", 16, 2, "bfloat16", "sdpa", "options"),
        ("model", "meta", "bf16-dtype"),
        ("prepare", "meta-model", Path("/source"), 8, 0),
    ]


def test_prepare_moe_weights_skips_random_init_and_rejects_an_indivisible_world():
    events = []

    assert _prepare(_adapter(events, random_init=True)) is None
    assert [event[0] for event in events] == ["config"]
    with pytest.raises(ValueError, match="divisible"):
        _prepare(_adapter([]), world_size=24)


def test_prepare_model_weights_dispatches_to_the_resolved_loader_hook(monkeypatch):
    from arctic_platform.model import loader

    received = []
    monkeypatch.setitem(
        loader._LOADERS,
        "with-hook",
        loader._LoaderEntry(fn=None, matches=None, prepare_weights=lambda spec, **kw: received.append(kw) or "path"),
    )
    monkeypatch.setitem(loader._LOADERS, "without-hook", loader._LoaderEntry(fn=None, matches=None))

    hooked = loader.prepare_model_weights(SimpleNamespace(loader="with-hook"), world_size=32, rank=8, local_rank=0)
    plain = loader.prepare_model_weights(SimpleNamespace(loader="without-hook"), world_size=32, rank=8, local_rank=0)

    assert (hooked, plain) == ("path", None)
    assert received == [{"world_size": 32, "rank": 8, "local_rank": 0}]


@pytest.mark.parametrize("name", ["glm5_next", "qwen4_exp", "qwen3_5_moe", "generic_moe", "glm_moe_dsa"])
def test_every_converting_moe_loader_registers_a_weight_hook(name):
    import arctic_platform.model.loaders  # noqa: F401
    from arctic_platform.model import loader

    assert loader._LOADERS[name].prepare_weights is not None
