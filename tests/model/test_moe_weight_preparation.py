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

import importlib
import json
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
    monkeypatch.setattr(builder, "wait_for_conversion_cache", lambda path: calls.append(("wait", {})))
    config = SimpleNamespace(dequantize_checkpoint=False)

    result = builder.prepare_weight_conversion(_FakePrimeModel(), config, source, rank=rank, local_rank=0)

    assert result == cache
    if scope == "node-local":
        assert [(kind, k["rank"], k["local_rank"]) for kind, k in calls] == [("node", rank, 0)]
    else:
        assert [kind for kind, _ in calls] == (["node"] if rank == 0 else ["wait"])


def test_shared_conversion_cache_waits_for_publication_and_times_out(monkeypatch, tmp_path):
    from arctic_platform.model.implementations.moe import conversion_cache

    cache = tmp_path / "prime"
    checks = iter([False, False, True])
    sleeps = []
    monkeypatch.setattr(conversion_cache, "conversion_cache_ready", lambda path: next(checks))
    monkeypatch.setattr(conversion_cache.time, "sleep", sleeps.append)
    conversion_cache.wait_for_conversion_cache(cache, timeout=10)
    assert sleeps == [0.25, 0.25]
    monkeypatch.setattr(conversion_cache, "conversion_cache_ready", lambda path: False)
    with pytest.raises(TimeoutError, match="did not become ready"):
        conversion_cache.wait_for_conversion_cache(cache, timeout=0)


@pytest.mark.parametrize("builder_name", ["qwen35", "glm52"])
def test_prepare_weight_conversion_selects_the_checkpoint_streamer(monkeypatch, tmp_path, builder_name):
    import importlib

    builder = importlib.import_module(f"arctic_platform.model.implementations.{builder_name}.model_builder")
    model = _FakePrimeModel()
    stream = lambda source, target: None  # noqa: E731
    model.convert_to_prime_checkpoint = stream
    calls = []
    monkeypatch.setattr(builder, "PreTrainedModelPrimeRL", _FakePrimeModel)
    monkeypatch.setattr(
        builder, "load_state_dict_keys", lambda path: ["model.layers.0.mlp.experts.0.gate_proj.weight"]
    )
    monkeypatch.setattr(builder, "resolve_conversion_cache_path", lambda *a: tmp_path / "prime")
    monkeypatch.setattr(builder, "conversion_cache_is_node_local", lambda path: False)
    monkeypatch.setattr(builder, "ensure_node_local_conversion_cache", lambda *a, **kw: calls.append(kw))
    config = SimpleNamespace(dequantize_checkpoint=False)
    assert builder.prepare_weight_conversion(model, config, tmp_path, rank=0, local_rank=0) == tmp_path / "prime"
    assert calls[0]["stream_convert_fn"] is stream


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


@pytest.mark.parametrize(
    "name, tiny_config, implementation, preparer",
    [
        ("glm5_next", "tiny_glm5_next_config", "glm53", "prepare_glm5_next_weights"),
        ("qwen4_exp", "tiny_qwen4_exp_config", "qwen38", "prepare_qwen4_exp_weights"),
    ],
)
def test_flash_moe_weight_hook_takes_the_spec_its_loader_defaulted(
    monkeypatch, tmp_path, name, tiny_config, implementation, preparer
):
    # The spec stores every FlashMoeOptions default, fp32_routed_weighting among them, and the hook revalidates it.
    import importlib

    from arctic_platform.model import loader
    from arctic_platform.model.moe_spec import moe_model_spec

    (tmp_path / "config.json").write_text(json.dumps(dict(model_type=name)))
    received = {}
    module = importlib.import_module(f"arctic_platform.model.implementations.{implementation}.deepspeed_integration")
    monkeypatch.setattr(module, preparer, lambda **kwargs: received.update(kwargs) or tmp_path / "cache")
    spec = moe_model_spec(model_name=str(tmp_path), ep_size=8, sp_size=8)

    path = loader.prepare_model_weights(spec, world_size=16, rank=8, local_rank=0)

    assert spec.loader == name
    assert spec.loader_options["fp32_routed_weighting"] is False
    assert path == tmp_path / "cache"
    assert received["options"].fp32_routed_weighting is False
    assert (received["ep_size"], received["world_size"], received["rank"]) == (8, 16, 8)


@pytest.mark.parametrize("name", ["glm5_next", "qwen4_exp", "qwen3_5_moe", "generic_moe", "glm_moe_dsa"])
def test_every_converting_moe_loader_registers_a_weight_hook(name):
    import arctic_platform.model.loaders  # noqa: F401
    from arctic_platform.model import loader

    assert loader._LOADERS[name].prepare_weights is not None


@pytest.mark.parametrize("streaming", [False, True])
def test_preparation_rejects_incomplete_conversion_before_returning_a_path(monkeypatch, tmp_path, streaming):
    from arctic_platform.model.implementations.qwen35 import model_builder as builder

    model = _FakePrimeModel()

    def incomplete(*args):
        target = args[-1]
        target.mkdir(parents=True)
        (target / "model.safetensors.index.json").write_text(
            json.dumps(dict(weight_map=dict(weight="missing.safetensors")))
        )

    if streaming:
        model.convert_to_prime_checkpoint = incomplete
    monkeypatch.setattr(builder, "PreTrainedModelPrimeRL", _FakePrimeModel)
    monkeypatch.setattr(
        builder, "load_state_dict_keys", lambda path: ["model.layers.0.mlp.experts.0.gate_proj.weight"]
    )
    monkeypatch.setattr(builder, "load_state_dict", lambda path: {})
    monkeypatch.setattr(builder, "save_state_dict", incomplete)
    cache = tmp_path / "prime"
    monkeypatch.setattr(builder, "resolve_conversion_cache_path", lambda *args: cache)
    monkeypatch.setattr(builder, "conversion_cache_is_node_local", lambda path: True)
    with pytest.raises(RuntimeError, match="did not publish a complete checkpoint"):
        builder.prepare_weight_conversion(
            model, SimpleNamespace(dequantize_checkpoint=False), tmp_path, rank=0, local_rank=0
        )
    assert not cache.exists()


def _tiny_preparation_model(family):
    import torch

    if family == "glm5_next":
        from test_glm5_next import _tiny_config

        from arctic_platform.model.implementations.glm53.modeling_glm5_next import (
            Glm5NextForConditionalGenerationPrimeRL,
        )

        config, model_type = _tiny_config(), Glm5NextForConditionalGenerationPrimeRL
    elif family == "qwen4_exp":
        from test_qwen4_exp import _tiny_config

        from arctic_platform.model.implementations.qwen38.modeling_qwen4_exp import (
            Qwen4ExpForConditionalGenerationPrimeRL,
        )

        config, model_type = _tiny_config(), Qwen4ExpForConditionalGenerationPrimeRL
    elif family == "qwen3_5_moe":
        from test_qwen35_moe_numerics import _tiny_config

        from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe import Qwen3_5MoeForCausalLM

        config, model_type = _tiny_config(), Qwen3_5MoeForCausalLM
    else:
        if family == "glm_moe_dsa":
            from arctic_platform.model.implementations.glm52.models.glm_moe_dsa import GlmMoeDsaConfig as Config
            from arctic_platform.model.implementations.glm52.models.glm_moe_dsa import (
                GlmMoeDsaForCausalLM as model_type,
            )
        else:
            from arctic_platform.model.implementations.qwen3_moe.configuration_qwen3_moe import (
                Qwen3MoeConfig as Config,
            )
            from arctic_platform.model.implementations.qwen3_moe.modeling_qwen3_moe import (
                Qwen3MoeForCausalLM as model_type,
            )

        config = Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_experts=4,
            n_routed_experts=4,
            num_experts_per_tok=2,
            first_k_dense_replace=0,
            q_lora_rank=8,
            kv_lora_rank=4,
            qk_rope_head_dim=4,
            qk_nope_head_dim=4,
            v_head_dim=4,
            index_n_heads=2,
            index_head_dim=4,
            index_topk=4,
            pad_token_id=0,
            use_grouped_mm=False,
        )
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        return model_type(config)


@pytest.mark.parametrize("family", ["qwen3_5_moe", "generic_moe", "glm_moe_dsa", "glm5_next", "qwen4_exp"])
def test_real_family_preparation_and_loading_choose_the_same_cache(monkeypatch, tmp_path, family):
    from arctic_platform.model.implementations.moe.layers.lm_head import inject_prime_lm_head

    builder_name = "glm52" if family == "glm_moe_dsa" else "qwen35"
    builder = importlib.import_module(f"arctic_platform.model.implementations.{builder_name}.model_builder")
    model = _tiny_preparation_model(family)
    hf_keys = dict.fromkeys(
        [
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.gate_up_proj",
        ]
    )
    assert model.is_hf_state_dict(hf_keys)
    monkeypatch.setattr(builder, "load_state_dict_keys", lambda path: hf_keys)
    source, cache = tmp_path / "source", tmp_path / "prime"
    monkeypatch.setattr(builder, "resolve_weight_source", lambda config: source)
    monkeypatch.setattr(builder, "resolve_conversion_cache_path", lambda *args: cache)
    monkeypatch.setattr(builder, "conversion_cache_is_node_local", lambda path: True)
    monkeypatch.setattr(builder, "conversion_cache_ready", lambda path: True)
    monkeypatch.setattr(builder, "ensure_node_local_conversion_cache", lambda *args, **kwargs: None)
    config = SimpleNamespace(
        fsdp_cpu_offload=True, debug=SimpleNamespace(random_init=False), dequantize_checkpoint=False
    )
    prepared = builder.prepare_weight_conversion(model, config, source, rank=0, local_rank=0)
    inject_prime_lm_head(model, chunk_size=2, fused_cross_entropy=False, fp32_lm_head=False)
    loaded = []
    monkeypatch.setattr(builder, "get_world", lambda: SimpleNamespace(rank=0, local_rank=0))
    monkeypatch.setattr(builder.torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(builder, "dcp_load", lambda state, *, storage_reader: loaded.append(Path(storage_reader.path)))
    monkeypatch.setattr(builder, "_move_buffers_to_cuda", lambda *args: None)
    monkeypatch.setattr(builder, "_init_buffers_post_meta", lambda *args: None, raising=False)
    monkeypatch.setattr(type(model), "init_buffers_post_meta", lambda self: None, raising=False)
    builder.load_dcp_from_hf(model, config, None)
    assert prepared == cache
    assert loaded == [prepared]
