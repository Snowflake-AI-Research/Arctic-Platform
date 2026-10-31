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
"""A node's HF->Prime conversion runs before its ranks start, and the ranks' load then reuses it."""

import shutil

import pytest
import torch


def _qwen3_moe_checkpoint(tmp_path):
    transformers = pytest.importorskip("transformers")
    source = tmp_path / "Qwen3-MoE-tiny"
    config = transformers.Qwen3MoeConfig(vocab_size=64, hidden_size=16, intermediate_size=32, moe_intermediate_size=8,
                                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                                         num_experts=4, num_experts_per_tok=2)  # fmt: skip
    transformers.Qwen3MoeForCausalLM(config).save_pretrained(source)
    return source


def _forbid(monkeypatch, target, name, message):
    def forbidden(*_args, **_kwargs):
        raise AssertionError(message)

    monkeypatch.setattr(target, name, forbidden)


def test_prepare_moe_weights_converts_without_a_collective_and_the_rank_load_reuses_it(tmp_path, monkeypatch):
    from arctic_platform.model import conversion_cache_ready
    from arctic_platform.model import moe_model_spec
    from arctic_platform.model import prepare_moe_weights
    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen_ds
    from arctic_platform.model.implementations.qwen35 import model_builder
    from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions

    source = _qwen3_moe_checkpoint(tmp_path)
    options = {"attn": "sdpa", "weight_conversion_cache_dir": str(tmp_path / "cache")}
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", "node")
    _forbid(monkeypatch, torch.distributed, "barrier", "the per-node preparation ran a collective")
    assert prepare_moe_weights(model_name=str(source), ep_size=2, loader_options=options) is True

    [cache] = (tmp_path / "cache").glob("*/prime")
    assert conversion_cache_ready(cache)

    # What load_dcp_from_hf does on each rank: it reads the same directory and converts nothing.
    spec = moe_model_spec(model_name=str(source), ep_size=2, loader_options=options)
    model_config = qwen_ds._build_model_config(str(source), 2, 1, spec.dtype, spec.attn_implementation,
                                               Qwen3_5MoeOptions.model_validate(spec.loader_options))  # fmt: skip
    model = qwen_ds._generic_adapter().get_model(model_config, device=torch.device("meta"))
    _forbid(monkeypatch, model_builder, "_write_conversion_cache", "the rank converted again")
    assert model_builder._convert_snapshot(model, model_config) == (cache, True)


@pytest.mark.parametrize("scope", ["node", "shared"])
def test_prepare_moe_weights_reuses_a_prebaked_sibling_cache(tmp_path, monkeypatch, scope):
    from arctic_platform.model import prepare_moe_weights
    from arctic_platform.model.implementations.qwen35 import model_builder

    source = _qwen3_moe_checkpoint(tmp_path)
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", "node")
    baked = {"attn": "sdpa", "weight_conversion_cache_dir": str(tmp_path / "bake")}
    assert prepare_moe_weights(model_name=str(source), ep_size=2, loader_options=baked) is True
    [converted] = (tmp_path / "bake").glob("*/prime")
    shutil.move(converted, source / "prime")

    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", scope)
    _forbid(monkeypatch, model_builder, "_write_conversion_cache", "converted despite a pre-baked cache")
    options = {"attn": "sdpa", "weight_conversion_cache_dir": str(tmp_path / "cache")}
    assert prepare_moe_weights(model_name=str(source), ep_size=2, loader_options=options) is True
    assert not (tmp_path / "cache").exists()


def test_prepare_moe_weights_leaves_an_unwritten_shared_cache_to_the_ranks(tmp_path, monkeypatch):
    from arctic_platform.model import prepare_moe_weights

    source = _qwen3_moe_checkpoint(tmp_path)
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", "shared")
    options = {"attn": "sdpa", "weight_conversion_cache_dir": str(tmp_path / "cache")}
    assert prepare_moe_weights(model_name=str(source), ep_size=2, loader_options=options) is False
    assert not list(tmp_path.glob("cache/*/prime"))


def test_prepare_weights_reports_a_loader_without_preparation(tmp_path):
    from arctic_platform.model import ModelSpec
    from arctic_platform.model.loader import prepare_weights

    spec = ModelSpec(model_path_or_name=str(_qwen3_moe_checkpoint(tmp_path)))
    assert prepare_weights(spec) is False


def _flash_checkpoint(tmp_path, family):
    transformers = pytest.importorskip("transformers")
    if family == "glm5_next":
        from tests.model.test_glm5_next import _tiny_config

        model_cls = "Glm5NextForConditionalGeneration"
    else:
        from tests.model.test_qwen4_exp import _tiny_config

        model_cls = "Qwen4ExpForConditionalGeneration"
    source = tmp_path / family
    getattr(transformers, model_cls)(_tiny_config()).save_pretrained(source)
    return source


def _deepseek_checkpoint(tmp_path, family):
    """A tiny model's own weights in the published layout, which the load converts to Prime."""
    from arctic_platform.model.implementations.moe.weights import save_state_dict

    if family == "deepseek_v4":
        from arctic_platform.model.implementations.deepseek_v4 import DeepseekV4ForCausalLM as model_cls
        from tests.model.test_deepseek_v4 import _requires_sparse_attention_kernel
        from tests.model.test_deepseek_v4 import _tiny_config

        _requires_sparse_attention_kernel()
    else:
        from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41ForCausalLM as model_cls
        from tests.model.test_deepseek_v41_checkpoint import _tiny_config
    config = _tiny_config()
    model = model_cls._from_config(config, dtype=torch.bfloat16)
    source = tmp_path / family
    config.save_pretrained(source)
    save_state_dict(
        model.convert_to_hf({name: tensor.detach().clone() for name, tensor in model.state_dict().items()}), source
    )
    return source


@pytest.mark.parametrize("family", ["glm5_next", "qwen4_exp", "deepseek_v4", "deepseek_v41"])
def test_prepare_builds_on_meta_and_writes_the_cache(tmp_path, monkeypatch, family):
    from arctic_platform.model import conversion_cache_ready
    from arctic_platform.model import moe_model_spec
    from arctic_platform.model import prepare_moe_weights
    from arctic_platform.model.implementations.qwen35 import model_builder

    make_checkpoint = _deepseek_checkpoint if family.startswith("deepseek") else _flash_checkpoint
    source = make_checkpoint(tmp_path, family)
    options = {"weight_conversion_cache_dir": str(tmp_path / "cache")}
    assert moe_model_spec(model_name=str(source), ep_size=2, loader_options=options).loader == family
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", "node")
    _forbid(monkeypatch, torch.distributed, "barrier", "the per-node preparation ran a collective")
    assert prepare_moe_weights(model_name=str(source), ep_size=2, loader_options=options) is True

    [cache] = (tmp_path / "cache").glob("*/prime")
    assert conversion_cache_ready(cache)
    _forbid(monkeypatch, model_builder, "_write_conversion_cache", "converted a ready cache again")
    assert prepare_moe_weights(model_name=str(source), ep_size=2, loader_options=options) is True


@pytest.mark.parametrize("family", ["deepseek_v4", "deepseek_v41"])
@pytest.mark.parametrize("peft", [None, {"peft_type": "Lora", "r": 8}])
def test_deepseek_prepare_dequantizes_the_checkpoint_the_load_does(tmp_path, monkeypatch, family, peft):
    import importlib
    import json

    from arctic_platform.model import prepare_moe_weights

    integration = importlib.import_module(f"arctic_platform.model.implementations.{family}.deepspeed_integration")
    calls = []
    monkeypatch.setattr(integration, f"prepare_{family}_weights", lambda **kwargs: calls.append(kwargs) or True)
    source = tmp_path / family
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": family, "n_routed_experts": 8}))

    assert prepare_moe_weights(model_name=str(source), ep_size=8, peft_config=peft) is True
    assert calls[0]["dequantize_checkpoint"] is (peft is None)
