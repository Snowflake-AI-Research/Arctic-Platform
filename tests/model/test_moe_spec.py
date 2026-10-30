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

"""MoE training options become one ModelSpec."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from arctic_platform.model import ModelSpec
from arctic_platform.model import ParallelismConfig
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.loaders.flash_moe import _matches as flash_matches
from arctic_platform.model.loaders.qwen3_5_moe import _matches as qwen_matches
from arctic_platform.model.moe_spec import hf_text_model_type
from arctic_platform.model.moe_spec import is_glm_moe_model
from arctic_platform.model.moe_spec import moe_model_spec


def _write_config(path, payload):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(payload))
    return str(path)


def test_text_model_type_reads_a_nested_glm_config(tmp_path):
    checkpoint = _write_config(
        tmp_path / "glm",
        {"model_type": "glm4v_moe", "text_config": {"model_type": "glm_moe_dsa"}},
    )
    assert hf_text_model_type(checkpoint) == "glm_moe_dsa"
    assert is_glm_moe_model(checkpoint)


@pytest.mark.parametrize("model_type", ["qwen3_5_moe", "qwen3_5_moe_text", "glm", "glm4", "glm4_text"])
def test_non_glm_moe_does_not_use_the_glm_training_policy(tmp_path, model_type):
    checkpoint = _write_config(tmp_path / "non_glm_moe", {"model_type": model_type})

    assert not is_glm_moe_model(checkpoint)


@pytest.mark.parametrize("family", ["glm_moe_dsa", "glm5_next"])
@pytest.mark.parametrize("suffix", ["", "_text"])
@pytest.mark.parametrize("nested", [False, True])
def test_glm_moe_training_policy_uses_the_text_family(tmp_path, family, suffix, nested):
    payload = {"model_type": family + suffix}
    if nested:
        payload = {"model_type": "vision_wrapper", "text_config": payload}
    checkpoint = _write_config(tmp_path / "glm", payload)

    assert is_glm_moe_model(checkpoint)


@pytest.mark.parametrize("model_type", ["qwen3_5_moe", "qwen3_5_moe_text"])
def test_qwen_moe_matches_the_text_family_in_an_unrelated_wrapper(tmp_path, model_type):
    checkpoint = _write_config(
        tmp_path / "qwen",
        {"model_type": "vision_wrapper", "text_config": {"model_type": model_type}},
    )

    assert moe_model_spec(model_name=checkpoint, ep_size=8).loader == "qwen3_5_moe"


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"model_type": "glm5_next"}, "glm5_next"),
        ({"model_type": "glm5_next_text"}, "glm5_next"),
        ({"model_type": "qwen4_exp_text"}, "qwen4_exp"),
        ({"model_type": "qwen3_5_moe_text"}, "qwen3_5_moe"),
        ({"model_type": "llama4_text"}, "llama4"),
        ({"model_type": "new_family_text"}, "new_family"),
        ({"model_type": "text_encoder"}, "text_encoder"),
        ({"model_type": "new_text_encoder"}, "new_text_encoder"),
        ({"model_type": "mistral3", "text_config": {"model_type": "mistral"}}, "mistral"),
        ({"model_type": "vision_wrapper", "text_config": {"model_type": "glm5_next_text"}}, "glm5_next"),
        ({"model_type": "glm5_next", "text_config": {"model_type": "qwen4_exp_text"}}, "qwen4_exp"),
        ({"model_type": "glm5_next", "text_config": {}}, "glm5_next"),
        ({"model_type": "glm5_next", "text_config": None}, "glm5_next"),
        ({}, ""),
    ],
)
@pytest.mark.parametrize("config_object", [False, True])
def test_loader_context_resolves_one_text_model_family(monkeypatch, payload, expected, config_object):
    config = payload
    if config_object:
        config = SimpleNamespace(**payload)
        if isinstance(payload.get("text_config"), dict):
            config.text_config = SimpleNamespace(**payload["text_config"])
    monkeypatch.setattr("arctic_platform.model.loader._load_hf_config", lambda _path: config)
    ctx = LoaderContext(ModelSpec(model_path_or_name="unused", loader="huggingface"))

    assert ctx.hf_model_family == expected


@pytest.mark.parametrize("family", ["glm5_next", "qwen4_exp", "qwen3_5_moe"])
@pytest.mark.parametrize("nested", [False, True])
def test_text_family_matching_does_not_enable_expert_loaders_for_dense_jobs(monkeypatch, family, nested):
    payload = {"model_type": family + "_text"}
    if nested:
        payload = {"model_type": "vision_wrapper", "text_config": payload}
    monkeypatch.setattr("arctic_platform.model.loader._load_hf_config", lambda _path: payload)
    ctx = LoaderContext(
        ModelSpec(
            model_path_or_name="unused",
            loader="huggingface",
            parallelism=ParallelismConfig(expert_parallel=1),
        )
    )

    matches = qwen_matches if family == "qwen3_5_moe" else flash_matches(family)
    assert not matches(ctx)


@pytest.mark.parametrize("family", ["glm5_next", "qwen4_exp", "qwen3_5_moe", "llama4"])
def test_moe_matchers_use_the_nested_family_instead_of_the_wrapper(monkeypatch, family):
    payload = {"model_type": "glm5_next", "text_config": {"model_type": family + "_text"}}
    monkeypatch.setattr("arctic_platform.model.loader._load_hf_config", lambda _path: payload)
    ctx = LoaderContext(
        ModelSpec(
            model_path_or_name="unused",
            loader="huggingface",
            parallelism=ParallelismConfig(expert_parallel=8),
        )
    )

    assert flash_matches("glm5_next")(ctx) == (family == "glm5_next")
    assert flash_matches("qwen4_exp")(ctx) == (family == "qwen4_exp")
    assert qwen_matches(ctx) == (family == "qwen3_5_moe")


@pytest.mark.parametrize("family,attention", [("glm5_next", "sparse_mla"), ("qwen4_exp", "qsa_flex")])
@pytest.mark.parametrize("suffix", ["", "_text"])
@pytest.mark.parametrize("wrapper", [None, "family", "vision_wrapper"])
def test_flash_moe_text_config_preserves_training_options(tmp_path, family, attention, suffix, wrapper):
    payload = {"model_type": family + suffix}
    if wrapper is not None:
        payload = {"model_type": family if wrapper == "family" else wrapper, "text_config": payload}
    checkpoint = _write_config(tmp_path / family, payload)
    spec = moe_model_spec(
        model_name=checkpoint,
        ep_size=8,
        sp_size=8,
        loader_options={
            "attn": attention,
            "ac_config": {"mode": "full"},
            "seq_len": 1024,
            "ep_comm_backend": "uccl",
            "fused_cross_entropy": "liger",
            "deepep_token_chunk_size": 1024,
            "debug": {"full_determinism": True},
        },
        tiled_mlp_token_chunk_size=1024,
    )

    assert spec.loader == family
    assert spec.attn_implementation == attention
    assert spec.parallelism.expert_parallel == spec.parallelism.sequence_parallel == 8
    assert spec.loader_options["ac_config"]["mode"] == "full"
    assert spec.loader_options["seq_len"] == 1024
    assert spec.loader_options["ep_comm_backend"] == "uccl"
    assert spec.loader_options["fused_cross_entropy"] == "liger"
    assert spec.loader_options["deepep_token_chunk_size"] == 1024
    assert spec.loader_options["tiled_mlp_token_chunk_size"] == 1024
    assert spec.loader_options["debug"]["full_determinism"] is True


def test_qwen_spec_lifts_dtype_attention_and_parallelism(tmp_path):
    checkpoint = _write_config(tmp_path / "qwen", {"model_type": "qwen3_5_moe"})
    spec = moe_model_spec(
        model_name=checkpoint,
        ep_size=8,
        sp_size=2,
        loader_options={
            "ep_comm_backend": "uccl",
            "optimization_dtype": "float32",
            "attn": "flash_attention_3",
            "fused_cross_entropy": False,
            "debug": {"random_init": True},
        },
        tiled_mlp_token_chunk_size=64,
        peft_config={"peft_type": "LORA", "r": 8, "target_modules": ["q_proj"]},
    )

    assert spec.loader == "qwen3_5_moe"
    assert spec.dtype == "float32"
    assert spec.attn_implementation == "flash_attention_3"
    assert spec.parallelism.expert_parallel == 8
    assert spec.parallelism.sequence_parallel == 2
    assert spec.patches.peft["r"] == 8
    assert spec.loader_options["ep_comm_backend"] == "uccl"
    assert spec.loader_options["fused_cross_entropy"] is False
    assert spec.loader_options["tiled_mlp_token_chunk_size"] == 64
    assert spec.loader_options["debug"]["random_init"] is True
    assert "optimization_dtype" not in spec.loader_options
    assert "attn" not in spec.loader_options


def test_omitted_fused_cross_entropy_defaults_to_liger(tmp_path):
    checkpoint = _write_config(tmp_path / "glm", {"model_type": "glm_moe_dsa"})
    spec = moe_model_spec(model_name=checkpoint, ep_size=8)
    assert spec.loader_options["fused_cross_entropy"] == "liger"


def test_none_offload_threshold_keeps_the_loader_default(tmp_path):
    checkpoint = _write_config(tmp_path / "qwen", {"model_type": "qwen3_5_moe"})
    spec = moe_model_spec(
        model_name=checkpoint,
        ep_size=8,
        loader_options={
            "fused_cross_entropy": False,
            "ac_config": {
                "mode": "full",
                "offload_config": {"enabled": True, "tensor_size_threshold": None},
            },
        },
    )
    offload = spec.loader_options["ac_config"]["offload_config"]
    assert offload["enabled"] is True
    assert offload["tensor_size_threshold"] == 1 << 20


def test_fused_cross_entropy_with_chunked_lm_head_is_rejected(tmp_path):
    checkpoint = _write_config(tmp_path / "qwen", {"model_type": "qwen3_5_moe"})
    with pytest.raises(ValueError, match="cannot combine fused_cross_entropy"):
        moe_model_spec(
            model_name=checkpoint,
            ep_size=8,
            loader_options={
                "fused_cross_entropy": "liger",
                "fused_lm_head_token_chunk_size": 128,
            },
        )


def test_glm_spec_rejects_sequence_parallelism(tmp_path):
    checkpoint = _write_config(tmp_path / "glm", {"model_type": "glm_moe_dsa"})
    with pytest.raises(ValueError, match="glm_moe_dsa does not support sequence parallelism"):
        moe_model_spec(model_name=checkpoint, ep_size=8, sp_size=2)


@pytest.mark.parametrize("nested", [False, True])
def test_glm53_uses_the_glm_moe_training_policy(tmp_path, nested):
    payload = {"model_type": "glm5_next_text"}
    if nested:
        payload = {"model_type": "glm5_next", "text_config": payload}
    checkpoint = _write_config(tmp_path / "glm53", payload)

    assert hf_text_model_type(checkpoint) == "glm5_next_text"
    assert is_glm_moe_model(checkpoint)
