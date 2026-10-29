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

import builtins
import json

import pytest

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


@pytest.mark.parametrize("nested", [False, True])
def test_glm53_uses_the_glm_moe_training_policy(tmp_path, nested):
    payload = {"model_type": "glm5_next_text"}
    if nested:
        payload = {"model_type": "glm5_next", "text_config": payload}
    checkpoint = _write_config(tmp_path / "glm53", payload)

    assert hf_text_model_type(checkpoint) == "glm5_next_text"
    assert is_glm_moe_model(checkpoint)


def test_non_glm_moe_does_not_use_the_glm_training_policy(tmp_path):
    checkpoint = _write_config(tmp_path / "qwen", {"model_type": "qwen3_5_moe"})

    assert not is_glm_moe_model(checkpoint)


@pytest.mark.parametrize("family,attention", [("glm5_next", "sparse_mla"), ("qwen4_exp", "qsa_flex")])
@pytest.mark.parametrize("nested", [False, True])
def test_flash_moe_text_config_preserves_training_options(tmp_path, family, attention, nested):
    payload = {"model_type": f"{family}_text"}
    if nested:
        payload = {"model_type": family, "text_config": payload}
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


@pytest.mark.parametrize(
    "attention,ep_size,error",
    [
        (None, 16, None),
        ("qsa_flex", 16, None),
        ("flex_attention", 16, None),
        ("sdpa", 16, "requires QSA FlexAttention"),
        ("qsa_flex", 3, "must divide 512"),
    ],
)
def test_qwen38_spec_validation_does_not_import_training(tmp_path, monkeypatch, attention, ep_size, error):
    checkpoint = _write_config(tmp_path / "qwen38", {"model_type": "qwen4_exp"})
    original_import = builtins.__import__

    def reject_training_import(name, *args, **kwargs):
        if name.startswith("arctic_platform.model.implementations.qwen38.deepspeed_integration"):
            raise AssertionError("Qwen3.8 spec validation must not import the training implementation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_training_import)
    options = {} if attention is None else {"attn": attention}
    if error is not None:
        with pytest.raises(ValueError, match=error):
            moe_model_spec(model_name=checkpoint, ep_size=ep_size, loader_options=options)
    else:
        spec = moe_model_spec(model_name=checkpoint, ep_size=ep_size, loader_options=options)
        assert spec.attn_implementation == ("qsa_flex" if attention is None else attention)
        assert spec.parallelism.expert_parallel == ep_size


@pytest.mark.parametrize("attention", [None, "sparse_mla", "flashinfer", "flashmla", "sdpa"])
def test_glm53_spec_validation_does_not_import_training(tmp_path, monkeypatch, attention):
    checkpoint = _write_config(tmp_path / "glm53", {"model_type": "glm5_next"})
    original_import = builtins.__import__

    def reject_training_import(name, *args, **kwargs):
        if name.startswith(
            (
                "arctic_platform.model.implementations.glm53.deepspeed_integration",
                "arctic_platform.model.implementations.glm53.context_parallel",
            )
        ):
            raise AssertionError("GLM-5.3 spec validation must not import the training implementation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_training_import)
    options = {} if attention is None else {"attn": attention}
    if attention == "sdpa":
        with pytest.raises(ValueError, match="requires sparse MLA"):
            moe_model_spec(model_name=checkpoint, ep_size=16, sp_size=16, loader_options=options)
    else:
        spec = moe_model_spec(model_name=checkpoint, ep_size=16, sp_size=16, loader_options=options)
        assert spec.attn_implementation == ("sparse_mla" if attention is None else attention)
        assert spec.parallelism.expert_parallel == spec.parallelism.sequence_parallel == 16


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
