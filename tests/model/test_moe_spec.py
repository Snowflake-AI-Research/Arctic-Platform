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
