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

"""Dense training options become one Hugging Face ModelSpec."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from arctic_platform.model.dense_spec import dense_model_spec


def _training_options():
    return {
        "activation_checkpointing": 2,
        "ac_config": {"offload_config": {"enabled": True, "keep_last_n": 3, "use_streams": False}},
        "compile": {"fullgraph": False},
        "fp32_lm_head": True,
        "fused_lm_head_token_chunk_size": 128,
        "fused_lm_head_vocab_chunk_size": 4096,
        "tiled_mlp_token_chunk_size": 64,
        "peft_config": {"peft_type": "Lora", "r": 8, "target_modules": ["q_proj"]},
    }


def test_dense_spec_transfers_setup_into_patches():
    spec = dense_model_spec(
        model_path="Qwen/Qwen3-8B",
        dtype="bfloat16",
        attn_implementation="flash_attention_3",
        sequence_parallel=4,
        liger=True,
        training_options=_training_options(),
    )

    assert spec.loader == "huggingface"
    assert spec.dtype == "bfloat16"
    assert spec.parallelism.sequence_parallel == 4
    assert spec.patches.liger is True
    assert spec.patches.peft["r"] == 8
    assert spec.patches.gradient_checkpointing == 2
    assert spec.patches.activation_offload.keep_last_n == 3
    assert spec.patches.activation_offload.use_streams is False
    assert spec.patches.compile.fullgraph is False
    assert spec.patches.lm_head.fp32 is True
    assert spec.patches.lm_head.token_chunk_size == 128
    assert spec.patches.lm_head.vocab_chunk_size == 4096
    assert spec.patches.tiled_mlp.token_chunk_size == 64


def test_dense_offload_omits_a_none_tensor_size_threshold():
    options = _training_options()
    options["ac_config"]["offload_config"]["tensor_size_threshold"] = None
    spec = dense_model_spec(
        model_path="Qwen/Qwen3-8B",
        dtype="bfloat16",
        attn_implementation="flash_attention_3",
        sequence_parallel=1,
        liger=False,
        training_options=options,
    )

    assert spec.patches.activation_offload.tensor_size_threshold == 1 << 20


def _spec(**kwargs):
    args = {
        "model_path": "Qwen/Qwen3-8B",
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "sequence_parallel": 1,
        "liger": False,
        "training_options": {},
    }
    args.update(kwargs)
    return dense_model_spec(**args)


def test_omitted_loader_options_are_not_passed(monkeypatch):
    import arctic_platform.model.dense_spec as dense_spec

    seen = []
    real = dense_spec.ModelSpec

    def recording(**kwargs):
        seen.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(dense_spec, "ModelSpec", recording)
    _spec()

    assert "loader_options" not in seen[0]


def test_loader_fields_in_training_options_are_rejected():
    with pytest.raises(ValueError, match="not in training_options; move fp32_parity"):
        _spec(training_options={"fp32_parity": False})
    with pytest.raises(ValueError, match="not in training_options; move fp32_parity, fp32_residual"):
        _spec(training_options={"fp32_parity": False, "fp32_residual": True})


def test_loader_options_are_forwarded_whole(monkeypatch):
    import arctic_platform.model.dense_spec as dense_spec

    seen = []
    real = dense_spec.ModelSpec

    def recording(**kwargs):
        seen.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(dense_spec, "ModelSpec", recording)
    given = {"fp32_parity": False, "fp32_residual": False}
    _spec(loader_options=given)

    assert seen[0]["loader_options"] == given


def test_ac_config_layers_checkpoint_exactly_those_layers():
    spec = _spec(training_options={"activation_checkpointing": True, "ac_config": {"layers": [0, 1, 2, 4]}})

    assert spec.patches.gradient_checkpointing == [0, 1, 2, 4]
    with pytest.raises(ValueError, match="ac_config.layers requires activation_checkpointing true"):
        _spec(training_options={"activation_checkpointing": False, "ac_config": {"layers": [0]}})
    with pytest.raises(ValidationError, match="lists a layer twice"):
        _spec(training_options={"activation_checkpointing": True, "ac_config": {"layers": [0, 0]}})
    with pytest.raises(ValidationError, match="layers is empty"):
        _spec(training_options={"activation_checkpointing": True, "ac_config": {"layers": []}})


def test_loader_options_are_validated_by_the_huggingface_schema():
    parity_off = _spec(loader_options={"fp32_parity": False})
    residual_on = _spec(loader_options={"fp32_residual": True})

    assert parity_off.loader_options == {"fp32_parity": False, "fp32_residual": False}
    assert residual_on.loader_options == {"fp32_parity": True, "fp32_residual": True}
    with pytest.raises(ValidationError, match="fp32_residual requires fp32_parity"):
        _spec(loader_options={"fp32_parity": False, "fp32_residual": True})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _spec(loader_options={"not_a_loader_option": True})
