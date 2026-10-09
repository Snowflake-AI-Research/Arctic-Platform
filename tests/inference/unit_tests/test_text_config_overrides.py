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

import json

import pytest
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from vllm.config import ModelConfig as VllmModelConfig

from arctic_platform.inference.server.config import ModelConfig as SamplerModelConfig


@pytest.mark.parametrize("layout", ["composite", "flat"])
def test_text_config_overrides_match_trainer_and_sampler(tmp_path, layout):
    composite = tmp_path / "composite"
    composite.mkdir()
    (composite / "config.json").write_text(
        json.dumps(
            dict(
                model_type="qwen3_5",
                architectures=["Qwen3_5ForConditionalGeneration"],
                text_config=dict(
                    model_type="qwen3_5_text",
                    vocab_size=32,
                    hidden_size=64,
                    intermediate_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=1,
                    num_key_value_heads=1,
                    head_dim=256,
                    layer_types=["full_attention"],
                    max_position_embeddings=262144,
                    rope_parameters=dict(
                        rope_type="default",
                        rope_theta=1e7,
                        partial_rotary_factor=0.25,
                        mrope_section=[11, 11, 10],
                        mrope_interleaved=True,
                    ),
                ),
            )
        )
    )
    flat = tmp_path / "flat"
    AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(composite).get_text_config()).save_pretrained(flat)
    path = str(tmp_path / layout)
    overrides = dict(
        max_position_embeddings=524288,
        rope_parameters=dict(rope_type="yarn", factor=2.0, original_max_position_embeddings=262144),
    )
    kwargs = SamplerModelConfig(model=path, text_config_overrides=overrides).to_engine_kwargs()
    hf_overrides = kwargs["hf_overrides"]
    assert isinstance(hf_overrides, dict)
    assert "text_config_overrides" not in kwargs

    from arctic_platform.common.text_config_overrides import apply_text_config_overrides
    from arctic_platform.common.text_config_overrides import text_config_hf_overrides

    trainer = AutoConfig.from_pretrained(path)
    apply_text_config_overrides(trainer, overrides)
    assert hf_overrides == text_config_hf_overrides(path, overrides)
    sampler = VllmModelConfig(model=path, dtype="float32", hf_overrides=hf_overrides).hf_text_config
    expected = trainer.get_text_config()
    assert sampler.max_position_embeddings == expected.max_position_embeddings == 524288
    assert sampler.rope_parameters == expected.rope_parameters
    assert sampler.rope_parameters["mrope_section"] == [11, 11, 10]
    assert sampler.rope_parameters["rope_type"] == "yarn"
    with pytest.raises(ValueError, match="hf_overrides"):
        SamplerModelConfig(
            model=path, text_config_overrides=overrides, extra_engine_kwargs=dict(hf_overrides={})
        ).to_engine_kwargs()
