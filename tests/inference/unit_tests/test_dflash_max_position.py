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

import pytest
import torch
from transformers import Qwen3Config
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from vllm.config import DeviceConfig
from vllm.config import ModelConfig
from vllm.config import ParallelConfig
from vllm.config import SpeculativeConfig
from vllm.config import VllmConfig
from vllm.config import set_current_vllm_config
from vllm.model_executor.models import qwen3_dflash

from arctic_platform.inference.vllm import plugin

NATIVE, TARGET = 262144, 393216
# A draft config.json hand-edited to plain YaRN sizes vLLM's table at original x factor, the target window exactly.
YARN = dict(rope_type="yarn", rope_theta=1e7, factor=1.5, original_max_position_embeddings=NATIVE)


@pytest.mark.parametrize("rope", [dict(rope_type="default", rope_theta=1e7), YARN])
def test_dflash_rotary_cache_covers_target_positions(tmp_path, monkeypatch, rope):
    monkeypatch.setenv("ARCTIC_INFERENCE_ENABLED", "0")
    plugin.arctic_inference_plugin()

    draft_length = TARGET if rope is YARN else NATIVE
    for name, length, parameters in (("target", TARGET, None), ("draft", draft_length, rope)):
        Qwen3Config(
            architectures=["Qwen3ForCausalLM" if name == "target" else "DFlash2DraftModel"],
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=length,
            **({} if parameters is None else dict(rope_parameters=dict(parameters))),
        ).save_pretrained(tmp_path / name)
    target = ModelConfig(model=str(tmp_path / "target"), dtype="float32")
    spec = SpeculativeConfig(
        model=str(tmp_path / "draft"),
        method="dflash",
        num_speculative_tokens=7,
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
    )
    draft = spec.draft_model_config.hf_config
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        # The name DFlashQwen3Attention resolves when it builds its rotary embedding.
        rope_module = qwen3_dflash.get_rope(
            head_size=8,
            max_position=draft.max_position_embeddings,
            rope_parameters=draft.rope_parameters,
            dtype=torch.float32,
        )
    positions = torch.tensor([NATIVE, TARGET - 1, TARGET + spec.num_speculative_tokens - 1])
    query = torch.ones(len(positions), 8)
    rotated, _ = rope_module.forward_native(positions, query, query)
    assert torch.isfinite(rotated).all()
    assert draft.max_position_embeddings == target.max_model_len + spec.num_speculative_tokens
    if rope is YARN:
        inv_freq, scale = ROPE_INIT_FUNCTIONS["yarn"](Qwen3Config(head_dim=8, rope_parameters=dict(YARN)), "cpu")
        freqs = torch.outer(positions.float(), inv_freq)
        expected = torch.cat((freqs.cos(), freqs.sin()), dim=-1) * scale
        torch.testing.assert_close(rope_module.cos_sin_cache[positions], expected, rtol=0, atol=0)
