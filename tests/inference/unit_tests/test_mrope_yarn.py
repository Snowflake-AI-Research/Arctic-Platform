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
from transformers import PreTrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope

from arctic_platform.inference.vllm.plugin import arctic_inference_plugin


@pytest.mark.parametrize("factor", [1.5, 2.0])
def test_mrope_yarn_matches_transformers(monkeypatch, factor):
    monkeypatch.setenv("ARCTIC_INFERENCE_ENABLED", "0")
    arctic_inference_plugin()
    rope_parameters = dict(
        rope_type="yarn",
        factor=factor,
        original_max_position_embeddings=262144,
        rope_theta=1e7,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
    )
    hf_config = PreTrainedConfig(
        hidden_size=256,
        num_attention_heads=1,
        head_dim=256,
        max_position_embeddings=int(262144 * factor),
        rope_parameters=dict(rope_parameters),
    )
    inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS["yarn"](hf_config, "cpu")
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
        rope = get_rope(
            head_size=256,
            max_position=int(262144 * factor),
            rope_parameters=dict(rope_parameters),
            dtype=torch.float32,
        )
    positions = torch.cat((torch.arange(4096), torch.arange(262140, 262150)))
    freqs = torch.outer(positions.float(), inv_freq)
    expected = torch.cat((freqs.cos(), freqs.sin()), dim=-1) * attention_scaling
    assert rope.cos_sin_cache.shape[0] == int(262144 * 4 * factor)
    # Both implementations calculate the same float32 frequencies and cache.
    torch.testing.assert_close(rope.cos_sin_cache[positions], expected, rtol=0, atol=0)
