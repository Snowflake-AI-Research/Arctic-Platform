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

import torch
from transformers import Qwen3Config
from vllm.config import DeviceConfig
from vllm.config import ModelConfig
from vllm.config import ParallelConfig
from vllm.config import SpeculativeConfig
from vllm.config import VllmConfig
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope

from arctic_platform.inference.vllm import plugin


def test_dflash_rotary_cache_covers_target_positions(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCTIC_INFERENCE_ENABLED", "0")
    plugin.arctic_inference_plugin()

    native_length, target_length = 262144, 393216
    for name, length in (("target", target_length), ("draft", native_length)):
        Qwen3Config(
            architectures=["Qwen3ForCausalLM" if name == "target" else "DFlash2DraftModel"],
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=length,
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
        rope = get_rope(
            head_size=8,
            max_position=draft.max_position_embeddings,
            rope_parameters=draft.rope_parameters,
            dtype=torch.float32,
        )
    positions = torch.tensor([native_length, target_length - 1])
    query = torch.ones(2, 8)
    rotated, _ = rope.forward_native(positions, query, query)
    assert torch.isfinite(rotated).all()
    assert draft.max_position_embeddings == target.max_model_len
