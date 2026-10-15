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
from vllm.config import VllmConfig
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.rotary_embedding import get_rope

from arctic_platform.inference.vllm import plugin

NATIVE, TARGET = 262144, 524288
DEFAULT = dict(rope_type="default", rope_theta=1e7)
YARN = dict(rope_type="yarn", rope_theta=1e7, factor=2.0, original_max_position_embeddings=NATIVE)


def _config(length, rope, **kwargs):
    return Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=length,
        rope_parameters=dict(rope),
        **kwargs,
    )


@pytest.mark.parametrize("follows", [False, True])
def test_dflash_draft_rotary_covers_target_positions(tmp_path, monkeypatch, follows):
    monkeypatch.setenv("ARCTIC_INFERENCE_ENABLED", "0")
    plugin.arctic_inference_plugin()
    _config(TARGET, YARN, architectures=["Qwen3ForCausalLM"]).save_pretrained(tmp_path / "target")
    _config(NATIVE, DEFAULT, architectures=["DFlash2DraftModel"]).save_pretrained(tmp_path / "draft")
    target = ModelConfig(model=str(tmp_path / "target"), dtype="float32")
    speculative_config = dict(model=str(tmp_path / "draft"), method="dflash", num_speculative_tokens=7)
    if follows:
        speculative_config["draft_rope_follows_target"] = True
    engine_args = EngineArgs(model=str(tmp_path / "target"), speculative_config=speculative_config)
    spec = engine_args.create_speculative_config(target, ParallelConfig())
    draft = spec.draft_model_config.hf_config
    # On: the plain-YaRN dict a hand-edited draft config.json carries. Off: the checkpoint's own rope, unchanged.
    assert draft.rope_parameters == (YARN if follows else DEFAULT)
    assert draft.max_position_embeddings == target.max_model_len + spec.num_speculative_tokens
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        rope = get_rope(
            head_size=8,
            max_position=draft.max_position_embeddings,
            rope_parameters=draft.rope_parameters,
            dtype=torch.float32,
        )
    positions = torch.tensor([NATIVE, TARGET - 1] + ([] if follows else [TARGET + spec.num_speculative_tokens - 1]))
    query = torch.ones(len(positions), 8)
    rotated, _ = rope.forward_native(positions, query, query)
    assert torch.isfinite(rotated).all()
    if follows:
        inv_freq, scale = ROPE_INIT_FUNCTIONS["yarn"](_config(TARGET, YARN), "cpu")
        freqs = torch.outer(positions.float(), inv_freq)
        expected = torch.cat((freqs.cos(), freqs.sin()), dim=-1) * scale
        torch.testing.assert_close(rope.cos_sin_cache[positions], expected, rtol=0, atol=0)
