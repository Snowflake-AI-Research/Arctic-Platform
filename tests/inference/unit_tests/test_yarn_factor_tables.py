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

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen3_5TextConfig
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

from arctic_platform.inference.vllm import yarn_factors


def _config():
    return Qwen3_5TextConfig(
        hidden_size=256,
        num_attention_heads=1,
        head_dim=256,
        max_position_embeddings=32,
        rope_parameters=dict(
            rope_type="yarn",
            factor=1.0,
            original_max_position_embeddings=32,
            rope_theta=1e7,
            partial_rotary_factor=0.25,
            mrope_section=[11, 11, 10],
            mrope_interleaved=True,
        ),
    )


def _rotary():
    from vllm.config import DeviceConfig
    from vllm.config import VllmConfig
    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        return MRotaryEmbedding(
            256,
            64,
            32,
            1e7,
            True,
            torch.float32,
            mrope_section=[11, 11, 10],
            mrope_interleaved=True,
        )


def test_tables_match_isolated_factors_and_native_default():
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    rotary = _rotary()
    native_cache = rotary.cos_sin_cache.clone()
    rows = native_cache.shape[0]
    model = torch.nn.ModuleList([rotary, _rotary()])
    factors = [1.0, 1.25, 1.5, 1.75, 2.0]
    assert yarn_factors.install_yarn_tables(model, _config(), factors, 32) == rows
    assert torch.equal(rotary.cos_sin_cache[:rows], native_cache)
    positions = torch.tensor([[0, 1, 31], [0, 2, 30], [0, 3, 29]])
    query, key = torch.randn(3, 512), torch.randn(3, 256)
    for slot, factor in enumerate(factors):
        isolated = _rotary()
        if factor > 1:
            config = _config()
            config.rope_parameters.update(rope_type="yarn", factor=factor, original_max_position_embeddings=32)
            inv_freq, scale = ROPE_INIT_FUNCTIONS["yarn"](config, torch.device("cpu"))
            freqs = torch.outer(torch.arange(rows, dtype=torch.float32), inv_freq)
            isolated.cos_sin_cache = torch.cat((freqs.cos() * scale, freqs.sin() * scale), -1)
        expected = isolated.forward_native(positions, query, key)
        actual = rotary.forward_native(positions + slot * rows, query, key)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert model[0].cos_sin_cache is model[1].cos_sin_cache


def test_v2_request_prefill_decode_readd_and_disabled(monkeypatch):
    from vllm.v1.worker.gpu.mm.rope import RopeState
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    writes = {}
    rope = object.__new__(RopeState)
    rope.num_dims = 3
    rope.max_model_len = 32
    rope.prefill_delta = SimpleNamespace(np=np.zeros(2, dtype=np.int32))
    rope.prefill_positions = SimpleNamespace(stage_write=lambda row, start, values: writes.__setitem__(row, values))
    positions = torch.arange(4).expand(3, -1).clone()
    from vllm.v1.worker.gpu.model_states import default

    model = torch.nn.ModuleList([_rotary()])
    model.get_mrope_input_positions = lambda *args: (positions, 2)
    config = SimpleNamespace(
        additional_config={"yarn_factors": [1.0, 1.5, 2.0]},
        model_config=SimpleNamespace(
            max_model_len=32,
            hf_text_config=_config(),
            enable_prompt_embeds=False,
            dtype=torch.float32,
            get_inputs_embeds_size=lambda: 256,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=32),
    )
    # Only the GPU-backed staging buffers are substituted.
    monkeypatch.setattr(default, "get_rope_state", lambda *args, **kwargs: rope)
    req = SimpleNamespace(
        prefill_token_ids=[1, 2, 3, 4],
        mm_features=[],
        sampling_params=SimpleNamespace(extra_args={"yarn_factor_slot": 2}),
    )
    # Restore patched classes after the test so registration tests stay independent.
    for cls, name in (
        (DefaultModelState, "__init__"),
        (DefaultModelState, "add_request"),
        (RopeState, "init_prefill_positions"),
    ):
        monkeypatch.setattr(cls, name, getattr(cls, name))
    monkeypatch.setattr(yarn_factors, "_PATCHED", False)
    yarn_factors.ensure_yarn_factor_patches()
    state = DefaultModelState(config, model, None, torch.device("cpu"))
    assert model[0].cos_sin_cache.shape == (384, 64)
    for _ in range(2):  # Preemption re-adds the request, never accumulating offsets.
        DefaultModelState.add_request(state, 0, req)
        assert writes[0] == [256, 257, 258, 259]
        assert rope.prefill_delta.np[0] == 258
        assert 4 + rope.prefill_delta.np[0] == 262  # V2 decode kernel expression.
        assert torch.equal(positions, torch.arange(4).expand(3, -1))
    req.sampling_params.extra_args = {}
    DefaultModelState.add_request(state, 0, req)
    assert writes[0] == [0, 1, 2, 3]
    assert rope.prefill_delta.np[0] == 2
    state._yarn_rows = 0
    DefaultModelState.add_request(state, 0, req)
    assert writes[0] == [0, 1, 2, 3]
    state._yarn_rows = 2**31
    req.sampling_params.extra_args = {"yarn_factor_slot": 1}
    with pytest.raises(ValueError, match="int32"):
        DefaultModelState.add_request(state, 0, req)
