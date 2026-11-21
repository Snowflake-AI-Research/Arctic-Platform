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
from vllm.config import ModelConfig
from vllm.config import ParallelConfig
from vllm.config import SpeculativeConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.request import Request

from arctic_platform.inference.vllm import plugin

# A hybrid attention + linear-attention target in Mamba "align" mode, where both groups share one block size.
BLOCK = 640
TURN1_PROMPT, TURN1_TOKENS = 16, 2000


def _request(request_id: str, token_ids: list[int]) -> Request:
    params = SamplingParams(max_tokens=4096)
    params.update_from_generation_config({}, eos_token_id=None)
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def test_dflash_next_turn_reuses_the_last_full_prefix_block(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCTIC_INFERENCE_ENABLED", "0")
    plugin.arctic_inference_plugin()
    for name, architecture in (("target", "Qwen3ForCausalLM"), ("draft", "DFlash2DraftModel")):
        Qwen3Config(
            architectures=[architecture],
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
        ).save_pretrained(tmp_path / name)
    spec = SpeculativeConfig(
        model=str(tmp_path / "draft"),
        method="dflash",
        num_speculative_tokens=7,
        target_model_config=ModelConfig(model=str(tmp_path / "target"), dtype="float32"),
        target_parallel_config=ParallelConfig(),
    )
    assert spec.use_eagle()

    init_none_hash(sha256)
    groups = [
        KVCacheGroupSpec(
            ["attn"], FullAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32)
        ),
        KVCacheGroupSpec(
            ["linear_attn"],
            MambaSpec(block_size=BLOCK, shapes=((1, 1),), dtypes=(torch.float32,), mamba_cache_mode="align"),
        ),
    ]
    # Scheduler.__init__ builds its manager with use_eagle=use_eagle_block_drop() and the drafter's lookahead of 1.
    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=64, kv_cache_tensors=[], kv_cache_groups=groups, prefix_cache_retention_interval=BLOCK
        ),
        max_model_len=8192,
        scheduler_block_size=BLOCK,
        hash_block_size=BLOCK,
        use_eagle=spec.use_eagle_block_drop(),
        num_prefill_lookahead=1,
    )

    # Turn 1 prefills its prompt, then decodes one token per step; its last sampled token is never computed.
    turn1 = _request("turn1", list(range(1, TURN1_PROMPT + 1)))
    num_new = TURN1_PROMPT
    while turn1.num_tokens < TURN1_TOKENS:
        assert manager.allocate_slots(turn1, num_new, num_lookahead_tokens=spec.num_speculative_tokens) is not None
        turn1.num_computed_tokens += num_new
        turn1.append_output_token_ids(10_000 + turn1.num_tokens)
        num_new = 1
    manager.free(turn1)
    assert turn1.num_computed_tokens == TURN1_TOKENS - 1

    # Turn 2 resends all of turn 1 plus new tokens: the hit is turn 1's three full blocks, not two.
    turn2 = _request("turn2", list(turn1.all_token_ids) + list(range(50_000, 51_500)))
    _, num_hit_tokens, _ = manager.get_computed_blocks(turn2)
    assert num_hit_tokens == (TURN1_TOKENS - 1) // BLOCK * BLOCK == 1920
