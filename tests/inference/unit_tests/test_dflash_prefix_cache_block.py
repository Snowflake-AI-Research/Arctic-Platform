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

BLOCK = 4


def _request(request_id: str, token_ids: list[int]) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=SamplingParams(max_tokens=2 * BLOCK),
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def test_dflash_next_turn_reuses_the_last_full_prefix_block(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCTIC_INFERENCE_ENABLED", "0")
    for name, architecture in (("target", "Qwen3ForCausalLM"), ("draft", "DFlash2DraftModel")):
        Qwen3Config(
            architectures=[architecture],
            hidden_size=8,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=16,
        ).save_pretrained(tmp_path / name)
    spec = SpeculativeConfig(
        model=str(tmp_path / "draft"),
        method="dflash",
        num_speculative_tokens=1,
        target_model_config=ModelConfig(model=str(tmp_path / "target"), dtype="float16"),
        target_parallel_config=ParallelConfig(),
    )
    # A pre-plugin config carries the upstream default, even if another test already installed the patch here.
    spec.disable_eagle_block_drop = False
    plugin.arctic_inference_plugin()
    assert spec.use_eagle()

    init_none_hash(sha256)
    groups = [
        KVCacheGroupSpec(
            ["attn"], FullAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float16)
        ),
        KVCacheGroupSpec(
            ["linear_attn"],
            MambaSpec(block_size=BLOCK, shapes=((1,),), dtypes=(torch.float16,), mamba_cache_mode="align"),
        ),
    ]
    manager = KVCacheManager(
        KVCacheConfig(num_blocks=8, kv_cache_tensors=[], kv_cache_groups=groups, prefix_cache_retention_interval=BLOCK),
        max_model_len=16,
        scheduler_block_size=BLOCK,
        hash_block_size=BLOCK,
        use_eagle=spec.use_eagle_block_drop(),
        num_prefill_lookahead=1,
    )

    turn1 = _request("turn1", [0])
    for token in range(1, 2 * BLOCK + 1):
        assert manager.allocate_slots(turn1, 1, num_lookahead_tokens=spec.num_speculative_tokens) is not None
        turn1.num_computed_tokens += 1
        turn1.append_output_token_ids(token)
    manager.free(turn1)

    turn2 = _request("turn2", [*turn1.all_token_ids, 9])
    _, num_hit_tokens, _ = manager.get_computed_blocks(turn2)
    assert num_hit_tokens == 2 * BLOCK
