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
from vllm.config import CacheConfig
from vllm.config import DeviceConfig
from vllm.config import ModelConfig
from vllm.config import ParallelConfig
from vllm.config import SpeculativeConfig
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

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


@pytest.mark.parametrize(("method", "draft_arch"), [("dflash", "DFlash2DraftModel"), ("dspark", "Qwen3DSparkModel")])
def test_next_turn_reuses_the_last_full_prefix_block(tmp_path, monkeypatch, method, draft_arch):
    monkeypatch.setenv("ARCTIC_INFERENCE_ENABLED", "0")
    for name, architecture in (("target", "Qwen3ForCausalLM"), ("draft", draft_arch)):
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
        method=method,
        num_speculative_tokens=1,
        target_model_config=ModelConfig(model=str(tmp_path / "target"), dtype="float16", skip_tokenizer_init=True),
        target_parallel_config=ParallelConfig(),
    )
    # Model a config received before plugin loading even if an earlier test patched its constructor.
    spec.disable_eagle_block_drop = False
    plugin.arctic_inference_plugin()

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
    # Exercise the V2 scheduler on CPU without starting its GPU model runner.
    monkeypatch.setattr("vllm.config.vllm.HAS_TRITON", True)
    config = VllmConfig(
        model_config=spec.target_model_config,
        speculative_config=spec,
        cache_config=CacheConfig(block_size=BLOCK, mamba_cache_mode="align"),
        device_config=DeviceConfig(device="cpu"),
    )
    cache = KVCacheConfig(
        num_blocks=8, kv_cache_tensors=[], kv_cache_groups=groups, prefix_cache_retention_interval=BLOCK
    )
    config.cache_config.num_gpu_blocks = cache.num_blocks
    scheduler = Scheduler(config, cache, StructuredOutputManager(config), block_size=BLOCK)
    assert scheduler.use_eagle

    turn1 = _request("turn1", [0])
    scheduler.add_request(turn1)
    for token in range(1, 2 * BLOCK + 1):
        scheduler.update_from_output(
            scheduler.schedule(),
            ModelRunnerOutput(req_ids=["turn1"], req_id_to_index={"turn1": 0}, sampled_token_ids=[[token]]),
        )
    scheduler.add_request(_request("turn2", [*turn1.all_token_ids, 9]))
    (continuation,) = scheduler.schedule().scheduled_new_reqs
    assert continuation.num_computed_tokens == 2 * BLOCK
