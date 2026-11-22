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
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec, MLAAttentionSpec
from vllm.v1.request import Request

from arctic_platform.inference.vllm.plugin import arctic_inference_plugin

BLOCK = 16


def _request(request_id, tokens):
    params = SamplingParams(max_tokens=10_000)
    params.update_from_generation_config({}, eos_token_id=None)
    return Request(
        request_id=request_id, prompt_token_ids=tokens, sampling_params=params, pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def _continuation_hit(dcp, prompt_len, output_len):
    """Hit of a request resending a finished request's prompt and output, on a hybrid MLA + Mamba cache."""
    config = KVCacheConfig(
        num_blocks=1024,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["mla"], MLAAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32)),
            KVCacheGroupSpec(
                ["mamba"], MambaSpec(block_size=BLOCK, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align")
            ),
        ],
        prefix_cache_retention_interval=None,
    )
    manager = KVCacheManager(
        config, max_model_len=1 << 16, enable_caching=True, hash_block_size=BLOCK,
        scheduler_block_size=BLOCK * dcp, dcp_world_size=dcp,
    )
    first = _request("first", list(range(prompt_len)))
    # The scheduler ends align-mode prefill chunks on block boundaries.
    for end in [*range(BLOCK, prompt_len, BLOCK), prompt_len]:
        assert manager.allocate_slots(first, end - first.num_computed_tokens) is not None
        first.num_computed_tokens = end
    for step in range(output_len):
        first.append_output_token_ids(10_000 + step)
        if step < output_len - 1:  # the last sampled token is never computed
            assert manager.allocate_slots(first, 1) is not None
            first.num_computed_tokens += 1
    manager.free(first)
    follow_up = _request("follow_up", list(first.all_token_ids) + [20_000, 20_001, 20_002])
    return manager.get_computed_blocks(follow_up)[1]


@pytest.mark.parametrize("prompt_len, output_len", [(130, 200), (300, 37), (257, 300), (100, 5)])
def test_follow_up_hits_the_finished_requests_last_hash_boundary_under_dcp(monkeypatch, prompt_len, output_len):
    monkeypatch.setenv("ARCTIC_INFERENCE_ENABLED", "0")
    arctic_inference_plugin()
    init_none_hash(sha256)

    computed = prompt_len + output_len - 1
    assert _continuation_hit(1, prompt_len, output_len) == computed - computed % BLOCK
    assert _continuation_hit(8, prompt_len, output_len) == computed - computed % BLOCK
