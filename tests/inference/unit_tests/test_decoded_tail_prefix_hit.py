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

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec, MLAAttentionSpec
from vllm.v1.request import Request, RequestStatus

from arctic_platform.inference.vllm.plugin import arctic_inference_plugin

BLOCK = 16


def _request(request_id, tokens):
    params = SamplingParams(max_tokens=10_000)
    params.update_from_generation_config({}, eos_token_id=None)
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def _cache(dcp):
    config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["mla"], MLAAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32)
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(block_size=BLOCK, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align"),
            ),
        ],
        prefix_cache_retention_interval=None,
    )
    return KVCacheManager(
        config,
        max_model_len=1 << 16,
        enable_caching=True,
        hash_block_size=BLOCK,
        scheduler_block_size=BLOCK * dcp,
        dcp_world_size=dcp,
    )


@pytest.fixture(autouse=True)
def _apply_patches(monkeypatch):
    monkeypatch.setenv("ARCTIC_INFERENCE_ENABLED", "0")
    arctic_inference_plugin()
    init_none_hash(sha256)


def _scheduler(manager, dcp):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.kv_cache_manager = manager
    scheduler.cache_config = SimpleNamespace(block_size=BLOCK)
    scheduler.block_size = BLOCK * dcp
    scheduler.hash_block_size = BLOCK
    scheduler.use_eagle_block_drop = False
    scheduler.mamba_partial_cache_hit = manager.coordinator.enable_partial_hash_hits
    scheduler.mamba_has_prefill_checkpoint_blocks = False
    scheduler.mamba_shared_prefix_checkpoint = False
    scheduler.num_hash_lookahead_tokens = 0
    return scheduler


@pytest.mark.parametrize("dcp", [1, 8])
@pytest.mark.parametrize("prompt_len, output_len", [(130, 206), (300, 37), (257, 300), (100, 5)])
def test_continuation_reuses_the_last_computed_boundary(dcp, prompt_len, output_len):
    manager = _cache(dcp)
    scheduler = _scheduler(manager, dcp)
    first = _request("first", list(range(prompt_len)))
    first.status = RequestStatus.RUNNING
    while first.num_computed_tokens < prompt_len:
        size = scheduler._mamba_block_aligned_split(first, prompt_len - first.num_computed_tokens)
        assert size > 0
        assert manager.allocate_slots(first, size) is not None
        first.num_computed_tokens += size
    for step in range(output_len - 1):
        first.append_output_token_ids(10_000 + step)
        assert manager.allocate_slots(first, 1) is not None
        first.num_computed_tokens += 1

    # The next async step has been scheduled when the stop arrives.
    assert manager.allocate_slots(first, 1) is not None
    first.num_computed_tokens += 1
    first.num_in_flight_tokens = 1
    first.num_output_placeholders = 1
    first.append_output_token_ids(10_000 + output_len - 1)
    first.status = RequestStatus.FINISHED_STOPPED
    assert scheduler._handle_stopped_request(first)
    manager.free(first)
    manager.new_step_starts()

    computed = prompt_len + output_len - 1
    boundary = computed // BLOCK * BLOCK
    if first.num_tokens % BLOCK == 0:
        assert manager.block_pool.get_cached_block(first.block_hashes[-1], [0]) is None

    follow_up = _request("follow_up", list(first.all_token_ids) + [20_000])
    blocks, hit, _ = manager.get_computed_blocks(follow_up)
    assert hit == boundary
    assert manager.allocate_slots(follow_up, 1, new_computed_blocks=blocks, num_new_computed_tokens=hit) is not None
    if hit % (BLOCK * dcp):
        source = blocks.blocks[0][-1]
        destination = manager.get_blocks(follow_up.request_id).blocks[0][-1]
        copies, retained = manager.take_kv_cache_block_copies()
        assert destination is not source
        assert (source.block_id, destination.block_id) in [(copy.src_block_id, copy.dst_block_id) for copy in copies]
        manager.block_pool.free_blocks(retained)


def test_failed_remote_load_does_not_publish_allocated_slots():
    manager = _cache(8)
    request = _request("failed", list(range(100)))
    assert manager.allocate_slots(request, 80, delay_cache_blocks=True) is not None
    request.num_computed_tokens = 80
    request.status = RequestStatus.FINISHED_ERROR
    manager.free(request)
    assert manager.block_pool.get_cached_block(request.block_hashes[4], [0]) is None
