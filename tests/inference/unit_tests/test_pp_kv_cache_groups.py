"""KV cache placement for a pipeline stage that holds no layer of a group, through vLLM's own planning on CPU: two
specs of different page sizes in two uniform-type groups, as for DeepSeek-V4.1's compressed layers."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig, VllmConfig
from vllm.v1.core import kv_cache_planning
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.utils import allocate_kv_cache

from arctic_platform.inference.vllm import pp_kv_cache_groups

BLOCK_SIZE = 16
FULL = FullAttentionSpec(
    block_size=BLOCK_SIZE, num_kv_heads=2, head_size=64, dtype=torch.bfloat16
)
SWA = SlidingWindowSpec(
    block_size=BLOCK_SIZE,
    num_kv_heads=1,
    head_size=64,
    dtype=torch.bfloat16,
    sliding_window=128,
)
MEMORY = 1 << 24


def _layer(i):
    return f"model.layers.{i}.attn"


# Stage specs: one stage holds both groups, the other only full attention.
SWA_ON_STAGE_1 = [
    {_layer(0): FULL, _layer(1): SWA, _layer(2): FULL, _layer(3): SWA},
    {_layer(4): FULL, _layer(5): FULL, _layer(6): FULL, _layer(7): FULL},
]


def _vllm_config(pp_size):
    config = VllmConfig(
        cache_config=CacheConfig(block_size=BLOCK_SIZE, enable_prefix_caching=True)
    )
    config.parallel_config.pipeline_parallel_size = pp_size
    # The planning reads these model fields only; no checkpoint is needed.
    config.model_config = SimpleNamespace(
        max_model_len=1024,
        original_max_model_len=1024,
        supports_kv_cache_grouping=False,
    )
    # Set once by the engine core; block-outermost packs mixed page sizes.
    config.cache_config.kv_cache_layout = "BLHNC"
    return config


def _plan(config, stage_specs):
    groups = kv_cache_planning.get_global_kv_cache_groups(config, stage_specs)
    return kv_cache_planning.get_kv_cache_configs(
        config, stage_specs, [MEMORY] * len(stage_specs), groups
    )


def _allocate(config, kv_cache_config):
    return allocate_kv_cache(
        kv_cache_config,
        torch.device("cpu"),
        config.cache_config.get_resolved_kv_cache_layout(),
    )


@pytest.fixture
def unpatched(monkeypatch):
    original = kv_cache_planning.get_kv_cache_config_from_groups
    monkeypatch.setattr(
        kv_cache_planning,
        "get_kv_cache_config_from_groups",
        getattr(original, "_orig", original),
    )


@pytest.fixture
def patched(unpatched):
    pp_kv_cache_groups.patch_get_kv_cache_config_from_groups(kv_cache_planning)


def test_every_stage_allocates_exactly_its_layers(patched):
    config = _vllm_config(pp_size=2)
    stage_specs = SWA_ON_STAGE_1
    for specs, kv_cache_config in zip(stage_specs, _plan(config, stage_specs)):
        assert set(_allocate(config, kv_cache_config)) == set(specs)
