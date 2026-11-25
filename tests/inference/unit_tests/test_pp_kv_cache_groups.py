"""KV cache placement for a pipeline stage that holds no layer of a group.

Built from vLLM's own planning and allocation (``get_global_kv_cache_groups``,
``get_kv_cache_configs``, ``generate_scheduler_kv_cache_config``,
``allocate_kv_cache``) on CPU, for a two-stage hybrid model: full-attention
layers on both stages, sliding-window layers on one stage only. The two specs
have different page sizes, so with a block-outermost layout vLLM packs them
into two uniform-type groups, as for DeepSeek-V4.1's compressed layers.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig, VllmConfig
from vllm.v1.core import kv_cache_planning
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
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
SWA_ON_STAGE_2 = [
    {_layer(0): FULL, _layer(1): FULL, _layer(2): FULL, _layer(3): FULL},
    {_layer(4): FULL, _layer(5): SWA, _layer(6): FULL, _layer(7): SWA},
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


@pytest.mark.parametrize("stage_specs", [SWA_ON_STAGE_1, SWA_ON_STAGE_2])
def test_vllm_gives_a_stage_another_stages_layers(unpatched, stage_specs):
    config = _vllm_config(pp_size=2)
    configs = _plan(config, stage_specs)
    lacking = next(i for i, specs in enumerate(stage_specs) if SWA not in specs.values())
    tensor_layers = {name for t in configs[lacking].kv_cache_tensors for name in t.layers}
    assert not tensor_layers <= set(stage_specs[lacking])
    with pytest.raises(StopIteration):
        _allocate(config, configs[lacking])


@pytest.mark.parametrize("stage_specs", [SWA_ON_STAGE_1, SWA_ON_STAGE_2])
def test_every_stage_allocates_exactly_its_layers(patched, stage_specs):
    config = _vllm_config(pp_size=2)
    for specs, kv_cache_config in zip(stage_specs, _plan(config, stage_specs)):
        assert set(_allocate(config, kv_cache_config)) == set(specs)


@pytest.mark.parametrize("stage_specs", [SWA_ON_STAGE_1, SWA_ON_STAGE_2])
def test_stages_and_scheduler_agree_on_groups_and_blocks(patched, stage_specs):
    config = _vllm_config(pp_size=2)
    configs = _plan(config, stage_specs)
    lacking = next(i for i, specs in enumerate(stage_specs) if SWA not in specs.values())

    # The lacking stage keeps the empty group, with the whole-model spec.
    empty = [g for g in configs[lacking].kv_cache_groups if not g.layer_names]
    assert len(empty) == 1 and isinstance(empty[0].kv_cache_spec, UniformTypeKVCacheSpecs)

    group_block_sizes = [[g.kv_cache_spec.block_size for g in c.kv_cache_groups] for c in configs]
    assert group_block_sizes[0] == group_block_sizes[1]
    assert len({c.num_blocks for c in configs}) == 1
    assert len({resolve_kv_cache_block_sizes(c, config) for c in configs}) == 1

    scheduler = generate_scheduler_kv_cache_config(configs)
    assert [g.kv_cache_spec.block_size for g in scheduler.kv_cache_groups] == group_block_sizes[0]
    assert scheduler.num_blocks == configs[0].num_blocks
    assert resolve_kv_cache_block_sizes(scheduler, config) == resolve_kv_cache_block_sizes(
        configs[0], config
    )


@pytest.mark.parametrize("stage_specs", [SWA_ON_STAGE_1, SWA_ON_STAGE_2])
def test_stage_holding_every_group_is_placed_as_before(unpatched, stage_specs):
    config = _vllm_config(pp_size=2)
    before = _plan(config, stage_specs)
    pp_kv_cache_groups.patch_get_kv_cache_config_from_groups(kv_cache_planning)
    after = _plan(config, stage_specs)
    holding = next(i for i, specs in enumerate(stage_specs) if SWA in specs.values())
    assert after[holding] == before[holding]
    # The lacking stage loses only the other stage's tensors.
    lacking = 1 - holding
    held = set(stage_specs[lacking])
    assert after[lacking].kv_cache_tensors == [
        t for t in before[lacking].kv_cache_tensors if set(t.layers) <= held
    ]


def test_pp1_placement_is_vllms(unpatched):
    specs = [{**SWA_ON_STAGE_1[0], **SWA_ON_STAGE_1[1]}]
    config = _vllm_config(pp_size=1)
    before = _plan(config, specs)
    pp_kv_cache_groups.patch_get_kv_cache_config_from_groups(kv_cache_planning)
    after = _plan(config, specs)
    assert after == before
    assert set(_allocate(config, after[0])) == set(specs[0])


def test_a_tensor_mixing_held_and_unheld_layers_is_rejected():
    group = SimpleNamespace(layer_names=[_layer(0)])
    tensor = SimpleNamespace(layers=[_layer(0), _layer(1)])
    config = SimpleNamespace(kv_cache_groups=[group], kv_cache_tensors=[tensor])
    with pytest.raises(ValueError, match="mixes layers"):
        pp_kv_cache_groups.drop_unheld_kv_cache_tensors(config)


def test_patch_applies_once(unpatched):
    pp_kv_cache_groups.patch_get_kv_cache_config_from_groups(kv_cache_planning)
    patched_fn = kv_cache_planning.get_kv_cache_config_from_groups
    pp_kv_cache_groups.patch_get_kv_cache_config_from_groups(kv_cache_planning)
    assert kv_cache_planning.get_kv_cache_config_from_groups is patched_fn
