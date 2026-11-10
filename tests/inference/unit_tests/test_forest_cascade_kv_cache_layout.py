"""FCA consumes native packed cache views across physical layouts."""

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec, KVCacheTensor, compute_layout_strides, create_kv_cache_views,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from arctic_platform.testing_utils import require_torch_gpu, set_seed, torch_assert_equal
from arctic_platform.inference.vllm.attention import ForestFlashAttentionBackend
from arctic_platform.inference.vllm.attention.flash_attn_forest_cascade import FlashAttentionImpl


@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_native_cache_views_preserve_keys_and_values(layout):
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=32, dtype=torch.float32)
    strides = compute_layout_strides(spec, 3, 2, layout)
    size = 2 * 3 * spec.page_size_bytes
    raw = torch.empty(size, dtype=torch.int8)
    caches = create_kv_cache_views(
        raw, spec, 3, layout,
        KVCacheTensor(size=size, layers=["a", "b"], layer_stride=strides[0], block_stride=strides[1]),
    )
    for index, cache in enumerate(caches):
        key, value = cache.transpose(1, 2).split(32, dim=-1)
        expected = torch.arange(key.numel(), dtype=torch.float32).reshape(key.shape) + index
        key.copy_(expected)
        value.copy_(-expected)
        torch_assert_equal(cache[..., :32].transpose(1, 2), expected)
        torch_assert_equal(cache[..., 32:].transpose(1, 2), -expected)
        assert cache.untyped_storage().data_ptr() == raw.data_ptr()
    assert ForestFlashAttentionBackend.get_impl_cls() is FlashAttentionImpl


@require_torch_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("layout", list(KVCacheLayout))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bulk_cache_update_matches_native_update_for_packed_views(layout, dtype):
    from arctic_platform.inference.py_custom_ops import try_load_torch_library, reshape_and_cache_flash_bulk
    if not try_load_torch_library():
        pytest.skip("Arctic custom ops are unavailable")
    set_seed(17)
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=32, dtype=dtype)
    strides = compute_layout_strides(spec, 3, 2, layout)
    size = 2 * 3 * spec.page_size_bytes
    tensor = KVCacheTensor(size=size, layers=["a", "b"], layer_stride=strides[0], block_stride=strides[1])
    raw = torch.zeros(size, dtype=torch.int8, device="cuda")
    reference_raw = raw.clone()
    caches = create_kv_cache_views(raw, spec, 3, layout, tensor)
    reference = create_kv_cache_views(reference_raw, spec, 3, layout, tensor)
    keys = torch.randn(5, 2 * 2 * 32, dtype=dtype, device="cuda")
    values = torch.randn_like(keys)
    slots = torch.tensor([0, 15, -1, 17, 34], device="cuda")
    scales = [torch.ones((), device="cuda") for _ in caches]
    views = [cache.transpose(1, 2).split(32, dim=-1) for cache in caches]
    reshape_and_cache_flash_bulk(keys, values, [view[0] for view in views],
        [view[1] for view in views], slots, "auto", scales, scales, 2, 32)
    for layer, cache in enumerate(reference):
        key_cache, value_cache = cache.transpose(1, 2).split(32, dim=-1)
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            keys.chunk(2, dim=-1)[layer].reshape(-1, 2, 32),
            values.chunk(2, dim=-1)[layer].reshape(-1, 2, 32),
            key_cache, value_cache, slots, "auto", scales[layer], scales[layer],
        )
    torch_assert_equal(raw, reference_raw)
