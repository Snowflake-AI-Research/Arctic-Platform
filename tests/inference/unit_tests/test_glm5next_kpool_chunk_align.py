import sys
import types
from dataclasses import dataclass

import pytest

from arctic_platform.inference.vllm import glm5next_kpool_chunk_align


@dataclass
class _CacheConfig:
    block_size: int
    mamba_cache_mode: str


class _Glm5NextIndexerCache:
    # Constructor contract of vLLM 0.31 models/glm5next/common/attention.py:Glm5NextIndexerCache.
    def __init__(self, *, head_dim, dtype, prefix, cache_config, index_kpool):
        assert index_kpool > 1
        assert cache_config.block_size % index_kpool == 0
        self._index_kpool = index_kpool


@pytest.fixture
def indexer_cache(monkeypatch):
    names = ["vllm", "vllm.models", "vllm.models.glm5next", "vllm.models.glm5next.common"]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    attention = types.ModuleType("vllm.models.glm5next.common.attention")
    cls = type("Glm5NextIndexerCache", (_Glm5NextIndexerCache,), {})
    attention.Glm5NextIndexerCache = cls
    monkeypatch.setitem(sys.modules, attention.__name__, attention)
    monkeypatch.setattr(glm5next_kpool_chunk_align, "_PATCHED", False)
    glm5next_kpool_chunk_align.apply_glm5next_kpool_chunk_align_check()
    return cls


def _build(cls, mode):
    # GLM-5.3-Flash: index_kpool 4, 768-token blocks after the indexer page fix.
    return cls(head_dim=128, dtype=None, prefix="model.layers.3.self_attn.indexer.k_cache",
               cache_config=_CacheConfig(block_size=768, mamba_cache_mode=mode), index_kpool=4)


def test_prefix_caching_off_is_refused_at_construction(indexer_cache):
    # Prefix caching off makes vLLM set mamba_cache_mode "none": chunks end wherever the token budget runs out.
    with pytest.raises(ValueError, match="enable_prefix_caching=True"):
        _build(indexer_cache, "none")


def test_aligned_chunks_construct_the_cache(indexer_cache):
    assert _build(indexer_cache, "align")._index_kpool == 4
