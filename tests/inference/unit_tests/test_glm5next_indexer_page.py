from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from arctic_platform.inference.vllm import glm5next_indexer_page
from arctic_platform.inference.vllm.glm5next_indexer_page import build_with_deepgemm_page
from arctic_platform.inference.vllm.glm5next_indexer_page import indexer_kernel_block_sizes
from arctic_platform.inference.vllm.glm5next_indexer_page import metadata_block_size
from arctic_platform.inference.vllm.glm5next_indexer_page import paged_mqa_page_sizes

# GLM-5.3-Flash packs one indexer state per 4 tokens. The only page this
# DeepGEMM accepts on Hopper is 64 states, so the kernel page is 256 tokens.
# vLLM then rounds the 640-token Mamba-aligned block up to 768.
_KPOOL = 4
_MANAGER_BLOCK = 768
_PAGES = paged_mqa_page_sizes(sm120=False)


@dataclass
class _Spec:
    block_size: int
    tokens_per_state: int
    storage_block_size: int | None = None


def test_hopper_kernel_page_is_64_states():
    sizes = indexer_kernel_block_sizes(_KPOOL, _PAGES)

    assert _PAGES == (64,)
    assert sizes == [256]
    assert 640 % 256 != 0
    assert _MANAGER_BLOCK % 256 == 0
    assert 256 // _KPOOL == 64


def test_sm120_also_keeps_only_the_64_state_page():
    pages = paged_mqa_page_sizes(sm120=True)

    assert indexer_kernel_block_sizes(_KPOOL, pages) == [256]
    assert pages == (64,)


def test_metadata_uses_the_64_state_page_for_a_768_token_block():
    kernel_block = 256

    seen = metadata_block_size(_MANAGER_BLOCK, kernel_block, _KPOOL, _PAGES)

    assert seen == kernel_block
    assert seen // _KPOOL == 64


def test_metadata_leaves_a_32_state_page_alone():
    # 128 tokens is 32 states. This DeepGEMM rejects that on Hopper.
    assert metadata_block_size(640, 128, _KPOOL, _PAGES) is None


def test_metadata_leaves_an_illegal_kernel_page_alone():
    # A 64-token kernel page is 16 states. DeepGEMM rejects that, and
    # shrinking the spec to it would not fix the assert.
    assert metadata_block_size(640, 64, _KPOOL, _PAGES) is None


def test_metadata_leaves_deepseek_and_already_legal_pages_alone():
    assert metadata_block_size(64, 64, 1, _PAGES) is None
    # 256 tokens is already one 64-state page, so no temporary view is needed.
    assert metadata_block_size(256, 256, _KPOOL, _PAGES) is None


def test_build_shows_the_kernel_page_and_restores_the_manager_block():
    spec = _Spec(block_size=_MANAGER_BLOCK, tokens_per_state=_KPOOL)
    builder = type("Builder", (), {})()
    builder.kv_cache_spec = spec
    builder.kernel_block_size = 256
    seen = {}

    def original_build(bound, *args, **kwargs):
        seen["block_size"] = bound.kv_cache_spec.block_size
        seen["states"] = bound.kv_cache_spec.block_size // bound.kv_cache_spec.tokens_per_state
        # The worker already expanded block 7 into three 256-token pages.
        # Recompressing (block_size != kernel) would fold them back to 7
        # and hand DeepGEMM 192 states. Equal sizes skip that fold.
        seen["recompress"] = bound.kv_cache_spec.block_size != bound.kernel_block_size
        return "metadata"

    result = build_with_deepgemm_page(builder, original_build, _PAGES, (), {})

    assert result == "metadata"
    assert seen == {"block_size": 256, "states": 64, "recompress": False}
    assert builder.kv_cache_spec is spec
    assert spec.block_size == _MANAGER_BLOCK


def test_build_restores_the_spec_when_metadata_build_fails():
    spec = _Spec(block_size=_MANAGER_BLOCK, tokens_per_state=_KPOOL)
    builder = type("Builder", (), {})()
    builder.kv_cache_spec = spec
    builder.kernel_block_size = 256

    def original_build(bound, *args, **kwargs):
        raise RuntimeError("capture failed")

    try:
        build_with_deepgemm_page(builder, original_build, _PAGES, (), {})
    except RuntimeError as exc:
        assert str(exc) == "capture failed"
    else:
        raise AssertionError("expected the metadata build to fail")

    assert builder.kv_cache_spec is spec


@pytest.mark.parametrize("block_size", [256, 512, 768])
def test_storage_page_is_64_compressed_states(block_size):
    assert glm5next_indexer_page.has_legal_storage_page(_Spec(block_size, _KPOOL, 256), _PAGES)


@pytest.mark.parametrize(
    "spec",
    [
        None,
        SimpleNamespace(block_size=512, tokens_per_state=4),
        _Spec(512, 4),
        _Spec(512, 4, 128),
        _Spec(512, 4, 64),
        _Spec(512, 4, 0),
        _Spec(512, 4, -256),
        _Spec(640, 4, 256),
        _Spec(512, 1, 64),
        _Spec(512, 0, 256),
        _Spec(512, 4, 256.0),
        _Spec(512, True, 64),
    ],
)
def test_missing_or_invalid_storage_page_keeps_legacy_behavior(spec):
    assert not glm5next_indexer_page.has_legal_storage_page(spec, _PAGES)


@pytest.mark.parametrize("fails", [False, True])
def test_storage_aware_build_preserves_spec_and_arguments(fails):
    spec = _Spec(256, _KPOOL, 256)
    builder = SimpleNamespace(kv_cache_spec=spec, kernel_block_size=64)
    arguments = (object(),)
    keyword_arguments = {"marker": object()}

    def original_build(bound, *args, **kwargs):
        assert bound.kv_cache_spec is spec
        assert args == arguments
        assert kwargs == keyword_arguments
        if fails:
            raise RuntimeError("upstream build failed")
        return "upstream metadata"

    if fails:
        with pytest.raises(RuntimeError, match="upstream build failed"):
            build_with_deepgemm_page(builder, original_build, _PAGES, arguments, keyword_arguments)
    else:
        assert (
            build_with_deepgemm_page(builder, original_build, _PAGES, arguments, keyword_arguments)
            == "upstream metadata"
        )
    assert builder.kv_cache_spec is spec


def test_backend_uses_actual_returned_spec_and_keeps_instances_separate(monkeypatch):
    from vllm.models.glm5next.common import attention
    from vllm.v1.attention.backends.mla import indexer

    stock_backend = object()
    legacy_backend = object()

    class Cache:
        _index_kpool = _KPOOL

        def __init__(self, spec):
            self.spec = spec

        def get_kv_cache_spec(self, vllm_config):
            assert vllm_config == "config"
            return self.spec

        def get_attn_backend(self):
            return stock_backend

    class Builder:
        def build(self):
            return "metadata"

    monkeypatch.setattr(attention, "Glm5NextIndexerCache", Cache)
    monkeypatch.setattr(indexer, "DeepseekV32IndexerMetadataBuilder", Builder)
    monkeypatch.setattr(glm5next_indexer_page, "_PATCHED", False)
    monkeypatch.setattr(glm5next_indexer_page, "_indexer_backend", lambda *args: legacy_backend)
    glm5next_indexer_page.apply_glm5next_indexer_page_fix()
    glm5next_indexer_page.apply_glm5next_indexer_page_fix()

    modern = Cache(_Spec(512, _KPOOL, 256))
    legacy = Cache(_Spec(768, _KPOOL))
    assert modern.get_attn_backend() is legacy_backend
    assert modern.get_kv_cache_spec("config") is modern.spec
    assert modern.get_attn_backend() is stock_backend
    assert legacy.get_kv_cache_spec("config") is legacy.spec
    assert legacy.get_attn_backend() is legacy_backend
    assert modern.get_attn_backend() is stock_backend

    modern.spec = _Spec(640, _KPOOL, 128)
    modern.get_kv_cache_spec("config")
    assert modern.get_attn_backend() is legacy_backend
