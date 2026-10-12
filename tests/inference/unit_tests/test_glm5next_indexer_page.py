from dataclasses import dataclass

from arctic_platform.inference.vllm.glm5next_indexer_page import (
    build_with_deepgemm_page,
    indexer_kernel_block_sizes,
    metadata_block_size,
    paged_mqa_page_sizes,
)

# GLM-5.3-Flash packs one indexer state per 4 tokens. vLLM's hybrid aligner
# then grows the shared block to 640 tokens so it covers the Mamba page.
_KPOOL = 4
_MANAGER_BLOCK = 640
_HOPPER_PAGES = paged_mqa_page_sizes(sm120=False)


@dataclass
class _Spec:
    block_size: int
    tokens_per_state: int


def test_hopper_kernel_pages_divide_the_aligned_block():
    sizes = indexer_kernel_block_sizes(_KPOOL, _HOPPER_PAGES)

    assert sizes == [128, 256]
    assert _MANAGER_BLOCK % 128 == 0
    assert _MANAGER_BLOCK % 256 != 0
    # 128 tokens / 4 tokens per state is the 32-state page DeepGEMM accepts.
    assert 128 // _KPOOL == 32


def test_sm120_keeps_only_the_64_state_page():
    pages = paged_mqa_page_sizes(sm120=True)

    assert indexer_kernel_block_sizes(_KPOOL, pages) == [256]
    assert pages == (64,)


def test_metadata_uses_the_32_state_page_for_a_640_token_block():
    kernel_block = 128

    seen = metadata_block_size(
        _MANAGER_BLOCK, kernel_block, _KPOOL, _HOPPER_PAGES)

    assert seen == kernel_block
    assert seen // _KPOOL == 32


def test_metadata_leaves_an_illegal_kernel_page_alone():
    # A 64-token kernel page is 16 states. DeepGEMM rejects that, and
    # shrinking the spec to it would not fix the assert.
    assert metadata_block_size(
        _MANAGER_BLOCK, 64, _KPOOL, _HOPPER_PAGES) is None


def test_metadata_leaves_deepseek_and_already_legal_pages_alone():
    assert metadata_block_size(64, 64, 1, _HOPPER_PAGES) is None
    # 256 tokens is already one 64-state page, so no temporary view is needed.
    assert metadata_block_size(256, 256, _KPOOL, _HOPPER_PAGES) is None


def test_build_shows_the_kernel_page_and_restores_the_manager_block():
    spec = _Spec(block_size=_MANAGER_BLOCK, tokens_per_state=_KPOOL)
    builder = type("Builder", (), {})()
    builder.kv_cache_spec = spec
    builder.kernel_block_size = 128
    seen = {}

    def original_build(bound, *args, **kwargs):
        seen["block_size"] = bound.kv_cache_spec.block_size
        seen["states"] = (
            bound.kv_cache_spec.block_size
            // bound.kv_cache_spec.tokens_per_state)
        # The worker already expanded block 7 into five 128-token pages.
        # Recompressing (block_size != kernel) would fold them back to 7
        # and hand DeepGEMM 160 states. Equal sizes skip that fold.
        seen["recompress"] = (
            bound.kv_cache_spec.block_size != bound.kernel_block_size)
        return "metadata"

    result = build_with_deepgemm_page(
        builder, original_build, _HOPPER_PAGES, (), {})

    assert result == "metadata"
    assert seen == {"block_size": 128, "states": 32, "recompress": False}
    assert builder.kv_cache_spec is spec
    assert spec.block_size == _MANAGER_BLOCK


def test_build_restores_the_spec_when_metadata_build_fails():
    spec = _Spec(block_size=_MANAGER_BLOCK, tokens_per_state=_KPOOL)
    builder = type("Builder", (), {})()
    builder.kv_cache_spec = spec
    builder.kernel_block_size = 128

    def original_build(bound, *args, **kwargs):
        raise RuntimeError("capture failed")

    try:
        build_with_deepgemm_page(
            builder, original_build, _HOPPER_PAGES, (), {})
    except RuntimeError as exc:
        assert str(exc) == "capture failed"
    else:
        raise AssertionError("expected the metadata build to fail")

    assert builder.kv_cache_spec is spec
