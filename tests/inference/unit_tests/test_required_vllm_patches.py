from arctic_platform.inference.vllm import config
from arctic_platform.inference.vllm import dense_prompt_logprobs
from arctic_platform.inference.vllm import dflash2_nan_fix
from arctic_platform.inference.vllm import glm5next_indexer_page
from arctic_platform.inference.vllm import kv_group_size
from arctic_platform.inference.vllm import mamba_completion_refresh
from arctic_platform.inference.vllm import mrope_yarn
from arctic_platform.inference.vllm import required_patches
from arctic_platform.inference.vllm import router_replay
from arctic_platform.inference.vllm import sparse_topk_order
from arctic_platform.inference.vllm import spec_decode_grammar
from arctic_platform.inference.vllm import xgrammar_stop_mask


def test_required_vllm_patches_are_applied_together(monkeypatch):
    calls = []
    monkeypatch.setattr(
        router_replay,
        "ensure_router_replay_vllm_patches",
        lambda: calls.append("router"),
    )
    monkeypatch.setattr(
        xgrammar_stop_mask,
        "ensure_xgrammar_stop_mask_fix",
        lambda: calls.append("xgrammar"),
    )
    monkeypatch.setattr(
        dense_prompt_logprobs,
        "ensure_dense_prompt_logprobs_patch",
        lambda: calls.append("dense_prompt_logprobs"),
    )
    monkeypatch.setattr(
        spec_decode_grammar,
        "ensure_spec_decode_grammar_fix",
        lambda: calls.append("spec_decode_grammar"),
    )
    monkeypatch.setattr(
        dflash2_nan_fix,
        "apply_dflash2_nan_fixes",
        lambda: calls.append("dflash2"),
    )
    monkeypatch.setattr(
        glm5next_indexer_page,
        "apply_glm5next_indexer_page_fix",
        lambda: calls.append("glm5next_indexer_page"),
    )
    monkeypatch.setattr(
        sparse_topk_order,
        "apply_sparse_topk_order_fix",
        lambda: calls.append("sparse_topk_order"),
    )
    monkeypatch.setattr(
        config,
        "ensure_dflash_max_position_patch",
        lambda: calls.append("dflash_max_position"),
    )
    monkeypatch.setattr(
        kv_group_size,
        "ensure_kv_group_size_patch",
        lambda: calls.append("kv_group_size"),
    )
    monkeypatch.setattr(
        mrope_yarn,
        "ensure_mrope_yarn_fix",
        lambda: calls.append("mrope_yarn"),
    )

    monkeypatch.setattr(
        mamba_completion_refresh,
        "ensure_mamba_completion_refresh",
        lambda: calls.append("mamba_completion_refresh"),
    )

    required_patches.apply_required_vllm_patches()

    assert calls == [
        "router",
        "xgrammar",
        "dense_prompt_logprobs",
        "spec_decode_grammar",
        "dflash2",
        "glm5next_indexer_page",
        "sparse_topk_order",
        "kv_group_size",
        "dflash_max_position",
        "mrope_yarn",
        "mamba_completion_refresh",
    ]
