from types import SimpleNamespace

import pytest
from vllm.config import ParallelConfig, VllmConfig
from vllm.config.kv_transfer import KVTransferConfig

from arctic_platform.inference.vllm.required_patches import apply_required_vllm_patches


def _capture_config(**parallel):
    # The fields the check reads. Building a VllmConfig would also select a model runner, which tests that enable the
    # Arctic optimizations change for the whole process.
    return SimpleNamespace(
        aux_output_config=SimpleNamespace(enabled=True),
        use_v2_model_runner=True,
        model_config=SimpleNamespace(runner_type="generate", is_moe=True),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        speculative_config=None,
        parallel_config=ParallelConfig(tensor_parallel_size=8, **parallel),
        kv_transfer_config=None,
    )


def test_routed_expert_capture_with_decode_context_parallelism():
    apply_required_vllm_patches()
    config = _capture_config(decode_context_parallel_size=8)
    VllmConfig._verify_aux_output_compatibility(config)
    assert config.parallel_config.decode_context_parallel_size == 8

    for parallel, message in (
        (dict(prefill_context_parallel_size=2), "context parallelism"),
        (dict(pipeline_parallel_size=2), "pipeline parallelism"),
    ):
        with pytest.raises(ValueError, match=message):
            VllmConfig._verify_aux_output_compatibility(_capture_config(decode_context_parallel_size=2, **parallel))
    config.kv_transfer_config = KVTransferConfig(kv_connector="NixlConnector", kv_role="kv_both")
    with pytest.raises(ValueError, match="NixlConnector"):
        VllmConfig._verify_aux_output_compatibility(config)
    assert config.parallel_config.decode_context_parallel_size == 8
