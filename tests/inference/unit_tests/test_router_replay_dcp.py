from types import SimpleNamespace

import pytest
from vllm.config import ParallelConfig, VllmConfig

from arctic_platform.inference.vllm.router_replay import ensure_router_replay_vllm_patches


def _capture_config(**parallel):
    """The fields vLLM's auxiliary-output check reads, for a routed-expert capture engine."""
    return SimpleNamespace(
        aux_output_config=SimpleNamespace(enabled=True),
        use_v2_model_runner=True,
        model_config=SimpleNamespace(runner_type="generate", is_moe=True),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        speculative_config=None,
        parallel_config=ParallelConfig(tensor_parallel_size=8, **parallel),
        kv_transfer_config=None,
    )


def test_routed_expert_capture_accepts_decode_context_parallelism():
    ensure_router_replay_vllm_patches()
    config = _capture_config(decode_context_parallel_size=8)

    VllmConfig._verify_aux_output_compatibility(config)

    assert config.parallel_config.decode_context_parallel_size == 8


@pytest.mark.parametrize(
    "parallel, message",
    [
        ({"prefill_context_parallel_size": 2}, "context parallelism"),
        ({"pipeline_parallel_size": 2}, "pipeline parallelism"),
    ],
)
def test_routed_expert_capture_preserves_other_parallelism_refusals(parallel, message):
    ensure_router_replay_vllm_patches()
    config = _capture_config(decode_context_parallel_size=2, **parallel)

    with pytest.raises(ValueError, match=message):
        VllmConfig._verify_aux_output_compatibility(config)

    assert config.parallel_config.decode_context_parallel_size == 2
