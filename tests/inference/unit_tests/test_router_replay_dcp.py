import pytest
from transformers import Qwen3MoeConfig
from vllm.config import AuxOutputConfig, CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.config.kv_transfer import KVTransferConfig

from arctic_platform.inference.vllm.plugin import arctic_inference_plugin


def test_routed_expert_capture_with_decode_context_parallelism(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCTIC_INFERENCE_ENABLED", "0")
    arctic_inference_plugin()
    model = Qwen3MoeConfig(architectures=["Qwen3MoeForCausalLM"], num_attention_heads=8, num_key_value_heads=1)
    model.save_pretrained(tmp_path)
    parallel = ParallelConfig(
        tensor_parallel_size=8, decode_context_parallel_size=8, distributed_executor_backend="mp"
    )
    config = VllmConfig(
        model_config=ModelConfig(model=str(tmp_path), dtype="float32", enforce_eager=True),
        device_config=DeviceConfig("cpu"),
        cache_config=CacheConfig(enable_prefix_caching=True),
        parallel_config=parallel,
        aux_output_config=AuxOutputConfig(enable_return_routed_experts=True),
    )
    assert config.parallel_config.decode_context_parallel_size == 8

    for field, size, message in (
        ("prefill_context_parallel_size", 8, "context parallelism"),
        ("pipeline_parallel_size", 2, "pipeline parallelism"),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(parallel, field, size)
            with pytest.raises(ValueError, match=message):
                config._verify_aux_output_compatibility()
    config.kv_transfer_config = KVTransferConfig(kv_connector="NixlConnector", kv_role="kv_both")
    with pytest.raises(ValueError, match="NixlConnector"):
        config._verify_aux_output_compatibility()
    assert config.parallel_config.decode_context_parallel_size == 8
