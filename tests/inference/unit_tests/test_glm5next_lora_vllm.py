from types import SimpleNamespace

import torch

from arctic_platform.inference.vllm.glm5next_lora_model import Glm5NextLoRAForConditionalGeneration
from vllm.config import LoRAConfig, VllmConfig, set_current_vllm_config
from vllm.config.device import DeviceConfig
from vllm.lora.layers.column_parallel_linear import MergedColumnParallelLinearWithLoRA
from vllm.lora.utils import from_layer
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.models.deepseek_v2 import DeepSeekV2FusedQkvAProjLinear


def test_real_fp8_fused_projection_wraps_with_trainer_slice_shapes(monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8.init_fp8_linear_kernel",
        lambda **kwargs: object(),
    )
    vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    with set_current_vllm_config(vllm_config):
        layer = DeepSeekV2FusedQkvAProjLinear(
            4096,
            [1536, 512],
            quant_config=Fp8Config(
                is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
            ),
        )
    config = LoRAConfig(max_lora_rank=32, lora_dtype=torch.bfloat16)
    assert from_layer(layer, 1, config, []) is layer
    wrapped = from_layer(
        layer,
        1,
        config,
        Glm5NextLoRAForConditionalGeneration.packed_modules_mapping.get("fused_qkv_a_proj", []),
    )
    assert isinstance(wrapped, MergedColumnParallelLinearWithLoRA)
    assert wrapped.output_slices == (1536, 512)
    assert tuple(weight.shape[-2:] for weight in wrapped.lora_b_stacked) == ((1536, 32), (512, 32))
