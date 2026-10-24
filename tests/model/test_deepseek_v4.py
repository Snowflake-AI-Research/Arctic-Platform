# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json

import pytest
import torch

from arctic_platform.model import ModelSpec
from arctic_platform.model import ParallelismConfig


def _checkpoint(tmp_path, *, n_routed_experts: int | None = None) -> str:
    path = tmp_path / "deepseek_v4"
    path.mkdir()
    config = {"model_type": "deepseek_v4"}
    if n_routed_experts is not None:
        config["n_routed_experts"] = n_routed_experts
    (path / "config.json").write_text(json.dumps(config))
    return str(path)


def _tiny_config():
    from arctic_platform.model.implementations.deepseek_v4 import DeepseekV4Config

    return DeepseekV4Config(
        vocab_size=32,
        hidden_size=64,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=32,
        num_key_value_heads=1,
        head_dim=32,
        q_lora_rank=32,
        max_position_embeddings=32,
        sliding_window=4,
        o_groups=8,
        o_lora_rank=16,
        layer_types=["sliding_attention", "compressed_sparse_attention"],
        index_n_heads=4,
        index_head_dim=16,
        index_topk=4,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        num_hash_layers=1,
        hc_mult=2,
        pad_token_id=0,
    )


def test_loader_selects_deepseek_v4(tmp_path):
    spec = ModelSpec(
        model_path_or_name=_checkpoint(tmp_path, n_routed_experts=256),
        parallelism=ParallelismConfig(expert_parallel=8),
    )
    assert spec.loader == "deepseek_v4"
    assert spec.attn_implementation == "eager"


def test_expert_parallel_one_stays_on_huggingface(tmp_path):
    spec = ModelSpec(
        model_path_or_name=_checkpoint(tmp_path),
        parallelism=ParallelismConfig(expert_parallel=1),
    )
    assert spec.loader == "huggingface"


def test_sequence_parallel_is_accepted(tmp_path):
    spec = ModelSpec(
        model_path_or_name=_checkpoint(tmp_path),
        parallelism=ParallelismConfig(expert_parallel=8, sequence_parallel=2),
    )
    assert spec.loader == "deepseek_v4"
    assert spec.parallelism.sequence_parallel == 2


def test_ep_must_divide_routed_experts(tmp_path):
    with pytest.raises(ValueError, match="must divide"):
        ModelSpec(
            model_path_or_name=_checkpoint(tmp_path, n_routed_experts=256),
            parallelism=ParallelismConfig(expert_parallel=6),
        )


def test_autoconfig_returns_the_ported_class(tmp_path):
    from transformers import AutoConfig

    from arctic_platform.model.implementations.deepseek_v4 import DeepseekV4Config
    from arctic_platform.model.implementations.qwen35.models import supports_custom_impl

    config = AutoConfig.from_pretrained(_checkpoint(tmp_path))
    assert isinstance(config, DeepseekV4Config)
    assert supports_custom_impl(config)
    assert config.num_hash_layers == 3
    assert config.scoring_func == "sqrtsoftplus"


def test_hash_layer_uses_tid2eid_and_learned_layer_keeps_expert_bias():
    from arctic_platform.model.implementations.deepseek_v4.moe import DeepseekV4MoE

    config = _tiny_config()
    hash_moe = DeepseekV4MoE(config, 0)
    learned_moe = DeepseekV4MoE(config, 1)
    assert hash_moe.is_hash
    assert hash_moe.router.tid2eid.shape == (32, 2)
    assert hash_moe.expert_bias is None
    assert learned_moe.expert_bias.shape == (4,)
    assert learned_moe.score_before_experts is False
    assert learned_moe.router.score_func == "sqrtsoftplus"


def test_published_expert_names_convert_to_platform_parameters():
    from arctic_platform.model.implementations.deepseek_v4.conversion_ops import apply_hf_to_prime
    from arctic_platform.model.implementations.deepseek_v4.converting_deepseek_v4 import conversion_chain

    config = _tiny_config()
    state = {
        "layers.0.ffn.gate.weight": torch.zeros(4, 64),
        "layers.0.ffn.gate.tid2eid": torch.zeros(32, 2, dtype=torch.long),
        "layers.0.ffn.gate.bias": torch.ones(4),
        "layers.0.ffn.shared_experts.w1.weight": torch.zeros(8, 64),
        "layers.0.ffn.experts.0.w1.weight": torch.zeros(8, 64),
        "layers.0.ffn.experts.1.w1.weight": torch.zeros(8, 64),
        "layers.1.ffn.gate.weight": torch.zeros(4, 64),
        "layers.1.ffn.gate.bias": torch.ones(4),
        "layers.1.ffn.shared_experts.w1.weight": torch.zeros(8, 64),
        "layers.1.ffn.experts.0.w1.weight": torch.zeros(8, 64),
        "embed.weight": torch.zeros(32, 64),
        "mtp.0.hc_attn_base": torch.zeros(1),
    }
    converted = apply_hf_to_prime(state, conversion_chain(config))
    assert "model.layers.0.mlp.router.tid2eid" in converted
    assert "model.layers.0.mlp.expert_bias" not in converted
    assert "model.layers.0.ffn.gate.bias" not in converted
    assert converted["model.layers.0.mlp.shared_expert.w1"].shape == (8, 64)
    assert converted["model.layers.0.mlp.experts.w1"].shape == (2, 8, 64)
    assert torch.equal(converted["model.layers.1.mlp.expert_bias"], torch.ones(4))
    assert "mtp.0.hc_attn_base" not in converted
    assert converted["model.embed_tokens.weight"].shape == (32, 64)


def test_checkpoint_fp32_parameters_are_built_in_fp32():
    from arctic_platform.model.implementations.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

    model = DeepseekV4ForCausalLM._from_config(_tiny_config(), dtype=torch.bfloat16)
    dtypes = {name: param.dtype for name, param in model.named_parameters()}
    fp32 = sorted(name for name, dtype in dtypes.items() if dtype == torch.float32)
    assert fp32 == sorted(name for name in dtypes if DeepseekV4ForCausalLM.keeps_fp32_parameter(name))
    assert "model.layers.0.attn_hc.fn" in fp32
    assert "model.layers.1.self_attn.sinks" in fp32
    assert "model.layers.1.self_attn.compressor.position_bias" in fp32
    assert "model.hc_head.hc_scale" in fp32
    assert dtypes["model.layers.0.input_layernorm.weight"] == torch.bfloat16
    assert dtypes["model.layers.0.self_attn.q_a_proj.weight"] == torch.bfloat16


def test_hf_export_writes_published_names_and_loads_back(monkeypatch):
    from arctic_platform.model.implementations.deepseek_v4.conversion_ops import apply_hf_to_prime
    from arctic_platform.model.implementations.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
    from arctic_platform.model.implementations.moe.deepspeed_integration import build_iter_full_hf_weights
    from arctic_platform.model.implementations.moe.vllm_weights import named_weight_sync_tensors

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    model = DeepseekV4ForCausalLM(_tiny_config())
    model.model.layers[0].mlp.router.tid2eid.copy_(torch.arange(64).view(32, 2) % 4)

    exported = dict(build_iter_full_hf_weights(model)())
    assert not [name for name in exported if name.startswith("model.")]
    assert "layers.1.attn.indexer.compressor.wkv.weight" in exported
    assert "layers.1.ffn.gate.bias" in exported
    assert "layers.0.ffn.gate.bias" not in exported

    expected = dict(named_weight_sync_tensors(model))
    expected["model.layers.0.mlp.router.tid2eid"] = model.model.layers[0].mlp.router.tid2eid
    loaded = apply_hf_to_prime(dict(exported), model.conversion_chain(model.config))
    assert sorted(loaded) == sorted(expected)
    for name, tensor in expected.items():
        assert torch.equal(loaded[name], tensor), name


def test_official_fp8_checkpoint_keeps_packed_expert_weights():
    from arctic_platform.model.implementations.deepseek_v4.conversion_ops import apply_hf_to_prime
    from arctic_platform.model.implementations.deepseek_v4.converting_deepseek_v4 import conversion_chain
    from arctic_platform.model.implementations.deepseek_v4.dequantize import decode_e8m0_scales_
    from arctic_platform.model.implementations.deepseek_v4.dequantize import encode_e8m0_scales_
    from arctic_platform.model.implementations.deepseek_v4.moe import DeepseekV4MoE

    config = _tiny_config()
    config.moe_intermediate_size = 64
    config.quantization_config = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    scale = torch.ones(8, 2, dtype=torch.float8_e8m0fnu)
    state = {
        "layers.1.ffn.experts.0.w1.weight": torch.ones(64, 32, dtype=torch.int8),
        "layers.1.ffn.experts.1.w1.weight": torch.full((64, 32), 2, dtype=torch.int8),
        "layers.1.ffn.experts.0.w1.scale": scale.clone(),
        "layers.1.ffn.experts.1.w1.scale": scale.clone(),
        "layers.1.attn.wq_a.weight": torch.zeros(32, 64, dtype=torch.float8_e4m3fn),
        "layers.1.attn.wq_a.scale": torch.ones(1, 1, dtype=torch.float8_e8m0fnu),
    }
    converted = apply_hf_to_prime(dict(state), conversion_chain(config))
    decode_e8m0_scales_(converted)
    experts = converted["model.layers.1.mlp.experts.w1"]
    assert experts.dtype == torch.int8
    assert experts.shape == (2, 64, 32)
    assert converted["model.layers.1.mlp.experts.w1_scale_inv"].dtype == torch.float32
    assert converted["model.layers.1.self_attn.q_a_proj.weight"].dtype == torch.float8_e4m3fn
    assert converted["model.layers.1.self_attn.q_a_proj.weight_scale_inv"].dtype == torch.float32

    encode_e8m0_scales_(converted)
    assert converted["model.layers.1.mlp.experts.w1_scale_inv"].dtype == torch.float8_e8m0fnu

    moe = DeepseekV4MoE(config, 1)
    assert moe.experts.mxfp4
    assert moe.experts.w1.dtype == torch.int8
    assert moe.experts.w1.shape == (4, 64, 32)
    assert moe.shared_expert.w1.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize("peft", [None, {"peft_type": "Lora", "r": 8}])
def test_only_lora_keeps_the_official_checkpoint_packed(tmp_path, monkeypatch, peft):
    from arctic_platform.model.implementations.deepseek_v4 import deepspeed_integration
    from arctic_platform.model.loader import LoaderContext
    from arctic_platform.model.loaders.deepseek_v4 import load_deepseek_v4

    calls = []
    monkeypatch.setattr(deepspeed_integration, "load_deepseek_v4_model", lambda **kwargs: calls.append(kwargs))
    spec = ModelSpec(
        model_path_or_name=_checkpoint(tmp_path),
        parallelism=ParallelismConfig(expert_parallel=8),
        patches={"peft": peft},
    )
    load_deepseek_v4(LoaderContext(spec=spec, parallel_groups={"ep_group": object()}))
    assert calls[0]["dequantize_checkpoint"] is (peft is None)


def test_dequantize_checkpoint_builds_bf16_modules_from_the_official_checkpoint(tmp_path):
    from arctic_platform.model.implementations.deepseek_v4.moe import Mxfp4GroupedExperts
    from arctic_platform.model.implementations.qwen35.config import ModelConfig
    from arctic_platform.model.implementations.qwen35.model_builder import get_model

    config = _tiny_config()
    config.moe_intermediate_size = 64
    config.quantization_config = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    config.save_pretrained(tmp_path)
    model = get_model(
        ModelConfig(name=str(tmp_path), attn="eager", impl="custom", dequantize_checkpoint=True),
        device=torch.device("meta"),
        dtype=torch.bfloat16,
    )
    assert not hasattr(model.config, "quantization_config")
    layer = model.model.layers[1]
    assert not isinstance(layer.mlp.experts, Mxfp4GroupedExperts)
    assert layer.mlp.experts.w1.dtype == torch.bfloat16
    assert layer.self_attn.q_a_proj.weight.dtype == torch.bfloat16

    state = {
        "layers.1.ffn.experts.0.w1.weight": torch.full((64, 32), 0x22, dtype=torch.int8),
        "layers.1.ffn.experts.1.w1.weight": torch.full((64, 32), 0x22, dtype=torch.int8),
        "layers.1.ffn.experts.0.w1.scale": torch.ones(64, 2, dtype=torch.float8_e8m0fnu),
        "layers.1.ffn.experts.1.w1.scale": torch.ones(64, 2, dtype=torch.float8_e8m0fnu),
        "layers.1.attn.wq_a.weight": torch.ones(32, 64, dtype=torch.float8_e4m3fn),
        "layers.1.attn.wq_a.scale": torch.ones(1, 1, dtype=torch.float8_e8m0fnu),
    }
    converted = model.convert_to_prime(state)
    assert not [name for name in converted if name.endswith("_scale_inv")]
    experts = converted["model.layers.1.mlp.experts.w1"]
    assert experts.dtype == torch.bfloat16
    assert experts.shape == (2, 64, 64)
    assert torch.all(experts == 1.0)
    assert torch.all(converted["model.layers.1.self_attn.q_a_proj.weight"] == 1.0)


def test_mxfp4_experts_match_dequantized_bf16_and_train_only_activations():
    from arctic_platform.model.implementations.deepseek_v4.dequantize import dequantize_mxfp4
    from arctic_platform.model.implementations.deepseek_v4.moe import Mxfp4GroupedExperts
    from arctic_platform.model.implementations.moe.layers.moe import _run_experts_for_loop_impl

    torch.manual_seed(0)
    experts = Mxfp4GroupedExperts(dim=64, hidden_dim=64, num_experts=2, swiglu_limit=10.0)
    with torch.no_grad():
        for name in ("w1", "w2", "w3"):
            weight = getattr(experts, name)
            weight.copy_(torch.randint(-7, 8, weight.shape, dtype=torch.int8))
            getattr(experts, f"{name}_scale_inv").fill_(1.0)
    x = torch.randn(4, 64, requires_grad=True)
    counts = torch.tensor([2, 2], dtype=torch.int32)
    output = experts(x, counts)
    reference = _run_experts_for_loop_impl(
        dequantize_mxfp4(experts.w1, experts.w1_scale_inv),
        dequantize_mxfp4(experts.w2, experts.w2_scale_inv),
        dequantize_mxfp4(experts.w3, experts.w3_scale_inv),
        x.detach().bfloat16(),
        counts,
        10.0,
    )
    assert torch.allclose(output, reference.float())
    output.sum().backward()
    assert x.grad is not None
    assert experts.w1.grad is None


def test_sqrtsoftplus_scores_match_softplus_sqrt():
    if not torch.cuda.is_available():
        pytest.skip("router histc on integer indices is CUDA-only")
    from arctic_platform.model.implementations.moe.layers.moe import TokenChoiceTopKRouter

    router = TokenChoiceTopKRouter(
        4,
        num_experts=3,
        top_k=1,
        score_func="sqrtsoftplus",
        route_norm=False,
        route_scale=1.0,
    ).cuda()
    with torch.no_grad():
        router.gate.weight.copy_(torch.eye(3, 4, device="cuda"))
    scores, indices, _ = router(torch.tensor([[2.0, 0.0, 0.0, 0.0]], device="cuda"))
    expected = torch.nn.functional.softplus(torch.tensor([2.0], device="cuda")).sqrt()
    assert indices.item() == 0
    assert torch.allclose(scores, expected)
