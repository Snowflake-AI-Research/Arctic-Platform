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
    from arctic_platform.model.implementations.deepseek_v4 import DeepseekV4ForCausalLM

    model = DeepseekV4ForCausalLM(_tiny_config())
    hash_moe = model.model.layers[0].mlp
    learned_moe = model.model.layers[1].mlp
    assert hash_moe.is_hash
    assert hash_moe.router.tid2eid.shape == (32, 2)
    assert hash_moe.expert_bias is None
    assert learned_moe.expert_bias.shape == (4,)
    assert learned_moe.score_before_experts is False
    assert learned_moe.router.score_func == "sqrtsoftplus"


def test_published_expert_names_convert_to_platform_parameters():
    from arctic_platform.model.implementations.deepseek_v4.converting_deepseek_v4 import conversion_chain
    from arctic_platform.model.implementations.deepseek_v4.conversion_ops import apply_hf_to_prime

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
