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

"""NemotronH router replay on CPU: the router gathers replayed routes, and the model hands each MoE layer its own."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from arctic_platform.model.implementations.gpu.router_replay_recompute import _router_supports_routed_experts
from arctic_platform.model.implementations.gpu.router_replay_recompute import install_self_router_replay
from arctic_platform.model.implementations.moe.layers.moe import ROUTER_REPLAY_FRESH
from arctic_platform.model.implementations.moe.layers.moe import LatentMoE
from arctic_platform.model.implementations.moe.layers.moe import NemotronHRouter
from arctic_platform.model.implementations.moe.vllm_weights import named_weight_sync_tensors
from arctic_platform.model.implementations.nemotron_h.configuration_nemotron_h import NemotronHConfig
from arctic_platform.model.implementations.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal

LAYERS_BLOCK_TYPE = ["mamba", "moe", "attention", "moe"]
MOE_LAYERS = [1, 3]
TOP_K = 2


def _llm_config(**overrides) -> dict:
    """A tiny NemotronH language-model config with the released checkpoint's (Hugging Face) field names."""
    config = dict(
        architectures=["NemotronHForCausalLM"],
        model_type="nemotron_h",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=16,
        layers_block_type=LAYERS_BLOCK_TYPE,
        mtp_layers_block_type=["attention", "moe"],
        num_nextn_predict_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        mamba_num_heads=8,
        mamba_head_dim=2,
        n_groups=8,
        ssm_state_size=4,
        conv_kernel=4,
        chunk_size=8,
        expand=1,
        n_routed_experts=4,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=8,
        moe_shared_expert_intermediate_size=8,
        moe_latent_size=8,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=5.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=11,
        tie_word_embeddings=False,
    )
    config.update(overrides)
    return config


class _Head(nn.Module):
    """Stands in for the injected PrimeRL LM head, which takes labels and temperature."""

    def forward(self, hidden_states, labels=None, temperature=None):
        return hidden_states


def _model() -> NemotronHForCausalLM:
    set_seed(0)
    model = NemotronHForCausalLM(NemotronHConfig.from_dict(_llm_config(), attn_implementation="sdpa"))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_()
    return model


# --------------------------------------------------------------------------- #
# Router replay
# --------------------------------------------------------------------------- #


def _router(n_group: int = 1, topk_group: int = 1) -> NemotronHRouter:
    set_seed(0)
    router = NemotronHRouter(
        dim=8, num_experts=6, top_k=TOP_K, n_group=n_group, topk_group=topk_group, norm_topk_prob=True
    )
    with torch.no_grad():
        router.gate.normal_()
        router.e_score_correction_bias.copy_(torch.linspace(0.0, 0.3, 6))
    return router


def _routes() -> torch.Tensor:
    return torch.tensor([[5, 4], [0, 1], [3, 2], [1, 0], [2, 5]], dtype=torch.int64)


@pytest.mark.parametrize("n_group,topk_group", [(1, 1), (2, 1)])
def test_router_gathers_replayed_routes(n_group, topk_group):
    router = _router(n_group, topk_group)
    x = torch.randn(5, 8)
    routed = _routes()

    top_scores, experts, num_tokens_per_expert = router(x, routed_experts=routed)

    torch_assert_equal(experts, routed)
    raw = torch.sigmoid(F.linear(x, router.gate)).gather(1, routed)
    torch_assert_close(top_scores, raw / raw.sum(dim=-1, keepdim=True))
    torch_assert_equal(num_tokens_per_expert, torch.bincount(routed.reshape(-1), minlength=6).float())


@pytest.mark.parametrize("n_group,topk_group", [(1, 1), (2, 1)])
def test_router_routes_fresh_tokens_like_replay_off(n_group, topk_group):
    router = _router(n_group, topk_group)
    router.enable_fresh_replay_routing()
    x = torch.randn(5, 8)
    routed = _routes()
    routed[[1, 3]] = ROUTER_REPLAY_FRESH
    fresh = (routed == ROUTER_REPLAY_FRESH).all(dim=1)

    off_scores, off_experts, _ = router(x)
    top_scores, experts, _ = router(x, routed_experts=routed)

    torch_assert_equal(experts[fresh], off_experts[fresh])
    torch_assert_equal(top_scores[fresh], off_scores[fresh])
    torch_assert_equal(experts[~fresh], routed[~fresh])


def test_router_is_wrapped_for_activation_checkpoint_replay():
    block = nn.Module()
    block.router = _router()

    assert _router_supports_routed_experts(block.router)
    assert install_self_router_replay(block) == 1


def test_router_correction_bias_is_exported_but_not_synced():
    block = nn.Module()
    block.router = _router()

    assert "router.e_score_correction_bias" in dict(named_weight_sync_tensors(block, include_export_buffers=True))
    assert "router.e_score_correction_bias" not in dict(named_weight_sync_tensors(block, include_export_buffers=False))


def test_model_hands_each_moe_layer_its_replayed_routes(monkeypatch):
    seen = []

    def moe_forward(self, x, routed_experts=None):
        seen.append(routed_experts)
        return torch.zeros_like(x)

    monkeypatch.setattr(LatentMoE, "forward", moe_forward)
    model = _model()
    model.lm_head = _Head()
    input_ids = torch.randint(0, 32, (1, 6))
    position_ids = torch.arange(6).unsqueeze(0)
    routed = torch.arange(6 * len(LAYERS_BLOCK_TYPE) * TOP_K).reshape(1, 6, len(LAYERS_BLOCK_TYPE), TOP_K) % 4

    with torch.no_grad():
        model(input_ids=input_ids, position_ids=position_ids, routed_experts=routed)
        model(input_ids=input_ids, position_ids=position_ids)

    assert len(seen) == 2 * len(MOE_LAYERS)
    for routes, layer_idx in zip(seen[: len(MOE_LAYERS)], MOE_LAYERS):
        torch_assert_equal(routes, routed[:, :, layer_idx, :])
    assert seen[len(MOE_LAYERS) :] == [None] * len(MOE_LAYERS)
