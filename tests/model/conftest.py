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
"""Fixtures shared by the model tests."""

import pytest
import torch
from torch import nn

from arctic_platform.testing_utils import single_rank_process_group


@pytest.fixture
def single_rank_group():
    """The default process group, with one rank; see ``single_rank_process_group``."""
    with single_rank_process_group() as group:
        yield group


@pytest.fixture
def tiny_glm_sparse_model():
    """Build the sparse GLM model shared by packed-row and attention-backend tests."""

    def build(*, kv_lora_rank=4, with_linear_attention=False):
        configuration = pytest.importorskip("transformers.models.glm5_next.configuration_glm5_next")
        modeling = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
        from arctic_platform.model.implementations.glm53.context_parallel import apply_context_parallelism

        hidden_size = 256 if with_linear_attention else 16
        layer_types = (
            ["deepseek_sparse_attention", "linear_attention"]
            if with_linear_attention
            else ["deepseek_sparse_attention"]
        )
        config = configuration.Glm5NextTextConfig(
            vocab_size=64,
            pad_token_id=0,
            hidden_size=hidden_size,
            intermediate_size=128 if with_linear_attention else 32,
            moe_intermediate_size=16,
            num_hidden_layers=len(layer_types),
            num_attention_heads=2,
            num_key_value_heads=2,
            n_routed_experts=2,
            num_experts_per_tok=1,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=8,
            qk_rope_head_dim=0,
            v_head_dim=4,
            qk_nope_head_dim=4,
            layer_types=layer_types,
            indexer_types=["full"] * len(layer_types),
            mlp_layer_types=["dense"] * len(layer_types),
            index_topk=4,
            index_head_dim=4,
            index_n_heads=2,
            index_kpool=2,
            linear_head_dim=128,
            linear_num_heads=2,
            linear_conv_kernel_dim=4,
            hc_mult=1,
            max_position_embeddings=32,
        )

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = modeling.Glm5NextTextModel(config)
                self.config = type("TextConfig", (), {"model_type": "test"})()

            def forward(self, **kwargs):
                return self.model(**kwargs)

        model = Model().to(torch.bfloat16).eval()
        apply_context_parallelism(model, 1, None)
        return model

    return build
