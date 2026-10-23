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
"""The ported counter matches ArcticTraining, and the H200 models match their reports."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from arctic_platform.common.flops_counter import VALID_CONFIG_TYPE
from arctic_platform.common.flops_counter import FlopsCounter
from arctic_platform.common.flops_counter import estimate_decoder_transformer_tflos
from arctic_platform.common.flops_counter import tflops_per_token

_ARCTIC_TRAINING_COUNTER = Path(
    "/code/users/abkulkarni/dss-monorepo/ArcticTraining-dss/arctic_training/trainer/flops_counter.py"
)


class _Config:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def _load_arctic_training_counter():
    if not _ARCTIC_TRAINING_COUNTER.is_file():
        return None
    debug_utils = types.ModuleType("arctic_training.debug.utils")
    debug_utils.print_rank = lambda *args, **kwargs: None
    for name, module in {
        "arctic_training": types.ModuleType("arctic_training"),
        "arctic_training.debug": types.ModuleType("arctic_training.debug"),
        "arctic_training.debug.utils": debug_utils,
    }.items():
        sys.modules.setdefault(name, module)
    sys.modules["arctic_training.debug.utils"] = debug_utils
    spec = importlib.util.spec_from_file_location("arctic_training_flops_counter", _ARCTIC_TRAINING_COUNTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _llama():
    return _Config(
        model_type="llama",
        hidden_size=64,
        vocab_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        head_dim=16,
    )


def _qwen3_moe():
    return _Config(
        model_type="qwen3_moe",
        hidden_size=32,
        vocab_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        moe_intermediate_size=48,
        num_experts_per_tok=2,
        num_experts=8,
        full_attention_interval=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=4,
    )


def _qwen3_next():
    config = _qwen3_moe()
    config.model_type = "qwen3_next"
    return config


def _deepseek_v3():
    return _Config(
        model_type="deepseek_v3",
        hidden_size=64,
        vocab_size=128,
        moe_intermediate_size=32,
        intermediate_size=96,
        num_hidden_layers=4,
        first_k_dense_replace=1,
        num_attention_heads=4,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        q_lora_rank=16,
        kv_lora_rank=16,
        v_head_dim=16,
    )


@pytest.mark.skipif(not _ARCTIC_TRAINING_COUNTER.is_file(), reason="ArcticTraining checkout is not in this workspace")
def test_ported_counter_matches_arctic_training():
    original = _load_arctic_training_counter()
    seqlens = [32, 40]
    qwen35 = _qwen3_next()
    qwen35.model_type = "qwen3_5_moe_text"
    for config in (_llama(), _qwen3_moe(), _qwen3_next(), qwen35, _deepseek_v3()):
        for checkpoint in (False, True):
            for forward_only in (False, True):
                ported = FlopsCounter(config, model_size=1000, enable_gradient_checkpointing=checkpoint)
                upstream = original.FlopsCounter(config, model_size=1000, enable_gradient_checkpointing=checkpoint)
                got = ported.estimate_tflos(seqlens, forward_only=forward_only)
                expected = upstream.estimate_tflos(seqlens, forward_only=forward_only)
                assert got == expected, (config.model_type, checkpoint, forward_only, got, expected)


def test_unknown_model_uses_the_dense_fallback():
    config = _Config(model_type="gemma3_text", num_hidden_layers=2, hidden_size=16)
    counter = FlopsCounter(config, model_size=10, enable_gradient_checkpointing=True)
    # 8 * 10 * 20 + 16 * 2 * 16 * 20**2, over 1e12.
    assert counter.estimate_tflos([20]) > 0
    assert "gemma3_text" in VALID_CONFIG_TYPE
    assert "gemma3_text" not in counter.estimate_func


def test_batch_helper_sums_rows():
    config = _llama()
    total, tokens = estimate_decoder_transformer_tflos(
        config,
        model_size=1000,
        batch_of_seqlens=[[8, 8], [16]],
        enable_gradient_checkpointing=True,
    )
    one = FlopsCounter(config, 1000, enable_gradient_checkpointing=True)
    assert tokens == 32
    assert total == one.estimate_tflos([8, 8]) + one.estimate_tflos([16])


def _deepseek_v4_flash():
    ratios = [0, 0] + [4, 128] * 19 + [0, 0, 0]
    kinds = {
        0: "sliding_attention",
        4: "compressed_sparse_attention",
        128: "heavily_compressed_attention",
    }
    return _Config(
        model_type="deepseek_v4",
        hidden_size=4096,
        vocab_size=129280,
        num_hidden_layers=43,
        num_attention_heads=64,
        head_dim=512,
        q_lora_rank=1024,
        o_groups=8,
        o_lora_rank=1024,
        moe_intermediate_size=2048,
        num_experts_per_tok=6,
        n_shared_experts=1,
        n_routed_experts=256,
        hc_mult=4,
        sliding_window=128,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        layer_types=[kinds[ratio] for ratio in ratios],
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 128,
        },
    )


def test_deepseek_v4_matches_the_reported_per_token_cost():
    config = _deepseek_v4_flash()
    assert abs(tflops_per_token(config, 65536) - 0.14061) < 5e-5
    assert abs(tflops_per_token(config, 262144) - 0.21807) < 5e-5
    # A longer context costs more per token, and two sequences add.
    short = FlopsCounter(config, 0, enable_gradient_checkpointing=True).estimate_tflos([1024])
    long = FlopsCounter(config, 0, enable_gradient_checkpointing=True).estimate_tflos([1024, 2048])
    assert long > short
    no_recompute = tflops_per_token(config, 65536, enable_gradient_checkpointing=False)
    assert abs(no_recompute - 0.14061 * 6 / 8) < 5e-5


def _glm53_flash():
    sparse = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43}
    return _Config(
        model_type="glm5_next_text",
        hidden_size=4096,
        vocab_size=154880,
        num_hidden_layers=45,
        num_attention_heads=64,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        v_head_dim=256,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        num_experts_per_tok=8,
        n_shared_experts=1,
        n_routed_experts=288,
        first_k_dense_replace=3,
        hc_mult=4,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        index_kpool=4,
        linear_num_heads=64,
        linear_head_dim=128,
        linear_conv_kernel_dim=4,
        layer_types=["deepseek_sparse_attention" if i in sparse else "linear_attention" for i in range(45)],
        mlp_layer_types=["dense"] * 3 + ["sparse"] * 42,
        indexer_types=["full"] * 45,
    )


def test_glm53_flash_matches_the_reported_attention_terms():
    config = _glm53_flash()
    # 8 x 16.742B activated parameters, plus 11 sparse layers. The H200 tables
    # rounded that activated count to 18B (0.17722 at 64K, 0.25694 at 256K).
    assert abs(tflops_per_token(config, 65536) - 0.16715) < 5e-5
    assert abs(tflops_per_token(config, 262144) - 0.24688) < 5e-5
    outer = _Config(model_type="glm5_next", text_config=config)
    assert tflops_per_token(outer, 65536) == tflops_per_token(config, 65536)
